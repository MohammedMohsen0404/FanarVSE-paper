"""Graph stage 1 as a CUDA kernel (NVRTC via CuPy): one 128-thread block per query.
Beam: sorted (score, id|expanded) arrays in shared memory. Each step expands the E best unexpanded beam entries, marks their
neighbours in a per-query visited bitmap (atomicOr), scores new ones with int8 codes (dp4a, 8 threads per code), keeps those above
the beam floor, sorts them (bitonic, shared) and merges them into the beam by rank (merge path)."""
import numpy as np, torch, cupy as cp
dev = "cuda"

SRC = r'''
#define NT 128
#if VHB
#define VIDX(x) ((unsigned)((x) * 2654435761u) >> (32 - VHB))
#else
#define VIDX(x) ((unsigned)(x))
#endif
extern "C" __global__ void walk(const int* __restrict__ qz8, const float* __restrict__ qsc, const float* __restrict__ qc, const float* __restrict__ qmn, const float* __restrict__ qsum,
    const float* __restrict__ seed_s, const int* __restrict__ seed_i,
    const int* __restrict__ G, const int4* __restrict__ Z, const __half* __restrict__ zs,
    unsigned* __restrict__ vis, int NW, int T, int* __restrict__ out_id, float* __restrict__ out_s, int* __restrict__ nev_out, long long* __restrict__ prof)
{
    // L, E, DEG, R, NS, KO, CP (pow2 >= E*DEG) are #defined at compile time
    __shared__ float bs[L]; __shared__ unsigned bi[L];             // beam (descending), bit 31 = expanded
    __shared__ float cs[CP]; __shared__ unsigned ci[CP];           // accepted candidates
    __shared__ float ts[L]; __shared__ unsigned ti[L];             // merge output
    __shared__ int qv[QW]; __shared__ int nodes[E]; __shared__ int ncand, nsel;
    const int q = blockIdx.x, tid = threadIdx.x;
    unsigned* vb = vis + (size_t)q * NW;
    for (int i = tid; i < QW; i += NT) qv[i] = qz8[q * QW + i];
    const float qscale = qsc[q], qcc = qc[q];
    for (int i = tid; i < L; i += NT) {
        if (i < NS) { int id = seed_i[q * NS + i]; bs[i] = seed_s[q * NS + i]; bi[i] = id; {
#if VMODE == 2
                atomicExch(vb + VIDX(id), (unsigned)id);
#else
                unsigned h = VIDX(id); atomicOr(vb + (h >> 5), 1u << (h & 31));
#endif
            } }
        else { bs[i] = -1e30f; bi[i] = 0x7fffffff; }
    }
    __syncthreads();
    int nev = 0; long long c0, p0 = 0, p1 = 0, p2 = 0, p3 = 0, it = 0, nsorted = 0;
    for (int t = 0; t < T; t++) {
        c0 = clock64(); it++;
        // 1) select E best unexpanded (warp 0 ballots, beam already sorted)
        if (tid < 32) {
            int got = 0;
            for (int b = 0; b < L && got < E; b += 32) {
                int i = b + tid; bool un = (i < L) && bs[i] > -1e29f && !(bi[i] & 0x80000000u);
                unsigned m = __ballot_sync(0xffffffff, un);
                while (m && got < E) { int l = __ffs(m) - 1; m &= m - 1;
                    if (tid == 0) { nodes[got] = bi[b + l] & 0x7fffffff; bi[b + l] |= 0x80000000u; }
                    got++; }
            }
            if (tid == 0) { nsel = got; ncand = 0; }
        }
        __syncthreads();
        { long long c1 = clock64(); p0 += c1 - c0; c0 = c1; }
        if (nsel == 0) break;
        float floor_s = bs[L - 1];
        // 2) neighbours: visited test, score new ones (TPC threads per code)
        for (int base = 0; base < nsel * DEG; base += NT / TPC) {
            int j = base + tid / TPC, sub = tid % TPC; int nb = -1; bool isnew = false;
            if (j < nsel * DEG) {
                nb = G[(size_t)nodes[j / DEG] * DEG + (j % DEG)];
                unsigned old = 0;
                unsigned h = VIDX(nb);
#if VMODE == 2
                if (sub == 0) old = atomicExch(vb + h, (unsigned)nb);
                old = __shfl_sync(0xffffffff, old, (tid & 31) & ~(TPC - 1));
                isnew = old != (unsigned)nb;
#else
                if (sub == 0) old = atomicOr(vb + (h >> 5), 1u << (h & 31));
                old = __shfl_sync(0xffffffff, old, (tid & 31) & ~(TPC - 1));
                isnew = !((old >> (h & 31)) & 1u);
#endif
            }
#if SIGN
            float acc = 0.f;
            if (isnew) {
                const int4* zr = Z + (size_t)nb * (R / 128);
                for (int k = sub; k < R / 128; k += TPC) { int4 z = zr[k]; int w4[4] = {z.x, z.y, z.z, z.w};
                    #pragma unroll
                    for (int u = 0; u < 4; u++) { int w = 4 * k + u; unsigned c = (unsigned)w4[u];
                        int pc = 0;
                        #pragma unroll
                        for (int p = 0; p < QP; p++) pc += __popc(c & (unsigned)qv[p * (R / 32) + w]) << p;
                        acc += qmn[q] * __popc(c) + qsc[q] * pc; } }
            }
            #pragma unroll
            for (int o = TPC / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
#else
            int acc = 0;
            if (isnew) {
                const int4* zr = Z + (size_t)nb * (R / 16);
                #pragma unroll
                for (int k = sub; k < R / 16; k += TPC) { int4 z = zr[k];
                    acc = __dp4a(z.x, qv[4 * k], acc); acc = __dp4a(z.y, qv[4 * k + 1], acc);
                    acc = __dp4a(z.z, qv[4 * k + 2], acc); acc = __dp4a(z.w, qv[4 * k + 3], acc); }
            }
            #pragma unroll
            for (int o = TPC / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
#endif
            if (isnew && sub == 0) {
                nev++;
                #if SIGN
                float s = qcc + __half2float(zs[nb]) * (2.f * acc - qsum[q]);
#else
                float s = qcc + __half2float(zs[nb]) * qscale * (float)acc;
#endif
                bool dup = false;
#if VMODE == 2
                if (s > floor_s) for (int u = 0; u < L; u++) dup |= (bi[u] & 0x7fffffff) == (unsigned)nb;
#endif
                if (s > floor_s && !dup) { int p = atomicAdd(&ncand, 1); cs[p] = s; ci[p] = nb; }
            }
        }
        __syncthreads();
        int nc = ncand;
        __syncthreads();
        { long long c1 = clock64(); p1 += c1 - c0; c0 = c1; }
        if (nc == 0) continue;
        nsorted += nc;
        // 3) bitonic sort of accepted candidates (descending), padded to pow2
        int P = 1; while (P < nc) P <<= 1;
        for (int i = nc + tid; i < P; i += NT) { cs[i] = -1e30f; ci[i] = 0x7fffffff; }
        __syncthreads();
        for (int k = 2; k <= P; k <<= 1) for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < P; i += NT) { int l = i ^ jj;
                if (l > i) { bool desc = ((i & k) == 0); float a = cs[i], b = cs[l];
                    if (desc ? (a < b) : (a > b)) { cs[i] = b; cs[l] = a; unsigned x = ci[i]; ci[i] = ci[l]; ci[l] = x; } } }
            __syncthreads(); }
        { long long c1 = clock64(); p2 += c1 - c0; c0 = c1; }
        // 4) merge beam (L) and candidates (nc) by rank, keep top L
        for (int i = tid; i < L; i += NT) {                         // beam element i: rank = i + #cand strictly greater
            float s = bs[i]; int lo = 0, hi = nc; while (lo < hi) { int m = (lo + hi) >> 1; if (cs[m] > s) lo = m + 1; else hi = m; }
            int r = i + lo; if (r < L) { ts[r] = s; ti[r] = bi[i]; } }
        for (int i = tid; i < nc; i += NT) {                        // cand i: rank = i + #beam greater or equal
            float s = cs[i]; int lo = 0, hi = L; while (lo < hi) { int m = (lo + hi) >> 1; if (bs[m] >= s) lo = m + 1; else hi = m; }
            int r = i + lo; if (r < L) { ts[r] = s; ti[r] = ci[i]; } }
        __syncthreads();
        for (int i = tid; i < L; i += NT) { bs[i] = ts[i]; bi[i] = ti[i]; }
        __syncthreads();
        { long long c1 = clock64(); p3 += c1 - c0; c0 = c1; }
    }
    if (tid == 0) { prof[q * 6] = p0; prof[q * 6 + 1] = p1; prof[q * 6 + 2] = p2; prof[q * 6 + 3] = p3; prof[q * 6 + 4] = it; prof[q * 6 + 5] = nsorted; }
    for (int i = tid; i < KO; i += NT) { out_id[q * KO + i] = bi[i] & 0x7fffffff; out_s[q * KO + i] = bs[i]; }
    __shared__ int red; if (tid == 0) red = 0; __syncthreads(); atomicAdd(&red, nev); __syncthreads();
    if (tid == 0) nev_out[q] = red;
}
'''

_mods = {}
def kernel(L, E, DEG, R, NS, KO, SIGN=0, QP=4, TPC=8, VHB=0, VMODE=1):
    key = (L, E, DEG, R, NS, KO, SIGN, QP, TPC, VHB, VMODE)
    if key not in _mods:
        CPn = 1
        while CPn < E * DEG: CPn <<= 1
        defs = f"#include <cuda_fp16.h>\n#define L {L}\n#define E {E}\n#define DEG {DEG}\n#define R {R}\n#define NS {NS}\n#define KO {KO}\n#define CP {CPn}\n"
        defs += f"#define SIGN {SIGN}\n#define QP {QP}\n#define TPC {TPC}\n#define QW {(QP * R // 32) if SIGN else R // 4}\n#define VHB {VHB}\n#define VMODE {VMODE}\n"
        _mods[key] = cp.RawModule(code=defs + SRC, options=("--std=c++17",)).get_function("walk")
    return _mods[key]

class GraphCU:
    def __init__(self, Z, zs, W, c, G, nseed=8192, seed=0, sign=False, QP=4, Zs=None):
        """Z int8 [N,R] (R multiple of 16), zs fp16 [N], W [d,R], c [d], G int32 [N,DEG]."""
        self.Z, self.zs, self.W, self.c, self.G = Z.contiguous(), zs.contiguous(), W, c, G.contiguous(); self.N, self.R = Z.shape; self.DEG = G.shape[1]
        g = torch.Generator().manual_seed(seed); self.smp = torch.randperm(self.N, generator=g)[:nseed].to(dev).int()
        self.sign, self.QP = sign, QP
        if sign: self.R = Z.shape[1] * 32
        self.Zs = Zs[self.smp.long()].half() if Zs is not None else (Z[self.smp.long()].float() * zs[self.smp.long()].float()[:, None]).half()
        self.NW0 = (self.N + 31) // 32; self.vis = {}
    def search(self, Qb, L=128, E=2, T=200, KO=128, NS=64, TPC=8, VHB=0, VMODE=1):
        B = len(Qb)
        self.NW = (1 << VHB) if VMODE == 2 else (1 << VHB) // 32 if VHB else self.NW0
        if (B, VHB, VMODE) not in self.vis: self.vis[(B, VHB, VMODE)] = torch.zeros(B, self.NW, dtype=torch.int32, device=dev)
        vis = self.vis[(B, VHB, VMODE)]; vis.fill_(-1) if VMODE == 2 else vis.zero_()
        qz = Qb @ self.W; qc = (Qb @ self.c).contiguous()
        if self.sign:                                                       # bit planes of the unsigned-quantised query
            mn = qz.amin(1); dl = (qz.amax(1) - mn) / (2 ** self.QP - 1); qu = torch.round((qz - mn[:, None]) / dl[:, None]).long()
            sh = torch.arange(32, device=dev, dtype=torch.int64)
            planes = [(((qu >> p) & 1).reshape(B, -1, 32) << sh).sum(2) for p in range(self.QP)]
            q8 = torch.stack(planes, 1).reshape(B, -1); q8 = torch.where(q8 >= 2 ** 31, q8 - 2 ** 32, q8).int().contiguous()
            qa, qmn, qsum = dl.float().contiguous(), mn.float().contiguous(), qz.sum(1).float().contiguous()
        else:
            qa = qz.abs().amax(1) / 127; q8 = torch.round(qz / qa[:, None]).to(torch.int8).contiguous(); qmn = qsum = qa
        v, i = (qz.half() @ self.Zs.T).float().topk(NS, 1); seeds = self.smp[i].contiguous(); v = (v + qc[:, None]).contiguous()
        oid = torch.empty(B, KO, dtype=torch.int32, device=dev); osc = torch.empty(B, KO, device=dev); nev = torch.empty(B, dtype=torch.int32, device=dev); self.prof = torch.zeros(B, 6, dtype=torch.int64, device=dev)
        f = kernel(L, E, self.DEG, self.R, NS, KO, int(self.sign), self.QP, TPC, VHB, VMODE)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            f((B,), (128,), (q8.data_ptr(), qa.float().contiguous().data_ptr(), qc.data_ptr(), qmn.data_ptr(), qsum.data_ptr(), v.data_ptr(), seeds.data_ptr(), self.G.data_ptr(),
                             self.Z.data_ptr(), self.zs.data_ptr(), vis.data_ptr(), np.int32(self.NW), np.int32(T), oid.data_ptr(), osc.data_ptr(), nev.data_ptr(), self.prof.data_ptr()))
        return oid, osc, nev
