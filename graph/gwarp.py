"""Graph stage 1, one warp per query (no block barriers). Each step: the warp picks the E best unexpanded beam entries (ballot),
each lane takes E*DEG/32 neighbours (visited bitmap atomicOr, int8 dp4a score of the full code), the candidates above the beam floor
are sorted with a register/shuffle bitonic network and merged into the shared-memory beam by rank."""
import numpy as np, torch, cupy as cp
dev = "cuda"

SRC = r'''
#include <cuda_fp16.h>
#define C (E * DEG / 32)
#define CPAD (C == 3 ? 4 : C)
extern "C" __global__ void __launch_bounds__(WPB * 32) walkw(const int* __restrict__ qz8, const float* __restrict__ qsc, const float* __restrict__ qc,
    const float* __restrict__ seed_s, const int* __restrict__ seed_i, const int* __restrict__ G, const int4* __restrict__ Z,
    const __half* __restrict__ zs, unsigned* __restrict__ vis, int NW, int T, int B, int* __restrict__ out_id, float* __restrict__ out_s,
    int* __restrict__ nev_out)
{
    __shared__ float bsm[WPB][L]; __shared__ unsigned bim[WPB][L]; __shared__ int qvm[WPB][R / 4];
    __shared__ float csm[WPB][32 * CPAD];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, q = blockIdx.x * WPB + w;
    if (q >= B) return;
    float* bs = bsm[w]; unsigned* bi = bim[w]; int* qv = qvm[w]; float* cs = csm[w];
    unsigned* vb = vis + (size_t)q * NW;
    for (int i = lane; i < R / 4; i += 32) qv[i] = qz8[q * (R / 4) + i];
    const float qscale = qsc[q], qcc = qc[q];
    for (int i = lane; i < L; i += 32) {
        if (i < NS) { int id = seed_i[q * NS + i]; bs[i] = seed_s[q * NS + i]; bi[i] = id; atomicOr(vb + (id >> 5), 1u << (id & 31)); }
        else { bs[i] = -1e30f; bi[i] = 0x7fffffff; }
    }
    __syncwarp();
    int nev = 0;
    for (int t = 0; t < T; t++) {
        int nodes[E]; int npos[E]; int got = 0;
        for (int b = 0; b < L && got < E; b += 32) {
            int i = b + lane; bool un = bs[i] > -1e29f && !(bi[i] & 0x80000000u);
            unsigned m = __ballot_sync(0xffffffff, un);
            while (m && got < E) { int l = __ffs(m) - 1; m &= m - 1;
                nodes[got] = (int)(bi[b + l] & 0x7fffffff); npos[got] = b + l; got++; }
        }
        __syncwarp();
        if (lane == 0) for (int e = 0; e < got; e++) bi[npos[e]] |= 0x80000000u;
        __syncwarp();
        if (got == 0) break;
        const float floor_s = bs[L - 1];
        float v[CPAD]; unsigned id[CPAD];
        #pragma unroll
        for (int r = 0; r < CPAD; r++) {
            v[r] = -1e30f; id[r] = 0x7fffffff;
            int j = r * 32 + lane;
            if (r < C && j / DEG < got) {
                int nb = G[(size_t)nodes[j / DEG] * DEG + (j % DEG)];
                unsigned old = atomicOr(vb + (nb >> 5), 1u << (nb & 31));
                if (!((old >> (nb & 31)) & 1u)) {
                    nev++;
                    const int4* zr = Z + (size_t)nb * (R / 16); int acc = 0;
                    #pragma unroll
                    for (int k = 0; k < R / 16; k++) { int4 z = zr[k];
                        acc = __dp4a(z.x, qv[4 * k], acc); acc = __dp4a(z.y, qv[4 * k + 1], acc);
                        acc = __dp4a(z.z, qv[4 * k + 2], acc); acc = __dp4a(z.w, qv[4 * k + 3], acc); }
                    float s = qcc + __half2float(zs[nb]) * qscale * (float)acc;
                    if (s > floor_s) { v[r] = s; id[r] = nb; }
                }
            }
        }
        // any accepted?
        bool any = false;
        #pragma unroll
        for (int r = 0; r < CPAD; r++) any |= v[r] > -1e29f;
        if (!__any_sync(0xffffffff, any)) continue;
        // bitonic sort (descending) of 32*CPAD elements; element e = r*32 + lane
        #pragma unroll
        for (int k = 2; k <= 32 * CPAD; k <<= 1) {
            #pragma unroll
            for (int jj = k >> 1; jj > 0; jj >>= 1) {
                if (jj >= 32) {
                    const int rj = jj >> 5;
                    #pragma unroll
                    for (int r = 0; r < CPAD; r++) { int rp = r ^ rj;
                        if (rp > r) { int e = r * 32 + lane; bool desc = ((e & k) == 0);
                            if (desc ? (v[r] < v[rp]) : (v[r] > v[rp])) { float a = v[r]; v[r] = v[rp]; v[rp] = a; unsigned x = id[r]; id[r] = id[rp]; id[rp] = x; } } }
                } else {
                    #pragma unroll
                    for (int r = 0; r < CPAD; r++) {
                        float ov = __shfl_xor_sync(0xffffffff, v[r], jj); unsigned oi = __shfl_xor_sync(0xffffffff, id[r], jj);
                        int e = r * 32 + lane; bool desc = ((e & k) == 0); bool lower = ((lane & jj) == 0);
                        bool takeMax = (desc == lower);
                        bool swap = takeMax ? (ov > v[r]) : (ov < v[r]);
                        if (swap) { v[r] = ov; id[r] = oi; } }
                }
            }
        }
        int nc = 0;
        #pragma unroll
        for (int r = 0; r < CPAD; r++) nc += __popc(__ballot_sync(0xffffffff, v[r] > -1e29f));
        #pragma unroll
        for (int r = 0; r < CPAD; r++) cs[r * 32 + lane] = v[r];
        __syncwarp();
        // ranks: beam element i -> i + #cands > bs[i]; cand j -> j + #beam >= cs[j]
        float bv[L / 32]; unsigned bid[L / 32]; int br[L / 32];
        #pragma unroll
        for (int u = 0; u < L / 32; u++) { int i = u * 32 + lane; float s = bs[i]; bv[u] = s; bid[u] = bi[i];
            int lo = 0, hi = nc; while (lo < hi) { int m = (lo + hi) >> 1; if (cs[m] > s) lo = m + 1; else hi = m; } br[u] = i + lo; }
        int cr[CPAD];
        #pragma unroll
        for (int r = 0; r < CPAD; r++) { int j = r * 32 + lane; cr[r] = L;
            if (j < nc) { float s = v[r]; int lo = 0, hi = L; while (lo < hi) { int m = (lo + hi) >> 1; if (bs[m] >= s) lo = m + 1; else hi = m; } cr[r] = j + lo; } }
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < L / 32; u++) if (br[u] < L) { bs[br[u]] = bv[u]; bi[br[u]] = bid[u]; }
        #pragma unroll
        for (int r = 0; r < CPAD; r++) if (cr[r] < L) { bs[cr[r]] = v[r]; bi[cr[r]] = id[r]; }
        __syncwarp();
    }
    for (int i = lane; i < KO; i += 32) { out_id[q * KO + i] = bi[i] & 0x7fffffff; out_s[q * KO + i] = bs[i]; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) nev += __shfl_xor_sync(0xffffffff, nev, o);
    if (lane == 0) nev_out[q] = nev;
}
'''

_mods = {}
def kernel(L, E, DEG, R, NS, KO, WPB):
    key = (L, E, DEG, R, NS, KO, WPB)
    if key not in _mods:
        defs = f"#define L {L}\n#define E {E}\n#define DEG {DEG}\n#define R {R}\n#define NS {NS}\n#define KO {KO}\n#define WPB {WPB}\n"
        _mods[key] = cp.RawModule(code=defs + SRC, options=("--std=c++17",)).get_function("walkw")
    return _mods[key]

class GraphW:
    def __init__(self, Z, zs, W, c, G, nseed=8192, seed=0, **kw):
        self.Z, self.zs, self.W, self.c, self.G = Z.contiguous(), zs.contiguous(), W, c, G.contiguous(); self.N, self.R = Z.shape; self.DEG = G.shape[1]
        g = torch.Generator().manual_seed(seed); self.smp = torch.randperm(self.N, generator=g)[:nseed].to(dev).int()
        self.Zs = (Z[self.smp.long()].float() * zs[self.smp.long()].float()[:, None]).half()
        self.NW = (self.N + 31) // 32; self.vis = {}
    def search(self, Qb, L=256, E=2, T=1000, KO=128, NS=64, WPB=4, **kw):
        B = len(Qb)
        if B not in self.vis: self.vis[B] = torch.zeros(B, self.NW, dtype=torch.int32, device=dev)
        vis = self.vis[B]; vis.zero_()
        qz = Qb @ self.W; qc = (Qb @ self.c).contiguous()
        qa = (qz.abs().amax(1) / 127).float().contiguous(); q8 = torch.round(qz / qa[:, None]).to(torch.int8).contiguous()
        v, i = (qz.half() @ self.Zs.T).float().topk(NS, 1); seeds = self.smp[i].contiguous(); v = (v + qc[:, None]).contiguous()
        oid = torch.empty(B, KO, dtype=torch.int32, device=dev); osc = torch.empty(B, KO, device=dev); nev = torch.empty(B, dtype=torch.int32, device=dev)
        f = kernel(L, E, self.DEG, self.R, NS, KO, WPB)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            f(((B + WPB - 1) // WPB,), (32 * WPB,), (q8.data_ptr(), qa.data_ptr(), qc.data_ptr(), v.data_ptr(), seeds.data_ptr(), self.G.data_ptr(),
              self.Z.data_ptr(), self.zs.data_ptr(), vis.data_ptr(), np.int32(self.NW), np.int32(T), np.int32(B), oid.data_ptr(), osc.data_ptr(), nev.data_ptr()))
        return oid, osc, nev
