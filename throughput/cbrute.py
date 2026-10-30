"""Routing-free stage 1 in CUDA: int8 WMMA (m16n16k16) scan of all vectors with R=128 PCA codes, emit with a plain branch
(atomics only for kept pairs). Block = 4 warps, 64 vectors; loops over query tiles of 64. Threshold from a 5% sample (fp16 GEMM).
Also usable on a subset of rows (IVF-style) via a row-range list."""
import os as _os, sys as _sys
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch, cupy as cp
dev = "cuda"
SRC = r'''
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define R 128
#ifndef BN
#define BN 64
#endif
#define BQ 64
#ifndef PAD
#define PAD 16
#endif
#define RS (R + PAD)
#define NW (BN / 16)
extern "C" __global__ void __launch_bounds__(BQ / 16 * 32) scan(const signed char* __restrict__ Q, const float* __restrict__ qs, const signed char* __restrict__ Z,
    const __half* __restrict__ zs, const float* __restrict__ t0, int* __restrict__ cnt, float* __restrict__ bval, int* __restrict__ bpos,
    int Bpad, int B, int N, int CAP)
{
    __shared__ __align__(16) signed char zsm[BN * RS];
    __shared__ __align__(16) signed char qsm[BQ * RS];
    __shared__ __align__(16) int accsm[BQ / 16][16 * BN];
    __shared__ float zsc[BN];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31; const long n0 = (long)blockIdx.x * BN;
    for (int i = tid; i < BN * R / 16; i += blockDim.x) {
        int row = i / (R / 16); long gr = n0 + row;
        int4 v = make_int4(0, 0, 0, 0); if (gr < N) v = ((const int4*)(Z + gr * R))[i % (R / 16)];
        *(int4*)(zsm + row * RS + (i % (R / 16)) * 16) = v;
    }
    for (int t = tid; t < BN; t += blockDim.x) zsc[t] = (n0 + t < N) ? __half2float(zs[n0 + t]) : 0.f;
    for (int q0 = 0; q0 < Bpad; q0 += BQ) {
        __syncthreads();
        for (int i = tid; i < BQ * R / 16; i += blockDim.x) *(int4*)(qsm + (i / (R / 16)) * RS + (i % (R / 16)) * 16) = ((const int4*)(Q + (long)q0 * R))[i];
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < BN / 16; nt++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc; wmma::fill_fragment(acc, 0);
            #pragma unroll
            for (int k = 0; k < R / 16; k++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
                wmma::load_matrix_sync(a, qsm + (w * 16) * RS + k * 16, RS);
                wmma::load_matrix_sync(b, zsm + (nt * 16) * RS + k * 16, RS);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(accsm[w] + nt * 16, acc, BN, wmma::mem_row_major);
        }
        __syncwarp();
        #pragma unroll 4
        for (int e = lane; e < 16 * BN; e += 32) {
            int r = e / BN, c = e % BN; int qi = q0 + w * 16 + r; long n = n0 + c;
            if (qi < B && n < N) {
                float est = (float)accsm[w][e] * qs[qi] * zsc[c];
                if (est >= t0[qi]) { int idx = atomicAdd(cnt + qi, 1); if (idx < CAP) { bval[(long)qi * CAP + idx] = est; bpos[(long)qi * CAP + idx] = (int)n; } }
            }
        }
    }
}
'''
_mods = {}
def kern(BN=64, PAD=16):
    if (BN, PAD) not in _mods:
        hdr = "#define BN %d" % BN + chr(10) + "#define PAD %d" % PAD + chr(10)
        _mods[(BN, PAD)] = cp.RawModule(code=hdr + SRC, options=("--std=c++17", f"-I{_sys.prefix}/Lib/site-packages/nvidia/cuda_nvcc/include")).get_function("scan")
    return _mods[(BN, PAD)]

class CBrute:
    def __init__(self, X16, mu, V, R=128, frac=0.05, chunk=65536):
        assert R == 128; self.R, self.mu, self.V = R, mu, V[:, :R].contiguous(); N = len(X16); self.N = N
        self.Z = torch.empty(N, R, dtype=torch.int8, device=dev); self.zs = torch.empty(N, dtype=torch.float16, device=dev)
        for j in range(0, N, chunk):
            z = (X16[j:j + chunk].to(dev).float() - mu) @ self.V; a = z.abs().amax(1).clamp(min=1e-12) / 127
            self.Z[j:j + chunk] = torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8); self.zs[j:j + chunk] = a.half()
        g = torch.Generator(device=dev).manual_seed(0); self.sidx = torch.randperm(N, generator=g, device=dev)[:int(frac * N)]
        self.Sf = (self.Z[self.sidx].float() * self.zs[self.sidx].float()[:, None]).half().T.contiguous(); self.frac = frac
    def qcode(self, Qb):
        z = (Qb - self.mu) @ self.V; a = z.abs().amax(1).clamp(min=1e-12) / 127
        return torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8), a.float(), z
    def search(self, Qb, K, over=2.0, M=None, BN=64, PAD=16, only_kernel=False):
        B = len(Qb); Bp = (B + 63) // 64 * 64; M = M or max(4096, 8 * K)
        q8, qs, z = self.qcode(Qb)
        qp = torch.zeros(Bp, self.R, dtype=torch.int8, device=dev); qp[:B] = q8; qsp = torch.zeros(Bp, device=dev); qsp[:B] = qs
        r = max(1, int(round(over * K * self.frac)))
        t0 = (z.half() @ self.Sf).float().topk(r, 1).values[:, -1].contiguous()
        cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            kern(BN, PAD)(((self.N + BN - 1) // BN,), (128,), (qp.data_ptr(), qsp.data_ptr(), self.Z.data_ptr(), self.zs.data_ptr(), t0.data_ptr(), cnt.data_ptr(),
                   bval.data_ptr(), bpos.data_ptr(), np.int32(Bp), np.int32(B), np.int32(self.N), np.int32(M)))
        t = bval.topk(K, 1); return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= M), cnt
