"""CUDA (int8 WMMA) kernels for the IVF int8 stage 1 of EngineI8: same inputs and outputs as the Triton _first8c / _emit8c.
One block = one (cell, block of BN=32 vectors); loops over tiles of 64 queries probing the cell (4 warps x 16 rows), scores to shared
memory, then MODE 0 writes per-query block maxima and MODE 1 emits candidates above the threshold (atomics only for kept pairs)."""
import os as _os, sys as _sys
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch, cupy as cp
dev = "cuda"
SRC = r'''
#include <mma.h>
using namespace nvcuda;
#define R 256
#define BN 32
#define BQ 64
#define RS (R + 16)
extern "C" __global__ void __launch_bounds__(128) ivf8(const signed char* __restrict__ Q, const float* __restrict__ qs, const int* __restrict__ qrows,
    const int* __restrict__ c_qoff, const int* __restrict__ c_qn, const int* __restrict__ p_cell, const int* __restrict__ p_blk,
    const int* __restrict__ p_s, const int* __restrict__ p_e, const signed char* __restrict__ Z, const float* __restrict__ zs,
    const float* __restrict__ qcp, const float* __restrict__ t0, int* __restrict__ cnt, float* __restrict__ bval, int* __restrict__ bpos,
    float* __restrict__ bmc, int CAP, int MB)
{
    __shared__ __align__(16) signed char zsm[BN * RS];
    __shared__ __align__(16) signed char qsm[BQ * RS];
    __shared__ __align__(16) int accsm[4][16 * BN];
    __shared__ float zsc[BN]; __shared__ int qid_s[BQ]; __shared__ float qsc_s[BQ], qcp_s[BQ], thr_s[BQ];
    const int p = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int c = p_cell[p], qn = c_qn[c]; if (qn == 0) return;
    const int qoff = c_qoff[c], blk = p_blk[p], cs = p_s[p], ce = p_e[p]; const long n0 = (long)blk * BN;
    for (int i = tid; i < BN * R / 16; i += 128) {
        int row = i / (R / 16), col = i % (R / 16); long n = n0 + row;
        int4 v = make_int4(0, 0, 0, 0); if (n >= cs && n < ce) v = ((const int4*)(Z + n * R))[col];
        *(int4*)(zsm + row * RS + col * 16) = v;
    }
    if (tid < BN) { long n = n0 + tid; zsc[tid] = (n >= cs && n < ce) ? zs[n] : 0.f; }
    for (int q0 = 0; q0 < qn; q0 += BQ) {
        __syncthreads();
        if (tid < BQ) {
            int qi = q0 + tid;
            if (qi < qn) { int qr = qrows[qoff + qi]; qid_s[tid] = qr; qsc_s[tid] = qs[qr]; qcp_s[tid] = qcp[qoff + qi]; thr_s[tid] = MODE ? t0[qr] : 0.f; }
            else { qid_s[tid] = -1; qsc_s[tid] = 0.f; qcp_s[tid] = 0.f; thr_s[tid] = 3.0e38f; }
        }
        __syncthreads();
        for (int i = tid; i < BQ * R / 16; i += 128) {
            int row = i / (R / 16), col = i % (R / 16); int qr = qid_s[row];
            int4 v = make_int4(0, 0, 0, 0); if (qr >= 0) v = ((const int4*)(Q + (long)qr * R))[col];
            *(int4*)(qsm + row * RS + col * 16) = v;
        }
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
#if MODE == 1
        #pragma unroll 4
        for (int e = lane; e < 16 * BN; e += 32) {
            int r = w * 16 + e / BN, col = e % BN; int qr = qid_s[r];
            if (qr >= 0 && zsc[col] != 0.f) {
                float est = (float)accsm[w][e] * qsc_s[r] * zsc[col] + qcp_s[r];
                if (est >= thr_s[r]) { int idx = atomicAdd(cnt + qr, 1); if (idx < CAP) { bval[(long)qr * CAP + idx] = est; bpos[(long)qr * CAP + idx] = (int)(n0 + col); } }
            }
        }
#else
        if (lane < 16) {
            int r = w * 16 + lane; int qr = qid_s[r];
            if (qr >= 0) {
                float m = -1e30f;
                for (int col = 0; col < BN; col++) if (zsc[col] != 0.f) m = fmaxf(m, (float)accsm[w][lane * BN + col] * qsc_s[r] * zsc[col] + qcp_s[r]);
                bmc[(long)(qoff + q0 + r) * MB + (blk - cs / BN)] = m;
            }
        }
#endif
    }
}
'''
_mods = {}
def kern(mode):
    if mode not in _mods:
        _mods[mode] = cp.RawModule(code="#define MODE %d\n" % mode + SRC, options=("--std=c++17", f"-I{_sys.prefix}/Lib/site-packages/nvidia/cuda_nvcc/include")).get_function("ivf8")
    return _mods[mode]
def launch(mode, P, q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, Z, zs, qcp, t0, cnt, bval, bpos, bmc, CAP, MB):
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        kern(mode)((P,), (128,), tuple(x.data_ptr() for x in (q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, Z, zs, qcp, t0, cnt, bval, bpos, bmc)) + (np.int32(CAP), np.int32(MB)))
