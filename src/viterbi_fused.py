"""Fused Viterbi encoder for the bitshift trellis (k=1): one Triton program per vector, path costs in a double-buffered
scratch row (L1/L2), backpointers packed one bit per state, backtracking in the same kernel. Same convention as
trellis_scan.viterbi_paths: predecessors of state s are (s >> 1) | (u << (L-1)), u in {0, 1}; the value on entering s is vals[s]."""
import torch, triton, triton.language as tl
dev = "cuda"

@triton.jit
def _viterbi(x_ptr, vals_ptr, scr_ptr, back_ptr, path_ptr, n, L: tl.constexpr, S: tl.constexpr, SW: tl.constexpr):
    v = tl.program_id(0)
    s = tl.arange(0, S); vals = tl.load(vals_ptr + s)
    sa = scr_ptr + v * 2 * S; sb = sa + S
    x0 = tl.load(x_ptr + v * n)
    tl.store(sa + s, (x0 - vals) * (x0 - vals))
    tl.debug_barrier()
    p0 = s >> 1; p1 = p0 | (1 << (L - 1))
    bw = tl.arange(0, SW)
    for j in range(1, n):
        src = sa + ((j - 1) % 2) * S; dst = sa + (j % 2) * S
        c0 = tl.load(src + p0); c1 = tl.load(src + p1)
        take1 = c1 < c0
        xj = tl.load(x_ptr + v * n + j)
        tl.store(dst + s, tl.where(take1, c1, c0) + (xj - vals) * (xj - vals))
        bits = tl.reshape(take1.to(tl.int32), (SW, 32))
        word = tl.sum(bits << tl.arange(0, 32)[None, :], axis=1)
        tl.store(back_ptr + (v * (n - 1) + (j - 1)) * SW + bw, word)
        tl.debug_barrier()
    last = sa + ((n - 1) % 2) * S
    c = tl.load(last + s)
    m = tl.min(c, axis=0)
    st = tl.min(tl.where(c == m, s, S), axis=0)
    tl.store(path_ptr + v * n + n - 1, st)
    for jj in range(1, n):
        j = n - jj
        w = tl.load(back_ptr + (v * (n - 1) + (j - 1)) * SW + st // 32)
        u = (w >> (st % 32)) & 1
        st = (st >> 1) | (u << (L - 1))
        tl.store(path_ptr + v * n + j - 1, st)

def viterbi_paths_fused(Xs, T, L, alpha, chunk=8192):
    B, n = Xs.shape; S = 2 ** L; SW = S // 32; vals = (alpha * T).float().contiguous()
    path = torch.empty(B, n, dtype=torch.int32, device=dev)
    for i in range(0, B, chunk):
        X = Xs[i:i + chunk].float().contiguous(); b = len(X)
        scr = torch.empty(b, 2, S, device=dev); back = torch.empty(b, n - 1, SW, dtype=torch.int32, device=dev)
        _viterbi[(b,)](X, vals, scr, back, path[i:i + b], n, L=L, S=S, SW=SW, num_warps=8)
    return path.long()
