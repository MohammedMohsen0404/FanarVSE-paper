"""Alternative trellis decode: load each row's words once (BN x WP) and build the windows in registers."""
import torch, triton, triton.language as tl
import trellis_scan

@triton.jit
def _decode_tile2(words_ptr, rn, nmask, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, tab_ptr):
    j = tl.arange(0, RP); jm = j < R
    wi = j // 32; off = j % 32
    w0 = tl.zeros((rn.shape[0], RP), dtype=tl.int64); w1 = tl.zeros((rn.shape[0], RP), dtype=tl.int64)
    for w in tl.static_range(W):
        col = tl.load(words_ptr + rn * W + w, mask=nmask, other=0).to(tl.int64) & 0xFFFFFFFF
        w0 = tl.where(wi[None, :] == w, col[:, None], w0)
        w1 = tl.where(wi[None, :] + 1 == w, col[:, None], w1)
    comb = (w0 << 32) | w1
    win = (comb >> (64 - L - off[None, :])) & ((1 << L) - 1)
    m2 = nmask[:, None] & jm[None, :]
    return tl.where(m2, tl.load(tab_ptr + win, mask=m2, other=0.0), 0.0)
