"""Trellis stage-1 on GPU. Stream per vector: bits[0:L] = initial state, then one bit per coordinate (k=1);
state_j = bits[j : j+L] (MSB-first) -> value_j = alpha * T[state_j]. Stored as uint32 words (big-endian bit order).
Kernel: decode windows in registers -> fp16 tile -> tensor-core dot -> per-block max (exact block-max selection)."""
import torch, triton, triton.language as tl
from trellis import viterbi, make_table

dev = "cuda"


def paths_to_words(path, L):
    """path: (B, r) int states -> stream bits (B, r+L-1) -> uint32 words (B, W) as int32 tensor."""
    B, r = path.shape; nbits = r + L - 1; W = (nbits + 31) // 32
    bits = torch.zeros(B, W * 32, dtype=torch.int64, device=path.device)
    sh = torch.arange(L - 1, -1, -1, device=path.device)
    bits[:, :L] = (path[:, :1] >> sh) & 1                                          # initial state bits (MSB first)
    bits[:, L:nbits] = path[:, 1:] & 1                                               # one new bit per coordinate
    w = (bits.reshape(B, W, 32) << torch.arange(31, -1, -1, device=path.device)).sum(2)
    return (w & 0xFFFFFFFF).to(torch.int64)


@torch.no_grad()
def viterbi_paths(Xs, T, L, alpha):
    """same as trellis.viterbi (k=1) but returns the state path."""
    B, n = Xs.shape; S = 2 ** L; vals = alpha * T; sp = torch.arange(S, device=dev)
    preds = torch.stack([(sp >> 1) | (u << (L - 1)) for u in range(2)], 1)
    cost = (Xs[:, :1] - vals[None]) ** 2; back = torch.empty(n - 1, B, S, dtype=torch.uint8, device=dev)
    for j in range(1, n):
        best, arg = cost[:, preds].min(2); back[j - 1] = arg.to(torch.uint8); cost = best + (Xs[:, j:j + 1] - vals[None]) ** 2
    s = cost.argmin(1); path = torch.empty(B, n, dtype=torch.long, device=dev); path[:, -1] = s
    for j in range(n - 1, 0, -1):
        s = preds[s, back[j - 1].gather(1, s[:, None])[:, 0].long()]; path[:, j - 1] = s
    return path


@triton.jit
def _decode_tile(words_ptr, rn, nmask, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, tab_ptr):
    j = tl.arange(0, RP); jm = j < R
    wi = j // 32; off = j % 32
    m2 = nmask[:, None] & jm[None, :]
    w0 = tl.load(words_ptr + rn[:, None] * W + wi[None, :], mask=m2, other=0).to(tl.int64) & 0xFFFFFFFF
    w1 = tl.load(words_ptr + rn[:, None] * W + wi[None, :] + 1, mask=m2 & (wi[None, :] + 1 < W), other=0).to(tl.int64) & 0xFFFFFFFF
    comb = (w0 << 32) | w1
    win = (comb >> (64 - L - off[None, :])) & ((1 << L) - 1)
    return tl.where(m2, tl.load(tab_ptr + win, mask=m2, other=0.0), 0.0)


@triton.jit
def _trellis_blockmax(q_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, out_ptr, B, N,
                      W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NB):
    pid_n = tl.program_id(0); pid_q = tl.program_id(1)
    rq = pid_q * BQ + tl.arange(0, BQ); rn = pid_n * BN + tl.arange(0, BN); qmask = rq < B; nmask = rn < N
    j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    tl.store(out_ptr + rq * NB + pid_n, tl.max(est, axis=1), mask=qmask)


@triton.jit
def _trellis_rescore(q_ptr, pos_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, out_ptr, B, M,
                     W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BM: tl.constexpr):
    qi = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    rows = tl.load(pos_ptr + qi * M + rm, mask=mmask, other=0)
    v = _decode_tile(words_ptr, rows, mmask, W, R, RP, L, tab_ptr)
    j = tl.arange(0, RP); qrow = tl.load(q_ptr + qi * R + j, mask=j < R, other=0.0)
    acc = tl.sum(v * qrow[None, :], axis=1)
    s = tl.load(scale_ptr + rows, mask=mmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + qi * M + rm, acc * s + tl.load(qc_ptr + qi), mask=mmask)


def trellis_search(Qr, qc, words, tab, scale, L, kp, BN=32, BQ=32):
    """Qr: (B, R) queries in the rotated PCA-r basis; exact top-kp by estimate (positions)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    bm = torch.empty(B, NB, device=Qr.device, dtype=torch.float32)
    _trellis_blockmax[(NB, triton.cdiv(B, BQ))](Qr, words, tab, scale, qc, bm, B, N, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, NB=NB)
    blocks = bm.topk(kp, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=Qr.device)).reshape(B, -1).clamp(max=N - 1).contiguous(); M = pos.shape[1]
    est = torch.empty(B, M, device=Qr.device, dtype=torch.float32)
    _trellis_rescore[(triton.cdiv(M, 32), B)](Qr, pos, words, tab, scale, qc, est, B, M, W=W, R=R, RP=RP, L=L, BM=32)
    t = est.topk(kp, 1)
    return (pos.gather(1, t.indices), t.values) if return_est else pos.gather(1, t.indices), est.topk(kp, 1).values


@triton.jit
def _trellis_cell(q_ptr, qrows_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, out_ptr, B, NQS, N, blk0,
                  W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NB):
    pid_n = tl.program_id(0); pid_q = tl.program_id(1); bidx = blk0 + pid_n
    rqi = pid_q * BQ + tl.arange(0, BQ); qmask = rqi < NQS; rq = tl.load(qrows_ptr + rqi, mask=qmask, other=0)
    rn = bidx * BN + tl.arange(0, BN); nmask = rn < N; j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    tl.store(out_ptr + rq * NB + bidx, tl.max(est, axis=1), mask=qmask)


def trellis_search_ivf(Qr, Qorig, qc, words, tab, scale, L, kp, cen, off, nprobe, BN=32, BQ=32, return_est=False):
    """words/scale sorted by cell; returns positions (sorted order) of exact top-kp estimates over the probed cells."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    bm = torch.full((B, NB), -1e30, device=Qr.device, dtype=torch.float32)
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices; flat = probe.flatten(); qid = torch.arange(B, device=Qr.device).repeat_interleave(nprobe)
    o = torch.argsort(flat); flat, qid = flat[o], qid[o]
    cells, counts = torch.unique_consecutive(flat, return_counts=True); starts = (torch.cumsum(counts, 0) - counts).tolist()
    for cc, st, ct in zip(cells.tolist(), starts, counts.tolist()):
        s, e = int(off[cc]), int(off[cc + 1])
        if e == s: continue
        b0 = s // BN; nb = (e - 1) // BN - b0 + 1
        _trellis_cell[(nb, triton.cdiv(ct, BQ))](Qr, qid[st:st + ct].contiguous(), words, tab, scale, qc, bm, B, ct, N, b0,
                                                 W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, NB=NB)
    blocks = bm.topk(kp, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=Qr.device)).reshape(B, -1).clamp(max=N - 1).contiguous(); M = pos.shape[1]
    est = torch.empty(B, M, device=Qr.device, dtype=torch.float32)
    _trellis_rescore[(triton.cdiv(M, 32), B)](Qr, pos, words, tab, scale, qc, est, B, M, W=W, R=R, RP=RP, L=L, BM=32)
    t = est.topk(kp, 1)
    return (pos.gather(1, t.indices), t.values) if return_est else pos.gather(1, t.indices)
