"""Faster stage 1: (1) one launch per batch over a (cell, query-tile) work list, so many cells cost no launch overhead;
(2) the scan keeps the top-2 estimates (value, offset) of every block instead of the maximum only, and the top-K is read from these entries.
Exactness (extends the block-max lemma): let t' be the K-th largest stored value. A block can hold a third item at or above t' only if its
second stored value is >= t'; those blocks are rescored in full and merged. The result is the exact top-K of the estimates (ties aside)."""
import numpy as np, torch, triton, triton.language as tl
from trellis_scan import _decode_tile, _trellis_rescore
dev = "cuda"

@triton.jit
def _cell_top2(q_ptr, qrows_ptr, t_qoff_ptr, t_qn_ptr, t_blk0_ptr, t_nb_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr,
               v1_ptr, v2_ptr, i1_ptr, i2_ptr, N, NB,
               W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    pid_b = tl.program_id(0); task = tl.program_id(1)
    nb = tl.load(t_nb_ptr + task); valid = pid_b < nb
    blk = tl.load(t_blk0_ptr + task) + pid_b; qoff = tl.load(t_qoff_ptr + task); qn = tl.load(t_qn_ptr + task)
    rqi = tl.arange(0, BQ); qmask = (rqi < qn) & valid; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn < N) & valid; j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    m1 = tl.max(est, axis=1); a1 = tl.argmax(est, axis=1)
    cols = tl.arange(0, BN)
    est2 = tl.where(cols[None, :] == a1[:, None], -1e30, est)
    m2 = tl.max(est2, axis=1); a2 = tl.argmax(est2, axis=1)
    o = rq * NB + blk
    tl.store(v1_ptr + o, m1, mask=qmask); tl.store(v2_ptr + o, m2, mask=qmask)
    tl.store(i1_ptr + o, a1.to(tl.int32), mask=qmask); tl.store(i2_ptr + o, a2.to(tl.int32), mask=qmask)

def worklist(probe, off_t, BN, BQ):
    """probe (B, np) cells per query -> per-task arrays (sorted query rows, query offset/count, first block, number of blocks)."""
    B, npr = probe.shape; flat = probe.flatten(); qid = torch.arange(B, device=dev).repeat_interleave(npr)
    o = torch.argsort(flat); flat, qid = flat[o], qid[o].to(torch.int32).contiguous()
    cells, counts = torch.unique_consecutive(flat, return_counts=True); starts = torch.cumsum(counts, 0) - counts
    ntile = (counts + BQ - 1) // BQ; T = int(ntile.sum())
    tc = torch.repeat_interleave(torch.arange(len(cells), device=dev), ntile); first = torch.cumsum(ntile, 0) - ntile
    ti = torch.arange(T, device=dev) - first[tc]
    qoff = (starts[tc] + ti * BQ).to(torch.int32); qn = torch.minimum(counts[tc] - ti * BQ, torch.tensor(BQ, device=dev)).to(torch.int32)
    s, e = off_t[cells[tc]], off_t[cells[tc] + 1]; blk0 = (s // BN).to(torch.int32); nb = torch.where(e > s, (e - 1) // BN - s // BN + 1, 0).to(torch.int32)
    return qid, qoff.contiguous(), qn.contiguous(), blk0.contiguous(), nb.contiguous()

def search(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, BN=32, BQ=32, exact=True, stats=None):
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    v1 = torch.full((B, NB), -1e30, device=dev); v2 = torch.full((B, NB), -1e30, device=dev)
    i1 = torch.zeros((B, NB), dtype=torch.int32, device=dev); i2 = torch.zeros((B, NB), dtype=torch.int32, device=dev)
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    qrows, qoff, qn, blk0, nb = worklist(probe, off_t, BN, BQ); T = len(qoff); mx = int(nb.max())
    _cell_top2[(mx, T)](Qr, qrows, qoff, qn, blk0, nb, words, tab, scale, qc, v1, v2, i1, i2, N, NB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN)
    vals = torch.cat([v1, v2], 1); top = vals.topk(K, 1); j = top.indices
    blk = torch.where(j < NB, j, j - NB); offs = torch.where(j < NB, i1.gather(1, blk), i2.gather(1, blk.clamp(max=NB - 1)))
    pos = blk * BN + offs; est = top.values
    if exact:
        t = est[:, -1:]; flag = v2 >= t                                                # blocks that may hold a third item >= t
        nf = flag.sum(1)
        if stats is not None: stats.append(nf.float().cpu())
        F = int(nf.max())
        if F > 0:
            fb = flag.float().topk(F, 1); fblk = torch.where(fb.values > 0, fb.indices, -1)          # (B, F) flagged block ids or -1
            fpos = (fblk[:, :, None].clamp(min=0) * BN + torch.arange(BN, device=dev)).reshape(B, -1).clamp(max=N - 1).contiguous()
            fest = torch.empty(B, fpos.shape[1], device=dev)
            _trellis_rescore[(triton.cdiv(fpos.shape[1], 32), B)](Qr, fpos, words, tab, scale, qc, fest, B, fpos.shape[1], W=W, R=R, RP=RP, L=L, BM=32)
            fest = torch.where((fblk[:, :, None] >= 0).expand(B, F, BN).reshape(B, -1), fest, torch.full_like(fest, -1e30))
            inflag = flag.gather(1, blk)                                               # stored entries of flagged blocks are replaced by the full rescore
            est = torch.where(inflag, torch.full_like(est, -1e30), est)
            allp = torch.cat([pos, fpos], 1); alle = torch.cat([est, fest], 1); t2 = alle.topk(K, 1)
            pos, est = allp.gather(1, t2.indices), t2.values
    return pos.contiguous(), est

@triton.jit
def _cell_max(q_ptr, qrows_ptr, t_qoff_ptr, t_qn_ptr, t_blk0_ptr, t_nb_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, v1_ptr, N, NB,
              W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    pid_b = tl.program_id(0); task = tl.program_id(1)
    nb = tl.load(t_nb_ptr + task); valid = pid_b < nb
    blk = tl.load(t_blk0_ptr + task) + pid_b; qoff = tl.load(t_qoff_ptr + task); qn = tl.load(t_qn_ptr + task)
    rqi = tl.arange(0, BQ); qmask = (rqi < qn) & valid; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn < N) & valid; j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    tl.store(v1_ptr + rq * NB + blk, tl.max(est, axis=1), mask=qmask)

def search_wl(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, BN=32, BQ=32):
    """work-list scan (one launch) + the exact block-max selection of the paper (rescore K blocks)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous(); bm = torch.full((B, NB), -1e30, device=dev)
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices; qrows, qoff, qn, blk0, nb = worklist(probe, off_t, BN, BQ)
    _cell_max[(int(nb.max()), len(qoff))](Qr, qrows, qoff, qn, blk0, nb, words, tab, scale, qc, bm, N, NB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN)
    blocks = bm.topk(K, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=dev)).reshape(B, -1).clamp(max=N - 1).contiguous(); M = pos.shape[1]
    est = torch.empty(B, M, device=dev)
    _trellis_rescore[(triton.cdiv(M, 32), B)](Qr, pos, words, tab, scale, qc, est, B, M, W=W, R=R, RP=RP, L=L, BM=32)
    t = est.topk(K, 1); return pos.gather(1, t.indices).contiguous(), t.values

@triton.jit
def _cell_max_flat(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, v1_ptr, N, NB,
                   W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p)
    rqi = tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
    rn = blk * BN + tl.arange(0, BN); nmask = rn < N; j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    tl.store(v1_ptr + rq * NB + blk, tl.max(est, axis=1), mask=qmask)

def search_flat(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, BN=32, BQ=32):
    """flat work list: one program per (cell, query tile, block); exact block-max selection."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous(); bm = torch.full((B, NB), -1e30, device=dev)
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices; qrows, qoff, qn, blk0, nb = worklist(probe, off_t, BN, BQ)
    tid = torch.repeat_interleave(torch.arange(len(nb), device=dev), nb.long()); first = torch.cumsum(nb.long(), 0) - nb.long()
    pblk = (blk0[tid] + (torch.arange(len(tid), device=dev) - first[tid])).to(torch.int32).contiguous()
    _cell_max_flat[(len(tid),)](Qr, qrows, qoff[tid].contiguous(), qn[tid].contiguous(), pblk, words, tab, scale, qc, bm, N, NB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN)
    blocks = bm.topk(K, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=dev)).reshape(B, -1).clamp(max=N - 1).contiguous(); M = pos.shape[1]
    est = torch.empty(B, M, device=dev)
    _trellis_rescore[(triton.cdiv(M, 32), B)](Qr, pos, words, tab, scale, qc, est, B, M, W=W, R=R, RP=RP, L=L, BM=32)
    t = est.topk(K, 1); return pos.gather(1, t.indices).contiguous(), t.values

@triton.jit
def _cell_emit(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, N, CAP,
               W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    """append every item with estimate >= t0[query] to the query's buffer (atomic counter)."""
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p)
    rqi = tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = s[None, :] * dot + qc[:, None]
    t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
    keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
    ones = tl.full((BQ, BN), 1, tl.int32)
    idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), ones, mask=keep)
    ok = keep & (idx < CAP)
    tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
    tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

def _flat(probe, off_t, BN, BQ, bounds=False):
    qrows, qoff, qn, blk0, nb = worklist(probe, off_t, BN, BQ)
    tid = torch.repeat_interleave(torch.arange(len(nb), device=dev), nb.long()); first = torch.cumsum(nb.long(), 0) - nb.long()
    pblk = (blk0[tid] + (torch.arange(len(tid), device=dev) - first[tid])).to(torch.int32).contiguous()
    if not bounds: return qrows, qoff[tid].contiguous(), qn[tid].contiguous(), pblk
    # cell bounds per program (to skip rows of a neighbouring cell in straddling blocks)
    B, npr = probe.shape; flat = probe.flatten().sort().values; cells, counts = torch.unique_consecutive(flat, return_counts=True)
    ntile = (counts + BQ - 1) // BQ; tc = torch.repeat_interleave(torch.arange(len(cells), device=dev), ntile)
    cs, ce = off_t[cells[tc]][tid].to(torch.int32).contiguous(), off_t[cells[tc] + 1][tid].to(torch.int32).contiguous()
    return qrows, qoff[tid].contiguous(), qn[tid].contiguous(), pblk, cs, ce

@triton.jit
def _cell_max_flat_b(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, v1_ptr, N, NB,
                     W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    """block maxima over the rows of the probed cell only (atomic max: a straddling block is shared by two cells)."""
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p)
    rqi = tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
    q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    dot = tl.dot(q, tl.trans(v))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * dot + qc[:, None], -1e30)
    tl.atomic_max(v1_ptr + rq * NB + blk, tl.max(est, axis=1), mask=qmask)

def search_thresh(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, frac1=0.25, CAP=8192, BN=32, BQ=32, stats=None, mult=0.0):
    """exact top-K of the stage-1 estimates over the probed cells without rescoring: threshold from the nearest cells, then emit.
    The first pass covers max(frac1 * nprobe, enough cells for mult * K blocks) and masks rows outside each cell, so t0 is a valid bound."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices; n1 = max(1, int(round(nprobe * frac1)))
    if mult > 0:
        bpc = N / (len(off_t) - 1) / BN + 1                                              # blocks per cell, straddling included
        n1 = max(n1, int(np.ceil(mult * K / bpc)))
    n1 = min(n1, nprobe)
    bm = torch.full((B, NB), -1e30, device=dev); a = _flat(probe[:, :n1].contiguous(), off_t, BN, BQ, bounds=True)
    _cell_max_flat_b[(len(a[3]),)](Qr, *a, words, tab, scale, qc, bm, N, NB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN)
    t0 = bm.topk(K, 1).values[:, -1].contiguous(); del bm
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    a = _flat(probe, off_t, BN, BQ, bounds=True)
    _cell_emit[(len(a[3]),)](Qr, a[0], a[1], a[2], a[3], a[4], a[5], words, tab, scale, qc, t0, cnt, bval, bpos, N, CAP, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN)
    if stats is not None: stats.append(cnt.float().cpu())
    t = bval.topk(K, 1); pos, est = bpos.gather(1, t.indices).long(), t.values
    over = torch.nonzero(cnt > CAP).flatten()
    if len(over):                                                                       # buffer overflow: exact fallback for those queries
        p2, e2 = search_flat(Qr[over], Qorig[over], qc[over], words, tab, scale, L, K, cen, off_t, nprobe, BN, BQ)
        pos[over], est[over] = p2, e2
    return pos.contiguous(), est

# ---- one program per (cell, block); the block is decoded once and reused for every query tile that probes the cell
@triton.jit
def _emit_loop(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr,
               bm_ptr, N, CAP, NB, MAXMODE: tl.constexpr,
               W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    for t in range(0, tl.cdiv(qn, BQ)):
        rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
        q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
        dot = tl.dot(q, tl.trans(v))
        qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
        est = s[None, :] * dot + qc[:, None]
        if MAXMODE:
            est = tl.where(nmask[None, :], est, -1e30)
            tl.atomic_max(bm_ptr + rq * NB + blk, tl.max(est, axis=1), mask=qmask)
        else:
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

@triton.jit
def _first_compact(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, bmc_ptr, MB,
                   W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    """first pass with compact block maxima: one row of MB slots per (cell, query) pair of the sorted work list."""
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p); cs = tl.load(p_s_ptr + p)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    for t in range(0, tl.cdiv(qn, BQ)):
        rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
        q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
        dot = tl.dot(q, tl.trans(v))
        est = tl.where(nmask[None, :], s[None, :] * dot + tl.load(qc_ptr + rq, mask=qmask, other=0.0)[:, None], -1e30)
        tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)

def _cellblocks(probe, off_t, BN, ret_order=False):
    """per (probed cell, block): query-list offset and count for the whole cell, block id, cell bounds."""
    B, npr = probe.shape; flat = probe.flatten(); qid = torch.arange(B, device=dev).repeat_interleave(npr)
    o = torch.argsort(flat); flat, qid = flat[o], qid[o].to(torch.int32).contiguous()
    cells, counts = torch.unique_consecutive(flat, return_counts=True); starts = torch.cumsum(counts, 0) - counts
    s, e = off_t[cells], off_t[cells + 1]; nb = torch.where(e > s, (e - 1) // BN - s // BN + 1, 0)
    tid = torch.repeat_interleave(torch.arange(len(cells), device=dev), nb); first = torch.cumsum(nb, 0) - nb
    blk = (s[tid] // BN + torch.arange(len(tid), device=dev) - first[tid]).to(torch.int32)
    i32 = lambda x: x.to(torch.int32).contiguous()
    out = (qid, i32(starts[tid]), i32(counts[tid]), blk.contiguous(), i32(s[tid]), i32(e[tid]))
    return out + (o,) if ret_order else out

def search_thresh2(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, frac1=0.125, mult=1.5, CAP=16384, BN=32, BQ=32, stats=None, probe=None):
    """search_thresh with the per-(cell, block) loop kernels (one decode per block)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    if probe is None: probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    n1 = max(1, int(round(nprobe * frac1)), int(np.ceil(mult * K / (N / (len(off_t) - 1) / BN + 1)))); n1 = min(n1, nprobe)
    bm = torch.full((B, NB), -1e30, device=dev); a = _cellblocks(probe[:, :n1].contiguous(), off_t, BN)
    dummy = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    _emit_loop[(len(a[3]),)](Qr, *a, words, tab, scale, qc, dummy, di, dummy, di, bm, N, CAP, NB, MAXMODE=True, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    t0 = bm.topk(K, 1).values[:, -1].contiguous(); del bm
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    a = _cellblocks(probe, off_t, BN)
    _emit_loop[(len(a[3]),)](Qr, *a, words, tab, scale, qc, t0, cnt, bval, bpos, dummy, N, CAP, NB, MAXMODE=False, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    if stats is not None: stats.append(cnt.float().cpu())
    m = min(int(cnt.max()), CAP); t = bval[:, :max(m, K)].topk(K, 1); pos, est = bpos.gather(1, t.indices).long(), t.values
    over = torch.nonzero(cnt > CAP).flatten()
    if len(over):
        p2, e2 = search_flat(Qr[over], Qorig[over], qc[over], words, tab, scale, L, K, cen, off_t, nprobe, BN, 32)
        pos[over], est[over] = p2, e2
    return pos.contiguous(), est

_MB = {}
def search_guess(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, beta=0.7, rho=0.5, CAP=32768, BN=32, BQ=32, stats=None, n1=None):
    """guess-and-verify: t0 = the ceil(rho*K)-th largest block maximum over the n1 nearest cells (or their smallest block maximum if they have
    fewer blocks); one emit pass over all probed cells. If a query emitted at least K items (and at most CAP), at least K probed items are
    >= t0, so its top K are all in the buffer and the result is exact; other queries are recomputed with search_thresh2."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qh = Qr.half().contiguous(); qc = qc.float().contiguous()
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    if n1 is None: n1 = int(np.ceil(beta * K / (N / (len(off_t) - 1) / BN + 1)))
    n1 = min(max(n1, 1), nprobe)
    MB = _MB.get(off_t.data_ptr())
    if MB is None:
        o_ = (off_t[1:] - off_t[:-1]); MB = _MB[off_t.data_ptr()] = int((int(o_.max()) + BN - 1) // BN + 1)
    a = _cellblocks(probe[:, :n1].contiguous(), off_t, BN, ret_order=True); order = a[6]; a = a[:6]
    bmc = torch.full((B * n1, MB), -1e30, device=dev)
    _first_compact[(len(a[3]),)](Qh, *a, words, tab, scale, qc, bmc, MB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=dev)
    v = bmc[inv].reshape(B, n1 * MB)
    d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    r = int(np.ceil(rho * K)); top = v.topk(min(r, v.shape[1]), 1).values
    valid = top > -1e29; t0 = torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.empty((B, CAP), device=dev); bpos = torch.empty((B, CAP), dtype=torch.int32, device=dev)
    a = _cellblocks(probe, off_t, BN)
    _emit_loop[(len(a[3]),)](Qh, *a, words, tab, scale, qc, t0, cnt, bval, bpos, d, N, CAP, NB, MAXMODE=False, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    if stats is not None: stats.append(cnt.float().cpu())
    m = max(min(int(cnt.max()), CAP), K); ar = torch.arange(m, device=dev)
    t = torch.where(ar[None, :] < cnt[:, None], bval[:, :m], torch.full((), -1e30, device=dev)).topk(K, 1)
    pos, est = bpos[:, :m].gather(1, t.indices).long().clamp(min=0), t.values
    bad = torch.nonzero((cnt < K) | (cnt > CAP)).flatten()
    if len(bad):
        p2, e2 = search_thresh2(Qr[bad], Qorig[bad], qc[bad], words, tab, scale, L, K, cen, off_t, nprobe, CAP=CAP, probe=probe[bad])
        pos[bad], est[bad] = p2, e2
    if stats is not None: stats.append(len(bad))
    return pos.contiguous(), est

# ---- small batches: one program per (query, probed cell), no work list, no host synchronisation
@triton.jit
def _small_scan(q_ptr, probe_ptr, off_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, bm_ptr,
                NPR, NP_USE, CAP, NB, MAXMODE: tl.constexpr, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BN: tl.constexpr):
    q = tl.program_id(0); jj = tl.program_id(1)
    if jj < NP_USE:
        c = tl.load(probe_ptr + q * NPR + jj); s = tl.load(off_ptr + c).to(tl.int32); e = tl.load(off_ptr + c + 1).to(tl.int32)
        j = tl.arange(0, RP)
        qv = tl.load(q_ptr + q * R + j, mask=j < R, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + q)
        t0 = tl.load(t0_ptr + q)
        for b in range(s // BN, (e + BN - 1) // BN):
            rn = b * BN + tl.arange(0, BN); nmask = (rn >= s) & (rn < e)
            v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float32)
            sc = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
            est = tl.sum(v * qv[None, :], axis=1) * sc + qc
            if MAXMODE:
                tl.atomic_max(bm_ptr + q * NB + b, tl.max(tl.where(nmask, est, -1e30), axis=0))
            else:
                keep = nmask & (est >= t0)
                idx = tl.atomic_add(cnt_ptr + q + tl.zeros((BN,), dtype=tl.int32), tl.full((BN,), 1, tl.int32), mask=keep)
                ok = keep & (idx < CAP)
                tl.store(bval_ptr + q * CAP + idx, est, mask=ok); tl.store(bpos_ptr + q * CAP + idx, rn, mask=ok)

def search_small(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, CAP=8192, BN=32, beta=1.1, rho=0.75):
    """guess-and-verify selection without host synchronisation. Returns pos, est, ok (bool per query: exact)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R)
    Qr = Qr.float().contiguous(); qc = qc.float().contiguous()
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices.to(torch.int32).contiguous()
    n1 = min(nprobe, max(1, int(np.ceil(beta * K / (N / (len(off_t) - 1) / BN + 1))))); r = int(np.ceil(rho * K))
    bm = torch.full((B, NB), -1e30, device=dev); d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    big = torch.full((B,), 1e30, device=dev)
    _small_scan[(B, n1)](Qr, probe, off_t, words, tab, scale, qc, big, di, d, di, bm, nprobe, n1, CAP, NB, MAXMODE=True, W=W, R=R, RP=RP, L=L, BN=BN)
    t0 = bm.topk(r, 1).values[:, -1].contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    _small_scan[(B, nprobe)](Qr, probe, off_t, words, tab, scale, qc, t0, cnt, bval, bpos, d, nprobe, nprobe, CAP, NB, MAXMODE=False, W=W, R=R, RP=RP, L=L, BN=BN)
    t = bval.topk(K, 1)
    return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= CAP)

# ---- medium batches: dense (query x cell) probe mask, one program per (cell, block), no host synchronisation
@triton.jit
def _mask_scan(q_ptr, pm_ptr, off_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, bm_ptr, B, NC, CAP, NB,
               LEVEL, MAXMODE: tl.constexpr, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BN: tl.constexpr, BQ: tl.constexpr):
    c = tl.program_id(0); jb = tl.program_id(1)
    s = tl.load(off_ptr + c).to(tl.int32); e = tl.load(off_ptr + c + 1).to(tl.int32)
    b = s // BN + jb
    if b * BN < e:
        rq = tl.arange(0, BQ); qmask = rq < B
        lvl = tl.load(pm_ptr + rq * NC + c, mask=qmask, other=0).to(tl.int32)        # 0: not probed, 1: probed, 2: probed and in the first pass
        use = qmask & (lvl >= LEVEL)
        if tl.max(use.to(tl.int32), axis=0) > 0:
            rn = b * BN + tl.arange(0, BN); nmask = (rn >= s) & (rn < e); j = tl.arange(0, RP)
            v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=use[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            dot = tl.dot(q, tl.trans(v))
            sc = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
            est = sc[None, :] * dot + qc[:, None]
            m2 = use[:, None] & nmask[None, :]
            if MAXMODE:
                tl.atomic_max(bm_ptr + rq * NB + b, tl.max(tl.where(m2, est, -1e30), axis=1), mask=use)
            else:
                t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
                keep = m2 & (est >= t0[:, None])
                idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
                ok = keep & (idx < CAP)
                tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
                tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :] + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

def search_mask(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, MB, CAP=8192, BN=32, beta=1.1, rho=0.75):
    """guess-and-verify selection for batches up to 64 without host synchronisation (MB: max blocks per cell)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R); NC = len(off_t) - 1
    BQ = max(16, triton.next_power_of_2(B))
    Qh = Qr.half().contiguous(); qc = qc.float().contiguous()
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    n1 = min(nprobe, max(1, int(np.ceil(beta * K / (N / NC / BN + 1))))); r = int(np.ceil(rho * K))
    lvl = torch.ones(B, nprobe, dtype=torch.int8, device=dev); lvl[:, :n1] = 2
    pm = torch.zeros(B, NC, dtype=torch.int8, device=dev).scatter_(1, probe, lvl)
    bm = torch.full((B, NB), -1e30, device=dev); d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    _mask_scan[(NC, MB)](Qh, pm, off_t, words, tab, scale, qc, d, di, d, di, bm, B, NC, CAP, NB, 2, MAXMODE=True, W=W, R=R, RP=RP, L=L, BN=BN, BQ=BQ)
    t0 = bm.topk(r, 1).values[:, -1].contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    _mask_scan[(NC, MB)](Qh, pm, off_t, words, tab, scale, qc, t0, cnt, bval, bpos, d, B, NC, CAP, NB, 1, MAXMODE=False, W=W, R=R, RP=RP, L=L, BN=BN, BQ=BQ)
    t = bval.topk(K, 1)
    return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= CAP)

@triton.jit
def _mask_scan_t(q_ptr, pm_ptr, off_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, bm_ptr, B, NC, CAP, NB,
                 LEVEL, MAXMODE: tl.constexpr, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BN: tl.constexpr, BQ: tl.constexpr):
    """as _mask_scan, for any batch: loops over query tiles and skips tiles in which no query probes the cell. pm is (NC, B) (cell-major)."""
    c = tl.program_id(0); jb = tl.program_id(1)
    s = tl.load(off_ptr + c).to(tl.int32); e = tl.load(off_ptr + c + 1).to(tl.int32)
    b = s // BN + jb
    if b * BN < e:
        rn = b * BN + tl.arange(0, BN); nmask = (rn >= s) & (rn < e); j = tl.arange(0, RP)
        v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        sc = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(B, BQ)):
            rq = t * BQ + tl.arange(0, BQ); qmask = rq < B
            lvl = tl.load(pm_ptr + c * B + rq, mask=qmask, other=0).to(tl.int32)
            use = qmask & (lvl >= LEVEL)
            if tl.max(use.to(tl.int32), axis=0) > 0:
                q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=use[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
                dot = tl.dot(q, tl.trans(v))
                qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
                est = sc[None, :] * dot + qc[:, None]
                m2 = use[:, None] & nmask[None, :]
                if MAXMODE:
                    tl.atomic_max(bm_ptr + rq * NB + b, tl.max(tl.where(m2, est, -1e30), axis=1), mask=use)
                else:
                    t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
                    keep = m2 & (est >= t0[:, None])
                    idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
                    ok = keep & (idx < CAP)
                    tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
                    tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :] + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

def search_mask_t(Qr, Qorig, qc, words, tab, scale, L, K, cen, off_t, nprobe, MB, CAP=8192, BN=32, BQ=32, beta=None, rho=None):
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R); NC = len(off_t) - 1
    if beta is None: beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
    Qh = Qr.half().contiguous(); qc = qc.float().contiguous()
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    n1 = min(nprobe, max(1, int(np.ceil(beta * K / (N / NC / BN + 1))))); r = int(np.ceil(rho * K))
    lvl = torch.ones(B, nprobe, dtype=torch.int8, device=dev); lvl[:, :n1] = 2
    pm = torch.zeros(B, NC, dtype=torch.int8, device=dev).scatter_(1, probe, lvl).T.contiguous()
    bm = torch.full((B, NB), -1e30, device=dev); d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    _mask_scan_t[(NC, MB)](Qh, pm, off_t, words, tab, scale, qc, d, di, d, di, bm, B, NC, CAP, NB, 2, MAXMODE=True, W=W, R=R, RP=RP, L=L, BN=BN, BQ=BQ, num_stages=1)
    t0 = bm.topk(r, 1).values[:, -1].contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    _mask_scan_t[(NC, MB)](Qh, pm, off_t, words, tab, scale, qc, t0, cnt, bval, bpos, d, B, NC, CAP, NB, 1, MAXMODE=False, W=W, R=R, RP=RP, L=L, BN=BN, BQ=BQ, num_stages=1)
    t = bval.topk(K, 1)
    return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= CAP)
