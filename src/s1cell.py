"""Stage 1 on cell residuals (static kernels decode with the register-built decoder of decode2): codes encode x - mean(cell) instead of x - c, and the kernels add q . mean(cell) per (query, cell) pair
instead of q . c per query. The pair term is gathered from QM = Q @ means^T in the order of the sorted work list (qoff + i is unique
per pair), so the kernels below differ from s1fast only in where they load the constant."""
import numpy as np, torch, triton, triton.language as tl
from s1fast import _decode_tile, _cellblocks
from decode2 import _decode_tile2
_MB = {}
def _cache(d, t, BN, make):
    """per-tensor cache: keyed by address but checked by identity, since a freed tensor's address can be reused by another index."""
    k = (t.data_ptr(), BN); hit = d.get(k)
    if hit is None or hit[0] is not t: hit = d[k] = (t, make())
    return hit[1]
dev = "cuda"

@triton.jit
def _emit_loop_c(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr,
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
        qc = tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)
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
def _first_compact_c(q_ptr, qrows_ptr, p_qoff_ptr, p_qn_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, bmc_ptr, MB,
                     W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0)
    blk = tl.load(p_blk_ptr + p); qoff = tl.load(p_qoff_ptr + p); qn = tl.load(p_qn_ptr + p); cs = tl.load(p_s_ptr + p)
    rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
    v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    for t in range(0, tl.cdiv(qn, BQ)):
        rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
        q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
        dot = tl.dot(q, tl.trans(v))
        est = tl.where(nmask[None, :], s[None, :] * dot + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None], -1e30)
        tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)

def _work(probe, off_t, BN, QM):
    """work list of s1fast._cellblocks plus the pair constants q . mean(cell) in work-list order."""
    a = _cellblocks(probe, off_t, BN, ret_order=True); o = a[6]
    qcp = QM[a[0].long(), probe.flatten()[o]].float().contiguous()
    return a[:6], qcp, o

def search_thresh2(Qr, QM, words, tab, scale, K, off_t, probe, frac1=0.125, mult=1.5, CAP=16384, BN=32, BQ=32, L=10):
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R); nprobe = probe.shape[1]
    Qr = Qr.float().contiguous()
    n1 = max(1, int(round(nprobe * frac1)), int(np.ceil(mult * K / (N / (len(off_t) - 1) / BN + 1)))); n1 = min(n1, nprobe)
    bm = torch.full((B, NB), -1e30, device=dev); a, qcp, _ = _work(probe[:, :n1].contiguous(), off_t, BN, QM)
    dummy = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
    _emit_loop_c[(len(a[3]),)](Qr, *a, words, tab, scale, qcp, dummy, di, dummy, di, bm, N, CAP, NB, MAXMODE=True, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    t0 = bm.topk(K, 1).values[:, -1].contiguous(); del bm
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, CAP), -1e30, device=dev); bpos = torch.zeros((B, CAP), dtype=torch.int32, device=dev)
    a, qcp, _ = _work(probe, off_t, BN, QM)
    _emit_loop_c[(len(a[3]),)](Qr, *a, words, tab, scale, qcp, t0, cnt, bval, bpos, dummy, N, CAP, NB, MAXMODE=False, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    m = min(int(cnt.max()), CAP); t = bval[:, :max(m, K)].topk(K, 1); pos, est = bpos.gather(1, t.indices).long(), t.values
    over = torch.nonzero(cnt > CAP).flatten()
    if len(over):                                                                                  # rare: redo those queries with a larger buffer
        p2, e2 = search_thresh2(Qr[over], QM[over], words, tab, scale, K, off_t, probe[over], frac1, mult, CAP * 4, BN, BQ, L)
        pos[over], est[over] = p2, e2
    return pos.contiguous(), est

def select(bval, bpos, cnt, K, buckets=(1024, 4096, 16384)):
    """top K of each query's first cnt[q] buffer entries (cnt <= CAP); queries are grouped by cnt so most selections run on short rows."""
    B, CAP = bval.shape; pos = torch.empty((B, K), dtype=torch.long, device=dev); est = torch.empty((B, K), device=dev)
    cmax = min(int(cnt.max()), CAP); lo = -1
    for S in sorted({max(b, K) for b in buckets if b < cmax} | {min(max(cmax, K), CAP)}):
        q = torch.nonzero((cnt > lo) & (cnt <= S)).flatten(); lo = S
        if len(q):
            ar = torch.arange(S, device=dev); v = torch.where(ar[None, :] < cnt[q, None], bval[q, :S], torch.full((), -1e30, device=dev)).topk(K, 1)
            pos[q] = bpos[q, :S].gather(1, v.indices).long().clamp(min=0); est[q] = v.values
    return pos, est

def search_guess(Qr, QM, words, tab, scale, K, off_t, probe, beta=0.7, rho=0.5, CAP=65536, BN=64, BQ=16, L=10, stats=None, bucketed=True):
    """s1fast.search_guess with pair constants (same exactness argument: the threshold test is on the same estimates that are emitted)."""
    B, R = Qr.shape; N, W = words.shape; NB = triton.cdiv(N, BN); RP = triton.next_power_of_2(R); nprobe = probe.shape[1]
    Qh = Qr.half().contiguous()
    n1 = min(max(int(np.ceil(beta * K / (N / (len(off_t) - 1) / BN + 1))), 1), nprobe)
    MB = _cache(_MB, off_t, BN, lambda: int((int((off_t[1:] - off_t[:-1]).max()) + BN - 1) // BN + 1))
    a, qcp, order = _work(probe[:, :n1].contiguous(), off_t, BN, QM)
    bmc = torch.full((B * n1, MB), -1e30, device=dev)
    _first_compact_c[(len(a[3]),)](Qh, *a, words, tab, scale, qcp, bmc, MB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=dev)
    v = bmc[inv].reshape(B, n1 * MB)
    d = torch.zeros(1, device=dev)
    r = int(np.ceil(rho * K)); top = v.topk(min(r, v.shape[1]), 1).values
    valid = top > -1e29; t0 = torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.empty((B, CAP), device=dev); bpos = torch.empty((B, CAP), dtype=torch.int32, device=dev)
    a, qcp, _ = _work(probe, off_t, BN, QM)
    _emit_loop_c[(len(a[3]),)](Qh, *a, words, tab, scale, qcp, t0, cnt, bval, bpos, d, N, CAP, NB, MAXMODE=False, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    if bucketed: pos, est = select(bval, bpos, cnt.clamp(max=CAP), K)
    else:
        m = max(min(int(cnt.max()), CAP), K); ar = torch.arange(m, device=dev)
        t = torch.where(ar[None, :] < cnt[:, None], bval[:, :m], torch.full((), -1e30, device=dev)).topk(K, 1)
        pos, est = bpos[:, :m].gather(1, t.indices).long().clamp(min=0), t.values
    bad = torch.nonzero((cnt < K) | (cnt > CAP)).flatten()
    if len(bad):
        p2, e2 = search_thresh2(Qr[bad], QM[bad], words, tab, scale, K, off_t, probe[bad], CAP=CAP, L=L)
        pos[bad], est[bad] = p2, e2
    if stats is not None: stats.append(len(bad))
    return pos.contiguous(), est

# ---- static (synchronisation-free) batch path: fixed-size work list over all (cell, block) pairs, so the whole search can be a CUDA graph
@triton.jit
def _emit_loop_s(q_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr,
                 CAP, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
        v = _decode_tile2(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            est = s[None, :] * tl.dot(q, tl.trans(v)) + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

@triton.jit
def _first_compact_s(q_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, bmc_ptr, MB,
                     W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p); cs = tl.load(p_s_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
        v = _decode_tile2(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            est = tl.where(nmask[None, :], s[None, :] * tl.dot(q, tl.trans(v)) + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None], -1e30)
            tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)

_STATIC = {}
def _static_blocks(off_t, BN):
    """all (cell, block) programs of the index: cell, block id, cell bounds (fixed for the index)."""
    return _cache(_STATIC, off_t, BN, lambda: _static_make(off_t, BN))

def _static_make(off_t, BN):
    if True:
        s, e = off_t[:-1], off_t[1:]; nb = torch.where(e > s, (e - 1) // BN - s // BN + 1, 0)
        cell = torch.repeat_interleave(torch.arange(len(s), device=dev), nb); first = torch.cumsum(nb, 0) - nb
        blk = s[cell] // BN + torch.arange(len(cell), device=dev) - first[cell]
        MB = int((int((e - s).max()) + BN - 1) // BN + 1)
        i32 = lambda x: x.to(torch.int32).contiguous()
        return (i32(cell), i32(blk), i32(s[cell]), i32(e[cell]), MB)

def _pairs_static(probe, NC, QM):
    """query list per cell (CSR over cells) with fixed-size ops: qid sorted by cell, cell offsets and counts, pair constants."""
    B, npr = probe.shape; flat = probe.flatten(); o = torch.argsort(flat, stable=True)
    qid = (o // npr).to(torch.int32).contiguous(); cnt = torch.zeros(NC, dtype=torch.long, device=dev).scatter_add_(0, flat, torch.ones_like(flat)); off = (torch.cumsum(cnt, 0) - cnt)
    return qid, off.to(torch.int32).contiguous(), cnt.to(torch.int32).contiguous(), QM[qid.long(), flat[o]].float().contiguous(), o

def search_static(Qr, QM, words, tab, scale, K, off_t, probe, beta=0.7, rho=0.5, M=None, BN=32, BQ=32, L=10):
    """guess-and-verify with no host synchronisation. Returns pos, est, ok; ok[q] is False when query q emitted fewer than K items or
    more than M (its result is then not guaranteed exact and must be recomputed)."""
    B, R = Qr.shape; N, W = words.shape; RP = triton.next_power_of_2(R); nprobe = probe.shape[1]; NC = len(off_t) - 1
    if M is None: M = max(4096, 16 * K)
    pcell, pblk, ps, pe, MB = _static_blocks(off_t, BN); P = len(pcell)
    Qh = Qr.half().contiguous()
    n1 = min(max(int(np.ceil(beta * K / (N / NC / BN + 1))), 1), nprobe)
    qid, coff, ccnt, qcp, o = _pairs_static(probe[:, :n1].contiguous(), NC, QM)
    bmc = torch.full((B * n1, MB), -1e30, device=dev)
    _first_compact_s[(P,)](Qh, qid, coff, ccnt, pcell, pblk, ps, pe, words, tab, scale, qcp, bmc, MB, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    inv = torch.empty_like(o); inv[o] = torch.arange(len(o), device=dev)
    v = bmc[inv].reshape(B, n1 * MB)
    r = int(np.ceil(rho * K)); top = v.topk(min(r, v.shape[1]), 1).values
    valid = top > -1e29; t0 = torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
    qid, coff, ccnt, qcp, _ = _pairs_static(probe, NC, QM)
    _emit_loop_s[(P,)](Qh, qid, coff, ccnt, pcell, pblk, ps, pe, words, tab, scale, qcp, t0, cnt, bval, bpos, M, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    t = bval.topk(K, 1); pos, est = bpos.gather(1, t.indices).long(), t.values
    return pos.contiguous(), est, (cnt >= K) & (cnt <= M)

# ---- persistent variant: a fixed number of programs loop over the active (cell, block) items, compacted on the GPU without synchronisation
@triton.jit
def _emit_loop_p(q_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, act_ptr, nact_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr,
                 CAP, NPROG, W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    n = tl.load(nact_ptr)
    for it in range(tl.program_id(0), n, NPROG):
        p = tl.load(act_ptr + it); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
        v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            est = s[None, :] * tl.dot(q, tl.trans(v)) + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)

@triton.jit
def _first_compact_p(q_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, act_ptr, nact_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, words_ptr, tab_ptr, scale_ptr, qc_ptr, bmc_ptr, MB, NPROG,
                     W: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, L: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    n = tl.load(nact_ptr)
    for it in range(tl.program_id(0), n, NPROG):
        p = tl.load(act_ptr + it); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p); cs = tl.load(p_s_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, RP)
        v = _decode_tile(words_ptr, rn, nmask, W, R, RP, L, tab_ptr).to(tl.float16)
        s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None] & (j[None, :] < R), other=0.0).to(tl.float16)
            est = tl.where(nmask[None, :], s[None, :] * tl.dot(q, tl.trans(v)) + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None], -1e30)
            tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)

def _active(pcell, ccnt):
    """indices of the (cell, block) programs whose cell is probed, compacted with fixed-size operations; count stays on the GPU."""
    P = len(pcell); m = ccnt[pcell.long()] > 0; pos = torch.cumsum(m.to(torch.int32), 0) - 1
    act = torch.zeros(P + 1, dtype=torch.int32, device=dev)
    act.scatter_(0, torch.where(m, pos.long(), torch.full_like(pos, P, dtype=torch.long)), torch.arange(P, dtype=torch.int32, device=dev))
    return act, m.sum().to(torch.int32).reshape(1)

def search_persist(Qr, QM, words, tab, scale, K, off_t, probe, beta=0.7, rho=0.5, M=None, BN=32, BQ=32, L=10, NPROG=1024):
    """search_static with persistent programs over the active items (same results, same exactness flag)."""
    B, R = Qr.shape; N, W = words.shape; RP = triton.next_power_of_2(R); nprobe = probe.shape[1]; NC = len(off_t) - 1
    if M is None: M = max(4096, 16 * K)
    pcell, pblk, ps, pe, MB = _static_blocks(off_t, BN)
    Qh = Qr.half().contiguous()
    n1 = min(max(int(np.ceil(beta * K / (N / NC / BN + 1))), 1), nprobe)
    qid, coff, ccnt, qcp, o = _pairs_static(probe[:, :n1].contiguous(), NC, QM); act, nact = _active(pcell, ccnt)
    bmc = torch.full((B * n1, MB), -1e30, device=dev)
    _first_compact_p[(NPROG,)](Qh, qid, coff, ccnt, act, nact, pcell, pblk, ps, pe, words, tab, scale, qcp, bmc, MB, NPROG, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    inv = torch.empty_like(o); inv[o] = torch.arange(len(o), device=dev)
    v = bmc[inv].reshape(B, n1 * MB)
    r = int(np.ceil(rho * K)); top = v.topk(min(r, v.shape[1]), 1).values
    valid = top > -1e29; t0 = torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
    cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
    qid, coff, ccnt, qcp, _ = _pairs_static(probe, NC, QM); act, nact = _active(pcell, ccnt)
    _emit_loop_p[(NPROG,)](Qh, qid, coff, ccnt, act, nact, pcell, pblk, ps, pe, words, tab, scale, qcp, t0, cnt, bval, bpos, M, NPROG, W=W, R=R, RP=RP, L=L, BQ=BQ, BN=BN, num_stages=1)
    t = bval.topk(K, 1); pos, est = bpos.gather(1, t.indices).long(), t.values
    return pos.contiguous(), est, (cnt >= K) & (cnt <= M)
