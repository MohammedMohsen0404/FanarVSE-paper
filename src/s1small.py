"""Small-batch stage-1 kernels (copies of s1fast.search_small / search_mask) using the register-built trellis decode."""
import numpy as np, torch, triton, triton.language as tl
from decode2 import _decode_tile2 as _decode_tile
dev = "cuda"

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

