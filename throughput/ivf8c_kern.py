import triton, triton.language as tl
@triton.jit
def _first8c(q_ptr, qs_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, z_ptr, zs_ptr, qc_ptr, bmc_ptr, MB,
            R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p); cs = tl.load(p_s_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, R)
        z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
        zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
            est = tl.dot(q, tl.trans(z)).to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :] + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            est = tl.where(nmask[None, :], est, -1e30)
            tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)

@triton.jit
def _emit8c(q_ptr, qs_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, z_ptr, zs_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, CAP,
           R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, R)
        z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
        zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
            est = tl.dot(q, tl.trans(z)).to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :] + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)



@triton.jit
def _emit8x(q_ptr, qs_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, z_ptr, zs_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, CAP,
           R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, EMIT: tl.constexpr, DOT: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, R)
        z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
        zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0)
        for t in range(0, tl.cdiv(qn, BQ)):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
            if DOT:
                acc = tl.dot(q, tl.trans(z))
            else:
                acc = tl.zeros((BQ, BN), dtype=tl.int32) + tl.sum(q.to(tl.int32), axis=1)[:, None] + tl.sum(z.to(tl.int32), axis=1)[None, :]
            est = acc.to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :] + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            if EMIT:
              idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
              ok = keep & (idx < CAP)
              tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
              tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)
            else:
              tl.store(cnt_ptr + rq, tl.sum(keep.to(tl.int32), axis=1), mask=qmask & (tl.sum(keep.to(tl.int32), axis=1) == 123456))



@triton.jit
def _first8p(q_ptr, qs_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, z_ptr, zs_ptr, qc_ptr, bmc_ptr, MB,
            R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p); cs = tl.load(p_s_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= cs) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, R)
        z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
        zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0)
        for t in tl.range(0, tl.cdiv(qn, BQ), num_stages=NS):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
            est = tl.dot(q, tl.trans(z)).to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :] + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            est = tl.where(nmask[None, :], est, -1e30)
            tl.store(bmc_ptr + (qoff + rqi) * MB + (blk - cs // BN), tl.max(est, axis=1), mask=qmask)



@triton.jit
def _emit8p(q_ptr, qs_ptr, qrows_ptr, c_qoff_ptr, c_qn_ptr, p_cell_ptr, p_blk_ptr, p_s_ptr, p_e_ptr, z_ptr, zs_ptr, qc_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, CAP,
           R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr):
    p = tl.program_id(0); c = tl.load(p_cell_ptr + p); qn = tl.load(c_qn_ptr + c)
    if qn > 0:
        qoff = tl.load(c_qoff_ptr + c); blk = tl.load(p_blk_ptr + p)
        rn = blk * BN + tl.arange(0, BN); nmask = (rn >= tl.load(p_s_ptr + p)) & (rn < tl.load(p_e_ptr + p)); j = tl.arange(0, R)
        z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
        zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0)
        for t in tl.range(0, tl.cdiv(qn, BQ), num_stages=NS):
            rqi = t * BQ + tl.arange(0, BQ); qmask = rqi < qn; rq = tl.load(qrows_ptr + qoff + rqi, mask=qmask, other=0)
            q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
            est = tl.dot(q, tl.trans(z)).to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :] + tl.load(qc_ptr + qoff + rqi, mask=qmask, other=0.0)[:, None]
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :].to(tl.int32) + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)



