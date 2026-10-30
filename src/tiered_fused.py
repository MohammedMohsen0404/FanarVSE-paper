"""Fused stage 2 for the tiered code: one kernel computes est = <q,c> + sum_t scale_t <q_t, cb_t[idx_t]> and var = sum_t ||q_t||^2 sig_t^2
for all candidates (up to 6 tiers; each tier's width and bits are compile-time constants). Queries are rotated per tier by one
block-diagonal matmul."""
import torch, triton, triton.language as tl
import tiered
dev = "cuda"

@triton.jit
def _tier(acc, var, q, rows, mmask, B, N, t, qk_ptr, planes_ptr, cb_ptr, scale_ptr, sig_ptr, qn2_ptr,
          NBITS: tl.constexpr, KB: tl.constexpr, KP: tl.constexpr, BM: tl.constexpr):
    cj = tl.arange(0, KP); cmask = cj < KB; m2 = mmask[:, None] & cmask[None, :]
    a = tl.zeros((BM,), dtype=tl.float32)
    for k in tl.static_range(8):
        code = tl.zeros((BM, KP), dtype=tl.int32)
        for p in tl.static_range(NBITS):
            byt = tl.load(planes_ptr + p * N * KB + rows[:, None] * KB + cj[None, :], mask=m2, other=0)
            code += (((byt >> (7 - k)) & 1).to(tl.int32)) << (NBITS - 1 - p)
        val = tl.load(cb_ptr + code)
        qrow = tl.load(qk_ptr + k * B * KB + q * KB + cj, mask=cmask, other=0.0)
        a += tl.sum(tl.where(m2, val * qrow[None, :], 0.0), axis=1)
    s = tl.load(scale_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    g = tl.load(sig_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    return acc + a * s, var + tl.load(qn2_ptr + t * B + q) * g * g

@triton.jit
def _tiered_score(pos_ptr, qc_ptr, est_ptr, sd_ptr, scale_ptr, sig_ptr, qn2_ptr, B, M, N,
                  q0, p0, c0, q1, p1, c1, q2, p2, c2, q3, p3, c3, q4, p4, c4, q5, p5, c5,
                  NB0: tl.constexpr, KB0: tl.constexpr, KP0: tl.constexpr, NB1: tl.constexpr, KB1: tl.constexpr, KP1: tl.constexpr,
                  NB2: tl.constexpr, KB2: tl.constexpr, KP2: tl.constexpr, NB3: tl.constexpr, KB3: tl.constexpr, KP3: tl.constexpr,
                  NB4: tl.constexpr, KB4: tl.constexpr, KP4: tl.constexpr, NB5: tl.constexpr, KB5: tl.constexpr, KP5: tl.constexpr,
                  BM: tl.constexpr):
    q = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    rows = tl.load(pos_ptr + q * M + rm, mask=mmask, other=0)
    acc = tl.zeros((BM,), dtype=tl.float32) + tl.load(qc_ptr + q); var = tl.zeros((BM,), dtype=tl.float32)
    if NB0 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 0, q0, p0, c0, scale_ptr, sig_ptr, qn2_ptr, NB0, KB0, KP0, BM)
    if NB1 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 1, q1, p1, c1, scale_ptr, sig_ptr, qn2_ptr, NB1, KB1, KP1, BM)
    if NB2 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 2, q2, p2, c2, scale_ptr, sig_ptr, qn2_ptr, NB2, KB2, KP2, BM)
    if NB3 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 3, q3, p3, c3, scale_ptr, sig_ptr, qn2_ptr, NB3, KB3, KP3, BM)
    if NB4 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 4, q4, p4, c4, scale_ptr, sig_ptr, qn2_ptr, NB4, KB4, KP4, BM)
    if NB5 > 0: acc, var = _tier(acc, var, q, rows, mmask, B, N, 5, q5, p5, c5, scale_ptr, sig_ptr, qn2_ptr, NB5, KB5, KP5, BM)
    tl.store(est_ptr + q * M + rm, acc, mask=mmask); tl.store(sd_ptr + q * M + rm, tl.sqrt(var), mask=mmask)

class _HP:
    def __init__(self, t): self.t, self.dtype = t, t.dtype
    def data_ptr(self): return self.t.data_ptr()

def prepare(code):
    """block-diagonal rotation V @ diag(P_t) so that one matmul gives all rotated tier queries."""
    if "VP" in code: return
    d = code["V"].shape[0]; D = torch.zeros(d, d, device=dev)
    for (s, e, _), P in zip(code["TS"], code["rots"]): D[s:e, s:e] = P
    code["VP"] = (code["V"] @ D).contiguous(); code["cbf"] = [tiered.CB[bb][0].float().contiguous() for _, _, bb in code["TS"]]

def score(code, Qb, qc, pos, BM=32):
    prepare(code); TS = code["TS"]; assert len(TS) <= 6
    B = len(Qb); M = pos.shape[1]; N = code["planes"][0].shape[1]
    Zq = (Qb.to(code["VP"].dtype) @ code["VP"]).float()
    qk, args, cex = [], [], {}
    qn2 = torch.stack([(Zq[:, s:e] ** 2).sum(1) for s, e, _ in TS]).contiguous()
    for t in range(6):
        if t < len(TS):
            s, e, bb = TS[t]; KB = (e - s) // 8
            qt = Zq[:, s:e].reshape(B, KB, 8).permute(2, 0, 1).contiguous(); qk.append(qt)
            pl = code["planes"][t]; pl = pl if pl.is_cuda else _HP(pl)
            args += [qt, pl, code["cbf"][t]]; cex.update({f"NB{t}": bb, f"KB{t}": KB, f"KP{t}": max(16, triton.next_power_of_2(KB))})
        else:
            args += [qn2, qn2, qn2]; cex.update({f"NB{t}": 0, f"KB{t}": 1, f"KP{t}": 16})
    est = torch.empty(B, M, device=dev); sd = torch.empty(B, M, device=dev)
    sc, sg = code["scale"], code["sig"]
    if not sc.is_cuda: sc, sg = _HP(sc), _HP(sg)
    _tiered_score[(triton.cdiv(M, BM), B)](pos.contiguous(), qc.float().contiguous(), est, sd, sc, sg, qn2, B, M, N, *args, BM=BM, **cex)
    return est, sd

# ---- packed layout: one contiguous record per vector (all tiers' bit planes, then fp16 scale and sigma per tier), for zero-copy reads
@triton.jit
def _tier_rec(acc, var, q, rows, mmask, B, RB, toff, soff, t, qk_ptr, rec_ptr, cb_ptr, qn2_ptr,
              NBITS: tl.constexpr, KB: tl.constexpr, KP: tl.constexpr, BM: tl.constexpr):
    cj = tl.arange(0, KP); cmask = cj < KB; m2 = mmask[:, None] & cmask[None, :]
    base = rows.to(tl.int64) * RB
    a = tl.zeros((BM,), dtype=tl.float32)

    for k in tl.static_range(8):
        code = tl.zeros((BM, KP), dtype=tl.int32)
        for p in tl.static_range(NBITS):
            byt = tl.load(rec_ptr + base[:, None] + toff + p * KB + cj[None, :], mask=m2, other=0)
            code += (((byt >> (7 - k)) & 1).to(tl.int32)) << (NBITS - 1 - p)
        val = tl.load(cb_ptr + code)
        qrow = tl.load(qk_ptr + k * B * KB + q * KB + cj, mask=cmask, other=0.0)
        a += tl.sum(tl.where(m2, val * qrow[None, :], 0.0), axis=1)
    sp = (rec_ptr + base + soff + 4 * t).to(tl.pointer_type(tl.float16), bitcast=True)
    s = tl.load(sp, mask=mmask, other=0.0).to(tl.float32); g = tl.load(sp + 1, mask=mmask, other=0.0).to(tl.float32)
    return acc + a * s, var + tl.load(qn2_ptr + t * B + q) * g * g

@triton.jit
def _tiered_score_rec(pos_ptr, qc_ptr, est_ptr, sd_ptr, rec_ptr, qn2_ptr, B, M, RB, soff,
                      q0, c0, o0, q1, c1, o1, q2, c2, o2, q3, c3, o3, q4, c4, o4, q5, c5, o5,
                      NB0: tl.constexpr, KB0: tl.constexpr, KP0: tl.constexpr, NB1: tl.constexpr, KB1: tl.constexpr, KP1: tl.constexpr,
                      NB2: tl.constexpr, KB2: tl.constexpr, KP2: tl.constexpr, NB3: tl.constexpr, KB3: tl.constexpr, KP3: tl.constexpr,
                      NB4: tl.constexpr, KB4: tl.constexpr, KP4: tl.constexpr, NB5: tl.constexpr, KB5: tl.constexpr, KP5: tl.constexpr,
                      BM: tl.constexpr):
    q = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    rows = tl.load(pos_ptr + q * M + rm, mask=mmask, other=0)
    acc = tl.zeros((BM,), dtype=tl.float32) + tl.load(qc_ptr + q); var = tl.zeros((BM,), dtype=tl.float32)
    if NB0 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o0, soff, 0, q0, rec_ptr, c0, qn2_ptr, NB0, KB0, KP0, BM)
    if NB1 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o1, soff, 1, q1, rec_ptr, c1, qn2_ptr, NB1, KB1, KP1, BM)
    if NB2 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o2, soff, 2, q2, rec_ptr, c2, qn2_ptr, NB2, KB2, KP2, BM)
    if NB3 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o3, soff, 3, q3, rec_ptr, c3, qn2_ptr, NB3, KB3, KP3, BM)
    if NB4 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o4, soff, 4, q4, rec_ptr, c4, qn2_ptr, NB4, KB4, KP4, BM)
    if NB5 > 0: acc, var = _tier_rec(acc, var, q, rows, mmask, B, RB, o5, soff, 5, q5, rec_ptr, c5, qn2_ptr, NB5, KB5, KP5, BM)
    tl.store(est_ptr + q * M + rm, acc, mask=mmask); tl.store(sd_ptr + q * M + rm, tl.sqrt(var), mask=mmask)

def pack(code, pin=True):
    """records (N, RB) uint8: tier t plane p at toff[t] + p*KB_t; then (scale, sigma) fp16 pairs per tier. RB rounded up to 64."""
    TS = code["TS"]; N = code["planes"][0].shape[1]; offs, o = [], 0
    for (s, e, bb) in TS: offs.append(o); o += bb * (e - s) // 8
    soff = o; RB = (o + 4 * len(TS) + 63) // 64 * 64
    rec = torch.zeros(N, RB, dtype=torch.uint8)
    for t, (s, e, bb) in enumerate(TS):
        KB = (e - s) // 8; P = code["planes"][t].cpu()
        for p in range(bb): rec[:, offs[t] + p * KB: offs[t] + (p + 1) * KB] = P[p]
    ss = torch.stack([code["scale"].cpu(), code["sig"].cpu()], 2).reshape(len(TS), N, 2).permute(1, 0, 2).reshape(N, -1).contiguous()
    rec[:, soff:soff + 4 * len(TS)] = ss.view(torch.uint8)
    if pin:
        r2 = torch.empty_like(rec, pin_memory=True); r2.copy_(rec); rec = r2
    return dict(rec=rec, RB=RB, soff=soff, toff=offs)

def score_rec(code, pk, Qb, qc, pos, BM=32):
    prepare(code); TS = code["TS"]; B = len(Qb); M = pos.shape[1]
    Zq = (Qb.to(code["VP"].dtype) @ code["VP"]).float()
    qn2 = torch.stack([(Zq[:, s:e] ** 2).sum(1) for s, e, _ in TS]).contiguous(); args, cex = [], {}
    for t in range(6):
        if t < len(TS):
            s, e, bb = TS[t]; KB = (e - s) // 8
            qt = Zq[:, s:e].reshape(B, KB, 8).permute(2, 0, 1).contiguous()
            args += [qt, code["cbf"][t], pk["toff"][t]]; cex.update({f"NB{t}": bb, f"KB{t}": KB, f"KP{t}": max(16, triton.next_power_of_2(KB))})
        else:
            args += [qn2, qn2, 0]; cex.update({f"NB{t}": 0, f"KB{t}": 1, f"KP{t}": 16})
    rec = pk["rec"]; rp = rec if rec.is_cuda else _HP(rec)
    est = torch.empty(B, M, device=dev); sd = torch.empty(B, M, device=dev)
    _tiered_score_rec[(triton.cdiv(M, BM), B)](pos.contiguous(), qc.float().contiguous(), est, sd, rp, qn2, B, M, pk["RB"], pk["soff"], *args, BM=BM, **cex)
    return est, sd
