"""Stage 2 with uniform (affine) per-tier codebooks, scored by nibble lookup tables.
With levels Delta*(k - (2^b-1)/2), <q_t, cb[idx]> = Delta*(sum_p 2^(b-1-p) <q_t, plane_p> - mid * sum(q_t)), and <q_t, plane_p> is a sum of
per-nibble table entries LUT[nibble][4-bit value] (16 entries per 4 coordinates, built once per query). Theorem 1 holds for any codebook,
so estimates stay unbiased and sigma keeps its form (with the uniform code's c)."""
import numpy as np, torch, triton, triton.language as tl
import tiered
dev = "cuda"

def uniform_cb(b, n=2_000_000):
    g = torch.randn(n, device=dev, generator=torch.Generator(device=dev).manual_seed(3)); best = None
    mid = (2 ** b - 1) / 2
    for delta in np.linspace(0.02, 2.5, 600):
        lv = (torch.arange(2 ** b, device=dev).float() - mid) * delta; th = (lv[1:] + lv[:-1]) / 2
        mse = ((g - lv[torch.bucketize(g, th)]) ** 2).mean().item()
        if best is None or mse < best[0]: best = (mse, float(delta), lv, th)
    return best

def encode_uniform(code, host16, order, c, seed=0):
    """re-encode the tiered refinement with uniform codebooks (same TS, PCA basis V and centre c)."""
    saved = dict(tiered.CB); deltas = {}
    for b in sorted({bb for _, _, bb in code["TS"]}):
        mse, delta, lv, th = uniform_cb(b); tiered.CB[b] = (lv, th); deltas[b] = delta
        print(f"  uniform {b} bit: delta {delta:.4f} mse {mse:.5f} (Lloyd-Max {tiered.DIST[b]:.5f})")
    V = code["V"].to(dev) if not code["V"].is_cuda else code["V"]
    new = tiered.build(host16, order, c, V, code["TS"], seed=seed)
    tiered.CB.clear(); tiered.CB.update(saved)
    out = dict(code); out.update(new); out["V"] = V; out.pop("VP", None); out.pop("cbf", None); out["deltas"] = deltas
    out["cb_uniform"] = {b: (torch.arange(2 ** b, device=dev).float() - (2 ** b - 1) / 2) * d for b, d in deltas.items()}
    return out

def prepare_u(code):
    if "VP" in code: return
    d = code["V"].shape[0]; D = torch.zeros(d, d, device=dev)
    for (s, e, _), P in zip(code["TS"], code["rots"]): D[s:e, s:e] = P
    code["VP"] = (code["V"] @ D).contiguous(); code["cbf"] = [code["cb_uniform"][bb].contiguous() for _, _, bb in code["TS"]]
    BITS = torch.tensor([[(v >> (3 - i)) & 1 for v in range(16)] for i in range(4)], dtype=torch.float32, device=dev)   # (4, 16): coord i of nibble
    code["NIB"] = BITS

def luts(code, Qb, nt=None):
    """per query: rotated tier queries -> nibble tables (B, total_nibbles*16), plus sum(q_t) and ||q_t||^2 per tier."""
    prepare_u(code); TS = code["TS"][:nt] if nt else code["TS"]; B = len(Qb)
    if "VP16" not in code: code["VP16"] = code["VP"].half().contiguous()
    Zq = (Qb.half() @ code["VP16"]).float() if code.get("fp16_query", False) else (Qb.to(code["VP"].dtype) @ code["VP"]).float()
    L = torch.cat([(Zq[:, s:e].reshape(B, (e - s) // 4, 4) @ code["NIB"]).reshape(B, -1) for s, e, _ in TS], 1).contiguous()
    qsum = torch.stack([Zq[:, s:e].sum(1) for s, e, _ in TS]).contiguous(); qn2 = torch.stack([(Zq[:, s:e] ** 2).sum(1) for s, e, _ in TS]).contiguous()
    return L, qsum, qn2

@triton.jit
def _tier_lut(acc, var, q, rows, mmask, N, t, lut_ptr, LW, loff, planes_ptr, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, delta,
              NBITS: tl.constexpr, KB: tl.constexpr, KP: tl.constexpr, BM: tl.constexpr):
    cj = tl.arange(0, KP); cmask = cj < KB; m2 = mmask[:, None] & cmask[None, :]
    lq = lut_ptr + q.to(tl.int64) * LW + loff
    a = tl.zeros((BM,), dtype=tl.float32)
    for p in tl.static_range(NBITS):
        byt = tl.load(planes_ptr + p * N * KB + rows[:, None] * KB + cj[None, :], mask=m2, other=0).to(tl.int32)
        hi = tl.load(lq + (2 * cj[None, :]) * 16 + (byt >> 4), mask=m2, other=0.0).to(tl.float32)
        lo = tl.load(lq + (2 * cj[None, :] + 1) * 16 + (byt & 15), mask=m2, other=0.0).to(tl.float32)
        a += tl.sum(hi + lo, axis=1) * (1 << (NBITS - 1 - p))
    mid = ((1 << NBITS) - 1) * 0.5
    a = delta * (a - mid * tl.load(qsum_ptr + t * B + q))
    s = tl.load(scale_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    g = tl.load(sig_ptr + t * N + rows, mask=mmask, other=0.0).to(tl.float32)
    return acc + a * s, var + tl.load(qn2_ptr + t * B + q) * g * g

@triton.jit
def _score_lut(pos_ptr, qc_ptr, est_ptr, sd_ptr, lut_ptr, LW, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, M, N, ncnt_ptr,
               p0, o0, d0, p1, o1, d1, p2, o2, d2, p3, o3, d3, p4, o4, d4, p5, o5, d5,
               NB0: tl.constexpr, KB0: tl.constexpr, KP0: tl.constexpr, NB1: tl.constexpr, KB1: tl.constexpr, KP1: tl.constexpr,
               NB2: tl.constexpr, KB2: tl.constexpr, KP2: tl.constexpr, NB3: tl.constexpr, KB3: tl.constexpr, KP3: tl.constexpr,
               NB4: tl.constexpr, KB4: tl.constexpr, KP4: tl.constexpr, NB5: tl.constexpr, KB5: tl.constexpr, KP5: tl.constexpr,
               BM: tl.constexpr, CNT: tl.constexpr):
    q = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    lim = M
    if CNT:
        lim = tl.load(ncnt_ptr + q)
    if tl.program_id(0) * BM >= lim:
        tl.store(est_ptr + q * M + rm, tl.full((BM,), -1e30, tl.float32), mask=mmask); tl.store(sd_ptr + q * M + rm, tl.zeros((BM,), dtype=tl.float32), mask=mmask)
        return
    live = mmask & (rm < lim)
    rows = tl.load(pos_ptr + q * M + rm, mask=live, other=0)
    acc = tl.zeros((BM,), dtype=tl.float32) + tl.load(qc_ptr + q); var = tl.zeros((BM,), dtype=tl.float32)
    mmask = live
    if NB0 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 0, lut_ptr, LW, o0, p0, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d0, NB0, KB0, KP0, BM)
    if NB1 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 1, lut_ptr, LW, o1, p1, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d1, NB1, KB1, KP1, BM)
    if NB2 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 2, lut_ptr, LW, o2, p2, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d2, NB2, KB2, KP2, BM)
    if NB3 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 3, lut_ptr, LW, o3, p3, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d3, NB3, KB3, KP3, BM)
    if NB4 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 4, lut_ptr, LW, o4, p4, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d4, NB4, KB4, KP4, BM)
    if NB5 > 0: acc, var = _tier_lut(acc, var, q, rows, mmask, N, 5, lut_ptr, LW, o5, p5, scale_ptr, sig_ptr, qsum_ptr, qn2_ptr, B, d5, NB5, KB5, KP5, BM)
    acc = tl.where(live, acc, -1e30); var = tl.where(live, var, 0.0)
    tl.store(est_ptr + q * M + rm, acc, mask=rm < M); tl.store(sd_ptr + q * M + rm, tl.sqrt(var), mask=rm < M)

def score_lut(code, Qb, qc, pos, nt=None, BM=32, nw=4, pre=None, ncnt=None):
    TS = code["TS"][:nt] if nt else code["TS"]; B = len(Qb); M = pos.shape[1]; N = code["planes"][0].shape[1]
    L, qsum, qn2 = pre if pre is not None else luts(code, Qb, nt)
    args, cex, off = [], {}, 0
    for t in range(6):
        if t < len(TS):
            s, e, bb = TS[t]; KB = (e - s) // 8
            args += [code["planes"][t], off, float(code["deltas"][bb])]; off += (e - s) // 4 * 16
            cex.update({f"NB{t}": bb, f"KB{t}": KB, f"KP{t}": triton.next_power_of_2(KB)})
        else:
            args += [qn2, 0, 0.0]; cex.update({f"NB{t}": 0, f"KB{t}": 1, f"KP{t}": 1})
    est = torch.empty(B, M, device=dev); sd = torch.empty(B, M, device=dev)
    _score_lut[(triton.cdiv(M, BM), B)](pos.contiguous(), qc.float().contiguous(), est, sd, L, L.shape[1], code["scale"], code["sig"], qsum, qn2, B, M, N,
                                         ncnt if ncnt is not None else qn2, *args, BM=BM, num_warps=nw, CNT=ncnt is not None, **cex)
    return est, sd

@triton.jit
def _lut_kernel(z_ptr, out_ptr, D, NNIB, BLK: tl.constexpr):
    """out[q, n*16 + v] = sum_i z[q, 4n+i] * bit_i(v), bit_i(v) = (v >> (3-i)) & 1."""
    q = tl.program_id(0); nb = tl.program_id(1) * BLK + tl.arange(0, BLK); m = nb < NNIB; v = tl.arange(0, 16)
    acc = tl.zeros((BLK, 16), dtype=tl.float32)
    for i in tl.static_range(4):
        zi = tl.load(z_ptr + q.to(tl.int64) * D + 4 * nb + i, mask=m, other=0.0)
        acc += zi[:, None] * ((v[None, :] >> (3 - i)) & 1).to(tl.float32)
    tl.store(out_ptr + q.to(tl.int64) * (NNIB * 16) + nb[:, None] * 16 + v[None, :], acc, mask=m[:, None])

def luts_fused(code, Qb, nt=None, half=False):
    """same outputs as luts(); the tier boundaries are multiples of 4, so one table over the leading D dims serves every tier prefix."""
    prepare_u(code); TS = code["TS"][:nt] if nt else code["TS"]; B = len(Qb); D = TS[-1][1]
    if "VP16" not in code: code["VP16"] = code["VP"].half().contiguous()
    Zq = (Qb.half() @ code["VP16"][:, :D]).float().contiguous()
    L = torch.empty(B, D // 4 * 16, device=dev, dtype=torch.float16 if half else torch.float32)
    _lut_kernel[(B, triton.cdiv(D // 4, 128))](Zq, L, D, D // 4, BLK=128)
    qsum = torch.stack([Zq[:, s:e].sum(1) for s, e, _ in TS]).contiguous(); qn2 = torch.stack([(Zq[:, s:e] ** 2).sum(1) for s, e, _ in TS]).contiguous()
    return L, qsum, qn2
