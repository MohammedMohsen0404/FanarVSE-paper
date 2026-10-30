"""Fused GPU scan for 4-bit codes (Triton): decode in registers -> tensor-core dot -> per-block max (exact block-max selection).
Layout: byte j of a vector holds dim j (high nibble) and dim j+H (low nibble), H = padded half-dim, so no interleaving is needed:
    q.c = dot(q[:H], hi) + dot(q[H:], lo).
Score: est[q,n] = scale[n] * (a * (q.c) + b * sum(q)),  dequantised value = a*c + b.
Pass 1 (kernel): block maxima (B, N/BLOCK_N).  Pass 2 (torch): top-k' blocks per query -> exact rescoring of their members."""
import torch, triton, triton.language as tl


@triton.jit
def _scan4_blockmax(q_ptr, codes_ptr, scale_ptr, qsum_ptr, out_ptr, B, N, a, b,
                    H: tl.constexpr, HB: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NB):
    pid_n = tl.program_id(0); pid_q = tl.program_id(1)
    rq = pid_q * BQ + tl.arange(0, BQ); rn = pid_n * BN + tl.arange(0, BN); ch = tl.arange(0, H)
    qmask = rq < B; nmask = rn < N
    q_hi = tl.load(q_ptr + rq[:, None] * (2 * H) + ch[None, :], mask=qmask[:, None], other=0.0)           # (BQ, H) fp16
    q_lo = tl.load(q_ptr + rq[:, None] * (2 * H) + H + ch[None, :], mask=qmask[:, None], other=0.0)
    byt = tl.load(codes_ptr + rn[:, None] * HB + ch[None, :], mask=nmask[:, None] & (ch[None, :] < HB), other=0)   # (BN, H) uint8
    hi = (byt >> 4).to(tl.float16); lo = (byt & 15).to(tl.float16)
    dot = tl.dot(q_hi, tl.trans(hi)) + tl.dot(q_lo, tl.trans(lo))                                        # (BQ, BN) fp32 acc
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    qs = tl.load(qsum_ptr + rq, mask=qmask, other=0.0)
    est = s[None, :] * (a * dot + b * qs[:, None])
    est = tl.where(nmask[None, :], est, -1e30)
    tl.store(out_ptr + rq * NB + pid_n, tl.max(est, axis=1), mask=qmask)


def scan4_blockmax(Qh, codes, scale, a, b, H, BN=64, BQ=64):
    """Qh: (B, 2H) fp16 (padded), codes: (N, HB) uint8, scale: (N,) fp16 -> blockmax (B, ceil(N/BN)) fp32"""
    B = Qh.shape[0]; N, HB = codes.shape; NB = triton.cdiv(N, BN)
    out = torch.empty(B, NB, device=Qh.device, dtype=torch.float32); qsum = Qh.float().sum(1).contiguous()
    _scan4_blockmax[(NB, triton.cdiv(B, BQ))](Qh, codes, scale, qsum, out, B, N, float(a), float(b), H=H, HB=HB, BQ=BQ, BN=BN, NB=NB)
    return out


def search4(Qh, codes, scale, a, b, H, kp, BN=64):
    """exact top-kp by estimated score using block maxima; returns candidate positions (B, kp)."""
    bm = scan4_blockmax(Qh, codes, scale, a, b, H, BN)
    blocks = bm.topk(kp, 1).indices                                                    # (B, kp) blocks
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=Qh.device)).reshape(len(Qh), -1).clamp(max=codes.shape[0] - 1)
    byt = codes[pos]                                                                    # (B, kp*BN, HB)
    hi = (byt >> 4).half(); lo = (byt & 15).half(); HB = codes.shape[1]
    dot = torch.einsum("bmh,bh->bm", hi, Qh[:, :HB]) + torch.einsum("bmh,bh->bm", lo, Qh[:, H:H + HB])
    est = scale[pos].float() * (a * dot.float() + b * Qh.float().sum(1, keepdim=True))
    return pos.gather(1, est.topk(kp, 1).indices)


@triton.jit
def _scan1_blockmax(qk_ptr, codes_ptr, scale_ptr, qsum_ptr, qc_ptr, out_ptr, B, N,
                    KB: tl.constexpr, KP: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NB):
    """qk: (8, B, KB) fp16, qk[k, :, j] = q[:, 8j+k]; codes: (N, KB) uint8 (bit 7-k of byte j = sign of dim 8j+k)."""
    pid_n = tl.program_id(0); pid_q = tl.program_id(1)
    rq = pid_q * BQ + tl.arange(0, BQ); rn = pid_n * BN + tl.arange(0, BN); cj = tl.arange(0, KP)
    qmask = rq < B; nmask = rn < N; cmask = cj < KB
    byt = tl.load(codes_ptr + rn[:, None] * KB + cj[None, :], mask=nmask[:, None] & cmask[None, :], other=0)
    acc = tl.zeros((BQ, BN), dtype=tl.float32)
    for k in tl.static_range(8):
        qk = tl.load(qk_ptr + k * B * KB + rq[:, None] * KB + cj[None, :], mask=qmask[:, None] & cmask[None, :], other=0.0)
        bit = ((byt >> (7 - k)) & 1).to(tl.float16)
        acc += tl.dot(qk, tl.trans(bit))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    qs = tl.load(qsum_ptr + rq, mask=qmask, other=0.0); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = s[None, :] * (2 * acc - qs[:, None]) + qc[:, None]
    est = tl.where(nmask[None, :], est, -1e30)
    tl.store(out_ptr + rq * NB + pid_n, tl.max(est, axis=1), mask=qmask)


@triton.jit
def _rescore1(qk_ptr, pos_ptr, codes_ptr, scale_ptr, qsum_ptr, qc_ptr, out_ptr, B, M,
              KB: tl.constexpr, KP: tl.constexpr, BM: tl.constexpr):
    """exact 1-bit estimate for gathered candidates: one program = 1 query x BM candidate positions."""
    q = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    cj = tl.arange(0, KP); cmask = cj < KB
    rows = tl.load(pos_ptr + q * M + rm, mask=mmask, other=0)
    byt = tl.load(codes_ptr + rows[:, None] * KB + cj[None, :], mask=mmask[:, None] & cmask[None, :], other=0)
    acc = tl.zeros((BM,), dtype=tl.float32)
    for k in tl.static_range(8):
        qrow = tl.load(qk_ptr + k * B * KB + q * KB + cj, mask=cmask, other=0.0).to(tl.float32)
        bit = ((byt >> (7 - k)) & 1).to(tl.float32)
        acc += tl.sum(bit * qrow[None, :], axis=1)
    s = tl.load(scale_ptr + rows, mask=mmask, other=0.0).to(tl.float32)
    est = s * (2 * acc - tl.load(qsum_ptr + q)) + tl.load(qc_ptr + q)
    tl.store(out_ptr + q * M + rm, est, mask=mmask)


def search1(Qz, qc, codes, scale, kp, BN=64, BQ=64):
    """Qz: (B, D) rotated queries (float), qc: (B,) <q,c>; codes (N, D/8) uint8 packed MSB-first; exact top-kp by estimate."""
    B, D = Qz.shape; N, KB = codes.shape; NB = triton.cdiv(N, BN)
    qk = Qz.reshape(B, KB, 8).permute(2, 0, 1).contiguous().half()            # (8, B, KB)
    qsum = Qz.float().sum(1).contiguous(); qcf = qc.float().contiguous()
    bm = torch.empty(B, NB, device=Qz.device, dtype=torch.float32)
    _scan1_blockmax[(NB, triton.cdiv(B, BQ))](qk, codes, scale, qsum, qcf, bm, B, N, KB=KB, KP=max(16, triton.next_power_of_2(KB)), BQ=BQ, BN=BN, NB=NB)
    blocks = bm.topk(kp, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=Qz.device)).reshape(B, -1).clamp(max=N - 1)
    pos = pos.contiguous(); M = pos.shape[1]; est = torch.empty(B, M, device=Qz.device, dtype=torch.float32)
    _rescore1[(triton.cdiv(M, 64), B)](qk, pos, codes, scale, qsum, qcf, est, B, M, KB=KB, KP=max(16, triton.next_power_of_2(KB)), BM=64)
    return pos.gather(1, est.topk(kp, 1).indices)


@triton.jit
def _scan1_cell(qk_ptr, qrows_ptr, codes_ptr, scale_ptr, qsum_ptr, qc_ptr, out_ptr, B, NQS, N, blk0, nblk,
                KB: tl.constexpr, KP: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, NB):
    """block-max over blocks [blk0, blk0+nblk) for the query subset qrows (indices into the batch)."""
    pid_n = tl.program_id(0); pid_q = tl.program_id(1)
    bidx = blk0 + pid_n
    rqi = pid_q * BQ + tl.arange(0, BQ); qmask = rqi < NQS
    rq = tl.load(qrows_ptr + rqi, mask=qmask, other=0)
    rn = bidx * BN + tl.arange(0, BN); nmask = rn < N
    cj = tl.arange(0, KP); cmask = cj < KB
    byt = tl.load(codes_ptr + rn[:, None] * KB + cj[None, :], mask=nmask[:, None] & cmask[None, :], other=0)
    acc = tl.zeros((BQ, BN), dtype=tl.float32)
    for k in tl.static_range(8):
        qk = tl.load(qk_ptr + k * B * KB + rq[:, None] * KB + cj[None, :], mask=qmask[:, None] & cmask[None, :], other=0.0)
        acc += tl.dot(qk, tl.trans(((byt >> (7 - k)) & 1).to(tl.float16)))
    s = tl.load(scale_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    qs = tl.load(qsum_ptr + rq, mask=qmask, other=0.0); qc = tl.load(qc_ptr + rq, mask=qmask, other=0.0)
    est = tl.where(nmask[None, :], s[None, :] * (2 * acc - qs[:, None]) + qc[:, None], -1e30)
    tl.store(out_ptr + rq * NB + bidx, tl.max(est, axis=1), mask=qmask)


def search1_ivf(Qz, Qorig, qc, codes, scale, cen, off, nprobe, kp, BN=32, BQ=32):
    """codes/scale sorted by cell; off: cell offsets (numpy, nlist+1); cen: (nlist, d) centroids for routing (on original q)."""
    B, D = Qz.shape; N, KB = codes.shape; NB = triton.cdiv(N, BN); KP = max(16, triton.next_power_of_2(KB))
    qk = Qz.reshape(B, KB, 8).permute(2, 0, 1).contiguous().half()
    qsum = Qz.float().sum(1).contiguous(); qcf = qc.float().contiguous()
    bm = torch.full((B, NB), -1e30, device=Qz.device, dtype=torch.float32)
    probe = (Qorig @ cen.T).topk(nprobe, 1).indices
    flat = probe.flatten(); qid = torch.arange(B, device=Qz.device).repeat_interleave(nprobe)
    o = torch.argsort(flat); flat, qid = flat[o], qid[o]
    cells, counts = torch.unique_consecutive(flat, return_counts=True); starts = (torch.cumsum(counts, 0) - counts).tolist()
    for c, st, ct in zip(cells.tolist(), starts, counts.tolist()):
        s, e = int(off[c]), int(off[c + 1])
        if e == s: continue
        b0 = s // BN; nb = (e - 1) // BN - b0 + 1; qrows = qid[st:st + ct].contiguous()
        _scan1_cell[(nb, triton.cdiv(ct, BQ))](qk, qrows, codes, scale, qsum, qcf, bm, B, ct, N, b0, nb, KB=KB, KP=KP, BQ=BQ, BN=BN, NB=NB)
    blocks = bm.topk(kp, 1).indices
    pos = (blocks[:, :, None] * BN + torch.arange(BN, device=Qz.device)).reshape(B, -1).clamp(max=N - 1).contiguous()
    M = pos.shape[1]; est = torch.empty(B, M, device=Qz.device, dtype=torch.float32)
    _rescore1[(triton.cdiv(M, 64), B)](qk, pos, codes, scale, qsum, qcf, est, B, M, KB=KB, KP=KP, BM=64)
    return pos.gather(1, est.topk(kp, 1).indices)


@triton.jit
def _rescore_planes(qk_ptr, pos_ptr, planes_ptr, cb_ptr, s2_ptr, qc_ptr, out_ptr, B, M, N,
                    NBITS: tl.constexpr, KB: tl.constexpr, KP: tl.constexpr, BM: tl.constexpr):
    """b-bit codes stored as NBITS bit-planes (plane p: (N, KB) uint8). value = cb[code]; est = s2 * <q, value> + qc."""
    q = tl.program_id(1); rm = tl.program_id(0) * BM + tl.arange(0, BM); mmask = rm < M
    cj = tl.arange(0, KP); cmask = cj < KB; m2 = mmask[:, None] & cmask[None, :]
    rows = tl.load(pos_ptr + q * M + rm, mask=mmask, other=0)
    acc = tl.zeros((BM,), dtype=tl.float32)
    for k in tl.static_range(8):
        code = tl.zeros((BM, KP), dtype=tl.int32)
        for p in tl.static_range(NBITS):
            byt = tl.load(planes_ptr + p * N * KB + rows[:, None] * KB + cj[None, :], mask=m2, other=0)
            code += (((byt >> (7 - k)) & 1).to(tl.int32)) << (NBITS - 1 - p)
        val = tl.load(cb_ptr + code)
        qrow = tl.load(qk_ptr + k * B * KB + q * KB + cj, mask=cmask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(m2, val * qrow[None, :], 0.0), axis=1)
    s2 = tl.load(s2_ptr + rows, mask=mmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + q * M + rm, acc * s2 + tl.load(qc_ptr + q), mask=mmask)


def rescore_planes(Qf, qc, pos, planes, cb, s2):
    """Qf: (B, d) queries rotated into the refinement basis; planes: (NBITS, N, d/8) uint8; returns est (B, M)."""
    B, d = Qf.shape; NB_, N, KB = planes.shape; M = pos.shape[1]
    qk = Qf.reshape(B, KB, 8).permute(2, 0, 1).contiguous().float()
    out = torch.empty(B, M, device=Qf.device, dtype=torch.float32)
    _rescore_planes[(triton.cdiv(M, 32), B)](qk, pos.contiguous(), planes, cb.float().contiguous(), s2, qc.float().contiguous(), out, B, M, N,
                                             NBITS=NB_, KB=KB, KP=max(16, triton.next_power_of_2(KB)), BM=32)
    return out
