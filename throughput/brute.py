"""Routing-free stage 1: int8 tensor-core scan of ALL vectors with R-dim PCA codes (per-vector fp16 scale), fused threshold emit.
The per-query threshold comes from a fixed random sample of the database (the r-th best sample score, r chosen so ~2K items pass);
a query is exact (top K of the full scan) when K <= emitted <= M, else it is flagged (as in the cell engine)."""
import numpy as np, torch, triton, triton.language as tl
dev = "cuda"

@triton.jit
def _scan(q_ptr, qs_ptr, z_ptr, zs_ptr, t0_ptr, cnt_ptr, bval_ptr, bpos_ptr, out_ptr, B, N, CAP,
          R: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0); rn = pid * BN + tl.arange(0, BN); nmask = rn < N; j = tl.arange(0, R)
    z = tl.load(z_ptr + rn[:, None].to(tl.int64) * R + j[None, :], mask=nmask[:, None], other=0)
    zs = tl.load(zs_ptr + rn, mask=nmask, other=0.0).to(tl.float32)
    for t in range(0, tl.cdiv(B, BQ)):
        rq = t * BQ + tl.arange(0, BQ); qmask = rq < B
        q = tl.load(q_ptr + rq[:, None] * R + j[None, :], mask=qmask[:, None], other=0)
        acc = tl.dot(q, tl.trans(z))                                                # int32 (BQ, BN)
        est = acc.to(tl.float32) * tl.load(qs_ptr + rq, mask=qmask, other=0.0)[:, None] * zs[None, :]
        if MODE == 0:                                                               # dense scores (sample)
            tl.store(out_ptr + rq[:, None] * N + rn[None, :], est, mask=qmask[:, None] & nmask[None, :])
        elif MODE == 3:                                                             # emit, one atomic per row, prefix by tensor-core dot
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            kc = tl.sum(keep.to(tl.int32), axis=1)
            if tl.max(kc, axis=0) > 0:
                pre = tl.cumsum(keep.to(tl.int32), axis=1)
                base = tl.atomic_add(cnt_ptr + rq, kc, mask=qmask & (kc > 0))
                idx = base[:, None] + pre - 1
                ok = keep & (idx < CAP)
                tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
                tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :] + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)
        elif MODE == 1:                                                             # emit above threshold
            t0 = tl.load(t0_ptr + rq, mask=qmask, other=1e30)
            keep = (est >= t0[:, None]) & nmask[None, :] & qmask[:, None]
            idx = tl.atomic_add(cnt_ptr + rq[:, None] + tl.zeros((BQ, BN), dtype=tl.int32), tl.full((BQ, BN), 1, tl.int32), mask=keep)
            ok = keep & (idx < CAP)
            tl.store(bval_ptr + rq[:, None] * CAP + idx, est, mask=ok)
            tl.store(bpos_ptr + rq[:, None] * CAP + idx, rn[None, :] + tl.zeros((BQ, BN), dtype=tl.int32), mask=ok)
        else:                                                                       # GEMM only (timing)
            tl.store(out_ptr + rq, tl.max(est, axis=1), mask=qmask & (pid == 0))

class Brute:
    def __init__(self, X16, mu, V, R=128, sample_every=20, chunk=65536):
        self.R, self.mu, self.V = R, mu, V[:, :R].contiguous(); N = len(X16); self.N = N
        self.Z = torch.empty(N, R, dtype=torch.int8, device=dev); self.zs = torch.empty(N, dtype=torch.float16, device=dev)
        for j in range(0, N, chunk):
            z = (X16[j:j + chunk].to(dev).float() - mu) @ self.V; a = z.abs().amax(1).clamp(min=1e-12) / 127
            self.Z[j:j + chunk] = torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8); self.zs[j:j + chunk] = a.half()
        self.sidx = torch.arange(0, N, sample_every, device=dev); self.Zs, self.zss = self.Z[self.sidx].contiguous(), self.zs[self.sidx].contiguous()
        self.frac = len(self.sidx) / N
    def qcode(self, Qb):
        z = (Qb - self.mu) @ self.V; a = z.abs().amax(1).clamp(min=1e-12) / 127
        return torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8).contiguous(), a.float().contiguous()
    def scores(self, q, qs, Z, zs, BQ=64, BN=128, nw=4):
        B, n = len(q), len(Z); out = torch.empty(B, n, device=dev); d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
        _scan[(triton.cdiv(n, BN),)](q, qs, Z, zs, d, di, d, di, out, B, n, 1, R=self.R, BQ=BQ, BN=BN, MODE=0, num_warps=nw); return out
    def search(self, Qb, K, over=2.0, M=None, BQ=64, BN=128, nw=4, mode=1):
        q, qs = self.qcode(Qb); B = len(Qb); M = M or max(4096, 8 * K)
        r = max(1, int(round(over * K * self.frac)))
        t0 = self.scores(q, qs, self.Zs, self.zss).topk(r, 1).values[:, -1].contiguous()
        cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
        out = torch.empty(B, device=dev)
        _scan[(triton.cdiv(self.N, BN),)](q, qs, self.Z, self.zs, t0, cnt, bval, bpos, out, B, self.N, M, R=self.R, BQ=BQ, BN=BN, MODE=mode, num_warps=nw)
        t = bval.topk(K, 1); return bpos.gather(1, t.indices).long(), t.values, (cnt >= K) & (cnt <= M), cnt
