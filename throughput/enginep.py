"""EngineP: EngineS with the first pass of stage 1 replaced by a predicted threshold (one emit pass). The predictor is a linear fit of
the K-th stage-1 estimate on routing features, trained on pseudo-queries sampled from the database (no test queries), per (n_p, K).
t0 = prediction - margin; a query whose emit count falls outside [K, M] is flagged and recomputed exactly (as in EngineS)."""
import numpy as np, torch, triton
from engine_cell import EngineS, EngineC, route
from engine3 import Engine4, rf
import s1cell
from s1cell import _static_blocks, _pairs_static, _emit_loop_s
dev = "cuda"

def feats(QM, probe, Q):
    srt = QM.float().gather(1, probe).sort(1, descending=True).values; n = srt.shape[1]
    return torch.stack([torch.ones(len(Q), device=dev), srt[:, 0], srt[:, min(4, n - 1)], srt[:, min(16, n - 1)], srt[:, min(63, n - 1)], srt.mean(1), Q.norm(dim=1)], 1)

class EngineP(EngineS):
    MARGIN_Q = 0.001; MSIZE = 16384; NTRAIN = 3000
    def _fit(self, K, nprobe):
        key = (K, nprobe)
        if not hasattr(self, "W"): self.W = {}
        if key in self.W: return self.W[key]
        s1 = self.s1; g = torch.Generator().manual_seed(1)
        idx = torch.randperm(rf.N, generator=g)[:self.NTRAIN]; Q = rf.host16[idx].to(dev).float()
        Q = Q + 0.02 * torch.randn_like(Q) * Q.norm(dim=1, keepdim=True) / Q.shape[1] ** 0.5       # avoid exact self-matches
        tk, F = [], []
        for s in range(0, len(Q), 500):
            Qb = Q[s:s + 500]; QM = Qb.half() @ s1["means16"].T; probe = route(s1, Qb, QM, nprobe)
            beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
            _, est, ok = s1cell.search_static(Qb @ s1["W1"], QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, L=rf.L)
            tk.append(est[:, K - 1]); F.append(feats(QM, probe, Qb))
        tk, F = torch.cat(tk), torch.cat(F); w = torch.linalg.lstsq(F, tk[:, None]).solution.flatten()
        res = tk - F @ w; mg = float(-res.quantile(self.MARGIN_Q))
        self.W[key] = (w.contiguous(), mg); return self.W[key]
    def _gpu(self, Qb, K, z, nprobe, k):
        if len(Qb) <= EngineC.SMALL: return EngineC._gpu(self, Qb, K, z, nprobe, k)
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T
        probe = route(s1, Qb, QM, nprobe); w, mg = self.W[(K, nprobe)]
        t0 = (feats(QM, probe, Qb) @ w - mg).contiguous()
        words, tab, scale, off_t = s1["words"], s1["tab"], s1["scale1"], s1["off_t"]
        Qr = (Qb @ s1["W1"]).half().contiguous(); B, R = Qr.shape; N, W = words.shape; RP = triton.next_power_of_2(R); NC = len(off_t) - 1; M = self.MSIZE
        pcell, pblk, ps, pe, MB = _static_blocks(off_t, 32)
        qid, coff, ccnt, qcp, _ = _pairs_static(probe, NC, QM)
        cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
        _emit_loop_s[(len(pcell),)](Qr, qid, coff, ccnt, pcell, pblk, ps, pe, words, tab, scale, qcp, t0, cnt, bval, bpos, M, W=W, R=R, RP=RP, L=rf.L, BQ=32, BN=32, num_stages=1)
        t = bval.topk(K, 1); pos = bpos.gather(1, t.indices).long(); ok = (cnt >= K) & (cnt <= M)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        if len(Qb) > EngineC.SMALL: self._fit(K, nprobe)
        return EngineS.search(self, Qb, K, z, nprobe, k, ret_est)
