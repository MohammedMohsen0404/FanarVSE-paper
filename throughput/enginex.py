"""EngineX = predicted-threshold stage 1 (no first pass) + in-graph exact repair + LUT stage 2 (uniform refinement code).
Stage 1: t0 = linear prediction of the K-th estimate from routing features (fit on database pseudo-queries) minus a margin; one emit
pass; a query is exact when K <= emitted <= M. Up to RB failed queries per batch are recomputed inside the CUDA graph with the
guess-and-verify search (fixed size: the RB rows with failure flags first); any query still failing is recomputed outside (rare)."""
import numpy as np, torch, triton
from engine_cell import EngineS, EngineC, route
from engine3 import Engine8, Engine4, rf
import s1cell, s2lut
from s1cell import _static_blocks, _pairs_static, _emit_loop_s
from enginep import feats
dev = "cuda"

class EngineX(EngineS):
    MARGIN_Q = 0.01; MSIZE = 4096; NTRAIN = 4000; RB = 64; K2 = None; NT = 3
    def _score(self, Qb, QC, pos): return s2lut.score_lut(self.code, Qb, QC, pos, BM=16, nw=2)
    def _fit(self, K, nprobe):
        if not hasattr(self, "W"): self.W = {}
        if (K, nprobe) in self.W: return
        s1 = self.s1; g = torch.Generator().manual_seed(1); idx = torch.randperm(rf.N, generator=g)[:self.NTRAIN]
        Q = rf.host16[idx].to(dev).float(); Q = Q + 0.02 * torch.randn_like(Q) * Q.norm(dim=1, keepdim=True) / Q.shape[1] ** 0.5
        tk, F = [], []
        for s in range(0, len(Q), 500):
            Qb = Q[s:s + 500]; QM = Qb.half() @ s1["means16"].T; probe = route(s1, Qb, QM, nprobe); beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
            _, est, ok = s1cell.search_static(Qb @ s1["W1"], QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, L=rf.L)
            tk.append(est[:, K - 1]); F.append(feats(QM, probe, Qb))
        tk, F = torch.cat(tk), torch.cat(F); w = torch.linalg.lstsq(F, tk[:, None]).solution.flatten()
        self.W[(K, nprobe)] = (w.contiguous(), float(-(tk - F @ w).quantile(self.MARGIN_Q)))
    def _stage1(self, Qb, QM, probe, K):
        s1 = self.s1; w, mg = self.W[(K, probe.shape[1])]; self.MSIZE = max(self.MSIZE, 8 * K)
        t0 = (feats(QM, probe, Qb) @ w - mg).contiguous()
        words, tab, scale, off_t = s1["words"], s1["tab"], s1["scale1"], s1["off_t"]
        Qr = (Qb @ s1["W1"]); Qh = Qr.half().contiguous(); B, R = Qr.shape; N, W = words.shape; RP = triton.next_power_of_2(R); NC = len(off_t) - 1; M = self.MSIZE
        pcell, pblk, ps, pe, MB = _static_blocks(off_t, 32)
        qid, coff, ccnt, qcp, _ = _pairs_static(probe, NC, QM)
        cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
        _emit_loop_s[(len(pcell),)](Qh, qid, coff, ccnt, pcell, pblk, ps, pe, words, tab, scale, qcp, t0, cnt, bval, bpos, M, W=W, R=R, RP=RP, L=rf.L, BQ=32, BN=32, num_stages=1)
        pos = bpos.gather(1, bval.topk(K, 1).indices).long(); ok = (cnt >= K) & (cnt <= M)
        # in-graph repair of up to RB failed rows (fixed size)
        rb = min(self.RB, B); ridx = (~ok).to(torch.int32).topk(rb).indices
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        p2, _, ok2 = s1cell.search_static(Qr[ridx], QM[ridx], words, tab, scale, K, off_t, probe[ridx], beta=beta, rho=rho, L=rf.L)
        fail = ~ok[ridx]
        pos[ridx] = torch.where(fail[:, None], p2, pos[ridx]); ok = ok.clone(); ok[ridx] = torch.where(fail, ok2, ok[ridx])
        return pos.contiguous(), ok
    def _gpu(self, Qb, K, z, nprobe, k):
        if len(Qb) <= EngineC.SMALL: return EngineC._gpu(self, Qb, K, z, nprobe, k)
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T; probe = route(s1, Qb, QM, nprobe)
        pos, ok = self._stage1(Qb, QM, probe, K)
        if self.K2 is not None and self.K2 < K:
            pre = s2lut.luts(self.code, Qb)
            pe, _ = s2lut.score_lut(self.code, Qb, QC, pos, nt=self.NT, BM=16, nw=2, pre=pre)
            pos = pos.gather(1, pe.topk(self.K2, 1).indices).contiguous()
            est, sg = s2lut.score_lut(self.code, Qb, QC, pos, BM=16, nw=2, pre=pre)
        else: est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        B = len(Qb)
        if B <= EngineC.SMALL: return EngineS.search(self, Qb, K, z, nprobe, k, ret_est)
        self._fit(K, nprobe); key = (B, K, z, nprobe, k, self.K2)
        if key not in self.graphs:
            if Engine8.H is None: self._s3(Qb[:1], *[torch.zeros(1, 1, dtype=t, device=dev) for t in (torch.long, torch.float, torch.bool, torch.bool)])
            qs = Qb.clone(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(st); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs)
        g, qs, (ok, I, nr, ids) = self.graphs[key]
        qs.copy_(Qb); g.replay(); I, nr, ids = I.clone(), nr.clone(), ids.clone()
        if not bool(ok.all()):
            bad = torch.nonzero(~ok).flatten(); a, b, c = Engine4.search(self, Qb[bad], K, z, nprobe, k)
            I[bad], nr[bad] = a, b; ids[bad] = c[:, :ids.shape[1]]; self.nbad = getattr(self, "nbad", 0) + len(bad)
        return I, nr, ids
