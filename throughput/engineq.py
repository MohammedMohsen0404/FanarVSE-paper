"""EngineQ: EngineS with a progressive stage 2. The leading NT tiers of the refinement code are scored for all K stage-1 candidates;
only the best K2 by that partial estimate get the full tiered score, its error bars and stage 3. Same index, same memory."""
import torch
import tiered_fused
from engine_cell import EngineS, EngineC, route
from engine3 import rf
import s1cell
dev = "cuda"

class EngineQ(EngineS):
    NT = 3; K2 = 100
    def _sub(self):
        if not hasattr(self, "_c3"):
            c = self.code; tiered_fused.prepare(c)
            self._c3 = dict(c, TS=c["TS"][:self.NT], planes=c["planes"][:self.NT], cbf=c["cbf"][:self.NT])
        return self._c3
    def _gpu(self, Qb, K, z, nprobe, k):
        if len(Qb) <= EngineC.SMALL: return EngineC._gpu(self, Qb, K, z, nprobe, k)
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T
        probe = route(s1, Qb, QM, nprobe)
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        pos, _, ok = s1cell.search_static(Qb @ s1["W1"], QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, M=self.M, L=rf.L)
        K2 = min(self.K2, K)
        if K2 < K:
            pe, _ = tiered_fused.score(self._sub(), Qb, QC, pos); pos = pos.gather(1, pe.topk(K2, 1).indices).contiguous()
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        from engine3 import Engine8, Engine4
        B = len(Qb); key = (B, K, z, nprobe, k, self.K2, self.NT)
        if B <= EngineC.SMALL: return EngineS.search(self, Qb, K, z, nprobe, k, ret_est)
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
