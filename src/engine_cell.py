"""Engine v8 with stage 1 on cell residuals (codes of x - mean(cell), constant q . mean(cell) per probed cell). Stages 2 and 3 unchanged
(stage 2 still estimates q . x from the global centre, so Theorem 1 and the error bars are untouched)."""
import time, numpy as np, torch
from engine3 import Engine8, Engine4, rf
import s1cell, s1small_cell, viterbi_fused
dev = "cuda"

def cellify(s1):
    """re-encode the stage-1 codes of an index already routed to its final cells against the cell means."""
    t0 = time.perf_counter(); N = len(s1["order"]); off = torch.as_tensor(s1["off"], device=dev); NL = len(off) - 1
    oc = s1["order"].cpu(); W1, tab, L = s1["W1"], s1["tab"], rf.L; R = W1.shape[1]; Tt = rf.make_table(L); alpha = float(s1["alpha"])
    cell = torch.searchsorted(off, torch.arange(N, device=dev), right=True) - 1
    means = torch.zeros(NL, rf.d, device=dev)
    for j in range(0, N, 65536): means.index_add_(0, cell[j:j + 65536], rf.host16[oc[j:j + 65536]].to(dev).float())
    means /= (off[1:] - off[:-1]).clamp(min=1).float()[:, None]
    words = torch.empty_like(s1["words"]); scale1 = torch.empty_like(s1["scale1"])
    for j in range(0, N, 16384):
        Rr = rf.host16[oc[j:j + 16384]].to(dev).float() - means[cell[j:j + 16384]]; n = len(Rr)
        Z = Rr @ W1; nz = Z.norm(dim=1).clamp(min=1e-6); Zs = Z / nz[:, None] * R ** 0.5
        path = torch.cat([viterbi_fused.viterbi_paths_fused(Zs[i:i + 2048], Tt, L, alpha) for i in range(0, n, 2048)])
        words[j:j + n] = rf.paths_to_words(path, L).to(torch.int32); Y = tab[path]; yn = Y.norm(dim=1); fip = (Zs * Y).sum(1) / (yn * R ** 0.5)
        scale1[j:j + n] = (nz / (fip * yn)).half()
    out = dict(s1); out.update(words=words, scale1=scale1, means16=means.half().contiguous(), cellify_s=time.perf_counter() - t0)
    return out

def compact(s1c):
    """final memory layout: one fp16 table of cell means serves routing (q . mean / |mean|, i.e. the spherical centroid direction) and the
    stage-1 constant; ids are int32; the fp32 routing centroids are dropped."""
    out = dict(s1c); m = out["means16"].float(); nrm = m.norm(dim=1).clamp(min=1e-12)
    out["inv_mnorm"] = (1.0 / nrm).contiguous(); out["order"] = out["order"].to(torch.int32)
    out["cen"] = out["inv_mnorm"]                       # 1-D: the small-batch kernels then route by q . mean / |mean| from the product they already have
    return out

def slim(code):
    """drop the build-only matrices (PCA basis, tier rotations) from the GPU once the query-side matrix VP is folded."""
    import tiered_fused
    tiered_fused.prepare(code); code["V"] = code["V"].cpu(); code["rots"] = [r.cpu() for r in code["rots"]]
    torch.cuda.empty_cache(); return code

def route(s1, Qb, QM, nprobe):
    if "inv_mnorm" in s1: return (QM.float() * s1["inv_mnorm"]).topk(nprobe, 1).indices
    return (Qb @ s1["cen"].T).topk(nprobe, 1).indices

class EngineC(Engine8):
    OPTS = dict(BN=32, BQ=32, CAP=32768, bucketed=False)                                             # cell residuals only
    SMALL, TINY = 32, 8
    def _gpu(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T
        a = (Qb @ s1["W1"], Qb, QM, s1["words"], s1["tab"], s1["scale1"], rf.L, K, s1["cen"], s1["off_t"], nprobe)
        if len(Qb) <= self.TINY: pos, _, ok = s1small_cell.search_small(*a)
        else:
            if not hasattr(self, "MB"): o = np.diff(s1["off"]); self.MB = int((o.max() + 31) // 32 + 1)
            pos, _, ok = s1small_cell.search_mask(*a, self.MB)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def stage1(self, Qb, K, nprobe, probe=None):
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T
        probe = route(s1, Qb, QM, nprobe)
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        return s1cell.search_guess(Qb @ s1["W1"], QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, L=rf.L, **self.OPTS)[0], QC

class EngineCT(EngineC):
    OPTS = dict(BN=64, BQ=16, CAP=32768, bucketed=False)                                             # + 64 x 16 tiles
class EngineCX(EngineC):
    OPTS = dict(BN=64, BQ=16, CAP=65536, bucketed=True)                                              # + larger buffer, bucketed selection

class EngineS(EngineC):
    """EngineC with the synchronisation-free stage 1 for every batch size: the whole search is one CUDA graph per (B, K, z, nprobe, k);
    queries flagged by the exactness test (rare) are recomputed alone with the work-list path."""
    M = None
    def _gpu(self, Qb, K, z, nprobe, k):
        if len(Qb) <= EngineC.SMALL: return EngineC._gpu(self, Qb, K, z, nprobe, k)
        s1 = self.s1; QC = Qb @ s1["c"]; QM = Qb.half() @ s1["means16"].T
        probe = route(s1, Qb, QM, nprobe)
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        pos, _, ok = s1cell.search_static(Qb @ s1["W1"], QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, M=self.M, L=rf.L)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        if ret_est: return EngineC.search(self, Qb, K, z, nprobe, k, ret_est)
        B = len(Qb); key = (B, K, z, nprobe, k)
        if key not in self.graphs:
            if Engine8.H is None: self._s3(Qb[:1], *[torch.zeros(1, 1, dtype=t, device=dev) for t in (torch.long, torch.float, torch.bool, torch.bool)])
            qs = Qb.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs)
        g, qs, (ok, I, nr, ids) = self.graphs[key]
        qs.copy_(Qb); g.replay(); I, nr, ids = I.clone(), nr.clone(), ids.clone()
        if not bool(ok.all()):                                                                       # the only synchronisation
            bad = torch.nonzero(~ok).flatten(); a, b, c = Engine4.search(self, Qb[bad], K, z, nprobe, k)
            I[bad], nr[bad] = a, b; ids = ids if ids.shape[1] == c.shape[1] else ids; ids[bad] = c
            self.nbad = getattr(self, "nbad", 0) + len(bad)
        return I, nr, ids

import tiered, tiered_fused
def cellify_code(s1c, code, chunk=16384):
    """tiered stage-2 codes of x - mean(cell) (same PCA basis, tiers and rotations as `code`)."""
    t0 = time.perf_counter(); order = s1c["order"]; N = len(order); off = torch.as_tensor(s1c["off"], device=dev)
    cell = torch.searchsorted(off, torch.arange(N, device=dev), right=True) - 1; means = s1c["means16"].float()
    TS, rots, V = code["TS"], code["rots"], code["V"]; oc = order.cpu()
    planes = [torch.empty_like(p) for p in code["planes"]]; scale = torch.empty_like(code["scale"]); sig = torch.empty_like(code["sig"])
    for j in range(0, N, chunk):
        Z = (rf.host16[oc[j:j + chunk]].to(dev).float() - means[cell[j:j + chunk]]) @ V; n = len(Z)
        for t, ((s, e, bb), P) in enumerate(zip(TS, rots)):
            F = Z[:, s:e] @ P; nf = F.norm(dim=1).clamp(min=1e-12); dt = e - s; Fu = F / nf[:, None] * dt ** 0.5
            cb, th = tiered.CB[bb]; q = torch.bucketize(Fu, th); Y = cb[q]; yn = Y.norm(dim=1); ct = (Fu * Y).sum(1) / (yn * dt ** 0.5)
            for p in range(bb): planes[t][p, j:j + n] = tiered.pack(((q >> (bb - 1 - p)) & 1).bool())
            scale[t, j:j + n] = (nf / (ct * yn)).half(); sig[t, j:j + n] = (nf * torch.sqrt((1 - ct ** 2).clamp(min=0)) / (ct * (dt - 1) ** 0.5)).half()
    out = dict(code); out.update(planes=planes, scale=scale, sig=sig, cellcode_s=time.perf_counter() - t0); out.pop("VP", None)
    return out

class EngineS2(EngineS):
    """EngineS with stage 2 on cell residuals too: est = q . mean(cell) (fp32) + tiered estimate of q . (x - mean(cell))."""
    def _score(self, Qb, QC, pos):
        s1 = self.s1
        if "means32" not in s1: s1["means32"] = s1["means16"].float(); s1["off_l"] = torch.as_tensor(s1["off"], device=dev)
        est, sg = tiered_fused.score(self.code, Qb, torch.zeros_like(QC), pos)
        cell = torch.searchsorted(s1["off_l"], pos, right=True) - 1
        return est + (Qb.float() @ s1["means32"].T).gather(1, cell), sg

class EngineT(EngineS):
    """EngineS with a lean path for at most TINY queries: one product with the cell means serves routing and the stage-1 constants,
    compact first-pass maxima, a small emit buffer, and the exactness flags and ids copied to pinned host memory inside the graph,
    so a search is one graph replay and one stream synchronisation."""
    TBETA, TRHO, TCAP = 1.1, 0.75, 2048
    PROJ = False
    def _tiny(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]; Qr = Qb @ s1["W1"]
        if self.PROJ:
            if "meansR" not in s1: s1["meansR"] = ((s1["means16"].float() - s1["c"]) @ s1["W1"]).half().contiguous()
            QM = (Qr.half() @ s1["meansR"].T).float() + QC[:, None]                                # q . m_c in the stage-1 subspace
        else: QM = Qb.half() @ s1["means16"].T
        pos, _, ok = s1small_cell.search_tiny(Qr, QM, s1["inv_mnorm"], s1["words"], s1["tab"], s1["scale1"], rf.L, K, s1["off_t"], nprobe, beta=self.TBETA, rho=self.TRHO, CAP=self.TCAP)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return torch.cat([ok.long()[:, None], ids.gather(1, score.topk(k, 1).indices).long()], 1)
    def search_host(self, Qb, K, z, nprobe, k):
        """returns the top-k ids as a numpy array (the latency path)."""
        B = len(Qb)
        if B > 1: return self.search(Qb, K, z, nprobe, k)[0].cpu().numpy()
        key = ("tiny", B, K, z, nprobe, k)
        if key not in self.graphs:
            if Engine8.H is None: self._s3(Qb[:1], *[torch.zeros(1, 1, dtype=t, device=dev) for t in (torch.long, torch.float, torch.bool, torch.bool)])
            qs = Qb.clone(); host = torch.empty((B, k + 1), dtype=torch.long, pin_memory=True); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): host.copy_(self._tiny(qs, K, z, nprobe, k), non_blocking=True)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = self._tiny(qs, K, z, nprobe, k); host.copy_(out, non_blocking=True)
            self.graphs[key] = (g, qs, host.numpy())
        g, qs, h = self.graphs[key]
        qs.copy_(Qb); g.replay(); torch.cuda.current_stream().synchronize()
        if h[:, 0].all(): return h[:, 1:].copy()
        self.nbad = getattr(self, "nbad", 0) + 1
        return EngineC.search(self, Qb, K, z, nprobe, k)[0].cpu().numpy()
