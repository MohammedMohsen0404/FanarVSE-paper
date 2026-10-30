"""Space-extreme FanarVSE: every per-vector array in pinned host RAM, read by the GPU over PCIe; the GPU keeps only small shared tables
(cell means projected on the stage-1 basis, W1, VP; the id map lives on the host). Variants are measured one after another."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, numpy as np, torch
sys.path.insert(0, f"{FANAR}/host_resident"); os.environ.setdefault("TAG", "spacex")
from h import *
from engine3 import Engine10, Engine8
from engine_cell import EngineS, EngineC
import stage3_zc, s1cell, tiered_fused
dev = "cuda"

class HObj:
    def __init__(self, t): self.t, self.dtype, self.shape = t, t.dtype, t.shape
    def data_ptr(self): return self.t.data_ptr()

class SpaceX(EngineS, Engine10):
    SMALL = 0
    def setup(self, proj=True, ids_host=True, vp16=False):
        s1 = self.s1; self.proj, self.ids_host = proj, ids_host
        tiered_fused.prepare(self.code)
        if vp16: self.code["VP"] = self.code["VP"].half().contiguous(); s1["W1"] = s1["W1"].half().contiguous()
        s1["words"] = HObj(stage3_zc.pin(s1["words"].cpu())); s1["scale1"] = HObj(stage3_zc.pin(s1["scale1"].cpu()))
        for k_ in ("V", "rots"): self.code.pop(k_, None)
        s1.pop("cen", None)
        if proj:
            s1["meansR"] = ((s1["means16"].float() - s1["c"]) @ s1["W1"].float()).half().contiguous(); s1.pop("means16")
        if ids_host:
            self.order_np = s1["order"].cpu().numpy(); Engine8.H = stage3_zc.pin(rf.host16[torch.from_numpy(self.order_np.astype(np.int64))])
            s1["order"] = None
        else: Engine8.H = stage3_zc.pin(rf.host16)
        torch.cuda.empty_cache(); self.graphs = {}
    def _QM(self, Qb, Qr, QC):
        s1 = self.s1
        if self.proj: return (Qr.half() @ s1["meansR"].T).float() + QC[:, None]
        return (Qb.half() @ s1["means16"].T).float()
    def _gpu(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]; Qr = (Qb.to(s1["W1"].dtype) @ s1["W1"]).float(); QM = self._QM(Qb, Qr, QC)
        probe = (QM * s1["inv_mnorm"]).topk(nprobe, 1).indices
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        pos, _, ok = s1cell.search_static(Qr, QM, s1["words"], s1["tab"], s1["scale1"], K, s1["off_t"], probe, beta=beta, rho=rho, M=self.M, L=rf.L)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = pos if self.ids_host else s1["order"][pos]
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9)); score = stage3_zc.stage3(Engine8.H, ids, amb, Qb, score)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        B = len(Qb); key = (B, K, z, nprobe, k)
        if key not in self.graphs:
            qs = Qb.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs)
        g, qs, (ok, I, nr, ids) = self.graphs[key]
        qs.copy_(Qb); g.replay(); I, nr, ids = I.clone(), nr.clone(), ids.clone()
        if not bool(ok.all()):                                                                       # exactness: recompute flagged queries
            bad = torch.nonzero(~ok).flatten(); self.nbad = getattr(self, "nbad", 0) + len(bad)
            a, b, c = self._exact(Qb[bad], K, z, nprobe, k); I[bad], nr[bad], ids[bad] = a, b, c
        return I, nr, ids
    def _exact(self, Qb, K, z, nprobe, k):
        """flagged queries: the same pipeline with a buffer large enough for every probed item (no overflow, count >= K)."""
        old = self.M; self.M = 1 << 20
        try:
            s1 = self.s1; QC = Qb @ s1["c"].to(Qb.dtype) if False else None
            ok, I, nr, ids = self._gpu(Qb, K, z, nprobe, k)
        finally: self.M = old
        return I, nr, ids
    def run(self, K, z, nprobe, k=10, B=500, nq=2000):
        I, r, C = [], [], []
        for s in range(0, nq, B):
            a, b, c = self.search(rf.Qg[s:s + B], K, z, nprobe, k)
            a = a.cpu().numpy(); c = c.cpu().numpy()
            if self.ids_host: a, c = self.order_np[a], self.order_np[c]
            I.append(a); r.append(b.cpu().numpy()); C.append(c)
        return np.vstack(I), np.concatenate(r), np.vstack(C)

def gpu_bytes(e):
    seen, tot = set(), 0
    def walk(x):
        nonlocal tot
        if isinstance(x, torch.Tensor):
            if x.is_cuda and x.untyped_storage().data_ptr() not in seen: seen.add(x.untyped_storage().data_ptr()); tot += x.untyped_storage().nbytes()
        elif isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, (list, tuple)): [walk(v) for v in x]
    walk(e.s1); walk(e.code); walk(getattr(e, "pk", {})); return tot

if __name__ == "__main__":
    import copy
    for proj, ids_host, vp16 in [tuple(bool(int(c)) for c in v) for v in os.environ.get("VARIANTS", "000,110,111").split(",")]:
        e, mem, e3 = build(cls=SpaceX, lean=True); e.setup(proj=proj, ids_host=ids_host, vp16=vp16)
        N = rf.N; gb = gpu_bytes(e); m = {"gpu_B": round(gb / N, 2), "gpu_MB": round(gb / 1e6, 1), "proj": proj, "ids_host": ids_host, "vp16": vp16}
        bench(e, m, fallbacks=0); print("fallback queries:", getattr(e, "nbad", 0), flush=True)
        del e, e3; Engine8.H = None; torch.cuda.empty_cache()
