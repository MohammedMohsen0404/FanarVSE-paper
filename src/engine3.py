"""FanarVSE engine v3: threshold-and-emit stage 1 (decode-once kernel, many cells), tiered stage-2 refinement, stage 3 on the GPU
(ambiguous rows gathered from the fp16 host copy and scored on the GPU). run() returns final ids, reads per query, candidate ids."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from ideas_engine2 import Engine2, load, reroute, rf
import s1fast, tiered
dev = "cuda"

class Engine3(Engine2):
    CAP = 32768
    def stage1(self, Qb, K, nprobe, probe=None):
        s1 = self.s1; QC = Qb @ s1["c"]
        return s1fast.search_thresh2(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off_t"], nprobe, CAP=self.CAP, probe=probe)[0], QC
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        pos, QC = self.stage1(Qb, K, nprobe); est, sg = tiered.score(self.code, Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = self.s1["order"][pos]; bi, bk = torch.nonzero(amb, as_tuple=True)
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        if len(bi):
            rows = rf.host16.index_select(0, ids[bi, bk].cpu()).to(dev, non_blocking=True).float()
            score[bi, bk] = (rows * Qb[bi]).sum(1)
        out = ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
        return out + (est, sg) if ret_est else out
    def run(self, K, z, nprobe, k=10, B=500, nq=2000):
        I, r, C = [], [], []
        for s in range(0, nq, B):
            a, b, c = self.search(rf.Qg[s:s + B], K, z, nprobe, k); I.append(a.cpu().numpy()); r.append(b.cpu().numpy()); C.append(c.cpu().numpy())
        return np.vstack(I), np.concatenate(r), np.vstack(C)
    def bench(self, K, z, nprobe, k=10, B=500, nq=2000):
        self.run(K, z, nprobe, k, B, nq); q = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter(); I, r, C = self.run(K, z, nprobe, k, B, nq); torch.cuda.synchronize(); q.append(nq / (time.perf_counter() - t0))
        return I, r, C, float(np.mean(q)), float(np.std(q))

def make(R=256, nlist=4096, B=3840, seed=0):
    s1, code = load(B, seed); s1n, cn = reroute(s1, code, nlist, seed); s1n["off_t"] = torch.as_tensor(s1n["off"], device=dev)
    return Engine3(s1n, cn), s1["words"].shape[1] * 4 + 2 + cn["bytes"] + 10

import tiered_fused
class Engine4(Engine3):
    """v4: guess-and-verify stage 1 (fp16 queries) + fused tiered stage 2."""
    def stage1(self, Qb, K, nprobe, probe=None):
        s1 = self.s1; QC = Qb @ s1["c"]
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        return s1fast.search_guess(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off_t"], nprobe, beta=beta, rho=rho, CAP=self.CAP)[0], QC
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        pos, QC = self.stage1(Qb, K, nprobe); est, sg = self._score(Qb, QC, pos) if hasattr(self, "_score") else tiered_fused.score(self.code, Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = self.s1["order"][pos]; bi, bk = torch.nonzero(amb, as_tuple=True)
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        if len(bi):
            rows = rf.host16.index_select(0, ids[bi, bk].cpu()).to(dev, non_blocking=True).float()
            score[bi, bk] = (rows * Qb[bi]).sum(1)
        out = ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
        return out + (est, sg) if ret_est else out

class Engine5(Engine4):
    """v5: v4 plus a synchronisation-free stage 1 for batches of at most SMALL queries, and one host sync per search for small batches."""
    SMALL = 4
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        if len(Qb) > self.SMALL or ret_est: return super().search(Qb, K, z, nprobe, k, ret_est)
        s1 = self.s1; QC = Qb @ s1["c"]
        pos, _, ok = s1fast.search_small(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off_t"], nprobe)
        est, sg = tiered_fused.score(self.code, Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]
        flat = torch.where(amb, ids, torch.full_like(ids, -1)).flatten()
        host = torch.cat([ok.long(), flat]).cpu()                                 # the only synchronisation
        if not bool(host[:len(Qb)].all()): return super().search(Qb, K, z, nprobe, k)
        sel = torch.nonzero(host[len(Qb):] >= 0).flatten()
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        if len(sel):
            rows = rf.host16.index_select(0, host[len(Qb):][sel]).to(dev, non_blocking=True).float()
            selg = sel.to(dev); bi, bk = selg // pos.shape[1], selg % pos.shape[1]
            score[bi, bk] = (rows * Qb[bi]).sum(1)
        return ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids

class Engine6(Engine5):
    """v6: small batches replay a captured CUDA graph for stages 1-2 and the classification; one host sync, then the exact reads."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw); self.graphs = {}
    SMALL = 32
    def _gpu(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]
        a = (Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off_t"], nprobe)
        if len(Qb) <= 4: pos, _, ok = s1fast.search_small(*a)
        else:
            if not hasattr(self, "MB"): o = np.diff(s1["off"]); self.MB = int((o.max() + 31) // 32 + 1)
            pos, _, ok = s1fast.search_mask(*a, self.MB)
        est, sg = tiered_fused.score(self.code, Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; flat = torch.where(amb, ids, torch.full_like(ids, -1)).flatten()
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        return torch.cat([ok.long(), flat]), score, ids, amb
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        B = len(Qb)
        if B > self.SMALL or ret_est: return Engine4.search(self, Qb, K, z, nprobe, k, ret_est)
        key = (B, K, z, nprobe, k)
        if key not in self.graphs:
            qs = Qb.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs, torch.empty(outs[0].shape, dtype=torch.long).pin_memory())
        g, qs, (packed, score, ids, amb), hostbuf = self.graphs[key]
        qs.copy_(Qb); g.replay(); hostbuf.copy_(packed); host = hostbuf                 # one synchronisation
        if not bool(host[:B].all()): return Engine4.search(self, Qb, K, z, nprobe, k)
        sel = torch.nonzero(host[B:] >= 0).flatten(); sc = score.clone()
        if len(sel):
            rows = rf.host16.index_select(0, host[B:][sel]).to(dev, non_blocking=True).float()
            selg = sel.to(dev); bi, bk = selg // ids.shape[1], selg % ids.shape[1]
            sc[bi, bk] = (rows * qs[bi]).sum(1)
        return ids.gather(1, sc.topk(k, 1).indices), amb.sum(1), ids

from concurrent.futures import ThreadPoolExecutor
class Engine7(Engine6):
    """v7: throughput mode runs consecutive batches on two CUDA streams from two host threads, so the host synchronisations and
    the exact-read gather of one batch overlap the kernels of the other."""
    STREAMS = 2
    def run(self, K, z, nprobe, k=10, B=500, nq=2000):
        if not hasattr(self, "streams"): self.streams = [torch.cuda.Stream() for _ in range(self.STREAMS)]
        def work(i, s):
            with torch.cuda.stream(self.streams[i % self.STREAMS]):
                a, b, c = self.search(rf.Qg[s:s + B], K, z, nprobe, k)
                return a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()
        with ThreadPoolExecutor(self.STREAMS) as ex:
            res = list(ex.map(lambda t: work(*t), enumerate(range(0, nq, B))))
        return np.vstack([r[0] for r in res]), np.concatenate([r[1] for r in res]), np.vstack([r[2] for r in res])

import stage3_zc, s1small
class Engine8(Engine6):
    """v8: v6 (graph small-batch path, guess-and-verify batch path) with stage 3 read by zero-copy from pinned host memory:
    no CPU gather, and the small-batch path has a single synchronisation (the exactness flag) at the end."""
    H = None
    def _score(self, Qb, QC, pos): return tiered_fused.score(self.code, Qb, QC, pos)
    def _s3(self, Qb, ids, est, sure, amb):
        if Engine8.H is None: Engine8.H = stage3_zc.pin(rf.host16)
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        return stage3_zc.stage3(Engine8.H, ids, amb, Qb, score)
    def _gpu(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]
        a = (Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off_t"], nprobe)
        if len(Qb) <= self.TINY: pos, _, ok = s1small.search_small(*a)
        else:
            if not hasattr(self, "MB"): o = np.diff(s1["off"]); self.MB = int((o.max() + 31) // 32 + 1)
            pos, _, ok = s1small.search_mask(*a, self.MB)
        est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    SMALL, TINY = 32, 8
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        B = len(Qb)
        if B > self.SMALL or ret_est:
            pos, QC = self.stage1(Qb, K, nprobe); est, sg = self._score(Qb, QC, pos)
            lo, hi = est - z * sg, est + z * sg
            sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
            ids = self.s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
            out = ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
            return out + (est, sg) if ret_est else out
        key = (B, K, z, nprobe, k)
        if key not in self.graphs:
            self._s3(Qb[:1], *[torch.zeros(1, 1, dtype=t, device=dev) for t in (torch.long, torch.float, torch.bool, torch.bool)]) if Engine8.H is None else None
            qs = Qb.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs)
        g, qs, (ok, I, nr, ids) = self.graphs[key]
        qs.copy_(Qb); g.replay()
        if not bool(ok.all()): return Engine4.search(self, Qb, K, z, nprobe, k)              # the only synchronisation
        return I.clone(), nr.clone(), ids.clone()

class Engine9(Engine8):
    """lean v9: only the stage-1 code (and routing) on the GPU; the tiered refinement code (planes, scales, sigmas) lives in pinned
    host memory and the fused stage-2 kernel reads it over PCIe (zero-copy), like the full vectors in stage 3."""
    def __init__(self, s1, code, **kw):
        hc = dict(code)
        hc["planes"] = [stage3_zc.pin(p.cpu()) for p in code["planes"]]
        hc["scale"] = stage3_zc.pin(code["scale"].cpu()); hc["sig"] = stage3_zc.pin(code["sig"].cpu())
        super().__init__(s1, hc, **kw)

class Engine10(Engine8):
    """v10: v8 with the refinement code packed as one 512-byte record per vector (on the GPU, or in pinned host memory if lean)."""
    def __init__(self, s1, code, lean=False, **kw):
        super().__init__(s1, code, **kw); self.pk = tiered_fused.pack(code, pin=lean)
        if not lean: self.pk["rec"] = self.pk["rec"].cuda()
        tiered_fused.prepare(code)
        if lean:
            for k_ in ("planes",): code[k_] = None
            code["scale"] = code["scale"].cpu(); code["sig"] = code["sig"].cpu(); torch.cuda.empty_cache()
    def _score(self, Qb, QC, pos): return tiered_fused.score_rec(self.code, self.pk, Qb, QC, pos)

import nvme_async
class Engine12(Engine10):
    """lean + NVMe: stage 1 on the GPU (48 B), refinement records in pinned host RAM (zero-copy), full vectors on NVMe read by the
    asynchronous C reader; in throughput mode the reads of one batch overlap the GPU stages of the next."""
    def __init__(self, s1, code, path=f"{FANAR}/data/oai1m_ssd_tier.bin", qd=128, **kw):
        super().__init__(s1, code, lean=True, **kw); self.tier = nvme_async.AsyncTier(path, rf.d, qd)
        self.pool = ThreadPoolExecutor(1)
    def _gpu_part(self, Qb, K, z, nprobe, k):
        pos, QC = self.stage1(Qb, K, nprobe); est, sg = self._score(Qb, QC, pos)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = self.s1["order"][pos]; bi, bk = torch.nonzero(amb, as_tuple=True)
        want = ids[bi, bk].cpu().numpy()
        buf = torch.empty((len(want), rf.d), dtype=torch.float16, pin_memory=True)
        fut = self.pool.submit(self.tier.read_into, want, buf)
        return Qb, ids, est, sure, amb, bi, bk, buf, fut
    def _finish(self, part, k):
        Qb, ids, est, sure, amb, bi, bk, buf, fut = part; fut.result()
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9))
        if len(bi): score[bi, bk] = (buf.to(dev, non_blocking=True).float() * Qb[bi]).sum(1)
        return ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        return self._finish(self._gpu_part(Qb, K, z, nprobe, k), k)
    def run(self, K, z, nprobe, k=10, B=500, nq=2000):
        I, r, C = [], [], []; prev = None
        for s in range(0, nq + B, B):
            cur = self._gpu_part(rf.Qg[s:s + B], K, z, nprobe, k) if s < nq else None
            if prev is not None:
                a, b, c = self._finish(prev, k); I.append(a.cpu().numpy()); r.append(b.cpu().numpy()); C.append(c.cpu().numpy())
            prev = cur
        return np.vstack(I), np.concatenate(r), np.vstack(C)
