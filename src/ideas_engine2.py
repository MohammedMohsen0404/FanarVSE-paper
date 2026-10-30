"""Engine v2 for the ideas round: stage 1 (trellis, R=256) with a configurable block size, tiered stage-2 refinement, stage 3 unchanged."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, torch
sys.path.insert(0, f"{FANAR}/src")
from rf_engine import rf
import tiered
from trellis_scan import trellis_search_ivf
dev = "cuda"

class Engine2(rf.Engine):
    def __init__(self, s1, code, BN=32, BQ=32):
        self.s1, self.code, self.BN, self.BQ = s1, code, BN, BQ
    def stage1(self, Qb, K, nprobe):
        s1 = self.s1; QC = Qb @ s1["c"]
        return trellis_search_ivf(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off"], nprobe, BN=self.BN, BQ=self.BQ), QC
    def gpu_stage(self, Qb, K, z, nprobe, k=10, lean=False):
        pos, QC = self.stage1(Qb, K, nprobe)
        est, sg = tiered.score(self.code, Qb, QC, pos); lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        return self.s1["order"][pos].cpu().numpy(), est.cpu().numpy(), sure.cpu().numpy(), amb.cpu().numpy()

def load(B=3840, seed=0):
    s1 = rf.build_stage1(seed); lam, V = tiered.pca_basis(rf.host16, s1["c"]); TS = tiered.allocate(lam, B)
    return s1, tiered.build(rf.host16, s1["order"], s1["c"], V, TS, seed)

def reroute(s1, code, nlist, seed=0, iters=15):
    g = torch.Generator().manual_seed(seed)
    samp = rf.host16[torch.randperm(rf.N, generator=g)[:200000]].to(dev).float(); cen = samp[torch.randperm(len(samp), generator=g)[:nlist].to(dev)].clone()
    for _ in range(iters):
        a = torch.cat([(samp[j:j + 50000] @ cen.T).argmax(1) for j in range(0, len(samp), 50000)])
        cen = torch.nn.functional.normalize(torch.zeros_like(cen).index_add_(0, a, samp) + 1e-9, dim=1)
    del samp
    A = torch.cat([(rf.host16[j:j + 100000].to(dev).float() @ cen.T).argmax(1) for j in range(0, rf.N, 100000)]); order = torch.argsort(A)
    off = torch.searchsorted(A[order], torch.arange(nlist + 1, device=dev)).cpu().numpy()
    inv_old = torch.empty_like(s1["order"]); inv_old[s1["order"]] = torch.arange(rf.N, device=dev)    # original id -> old sorted position
    perm = inv_old[order]                                                                           # new sorted position -> old sorted position
    s1n = dict(s1); s1n.update(cen=cen, off=off, order=order, words=s1["words"][perm].contiguous(), scale1=s1["scale1"][perm].contiguous())
    cn = dict(code); cn.update(planes=[p[:, perm].contiguous() for p in code["planes"]], scale=code["scale"][:, perm].contiguous(), sig=code["sig"][:, perm].contiguous())
    return s1n, cn

