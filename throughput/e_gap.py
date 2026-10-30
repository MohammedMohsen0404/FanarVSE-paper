"""Experiment 112: how far below the query's 10th-best stage-1 (int8) estimate can a true top-10 neighbour fall? For each margin delta
(in units of |q| or absolute), the fraction of true neighbours (within the candidate set) kept and the mean number of candidates kept."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch
exec(open(FANAR + "/throughput/e_i8prof2.py").read().split('t = {}')[0])
E.RHO, E.MSZ, E.BQ, E.NW, E.NROUTE = 0.6, 3072, 16, 2, True
Q = rf.Qg; GT = torch.from_numpy(rf.GT10[:, :10].astype(np.int64)).to(dev)
E.meansN = (s1["means16"].float() * s1["inv_mnorm"][:, None]).half().contiguous(); E.mnorm = (1.0 / s1["inv_mnorm"]).float().contiguous()
P_, E_ = [], []
for s in range(0, 2000, 500):
    Qb = Q[s:s + 500]; QM = Qb.half() @ E.meansN.T; probe = QM.topk(384, 1).indices
    pos, ok = E.stage1_i8(Qb, QM, probe, 200)
    # stage-1 estimate of every candidate (recomputed exactly as the kernel does)
    z = Qb @ Vr; qs = z.abs().amax(1) / 127; q8 = torch.round(z / qs[:, None]).clamp(-127, 127)
    off = s1["off_t"].long(); cell = torch.searchsorted(off, pos.flatten(), right=True).reshape(pos.shape) - 1
    est = (Z[pos].float() * q8[:, None, :]).sum(-1) * qs[:, None] * zs[pos] + (Qb.half() @ s1["means16"].T).float().gather(1, cell)
    P_.append(pos); E_.append(est)
pos = torch.cat(P_); est = torch.cat(E_)
ids = s1["order"][pos]; isnn = (ids[:, :, None] == GT[:, None, :]).any(-1)                 # candidate is a true top-10 neighbour
e10 = est.sort(1, descending=True).values[:, 9:10]; gap = (e10 - est)                         # >0: below the 10th-best estimate
g = gap[isnn]; print("true neighbours in candidate set:", int(isnn.sum()), "gap quantiles 0.99/0.999/0.9999/max:",
                     [round(float(g.quantile(q)), 4) for q in (0.99, 0.999, 0.9999)], round(float(g.max()), 4))
for d in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
    keep = gap <= d
    print(f"delta {d}: neighbours kept {float((keep & isnn).sum() / isnn.sum()):.5f}, candidates kept mean {keep.sum(1).float().mean():.1f} p99 {keep.sum(1).float().quantile(0.99):.0f}")
