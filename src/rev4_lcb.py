"""Certified plug-in sigma per tier. Event A: |s_t_hat - s_t| <= z1 * ||q_t|| sig_t for all tiers (prob >= 1 - 6K*2exp(-z1^2/2)).
On A, ||q_t_perp||^2 <= ||q_t||^2 - max(0, |s_t_hat| - z1 ||q_t|| sig_t)^2 / ||o_t||^2 =: u_t^2, so sigma' = sqrt(sum u_t^2 sig_t^2) >= sigma_true,
and P(|err| > z2 sigma') <= 2 exp(-z2^2/2) + P(not A). Per query: K (12 exp(-z1^2/2) + 2 exp(-z2^2/2))."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf, Engine8
import tiered
from fused_scan import rescore_planes
e3, _ = make(256); e = Engine8(e3.s1, e3.code); s1, code = e.s1, e.code; dev = "cuda"; TS = code["TS"]
K, nprobe = 400, 768
def per_tier(Qb, QC, pos, ids):
    Zq = Qb @ code["V"]; zero = torch.zeros(len(Qb), device=dev); ests, qn, sg, on = [], [], [], []
    O = torch.stack([(rf.host16.index_select(0, ids[i].cpu()).to(dev).float() - s1["c"]) @ code["V"] for i in range(len(Qb))])   # (B, K, d) residual in PCA basis
    for t, ((s, e_, bb), P) in enumerate(zip(TS, code["rots"])):
        qt = (Zq[:, s:e_] @ P).contiguous()
        ests.append(rescore_planes(qt, zero, pos, code["planes"][t], tiered.CB[bb][0], code["scale"][t]))
        qn.append(qt.norm(dim=1, keepdim=True).expand_as(pos)); sg.append(code["sig"][t][pos].float()); on.append(O[:, :, s:e_].norm(dim=2).half().float())
    return [torch.stack(x) for x in (ests, qn, sg, on)]                                          # (T, B, K)
def bound(z1, z2): return K * (12 * np.exp(-z1 * z1 / 2) + 2 * np.exp(-z2 * z2 / 2))
res = []
cfgs = [("engine ||q||, z=4.9", None, 4.9), ("engine ||q||, z=2", None, 2.0)]
for z1, z2 in ((5.37, 5.04), (5.6, 4.95), (5.2, 5.2)): cfgs.append((f"LCB plug-in z1={z1} z2={z2}", z1, z2))
cfgs.append(("LCB plug-in heuristic z1=0 z2=2", 0.0, 2.0))
acc = {c[0]: dict(reads=[], unc=[], I=[]) for c in cfgs}
for s0 in range(0, 2000, 250):
    Qb = rf.Qg[s0:s0 + 250]; pos, QC = e.stage1(Qb, K, nprobe); ids = s1["order"][pos]
    est_t, qn, sg, on = per_tier(Qb, QC, pos, ids); est = QC[:, None] + est_t.sum(0)
    ex = torch.stack([(rf.host16.index_select(0, ids[i].cpu()).to(dev).float() @ Qb[i]) for i in range(len(Qb))])
    for name, z1, z in cfgs:
        if z1 is None: sigma = torch.sqrt(((qn * sg) ** 2).sum(0))
        else:
            lcb = torch.clamp(est_t.abs() - z1 * qn * sg, min=0) / on.clamp(min=1e-6)
            u2 = torch.clamp(qn ** 2 - lcb ** 2, min=0); sigma = torch.sqrt((u2 * sg ** 2).sum(0))
        lo, hi = est - z * sigma, est + z * sigma
        sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure
        score = torch.where(sure, est + 10, torch.full_like(est, -1e9)); score = torch.where(amb, ex, score)
        a = acc[name]; a["reads"].append(amb.sum(1).cpu()); a["unc"].append(((est - ex).abs() > z * sigma).any(1).cpu()); a["I"].append(ids.gather(1, score.topk(10, 1).indices).cpu())
for name, z1, z in cfgs:
    a = acc[name]; r = torch.cat(a["reads"]).float()
    x = {"sigma": name, "per_query_bound": (round(bound(z1, z), 4) if z1 else (round(K * 2 * np.exp(-z * z / 2), 4) if z1 is None else None)),
         "recall@10": rf.recall(torch.cat(a["I"]).numpy()), "reads_mean": round(float(r.mean()), 2), "reads_p99": float(torch.quantile(r, 0.99)), "uncovered_query_share": round(float(torch.cat(a["unc"]).float().mean()), 4)}
    print(json.dumps(x), flush=True); res.append(x)
json.dump(res, open(f"{FANAR}/results/rev4_lcb.json", "w"), indent=1)
