"""Reviewer Q: ||q|| vs a plug-in estimate of ||q_perp|| in sigma. sigma' = sigma * sqrt(max(f_min, 1 - s_hat^2 / (||q||^2 ||o||^2))),
s_hat = estimated residual score (est - <q,c>), ||o|| = residual norm (2 B/vector extra). Coverage measured against exact scores."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf, Engine8
import stage3_zc
e3, _ = make(256); e = Engine8(e3.s1, e3.code); s1 = e.s1; dev = "cuda"
# residual norms in the stage-1 sorted order -> per original id
onorm = torch.cat([(rf.host16[j:j + 100000].to(dev).float() - s1["c"]).norm(dim=1) for j in range(0, rf.N, 100000)]).half().float()
K, nprobe = 400, 768; out = []
def run(mode, z, fmin=0.0, nq=2000, B=500, cover=False):
    I_all, reads, unc, Zs = [], [], [], []
    for s in range(0, nq, B):
        Qb = rf.Qg[s:s + B]; pos, QC = e.stage1(Qb, K, nprobe); est, sg = e._score(Qb, QC, pos); ids = s1["order"][pos]
        if mode == "plugin":
            qn = Qb.norm(dim=1, keepdim=True); r = (est - QC[:, None]) / (qn * onorm[ids])
            sg = sg * torch.sqrt(torch.clamp(1 - r * r, min=fmin))
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure
        score = e._s3(Qb, ids, est, sure, amb); I_all.append(ids.gather(1, score.topk(10, 1).indices).cpu().numpy()); reads.append(amb.sum(1).cpu().numpy())
        if cover:
            ex = torch.stack([(stage3_zc.pin and rf.host16.index_select(0, ids[i].cpu()).to(dev).float() @ Qb[i]) for i in range(len(Qb))])
            unc.append(((est - ex).abs() > z * sg).any(1).cpu().numpy()); Zs.append(((est - ex) / sg).flatten().cpu())
    I = np.vstack(I_all); r = np.concatenate(reads)
    x = {"sigma": mode, "fmin": fmin, "z": z, "recall@10": rf.recall(I), "reads_mean": round(float(r.mean()), 2), "reads_p99": float(np.percentile(r, 99))}
    if cover: x.update(uncovered_query_share=round(float(np.concatenate(unc).mean()), 4), std_Z=round(float(torch.cat(Zs).std()), 3))
    return x
for z in (2.0, 4.9):
    for mode, fmin in (("||q|| (engine)", 0.0), ("plugin", 0.0), ("plugin", 0.25)):
        x = run("plugin" if mode == "plugin" else "q", z, fmin, cover=True)
        x["sigma"] = mode; print(json.dumps(x), flush=True); out.append(x)
json.dump(out, open(f"{FANAR}/results/rev3_qperp.json", "w"), indent=1)
