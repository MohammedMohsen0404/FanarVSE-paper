"""Experiment 81: CUDA int8-WMMA routing-free scan: correctness vs the Triton per-element version, time at B=500/2000/4000, candidate recall."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch, json
from cache import *
import cbrute, brute
e, s1, code, mem = load(); dev = "cuda"
P = torch.load(f"{FANAR}/data/oai1m_pca256.pt"); mu, V = P["mu"].to(dev), P["V"].to(dev)
cb = cbrute.CBrute(rf.host16, mu, V, R=128)
GT = rf.GT10[:, :10]
def recall(ids):
    ids = ids.cpu().numpy(); return float(np.mean([len(set(ids[i]) & set(GT[i])) / 10 for i in range(len(ids))]))
Qb = rf.Qg[:500]; p, est, ok, cnt = cb.search(Qb, 400)
print("ok", int(ok.sum()), "emitted median", int(cnt.float().median()))
# reference: exact top-400 of the same int8 scores via torch
q8, qs, z = cb.qcode(Qb); ref = ((q8.float() @ cb.Z.float().T) * qs[:, None] * cb.zs.float()[None]).topk(400, 1)
print("max |est - ref| (sorted):", float((est.sort(1).values - ref.values.sort(1).values).abs().max()))
for B in (500, 1000, 2000, 4000):
    Qb = torch.cat([rf.Qg] * (B // 2000 + 1))[:B]
    t = T(lambda: cb.search(Qb, 400), 5); print(f"B={B}: stage-1 scan {t:.2f} ms -> {B / t * 1000:.0f} q/s", flush=True)
P_, F = [], 0
for s in range(0, 2000, 500):
    p, est, ok, cnt = cb.search(rf.Qg[s:s + 500], 400); P_.append(p); F += int((~ok).sum())
print("candidate recall K=400:", recall(torch.cat(P_)), "flagged", F)
