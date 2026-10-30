"""Experiment 73: progressive stage 2. Score the leading tiers (384 principal directions, 216 of 504 bytes) for all K candidates, keep
the best K2 by that partial estimate (plus the stage-1-free part <q,c>), and run the full tiered score on those only."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, torch
from cache import *
import tiered_fused
e, s1, code, mem = load(); dev = "cuda"
for t in range(len(code["TS"])): pass
tiered_fused.prepare(code)
def sub(nt): return dict(code, TS=code["TS"][:nt], planes=code["planes"][:nt], cbf=code["cbf"][:nt])
def search(Qb, K, nprobe, nt, K2, z=2.0, k=10):
    pos, QC = e.stage1(Qb, K, nprobe)
    if K2 < K:
        pe, _ = tiered_fused.score(sub(nt), Qb, QC, pos); keep = pe.topk(K2, 1).indices; pos = pos.gather(1, keep)
    est, sg = e._score(Qb, QC, pos); lo, hi = est - z * sg, est + z * sg
    sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
    ids = s1["order"][pos]; score = e._s3(Qb, ids, est, sure, amb)
    return ids.gather(1, score.topk(k, 1).indices), ids, pos, QC
Qb = rf.Qg[:500]
pos, QC = e.stage1(Qb, 400, 384)
print("full stage 2, K=400: %.2f ms" % T(lambda: e._score(Qb, QC, pos)))
for nt in (1, 2, 3):
    print(f"partial tiers 0..{nt - 1}, K=400: %.2f ms" % T(lambda: tiered_fused.score(sub(nt), Qb, QC, pos)), "| full on 100: %.2f ms" % T(lambda: e._score(Qb, QC, pos[:, :100].contiguous())))
for nt in (1, 2, 3):
    for K2 in (50, 100, 200):
        I = torch.cat([search(rf.Qg[s:s + 500], 400, 384, nt, K2)[0] for s in range(0, 2000, 500)]).cpu().numpy()
        def full_stage23():
            pe, _ = tiered_fused.score(sub(nt), Qb, QC, pos); p2 = pos.gather(1, pe.topk(K2, 1).indices); return e._score(Qb, QC, p2)
        x = dict(tag="progressive", nprobe=384, K=400, tiers=nt, K2=K2, recall=rf.recall(I, 10), stage2_ms=round(T(full_stage23), 2))
        print(json.dumps(x), flush=True)
        with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
