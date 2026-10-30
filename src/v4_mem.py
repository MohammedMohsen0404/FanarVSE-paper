"""Memory sweep with engine v4: refinement bits per vector B (stage 1 R=256 fixed)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf, Engine4
out = []
for B in (1536, 2048, 2560, 3072, 3840, 4608):
    e3, nb = make(256, B=B); e = Engine4(e3.s1, e3.code); T = e3.code["TS"]
    if len(T) > 6: print("skip", B, len(T)); continue
    for nprobe, K in ((512, 400), (768, 400), (1024, 800)):
        I, r, C, m, s = e.bench(K, 2.0, nprobe)
        x = {"refine_bits": B, "gpu_bytes": nb, "tiers": [[e_ - s_, b] for s_, e_, b in T], "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C),
             "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(r)}
        print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(f"{FANAR}/results/v4_mem.json", "w"), indent=1)
    del e, e3; torch.cuda.empty_cache()
