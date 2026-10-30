"""k=100 with engine v3 at batch 500 (paper protocol)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf
out = []
eng, nbytes = make(256)
for nprobe, K in ((768, 1000), (1024, 1500), (1536, 1500), (1536, 2000), (2048, 2000), (3072, 3000)):
    I, reads, C, m, s = eng.bench(K, 2.0, nprobe, k=100, B=500)
    r = {"k": 100, "nprobe": nprobe, "K": K, "batch": 500, "gpu_bytes": nbytes, "recall@100": rf.recall(I, 100), "cand_ceiling": rf.cand_recall(C, 100), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(reads)}
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(f"{FANAR}/results/v3_k100.json", "w"), indent=1)
print("DONE")
