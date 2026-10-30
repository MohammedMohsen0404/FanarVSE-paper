"""Lean v10 (48/64 B on the GPU): R=256 and R=384 grids, and small-batch latency."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf, Engine10
R = int(os.environ.get("FANAR_R", 256)); out = []; OUT = f"{FANAR}/results/v10_lean_R{R}.json"
e3, _ = make(R); s1b = e3.s1["words"].shape[1] * 4 + 2 + 10; el = Engine10(e3.s1, e3.code, lean=True)
grid = ((128, 100), (192, 100), (256, 200), (384, 200), (384, 400), (512, 400), (768, 400), (768, 800), (1024, 800), (2048, 800)) if R == 256 else ((1536, 400), (2048, 800), (3072, 1200))
for nprobe, K in grid:
    I, r, C, m, s = el.bench(K, 2.0, nprobe)
    x = {"R": R, "gpu_bytes": s1b, "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(r)}
    print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(OUT, "w"), indent=1)
if R == 256:
    for nprobe, K in ((384, 200), (768, 400)):
        for B in (1, 16, 100):
            ts = []
            for i in range(min(200, 2000 // B) + 3):
                Qb = rf.Qg[(i * B) % 2000:(i * B) % 2000 + B]; torch.cuda.synchronize(); t0 = time.perf_counter()
                el.search(Qb, K, 2.0, nprobe, 10)[0].cpu(); ts.append(time.perf_counter() - t0)
            x = {"latency": True, "nprobe": nprobe, "K": K, "batch": B, "median_ms": round(float(np.median(ts[3:]) * 1000), 2)}
            print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(OUT, "w"), indent=1)
