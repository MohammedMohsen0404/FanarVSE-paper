"""single-query and small-batch latency: final engine (EngineS, paper protocol) vs the lean path (EngineT.search_host); recall checked."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, time, json, numpy as np, torch
sys.argv = [sys.argv[0], "sweep"]
src = open(f"{FANAR}/src/v19_final.py", encoding="utf-8").read().split("grid = lambda")[0]; exec(src)
from engine_cell import EngineT
e, mem = build(cls=EngineT); res = []; e.PROJ = bool(int(os.environ.get("PROJ", "0")))
for nprobe, K in ((128, 100), (192, 200), (256, 200), (384, 200), (384, 400), (512, 400), (768, 400)):
    for B in (1,):
        out = {}
        for name, f in (("graph+cpu", lambda Qb: e.search(Qb, K, 2.0, nprobe, 10)[0].cpu().numpy()), ("lean", lambda Qb: e.search_host(Qb, K, 2.0, nprobe, 10))):
            I = np.vstack([f(rf.Qg[s:s + B]) for s in range(0, 2000 // B * B, B)]); rec = rf.recall(np.vstack([I, np.zeros((2000 - len(I), 10), dtype=I.dtype)]))
            ts = []
            for i in range(min(300, 2000 // B) + 5):
                Qb = rf.Qg[(i * B) % 2000:(i * B) % 2000 + B]; torch.cuda.synchronize(); t0 = time.perf_counter(); f(Qb); ts.append(time.perf_counter() - t0)
            ts = np.array(ts[5:]) * 1000; out[name] = (round(rec, 4), round(float(np.median(ts)), 3), round(float(np.percentile(ts, 99)), 3))
        r = {"nprobe": nprobe, "K": K, "batch": B, **{f"{n}_recall": v[0] for n, v in out.items()}, **{f"{n}_median_ms": v[1] for n, v in out.items()}, **{f"{n}_p99_ms": v[2] for n, v in out.items()}, "fallbacks_cum": getattr(e, "nbad", 0)}
        print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(f"{FANAR}/results/v21_lat.json", "w"), indent=1)
