"""latency (paper protocol) of Engine8 and EngineC on one dataset; also recall at each config with the batch path."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
DS = sys.argv[1]; os.environ.setdefault("GRID", "1:1")
exec(open(f"{FANAR}/src/v18_run.py", encoding="utf-8").read().split("for R in")[0])
rf.R = 256; e3, nb = make(256, nlist=NL, B=int(BITS * rf.d)); s1c = cellify(e3.s1)
res = []
for name, eng in [(n, x) for n, x in (("v8", Engine8(e3.s1, e3.code)), ("cell", EngineC(s1c, e3.code)), ("static", EngineS(s1c, e3.code))) if n in os.environ.get("LENG", "v8,cell,static").split(",")]:
    Engine8.H = None
    for nprobe, K in [tuple(map(int, g.split(":"))) for g in os.environ.get("LGRID", "384:200,768:400,512:200").split(",")]:
        for B in (1, 16, 100, 500):
            nbt = min(200, 2000 // B); ts = []; Is = []
            for i in range(nbt + 3):
                Qb = rf.Qg[(i * B) % 2000:(i * B) % 2000 + B]; torch.cuda.synchronize(); t0 = time.perf_counter()
                I = eng.search(Qb, K, 2.0, nprobe, 10)[0].cpu(); ts.append(time.perf_counter() - t0)
                if i >= 3: Is.append((i * B) % 2000)
            ts = np.array(ts[3:]) * 1000
            r = {"ds": DS, "engine": name, "nprobe": nprobe, "K": K, "batch": B, "median_ms": round(float(np.median(ts)), 2), "p99_ms": round(float(np.percentile(ts, 99)), 2)}
            print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(f"{FANAR}/results/v18_lat_{DS}_{os.environ.get('LENG', 'all')}.json", "w"), indent=1)
