"""Engine8 vs EngineC (cell-residual stage 1) on one dataset. usage: v18_run.py DS  (env RS, GRID)"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
DS = sys.argv[1]
from rf_engine import rf
rf.X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy"); rf.host16 = torch.from_numpy(rf.X); rf.Qn = np.load(f"{FANAR}/data/{DS}_q.npy")
rf.Qg = torch.from_numpy(rf.Qn).cuda(); rf.GT10 = np.load(f"{FANAR}/data/{DS}_gt.npy"); rf.GT100 = np.load(f"{FANAR}/data/{DS}_gt100.npy")
rf.N, rf.d = rf.X.shape; rf.NQ = 2000
rf.ROOT = f"{FANAR}" if DS == "oai1m" else os.environ.get("CACHE", f"{FANAR}/cache/fanar_tmp_" + DS)
os.makedirs(rf.ROOT + "/data/rev_cache", exist_ok=True)
import viterbi_fused; rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
from engine3 import make, Engine8
from engine_cell import EngineC, EngineCT, EngineCX, EngineS, EngineS2, cellify, cellify_code
NL = int(os.environ.get("NL", 4096 if rf.N >= 900_000 else (1024 if rf.N > 150000 else 512)))
BITS = float(os.environ.get("BITS", "2.5")); OUT = f"{FANAR}/results/v18b_{DS}.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else []
def emit(x):
    print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(OUT, "w"), indent=1)
for R in [int(x) for x in os.environ.get("RS", "256").split(",")]:
    rf.R = R; e3, nb = make(R, nlist=NL, B=int(BITS * rf.d)); s1c = cellify(e3.s1)
    engines = {"v8": Engine8(e3.s1, e3.code), "cell": EngineC(s1c, e3.code), "cellT": EngineCT(s1c, e3.code), "cellX": EngineCX(s1c, e3.code), "static": EngineS(s1c, e3.code)}
    if "static2" in os.environ.get("ENGINES", ""): c2 = cellify_code(s1c, e3.code); engines["static2"] = EngineS2(s1c, c2)
    for name in os.environ.get("ENGINES", "v8,cell").split(","):
        e = engines[name]; Engine8.H = None
        for g in os.environ["GRID"].split(","):
            nprobe, K = map(int, g.split(":")); I, r, C, m, s = e.bench(K, 2.0, nprobe)
            emit({"ds": DS, "engine": name, "R": R, "nlist": NL, "bits_per_dim": BITS, "gpu_bytes": nb + (0 if name == "v8" else 0), "nprobe": nprobe, "K": K,
                  "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2), "cellify_s": round(s1c["cellify_s"], 1)})
    del engines, e3, s1c; Engine8.H = None; torch.cuda.empty_cache()
