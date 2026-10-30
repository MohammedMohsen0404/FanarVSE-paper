"""Engine v8 on the large / image sets prepared by v16_prep.py (refinement 2.5 bits/dim; 4,096 cells from 1M up, 1,024 below).
usage: v16_run.py {arxivL|siglip|dino}"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
DS = sys.argv[1]
from rf_engine import rf
rf.X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy"); rf.host16 = torch.from_numpy(rf.X); rf.Qn = np.load(f"{FANAR}/data/{DS}_q.npy")
rf.Qg = torch.from_numpy(rf.Qn).cuda(); rf.GT10 = np.load(f"{FANAR}/data/{DS}_gt.npy"); rf.GT100 = np.load(f"{FANAR}/data/{DS}_gt100.npy")
rf.N, rf.d = rf.X.shape; rf.NQ = 2000; rf.ROOT = f"{FANAR}/cache/fanar_tmp_" + DS
os.makedirs(rf.ROOT + "/data/rev_cache", exist_ok=True)
import viterbi_fused; rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
from engine3 import make, Engine8
NL = 4096 if rf.N >= 1_000_000 else 1024; OUT = f"{FANAR}/results/v16_{DS}.json"; out = []
def emit(x):
    print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(OUT, "w"), indent=1)
# fp16 brute force on the GPU (reference)
if not os.environ.get("SKIPBF"):
    G = rf.host16.cuda(); f = lambda: [(rf.Qg[s:s + 500].half() @ G.T).topk(10, 1) for s in range(0, 2000, 500)]
    f(); torch.cuda.synchronize(); qs = []
    for _ in range(3):
        t0 = time.perf_counter(); res = f(); torch.cuda.synchronize(); qs.append(2000 / (time.perf_counter() - t0))
    emit({"ds": DS, "engine": "fp16 brute force (GPU)", "N": rf.N, "d": rf.d, "gpu_bytes": rf.d * 2, "recall@10": rf.recall(np.vstack([r.indices.cpu().numpy() for r in res])), "QPS": round(float(np.mean(qs)), 1)})
    del G, res; torch.cuda.empty_cache()
RS = [int(x) for x in os.environ.get("RS", "256,384").split(",")]
GRID = os.environ.get("GRID")
if os.environ.get("SKIPBF"): out = json.load(open(OUT)) if os.path.exists(OUT) else []
for R in RS:
    rf.R = R
    t0 = time.perf_counter(); e3, nb = make(R, nlist=NL, B=int(float(os.environ.get("BITS", "2.5")) * rf.d)); bt = time.perf_counter() - t0
    e = Engine8(e3.s1, e3.code); Engine8.H = None
    grid = [tuple(map(int, g.split(":"))) for g in GRID.split(",")] if GRID else [(n, K) for n in ((64, 128, 256, 384, 512, 768, 1024) if NL == 4096 else (32, 64, 96, 128, 192, 256, 384)) for K in (100, 200, 400, 800)]
    for nprobe, K in grid:
        if True:
            I, r, C, m, s = e.bench(K, 2.0, nprobe)
            emit({"ds": DS, "R": R, "d": rf.d, "N": rf.N, "nlist": NL, "gpu_bytes": nb, "tiers": len(e3.code["TS"]), "build_s": round(bt, 1), "nprobe": nprobe, "K": K, "bits_per_dim": float(os.environ.get("BITS", "2.5")),
                  "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2)})
    del e, e3; Engine8.H = None; torch.cuda.empty_cache()
print("DONE", flush=True)
