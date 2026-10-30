"""Engine v8 on other embedding models (100k scale): OpenAI-3-large 1536-d and arXiv InstructorXL 768-d. Refinement 2.5 bits/dim, 512 cells."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
DS = sys.argv[1]
from rf_engine import rf
rf.X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy"); rf.host16 = torch.from_numpy(np.ascontiguousarray(rf.X)); rf.Qn = np.load(f"{FANAR}/data/{DS}_q.npy")
rf.Qg = torch.from_numpy(rf.Qn).cuda(); rf.GT10 = np.load(f"{FANAR}/data/{DS}_gt.npy"); rf.GT100 = np.load(f"{FANAR}/data/{DS}_gt100.npy")
rf.N, rf.d = rf.X.shape; rf.NQ = 2000; rf.ROOT = f"{FANAR}/cache/fanar_tmp_" + DS
os.makedirs(rf.ROOT + "/data/rev_cache", exist_ok=True)
import viterbi_fused; rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
from engine3 import make, Engine8
out = []
for R in (256, 384):
    rf.R = R
    t0 = time.perf_counter(); e3, nb = make(R, nlist=512, B=int(2.5 * rf.d)); bt = time.perf_counter() - t0
    e = Engine8(e3.s1, e3.code); Engine8.H = None
    for nprobe in (32, 48, 64, 96, 128, 192, 256):
        for K in (100, 200, 400):
            I, r, C, m, s = e.bench(K, 2.0, nprobe)
            x = {"ds": DS, "R": R, "d": rf.d, "N": rf.N, "gpu_bytes": nb, "tiers": len(e3.code["TS"]), "build_s": round(bt, 1), "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C),
                 "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2)}
            print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(f"{FANAR}/results/v13_{DS}.json", "w"), indent=1)
    del e, e3; torch.cuda.empty_cache()
# fp16 brute force on the GPU (reference)
G = rf.host16.cuda(); f = lambda: [(rf.Qg[s:s + 500].half() @ G.T).topk(10, 1) for s in range(0, 2000, 500)]
f(); torch.cuda.synchronize(); qs = []
for _ in range(3):
    t0 = time.perf_counter(); res = f(); torch.cuda.synchronize(); qs.append(2000 / (time.perf_counter() - t0))
I = np.vstack([r.indices.cpu().numpy() for r in res])
x = {"ds": DS, "engine": "fp16 brute force (GPU)", "gpu_bytes": rf.d * 2, "recall@10": rf.recall(I), "QPS": round(float(np.mean(qs)), 1)}
print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(f"{FANAR}/results/v13_{DS}.json", "w"), indent=1)
