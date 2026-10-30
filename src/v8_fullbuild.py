"""Full index build from scratch with the fused Viterbi encoder (no cache): stage 1, tiered refinement, 4,096 cells; then recall/QPS."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, time, json, torch
sys.path.insert(0, f"{FANAR}/src")
from ideas_engine2 import rf, reroute
import tiered, viterbi_fused
from engine3 import Engine8
rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
rf.ROOT = sys.argv[1]
out = {}
for R in (256, 384):
    rf.R = R
    torch.cuda.synchronize(); t0 = time.perf_counter()
    s1 = rf.build_stage1(0); torch.cuda.synchronize(); t1 = time.perf_counter()
    lam, V = tiered.pca_basis(rf.host16, s1["c"]); code = tiered.build(rf.host16, s1["order"], s1["c"], V, tiered.allocate(lam, 3840), 0); torch.cuda.synchronize(); t2 = time.perf_counter()
    s1n, cn = reroute(s1, code, 4096); s1n["off_t"] = torch.as_tensor(s1n["off"], device="cuda"); torch.cuda.synchronize(); t3 = time.perf_counter()
    e = Engine8(s1n, cn); I, r, C, m, s = e.bench(400, 2.0, 768 if R == 256 else 1536)
    out[R] = {"stage1_s": round(t1 - t0, 1), "tiered_s": round(t2 - t1, 1), "cells_s": round(t3 - t2, 1), "total_s": round(t3 - t0, 1), "recall@10": rf.recall(I), "QPS": round(m)}
    print(R, json.dumps(out[R]), flush=True)
    import os; os.remove(f"{rf.ROOT}/data/rev_cache/s1_seed0" + (f"_R{R}" if R != 256 else "") + ".pt")
json.dump(out, open(f"{FANAR}/results/v8_fullbuild.json", "w"), indent=1)
