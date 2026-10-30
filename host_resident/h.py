"""Harness for the ideas campaign: final engine (EngineS on compact(cellify)) with knobs. env: DS, R, NL, BATCH, GRID (n:K,...), TAG, BITS.
Appends rows to results/host_resident_results.jsonl."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
DS = os.environ.get("DS", "oai1m")
from rf_engine import rf
if DS != "oai1m":
    rf.X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy"); rf.host16 = torch.from_numpy(rf.X); rf.Qn = np.load(f"{FANAR}/data/{DS}_q.npy")
    rf.Qg = torch.from_numpy(rf.Qn).cuda(); rf.GT10 = np.load(f"{FANAR}/data/{DS}_gt.npy"); rf.GT100 = np.load(f"{FANAR}/data/{DS}_gt100.npy")
    rf.N, rf.d = rf.X.shape; rf.NQ = 2000
ROOT = os.environ.get("ROOTC", f"{FANAR}/cache/fanar_tmp_" + DS)
if DS != "oai1m" or os.environ.get("ROOTC"): rf.ROOT = ROOT; os.makedirs(rf.ROOT + "/data/rev_cache", exist_ok=True)
import viterbi_fused; rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
from engine3 import make
from engine_cell import EngineS, cellify, compact
R = int(os.environ.get("R", 256)); NL = int(os.environ.get("NL", 4096)); BATCH = int(os.environ.get("BATCH", 500)); BITS = float(os.environ.get("BITS", 2.5))
TAG = os.environ.get("TAG", "")

def build(cls=EngineS, **kw):
    rf.R = R; t0 = time.perf_counter(); e3, _ = make(R, nlist=NL, B=int(BITS * rf.d)); s1 = compact(cellify(e3.s1)); bt = time.perf_counter() - t0
    N = len(s1["order"]); per = (s1["words"].numel() * 4 + s1["scale1"].numel() * 2 + s1["order"].numel() * 4) / N
    tables = s1["means16"].numel() * 2 + s1["W1"].numel() * 4 + s1["inv_mnorm"].numel() * 4
    mem = {"stage1_B": round(per, 1), "refine_B": float(e3.code["bytes"]), "tables_B": round(tables / N, 1), "gpu_B": round(per + float(e3.code["bytes"]) + tables / N, 1), "lean_B": round(per + tables / N, 1)}
    return cls(s1, e3.code, **kw), mem, e3

def emit(x):
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/host_resident_results.jsonl", "a") as f: f.write(json.dumps(x) + "\n")

def bench(e, mem, grid=None, k=10, **extra):
    grid = grid or [tuple(map(int, g.split(":"))) for g in os.environ.get("GRID", "384:400").split(",")]
    for nprobe, K in grid:
        I, r, C, m, s = e.bench(K, 2.0, nprobe, k=k, B=BATCH)
        emit({"tag": TAG, "ds": DS, "R": R, "NL": NL, "batch": BATCH, "nprobe": nprobe, "K": K, "recall": rf.recall(I, k), "ceiling": rf.cand_recall(C, k),
              "QPS": round(m), "QPS_std": round(s), "reads": round(float(r.mean()), 2), **mem, **extra})

if __name__ == "__main__":
    e, mem, _ = build(); bench(e, mem)
