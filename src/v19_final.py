"""Measurements of the final engine (cell-residual stage 1, synchronisation-free batch path, compact tables) for the paper, with the
protocols of ideas_final.py. usage: v19_final.py MODE [DS]   MODE in sweep|data|lat|k100|z|seeds|lean|nvme"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
MODE = sys.argv[1]; DS = sys.argv[2] if len(sys.argv) > 2 else "oai1m"
from rf_engine import rf
if DS != "oai1m":
    rf.X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy"); rf.host16 = torch.from_numpy(rf.X); rf.Qn = np.load(f"{FANAR}/data/{DS}_q.npy")
    rf.Qg = torch.from_numpy(rf.Qn).cuda(); rf.GT10 = np.load(f"{FANAR}/data/{DS}_gt.npy"); rf.GT100 = np.load(f"{FANAR}/data/{DS}_gt100.npy")
    rf.N, rf.d = rf.X.shape; rf.NQ = 2000; rf.ROOT = f"{FANAR}/cache/fanar_tmp_" + DS
    os.makedirs(rf.ROOT + "/data/rev_cache", exist_ok=True)
import viterbi_fused; rf.viterbi_paths = viterbi_fused.viterbi_paths_fused
from engine3 import make, Engine8, Engine10, Engine12
from engine_cell import EngineS, EngineC, cellify, compact
dev = "cuda"
CFG = {"oai1m": (256, 4096), "siglip": (256, 4096), "dino": (512, 1024), "arxivL": (256, 8192), "oai3": (256, 512), "arxiv": (256, 512)}
R, NL = CFG[DS]; R = int(os.environ.get("FANAR_R", R)); BITS = 2.5
OUT = f"{FANAR}/results/v19_{MODE}" + ("" if DS == "oai1m" else f"_{DS}") + (f"_R{R}" if R != CFG[DS][0] else "") + ".json"; out = []
def emit(x):
    print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(OUT, "w"), indent=1)

class LeanF(EngineS, Engine10): pass
class NvmeF(EngineC, Engine12): pass

def build(seed=0, cls=EngineS, bits=None, **kw):
    rf.R = R; t0 = time.perf_counter(); e3, _ = make(R, nlist=NL, B=int(bits or BITS * rf.d), seed=seed); s1 = compact(cellify(e3.s1)); bt = time.perf_counter() - t0
    N = len(s1["order"]); per_s1 = (s1["words"].numel() * 4 + s1["scale1"].numel() * 2 + s1["order"].numel() * 4) / N
    tables = s1["means16"].numel() * 2 + s1["W1"].numel() * 4 + s1["inv_mnorm"].numel() * 4 + s1["c"].numel() * 4 + s1["tab"].numel() * 4
    mem = {"stage1_B_per_vec": round(per_s1, 1), "refine_B_per_vec": round(float(e3.code["bytes"]), 1), "tables_MB": round(tables / 1e6, 2),
           "gpu_B_per_vec_total": round(per_s1 + float(e3.code["bytes"]) + tables / N, 1), "lean_B_per_vec_total": round(per_s1 + tables / N, 1), "build_s": round(bt, 1)}
    return cls(s1, e3.code, **kw), mem

def row(e, nprobe, K, k=10, z=2.0, mem=None, **extra):
    I, r, C, m, s = e.bench(K, z, nprobe, k=k)
    x = {"ds": DS, "R": R, "nlist": NL, "nprobe": nprobe, "K": K, "k": k, f"recall@{k}": rf.recall(I, k), "cand_ceiling": rf.cand_recall(C, k),
         "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(r), **extra}
    if mem: x.update(mem)
    emit(x)

grid = lambda ns, Ks: [(n, K) for n in ns for K in Ks]
if MODE in ("sweep", "data"):
    e, mem = build()
    G = {"oai1m": grid((64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048), (100, 200, 400, 800)),
         "siglip": grid((32, 64, 128, 256, 512, 1024), (100, 200, 400, 800)), "dino": grid((16, 32, 64, 128), (100, 200, 400, 800, 1600)),
         "arxivL": grid((256, 512, 1024, 2048, 3072), (400, 800, 1600)), "oai3": grid((32, 64, 128, 256, 512), (100, 200, 400, 800)),
         "arxiv": grid((32, 64, 128, 256, 512), (100, 200, 400, 800))}[DS]
    if R == 384 and DS == "oai1m": G = grid((768, 1024, 1536, 2048, 3072), (400, 800, 1200))
    for nprobe, K in G: row(e, nprobe, K, mem=mem)
elif MODE == "lat":
    e, mem = build()
    for nprobe, K in ((256, 200), (384, 200), (384, 400), (512, 400), (768, 400)):
        I, r, C, m, s = e.bench(K, 2.0, nprobe); rec = rf.recall(I)
        for B in (1, 4, 16, 100, 500):
            nb = min(200, 2000 // B); ts = []
            for i in range(nb + 3):
                Qb = rf.Qg[(i * B) % 2000:(i * B) % 2000 + B]; torch.cuda.synchronize(); t0 = time.perf_counter()
                e.search(Qb, K, 2.0, nprobe, 10)[0].cpu(); ts.append(time.perf_counter() - t0)
            ts = np.array(ts[3:]) * 1000
            emit({"nprobe": nprobe, "K": K, "recall@10": rec, "batch": B, "median_ms": round(float(np.median(ts)), 2), "p99_ms": round(float(np.percentile(ts, 99)), 2)})
elif MODE == "k100":
    e, mem = build()
    for nprobe, K in ((512, 1000), (768, 1000), (1024, 1500), (1536, 1500), (1536, 2000), (2048, 2000), (3072, 3000)): row(e, nprobe, K, k=100, mem=mem)
elif MODE == "z":
    e, mem = build()
    for nprobe, K in ((384, 400), (768, 400)):
        E, S, X = [], [], []
        for s in range(0, 2000, 250):
            Qb = rf.Qg[s:s + 250]; _, _, ids, est, sg = e.search(Qb, K, 2.0, nprobe, 10, ret_est=True)
            ex = torch.stack([(rf.host16.index_select(0, ids[i].long().cpu()).to(dev).float() @ Qb[i]) for i in range(len(Qb))])
            E.append(est.cpu()); S.append(sg.cpu()); X.append(ex.cpu())
        E, S, X = torch.cat(E), torch.cat(S), torch.cat(X); Z = (E - X) / S; zs = (1, 2, 3, 4, 4.36, 4.9, 5.5)
        emit({"nprobe": nprobe, "K": K, "std_Z_engine_sigma": round(float(Z.std()), 4), "mean_Z": round(float(Z.mean()), 4),
              "uncovered_query_share": {str(z): round(float(((Z.abs() > z).any(1)).float().mean()), 5) for z in zs},
              "pair_tail": {str(z): round(float((Z.abs() > z).float().mean()), 6) for z in zs}})
        for z in (2.0, 3.0, 4.36, 4.9, 1e4): row(e, nprobe, K, z=z, zval=("exact rerank of all K" if z == 1e4 else z))
elif MODE == "seeds":
    for seed in range(5):
        e, mem = build(seed)
        for nprobe, K in ((384, 400), (768, 400), (1024, 800)): row(e, nprobe, K, seed=seed)
        del e; torch.cuda.empty_cache()
elif MODE == "lean":
    e, mem = build(cls=LeanF, lean=True)
    G = grid((192, 256, 384, 512, 768, 1024), (100, 200, 400, 800)) if R == 256 else ((1536, 400), (2048, 800), (3072, 1200))
    for nprobe, K in G: row(e, nprobe, K, mem=mem, full_vectors="host RAM")
elif MODE == "nvme":
    e, mem = build(cls=NvmeF)
    G = ((128, 100), (192, 100), (256, 200), (384, 200), (384, 400), (512, 400), (768, 400), (768, 800), (1024, 800)) if R == 256 else ((1536, 400), (2048, 800), (3072, 1200))
    for nprobe, K in G: row(e, nprobe, K, mem=mem, full_vectors="NVMe")
elif MODE == "mem":
    for Bb in (1536, 2048, 2560, 3072, 3840, 4608):
        e, mem = build(bits=Bb); T = len(e.code["TS"])
        for nprobe, K in ((384, 400), (1024, 800)): row(e, nprobe, K, mem=mem, refine_bits=Bb, tiers=T)
        del e; torch.cuda.empty_cache()
elif MODE == "fullbuild":
    import shutil, tiered
    from ideas_engine2 import reroute
    tmp = f"{FANAR}/cache/fanar_fullbuild"; shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp + "/data/rev_cache"); rf.ROOT = tmp
    for R_ in (256, 384):
        rf.R = R_; torch.cuda.synchronize(); t0 = time.perf_counter()
        s1 = rf.build_stage1(0); torch.cuda.synchronize(); t1 = time.perf_counter()
        lam, V = tiered.pca_basis(rf.host16, s1["c"]); code = tiered.build(rf.host16, s1["order"], s1["c"], V, tiered.allocate(lam, 3840), 0); torch.cuda.synchronize(); t2 = time.perf_counter()
        s1n, cn = reroute(s1, code, 4096); s1n["off_t"] = torch.as_tensor(s1n["off"], device=dev); torch.cuda.synchronize(); t3 = time.perf_counter()
        s1c = compact(cellify(s1n)); torch.cuda.synchronize(); t4 = time.perf_counter()
        e = EngineS(s1c, cn); I, r, C, m, s = e.bench(400, 2.0, 768 if R_ == 256 else 1536)
        emit({"R": R_, "stage1_s": round(t1 - t0, 1), "tiered_s": round(t2 - t1, 1), "cells_s": round(t3 - t2, 1), "cell_reencode_s": round(t4 - t3, 1), "total_s": round(t4 - t0, 1),
              "recall@10": rf.recall(I), "QPS": round(m)})
        os.remove(f"{tmp}/data/rev_cache/s1_seed0" + (f"_R{R_}" if R_ != 256 else "") + ".pt"); del e, s1, s1n, s1c, code, cn; torch.cuda.empty_cache()
print("DONE", flush=True)
