"""Measurements of engine v3 for the paper. usage: ideas_final.py {sweep|zsweep|latency|seeds|chain}  (FANAR_R=256|384)"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import Engine3, Engine4, Engine8, rf, load, reroute
from engine3 import make as _make
def make(R=256, nlist=4096, B=3840, seed=0):
    e, nb = _make(R, nlist, B, seed); return Engine8(e.s1, e.code), nb
import s1fast, tiered
from scipy.special import betainc
dev = "cuda"; MODE = sys.argv[1]; R = int(os.environ.get("FANAR_R", 256)); out = []
OUT = f"{FANAR}/results/v8_{MODE}" + (f"_R{R}" if MODE == "sweep" else "") + ".json"
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)

if MODE == "sweep":
    eng, nbytes = make(R)
    grid = [(n, K) for n in (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048) for K in (100, 200, 400, 800)]
    if R == 384: grid += [(3072, 800), (3072, 1200), (4096, 800), (4096, 1200)]
    for nprobe, K in grid:
        I, reads, C, m, s = eng.bench(K, 2.0, nprobe)
        emit({"R": R, "nlist": 4096, "nprobe": nprobe, "K": K, "gpu_bytes": nbytes, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C),
              "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(reads)})

elif MODE == "zsweep":
    eng, nbytes = make(256); nprobe, K = 768, 400
    # coverage: exact scores of all candidates; share of queries with an uncovered candidate; std of the standardised error (engine sigma)
    E, S, X = [], [], []
    for s in range(0, 2000, 250):
        Qb = rf.Qg[s:s + 250]; _, _, ids, est, sg = eng.search(Qb, K, 2.0, nprobe, 10, ret_est=True)
        ex = torch.stack([(rf.host16.index_select(0, ids[i].cpu()).to(dev).float() @ Qb[i]) for i in range(len(Qb))])
        E.append(est.cpu()); S.append(sg.cpu()); X.append(ex.cpu())
    E, S, X = torch.cat(E), torch.cat(S), torch.cat(X); Z = (E - X) / S
    zs = (1, 2, 3, 4, 4.36, 4.9, 5.5)
    emit({"std_Z_engine_sigma": round(float(Z.std()), 4), "mean_Z": round(float(Z.mean()), 4),
          "uncovered_query_share": {str(z): round(float(((Z.abs() > z).any(1)).float().mean()), 5) for z in zs},
          "subgaussian_bound_2Kexp": {str(z): round(min(1, K * 2 * np.exp(-z * z / 2)), 5) for z in zs},
          "pair_tail": {str(z): round(float((Z.abs() > z).float().mean()), 6) for z in zs}})
    for z in (1.0, 2.0, 3.0, 4.36, 4.9):
        I, reads, C, m, s = eng.bench(K, z, nprobe)
        emit({"z": z, "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(reads)})
    I, reads, C, m, s = eng.bench(K, 1e4, nprobe)
    emit({"z": "exact rerank of all K", "recall@10": rf.recall(I), "QPS": round(m, 1), **rf.read_stats(reads)})

elif MODE == "latency":
    eng, nbytes = make(256)
    for nprobe, K in ((256, 200), (384, 200), (512, 400), (768, 400)):
        for B in (1, 4, 16, 100, 500):
            nb = min(200, 2000 // B); ts = []
            for i in range(nb + 3):
                Qb = rf.Qg[(i * B) % 2000:(i * B) % 2000 + B]; torch.cuda.synchronize(); t0 = time.perf_counter()
                I = eng.search(Qb, K, 2.0, nprobe, 10)[0].cpu(); ts.append(time.perf_counter() - t0)
            ts = np.array(ts[3:]) * 1000
            emit({"nprobe": nprobe, "K": K, "batch": B, "median_ms": round(float(np.median(ts)), 2), "p99_ms": round(float(np.percentile(ts, 99)), 2)})

elif MODE == "seeds":
    for seed in range(5):
        eng, nbytes = make(256, seed=seed)
        for nprobe, K in ((768, 400), (1024, 800)):
            I, reads, C, m, s = eng.bench(K, 2.0, nprobe)
            emit({"seed": seed, "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(reads)})
        del eng; torch.cuda.empty_cache()

elif MODE == "k100":
    eng, nbytes = make(256)
    for nprobe, K in ((768, 1000), (1024, 1500), (1536, 1500), (1536, 2000), (2048, 2000), (3072, 3000)):
        I, reads, C, m, s = eng.bench(K, 2.0, nprobe, k=100, B=500)
        emit({"k": 100, "nprobe": nprobe, "K": K, "batch": 500, "gpu_bytes": nbytes, "recall@100": rf.recall(I, 100), "cand_ceiling": rf.cand_recall(C, 100), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(reads)})

elif MODE == "mem":
    for Bb in (1536, 2048, 2560, 3072, 3840, 4608):
        eng, nbytes = make(256, B=Bb); T = eng.code["TS"]
        for nprobe, K in ((512, 400), (768, 400), (1024, 800)):
            I, r, C, m, s = eng.bench(K, 2.0, nprobe)
            emit({"refine_bits": Bb, "gpu_bytes": nbytes, "tiers": [[e_ - s_, b] for s_, e_, b in T], "nprobe": nprobe, "K": K, "recall@10": rf.recall(I), "cand_ceiling": rf.cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **rf.read_stats(r)})
        del eng; torch.cuda.empty_cache()

elif MODE == "chain":
    # each component added in turn, same queries and ground truth
    from ideas_engine2 import Engine2
    s1, code = load(3840)
    s2 = rf.build_stage2(s1, rf.rotation("haar")); base = rf.Engine(s1, s2)
    I, r, C, m, s = base.bench(400, 2.0, 32); emit({"step": "paper engine: 64 cells, 4-bit stage 2 (820 B)", "nprobe": 32, "K": 400, "recall@10": rf.recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2)})
    del base, s2; torch.cuda.empty_cache()
    e2 = Engine2(s1, code); I, r, C, m, s = e2.bench(400, 2.0, 32)
    emit({"step": "+ tiered stage 2 (552 B)", "nprobe": 32, "K": 400, "recall@10": rf.recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2)})
    s1n, cn = reroute(s1, code, 4096); s1n["off_t"] = torch.as_tensor(s1n["off"], device=dev)
    class A(Engine2):
        def stage1(self, Qb, K, nprobe):
            t = self.s1; QC = Qb @ t["c"]; return s1fast.search_flat(Qb @ t["W1"], Qb, QC, t["words"], t["tab"], t["scale1"], 10, K, t["cen"], t["off_t"], nprobe)[0], QC
    class Bt(Engine2):
        def stage1(self, Qb, K, nprobe):
            t = self.s1; QC = Qb @ t["c"]; return s1fast.search_thresh(Qb @ t["W1"], Qb, QC, t["words"], t["tab"], t["scale1"], 10, K, t["cen"], t["off_t"], nprobe, frac1=0.125, CAP=16384)[0], QC
    class Cl(Engine2):
        def stage1(self, Qb, K, nprobe):
            t = self.s1; QC = Qb @ t["c"]; return s1fast.search_thresh2(Qb @ t["W1"], Qb, QC, t["words"], t["tab"], t["scale1"], 10, K, t["cen"], t["off_t"], nprobe, CAP=16384)[0], QC
    for name, cls in (("+ 4,096 cells, one-launch work list, block-max selection", A), ("+ threshold-and-emit selection", Bt), ("+ decode-once kernel, adaptive first pass", Cl), ("+ stage 3 on the GPU", Engine3),
                      ("+ guess-and-verify threshold, fp16 queries, fused stage 2", Engine4), ("+ compact first pass, zero-copy stage 3", Engine8)):
        e = cls(s1n, cn); I, r, C, m, s = e.bench(400, 2.0, 768)
        emit({"step": name, "nprobe": 768, "K": 400, "recall@10": rf.recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads": round(float(r.mean()), 2)})
print("DONE", flush=True)
