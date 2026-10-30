"""Reviewer round, GPU baselines on 998k OpenAI-1536 (WSL2): IVF-RaBitQ (cuVS 26.10, official GPU integration), k=100 for every engine,
batch-size latency for CAGRA and IVF-RaBitQ. 3 timed repeats after one warm-up; mean and std."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, os, sys, time, numpy as np, cupy as cp, faiss
from cuvs.neighbors import cagra
from cuvs_rabitq import IVFRaBitQ
X16 = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT10 = np.load(f"{FANAR}/data/oai1m_gt.npy")
N, d = X16.shape; NQ = len(Q); REPS = 3; OUT = f"{FANAR}/results/rev_baselines.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {json.dumps({k: r[k] for k in ("engine", "cfg", "k", "batch")}) for r in out}
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
def key(engine, cfg, k, batch): return json.dumps({"engine": engine, "cfg": cfg, "k": k, "batch": batch})
sync = lambda: cp.cuda.Device().synchronize()

# exact top-100 ground truth (inner product on the fp16 data, fp32 accumulate)
GT100_F = f"{FANAR}/data/oai1m_gt100.npy"
if not os.path.exists(GT100_F):
    Qg = cp.asarray(Q); best_s = cp.full((NQ, 100), -1e9, cp.float32); best_i = cp.zeros((NQ, 100), cp.int64)
    for j in range(0, N, 100000):
        S = Qg @ cp.asarray(X16[j:j + 100000]).astype(cp.float32).T
        s = cp.concatenate([best_s, S], 1); i = cp.concatenate([best_i, cp.arange(j, j + S.shape[1])[None].repeat(NQ, 0)], 1)
        o = cp.argsort(-s, 1)[:, :100]; best_s = cp.take_along_axis(s, o, 1); best_i = cp.take_along_axis(i, o, 1)
    np.save(GT100_F, cp.asnumpy(best_i)); del Qg
GT100 = np.load(GT100_F); assert np.mean([len(set(GT100[q, :10]) & set(GT10[q])) for q in range(NQ)]) > 9.9
def rec(I, k):
    G = GT10 if k == 10 else GT100[:, :k]; I = np.asarray(I)[:, :k]
    return round(float(np.mean([len(set(I[i]) & set(G[i])) / k for i in range(len(I))])), 4)
def timed(fn, nq=NQ):
    fn(); sync(); qps = []
    for _ in range(REPS):
        t0 = time.perf_counter(); I = fn(); sync(); qps.append(nq / (time.perf_counter() - t0))
    return I, round(float(np.mean(qps)), 1), round(float(np.std(qps)), 1)
def batched(search, Qg, B):
    return lambda: cp.concatenate([search(Qg[s:s + B]) for s in range(0, len(Qg), B)])

Qg = cp.asarray(Q)
# ---------------- IVF-RaBitQ ----------------
F32 = os.path.expanduser("~/oai1m_fp32.bin")
if not os.path.exists(F32):
    mm = np.memmap(F32, np.float32, "w+", shape=(N, d))
    for j in range(0, N, 50000): mm[j:j + 50000] = X16[j:j + 50000]
    mm.flush(); del mm
X32 = np.memmap(F32, np.float32, "r", shape=(N, d))
for bits in (2, 3, 4, 5, 7, 9):
    if all(key("IVF-RaBitQ", {"bits": bits, "n_lists": 1024, "n_probes": p, "mode": "QUANT4" if bits <= 5 else "QUANT8"}, 10, 500) in done for p in (16, 32, 64, 128, 256)) and bits not in (4, 7): continue
    free0 = cp.cuda.runtime.memGetInfo()[0]
    idx = IVFRaBitQ(n_lists=1024, bits_per_dim=bits); sync(); t0 = time.perf_counter(); idx.build(X32); sync(); bt = round(time.perf_counter() - t0, 1)
    gpu_b = round((free0 - cp.cuda.runtime.memGetInfo()[0]) / N, 1)
    print(f"IVF-RaBitQ bits={bits} built in {bt}s, ~{gpu_b} B/vec on GPU", flush=True)
    modes = ("QUANT4", "QUANT8", "LUT16") if bits == 4 else ("QUANT4" if bits <= 5 else "QUANT8",)
    for mode in modes:
        for npb in (16, 32, 64, 128, 256):
            cfg = {"bits": bits, "n_lists": 1024, "n_probes": npb, "mode": mode}
            if key("IVF-RaBitQ", cfg, 10, 500) in done: continue
            try:
                I, m, s = timed(batched(lambda q: idx.search(q, 10, npb, mode)[0], Qg, 500))
                emit({"engine": "IVF-RaBitQ", "cfg": cfg, "k": 10, "batch": 500, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s, "gpu_bytes_measured": gpu_b, "build_s": bt})
            except Exception as e: print("rabitq failed", cfg, str(e)[:200], flush=True)
    if bits in (4, 7):                                                                     # k=100 and small-batch latency
        for npb in (64, 128, 256):
            cfg = {"bits": bits, "n_lists": 1024, "n_probes": npb, "mode": "QUANT4" if bits <= 5 else "QUANT8"}
            if key("IVF-RaBitQ", cfg, 100, 500) not in done:
                I, m, s = timed(batched(lambda q: idx.search(q, 100, npb, cfg["mode"])[0], Qg, 500))
                emit({"engine": "IVF-RaBitQ", "cfg": cfg, "k": 100, "batch": 500, "recall": rec(cp.asnumpy(I), 100), "QPS": m, "QPS_std": s, "gpu_bytes_measured": gpu_b})
        cfg = {"bits": bits, "n_lists": 1024, "n_probes": 128, "mode": "QUANT4" if bits <= 5 else "QUANT8"}
        for B, nq in ((1, 200), (16, 480), (100, 2000)):
            if key("IVF-RaBitQ", cfg, 10, B) in done: continue
            I, m, s = timed(batched(lambda q: idx.search(q, 10, 128, cfg["mode"])[0], Qg[:nq], B), nq)
            emit({"engine": "IVF-RaBitQ", "cfg": cfg, "k": 10, "batch": B, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s, "latency_ms": round(B / m * 1000, 3)})
    del idx; cp.get_default_memory_pool().free_all_blocks()
del X32

# ---------------- CAGRA: k=100 and small batches ----------------
Xg = cp.asarray(np.ascontiguousarray(X16)); Qh = cp.asarray(Q.astype(np.float16))
for deg in (32, 64):
    idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg), Xg); sync()
    for itopk in ((128, 256, 512, 1024) if deg == 32 else (512, 1024)):
        cfg = {"graph_degree": deg, "itopk": itopk}
        if key("CAGRA", cfg, 100, 500) in done: continue
        sp = cagra.SearchParams(itopk_size=itopk)
        I, m, s = timed(batched(lambda q: cp.asarray(cagra.search(sp, idx, q, 100)[1]), Qh, 500))
        emit({"engine": "CAGRA", "cfg": cfg, "k": 100, "batch": 500, "recall": rec(cp.asnumpy(I), 100), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d + 4 * deg})
    if deg == 32:
        for itopk in (256, 512):
            sp = cagra.SearchParams(itopk_size=itopk); cfg = {"graph_degree": 32, "itopk": itopk}
            for B, nq in ((1, 200), (16, 480), (100, 2000)):
                if key("CAGRA", cfg, 10, B) in done: continue
                I, m, s = timed(batched(lambda q: cp.asarray(cagra.search(sp, idx, q, 10)[1]), Qh[:nq], B), nq)
                emit({"engine": "CAGRA", "cfg": cfg, "k": 10, "batch": B, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s, "latency_ms": round(B / m * 1000, 3)})
    del idx; cp.get_default_memory_pool().free_all_blocks()
del Xg; cp.get_default_memory_pool().free_all_blocks()

# ---------------- FAISS-GPU flat fp16, k=100 ----------------
if key("FAISS-GPU flat fp16", {}, 100, 500) not in done:
    res = faiss.StandardGpuResources(); fc = faiss.GpuIndexFlatConfig(); fc.useFloat16 = True; fl = faiss.GpuIndexFlatIP(res, d, fc)
    for j in range(0, N, 100000): fl.add(np.asarray(X16[j:j + 100000], np.float32))
    I, m, s = timed(lambda: fl.search(Q, 100)[1])
    emit({"engine": "FAISS-GPU flat fp16", "cfg": {}, "k": 100, "batch": 2000, "recall": rec(I, 100), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d})
print("DONE", flush=True)
