"""Second pass with the RMM pool: CAGRA k=10 batch 500 at large itopk with multi_cta (best algo at k=100), both graph degrees;
IVF-RaBitQ small-batch latency with the pool (best high-recall configuration)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, os, sys, time, numpy as np, cupy as cp, rmm
rmm.reinitialize(pool_allocator=True, initial_pool_size=2 ** 30)
from rmm.allocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)
from cuvs.neighbors import cagra
from cuvs_rabitq import IVFRaBitQ
X16 = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy")
GT10 = np.load(f"{FANAR}/data/oai1m_gt.npy"); GT100 = np.load(f"{FANAR}/data/oai1m_gt100.npy"); N, d = X16.shape
OUT = f"{FANAR}/results/rev_diag2.json"; out = []; sync = lambda: cp.cuda.Device().synchronize()
def rec(I, k):
    G = GT10 if k == 10 else GT100[:, :k]; return round(float(np.mean([len(set(I[i, :k]) & set(G[i])) / k for i in range(len(I))])), 4)
def timed(f, nq):
    f(); sync(); q = []
    for _ in range(3):
        t0 = time.perf_counter(); I = f(); sync(); q.append(nq / (time.perf_counter() - t0))
    return I, round(float(np.mean(q)), 1), round(float(np.std(q)), 1)
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
Qg32 = cp.asarray(Q)
X32 = np.memmap(os.path.expanduser("~/oai1m_fp32.bin"), np.float32, "r", shape=(N, d))
for bits, nl, npb in ((9, 4096, 384), (9, 4096, 768), (7, 1024, 160)):
    idx = IVFRaBitQ(n_lists=nl, bits_per_dim=bits); idx.build(X32)
    for B, nq in ((1, 200), (16, 480), (100, 2000), (500, 2000)):
        I, m, s = timed(lambda: cp.concatenate([idx.search(Qg32[st:st + B], 10, npb, "QUANT8")[0] for st in range(0, nq, B)]), nq)
        emit({"engine": "IVF-RaBitQ", "bits": bits, "n_lists": nl, "n_probes": npb, "k": 10, "batch": B, "rmm_pool": True, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s, "latency_ms": round(B / m * 1000, 3)})
    for p2 in ((npb, 2 * npb) if nl == 4096 and npb == 384 else ()):
        I, m, s = timed(lambda: cp.concatenate([idx.search(Qg32[st:st + 500], 100, p2, "QUANT8")[0] for st in range(0, 2000, 500)]), 2000)
        emit({"engine": "IVF-RaBitQ", "bits": bits, "n_lists": nl, "n_probes": p2, "k": 100, "batch": 500, "rmm_pool": True, "recall": rec(cp.asnumpy(I), 100), "QPS": m, "QPS_std": s})
    if nl == 4096 and npb == 768:
        for p2 in (1536, 2048):                                                          # recall ceiling of the 9-bit estimator
            I, m, s = timed(lambda: cp.concatenate([idx.search(Qg32[st:st + 500], 10, p2, "QUANT8")[0] for st in range(0, 2000, 500)]), 2000)
            emit({"engine": "IVF-RaBitQ", "bits": bits, "n_lists": nl, "n_probes": p2, "k": 10, "batch": 500, "rmm_pool": True, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s})
    idx.free()
del X32
Xg = cp.asarray(np.ascontiguousarray(X16)); Qh = cp.asarray(Q.astype(np.float16))
for deg in (32, 64):
    idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg), Xg); sync()
    for itopk in (256, 384, 512, 768, 1024):
        for algo in ("auto", "multi_cta"):
            sp = cagra.SearchParams(itopk_size=itopk, algo=algo)
            I, m, s = timed(lambda: cp.concatenate([cp.asarray(cagra.search(sp, idx, Qh[st:st + 500], 10)[1]) for st in range(0, 2000, 500)]), 2000)
            emit({"engine": "CAGRA", "graph_degree": deg, "itopk": itopk, "algo": algo, "k": 10, "batch": 500, "rmm_pool": True, "recall": rec(cp.asnumpy(I), 10), "QPS": m, "QPS_std": s})
        if deg == 64 and itopk >= 512:
            sp = cagra.SearchParams(itopk_size=itopk, algo="multi_cta")
            I, m, s = timed(lambda: cp.concatenate([cp.asarray(cagra.search(sp, idx, Qh[st:st + 500], 100)[1]) for st in range(0, 2000, 500)]), 2000)
            emit({"engine": "CAGRA", "graph_degree": deg, "itopk": itopk, "algo": "multi_cta", "k": 100, "batch": 500, "rmm_pool": True, "recall": rec(cp.asnumpy(I), 100), "QPS": m, "QPS_std": s})
    del idx; cp.get_default_memory_pool().free_all_blocks()
print("DONE", flush=True)
