"""Diagnose CAGRA at small batches and k=100: search algorithm (auto / single_cta / multi_cta / multi_kernel) and RMM pool allocator.
Writes the best setting per (batch, k, itopk) so the paper reports CAGRA at its best."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, os, sys, time, numpy as np, cupy as cp, rmm
rmm.reinitialize(pool_allocator=True, initial_pool_size=2 ** 30)
from rmm.allocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)
from cuvs.neighbors import cagra
X16 = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy")
GT10 = np.load(f"{FANAR}/data/oai1m_gt.npy"); GT100 = np.load(f"{FANAR}/data/oai1m_gt100.npy"); d = X16.shape[1]
OUT = f"{FANAR}/results/rev_cagra_diag.json"; out = []
def rec(I, k):
    G = GT10 if k == 10 else GT100[:, :k]; return round(float(np.mean([len(set(I[i, :k]) & set(G[i])) / k for i in range(len(I))])), 4)
Xg = cp.asarray(np.ascontiguousarray(X16)); Qh = cp.asarray(Q.astype(np.float16)); sync = lambda: cp.cuda.Device().synchronize()
idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=32, intermediate_graph_degree=64), Xg); sync()
for k, itopk, B, nq in ((10, 256, 1, 200), (10, 256, 16, 480), (10, 256, 100, 2000), (10, 256, 500, 2000), (10, 512, 1, 200), (10, 512, 16, 480), (10, 512, 100, 2000),
                        (100, 256, 500, 2000), (100, 512, 500, 2000), (100, 1024, 500, 2000)):
    for algo in ("auto", "single_cta", "multi_cta", "multi_kernel"):
        try:
            sp = cagra.SearchParams(itopk_size=itopk, algo=algo)
            f = lambda: cp.concatenate([cp.asarray(cagra.search(sp, idx, Qh[s:s + B], k)[1]) for s in range(0, nq, B)])
            f(); sync(); qps = []
            for _ in range(3):
                t0 = time.perf_counter(); I = f(); sync(); qps.append(nq / (time.perf_counter() - t0))
            r = {"engine": "CAGRA", "graph_degree": 32, "itopk": itopk, "k": k, "batch": B, "algo": algo, "rmm_pool": True, "recall": rec(cp.asnumpy(I), k),
                 "QPS": round(float(np.mean(qps)), 1), "QPS_std": round(float(np.std(qps)), 1), "latency_ms": round(B / float(np.mean(qps)) * 1000, 3)}
        except Exception as e: r = {"itopk": itopk, "k": k, "batch": B, "algo": algo, "error": str(e)[:150]}
        print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
print("DONE", flush=True)
