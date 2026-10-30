"""CAGRA on the 998k OpenAI set at batch 500 / 2000 / 4000 (4000 = the 2000 queries twice), itopk x algorithm sweep, RMM pool,
3 timed repeats (best of three). Fair large-batch baseline for the FanarVSE round-11 engines."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, time, numpy as np, cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
rmm.reinitialize(pool_allocator=True, initial_pool_size=2 ** 31); cp.cuda.set_allocator(rmm_cupy_allocator)
from cuvs.neighbors import cagra
X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT = np.load(f"{FANAR}/data/oai1m_gt.npy")
N, d = X.shape; NQ = len(Q); out = []
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i, :10])) / 10 for i in range(NQ)])), 4)
def emit(r): print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(f"{FANAR}/results/cagra_batch.json", "w"), indent=1)
Xg = cp.asarray(np.ascontiguousarray(X)); Q2 = np.concatenate([Q, Q]); Qg = cp.asarray(Q2.astype(np.float16))
ip = cagra.IndexParams(metric="inner_product", graph_degree=32, intermediate_graph_degree=64)
idx = cagra.build(ip, Xg); cp.cuda.Device().synchronize()
for itopk in (128, 256, 384, 512):
    for algo in ("single_cta", "multi_cta", "multi_kernel", "auto"):
        try:
            sp = cagra.SearchParams(itopk_size=itopk, algo=algo)
            for B in (500, 2000, 4000):
                def run():
                    Is = []
                    for s in range(0, max(B, NQ), B): Is.append(cp.asarray(cagra.search(sp, idx, Qg[s:s + B], 10)[1]))
                    return cp.concatenate(Is)
                I = run(); cp.cuda.Device().synchronize(); best = 0
                n = max(B, NQ)
                for _ in range(3):
                    t0 = time.perf_counter(); run(); cp.cuda.Device().synchronize(); best = max(best, n / (time.perf_counter() - t0))
                emit({"engine": "CAGRA deg32 fp16", "itopk": itopk, "algo": algo, "batch": B, "recall@10": rec(cp.asnumpy(I)[:NQ]), "QPS": round(best)})
        except Exception as e: print("fail", itopk, algo, str(e)[:150], flush=True)
print("DONE", flush=True)
