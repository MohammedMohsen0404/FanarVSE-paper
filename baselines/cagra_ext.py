"""Paper rerun: CAGRA extended to itopk 512/1024 and graph degree 64, 3 timed repeats. usage: python cagra_ext.py <dbp|glove>
usage: python cagra_bench.py <dbp|glove>"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, cupy as cp
from cuvs.neighbors import cagra, ivf_pq
from cuvs.neighbors.refine import refine
ds = sys.argv[1]; NQ = 2000; out = []
if ds == "glove":
    import h5py; f = h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r"); X = np.array(f["train"], np.float32); Qr = np.array(f["test"][:NQ], np.float32)
    mu = X[:100000].mean(0); X -= mu; X /= np.linalg.norm(X, axis=1, keepdims=True); Qr -= mu; Qr /= np.linalg.norm(Qr, axis=1, keepdims=True); T, Q = X, Qr
else:
    import pyarrow.parquet as pq_
    files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
    X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
    X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
    T, Q = np.ascontiguousarray(X[:-NQ]), np.ascontiguousarray(X[-NQ:])
N, d = T.shape; GT = np.load(f"{FANAR}/results/gt_{ds}_wsl.npy")
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
Tg, Qg = cp.asarray(T), cp.asarray(Q); del X
def emit(r): print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(f"{FANAR}/results/cagra_ext_{ds}.json", "w"), indent=1)
for deg, itopks in ((32, (128, 256, 512, 1024)), (64, (256, 512, 1024))):
    try:
        ip = cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg)
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); idx = cagra.build(ip, Tg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
        for itopk in itopks:
            sp = cagra.SearchParams(itopk_size=itopk); cagra.search(sp, idx, Qg, 10); cp.cuda.Device().synchronize(); qps = []
            for _ in range(3):
                t0 = time.perf_counter(); _, I = cagra.search(sp, idx, Qg, 10); cp.cuda.Device().synchronize(); qps.append(NQ / (time.perf_counter() - t0))
            emit({"ds": ds, "N": N, "engine": f"CAGRA deg{deg} (fp32 dataset)", "itopk": itopk, "recall@10": rec(cp.asnumpy(cp.asarray(I))), "QPS": round(float(np.mean(qps)), 1),
                  "QPS_std": round(float(np.std(qps)), 1), "index_bytes_per_vec": 4 * d + 4 * deg, "build_s": round(bt, 2)})
        del idx; cp.get_default_memory_pool().free_all_blocks()
    except Exception as e: print("deg", deg, "failed:", str(e)[:200], flush=True)
print("DONE", flush=True)
