"""CAGRA (cuVS) baselines, same data/queries/GT as gpu_baselines.py. Full-precision dataset and VPQ-compressed dataset.
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
Tg, Qg = cp.asarray(T), cp.asarray(Q)
configs = [("CAGRA deg32 (full fp32 dataset)", dict(), 4 * d + 32 * 4)]
for name, extra, bpv in configs:
    try:
        ip = cagra.IndexParams(metric="inner_product", graph_degree=32, intermediate_graph_degree=64, **extra)
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); idx = cagra.build(ip, Tg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
        for itopk in (32, 64, 128, 256):
            sp = cagra.SearchParams(itopk_size=itopk); cagra.search(sp, idx, Qg[:100], 10); cp.cuda.Device().synchronize()
            t0 = time.perf_counter(); _, I = cagra.search(sp, idx, Qg, 10); cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0
            r = {"ds": ds, "N": N, "engine": name, "itopk": itopk, "recall@10": rec(cp.asnumpy(cp.asarray(I))), "QPS": round(NQ / dt, 1), "index_bytes_per_vec": bpv, "build_s": round(bt, 2)}
            print(json.dumps(r), flush=True); out.append(r)
        del idx
    except Exception as e: print(name, "failed:", str(e)[:200], flush=True)
# cuVS IVF-PQ (+ GPU refine on the raw dataset) at FanarVSE-like code sizes
for pq_dim in ((96, 48) if d >= 512 else (50, 25)):
    try:
        ipp = ivf_pq.IndexParams(n_lists=1024, metric="inner_product", pq_dim=pq_dim, pq_bits=8)
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); pidx = ivf_pq.build(ipp, Tg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
        for npb in (32, 64, 128):
            for kp in (10, 40, 100):
                spp = ivf_pq.SearchParams(n_probes=npb); ivf_pq.search(spp, pidx, Qg[:100], kp); cp.cuda.Device().synchronize(); t0 = time.perf_counter()
                _, C = ivf_pq.search(spp, pidx, Qg, kp)
                if kp > 10: _, I = refine(Tg, Qg, cp.asarray(C), k=10, metric="inner_product")
                else: I = C
                cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0
                r = {"ds": ds, "N": N, "engine": f"cuVS IVF-PQ pq_dim={pq_dim}" + (f" +refine{kp}" if kp > 10 else ""), "nprobe": npb, "recall@10": rec(cp.asnumpy(cp.asarray(I))),
                     "QPS": round(NQ / dt, 1), "index_bytes_per_vec": pq_dim + 8, "build_s": round(bt, 2)}
                print(json.dumps(r), flush=True); out.append(r)
        del pidx
    except Exception as e: print("ivf_pq", pq_dim, "failed:", str(e)[:200], flush=True)
json.dump(out, open(f"{FANAR}/results/cagra_{ds}.json", "w"), indent=1)
