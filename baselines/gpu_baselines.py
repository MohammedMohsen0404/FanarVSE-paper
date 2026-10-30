"""Fair GPU baselines (FAISS-GPU, CAGRA if available) on the same data/queries/ground truth as FanarVSE.
usage: python gpu_baselines.py <dbp|glove>   (run inside WSL venv)"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, faiss
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
N, d = T.shape; res = faiss.StandardGpuResources(); print(ds, N, d, flush=True)
flat = faiss.GpuIndexFlatIP(res, d); flat.add(T); _, GT = flat.search(Q, 10)
np.save(f"{FANAR}/results/gt_{ds}_wsl.npy", GT)
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)

def bench(name, idx, bpv, build_s, setp=None, params=(None,), batch=NQ):
    for p in params:
        if setp: setp(p)
        idx.search(Q[:100], 10); t0 = time.perf_counter()
        I = np.vstack([idx.search(Q[s:s + batch], 10)[1] for s in range(0, NQ, batch)]); dt = time.perf_counter() - t0
        r = {"ds": ds, "N": N, "engine": name, "param": p, "batch": batch, "recall@10": rec(I), "QPS": round(NQ / dt, 1), "index_bytes_per_vec": bpv, "build_s": round(build_s, 2)}
        print(json.dumps(r), flush=True); out.append(r)

# exact brute force (fp32, fp16)
bench("FAISS-GPU Flat fp32", flat, 4 * d, 0)
cfg = faiss.GpuIndexFlatConfig(); cfg.useFloat16 = True; f16 = faiss.GpuIndexFlatIP(res, d, cfg); t0 = time.perf_counter(); f16.add(T); bench("FAISS-GPU Flat fp16", f16, 2 * d, time.perf_counter() - t0); del f16
nlist = 1024 if N < 500000 else 2048
# IVF-Flat
t0 = time.perf_counter(); ivf = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_INNER_PRODUCT); ivf.train(T[:200000]); ivf.add(T); bt = time.perf_counter() - t0
bench("FAISS-GPU IVF-Flat", ivf, 4 * d + 8, bt, lambda p: setattr(ivf, "nprobe", p), (16, 32, 64, 128)); del ivf
# IVF-PQ (+ optional CPU-side exact rerank of the GPU candidates)
for M in ((96, 48) if d == 1536 else (50, 20)):
    try:
        t0 = time.perf_counter(); pq = faiss.GpuIndexIVFPQ(res, d, nlist, M, 8, faiss.METRIC_INNER_PRODUCT); pq.train(T[:200000]); pq.add(T); bt = time.perf_counter() - t0
        bench(f"FAISS-GPU IVF-PQ M={M}", pq, M + 8, bt, lambda p: setattr(pq, "nprobe", p), (32, 64, 128))
        # manual rerank: GPU returns top-k', exact re-score on CPU
        for kp in (40, 100):
            for p in (64, 128):
                pq.nprobe = p; t0 = time.perf_counter(); _, C = pq.search(Q, kp)
                ex = np.einsum("bkd,bd->bk", T[C], Q); I = np.take_along_axis(C, np.argsort(-ex, 1)[:, :10], 1); dt = time.perf_counter() - t0
                r = {"ds": ds, "N": N, "engine": f"FAISS-GPU IVF-PQ M={M} +rerank{kp}", "param": p, "batch": NQ, "recall@10": rec(I), "QPS": round(NQ / dt, 1), "index_bytes_per_vec": M + 8, "build_s": round(bt, 2)}
                print(json.dumps(r), flush=True); out.append(r)
        del pq
    except Exception as e: print("IVFPQ", M, "failed:", str(e)[:160], flush=True)
# CAGRA via FAISS (if compiled with cuVS)
try:
    cp = faiss.GpuIndexCagraConfig(); cp.graph_degree = 32; cp.intermediate_graph_degree = 64
    t0 = time.perf_counter(); cg = faiss.GpuIndexCagra(res, d, faiss.METRIC_INNER_PRODUCT, cp); cg.train(T); bt = time.perf_counter() - t0
    for itopk in (32, 64, 128, 256):
        sp = faiss.SearchParametersCagra(); sp.itopk_size = itopk; cg.search(Q[:100], 10, params=sp); t0 = time.perf_counter(); _, I = cg.search(Q, 10, params=sp); dt = time.perf_counter() - t0
        r = {"ds": ds, "N": N, "engine": "FAISS-GPU CAGRA (deg32)", "param": itopk, "batch": NQ, "recall@10": rec(I), "QPS": round(NQ / dt, 1), "index_bytes_per_vec": 4 * d + 32 * 4, "build_s": round(bt, 2)}
        print(json.dumps(r), flush=True); out.append(r)
except Exception as e: print("CAGRA via faiss unavailable:", str(e)[:200], flush=True)
json.dump(out, open(f"{FANAR}/results/gpu_baselines_{ds}.json", "w"), indent=1)
