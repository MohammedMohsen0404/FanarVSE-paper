"""Reviewer round on GloVe-100 (1.18M) and OpenAI 229k: IVF-RaBitQ (cuVS) sweep and CAGRA with the multi_cta algorithm + RMM pool.
usage: python small_ds.py <dbp|glove>   (same preprocessing, queries and ground truth as cagra_ext.py)"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, cupy as cp, rmm
rmm.reinitialize(pool_allocator=True, initial_pool_size=2 ** 30)
from rmm.allocators.cupy import rmm_cupy_allocator
cp.cuda.set_allocator(rmm_cupy_allocator)
from cuvs.neighbors import cagra
from cuvs_rabitq import IVFRaBitQ
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
Tg, Qg = cp.asarray(np.ascontiguousarray(T)), cp.asarray(Q); del X; sync = lambda: cp.cuda.Device().synchronize()
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(f"{FANAR}/results/rev_small_{ds}.json", "w"), indent=1)
def timed(f):
    f(); sync(); q = []
    for _ in range(3):
        t0 = time.perf_counter(); I = f(); sync(); q.append(NQ / (time.perf_counter() - t0))
    return I, round(float(np.mean(q)), 1), round(float(np.std(q)), 1)
lists = (1024, 4096) if ds == "glove" else (512, 1024)
for bits in (4, 7, 9):
    for nl in lists:
        free0 = cp.cuda.runtime.memGetInfo()[0]; idx = IVFRaBitQ(n_lists=nl, bits_per_dim=bits); t0 = time.perf_counter(); idx.build(Tg); sync(); bt = round(time.perf_counter() - t0, 1)
        gb = round((free0 - cp.cuda.runtime.memGetInfo()[0]) / N, 1); mode = "QUANT4" if bits <= 5 else "QUANT8"
        for frac in (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2):
            npb = int(nl * frac)
            I, m, s = timed(lambda: cp.concatenate([idx.search(Qg[st:st + 500], 10, npb, mode)[0] for st in range(0, NQ, 500)]))
            emit({"ds": ds, "N": N, "engine": "IVF-RaBitQ", "bits": bits, "n_lists": nl, "n_probes": npb, "recall@10": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes_measured": gb, "build_s": bt})
        idx.free()
for deg, itopks in ((32, (256, 512, 1024)), (64, (256, 512, 1024))):
    idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg), Tg); sync()
    for itopk in itopks:
        sp = cagra.SearchParams(itopk_size=itopk, algo="multi_cta")
        I, m, s = timed(lambda: cp.asarray(cagra.search(sp, idx, Qg, 10)[1]))
        emit({"ds": ds, "N": N, "engine": f"CAGRA deg{deg} multi_cta (fp32 dataset)", "itopk": itopk, "recall@10": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 4 * d + 4 * deg})
    del idx
print("DONE", flush=True)
