"""turbovec (TurboQuant, CPU SIMD brute force) on the same data/queries/GT. With and without exact rerank of its top-k'.
usage: bench_turbovec.py <dbp229|oai1m|glove>"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, turbovec
ds = sys.argv[1]; out = []
if ds == "glove":
    import h5py; f = h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r"); X = np.array(f["train"], np.float32); Q = np.array(f["test"][:2000], np.float32)
    mu = X[:100000].mean(0); X -= mu; X /= np.linalg.norm(X, axis=1, keepdims=True); Q -= mu; Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    X = np.hstack([X, np.zeros((len(X), 4), np.float32)]); Q = np.hstack([Q, np.zeros((len(Q), 4), np.float32)]); GT = np.load(f"{FANAR}/results/gt_glove_wsl.npy")
elif ds == "dbp229":
    import pyarrow.parquet as pq_
    files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
    Z = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
    Z = Z[np.random.default_rng(0).permutation(len(Z))]; Z -= Z[:50000].mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    X, Q = np.ascontiguousarray(Z[:-2000]), np.ascontiguousarray(Z[-2000:]); del Z; GT = np.load(f"{FANAR}/results/gt_dbp_wsl.npy")
else:
    X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT = np.load(f"{FANAR}/data/oai1m_gt.npy")
N, d = X.shape; NQ = len(Q)
rec = lambda I: round(float(np.mean([len(set(np.asarray(I[i])[:10]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
for bits in ((2,) if ds == "oai1m" else (2, 4)):
    for calib in ((False,) if ds == "oai1m" else (False, True)):
        try: idx = turbovec.TurboQuantIndex(d, bits)
        except TypeError: idx = turbovec.TurboQuantIndex(dim=d, bit_width=bits)
        t0 = time.perf_counter()
        if calib: idx.calibrate(np.ascontiguousarray(np.asarray(X[np.random.default_rng(0).choice(N, 4096, replace=False)], np.float32)))
        for j in range(0, N, 100000): idx.add(np.ascontiguousarray(np.asarray(X[j:j + 100000], np.float32)))
        bt = time.perf_counter() - t0
        for kp in (10, 40, 100):
            idx.search(Q[:50], kp); t0 = time.perf_counter(); res = idx.search(Q, kp); dt_s = time.perf_counter() - t0
            I = res[1] if isinstance(res, tuple) else res
            I = np.asarray(I)
            if I.shape[1] != kp and isinstance(res, tuple): I = np.asarray(res[0]) if np.asarray(res[0]).dtype.kind in "iu" else I
            if kp > 10:
                t1 = time.perf_counter()
                Xc = np.asarray(X[np.sort(np.unique(I))], np.float32) if False else None
                ex = np.einsum("bkd,bd->bk", np.asarray(X[I.reshape(-1)], np.float32).reshape(NQ, kp, d), Q); I = np.take_along_axis(I, np.argsort(-ex, 1)[:, :10], 1)
                dt = dt_s + (time.perf_counter() - t1)
            else: dt = dt_s
            r = {"ds": ds, "N": N, "engine": f"turbovec {bits}-bit" + (" +calibrate" if calib else ""), "k'": kp, "rerank": kp > 10, "bytes_per_vec": d * bits // 8 + 4,
                 "recall@10": rec(I), "QPS": round(NQ / dt, 1), "build_s": round(bt, 1)}
            print(json.dumps(r), flush=True); out.append(r)
        del idx
json.dump(out, open(f"{FANAR}/results/turbovec_{ds}.json", "w"), indent=1)
