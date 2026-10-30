import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, time, numpy as np, faiss
X16 = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT100 = np.load(f"{FANAR}/data/oai1m_gt100.npy"); N, d = X16.shape
res = faiss.StandardGpuResources(); fc = faiss.GpuIndexFlatConfig(); fc.useFloat16 = True; fl = faiss.GpuIndexFlatIP(res, d, fc)
for j in range(0, N, 100000): fl.add(np.asarray(X16[j:j + 100000], np.float32))
for k in (10, 100):
    fl.search(Q, k); q = []
    for _ in range(3):
        t0 = time.perf_counter(); I = fl.search(Q, k)[1]; q.append(len(Q) / (time.perf_counter() - t0))
    G = GT100[:, :k]; r = {"engine": "FAISS-GPU flat fp16", "k": k, "recall": round(float(np.mean([len(set(I[i]) & set(G[i])) / k for i in range(len(Q))])), 4), "QPS": round(float(np.mean(q)), 1), "QPS_std": round(float(np.std(q)), 1)}
    print(json.dumps(r), flush=True); json.dump(r, open(f"{FANAR}/results/rev_faiss_k{k}.json", "w"))
