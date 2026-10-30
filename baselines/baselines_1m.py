"""GPU baselines on the 998k OpenAI set (fp16 dataset, same queries/GT as FanarVSE): FAISS-GPU flat fp16, CAGRA (fp16), cuVS IVF-PQ+refine."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, time, numpy as np, cupy as cp, faiss
from cuvs.neighbors import cagra, ivf_pq
from cuvs.neighbors.refine import refine
X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT = np.load(f"{FANAR}/data/oai1m_gt.npy")
N, d = X.shape; NQ = len(Q); out = []
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
def emit(r): print(json.dumps(r), flush=True); out.append(r)
# FAISS-GPU flat fp16
try:
    res = faiss.StandardGpuResources(); cfg = faiss.GpuIndexFlatConfig(); cfg.useFloat16 = True
    fl = faiss.GpuIndexFlatIP(res, d, cfg)
    for j in range(0, N, 100000): fl.add(np.asarray(X[j:j + 100000], np.float32))
    fl.search(Q[:100], 10); t0 = time.perf_counter(); _, I = fl.search(Q, 10); dt = time.perf_counter() - t0
    emit({"engine": "FAISS-GPU Flat fp16", "N": N, "recall@10": rec(I), "QPS": round(NQ / dt, 1), "gpu_bytes": 2 * d})
    del fl, res
except Exception as e: print("flat fp16 failed:", str(e)[:200], flush=True)
Xg = cp.asarray(np.ascontiguousarray(X)); Qg = cp.asarray(Q.astype(np.float16)); Qf = cp.asarray(Q)
# CAGRA on fp16 dataset
try:
    ip = cagra.IndexParams(metric="inner_product", graph_degree=32, intermediate_graph_degree=64)
    cp.cuda.Device().synchronize(); t0 = time.perf_counter(); idx = cagra.build(ip, Xg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
    for itopk in (32, 64, 128, 256):
        sp = cagra.SearchParams(itopk_size=itopk); cagra.search(sp, idx, Qg[:100], 10); cp.cuda.Device().synchronize()
        t0 = time.perf_counter(); _, I = cagra.search(sp, idx, Qg, 10); cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0
        emit({"engine": "CAGRA deg32 (fp16 dataset)", "N": N, "itopk": itopk, "recall@10": rec(cp.asnumpy(cp.asarray(I))), "QPS": round(NQ / dt, 1), "gpu_bytes": 2 * d + 128, "build_s": round(bt, 1)})
    del idx
except Exception as e: print("CAGRA failed:", str(e)[:300], flush=True)
cp.get_default_memory_pool().free_all_blocks()
# cuVS IVF-PQ (+refine on fp16 dataset)
for pq_dim in (96, 48):
    try:
        ipp = ivf_pq.IndexParams(n_lists=2048, metric="inner_product", pq_dim=pq_dim, pq_bits=8)
        t0 = time.perf_counter(); pidx = ivf_pq.build(ipp, Xg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
        for npb in (64, 128):
            for kp in (40, 100):
                spp = ivf_pq.SearchParams(n_probes=npb); ivf_pq.search(spp, pidx, Qg[:100], kp); cp.cuda.Device().synchronize(); t0 = time.perf_counter()
                _, C = ivf_pq.search(spp, pidx, Qg, kp); _, I = refine(Xg, Qg, cp.asarray(C), k=10, metric="inner_product"); cp.cuda.Device().synchronize(); dt = time.perf_counter() - t0
                emit({"engine": f"cuVS IVF-PQ pq{pq_dim} +refine{kp}", "N": N, "nprobe": npb, "recall@10": rec(cp.asnumpy(cp.asarray(I))), "QPS": round(NQ / dt, 1), "gpu_bytes": pq_dim + 8, "build_s": round(bt, 1)})
        del pidx; cp.get_default_memory_pool().free_all_blocks()
    except Exception as e: print("ivf_pq", pq_dim, "failed:", str(e)[:200], flush=True)
json.dump(out, open(f"{FANAR}/results/baselines_1m.json", "w"), indent=1)
