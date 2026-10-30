"""Paper rerun on the 998k OpenAI set: GPU baselines with 3 timed repeats each (mean, std), CAGRA extended to itopk 512/1024."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, time, numpy as np, cupy as cp, faiss
from cuvs.neighbors import cagra, ivf_pq
from cuvs.neighbors.refine import refine
X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT = np.load(f"{FANAR}/data/oai1m_gt.npy")
N, d = X.shape; NQ = len(Q); out = []; REPS = 3
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
def emit(r): print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(f"{FANAR}/results/baselines_1m_rep.json", "w"), indent=1)
def timed(fn):
    fn(); cp.cuda.Device().synchronize(); qps = []
    for _ in range(REPS):
        t0 = time.perf_counter(); I = fn(); cp.cuda.Device().synchronize(); qps.append(NQ / (time.perf_counter() - t0))
    return I, round(float(np.mean(qps)), 1), round(float(np.std(qps)), 1)
# FAISS-GPU flat fp16
try:
    res = faiss.StandardGpuResources(); cfg = faiss.GpuIndexFlatConfig(); cfg.useFloat16 = True
    fl = faiss.GpuIndexFlatIP(res, d, cfg)
    for j in range(0, N, 100000): fl.add(np.asarray(X[j:j + 100000], np.float32))
    I, m, s = timed(lambda: fl.search(Q, 10)[1])
    emit({"engine": "FAISS-GPU Flat fp16", "recall@10": rec(I), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d})
    del fl, res
except Exception as e: print("flat fp16 failed:", str(e)[:200], flush=True)
Xg = cp.asarray(np.ascontiguousarray(X)); Qg = cp.asarray(Q.astype(np.float16))
# CAGRA on fp16 dataset, extended itopk range
try:
    ip = cagra.IndexParams(metric="inner_product", graph_degree=32, intermediate_graph_degree=64)
    cp.cuda.Device().synchronize(); t0 = time.perf_counter(); idx = cagra.build(ip, Xg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
    for itopk in (64, 128, 256, 512, 1024):
        try:
            sp = cagra.SearchParams(itopk_size=itopk)
            I, m, s = timed(lambda: cp.asarray(cagra.search(sp, idx, Qg, 10)[1]))
            emit({"engine": "CAGRA deg32 (fp16 dataset)", "itopk": itopk, "recall@10": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d + 128, "build_s": round(bt, 1)})
        except Exception as e: print("CAGRA itopk", itopk, "failed:", str(e)[:200], flush=True)
    # wider graph for the high-recall end
    del idx; cp.get_default_memory_pool().free_all_blocks()
    ip = cagra.IndexParams(metric="inner_product", graph_degree=64, intermediate_graph_degree=128)
    t0 = time.perf_counter(); idx = cagra.build(ip, Xg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
    for itopk in (256, 512, 1024):
        try:
            sp = cagra.SearchParams(itopk_size=itopk)
            I, m, s = timed(lambda: cp.asarray(cagra.search(sp, idx, Qg, 10)[1]))
            emit({"engine": "CAGRA deg64 (fp16 dataset)", "itopk": itopk, "recall@10": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d + 256, "build_s": round(bt, 1)})
        except Exception as e: print("CAGRA64 itopk", itopk, "failed:", str(e)[:200], flush=True)
    del idx
except Exception as e: print("CAGRA failed:", str(e)[:300], flush=True)
cp.get_default_memory_pool().free_all_blocks()
# cuVS IVF-PQ pq96 + refine (best config of the earlier sweep)
try:
    ipp = ivf_pq.IndexParams(n_lists=2048, metric="inner_product", pq_dim=96, pq_bits=8)
    t0 = time.perf_counter(); pidx = ivf_pq.build(ipp, Xg); cp.cuda.Device().synchronize(); bt = time.perf_counter() - t0
    for npb, kp in ((128, 100), (256, 200)):
        spp = ivf_pq.SearchParams(n_probes=npb)
        def f():
            _, C = ivf_pq.search(spp, pidx, Qg, kp); return cp.asarray(refine(Xg, Qg, cp.asarray(C), k=10, metric="inner_product")[1])
        I, m, s = timed(f)
        emit({"engine": f"cuVS IVF-PQ pq96 +refine{kp}", "nprobe": npb, "recall@10": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 104, "build_s": round(bt, 1)})
except Exception as e: print("ivf_pq failed:", str(e)[:200], flush=True)
print("DONE", flush=True)
