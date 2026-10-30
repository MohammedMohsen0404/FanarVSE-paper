"""GPU baselines on the two extra embedding sets (98k OpenAI-3-large 1536-d, 98k arXiv InstructorXL 768-d), same protocol as
v13_other.py: 2,000 queries, k=10, batch 500, 3 timed repeats after one warm-up. CAGRA, IVF-RaBitQ (cuVS 26.10), IVF-PQ + refine, fp16 flat."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, os, time, numpy as np, cupy as cp, faiss
from cuvs.neighbors import cagra, ivf_pq, refine
from cuvs_rabitq import IVFRaBitQ
OUT = f"{FANAR}/results/v15_other_baselines.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {json.dumps({k: r[k] for k in ("ds", "engine", "cfg")}) for r in out}; REPS = 3
sync = lambda: cp.cuda.Device().synchronize()
def key(ds, e, c): return json.dumps({"ds": ds, "engine": e, "cfg": c})
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
def timed(fn, nq):
    fn(); sync(); qps = []
    for _ in range(REPS):
        t0 = time.perf_counter(); I = fn(); sync(); qps.append(nq / (time.perf_counter() - t0))
    return I, round(float(np.mean(qps)), 1), round(float(np.std(qps)), 1)
def batched(search, Qg, B=500): return lambda: cp.concatenate([search(Qg[s:s + B]) for s in range(0, len(Qg), B)])

for ds in ("oai3", "arxiv"):
    X16 = np.load(f"{FANAR}/data/{ds}_base_fp16.npy"); Q = np.load(f"{FANAR}/data/{ds}_q.npy").astype(np.float32)
    GT = np.load(f"{FANAR}/data/{ds}_gt.npy"); N, d = X16.shape; NQ = len(Q)
    rec = lambda I: round(float(np.mean([len(set(np.asarray(I)[i, :10]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
    Qg = cp.asarray(Q); Qh = Qg.astype(cp.float16); Xg16 = cp.asarray(X16); Xg32 = Xg16.astype(cp.float32)
    # ---- fp16 flat (FAISS-GPU) ----
    if key(ds, "FAISS-GPU flat fp16", {}) not in done:
        res = faiss.StandardGpuResources(); fc = faiss.GpuIndexFlatConfig(); fc.useFloat16 = True; fl = faiss.GpuIndexFlatIP(res, d, fc); fl.add(X16.astype(np.float32))
        I, m, s = timed(lambda: fl.search(Q, 10)[1], NQ)
        emit({"ds": ds, "engine": "FAISS-GPU flat fp16", "cfg": {}, "recall": rec(I), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d}); del fl, res
    # ---- CAGRA ----
    for deg in (32, 64):
        cfgs = [{"graph_degree": deg, "itopk": t} for t in (32, 64, 128, 256, 512)]
        if all(key(ds, "CAGRA", c) in done for c in cfgs): continue
        sync(); t0 = time.perf_counter(); idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg), Xg16); sync(); bt = round(time.perf_counter() - t0, 1)
        for c in cfgs:
            if key(ds, "CAGRA", c) in done: continue
            sp = cagra.SearchParams(itopk_size=c["itopk"])
            I, m, s = timed(batched(lambda q: cp.asarray(cagra.search(sp, idx, q, 10)[1]), Qh), NQ)
            emit({"ds": ds, "engine": "CAGRA", "cfg": c, "recall": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d + 4 * deg, "build_s": bt})
        del idx; cp.get_default_memory_pool().free_all_blocks()
    # ---- IVF-RaBitQ (512 lists, as FanarVSE's 512 cells) ----
    for bits in (3, 4, 5, 7, 9):
        mode = "QUANT4" if bits <= 5 else "QUANT8"
        cfgs = [{"bits": bits, "n_lists": 512, "n_probes": p, "mode": mode} for p in (8, 16, 32, 64, 128, 256, 512)]
        if all(key(ds, "IVF-RaBitQ", c) in done for c in cfgs): continue
        free0 = cp.cuda.runtime.memGetInfo()[0]
        idx = IVFRaBitQ(n_lists=512, bits_per_dim=bits); sync(); t0 = time.perf_counter(); idx.build(Xg32); sync(); bt = round(time.perf_counter() - t0, 1)
        gb = round((free0 - cp.cuda.runtime.memGetInfo()[0]) / N, 1)
        for c in cfgs:
            if key(ds, "IVF-RaBitQ", c) in done: continue
            try:
                I, m, s = timed(batched(lambda q: idx.search(q, 10, c["n_probes"], mode)[0], Qg), NQ)
                emit({"ds": ds, "engine": "IVF-RaBitQ", "cfg": c, "recall": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes_measured": gb, "gpu_bytes": round(bits * d / 8 + 12, 1), "build_s": bt})
            except Exception as e: print("rabitq failed", c, str(e)[:200], flush=True)
        idx.free(); del idx; cp.get_default_memory_pool().free_all_blocks()
    # ---- IVF-PQ + exact refine (fp32 vectors kept on the GPU for the refine) ----
    for pq_dim in (d // 8, d // 4):
        cfgs = [{"pq_dim": pq_dim, "n_lists": 512, "n_probes": p, "refine": r} for p in (16, 32, 64) for r in (40, 100)]
        if all(key(ds, "IVF-PQ+refine", c) in done for c in cfgs): continue
        sync(); t0 = time.perf_counter(); idx = ivf_pq.build(ivf_pq.IndexParams(n_lists=512, metric="inner_product", pq_dim=pq_dim, pq_bits=8), Xg32); sync(); bt = round(time.perf_counter() - t0, 1)
        for c in cfgs:
            if key(ds, "IVF-PQ+refine", c) in done: continue
            sp = ivf_pq.SearchParams(n_probes=c["n_probes"])
            def f(q, c=c, sp=sp):
                _, cand = ivf_pq.search(sp, idx, q, c["refine"]); return cp.asarray(refine(Xg32, q, cand, 10, metric="inner_product")[1])
            I, m, s = timed(batched(f, Qg), NQ)
            emit({"ds": ds, "engine": "IVF-PQ+refine", "cfg": c, "recall": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": pq_dim + 4 * d, "build_s": bt})
        del idx; cp.get_default_memory_pool().free_all_blocks()
    del Xg16, Xg32, Qg, Qh; cp.get_default_memory_pool().free_all_blocks()
print("DONE", flush=True)
