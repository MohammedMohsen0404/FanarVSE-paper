"""GPU baselines on the image / large sets prepared by v16_prep.py (SigLIP ImageNet 1.28M x 768, Met DINOv2-giant x 1536).
Same protocol as other_baselines.py; IVF-RaBitQ is built from a host fp32 memmap (6 GB GPU). usage: big_baselines.py DS"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, os, sys, time, numpy as np, cupy as cp
from cuvs.neighbors import cagra
from cuvs_rabitq import IVFRaBitQ
DS = sys.argv[1]; OUT = f"{FANAR}/results/v17_baselines_{DS}.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {json.dumps({k: r[k] for k in ("engine", "cfg")}) for r in out}; REPS = 3
sync = lambda: cp.cuda.Device().synchronize()
def key(e, c): return json.dumps({"engine": e, "cfg": c})
def emit(r):
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
def timed(fn, nq):
    fn(); sync(); qps = []
    for _ in range(REPS):
        t0 = time.perf_counter(); I = fn(); sync(); qps.append(nq / (time.perf_counter() - t0))
    return I, round(float(np.mean(qps)), 1), round(float(np.std(qps)), 1)
def batched(search, Qg, B=500): return lambda: cp.concatenate([search(Qg[s:s + B]) for s in range(0, len(Qg), B)])

X16 = np.load(f"{FANAR}/data/{DS}_base_fp16.npy", mmap_mode="r"); Q = np.load(f"{FANAR}/data/{DS}_q.npy").astype(np.float32)
GT = np.load(f"{FANAR}/data/{DS}_gt.npy"); N, d = X16.shape; NQ = len(Q); NL = 4096 if N >= 1_000_000 else 1024
rec = lambda I: round(float(np.mean([len(set(np.asarray(I)[i, :10]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
Qg = cp.asarray(Q); Qh = Qg.astype(cp.float16)
# ---- CAGRA ----
Xg16 = cp.asarray(np.ascontiguousarray(X16))
for deg in (32, 64):
    cfgs = [{"graph_degree": deg, "itopk": t} for t in (32, 64, 128, 256, 512)]
    if all(key("CAGRA", c) in done for c in cfgs): continue
    sync(); t0 = time.perf_counter(); idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=deg, intermediate_graph_degree=2 * deg), Xg16); sync(); bt = round(time.perf_counter() - t0, 1)
    for c in cfgs:
        if key("CAGRA", c) in done: continue
        sp = cagra.SearchParams(itopk_size=c["itopk"])
        I, m, s = timed(batched(lambda q: cp.asarray(cagra.search(sp, idx, q, 10)[1]), Qh), NQ)
        emit({"ds": DS, "engine": "CAGRA", "cfg": c, "recall": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes": 2 * d + 4 * deg, "build_s": bt})
    del idx; cp.get_default_memory_pool().free_all_blocks()
del Xg16; cp.get_default_memory_pool().free_all_blocks()
# ---- IVF-RaBitQ (host fp32 memmap) ----
F32 = os.path.expanduser(f"~/{DS}_fp32.bin")
if not os.path.exists(F32):
    mm = np.memmap(F32, np.float32, "w+", shape=(N, d))
    for j in range(0, N, 100000): mm[j:j + 100000] = X16[j:j + 100000]
    mm.flush(); del mm
X32 = np.memmap(F32, np.float32, "r", shape=(N, d))
if N * d * 4 < 2.5e9: X32 = cp.asarray(np.asarray(X32))                                  # device build when it fits
for bits in (4, 5, 7, 9):
    mode = "QUANT4" if bits <= 5 else "QUANT8"
    cfgs = [{"bits": bits, "n_lists": NL, "n_probes": p, "mode": mode} for p in ((32, 64, 128, 256, 512, 1024) if NL == 4096 else (16, 32, 64, 128, 256, 512))]
    if all(key("IVF-RaBitQ", c) in done for c in cfgs): continue
    free0 = cp.cuda.runtime.memGetInfo()[0]
    idx = IVFRaBitQ(n_lists=NL, bits_per_dim=bits); sync(); t0 = time.perf_counter(); idx.build(X32); sync(); bt = round(time.perf_counter() - t0, 1)
    gb = round((free0 - cp.cuda.runtime.memGetInfo()[0]) / N, 1)
    for c in cfgs:
        if key("IVF-RaBitQ", c) in done: continue
        try:
            I, m, s = timed(batched(lambda q: idx.search(q, 10, c["n_probes"], mode)[0], Qg), NQ)
            emit({"ds": DS, "engine": "IVF-RaBitQ", "cfg": c, "recall": rec(cp.asnumpy(I)), "QPS": m, "QPS_std": s, "gpu_bytes_measured": gb, "gpu_bytes": round(bits * d / 8 + 12, 1), "build_s": bt})
        except Exception as e: print("rabitq failed", c, str(e)[:200], flush=True)
    idx.free(); del idx; cp.get_default_memory_pool().free_all_blocks()
print("DONE", flush=True)
