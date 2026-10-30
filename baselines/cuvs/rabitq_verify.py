"""Fresh-process check of IVF-RaBitQ at high bit widths: GPU memory (index freed between builds), n_lists 1024 vs 4096, finer probe sweep.
usage: rabitq_verify.py bits n_lists"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, os, time, numpy as np, cupy as cp
from cuvs_rabitq import IVFRaBitQ
bits, nl = int(sys.argv[1]), int(sys.argv[2])
Q = np.load(f"{FANAR}/data/oai1m_q.npy"); GT = np.load(f"{FANAR}/data/oai1m_gt.npy"); N, d = 998000, 1536
N = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r").shape[0]
X32 = np.memmap(os.path.expanduser("~/oai1m_fp32.bin"), np.float32, "r", shape=(N, d)); Qg = cp.asarray(Q); NQ = len(Q)
OUT = f"{FANAR}/results/rev_rabitq_verify.json"; out = json.load(open(OUT)) if os.path.exists(OUT) else []
rec = lambda I: round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
free0 = cp.cuda.runtime.memGetInfo()[0]; idx = IVFRaBitQ(n_lists=nl, bits_per_dim=bits)
t0 = time.perf_counter(); idx.build(X32); cp.cuda.Device().synchronize(); bt = round(time.perf_counter() - t0, 1)
gb = round((free0 - cp.cuda.runtime.memGetInfo()[0]) / N, 1); print("bits", bits, "n_lists", nl, "build", bt, "s, GPU B/vec", gb, flush=True)
mode = "QUANT4" if bits <= 5 else "QUANT8"
for npb in ((64, 96, 128, 160, 192, 256, 384) if nl == 1024 else (128, 256, 384, 512, 768, 1024)):
    f = lambda: cp.concatenate([idx.search(Qg[s:s + 500], 10, npb, mode)[0] for s in range(0, NQ, 500)])
    f(); cp.cuda.Device().synchronize(); q = []
    for _ in range(3):
        t0 = time.perf_counter(); I = f(); cp.cuda.Device().synchronize(); q.append(NQ / (time.perf_counter() - t0))
    r = {"engine": "IVF-RaBitQ", "bits": bits, "n_lists": nl, "n_probes": npb, "mode": mode, "recall": rec(cp.asnumpy(I)), "QPS": round(float(np.mean(q)), 1), "QPS_std": round(float(np.std(q)), 1),
         "gpu_bytes_measured": gb, "gpu_bytes_payload": round(d * bits / 8, 1), "build_s": bt}
    print(json.dumps(r), flush=True); out.append(r); json.dump(out, open(OUT, "w"), indent=1)
idx.free()
