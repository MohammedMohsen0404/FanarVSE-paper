"""[reviewer round: 3 timed repeats, results to rev_v2_glove.json] Pipelined 4-bit engine on GloVe-1.18M: GPU fused 4-bit scan (codes only) || CPU exact rerank from fp16 host vectors."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, time, numpy as np, torch, h5py
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, f"{FANAR}/src")
from fused_scan import search4
dev = "cuda"; NQ = 2000; torch.manual_seed(0)
f = h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r"); X = np.array(f["train"], np.float32); Qr = np.array(f["test"][:NQ], np.float32)
mu = X[:100000].mean(0); X -= mu; X /= np.linalg.norm(X, axis=1, keepdims=True); Qr -= mu; Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
GT = np.load(f"{FANAR}/results/gt_glove_wsl.npy"); N, d = X.shape; host16 = torch.from_numpy(X).half()
P = torch.linalg.qr(torch.randn(d, d, device=dev))[0]; H = 64; HB = (d + 1) // 2; lim = 3.0 / d ** 0.5; L = 15; a = 2 * lim / L; b = -lim
codes = torch.empty(N, HB, dtype=torch.uint8, device=dev); scale = torch.empty(N, dtype=torch.float16, device=dev)
for j in range(0, N, 200000):
    Z = torch.from_numpy(X[j:j + 200000]).to(dev) @ P; nz = Z.norm(dim=1); c = torch.clamp(((Z / nz[:, None] - b) / a).round(), 0, L).to(torch.uint8)
    codes[j:j + len(c)] = (c[:, :HB] << 4) | c[:, HB:2 * HB]; scale[j:j + len(c)] = nz.half()
Qg = torch.from_numpy(Qr).to(dev); QZ = Qg @ P; Qc = torch.from_numpy(Qr)
Qh = torch.zeros(NQ, 2 * H, dtype=torch.float16, device=dev); Qh[:, :HB] = QZ[:, :HB].half(); Qh[:, H:H + (d - HB)] = QZ[:, HB:].half()
def rerank(Qb, cand, k=10):
    Xc = host16.index_select(0, cand.reshape(-1)).reshape(cand.shape[0], cand.shape[1], d).float()
    return cand.gather(1, torch.einsum("bkd,bd->bk", Xc, Qb).topk(k, 1).indices).numpy()
def run(kp, batch, workers=4):
    futs = []
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, NQ, batch):
            cand = search4(Qh[s:s + batch], codes, scale, a, b, H, kp).cpu(); futs.append(ex.submit(rerank, Qc[s:s + batch], cand))
        return np.vstack([f.result() for f in futs])
out = []
for kp in (10, 20, 50):
    for batch in (500, 1000):
        run(kp, batch); torch.cuda.synchronize(); q = []
        for _ in range(3):
            t0 = time.perf_counter(); I = run(kp, batch); q.append(NQ / (time.perf_counter() - t0))
        out.append({"ds": "glove", "kp": kp, "batch": batch, "gpu_bytes": HB + 2, "recall@10": round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4), "QPS": round(float(np.mean(q)), 1), "QPS_std": round(float(np.std(q)), 1)})
        print(json.dumps(out[-1]), flush=True); json.dump(out, open(f"{FANAR}/results/rev_v2_glove.json", "w"), indent=1)
