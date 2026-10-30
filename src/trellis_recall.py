"""Recall on OpenAI-1536 (229k): TurboQuant (rotation + Lloyd-Max per coordinate) vs trellis (rotation + bitshift trellis)
at the same bits/dim, same unbiased estimator est = <q,c> + |r| <q', y>/(|y| <o, y/|y|>), brute force over all vectors.
Reports recall@10 from the estimator alone and after exact rerank of top-40."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from trellis import viterbi, make_table, lloyd_max
import pyarrow.parquet as pq_
dev = "cuda"; torch.manual_seed(0); NQ = 2000; SEG = 256
files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
T_, Qn = np.ascontiguousarray(X[:-NQ]), np.ascontiguousarray(X[-NQ:]); del X
GT = np.load(f"{FANAR}/results/gt_dbp_wsl.npy"); N, d = T_.shape; host = torch.from_numpy(T_); Q = torch.from_numpy(Qn).to(dev)
c = host[:50000].to(dev).mean(0); P = torch.linalg.qr(torch.randn(d, d, device=dev))[0]
QR = Q @ P; QC = Q @ c; sd = d ** -0.5
def encode(kind, k, L=10):
    Y = torch.empty(N, d, dtype=torch.float16, device=dev); nr = torch.empty(N, device=dev); fip = torch.empty(N, device=dev)
    if kind == "trellis":
        Tb = make_table(L); G = torch.randn(512, SEG, device=dev); best = None
        for a in np.linspace(0.6, 1.3, 15):
            m = float(((G - viterbi(G, Tb, L, k, float(a))) ** 2).mean())
            if best is None or m < best[0]: best = (m, float(a))
        alpha = best[1]
    else:
        cb = lloyd_max(k); thr = (cb[1:] + cb[:-1]) / 2
    for j in range(0, N, 4096):
        R = host[j:j + 4096].to(dev) - c; n_ = R.norm(dim=1); O = (R / n_[:, None]) @ P; Z = O / sd          # ~N(0,1) coords
        if kind == "trellis":
            Zs = Z.reshape(-1, SEG); Yh = torch.cat([viterbi(Zs[i:i + 2048], Tb, L, k, alpha) for i in range(0, len(Zs), 2048)]).reshape(Z.shape)
        else: Yh = cb[torch.bucketize(Z, thr)]
        yn = Yh.norm(dim=1); Y[j:j + len(Z)] = (Yh / yn[:, None]).half(); nr[j:j + len(Z)] = n_; fip[j:j + len(Z)] = (O * Yh).sum(1) / yn
    return Y, nr, fip
res = []
for k in (1, 2):
    for kind in ("turboquant", "trellis"):
        t0 = time.perf_counter(); Y, nr, fip = encode(kind, k); torch.cuda.synchronize(); et = time.perf_counter() - t0
        scale = nr / fip; I40 = []
        for s in range(0, NQ, 250):
            best = None
            for j in range(0, N, 50000):
                est = (QR[s:s + 250].half() @ Y[j:j + 50000].T).float() * scale[j:j + 50000][None] + QC[s:s + 250, None]
                v, ix = est.topk(40, 1); ix = ix + j
                if best is None: best = (v, ix)
                else:
                    vv = torch.cat([best[0], v], 1); ii = torch.cat([best[1], ix], 1); t = vv.topk(40, 1); best = (t.values, ii.gather(1, t.indices))
            I40.append(best[1].cpu())
        I40 = torch.cat(I40).numpy()
        ex = np.einsum("bkd,bd->bk", T_[I40.reshape(-1)].reshape(NQ, 40, d), Qn); Ir = np.take_along_axis(I40, np.argsort(-ex, 1)[:, :10], 1)
        rc = lambda I: round(float(np.mean([len(set(I[i][:10]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
        r = {"code": kind, "bits_per_dim": k, "mean_cos": round(float(fip.mean()), 4), "recall@10_estimator_only": rc(I40), "recall@10_rerank40": rc(Ir),
             "bytes_per_vec": int(d * k / 8 + (d // SEG) * 10 / 8 + 4) if kind == "trellis" else int(d * k / 8 + 4), "encode_s": round(et, 1)}
        print(json.dumps(r), flush=True); res.append(r); del Y
json.dump(res, open(f"{FANAR}/results/trellis_recall_229k.json", "w"), indent=1)
