"""T1 practical: how few stage-1 bits keep the true top-10 inside the top-K candidates? (OpenAI 229k)
Codes on the top-r PCA dims (random-rotated): sign (1 bit/dim) vs trellis (0.5 bit/dim via V=2, 1 bit/dim via V=1).
Candidate recall = fraction of true top-10 inside top-K by the unbiased estimator. Identification bound ~ 143 bits/vector."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from trellis import viterbi, viterbi_v, make_table, make_table_v
import pyarrow.parquet as pq_
dev = "cuda"; torch.manual_seed(0); NQ = 1000
files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
T = torch.from_numpy(X[:-2000]).to(dev); Q = torch.from_numpy(X[-2000:][:NQ]).to(dev); del X; N = len(T)
GT = (Q @ T.T).topk(10, 1).indices
c = T[:50000].mean(0); ev, V = torch.linalg.eigh(torch.cov((T[:50000] - c).T)); V = V.flip(1)
res = []
def cand_recall(Y, nr, fip, Wq, Ks=(100, 200, 400)):
    est = (Q @ Wq) @ (Y / Y.norm(dim=1, keepdim=True)).T * (nr / fip)[None] + (Q @ c)[:, None]
    out = {}
    for K in Ks:
        top = est.topk(K, 1).indices; out[f"cand_recall@K{K}"] = round(float((GT[:, :, None] == top[:, None, :]).any(2).float().mean()), 4)
    return out
for r in (128, 192, 256, 384):
    W = V[:, :r] @ torch.linalg.qr(torch.randn(r, r, device=dev))[0]
    Z = (T - c) @ W; nr = Z.norm(dim=1); O = Z / nr[:, None]; Zs = O * r ** 0.5
    # sign code, r bits
    Y = torch.sign(O); fip = (O * Y).sum(1) / Y.norm(dim=1)
    res.append({"code": "sign", "pca_dims": r, "bits": r, **cand_recall(Y, nr, fip, W)}); print(json.dumps(res[-1]), flush=True)
    # trellis at 1 bit/dim (V=1,k=1) and 0.5 bit/dim (V=2,k=1)
    for V_, label in ((1, "trellis 1b/dim"), (2, "trellis 0.5b/dim")):
        L = 10; Tb = make_table(L) if V_ == 1 else make_table_v(L, 2)
        enc = (lambda Xs, a: viterbi(Xs, Tb, L, 1, a)) if V_ == 1 else (lambda Xs, a: viterbi_v(Xs, Tb, L, 1, a))
        G = torch.randn(512, r, device=dev); best = None
        for a in np.linspace(0.4, 1.3, 12):
            m = float(((G - enc(G, float(a))) ** 2).mean())
            if best is None or m < best[0]: best = (m, float(a))
        t0 = time.perf_counter()
        Y = torch.cat([enc(Zs[i:i + 2048], best[1]) for i in range(0, N, 2048)]); fip = (O * Y).sum(1) / Y.norm(dim=1)
        bits = r // V_ + L
        res.append({"code": label, "pca_dims": r, "bits": bits, "mean_cos": round(float(fip.mean()), 4), "encode_s": round(time.perf_counter() - t0, 1), **cand_recall(Y, nr, fip, W)})
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(f"{FANAR}/results/t1_identification.json", "w"), indent=1)
