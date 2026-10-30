"""N1: Bussgang / Van Vleck LMMSE decoding of 1-bit codes (zero extra storage).
Codes y = sign(z) of rotated unit residuals z. Standard (RaBitQ) estimator uses y/|y| scaled by <o,o_bar>.
LMMSE: z_hat = A y with A = C_zy C_yy^-1 (estimated on a training sample; Bussgang/arcsine law in closed form for Gaussians).
Scoring: <q', z_hat> = <A^T q', y>  -> only the QUERY is transformed (one d x d matvec), codes unchanged.
Tested on full-dim (1536) and PCA-truncated (256) 1-bit codes, OpenAI 229k, estimator-only and rerank recall."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import glob, json, numpy as np, torch, pyarrow.parquet as pq_
dev = "cuda"; torch.manual_seed(0); NQ = 1000
files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
T = torch.from_numpy(X[:-2000]).to(dev); Q = torch.from_numpy(X[-2000:][:NQ]).to(dev); del X; N, d = T.shape
GT = (Q @ T.T).topk(10, 1).indices
c = T[:50000].mean(0); ev, V = torch.linalg.eigh(torch.cov((T[:50000] - c).T)); V = V.flip(1)
def recall(est, Ks=(10, 40, 100)):
    out = {}
    for K in Ks:
        top = est.topk(K, 1).indices
        if K == 10: out["recall@10_estimator"] = round(float((GT[:, :, None] == top[:, None, :]).any(2).float().mean()), 4)
        else:
            ex = torch.gather(Q @ T.T, 1, top); rr = top.gather(1, ex.topk(10, 1).indices)
            out[f"recall@10_rerank{K}"] = round(float((GT[:, :, None] == rr[:, None, :]).any(2).float().mean()), 4)
    return out
res = []
for r, name in ((d, "full-1536 cell-residual"),):
    W = (V[:, :r] if r < d else torch.eye(d, device=dev)) @ torch.linalg.qr(torch.randn(r, r, device=dev))[0]
    Cc = T[torch.randperm(N, device=dev)[:1024]].clone()
    for _ in range(10):
        a_ = (T @ Cc.T).argmax(1); Cc = torch.nn.functional.normalize(torch.zeros_like(Cc).index_add_(0, a_, T) + 1e-9, dim=1)
    a_ = (T @ Cc.T).argmax(1)
    Z = (T - Cc[a_]) @ W; nr = Z.norm(dim=1); O = Z / nr[:, None]; del Z; Y = torch.sign(O).half(); torch.cuda.empty_cache()
    Qr = Q @ W; qc = (Q @ Cc.T)[:, a_]
    # standard RaBitQ-style estimator
    fip = (O * Y.float()).sum(1) / r ** 0.5; est_std = (Qr.half() @ Y.T).float() / r ** 0.5 * (nr / fip)[None] + qc
    res.append({"codes": name, "decoder": "standard (RaBitQ/TurboQuant)", **recall(est_std)}); print(json.dumps(res[-1]), flush=True)
    # Bussgang LMMSE decoder from a training sample
    tr = torch.randperm(N, device=dev)[:100000]; Ot, Yt = O[tr], Y[tr].float()
    Czy = Ot.T @ Yt / len(tr); Cyy = Yt.T @ Yt / len(tr)
    A = Czy @ torch.linalg.inv(Cyy + 1e-3 * torch.eye(r, device=dev))
    sm = torch.randperm(N, device=dev)[:20000]; Ys = Y[sm].float(); Zh = Ys @ A.T
    est_lm = ((Qr @ A).half() @ Y.T).float() * nr[None] + qc               # raw LMMSE (biased per vector)
    oz = torch.cat([(O[i:i + 20000] * (Y[i:i + 20000].float() @ A.T)).sum(1) for i in range(0, N, 20000)])   # <o, z_hat> per vector (stored factor)
    est_lmu = ((Qr @ A).half() @ Y.T).float() * (nr / oz)[None] + qc      # Bussgang direction + unbiased per-vector scale
    mse_std = float(((O[sm] - Ys / r ** 0.5 * fip[sm, None]) ** 2).sum(1).mean()); mse_lm = float(((O[sm] - Zh) ** 2).sum(1).mean())
    res.append({"codes": name, "decoder": "Bussgang LMMSE raw", "mse_std": round(mse_std, 4), "mse_lmmse": round(mse_lm, 4), **recall(est_lm)}); print(json.dumps(res[-1]), flush=True)
    res.append({"codes": name, "decoder": "Bussgang direction + unbiased scale", **recall(est_lmu)}); print(json.dumps(res[-1]), flush=True)
json.dump(res, open(f"{FANAR}/results/n1_bussgang_cellres.json", "w"), indent=1)
