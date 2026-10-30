"""Reviewer round, theory checks on OpenAI-ada-1536 (first 228k vectors of the 998k set, centred and normalised as in Sec. 3).
usage: rev_theory.py {ee|ident}
  ee     error covariance E[eps eps^T] of the decoded unit residual (eps = o_hat - o, o_hat = what the estimator multiplies q by)
         for: sign codes + standard (RaBitQ) decoder, sign + Bussgang LMMSE (raw / unbiased scale), 1-bit trellis (PCA-256),
         4-bit Lloyd-Max (TurboQuant-style). Reports MSE = tr(E), ranking error tr(Sigma_q E), the isotropic reference
         tr(Sigma_q) tr(E) / d, their ratio rho, and spectrum shape (lambda_max / mean, participation ratio / d, share of tr(E)
         in the top-50 data principal directions).
  ident  identification-rate bound: tau (cosine of the k-th neighbour) quantiles, effective dimension under several definitions,
         bits = d_eff * log2(1/tau)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch, faiss
sys.path.insert(0, f"{FANAR}/src")
from trellis import make_table, viterbi
from trellis_scan import viterbi_paths
dev = "cuda"; torch.manual_seed(0); EXP = sys.argv[1]
X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r")
T = torch.from_numpy(np.ascontiguousarray(X[:228000])).to(dev).float(); Q = torch.from_numpy(np.load(f"{FANAR}/data/oai1m_q.npy")).to(dev)
mu = T[:50000].mean(0); T = torch.nn.functional.normalize(T - mu, dim=1); Qc = torch.nn.functional.normalize(Q - mu, dim=1); N, d = T.shape
res = []; OUT = f"{FANAR}/results/rev_theory_{EXP}.json"
def emit(r):
    print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(OUT, "w"), indent=1)
ev, V = torch.linalg.eigh(torch.cov(T[:50000].T)); ev, V = ev.flip(0), V.flip(1)

if EXP == "ee":
    Sq = Qc.T @ Qc / len(Qc)                                                              # query second moment (queries in the same centred frame)
    NQE, KN = 400, 50                                                                     # near-threshold pairs: each query with its true top-50
    top = torch.cat([(Qc[i:i + 200] @ T.T).topk(KN, 1).indices for i in range(0, NQE, 200)]); Qrep = Qc[:NQE].repeat_interleave(KN, 0)
    def report(name, dec, space_dim, proj=None):
        """dec: unit residuals (n, D) -> decoded o_hat (n, D) as used by the estimator <q, o_hat>."""
        O = Tsp[S]; eps = dec(O) - O
        E = eps.T @ eps / len(eps); lam = torch.linalg.eigvalsh(E).clamp(min=0); trE = float(E.trace())
        rank_err = float((Sq * E).sum()); iso = float(Sq.trace()) * trE / space_dim
        top50 = float((V[:, :50].T @ E @ V[:, :50]).trace()) / trE
        Oc = Tsp[top.flatten()]; Qp = Qrep if proj is None else Qrep @ proj
        err = ((dec(Oc) - Oc) * Qp).sum(1).reshape(NQE, KN); true = (Oc * Qp).sum(1).reshape(NQE, KN); est = true + err
        within = float(err.var(1).mean())                                                 # error variance across one query's candidates (offsets cancel)
        r10 = float((est.topk(10, 1).indices < 10).float().mean())                        # true top-10 recovered among the true top-50
        emit({"decoder": name, "dim": space_dim, "mse_trE": round(trE, 5), "rank_err_tr(SqE)": round(rank_err, 7), "iso_ref": round(iso, 7),
              "rho=rank_err/iso": round(rank_err / iso, 3), "lam_max/lam_mean": round(float(lam.max() / lam[-space_dim:].mean()), 2),
              "PR/dim": round(float(lam.sum() ** 2 / (lam ** 2).sum()) / space_dim, 3), "share_trE_top50_PCs": round(top50, 4), "top50_reference": round(50 / space_dim, 4),
              "near_threshold_err_var": round(within, 8), "recall10_within_top50": round(r10, 4)})
    S = torch.randperm(N, device=dev)[:20000]; Tsp = T
    # --- 1-bit sign codes on all 1536 dims after a Haar rotation
    P = torch.linalg.qr(torch.randn(d, d, device=dev))[0]
    def d_sign(O):
        Y = torch.sign(O @ P); c = ((O @ P) * Y).sum(1) / d ** 0.5; return (Y / d ** 0.5) @ P.T / c[:, None]
    report("sign + standard (unbiased, Thm 1)", d_sign, d)
    tr = torch.randperm(N, device=dev)[:100000]; Ztr = T[tr] @ P; Ytr = torch.sign(Ztr)
    A = (Ztr.T @ Ytr / len(tr)) @ torch.linalg.inv(Ytr.T @ Ytr / len(tr) + 1e-3 * torch.eye(d, device=dev)); del Ztr, Ytr
    d_lm = lambda O: (torch.sign(O @ P) @ A.T) @ P.T
    report("sign + Bussgang LMMSE (raw)", d_lm, d)
    def d_lmu(O):
        o = d_lm(O); return o / (O * o).sum(1, keepdim=True)
    report("sign + Bussgang direction, unbiased scale", d_lmu, d)
    # --- 4-bit Lloyd-Max after rotation (stage 2 of FanarVSE)
    g = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32); km = faiss.Kmeans(1, 16, niter=60, seed=1); km.train(g)
    cb = torch.from_numpy(np.sort(km.centroids[:, 0])).to(dev) * d ** -0.5; thr = (cb[1:] + cb[:-1]) / 2
    def d_lloyd(O):
        F = O @ P; Yf = cb[torch.bucketize(F, thr)]; ynf = Yf.norm(dim=1); cf = (F * Yf).sum(1) / ynf; return (Yf / ynf[:, None]) @ P.T / cf[:, None]
    report("4-bit Lloyd-Max + standard", d_lloyd, d)
    # --- 1-bit trellis vs sign on the top-256 PCA dims (stage 1 of FanarVSE); errors measured inside the 256-dim subspace
    R = 256; W = V[:, :R] @ torch.linalg.qr(torch.randn(R, R, device=dev))[0]
    Sq = W.T @ Sq @ W; V = W.T @ V; Zp = T @ W; Tsp = Zp / Zp.norm(dim=1, keepdim=True); del Zp     # unit residuals inside the subspace
    def d_psign(O):
        Ys = torch.sign(O); return Ys / R ** 0.5 / ((O * Ys).sum(1, keepdim=True) / R ** 0.5)
    report("PCA-256 sign + standard", d_psign, R, proj=W)
    Tt = make_table(10); Gs = torch.randn(512, R, device=dev); best = min(((float(((Gs - viterbi(Gs, Tt, 10, 1, float(a))) ** 2).mean()), float(a)) for a in np.linspace(0.6, 1.3, 15)))
    tab = best[1] * Tt
    def d_trellis(O):
        path = torch.cat([viterbi_paths(O[i:i + 2048] * R ** 0.5, Tt, 10, best[1]) for i in range(0, len(O), 2048)])
        Yt = tab[path]; yt = Yt / Yt.norm(dim=1, keepdim=True); return yt / (O * yt).sum(1, keepdim=True)
    report("PCA-256 trellis L=10 + standard", d_trellis, R, proj=W)
    Ztr = Tsp[tr[:50000]]; Ytr = torch.sign(Ztr); A2 = (Ztr.T @ Ytr) @ torch.linalg.inv(Ytr.T @ Ytr + 1e-3 * torch.eye(R, device=dev))
    report("PCA-256 sign + Bussgang LMMSE (raw)", lambda O: torch.sign(O) @ A2.T, R, proj=W)

elif EXP == "ident":
    lam = ev.clamp(min=0); cum = torch.cumsum(lam, 0) / lam.sum()
    deff = {"participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()), "dims_90pct_var": int((cum < 0.90).sum()) + 1,
            "dims_95pct_var": int((cum < 0.95).sum()) + 1, "dims_99pct_var": int((cum < 0.99).sum()) + 1,
            "entropy_rank": float(torch.exp(-(lam / lam.sum() * torch.log(lam / lam.sum() + 1e-30)).sum()))}
    for k in (1, 10, 100):
        tau = torch.cat([(Qc[i:i + 500] @ T.T).topk(k, 1).values[:, -1] for i in range(0, len(Qc), 500)])
        q = {f"tau_p{p}": round(float(torch.quantile(tau, p / 100)), 4) for p in (10, 50, 90)}
        bits = {name: {f"p{p}": round(v * float(np.log2(1 / q[f'tau_p{p}'])), 1) for p in (10, 50, 90)} for name, v in deff.items()}
        emit({"k": k, "N": N, **q, "d_eff": {kk: round(vv, 1) for kk, vv in deff.items()}, "bits_per_vector": bits})
print("DONE", flush=True)
