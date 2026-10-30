"""Offline test of stage-1 cell-residual encoding: candidate ceiling (top-K stage-1 estimate within n_p probed cells, recall of the
true top-10) for (A) global-centre residual (the engine), (B) cell-mean residual with the global PCA, (C) cell-mean residual with
a PCA of the cell residuals. Same trellis encoder, R, L and scale as the engine. usage: v18_cellres_sim.py DS [R]"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
DS = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 256
from rf_engine import rf
import viterbi_fused
dev = "cuda"; X = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_base_fp16.npy")); Q = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_q.npy")).to(dev)
GT = np.load(f"{FANAR}/data/{DS}_gt.npy"); N, d = X.shape; NL = 4096 if N >= 1_000_000 else (1024 if N > 150000 else 512); L = rf.L
torch.manual_seed(0)
samp = X[torch.randperm(N)[:200000]].to(dev).float(); cen = samp[torch.randperm(len(samp), device=dev)[:NL]].clone()
for _ in range(15):
    a_ = (samp @ cen.T).argmax(1); cen = torch.nn.functional.normalize(torch.zeros_like(cen).index_add_(0, a_, samp) + 1e-9, dim=1)
A = torch.cat([(X[j:j + 100000].to(dev).float() @ cen.T).argmax(1) for j in range(0, N, 100000)])
cnt = torch.bincount(A, minlength=NL).float().clamp(min=1); mean = torch.zeros(NL, d, device=dev)
for j in range(0, N, 100000): mean.index_add_(0, A[j:j + 100000], X[j:j + 100000].to(dev).float())
mean /= cnt[:, None]; c = X[:50000].to(dev).float().mean(0)
def pca(S): ev, V = torch.linalg.eigh(torch.cov(S.T)); return V.flip(1)[:, :R] @ torch.linalg.qr(torch.randn(R, R, device=dev))[0], ev.flip(0)
idx = torch.randperm(N)[:50000]
Wg, evg = pca(X[idx].to(dev).float() - c); Wc, evc = pca(X[idx].to(dev).float() - mean[A[idx.to(dev)]])
Tt = rf.make_table(L); G = torch.randn(512, R, device=dev); best = None
for a in np.linspace(0.6, 1.3, 15):
    m = float(((G - rf.viterbi(G, Tt, L, 1, float(a))) ** 2).mean()); best = (m, float(a)) if best is None or m < best[0] else best
alpha = best[1]; tab = alpha * Tt
def encode(cent_of, W):
    """returns the stage-1 estimate of the residual part as (Y * scale) in R dims (fp16), per vector"""
    out = torch.empty(N, R, dtype=torch.float16, device=dev)
    for j in range(0, N, 16384):
        Rr = X[j:j + 16384].to(dev).float() - cent_of(j); Z = Rr @ W; nz = Z.norm(dim=1).clamp(min=1e-6); Zs = Z / nz[:, None] * R ** 0.5
        path = torch.cat([viterbi_fused.viterbi_paths_fused(Zs[i:i + 2048], Tt, L, alpha) for i in range(0, len(Zs), 2048)])
        Y = tab[path]; yn = Y.norm(dim=1); fip = (Zs * Y).sum(1) / (yn * R ** 0.5)
        out[j:j + len(Zs)] = (Y * (nz / (fip * yn))[:, None]).half()
    return out
def ceiling(Yhat, W, cterm, nprobes=(32, 64, 128, 256), Ks=(100, 200, 400, 800)):
    QW = (Q @ W).half(); probe_rank = torch.argsort(-(Q @ cen.T), 1); res = {}
    for s in range(0, len(Q), 250):
        qb = slice(s, s + 250); est = (QW[qb] @ Yhat.T).float() + cterm(qb)                       # 250 x N
        for npb in nprobes:
            mask = torch.zeros(est.shape[0], NL, dtype=torch.bool, device=dev); mask.scatter_(1, probe_rank[qb, :npb], True)
            e = torch.where(mask[:, A], est, torch.tensor(-1e9, device=dev))
            for K in Ks:
                top = e.topk(K, 1).indices.cpu().numpy()
                res.setdefault((npb, K), []).extend(len(set(top[i]) & set(GT[s + i])) / 10 for i in range(len(top)))
        del est
    return {f"{k[0]}:{k[1]}": round(float(np.mean(v)), 4) for k, v in res.items()}
out = {"ds": DS, "R": R, "nlist": NL, "energy_topR_global": round(float(evg[:R].sum() / evg.sum()), 3), "energy_topR_cell": round(float(evc[:R].sum() / evc.sum()), 3),
       "resid_norm2_global": round(float(((X[idx].to(dev).float() - c) ** 2).sum(1).mean()), 4), "resid_norm2_cell": round(float(((X[idx].to(dev).float() - mean[A[idx.to(dev)]]) ** 2).sum(1).mean()), 4)}
print(json.dumps(out), flush=True)
Ag = encode(lambda j: c, Wg); out["A_global"] = ceiling(Ag, Wg, lambda qb: (Q[qb] @ c)[:, None]); del Ag; print(json.dumps(out), flush=True)
QM = Q @ mean.T                                                                                        # q . cell mean, exact (routing-like)
Bc = encode(lambda j: mean[A[j:j + 16384]], Wg); out["B_cell_globalPCA"] = ceiling(Bc, Wg, lambda qb: QM[qb][:, A]); del Bc; print(json.dumps(out), flush=True)
Cc = encode(lambda j: mean[A[j:j + 16384]], Wc); out["C_cell_cellPCA"] = ceiling(Cc, Wc, lambda qb: QM[qb][:, A]); del Cc
print(json.dumps(out), flush=True); json.dump(out, open(f"{FANAR}/results/v18_cellres_{DS}_R{R}.json", "w"), indent=1)
