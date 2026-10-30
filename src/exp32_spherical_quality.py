"""Exp 32: codebook quality for the RaBitQ estimator. For each codebook, encode random unit vectors (d=1536, i.i.d. after
rotation) block-wise with one global scale, and report c = <o, o_bar> and the predicted relative error sqrt(1-c^2)/c."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch, faiss
sys.path.insert(0, f"{FANAR}/src")
from lattices import e8_shell, d4_shell, bw16_shell, bw16_three_shells
dev = "cuda"; torch.manual_seed(0); d = 1536; n = 20000
O = torch.randn(n, d, device=dev); O = O / O.norm(dim=1, keepdim=True)

def nearest(B, C, chunk):
    return torch.cat([torch.cdist(B[i:i + chunk], C).argmin(1) for i in range(0, len(B), chunk)])

def block_code(O, C, blk, chunk):
    C = torch.from_numpy(C).to(dev); B = O.reshape(-1, blk); best = None
    sample = B[: (d // blk) * 2000]
    for a in np.geomspace(2, 400, 36):
        Y = C[nearest(sample * a, C, chunk)].reshape(-1, d); cs = float((O[: len(Y)] * Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-9)).sum(1).mean())
        if best is None or cs > best[0]: best = (cs, a)
    Y = C[nearest(B * best[1], C, chunk)].reshape(O.shape)
    return float((O * Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-9)).sum(1).mean())

def two_stage(O, C, blk, chunk):
    """Residual lattice coding: stage 1 then stage 2 on the residual, each with its own global scale (16 bits / 8 dims)."""
    Ct = torch.from_numpy(C).to(dev); B = O.reshape(-1, blk); bestv = None
    for a1 in np.geomspace(5, 100, 12):
        Y1 = Ct[nearest(B[:200000] * a1, Ct, chunk)] / a1; Rr = B[:200000] - Y1
        for a2 in np.geomspace(a1 * 1.5, a1 * 20, 10):
            Y = Y1 + Ct[nearest(Rr * a2, Ct, chunk)] / a2
            err = float(((B[:200000] - Y) ** 2).sum())
            if bestv is None or err < bestv[0]: bestv = (err, a1, a2)
    _, a1, a2 = bestv; Y1 = Ct[nearest(B * a1, Ct, chunk)] / a1; Y = Y1 + Ct[nearest((B - Y1) * a2, Ct, chunk)] / a2
    Y = Y.reshape(O.shape); return float((O * Y / Y.norm(dim=1, keepdim=True)).sum(1).mean())

def lloyd_scalar(bits):
    g = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32)
    km = faiss.Kmeans(1, 2 ** bits, niter=60, seed=1); km.train(g); return np.sort(km.centroids, 0)

out = []
def rep(rate, name, c):
    r = {"bits_per_dim": rate, "codebook": name, "c": round(c, 4), "pred_rel_err": round(float(np.sqrt(1 - c * c) / c), 4)}
    print(json.dumps(r), flush=True); out.append(r)
rep(1.0, "hypercube (RaBitQ)", float((O * torch.sign(O) / d ** 0.5).sum(1).mean()))
rep(1.0, "E8 shell-256 (8-d)", block_code(O, e8_shell(), 8, 1_000_000))
rep(1.0, "BW16 3-shell-65536 (16-d)", block_code(O, bw16_three_shells(), 16, 4096))
rep(0.5, "BW16 shell-256 (16-d)", block_code(O, bw16_shell(), 16, 500_000))
rep(2.0, "scalar Lloyd 2-bit (~extended RaBitQ grid)", block_code(O, lloyd_scalar(2), 1, 4_000_000))
rep(2.0, "D4 shell-256 (4-d)", block_code(O, d4_shell(), 4, 1_000_000))
rep(2.0, "E8 two-stage residual (8-d, 16 bits)", two_stage(O, e8_shell(), 8, 1_000_000))
json.dump(out, open(f"{FANAR}/results/exp32_spherical_quality.json", "w"), indent=1)
