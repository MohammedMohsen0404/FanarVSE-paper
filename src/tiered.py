"""Tiered stage-2 refinement (MSE water-filling over 32-dim PCA groups, min 1 bit; one Haar rotation + Lloyd-Max per tier; bit planes).
est = <q,c> + sum_t scale_t <q_t, cb_t[idx_t]>,  sigma^2 = sum_t ||q_t||^2 sig_t^2  (engine-style conservative interval, Theorem 1 per tier)."""
import numpy as np, torch, faiss
from fused_scan import rescore_planes
dev = "cuda"
_g = np.random.default_rng(1).standard_normal((1_000_000, 1)).astype(np.float32); CB, DIST = {}, {0: 1.0}
for _b in range(1, 9):
    _km = faiss.Kmeans(1, 2 ** _b, niter=60, seed=1); _km.train(_g[:400000]); _cb = np.sort(_km.centroids[:, 0])
    _th = (_cb[1:] + _cb[:-1]) / 2; DIST[_b] = float(((_g[:, 0] - _cb[np.searchsorted(_th, _g[:, 0])]) ** 2).mean())
    CB[_b] = (torch.from_numpy(_cb).to(dev), torch.from_numpy(_th).to(dev))
SH = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
pack = lambda bit: ((bit.reshape(bit.shape[0], -1, 8).to(torch.uint8)) << SH).sum(2).to(torch.uint8)
G = 32

def pca_basis(host16, c, n=50000):
    S = host16[:n].to(dev).float() - c; S = S / S.norm(dim=1, keepdim=True)
    ev, V = torch.linalg.eigh(torch.cov(S.T)); return ev.flip(0).clamp(min=1e-12), V.flip(1).contiguous()

def allocate(lam, B, bmin=1, bmax=8):
    d = len(lam); NG = d // G; lamg = lam.reshape(NG, G).sum(1).cpu().numpy(); b = np.full(NG, bmin); left = B - bmin * d
    while left >= G:
        gain = [lamg[i] * (DIST[b[i]] - DIST[b[i] + 1]) if b[i] < bmax else -1 for i in range(NG)]
        i = int(np.argmax(gain)); b[i] += 1; left -= G
    T, st = [], 0
    for i in range(1, NG + 1):
        if i == NG or b[i] != b[st]: T.append((st * G, i * G, int(b[st]))); st = i
    return T

def build(host16, order, c, V, TS, seed=0, chunk=16384):
    """codes for vectors in the given order (sorted positions). Returns dict with per-tier planes (bb, N, dt/8), scale, sig (fp16)."""
    N = len(order); gen = torch.Generator(device=dev).manual_seed(100 + seed)
    rots = [torch.linalg.qr(torch.randn(e - s, e - s, device=dev, generator=gen))[0] for s, e, _ in TS]
    planes = [torch.empty(bb, N, (e - s) // 8, dtype=torch.uint8, device=dev) for s, e, bb in TS]
    scale = torch.empty(len(TS), N, dtype=torch.float16, device=dev); sig = torch.empty(len(TS), N, dtype=torch.float16, device=dev)
    oc = order.cpu()
    for j in range(0, N, chunk):
        Z = (host16[oc[j:j + chunk]].to(dev).float() - c) @ V; n = len(Z)
        for t, ((s, e, bb), P) in enumerate(zip(TS, rots)):
            F = Z[:, s:e] @ P; nf = F.norm(dim=1).clamp(min=1e-12); dt = e - s; Fu = F / nf[:, None] * dt ** 0.5
            cb, th = CB[bb]; q = torch.bucketize(Fu, th); Y = cb[q]; yn = Y.norm(dim=1); ct = (Fu * Y).sum(1) / (yn * dt ** 0.5)
            for p in range(bb): planes[t][p, j:j + n] = pack(((q >> (bb - 1 - p)) & 1).bool())
            scale[t, j:j + n] = (nf / (ct * yn)).half(); sig[t, j:j + n] = (nf * torch.sqrt((1 - ct ** 2).clamp(min=0)) / (ct * (dt - 1) ** 0.5)).half()
    nbytes = sum((e - s) * bb for s, e, bb in TS) / 8 + 4 * len(TS)
    return dict(TS=TS, rots=rots, planes=planes, scale=scale, sig=sig, V=V, bytes=nbytes)

def score(code, Qb, qc, pos):
    """Qb (B, d) queries, qc (B,) = <q, c>, pos (B, M) sorted positions -> est (B, M), sigma (B, M)."""
    Zq = Qb @ code["V"]; est = qc[:, None].float().expand(pos.shape).clone(); var = torch.zeros(pos.shape, device=dev)
    zero = torch.zeros(len(Qb), device=dev)
    for t, ((s, e, bb), P) in enumerate(zip(code["TS"], code["rots"])):
        qt = (Zq[:, s:e] @ P).contiguous()
        est += rescore_planes(qt, zero, pos, code["planes"][t], CB[bb][0], code["scale"][t])
        var += (qt * qt).sum(1, keepdim=True) * code["sig"][t][pos].float() ** 2
    return est, var.sqrt()
