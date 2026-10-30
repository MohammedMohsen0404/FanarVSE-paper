"""Exp 35: Lattice-RaBitQ in IVF, general version. Same coarse cells, same per-vector factors (|r|, <o,o_bar>, |Y|), only the
codebook changes. Codebooks by rate: 1 bit/dim: hypercube | E8-256 | BW16-65536 ; 2 bit/dim: scalar-Lloyd-2 | D4-256 | E8 two-stage.
usage: rev_cfail <dbp|oai3|arxiv|glove> N rate(1|2) rot|norot
Reviewer Q12: does the codebook order predicted by c survive without the random rotation (non-Gaussian, anisotropic coordinates)?"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, numpy as np, torch, faiss
sys.path.insert(0, f"{FANAR}/src")
from hvb import Prep
from lattices import e8_shell, d4_shell, bw16_three_shells
ds, N, RATE, ROT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; dev = "cuda"; torch.manual_seed(0); NQ = 1000
import pyarrow.parquet as pq_
def load():
    g = np.random.default_rng(7)
    if ds == "t3":                                                                       # heavy-tailed coordinates (Student-t, 3 dof), 128 dims
        return g.standard_t(3, size=(N + 60000, 128)).astype(np.float32)
    if ds == "sparse":                                                                   # 16 non-zeros out of 256, Gaussian values
        Xs = np.zeros((N + 60000, 256), np.float32); idx = np.argsort(g.random((N + 60000, 256)), 1)[:, :16]
        np.put_along_axis(Xs, idx, g.standard_normal((N + 60000, 16)).astype(np.float32), 1); return Xs
    if ds == "glove":
        import h5py; return np.array(h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r")["train"][:N + 60000], np.float32)
    pat, col = {"dbp": ("dbp/shard*.parquet", "openai"), "oai3": ("oai3/part*.parquet", None), "arxiv": ("arxiv/part*.parquet", None)}[ds]
    parts = []
    for p in sorted(glob.glob(f"{FANAR}/data/{pat}")):
        t = pq_.read_table(p)
        c = col or [n for n in t.schema.names if t.schema.field(n).type.__class__.__name__ in ("ListType", "FixedSizeListType", "LargeListType")][0]
        parts.append(np.array(t.column(c).to_pylist(), np.float32))
    return np.concatenate(parts)
X = load(); X = X[np.random.default_rng(0).permutation(len(X))]; Xp = Prep().fit(X[:min(50000, len(X) // 2)])(X); del X
d0 = Xp.shape[1]; d = int(np.ceil(d0 / 16) * 16); Xp = np.hstack([Xp, np.zeros((len(Xp), d - d0), np.float32)])
N = min(N, len(Xp) - NQ); Tn, Qn = np.ascontiguousarray(Xp[:N]), np.ascontiguousarray(Xp[N:N + NQ]); del Xp
nlist = int(4 * np.sqrt(N)); T, Q = torch.from_numpy(Tn).to(dev), torch.from_numpy(Qn).to(dev)
GT = (Q @ T.T).topk(10, 1).indices.cpu().numpy()
km = faiss.Kmeans(d, nlist, niter=20, seed=1); km.train(Tn[: min(N, 100000)]); C = torch.from_numpy(km.centroids).to(dev)
A = (T @ C.T).argmax(1)
P, _ = torch.linalg.qr(torch.randn(d, d, device=dev))
if ROT == "norot": P = torch.eye(d, device=dev)
R = T - C[A]; nr = R.norm(dim=1); O = R / nr[:, None]; Z = O @ P
zc = Z[:20000] * d ** 0.5; zc = zc[:, zc.var(0) > 1e-8]; KURT = float((((zc - zc.mean(0)) ** 4).mean(0) / zc.var(0) ** 2 - 3).median()); SPREAD = float(zc.var(0).max() / zc.var(0).median())
lists = [torch.nonzero(A == c).flatten() for c in range(nlist)]

def nearest(B, Cb, chunk):
    return torch.cat([torch.cdist(B[i:i + chunk], Cb).argmin(1) for i in range(0, len(B), chunk)])
def fit_alpha(Cb, blk, chunk):
    B = Z.reshape(-1, blk); m = (d // blk) * 1500; best = None
    for a in np.geomspace(2, 400, 30):
        Y = Cb[nearest(B[:m] * a, Cb, chunk)].reshape(-1, d)
        cs = float((Z[: len(Y)] * Y / Y.norm(dim=1, keepdim=True).clamp(min=1e-9)).sum(1).mean())
        if best is None or cs > best[0]: best = (cs, a)
    return best[1]
def lloyd(bits):
    g = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32)
    k = faiss.Kmeans(1, 2 ** bits, niter=60, seed=1); k.train(g); return np.sort(k.centroids, 0)

def encode(kind):
    """returns list of (codebook tensor, block size, index tensor (N, nblocks)) stages, and Y (N,d) reconstruction"""
    if kind == "hypercube": return None, torch.sign(Z)
    spec = {"E8-256": (e8_shell(), 8, 1_000_000), "BW16-65536": (bw16_three_shells(), 16, 4096),
            "scalar-Lloyd-2": (lloyd(2), 1, 4_000_000), "D4-256": (d4_shell(), 4, 1_000_000), "E8x2": (e8_shell(), 8, 1_000_000)}[kind]
    Cb, blk, ch = torch.from_numpy(spec[0]).to(dev), spec[1], spec[2]
    if kind != "E8x2":
        a = fit_alpha(Cb, blk, ch); idx = nearest(Z.reshape(-1, blk) * a, Cb, ch).reshape(N, -1)
        return [(Cb / a, blk, idx)], (Cb[idx] / a).reshape(N, d)
    B = Z.reshape(-1, blk); best = None
    for a1 in np.geomspace(5, 100, 10):
        Y1 = Cb[nearest(B[:200000] * a1, Cb, ch)] / a1; Rr = B[:200000] - Y1
        for a2 in np.geomspace(a1 * 1.5, a1 * 20, 8):
            err = float(((Rr - Cb[nearest(Rr * a2, Cb, ch)] / a2) ** 2).sum())
            if best is None or err < best[0]: best = (err, a1, a2)
    _, a1, a2 = best; i1 = nearest(B * a1, Cb, ch); Y1 = Cb[i1] / a1; i2 = nearest((B - Y1) * a2, Cb, ch)
    return [(Cb / a1, blk, i1.reshape(N, -1)), (Cb / a2, blk, i2.reshape(N, -1))], (Y1 + Cb[i2] / a2).reshape(N, d)

def run(kind):
    stages, Y = encode(kind); yn = Y.norm(dim=1); Ob = (Y / yn[:, None]) @ P.T; fip = (O * Ob).sum(1)
    PQm = Q @ P; qc = Q @ C.T; res = []
    for npb in (16, 64):
        probe = qc.topk(min(npb, nlist), 1).indices; out0, out20, out40 = [], [], []
        for i in range(NQ):
            cand = torch.cat([lists[c] for c in probe[i].tolist()])
            if stages is None: qy = Y[cand] @ PQm[i]
            else:
                qy = 0
                for Cb, blk, idx in stages:
                    lut = PQm[i].reshape(-1, blk) @ Cb.T; qy = qy + lut.gather(1, idx[cand].T.long()).sum(0)
            est = (qy / yn[cand]) * nr[cand] / fip[cand] + qc[i, A[cand]]
            top = cand[est.topk(min(40, len(cand))).indices]; out0.append(top[:10].cpu().numpy())
            out40.append(top[(T[top] @ Q[i]).topk(10).indices].cpu().numpy()); t20 = top[:20]; out20.append(t20[(T[t20] @ Q[i]).topk(min(10, len(t20))).indices].cpu().numpy())
        rc = lambda o: round(float(np.mean([len(set(o[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4)
        r = {"ds": ds, "N": N, "rate": RATE, "rotation": ROT, "coord_excess_kurtosis_median": round(KURT, 3), "coord_var_max_over_median": round(SPREAD, 2), "codebook": kind, "mean_cos": round(float(fip.mean()), 4), "nprobe": npb, "R@10": rc(out0), "R@10_rerank20": rc(out20), "R@10_rerank40": rc(out40)}
        print(json.dumps(r), flush=True); res.append(r)
    return res
kinds = ["hypercube", "E8-256", "BW16-65536"] if RATE == 1 else ["scalar-Lloyd-2", "D4-256", "E8x2"]
allres = sum([run(k) for k in kinds], [])
json.dump(allres, open(f"{FANAR}/results/rev_cfail_{ds}_{N}_r{RATE}_{ROT}.json", "w"), indent=1)
