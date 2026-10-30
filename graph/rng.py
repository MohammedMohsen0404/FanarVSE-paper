"""diversified (RNG / Vamana-alpha) graph from the exact 64-NN lists: keep v_i unless a kept v_j has alpha*d(v_j,v_i) <= d(u,v_i);
then add reverse edges; final degree DEG (forward kept first, then reverse by similarity, padding with plain neighbours)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, time, numpy as np, torch
DS = sys.argv[1]; ALPHA = float(sys.argv[2]); DEG = int(sys.argv[3]); dev = "cuda"
X = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_base_fp16.npy", mmap_mode="r")[:]).to(dev); N = len(X)
G0 = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_knn64.npy")).to(dev).long(); S0 = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_knn64_s.npy")).to(dev).float()
keep = torch.zeros(N, 64, dtype=torch.bool, device=dev); t0 = time.time(); C = 1024
for s in range(0, N, C):
    g = G0[s:s + C]; V = X[g]; S = torch.bmm(V, V.transpose(1, 2)).float(); du = 1 - S0[s:s + C]; dvv = 1 - S
    k = torch.zeros(len(g), 64, dtype=torch.bool, device=dev)
    for i in range(64):
        occ = (k[:, :i] & (ALPHA * dvv[:, i, :i] <= du[:, i:i + 1])).any(1) if i else torch.zeros(len(g), dtype=torch.bool, device=dev)
        k[:, i] = ~occ
    keep[s:s + C] = k
del X, V, S, dvv; torch.cuda.empty_cache()
print("kept/node", keep.sum(1).float().mean().item(), round(time.time() - t0), flush=True)
# forward kept list (rank order), cap DEG//2 .. DEG
pri = torch.where(keep, torch.arange(64, device=dev)[None], 1000)
fo = torch.argsort(pri, dim=1, stable=True); fwd = torch.where(pri.gather(1, fo) < 1000, G0.gather(1, fo), -1)[:, :DEG]
fs = torch.where(fwd >= 0, S0.gather(1, fo)[:, :DEG], -9)
# reverse edges of kept forward edges
u = torch.arange(N, device=dev)[:, None].expand_as(fwd)[fwd >= 0]; v = fwd[fwd >= 0]; sv = fs[fwd >= 0]
o = torch.argsort(v * 4 - sv); v, u, sv = v[o], u[o], sv[o]
cnt = torch.bincount(v, minlength=N); st = torch.cumsum(cnt, 0) - cnt; rk = torch.arange(len(v), device=dev) - st[v]
RV = DEG; ok = rk < RV; rev = torch.full((N, RV), -1, dtype=torch.long, device=dev); rs = torch.full((N, RV), -9.0, device=dev)
rev[v[ok], rk[ok]] = u[ok]; rs[v[ok], rk[ok]] = sv[ok]
# merge: forward kept (priority 0..), reverse (by sim), then plain kNN; dedupe; take DEG (chunked)
Gf = torch.empty(N, DEG, dtype=torch.long, device=dev)
for a in range(0, N, 131072):
    b = slice(a, a + 131072); n = len(fwd[b])
    cand = torch.cat([fwd[b], rev[b], G0[b]], 1); pr = torch.cat([torch.arange(DEG, device=dev)[None].expand(n, -1).float(),
           torch.where(rev[b] >= 0, 100 - rs[b], 1e9), 1000 + torch.arange(64, device=dev)[None].expand(n, -1).float()], 1)
    pr = torch.where(cand < 0, 1e9, pr)
    key = torch.where(cand < 0, N + 1, cand).double() * 1e4 + pr.clamp(max=5e3).double()
    srt, oo = key.sort(1); idd = (srt / 1e4).floor(); dup = torch.zeros_like(idd, dtype=torch.bool); dup[:, 1:] = idd[:, 1:] == idd[:, :-1]
    dmask = torch.zeros_like(dup); dmask.scatter_(1, oo, dup); pr = torch.where(dmask, 1e9, pr)
    Gf[b] = cand.gather(1, pr.topk(DEG, 1, largest=False).indices)
print("fwd", (fwd >= 0).sum(1).float().mean().item(), "rev", (rev >= 0).sum(1).float().mean().item(), "invalid", (Gf < 0).sum().item())
np.save(f"{FANAR}/data/{DS}_rng_a{ALPHA}_d{DEG}.npy", Gf.int().cpu().numpy()); print("done", round(time.time() - t0))
