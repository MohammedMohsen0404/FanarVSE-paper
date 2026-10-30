"""Idea 34 simulation: graph traversal over compressed vectors (PCA-R fp16 of x - mean) instead of IVF scanning.
Measures evaluations per query vs candidate recall (top-10 inside the final list), which bounds the refined recall."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, time, numpy as np, torch
DS = os.environ.get("DS", "oai1m"); R = int(os.environ.get("R", 256)); DEG = int(os.environ.get("DEG", 32)); dev = "cuda"
X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy", mmap_mode="r"); N, d = X.shape
Q = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_q.npy")).float().to(dev); GT = np.load(f"{FANAR}/data/{DS}_gt.npy")[:, :10]
G0 = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_knn64.npy")).to(dev)
S0 = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_knn64_s.npy")).to(dev)

def build_graph(deg):
    """forward kNN (deg/2) + reverse edges, filled to deg with further forward neighbours."""
    f = deg // 2; u = torch.arange(N, device=dev).repeat_interleave(f); v = G0[:, :f].reshape(-1).long(); s = S0[:, :f].reshape(-1).float()
    key = v * 4 - s                                                                   # group by v, best sim first (s in [-1,1])
    o = torch.argsort(key); v, u = v[o], u[o]
    cnt = torch.bincount(v, minlength=N); start = torch.cumsum(cnt, 0) - cnt; rank = torch.arange(len(v), device=dev) - start[v]
    keep = rank < f; rev = torch.full((N, f), -1, dtype=torch.long, device=dev); rev[v[keep], rank[keep]] = u[keep]
    cand = torch.cat([G0[:, :f].long(), rev, G0[:, f:].long()], 1)                    # forward, reverse, extra forward
    srt, oo = torch.sort(torch.where(cand < 0, N + torch.arange(cand.shape[1], device=dev)[None], cand), dim=1, stable=True)
    dup = torch.zeros_like(srt, dtype=torch.bool); dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
    pos_ok = torch.zeros_like(dup); pos_ok.scatter_(1, oo, ~dup & (srt < N))
    pri = torch.where(pos_ok, torch.arange(cand.shape[1], device=dev)[None].expand_as(cand), 10 ** 6)
    sel = pri.topk(deg, 1, largest=False).indices; g = cand.gather(1, sel)
    bad = pri.gather(1, sel) >= 10 ** 6; g[bad] = G0[:, :deg].long()[bad]
    return g.int().contiguous()

# compressed navigation vectors: PCA-R of x - c (fp16)
c = torch.from_numpy(X[:200000].astype(np.float32).mean(0)).to(dev)
Sm = torch.from_numpy(X[::20].astype(np.float32)).to(dev) - c; ev, V = torch.linalg.eigh(torch.cov(Sm.T)); W = V.flip(1)[:, :R].contiguous(); del Sm
Z = torch.empty(N, R, dtype=torch.float16, device=dev)
for s in range(0, N, 65536): Z[s:s + 65536] = ((torch.from_numpy(X[s:s + 65536]).to(dev).float() - c) @ W).half()
ZT = os.environ.get("ZT", "fp16")
for s_ in range(0, N, 65536):
    z = Z[s_:s_ + 65536].float()
    if ZT == "sign": y = torch.sign(z); Z[s_:s_ + 65536] = (y * ((z * z).sum(1, keepdim=True) / (z * y).sum(1, keepdim=True))).half()
    if ZT == "int8": a = z.abs().amax(1, keepdim=True) / 127; Z[s_:s_ + 65536] = (torch.round(z / a) * a).half()
    if ZT == "int4": a = z.abs().amax(1, keepdim=True) / 7; Z[s_:s_ + 65536] = (torch.round(z / a) * a).half()
smp = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:int(os.environ.get("NSEED", 8192))].to(dev)

def search(Qb, G, L, w, T):
    B = len(Qb); qz = ((Qb - 0) @ W).half(); qc = Qb @ c
    sc = (qz.float() @ Z[smp].float().T); v, i = sc.topk(L, 1); ids = smp[i]; dd = v
    expd = torch.zeros_like(ids, dtype=torch.bool)
    vis = torch.zeros(B, N, dtype=torch.bool, device=dev); vis.scatter_(1, ids, True); evals = torch.full((B,), L, device=dev)
    for t in range(T):
        sel = torch.where(expd, torch.full_like(dd, -1e9), dd).topk(w, 1).indices
        expd.scatter_(1, sel, True); nodes = ids.gather(1, sel)
        nb = G[nodes].reshape(B, -1).long(); new = ~vis.gather(1, nb); vis.scatter_(1, nb, True)
        dn = torch.bmm(Z[nb], qz[:, :, None]).squeeze(2).float(); dn = torch.where(new, dn, torch.full_like(dn, -1e9)); evals += new.sum(1)
        ids = torch.cat([ids, nb], 1); dd = torch.cat([dd, dn], 1); expd = torch.cat([expd, torch.zeros_like(new)], 1)
        dd, o = dd.topk(L, 1); ids = ids.gather(1, o); expd = expd.gather(1, o)
    return ids, evals

for deg in [int(x) for x in os.environ.get("DEGS", "32").split(",")]:
    G = build_graph(deg) if deg < 64 else G0
    for L, w, T in [tuple(map(int, s.split(":"))) for s in os.environ.get("GRID", "128:4:40,256:4:80,512:8:80").split(",")]:
        I, E = [], []; torch.cuda.synchronize(); t0 = time.time()
        for s in range(0, 2000, 250):
            ids, ev = search(Q[s:s + 250], G, L, w, T); I.append(ids.cpu().numpy()); E.append(ev.cpu().numpy())
        torch.cuda.synchronize(); dt = time.time() - t0; I = np.vstack(I); E = np.concatenate(E)
        rec = np.mean([len(set(GT[q]) & set(I[q])) / 10 for q in range(2000)])
        rec10 = np.mean([len(set(GT[q]) & set(I[q, :200])) / 10 for q in range(2000)])
        print(dict(ZT=ZT, R=R, deg=deg, L=L, w=w, T=T, cand_recall=round(rec, 4), recall_in_top200=round(rec10, 4), evals=int(E.mean()), torch_qps=round(2000 / dt)), flush=True)
