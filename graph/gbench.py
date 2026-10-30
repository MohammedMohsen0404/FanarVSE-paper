"""benchmark graph stage 1 kernel: candidate recall of top-KO and time per batch."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/graph"); from gkern import GraphS1
KER = os.environ.get('KER', 'cu')
if KER == 'cu': from gcuda import GraphCU
if KER == 'w': from gwarp import GraphW as GraphCU
DS = os.environ.get("DS", "oai1m"); R = int(os.environ.get("R", 256)); DEG = int(os.environ.get("DEG", 64)); dev = "cuda"; B = int(os.environ.get("B", 500))
X = np.load(f"{FANAR}/data/{DS}_base_fp16.npy", mmap_mode="r"); N, d = X.shape
Q = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_q.npy")).float().to(dev); GT = np.load(f"{FANAR}/data/{DS}_gt.npy")[:, :10]
G = torch.from_numpy(np.load(os.environ.get("GF", f"{FANAR}/data/{DS}_knn64.npy"))[:, :DEG].copy()).to(dev)
c = torch.from_numpy(X[:200000].astype(np.float32).mean(0)).to(dev)
Sm = torch.from_numpy(X[::20].astype(np.float32)).to(dev) - c; ev, V = torch.linalg.eigh(torch.cov(Sm.T)); W = V.flip(1)[:, :R].contiguous(); del Sm
CODE = os.environ.get("CODE", "int8"); QP = int(os.environ.get("QP", 4)); TPC = int(os.environ.get("TPC", 8))
if CODE == "sign":
    Z = torch.empty(N, R // 32, dtype=torch.int32, device=dev); zs = torch.empty(N, dtype=torch.float16, device=dev); Zsf = torch.empty(N, R, dtype=torch.float16, device=dev)
    sh = torch.arange(32, device=dev, dtype=torch.int64)
    for s in range(0, N, 65536):
        z = (torch.from_numpy(X[s:s + 65536]).to(dev).float() - c) @ W; y = torch.where(z >= 0, 1.0, -1.0)
        zs[s:s + 65536] = ((z * z).sum(1) / (z * y).sum(1)).half(); Zsf[s:s + 65536] = (y * zs[s:s + 65536].float()[:, None]).half()
        wv = ((z >= 0).long().reshape(len(z), -1, 32) << sh).sum(2); Z[s:s + 65536] = torch.where(wv >= 2 ** 31, wv - 2 ** 32, wv).int()
else:
    Z = torch.empty(N, R, dtype=torch.int8, device=dev); zs = torch.empty(N, dtype=torch.float16, device=dev); Zsf = None
    for s in range(0, N, 65536):
        z = (torch.from_numpy(X[s:s + 65536]).to(dev).float() - c) @ W; a = z.abs().amax(1) / 127
        Z[s:s + 65536] = torch.round(z / a[:, None]).to(torch.int8); zs[s:s + 65536] = a.half()
g = GraphCU(Z, zs, W, c, G, sign=CODE == 'sign', QP=QP, Zs=Zsf) if KER in ('cu', 'w') else GraphS1(Z, zs, W, c, G)
for L, E, T, KO, DBG, NWp in [tuple(map(int, (x + ':0:4').split(":")[:6])) for x in os.environ.get("GRID", "256:2:60:128,256:2:100:128,384:2:120:128").split(",")]:
    I, NE = [], []; kw = dict(TPC=TPC, VHB=int(os.environ.get('VHB', 0)), VMODE=int(os.environ.get('VMODE', 1))) if KER == 'cu' else dict(TPC=TPC) if KER in ('cu', 'w') else dict(DBG=DBG, nw=NWp)
    for _ in range(3): g.search(Q[:B], L, E, T, KO, **kw); torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in range(0, 2000, B):
        oid, osc, nev = g.search(Q[s:s + B], L, E, T, KO, **kw); I.append(oid); NE.append(nev)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    I = torch.cat(I).cpu().numpy(); NE = torch.cat(NE).float().mean().item()
    rec = lambda k: np.mean([len(set(GT[q]) & set(I[q, :k])) / 10 for q in range(2000)])
    if hasattr(g, 'prof'): p = g.prof.double().mean(0).tolist(); tot = sum(p[:4]); print('prof: sel %.2f score %.2f sort %.2f merge %.2f | iters %.0f sorted/iter %.1f | cycles/iter %.0f' % (p[0]/tot, p[1]/tot, p[2]/tot, p[3]/tot, p[4], p[5]/p[4], tot/p[4]))
    print(dict(VMODE=os.environ.get('VMODE', 1), VHB=os.environ.get('VHB', 0), CODE=CODE, QP=QP, TPC=TPC, GF=os.environ.get("GF", "knn")[-16:], R=R, DEG=DEG, L=L, E=E, T=T, B=B, DBG=DBG, nw=NWp, cand100=round(rec(100), 4), candKO=round(rec(KO), 4), evals=round(NE), ms_per_batch=round(dt / (2000 / B) * 1e3, 2), qps=round(2000 / dt)), flush=True)
