"""End to end: graph stage 1 (int8 navigation codes + pruned graph, CUDA kernel) -> FanarVSE stage 2 (error bars) -> stage 3 (exact).
env GF graph file, DEG, NR (navigation dims), GRID L:E:T:K,..."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/host_resident"); sys.path.insert(0, f"{FANAR}/graph"); os.environ.setdefault("TAG", "graph")
from h import *
from gcuda import GraphCU
dev = "cuda"; NR = int(os.environ.get("NR", 256)); DEG = int(os.environ.get("DEG", 32))
e, mem, e3 = build(); s1 = e.s1; N = rf.N
pos_of = torch.empty(N, dtype=torch.long, device=dev); pos_of[s1["order"].long()] = torch.arange(N, device=dev)
X = rf.X; c = torch.from_numpy(X[:200000].astype(np.float32).mean(0)).to(dev)
Sm = torch.from_numpy(X[::20].astype(np.float32)).to(dev) - c; ev, V = torch.linalg.eigh(torch.cov(Sm.T)); W = V.flip(1)[:, :NR].contiguous(); del Sm
Z = torch.empty(N, NR, dtype=torch.int8, device=dev); zs = torch.empty(N, dtype=torch.float16, device=dev)
for s in range(0, N, 65536):
    z = (torch.from_numpy(X[s:s + 65536]).to(dev).float() - c) @ W; a = z.abs().amax(1) / 127
    Z[s:s + 65536] = torch.round(z / a[:, None]).to(torch.int8); zs[s:s + 65536] = a.half()
G = torch.from_numpy(np.load(os.environ.get("GF", f"{FANAR}/data/oai1m_rng_a1.2_d32.npy"))[:, :DEG].copy()).to(dev)
g = GraphCU(Z, zs, W, c, G); torch.cuda.empty_cache()
graph_B = (Z.numel() + zs.numel() * 2 + G.numel() * 4) / N

def search(Qb, L, E, T, K, k=10, z=2.0):
    oid, osc, nev = g.search(Qb, L, E, T, KO=K, TPC=4, VMODE=2, VHB=12)
    pos = pos_of[oid.long()]; QC = Qb @ s1["c"]; est, sg = e._score(Qb, QC, pos)
    lo, hi = est - z * sg, est + z * sg
    sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
    ids = oid.long(); score = e._s3(Qb, ids, est, sure, amb)
    return ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids, nev

for L, E, T, K in [tuple(map(int, x.split(":"))) for x in os.environ.get("GRID", "128:2:200:128").split(",")]:
    B = BATCH; best = 1e9
    for rep in range(3):
        I, C, NE, RD = [], [], [], []; torch.cuda.synchronize(); t0 = time.perf_counter()
        for s in range(0, 2000, B):
            a, r, cc, ne = search(rf.Qg[s:s + B], L, E, T, K); I.append(a); C.append(cc); NE.append(ne); RD.append(r)
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t0)
    I = torch.cat(I).cpu().numpy(); C = torch.cat(C).cpu().numpy()
    emit({"tag": TAG, "ds": DS, "graph": os.path.basename(os.environ.get("GF", "rng_a1.2_d32")), "NR": NR, "L": L, "E": E, "T": T, "K": K, "batch": B,
          "recall": rf.recall(I, 10), "ceiling": rf.cand_recall(C, 10), "QPS": round(2000 / best), "evals": round(torch.cat(NE).float().mean().item()),
          "reads": round(torch.cat(RD).float().mean().item(), 2), "graph_B": round(graph_B, 1), "gpu_B": round(mem["gpu_B"] - mem["stage1_B"] + graph_B, 1)})
if os.environ.get("DUPCHK"):
    a, r, cc, ne = search(rf.Qg[:500], 384, 2, 1000, 128)
    cc = cc.cpu().numpy(); a = a.cpu().numpy()
    print("rows with dup candidates", sum(len(set(x)) < len(x) for x in cc), "rows with dup top10", sum(len(set(x)) < len(x) for x in a))
