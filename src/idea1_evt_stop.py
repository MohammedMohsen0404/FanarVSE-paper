"""Idea 1: extreme-value-theory adaptive probing for IVF (training-free, per query).
Probe cells in centroid order. After each cell, fit (centroid score s_j -> observed cell max m_j) on the probed cells:
m = a + b s + Gumbel noise. For each unprobed cell, P(M_j > theta) with theta = current 10th best score.
Stop when the expected number of unprobed cells beating theta < eps. Compare with fixed nprobe at equal mean #cells.
usage: idea1_evt_stop.py <dbp|glove> N"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, numpy as np, torch, faiss
sys.path.insert(0, f"{FANAR}/src")
from hvb import Prep
ds, N = sys.argv[1], int(sys.argv[2]); NQ = 1000; dev = "cuda"
if ds == "glove":
    import h5py; X = np.array(h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r")["train"][:N + 60000], np.float32)
else:
    import pyarrow.parquet as pq_
    X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))])
X = X[np.random.default_rng(0).permutation(len(X))]; Xp = Prep().fit(X[:50000])(X); del X
N = min(N, len(Xp) - NQ); T = torch.from_numpy(Xp[:N]).to(dev); Q = torch.from_numpy(Xp[N:N + NQ]).to(dev); d = T.shape[1]
nlist = 1024; km = faiss.Kmeans(d, nlist, niter=20, seed=1); km.train(Xp[:min(N, 200000)]); C = torch.from_numpy(km.centroids).to(dev); del Xp
A = (T @ C.T).argmax(1); lists = [torch.nonzero(A == c).flatten() for c in range(nlist)]
GT = torch.cat([(Q[i:i + 100] @ T.T).topk(10, 1).indices for i in range(0, NQ, 100)]).cpu().numpy()
QC = Q @ C.T; order = QC.argsort(1, descending=True)

def run_query(i, max_probe, eps=None, min_probe=4):
    """returns (#cells probed, top10 ids). eps=None -> fixed max_probe."""
    best_s = torch.empty(0, device=dev); best_i = torch.empty(0, dtype=torch.long, device=dev)
    s_obs, m_obs = [], []; cells = order[i]; theta = -1e9
    for j in range(max_probe):
        c = int(cells[j]); ids = lists[c]
        if len(ids):
            sc = T[ids] @ Q[i]
            best_s = torch.cat([best_s, sc]); best_i = torch.cat([best_i, ids])
            if len(best_s) > 10: tk = best_s.topk(10); best_s, best_i = tk.values, best_i[tk.indices]
            s_obs.append(float(QC[i, c])); m_obs.append(float(sc.max()))
            theta = float(best_s.min()) if len(best_s) >= 10 else -1e9
        if eps is not None and j + 1 >= min_probe and len(s_obs) >= 3:
            s = np.array(s_obs); m = np.array(m_obs)
            b, a = np.polyfit(s, m, 1); res = m - (a + b * s)
            beta = max(res.std() * np.sqrt(6) / np.pi, 1e-4); mu0 = -0.5772 * beta          # Gumbel with zero-mean residual
            rest = QC[i, cells[j + 1:max_probe]].cpu().numpy(); loc = a + b * rest + mu0
            p = 1 - np.exp(-np.exp(-(theta - loc) / beta))                                  # P(M_j > theta)
            if p.sum() < eps: return j + 1, best_i.cpu().numpy()
    return max_probe, best_i.cpu().numpy()

rec = lambda outs: float(np.mean([len(set(o) & set(GT[i])) / 10 for i, o in enumerate(outs)]))
res = []
for npb in (8, 16, 32, 64, 128):
    outs = [run_query(i, npb)[1] for i in range(NQ)]
    r = {"ds": ds, "method": "fixed", "mean_probes": npb, "recall@10": round(rec(outs), 4)}; print(json.dumps(r), flush=True); res.append(r)
for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
    o = [run_query(i, 256, eps) for i in range(NQ)]
    r = {"ds": ds, "method": f"EVT eps={eps}", "mean_probes": round(float(np.mean([x[0] for x in o])), 1),
         "p90_probes": int(np.percentile([x[0] for x in o], 90)), "recall@10": round(rec([x[1] for x in o]), 4)}
    print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(f"{FANAR}/results/idea1_evt_{ds}.json", "w"), indent=1)
