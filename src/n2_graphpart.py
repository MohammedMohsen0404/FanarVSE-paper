"""N2: kNN-graph min-cut partitioning (balanced label propagation) vs k-means, for IVF routing. OpenAI 229k.
Routing recall = fraction of true top-10 lying in the probed cells (probes chosen by query . cell-centroid).
Also reports edge-cut (fraction of 10-NN graph edges crossing cells) and cell-size balance."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import glob, json, numpy as np, torch, pyarrow.parquet as pq_
dev = "cuda"; torch.manual_seed(0); NQ = 2000
files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
T = torch.from_numpy(X[:-NQ]).to(dev); Q = torch.from_numpy(X[-NQ:]).to(dev); del X; N = len(T)
GT = torch.cat([(Q[i:i + 500] @ T.T).topk(10, 1).indices for i in range(0, NQ, 500)])
KNN = torch.cat([(T[i:i + 2000] @ T.T).topk(11, 1).indices[:, 1:] for i in range(0, N, 2000)])     # exact 10-NN graph
print("graph built", flush=True)

def kmeans(k, it=20):
    C = T[torch.randperm(N, device=dev)[:k]].clone()
    for _ in range(it):
        a = torch.cat([(T[i:i + 50000] @ C.T).argmax(1) for i in range(0, N, 50000)])
        C = torch.nn.functional.normalize(torch.zeros_like(C).index_add_(0, a, T) + 1e-9, dim=1)
    return a

def label_prop(a, k, iters=10, lam=1.0, symmetric=True):
    """balanced label propagation: move each node to the label maximising (#neighbours with that label) - lam * size/avg."""
    src = torch.arange(N, device=dev).repeat_interleave(10); dst = KNN.reshape(-1)
    if symmetric: src, dst = torch.cat([src, dst]), torch.cat([dst, src])
    avg = N / k
    for _ in range(iters):
        votes = torch.zeros(N, k, device=dev); votes.index_put_((src, a[dst]), torch.ones_like(src, dtype=torch.float), accumulate=True)
        votes[torch.arange(N, device=dev), a] += 0.5                                       # inertia: keep current label on ties
        size = torch.bincount(a, minlength=k).float()
        a = (votes - lam * size[None] / avg).argmax(1)
    return a

def evaluate(a, k, name):
    C = torch.nn.functional.normalize(torch.zeros(k, T.shape[1], device=dev).index_add_(0, a, T) + 1e-9, dim=1)
    gc = a[GT]; order = (Q @ C.T).argsort(1, descending=True); size = torch.bincount(a, minlength=k).float()
    cut = float((a[torch.arange(N, device=dev).repeat_interleave(10)] != a[KNN.reshape(-1)]).float().mean())
    r = {"partition": name, "k": k, "edge_cut": round(cut, 4), "max_cell/avg": round(float(size.max() / size.mean()), 2)}
    for p in (2, 4, 8, 16, 32):
        sel = order[:, :p]; r[f"route@np{p}"] = round(float((gc[:, :, None] == sel[:, None, :]).any(2).float().mean()), 4)
        r[f"scan%@np{p}"] = round(float(size[sel].sum(1).mean() / N * 100), 2)
    print(json.dumps(r), flush=True); return r

res = []
for k in (64, 256):
    a0 = kmeans(k); res.append(evaluate(a0, k, "k-means"))
    for lam in (0.5, 1.0, 2.0):
        a1 = label_prop(a0.clone(), k, lam=lam); res.append(evaluate(a1, k, f"k-means + graph min-cut LP (lam={lam})"))
json.dump(res, open(f"{FANAR}/results/n2_graphpart.json", "w"), indent=1)
