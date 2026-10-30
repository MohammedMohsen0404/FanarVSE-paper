"""Build the 1M OpenAI dataset once: normalised fp16 base (for WSL baselines), 2000 held-out queries, exact fp32 ground truth."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import glob, numpy as np, torch, pyarrow.parquet as pq_
files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet")); d = 1536; NQ = 2000
sizes = [pq_.ParquetFile(f).metadata.num_rows for f in files]; Ntot = sum(sizes)
X = np.empty((Ntot, d), np.float32); o = 0
for f, n in zip(files, sizes):
    X[o:o + n] = np.array(pq_.read_table(f, columns=["openai"]).column("openai").to_pylist(), np.float32); o += n
mu = X[:50000].mean(0)
for j in range(0, Ntot, 100000): X[j:j + 100000] -= mu; X[j:j + 100000] /= np.linalg.norm(X[j:j + 100000], axis=1, keepdims=True)
qidx = np.sort(np.random.default_rng(0).choice(Ntot - NQ, NQ, replace=False)); Q = X[qidx].copy(); X[qidx] = X[Ntot - NQ:]
T = X[: Ntot - NQ]; N = len(T); print("N", N, flush=True)
Qg = torch.from_numpy(Q).cuda(); best = None
for j in range(0, N, 100000):
    s = Qg @ torch.from_numpy(T[j:j + 100000]).cuda().T; v, ix = s.topk(10, 1); ix = ix + j
    if best is None: best = (v, ix)
    else:
        vv = torch.cat([best[0], v], 1); ii = torch.cat([best[1], ix], 1); t = vv.topk(10, 1); best = (t.values, ii.gather(1, t.indices))
np.save(f"{FANAR}/data/oai1m_gt.npy", best[1].cpu().numpy()); np.save(f"{FANAR}/data/oai1m_q.npy", Q)
np.save(f"{FANAR}/data/oai1m_base_fp16.npy", T.astype(np.float16)); print("saved", flush=True)
