"""Prepare OpenAI-3-large (1536) and arXiv InstructorXL (768) sets: normalised fp16 base, last 2,000 rows as queries, exact top-10/100 on GPU."""
import numpy as np, torch, pyarrow.parquet as pq, glob
for name, pat, col in (("oai3", "data/oai3/*.parquet", "text-embedding-3-large-1536-embedding"), ("arxiv", "data/arxiv/*.parquet", "vector")):
    X = np.concatenate([np.stack(pq.read_table(f, columns=[col]).column(0).to_numpy(zero_copy_only=False)).astype(np.float32) for f in sorted(glob.glob(pat))])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q, B = X[-2000:], X[:-2000]
    G = torch.from_numpy(B).cuda(); gt = []
    for s in range(0, 2000, 250):
        gt.append((torch.from_numpy(Q[s:s + 250]).cuda() @ G.T).topk(100, 1).indices.cpu().numpy())
    np.save(f"data/{name}_base_fp16.npy", B.astype(np.float16)); np.save(f"data/{name}_q.npy", Q); gt = np.vstack(gt)
    np.save(f"data/{name}_gt.npy", gt[:, :10]); np.save(f"data/{name}_gt100.npy", gt)
    print(name, B.shape, Q.shape)
