"""Prepare the large/image sets: full arXiv InstructorXL (~2.2M, 768), ImageNet-1k SigLIP (1.28M, 768), Met DINOv2-giant (1536).
L2-normalised fp16 base, 2,000 random held-out queries (seed 0), exact top-10/100 inner product on the GPU. usage: v16_prep.py {arxivL|siglip|dino}"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, numpy as np, torch, pyarrow.parquet as pq
name = sys.argv[1]
def norm16(A):
    A = A.astype(np.float32); A = A[np.isfinite(A).all(1) & (np.linalg.norm(A, axis=1) > 0)]
    return (A / np.linalg.norm(A, axis=1, keepdims=True)).astype(np.float16)
if name == "siglip":
    with np.load(f"{FANAR}/data/siglip/train_embeddings.npz", allow_pickle=False) as z:
        print({k: (z[k].shape, z[k].dtype) for k in z.keys()}); key = [k for k in z.keys() if z[k].ndim == 2][0]; E = z[key]
    X = np.concatenate([norm16(E[j:j + 200000]) for j in range(0, len(E), 200000)]); del E
else:
    pat, col = {"arxivL": (f"{FANAR}/data/arxiv/part*.parquet", "vector"), "dino": (f"{FANAR}/data/dino/part*.parquet", "embedding")}[name]
    X = np.concatenate([norm16(np.stack(pq.read_table(f, columns=[col]).column(0).to_numpy(zero_copy_only=False))) for f in sorted(glob.glob(pat))])
perm = np.random.default_rng(0).permutation(len(X)); qi = np.sort(perm[:2000]); keep = np.ones(len(X), bool); keep[qi] = False
Q = X[qi].astype(np.float32); Q /= np.linalg.norm(Q, axis=1, keepdims=True); B = X[keep]; del X
gt = []; Qg = torch.from_numpy(Q).cuda()
for s in range(0, 2000, 500):
    bs, bi = None, None
    for j in range(0, len(B), 200000):
        S = Qg[s:s + 500] @ torch.from_numpy(B[j:j + 200000]).cuda().float().T; v, i = S.topk(100, 1); i += j
        if bs is None: bs, bi = v, i
        else: c = torch.cat([bs, v], 1); o = c.topk(100, 1).indices; bs, bi = c.gather(1, o), torch.cat([bi, i], 1).gather(1, o)
    gt.append(bi.cpu().numpy())
gt = np.vstack(gt)
np.save(f"{FANAR}/data/{name}_base_fp16.npy", B); np.save(f"{FANAR}/data/{name}_q.npy", Q)
np.save(f"{FANAR}/data/{name}_gt.npy", gt[:, :10]); np.save(f"{FANAR}/data/{name}_gt100.npy", gt)
print(name, B.shape, Q.shape)
