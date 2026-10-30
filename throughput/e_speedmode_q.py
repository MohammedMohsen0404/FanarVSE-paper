"""Experiment 82: quality of a 'speed mode': IVF (the engine's 4096 cells, centroid routing) + int8 PCA-R codes (R=128/256) for
ranking + exact rerank of the top Kr from fp16 vectors. Reports recall@10 against (n_p, R, Kr)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch
from cache import *
e, s1, code, mem = load(); dev = "cuda"
N = rf.N; off = s1["off_t"].long(); order = s1["order"].long(); NC = len(off) - 1
P = torch.load(f"{FANAR}/data/oai1m_pca256.pt"); mu, V = P["mu"].to(dev), P["V"].to(dev)
Xc = torch.empty(N, rf.d, dtype=torch.float16, device=dev)
for j in range(0, N, 131072): Xc[j:j + 131072] = rf.host16[order[j:j + 131072].cpu()].to(dev)        # cell order
Z = {}
for R in (128, 256):
    z8 = torch.empty(N, R, dtype=torch.int8, device=dev); zs = torch.empty(N, device=dev)
    for j in range(0, N, 65536):
        z = (Xc[j:j + 65536].float() - mu) @ V[:, :R]; a = z.abs().amax(1).clamp(min=1e-12) / 127
        z8[j:j + 65536] = torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8); zs[j:j + 65536] = a
    Z[R] = (z8, zs)
Q = rf.Qg; GT = rf.GT10[:, :10]
QMn = (Q.half() @ s1["means16"].T).float() * s1["inv_mnorm"]
def recall(ids):
    ids = ids.cpu().numpy(); return float(np.mean([len(set(ids[i]) & set(GT[i])) / 10 for i in range(len(ids))]))
lens = (off[1:] - off[:-1])
for nprobe in (128, 256, 384, 768):
    probe = QMn.topk(nprobe, 1).indices
    for R in (128, 256):
        z8, zs = Z[R]; res = {Kr: [] for Kr in (10, 20, 32, 64, 128)}
        for i in range(len(Q)):
            pos = torch.cat([torch.arange(off[c], off[c + 1], device=dev) for c in probe[i].tolist()])
            qz = ((Q[i] - mu) @ V[:, :R]); sc = (z8[pos].float() @ qz) * zs[pos]
            top = pos[sc.topk(128).indices]; ex = Xc[top].float() @ Q[i]
            for Kr in res: res[Kr].append(order[top[:Kr][ex[:Kr].topk(10).indices]])
        print(f"n_p={nprobe:4d} R={R}: " + "  ".join(f"rerank {Kr}: {recall(torch.stack(v)):.4f}" for Kr, v in res.items()), flush=True)
