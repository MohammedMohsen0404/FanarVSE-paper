"""Experiment 76: candidate recall of brute force over ALL vectors with low-dimensional codes (global PCA of x - mu, R dims),
float and int8 (per-vector scale), top-K by score. If high, routing can be dropped: an int8 GEMM over 1M x R is cheap on tensor cores."""
import numpy as np, torch
from cache import *
e, s1, code, mem = load(); dev = "cuda"
X16 = rf.host16; N = rf.N; Q = rf.Qg; GT = rf.GT10[:, :10]
mu = torch.from_numpy(rf.X[:200000].astype(np.float32).mean(0)).to(dev)
S = torch.from_numpy(rf.X[::10].astype(np.float32)).to(dev) - mu; ev, V = torch.linalg.eigh(torch.cov(S.T)); V = V.flip(1).contiguous(); del S
RM = 256; Z = torch.empty(N, RM, dtype=torch.float16, device=dev)
for j in range(0, N, 65536): Z[j:j + 65536] = ((X16[j:j + 65536].to(dev).float() - mu) @ V[:, :RM]).half()
Qz = (Q - mu) @ V[:, :RM]; qmu = Q @ mu
def recall(ids):
    ids = ids.cpu().numpy(); return np.mean([len(set(ids[i]) & set(GT[i])) / 10 for i in range(len(GT))])
Ks = (400, 1000, 2000, 4000, 8000)
for R in (32, 64, 96, 128, 192, 256):
    for mode in ("fp16", "int8"):
        if mode == "fp16": Zr = Z[:, :R]
        else:
            a = Z[:, :R].float().abs().amax(1, keepdim=True) / 127; Zr = (torch.round(Z[:, :R].float() / a).clamp(-127, 127) * a).half()
        top = []
        for s in range(0, 2000, 250):
            sc = (Qz[s:s + 250, :R].half() @ Zr.T).float(); top.append(sc.topk(max(Ks), 1).indices)
        top = torch.cat(top)
        print(f"R={R:3d} {mode}: " + "  ".join(f"K={K}: {recall(top[:, :K]):.4f}" for K in Ks), flush=True)
