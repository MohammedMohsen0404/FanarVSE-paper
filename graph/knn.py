"""exact kNN graph (k=64, self excluded) of a dataset by brute force on the GPU; saves int32 ids [N,64] and fp16 sims."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, time, numpy as np, torch
DS = sys.argv[1] if len(sys.argv) > 1 else "oai1m"; K = 64
X = torch.from_numpy(np.load(f"{FANAR}/data/{DS}_base_fp16.npy", mmap_mode="r")[:]).cuda(); N = len(X)
I = np.empty((N, K), np.int32); S = np.empty((N, K), np.float16); t0 = time.time(); B = 512
for s in range(0, N, B):
    sim = X[s:s + B] @ X.T; sim[torch.arange(len(sim)), torch.arange(s, s + len(sim))] = -10
    v, i = sim.topk(K, 1); I[s:s + B] = i.int().cpu().numpy(); S[s:s + B] = v.cpu().numpy()
    if s % (B * 200) == 0: print(s, round(time.time() - t0), flush=True)
np.save(f"{FANAR}/data/{DS}_knn64.npy", I); np.save(f"{FANAR}/data/{DS}_knn64_s.npy", S); print("done", time.time() - t0)
