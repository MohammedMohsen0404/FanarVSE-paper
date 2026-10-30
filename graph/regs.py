import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, torch, numpy as np
sys.path.insert(0, f"{FANAR}/graph"); import gkern
N, R, DEG = 100000, 256, 32
Z = torch.zeros(N, R, dtype=torch.int8, device="cuda"); zs = torch.ones(N, dtype=torch.float16, device="cuda")
G = torch.randint(0, N, (N, DEG), dtype=torch.int32, device="cuda"); W = torch.randn(1536, R, device="cuda"); c = torch.zeros(1536, device="cuda")
g = gkern.GraphS1(Z, zs, W, c, G, nseed=1024)
for nw in [2, 4, 8]:
    for E in [2, 4]:
        [g.search(torch.randn(64, 1536, device="cuda"), 256, E, 10, 128, nw=nw, RC=rc) for rc in (16, 32, 64)]
for k, v in gkern._walk.cache[0].items():
    print(str(k)[-120:], "regs", v.n_regs, "spills", v.n_spills, "smem", v.metadata.shared)
