import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf
dev = "cuda"
e, _ = make(256); cen = e.s1["cen"]; N = rf.N
SV, S = [], []
for j in range(0, N, 100000):
    t = (rf.host16[j:j + 100000].to(dev).float() @ cen.T).topk(2, 1); SV.append(t.values); S.append(t.indices)
SV, S = torch.cat(SV), torch.cat(S); P, Q2 = S[:, 0], S[:, 1]; gap = SV[:, 0] - SV[:, 1]
G = torch.as_tensor(rf.GT10, device=dev)
rank = torch.cat([(rf.Qg[s:s + 500] @ cen.T).argsort(1, descending=True).argsort(1) for s in range(0, 2000, 500)])
gp = rank.gather(1, P[G].reshape(2000, -1)).reshape(2000, 10); res = []
for f in (0.0, 0.1, 0.2, 0.4):
    sp = gap <= torch.quantile(gap[:200000], f) if f > 0 else torch.zeros(N, dtype=torch.bool, device=dev)
    c2 = torch.bincount(P, minlength=len(cen)).float() + torch.bincount(Q2[sp], minlength=len(cen)).float()
    gs = rank.gather(1, Q2[G].reshape(2000, -1)).reshape(2000, 10); spg = sp[G]
    for npr in (192, 256, 320, 384, 512, 640, 768):
        hit = (gp < npr) | (spg & (gs < npr)); scanned = float(((rank < npr).float() @ c2).mean())
        r = {"spill_frac": f, "nprobe": npr, "coverage@10": round(float(hit.float().mean()), 4), "items_scanned": round(scanned)}
        print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(f"{FANAR}/results/v4_spill_sim2.json", "w"), indent=1)
