"""Spilling simulation: coverage of the true top-10 vs items scanned, primary-only vs primary+secondary assignment
(secondary = 2nd nearest centroid, or SOAR-style: argmax <x,c> - lam * |<r, c-c1>|-like orthogonality penalty), with spill fraction f."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf
dev = "cuda"
e, _ = make(256); cen = e.s1["cen"]; N = rf.N
S = torch.cat([(rf.host16[j:j + 100000].to(dev).float() @ cen.T).topk(8, 1).indices for j in range(0, N, 100000)])          # (N, 8) nearest cells
P = S[:, 0]
# SOAR: secondary cell minimising |<r, r'>| where r = x - c1, r' = x - c; approximated by score <x,c> - lam * <r, c - c1>^2 / |c - c1|^2 among top-8
X = None
def soar(lam):
    out = torch.empty(N, dtype=torch.long, device=dev)
    for j in range(0, N, 50000):
        x = rf.host16[j:j + 50000].to(dev).float(); c1 = cen[S[j:j + 50000, 0]]; r = x - c1; cand = S[j:j + 50000, 1:]
        cc = cen[cand]; sc = (x[:, None, :] * cc).sum(-1); diff = cc - c1[:, None, :]
        par = ((r[:, None, :] * diff).sum(-1)) ** 2 / (diff.norm(dim=-1) ** 2 + 1e-9) / (r.norm(dim=1, keepdim=True) ** 2 + 1e-9)
        out[j:j + 50000] = cand.gather(1, (sc - lam * par).argmax(1, keepdim=True)).squeeze(1)
    return out
cnt = torch.bincount(P, minlength=len(cen)).float()
G = torch.as_tensor(rf.GT10, device=dev)
rank = torch.cat([(rf.Qg[s:s + 500] @ cen.T).argsort(1, descending=True).argsort(1) for s in range(0, 2000, 500)])        # rank of each cell per query
gp = rank.gather(1, P[G].reshape(2000, -1)).reshape(2000, 10)
res = []
for name, sec in (("primary", None), ("2nd-nearest", S[:, 1]), ("soar lam=1", soar(1.0)), ("soar lam=4", soar(4.0))):
    gs = None if sec is None else rank.gather(1, sec[G].reshape(2000, -1)).reshape(2000, 10)
    c2 = cnt if sec is None else cnt + torch.bincount(sec, minlength=len(cen)).float()
    for npr in (128, 192, 256, 384, 512, 768, 1024):
        hit = gp < npr if gs is None else (gp < npr) | (gs < npr)
        top = rank < npr; scanned = float((top.float() @ c2).mean())
        r = {"assign": name, "nprobe": npr, "coverage@10": round(float(hit.float().mean()), 4), "items_scanned": round(scanned)}
        print(json.dumps(r), flush=True); res.append(r)
json.dump(res, open(f"{FANAR}/results/v4_spill_sim.json", "w"), indent=1)
