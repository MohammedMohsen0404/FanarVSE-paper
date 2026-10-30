"""Experiments 77-79: speed of the routing-free int8 scan (GEMM only / with emit), tile sweep, sample-threshold failure rate, candidate recall."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch, json
from cache import *
import brute
e, s1, code, mem = load(); dev = "cuda"
mu = torch.from_numpy(rf.X[:200000].astype(np.float32).mean(0)).to(dev)
S = torch.from_numpy(rf.X[::10].astype(np.float32)).to(dev) - mu; ev, V = torch.linalg.eigh(torch.cov(S.T)); V = V.flip(1).contiguous(); del S
torch.save({"mu": mu.cpu(), "V": V[:, :256].cpu()}, f"{FANAR}/data/oai1m_pca256.pt")
GT = rf.GT10[:, :10]
def recall(ids):
    ids = ids.cpu().numpy(); return float(np.mean([len(set(ids[i]) & set(GT[i])) / 10 for i in range(len(ids))]))
for R in (128, 64, 256):
    b = brute.Brute(rf.host16, mu, V, R=R); Qb = rf.Qg[:500]; q, qs = b.qcode(Qb)
    for BQ, BN, nw in ((64, 128, 4), (128, 128, 8), (64, 256, 8), (128, 64, 4), (32, 128, 4)):
        try:
            g = T(lambda: b.search(Qb, 400, BQ=BQ, BN=BN, nw=nw, mode=2)); m = T(lambda: b.search(Qb, 400, BQ=BQ, BN=BN, nw=nw, mode=1))
            print(f"R={R} BQ={BQ} BN={BN} nw={nw}: GEMM-only path {g:.2f} ms, with emit {m:.2f} ms", flush=True)
        except Exception as ex: print(R, BQ, BN, nw, "fail", str(ex)[:80])
    for K, over in ((400, 2.0), (400, 3.0), (1000, 2.0)):
        P, F, C = [], 0, []
        for s in range(0, 2000, 500):
            pos, est, ok, cnt = b.search(rf.Qg[s:s + 500], K, over=over); P.append(pos); F += int((~ok).sum()); C.append(cnt)
        cnt = torch.cat(C).float()
        x = dict(tag="brute", R=R, K=K, over=over, cand_recall=recall(torch.cat(P)), flagged=F, emitted_p1=int(cnt.quantile(0.01)), emitted_p50=int(cnt.median()), emitted_p99=int(cnt.quantile(0.99)))
        print(json.dumps(x), flush=True)
        with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
    del b; torch.cuda.empty_cache()
