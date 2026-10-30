import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, time, torch
sys.path.insert(0, f"{FANAR}/host_resident"); os.environ.setdefault("TAG", "prof23")
from h import *
e, mem, e3 = build(); s1 = e.s1; Qb = rf.Qg[:500]
def t(f, n=20):
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3
for K in [100, 200, 400]:
    pos, QC = e.stage1(Qb, K, 384)
    est, sg = e._score(Qb, QC, pos); lo, hi = est - 2 * sg, est + 2 * sg
    sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure; ids = s1["order"][pos]
    print(dict(K=K, s1_ms=round(t(lambda: e.stage1(Qb, K, 384)), 2), s2_ms=round(t(lambda: e._score(Qb, QC, pos)), 2),
               s3_ms=round(t(lambda: e._s3(Qb, ids, est, sure, amb)), 2), reads=float(amb.sum(1).float().mean()), full_ms=round(t(lambda: e.search(Qb, K, 2.0, 384, 10)), 2)), flush=True)
