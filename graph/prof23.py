"""time of stage 2+3 alone (K candidates per query, batch 500) on the final engine, vs the full search."""
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
for K in [200, 400]:
    pos = torch.randint(0, rf.N, (500, K), device="cuda")
    def s23():
        QC = Qb @ s1["c"]; est, sg = e._score(Qb, QC, pos); lo, hi = est - 2 * sg, est + 2 * sg
        sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure
        return e._s3(Qb, s1["order"][pos], est, sure, amb)
    full = t(lambda: e.search(Qb, K, 2.0, 384, 10))
    print(dict(K=K, stage23_ms=round(t(s23), 2), full_ms=round(full, 2)), flush=True)
