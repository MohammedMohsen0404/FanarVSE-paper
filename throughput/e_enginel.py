"""Experiments 88-90: EngineL (LUT stage 2, full and progressive) end to end vs EngineS / EngineQ, batch 500 / 2000 / 4000."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, torch, numpy as np
from cache import *
import s2lut
from enginel import EngineL
from engineq import EngineQ
e, s1, code, mem = load(); dev = "cuda"
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); cu["V"] = cu["V"]
def row(tag, eng, nprobe, K, B=500, **kw):
    Qall = torch.cat([rf.Qg] * max(1, B // 2000))
    I, r, C, m, s = eng.bench(K, 2.0, nprobe, k=10, B=B) if B <= 2000 else (None,) * 5
    if B > 2000:
        out = [eng.search(Qall[i:i + B], K, 2.0, nprobe, 10) for i in range(0, len(Qall), B)]
        I = torch.cat([o[0] for o in out])[:2000].cpu().numpy(); r = torch.cat([o[1] for o in out]).float().cpu()
        ms = min(T(lambda: eng.search(Qall[:B], K, 2.0, nprobe, 10), 3) for _ in range(3)); m = B / ms * 1000
    x = dict(tag=tag, nprobe=nprobe, K=K, batch=B, recall=rf.recall(I, 10), QPS=round(m), reads=round(float(np.asarray(r, dtype=np.float64).mean()), 2), **kw)
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
L = EngineL(s1, cu); LP = EngineL(s1, cu); LP.K2 = 100; Q = EngineQ(s1, code); Q.K2 = 100
for B in (500, 2000, 4000):
    for nprobe, K in ((384, 400), (1024, 800)):
        row("engineS", e, nprobe, K, B); row("engineQ", Q, nprobe, K, B, K2=100); row("engineL", L, nprobe, K, B); row("engineLP", LP, nprobe, K, B, K2=100)
