"""Experiments 93-95: EngineX end to end (one process per batch size to avoid graph-pool build-up). usage: e_enginex.py B"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, torch, numpy as np
from cache import *
import s2lut
from enginex import EngineX
from enginel import EngineL
e, s1, code, mem = load(); dev = "cuda"; B = int(sys.argv[1])
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu)
del code; torch.cuda.empty_cache()
Qall = torch.cat([rf.Qg] * max(1, B // 2000))
def row(tag, eng, nprobe, K, **kw):
    eng.nbad = 0
    out = [eng.search(Qall[i:i + B], K, 2.0, nprobe, 10) for i in range(0, len(Qall), B)]
    I = torch.cat([o[0] for o in out])[:2000].cpu().numpy(); r = torch.cat([o[1] for o in out]).float()
    ms = min(T(lambda: eng.search(Qall[:B], K, 2.0, nprobe, 10), 3) for _ in range(3))
    x = dict(tag=tag, nprobe=nprobe, K=K, batch=B, recall=rf.recall(I, 10), QPS=round(B / ms * 1000), reads=round(float(r.mean()), 2), outside_fallback=eng.nbad, **kw)
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
L = EngineL(s1, cu); X = EngineX(s1, cu)
for nprobe, K in ((384, 400), (1024, 800)):
    row("engineL", L, nprobe, K); row("engineX", X, nprobe, K)
    L.graphs.clear(); X.graphs.clear(); torch.cuda.empty_cache()
for mq, M in ((0.03, 4096), (0.01, 8192), (0.003, 8192)):
    Y = EngineX(s1, cu); Y.MARGIN_Q, Y.MSIZE = mq, M; row("engineX", Y, 384, 400, margin_q=mq, M=M); Y.graphs.clear(); torch.cuda.empty_cache()
