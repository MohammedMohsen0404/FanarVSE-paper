"""Experiments 74-76: progressive stage 2 end to end (CUDA graph), alternating with EngineS; recall frontier; batch 2000."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, torch
from cache import *
from engineq import EngineQ
e, s1, code, mem = load()
def row(tag, eng, nprobe, K, B=500, **kw):
    I, r, C, m, s = eng.bench(K, 2.0, nprobe, k=10, B=B)
    x = dict(tag=tag, nprobe=nprobe, K=K, batch=B, recall=rf.recall(I, 10), QPS=round(m), QPS_std=round(s), reads=round(float(r.mean()), 2), **kw)
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
cfgs = [(256, 200, 80), (384, 400, 100), (384, 400, 60), (512, 400, 100), (768, 400, 100), (1024, 800, 150), (1536, 800, 150)]
for nprobe, K, K2 in cfgs:
    q = EngineQ(s1, code); q.K2 = K2
    for rep in range(2):
        row("engineS", e, nprobe, K); row("engineQ", q, nprobe, K, K2=K2)
for nprobe, K, K2 in ((384, 400, 100), (1024, 800, 150)):
    q = EngineQ(s1, code); q.K2 = K2; row("engineS", e, nprobe, K, B=2000); row("engineQ", q, nprobe, K, B=2000, K2=K2)
