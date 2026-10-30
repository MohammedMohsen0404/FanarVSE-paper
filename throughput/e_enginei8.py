"""Experiments 97-99: EngineI8 end to end (int8 cell-residual IVF stage 1 + LUT stage 2), one batch size per process. usage: B"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, torch, numpy as np
from cache import *
import s2lut
from enginei8 import EngineI8, build_codes
e, s1, code, mem = load(); dev = "cuda"; B = int(sys.argv[1])
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); del code; torch.cuda.empty_cache()
V = torch.load(f"{FANAR}/data/oai1m_pca256.pt")["V"].to(dev)
Z, zs, Vr = build_codes(s1, rf.host16, V, R=256)
s1["words"] = None; torch.cuda.empty_cache()                                     # trellis codes not needed by this engine
E = EngineI8(s1, cu, Z, zs, Vr)
Qall = torch.cat([rf.Qg] * max(1, B // 2000))
def row(nprobe, K):
    E.nbad = 0
    out = [E.search(Qall[i:i + B], K, 2.0, nprobe, 10) for i in range(0, len(Qall), B)]
    I = torch.cat([o[0] for o in out])[:2000].cpu().numpy(); r = torch.cat([o[1] for o in out]).float()
    C = torch.cat([o[2] for o in out])[:2000].cpu().numpy()
    ms = min(T(lambda: E.search(Qall[:B], K, 2.0, nprobe, 10), 3) for _ in range(3))
    x = dict(tag="engineI8", R=256, nprobe=nprobe, K=K, batch=B, recall=rf.recall(I, 10), ceiling=rf.cand_recall(C, 10), QPS=round(B / ms * 1000),
             reads=round(float(r.mean()), 2), unrepaired_flags=E.nbad, gpu_B=round(E.gpu_bytes(mem["gpu_B"]), 1))
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
    E.graphs.clear(); torch.cuda.empty_cache()
for nprobe, K in ((256, 100), (384, 100), (384, 200), (512, 200), (768, 200), (1024, 400)): row(nprobe, K)
