"""Exact GPU bytes held by EngineI8 v3: every CUDA tensor reachable from its s1 / code dicts and its own arrays (deduplicated by storage)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import torch
from cache import *
import s2lut
from enginei8 import EngineI8, build_codes
e, s1, code, mem = load(); dev = "cuda"
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); cu["fp16_query"] = True; del code
V = torch.load(f"{FANAR}/data/oai1m_pca256.pt")["V"].to(dev); Z, zs, Vr = build_codes(s1, rf.host16, V, R=256); s1["words"] = None; del V; torch.cuda.empty_cache()
E = EngineI8(s1, cu, Z, zs, Vr); E.FASTPAIRS = E.ROUTE16 = E.LUTF = True
E.RHO, E.MSZ, E.BQ, E.NW, E.NROUTE = 0.6, 3072, 16, 2, True; E.DELTA, E.LUTH = 0.07, True
E._gpu(rf.Qg[:500], 175, 1.25, 384, 10)
seen, rows = set(), []
def walk(name, x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda and x.untyped_storage().data_ptr() not in seen:
            seen.add(x.untyped_storage().data_ptr()); rows.append((name, tuple(x.shape), str(x.dtype), x.untyped_storage().nbytes()))
    elif isinstance(x, dict):
        for k, v in x.items(): walk(f"{name}.{k}", v)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): walk(f"{name}[{i}]", v)
walk("s1", s1); walk("code", cu)
for k in ("Z", "zs", "Vr", "meansN", "mnorm"): walk(k, getattr(E, k))
N = len(s1["order"]); tot = sum(r[3] for r in rows)
for r in sorted(rows, key=lambda r: -r[3]): print(f"{r[3] / N:8.2f} B/vec  {r[3] / 1e6:8.2f} MB  {r[0]} {r[1]} {r[2]}")
print(f"TOTAL {tot / N:.1f} B/vec ({tot / 1e6:.1f} MB)")
