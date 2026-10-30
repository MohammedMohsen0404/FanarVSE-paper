"""Exact GPU bytes held by the final engine: every CUDA tensor reachable from the engine's s1 and code dicts (deduplicated by storage)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, torch
sys.path.insert(0, f"{FANAR}/host_resident")
from h import *
from engine_cell import slim
e, mem, e3 = build(); slim(e.code)
for B in (500, 16, 1):
    I, r, C, m, sd = e.bench(400, 2.0, 384, B=B, nq=2000 if B > 1 else 400); print("batch", B, "recall", rf.recall(I, nq=len(I)) if B > 1 else None, "QPS", round(m), flush=True)
seen, rows = set(), []
def walk(name, x):
    if isinstance(x, torch.Tensor):
        if x.is_cuda and x.untyped_storage().data_ptr() not in seen:
            seen.add(x.untyped_storage().data_ptr()); rows.append((name, tuple(x.shape), str(x.dtype), x.untyped_storage().nbytes()))
    elif isinstance(x, dict):
        for k, v in x.items(): walk(f"{name}.{k}", v)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): walk(f"{name}[{i}]", v)
walk("s1", e.s1); walk("code", e.code)
N = len(e.s1["order"]); tot = sum(r[3] for r in rows)
for r in sorted(rows, key=lambda r: -r[3]): print(f"{r[3] / N:8.2f} B/vec  {r[3] / 1e6:8.2f} MB  {r[0]} {r[1]} {r[2]}")
print(f"TOTAL {tot / N:.1f} B/vec ({tot / 1e6:.1f} MB); torch allocated {torch.cuda.memory_allocated() / 1e6:.1f} MB")
