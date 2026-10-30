"""Round 10 harness: build the final engine once, cache it to disk, and reload it in seconds.
load() -> (e, s1, code, mem); also returns the reference candidates of the 384:400 operating point."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, sys, time, torch
sys.path.insert(0, f"{FANAR}/host_resident")
from h import *
from engine_cell import EngineS
CP = f"{FANAR}/data/oai1m_engine_r10.pt"

def load():
    if os.path.exists(CP):
        d = torch.load(CP, weights_only=False); s1, code, mem = d["s1"], d["code"], d["mem"]
        for D in (s1, code):
            for k, v in D.items():
                if torch.is_tensor(v) and k not in ("rots", "V"): D[k] = v.cuda() if v.device.type == "cpu" and k != "host" else v
                elif isinstance(v, list) and v and torch.is_tensor(v[0]) and k not in ("rots",): D[k] = [x.cuda() for x in v]
        return EngineS(s1, code), s1, code, mem
    e, mem, e3 = build()
    torch.save({"s1": e.s1, "code": e.code, "mem": mem}, CP)
    return e, e.s1, e.code, mem

def T(f, n=20):
    f(); torch.cuda.synchronize(); a = torch.cuda.Event(True); b = torch.cuda.Event(True); a.record()
    for _ in range(n): f()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n

if __name__ == "__main__":
    t0 = time.time(); e, s1, code, mem = load(); print("loaded", round(time.time() - t0, 1), mem)
    I, r, C, m, s = e.bench(400, 2.0, 384, k=10, B=500); print("recall", rf.recall(I, 10), "QPS", round(m))
