"""Experiments 107-108: CUDA WMMA int8 IVF stage 1 + int32 pairing in EngineI8: correctness (same candidates) and end-to-end QPS."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, torch
from cache import *
import s2lut
from enginei8 import EngineI8, build_codes
from engine_cell import route
e, s1, code, mem = load(); dev = "cuda"; B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); cu["fp16_query"] = True; del code
V = torch.load(f"{FANAR}/data/oai1m_pca256.pt")["V"].to(dev); Z, zs, Vr = build_codes(s1, rf.host16, V, R=256); s1["words"] = None; torch.cuda.empty_cache()
E = EngineI8(s1, cu, Z, zs, Vr)
Qb = rf.Qg[:500]; QM = Qb.half() @ s1["means16"].T; probe = route(s1, Qb, QM, 384)
p0, ok0 = E.stage1_i8(Qb, QM, probe, 200); E.CUDA = True; E.FASTPAIRS = True; p1, ok1 = E.stage1_i8(Qb, QM, probe, 200)
same = sum(len(set(a) & set(b)) for a, b in zip(p0.tolist(), p1.tolist())) / p0.numel()
print("candidate overlap Triton vs CUDA:", round(same, 5), "ok", int(ok0.sum()), int(ok1.sum()), flush=True)
Qb = rf.Qg[:B] if B <= 2000 else torch.cat([rf.Qg] * (B // 2000)); QM = Qb.half() @ s1["means16"].T; probe = route(s1, Qb, QM, 384)
for cuda, fp in ((False, False), (False, True), (True, True)):
    E.CUDA, E.FASTPAIRS = cuda, fp; print(f"stage1 CUDA={cuda} fastpairs={fp}: {T(lambda: E.stage1_i8(Qb, QM, probe, 200)) / B * 1000:.2f} us/q", flush=True)
Qall = torch.cat([rf.Qg] * max(1, B // 2000))
def row(tag, nprobe, K):
    E.nbad = 0; E.graphs.clear(); torch.cuda.empty_cache()
    out = [E.search(Qall[i:i + B], K, 2.0, nprobe, 10) for i in range(0, len(Qall), B)]
    I = torch.cat([o[0] for o in out])[:2000].cpu().numpy(); r = torch.cat([o[1] for o in out]).float()
    ms = min(T(lambda: E.search(Qall[:B], K, 2.0, nprobe, 10), 3) for _ in range(3))
    x = dict(tag=tag, nprobe=nprobe, K=K, batch=B, recall=rf.recall(I, 10), QPS=round(B / ms * 1000), reads=round(float(r.mean()), 2), unrepaired=E.nbad)
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
E.CUDA, E.FASTPAIRS = True, True
for nprobe, K in ((256, 100), (384, 100), (384, 150), (384, 200), (512, 200), (768, 200), (1024, 400)): row("I8 CUDA", nprobe, K)
