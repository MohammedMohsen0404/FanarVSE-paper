"""Experiments 115-118: fused LUT build, progressive LUT stage 2 (K2), projected routing in fp16; B from argv."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, torch
from cache import *
import s2lut
from enginei8 import EngineI8, build_codes
e, s1, code, mem = load(); dev = "cuda"; B = int(sys.argv[1])
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); cu["fp16_query"] = True; del code
V = torch.load(f"{FANAR}/data/oai1m_pca256.pt")["V"].to(dev); Z, zs, Vr = build_codes(s1, rf.host16, V, R=256); s1["words"] = None; torch.cuda.empty_cache()
E = EngineI8(s1, cu, Z, zs, Vr); E.FASTPAIRS = True; E.ROUTE16 = True; Qall = torch.cat([rf.Qg] * max(1, B // 2000))
Qb = Qall[:B]
if False: print("fused LUT max diff", float((a[0] - b[0]).abs().max()), " time old %.2f us/q, fused %.2f us/q" % (T(lambda: s2lut.luts(cu, Qb)) / B * 1000, T(lambda: s2lut.luts_fused(cu, Qb)) / B * 1000), flush=True)
def row(tag, nprobe, K, z=1.5):
    E.nbad = 0; E.graphs.clear(); torch.cuda.empty_cache()
    out = [E.search(Qall[i:i + B], K, z, nprobe, 10) for i in range(0, len(Qall), B)]
    I = torch.cat([o[0] for o in out])[:2000].cpu().numpy(); r = torch.cat([o[1] for o in out]).float()
    ms = min(T(lambda: E.search(Qall[:B], K, z, nprobe, 10), 3) for _ in range(3))
    x = dict(tag=tag, nprobe=nprobe, K=K, z=z, batch=B, recall=rf.recall(I, 10), QPS=round(B / ms * 1000), reads=round(float(r.mean()), 2), fallback=E.nbad,
             lutf=E.LUTF, K2=E.K2, proj=E.PROJ)
    print(json.dumps(x), flush=True)
    with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")
E.LUTF = True; E.K2 = None; E.PROJ = False
E.RHO, E.MSZ, E.BQ, E.NW, E.NROUTE = 0.6, 3072, 16, 2, True
E.DELTA, E.LUTH = 0.07, True
for nprobe, K, M in ((256, 150, 3072), (320, 175, 3072), (352, 175, 3072), (384, 175, 3072), (448, 250, 4096), (512, 300, 4096), (768, 400, 4096), (1024, 400, 4096), (1536, 600, 8192), (2048, 800, 8192)):
    E.MSZ = M; row("v3 FINAL", nprobe, K, 1.25)
print("fallback still failing:", getattr(E, "nbad2", 0))
