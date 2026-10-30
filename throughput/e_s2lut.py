"""Experiments 85-87: uniform-codebook refinement (re-encoded), nibble-LUT stage-2 kernel: correctness vs the gather kernel on the same
codes, speed (full and 3-tier partial), and end-to-end recall / reads with the uniform code."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import os, json, torch
from cache import *
import tiered_fused, s2lut
e, s1, code, mem = load(); dev = "cuda"
UP = f"{FANAR}/data/oai1m_code_uniform.pt"
if os.path.exists(UP):
    cu = torch.load(UP, weights_only=False)
    for k, v in list(cu.items()):
        if torch.is_tensor(v): cu[k] = v.to(dev)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
        elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
else:
    cu = s2lut.encode_uniform(code, rf.host16, s1["order"].long(), s1["c"]); torch.save({k: v for k, v in cu.items() if k not in ("VP", "cbf", "NIB")}, UP)
s2lut.prepare_u(cu)
Qb = rf.Qg[:500]; QC = Qb @ s1["c"]; pos = e.stage1(Qb, 400, 384)[0]
ref = tiered_fused.score(cu, Qb, QC, pos)                     # gather kernel with the uniform codebook (cbf from prepare_u)
lut = s2lut.score_lut(cu, Qb, QC, pos)
print("est max diff", float((ref[0] - lut[0]).abs().max()), "sd max diff", float((ref[1] - lut[1]).abs().max()), flush=True)
print("gather kernel (full, K=400): %.2f ms" % T(lambda: tiered_fused.score(cu, Qb, QC, pos)))
for BM, nw in ((16, 2), (32, 4), (64, 4), (32, 2), (64, 8)):
    print(f"LUT kernel BM={BM} nw={nw} (full, K=400): %.2f ms" % T(lambda: s2lut.score_lut(cu, Qb, QC, pos, BM=BM, nw=nw)), flush=True)
print("LUT build only: %.2f ms" % T(lambda: s2lut.luts(cu, Qb)))
print("LUT partial 3 tiers K=400: %.2f ms" % T(lambda: s2lut.score_lut(cu, Qb, QC, pos, nt=3)))
# end-to-end quality with the uniform code (gather kernel path of the engine, so only the code changes)
from engine_cell import EngineS
eu = EngineS(s1, cu)
for nprobe, K in ((384, 400), (1024, 800)):
    I, r, C, m, s = eu.bench(K, 2.0, nprobe, k=10, B=500)
    x = dict(tag="uniform_code", nprobe=nprobe, K=K, recall=rf.recall(I, 10), reads=round(float(r.mean()), 2), QPS=round(m))
    print(json.dumps(x), flush=True)
