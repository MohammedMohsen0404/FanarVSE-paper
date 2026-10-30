"""Experiment 106: full per-query breakdown of the best EngineI8 configuration (B=4000, 384:200, z=1.5, fused LUT), stage-1 parts included."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import numpy as np, torch
from cache import *
import s2lut
from enginei8 import EngineI8, build_codes, pairs_fast
from s1cell import _static_blocks
from ivf8c_kern import _first8c, _emit8c
e, s1, code, mem = load(); dev = "cuda"
cu = torch.load(f"{FANAR}/data/oai1m_code_uniform.pt", weights_only=False)
for k, v in list(cu.items()):
    if torch.is_tensor(v): cu[k] = v.to(dev)
    elif isinstance(v, list) and v and torch.is_tensor(v[0]): cu[k] = [x.to(dev) for x in v]
    elif isinstance(v, dict) and v and torch.is_tensor(next(iter(v.values()))): cu[k] = {a: b.to(dev) for a, b in v.items()}
s2lut.prepare_u(cu); cu["fp16_query"] = True; del code
V = torch.load(f"{FANAR}/data/oai1m_pca256.pt")["V"].to(dev); Z, zs, Vr = build_codes(s1, rf.host16, V, R=256); s1["words"] = None; torch.cuda.empty_cache()
E = EngineI8(s1, cu, Z, zs, Vr); E.FASTPAIRS = E.ROUTE16 = E.LUTF = True
B = 4000; Qb = torch.cat([rf.Qg] * 2); K, nprobe, R, BQ, BN = 200, 384, 256, 32, 32
t = {}
t["QC"] = T(lambda: Qb @ s1["c"]); QC = Qb @ s1["c"]
t["QM matmul"] = T(lambda: Qb.half() @ s1["means16"].T); QM = Qb.half() @ s1["means16"].T
t["route topk fp16"] = T(lambda: (QM.float() * s1["inv_mnorm"]).half().topk(nprobe, 1)); probe = (QM.float() * s1["inv_mnorm"]).half().topk(nprobe, 1).indices
t["q int8 code"] = T(lambda: torch.round((Qb @ Vr) / ((Qb @ Vr).abs().amax(1, keepdim=True) / 127)).to(torch.int8))
z = Qb @ Vr; qs = (z.abs().amax(1) / 127); q8 = torch.round(z / qs[:, None]).clamp(-127, 127).to(torch.int8).contiguous(); qs = qs.float().contiguous()
off_t = s1["off_t"]; NC = len(off_t) - 1; pcell, pblk, ps, pe, MB = _static_blocks(off_t, BN); P = len(pcell); M = 4096
n1 = min(max(int(np.ceil(1.1 * K / (rf.N / NC / BN + 1))), 1), nprobe)
t["pairs n1"] = T(lambda: pairs_fast(probe[:, :n1].contiguous(), NC, QM)); qid, coff, ccnt, qcp, o = pairs_fast(probe[:, :n1].contiguous(), NC, QM)
bmc = torch.full((B * n1, MB), -1e30, device=dev)
t["bmc fill"] = T(lambda: bmc.fill_(-1e30))
t["first pass"] = T(lambda: _first8c[(P,)](q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, Z, zs, qcp, bmc, MB, R=R, BQ=BQ, BN=BN, num_stages=1))
def thr():
    inv = torch.empty_like(o); inv[o] = torch.arange(len(o), device=dev); v = bmc[inv].reshape(B, n1 * MB)
    top = v.topk(int(np.ceil(0.75 * K)), 1).values; valid = top > -1e29; return torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
t["threshold"] = T(thr); t0 = thr()
t["pairs all"] = T(lambda: pairs_fast(probe, NC, QM)); qid2, coff2, ccnt2, qcp2, _ = pairs_fast(probe, NC, QM)
cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
t["emit buffers"] = T(lambda: (cnt.zero_(), bval.fill_(-1e30)))
t["emit"] = T(lambda: _emit8c[(P,)](q8, qs, qid2, coff2, ccnt2, pcell, pblk, ps, pe, Z, zs, qcp2, t0, cnt.zero_(), bval, bpos, M, R=R, BQ=BQ, BN=BN, num_stages=1))
t["final topk"] = T(lambda: bval.topk(K, 1)); pos = bpos.gather(1, bval.topk(K, 1).indices).long()
t["LUT build"] = T(lambda: s2lut.luts_fused(cu, Qb)); pre = s2lut.luts_fused(cu, Qb)
t["stage 2 LUT"] = T(lambda: s2lut.score_lut(cu, Qb, QC, pos, BM=16, nw=2, pre=pre)); est, sg = s2lut.score_lut(cu, Qb, QC, pos, BM=16, nw=2, pre=pre)
lo, hi = est - 1.5 * sg, est + 1.5 * sg
t["select"] = T(lambda: (hi.topk(11, 1), lo.topk(10, 1))); sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure; ids = s1["order"][pos]
t["stage 3"] = T(lambda: E._s3(Qb, ids, est, sure, amb))
print("emitted median", int(cnt.float().median()), "p99", int(cnt.float().quantile(0.99)), "max", int(cnt.max()))
tot = sum(t.values())
for k, v in t.items(): print(f"{k:18s} {v / B * 1000:6.2f} us/q  {100 * v / tot:5.1f}%")
print(f"{'TOTAL':18s} {tot / B * 1000:6.2f} us/q -> {B / tot * 1000:.0f} q/s (unfused sum)")
