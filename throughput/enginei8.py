"""EngineI8: stage 1 = IVF over the engine's cells with int8 codes of the cell residual x - m_c (global PCA basis, R=256, per-vector
scale) on int8 tensor cores, with the engine's exact guess-and-verify selection (block-max first pass + threshold emit);
stage 2 = nibble-LUT tiered refinement (uniform codebooks); stage 3 unchanged. Replaces the 1-bit trellis codes (42 B) by 260 B."""
import numpy as np, torch
from engine_cell import EngineS, EngineC, route
from engine3 import Engine8, Engine4, rf
from s1cell import _static_blocks, _pairs_static
from ivf8c_kern import _first8c, _emit8c
import s2lut, civf8

def pairs_fast(probe, NC, QM, scale=None):
    B, npr = probe.shape; flat = probe.flatten().to(torch.int32); _, o = torch.sort(flat, stable=True)
    qid = (o // npr).to(torch.int32).contiguous(); cnt = torch.zeros(NC, dtype=torch.int32, device=dev).scatter_add_(0, flat.long(), torch.ones_like(flat)); off = torch.cumsum(cnt, 0) - cnt
    fo = flat[o].long(); qc = QM[qid.long(), fo].float()
    if scale is not None: qc = qc * scale[fo]
    return qid, off.to(torch.int32).contiguous(), cnt.contiguous(), qc.contiguous(), o
dev = "cuda"

def build_codes(s1, X16, V, R=256, chunk=65536):
    order = s1["order"].long(); N = len(order); off = s1["off_t"].long(); means = s1["means16"].float()
    cell = torch.searchsorted(off, torch.arange(N, device=dev), right=True) - 1; Vr = V[:, :R].contiguous()
    Z = torch.empty(N, R, dtype=torch.int8, device=dev); zs = torch.empty(N, device=dev)
    for j in range(0, N, chunk):
        x = X16[order[j:j + chunk].cpu()].to(dev).float(); z = (x - means[cell[j:j + chunk]]) @ Vr
        a = z.abs().amax(1).clamp(min=1e-12) / 127
        Z[j:j + chunk] = torch.round(z / a[:, None]).clamp(-127, 127).to(torch.int8); zs[j:j + chunk] = a
    return Z, zs, Vr

class EngineI8(EngineS):
    BQ, BN = 32, 32; PROJ = False; UNSORTED = False; CUDA = False; FASTPAIRS = False; NW = 4; SORTROUTE = False; MSZ = None; ROUTE16 = False; LUTF = False; K2 = None; NT = 3; BETA = None; RHO = None; NROUTE = False; DELTA = None; LUTH = False
    def __init__(self, s1, code, Z, zs, Vr):
        super().__init__(s1, code); self.Z, self.zs, self.Vr = Z, zs, Vr
    def _score(self, Qb, QC, pos, pre=None):
        if pre is None and self.LUTF: pre = s2lut.luts_fused(self.code, Qb)
        return s2lut.score_lut(self.code, Qb, QC, pos, BM=16, nw=2, pre=pre)
    def stage1_i8(self, Qb, QM, probe, K, rho_override=None, M_override=None):
        s1 = self.s1; off_t = s1["off_t"]; NC = len(off_t) - 1; B = len(Qb); R = self.Z.shape[1]; BQ, BN = self.BQ, self.BN
        z = Qb @ self.Vr; qs = (z.abs().amax(1).clamp(min=1e-12) / 127); q8 = torch.round(z / qs[:, None]).clamp(-127, 127).to(torch.int8).contiguous(); qs = qs.float().contiguous()
        pcell, pblk, ps, pe, MB = _static_blocks(off_t, BN); P = len(pcell); M = M_override or self.MSZ or max(4096, 16 * K)
        beta, rho = (1.1, 0.75) if K <= 400 else (0.7, 0.5)
        if self.BETA: beta = self.BETA
        if self.RHO: rho = self.RHO
        if rho_override: beta, rho = 1.5, rho_override
        n1 = min(max(int(np.ceil(beta * K / (self.Z.shape[0] / NC / BN + 1))), 1), probe.shape[1])
        pf0 = pairs_fast if self.FASTPAIRS else _pairs_static
        pf = (lambda a, b, c: pairs_fast(a, b, c, self.mnorm)) if self.NROUTE else pf0
        qid, coff, ccnt, qcp, o = pf(probe[:, :n1].contiguous(), NC, QM)
        bmc = torch.full((B * n1, MB), -1e30, device=dev)
        if self.CUDA:
            d = torch.zeros(1, device=dev); di = torch.zeros(1, dtype=torch.int32, device=dev)
            civf8.launch(0, P, q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, self.Z, self.zs, qcp, d, di, d, di, bmc, 1, MB)
        else: _first8c[(P,)](q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, self.Z, self.zs, qcp, bmc, MB, R=R, BQ=BQ, BN=BN, num_stages=1, num_warps=self.NW)
        inv = torch.empty_like(o); inv[o] = torch.arange(len(o), device=dev); v = bmc[inv].reshape(B, n1 * MB)
        top = v.topk(min(int(np.ceil(rho * K)), v.shape[1]), 1).values; valid = top > -1e29
        t0 = torch.where(valid, top, torch.full_like(top, 1e30)).min(1).values.contiguous()
        cnt = torch.zeros(B, dtype=torch.int32, device=dev); bval = torch.full((B, M), -1e30, device=dev); bpos = torch.zeros((B, M), dtype=torch.int32, device=dev)
        qid, coff, ccnt, qcp, _ = pf(probe, NC, QM)
        if self.CUDA: civf8.launch(1, P, q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, self.Z, self.zs, qcp, t0, cnt, bval, bpos, bmc, M, MB)
        else: _emit8c[(P,)](q8, qs, qid, coff, ccnt, pcell, pblk, ps, pe, self.Z, self.zs, qcp, t0, cnt, bval, bpos, M, R=R, BQ=BQ, BN=BN, num_stages=1, num_warps=self.NW)
        t = bval.topk(K, 1); self._est1 = t.values; return bpos.gather(1, t.indices).long().contiguous(), (cnt >= K) & (cnt <= M)
    def _gpu(self, Qb, K, z, nprobe, k):
        if len(Qb) <= EngineC.SMALL: raise RuntimeError("small batches not wired for EngineI8")
        s1 = self.s1; QC = Qb @ s1["c"]
        if self.NROUTE:
            if not hasattr(self, "meansN"):
                self.meansN = (s1["means16"].float() * s1["inv_mnorm"][:, None]).half().contiguous(); self.mnorm = (1.0 / s1["inv_mnorm"]).float().contiguous()
            QM = Qb.half() @ self.meansN.T                                          # normalised scores: route directly; pair constants rescaled by |m_c|
            probe = QM.topk(nprobe, 1).indices
            pos, ok = self.stage1_i8(Qb, QM, probe, K)
            return self._finish(Qb, QC, pos, ok, K, z, k)
        if self.PROJ:
            if not hasattr(self, "mproj"): self.mproj = ((s1["means16"].float() - s1["c"]) @ s1["W1"].float()).half().T.contiguous()
            if not hasattr(self, "W1h"): self.W1h = s1["W1"].half().contiguous()
            QM = (QC[:, None] + ((Qb.half() @ self.W1h) @ self.mproj).float()).half()
        else: QM = Qb.half() @ s1["means16"].T
        sc = (QM.float() * s1["inv_mnorm"]).half() if self.ROUTE16 else QM.float() * s1["inv_mnorm"]
        probe = sc.sort(1, descending=True).indices[:, :nprobe].contiguous() if self.SORTROUTE else sc.topk(nprobe, 1, sorted=not self.UNSORTED).indices
        pos, ok = self.stage1_i8(Qb, QM, probe, K)
        return self._finish(Qb, QC, pos, ok, K, z, k)
    def _finish(self, Qb, QC, pos, ok, K, z, k):
        s1 = self.s1
        pre = s2lut.luts_fused(self.code, Qb, half=self.LUTH) if self.LUTF else None
        if self.K2 is not None and self.K2 < K:
            pre = pre if pre is not None else s2lut.luts(self.code, Qb)
            pe, _ = s2lut.score_lut(self.code, Qb, QC, pos, nt=self.NT, BM=16, nw=2, pre=pre)
            pos = pos.gather(1, pe.topk(self.K2, 1).indices).contiguous()
        if self.DELTA is not None:
            e1 = self._est1; ncnt = (e1 >= e1[:, 9:10] - self.DELTA).sum(1).to(torch.int32).contiguous()
            pre = pre if pre is not None else s2lut.luts_fused(self.code, Qb)
            est, sg = s2lut.score_lut(self.code, Qb, QC, pos, BM=16, nw=2, pre=pre, ncnt=ncnt)
        else: est, sg = self._score(Qb, QC, pos, pre)
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb)
        return ok, ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def search(self, Qb, K, z, nprobe, k, ret_est=False):
        B = len(Qb); key = (B, K, z, nprobe, k)
        if key not in self.graphs:
            if Engine8.H is None: self._s3(Qb[:1], *[torch.zeros(1, 1, dtype=t, device=dev) for t in (torch.long, torch.float, torch.bool, torch.bool)])
            qs = Qb.clone(); st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3): self._gpu(qs, K, z, nprobe, k)
            torch.cuda.current_stream().wait_stream(st); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): outs = self._gpu(qs, K, z, nprobe, k)
            self.graphs[key] = (g, qs, outs)
        g, qs, (ok, I, nr, ids) = self.graphs[key]
        qs.copy_(Qb); g.replay(); I, nr, ids = I.clone(), nr.clone(), ids.clone()
        if not bool(ok.all()):
            bad = torch.nonzero(~ok).flatten(); self.nbad = getattr(self, "nbad", 0) + len(bad)
            a, b, c = self._fallback(Qb[bad], K, z, nprobe, k); I[bad], nr[bad] = a, b; ids[bad] = c[:, :ids.shape[1]]
        return I, nr, ids
    def _fallback(self, Qb, K, z, nprobe, k):
        s1 = self.s1; QC = Qb @ s1["c"]
        if self.NROUTE: QM = Qb.half() @ self.meansN.T; probe = QM.topk(nprobe, 1).indices      # same tables as the main path
        else: QM = Qb.half() @ s1["means16"].T; probe = (QM.float() * s1["inv_mnorm"]).topk(nprobe, 1).indices
        pos, ok = self.stage1_i8(Qb, QM, probe, K, rho_override=1.0, M_override=65536)
        est, sg = self._score(Qb, QC, pos); lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        ids = s1["order"][pos]; score = self._s3(Qb, ids, est, sure, amb); self.nbad2 = getattr(self, "nbad2", 0) + int((~ok).sum())
        return ids.gather(1, score.topk(k, 1).indices), amb.sum(1), ids
    def gpu_bytes(self, base):
        return base - 42.0 + (self.Z.shape[1] + 4)                               # replaces trellis words+scale (40 B) by int8 codes + fp32 scale
