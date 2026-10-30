"""Graph stage 1: one Triton program per query walks a proximity graph, scoring neighbours with compact navigation codes
(int8 projection of x - c on R principal directions, per-vector scale). The beam is a register list of packed int64 keys
(order-preserving score bits << 32 | id << 1 | expanded), re-sorted after every expansion; a per-query visited bitmap in global
memory (atomic_or) guarantees every code is scored once."""
import numpy as np, torch, triton, triton.language as tl
dev = "cuda"

@triton.jit
def _pack(score, ids, ex):
    b = score.to(tl.int32, bitcast=True)
    k32 = tl.where(b >= 0, b, b ^ 0x7FFFFFFF)
    return (k32.to(tl.int64) << 32) | ((ids.to(tl.int64) << 1) | ex.to(tl.int64))

@triton.jit
def _unpack_score(key):
    k32 = (key >> 32).to(tl.int32)
    return tl.where(k32 >= 0, k32, k32 ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)

@triton.jit
def _walk(qz_ptr, qc_ptr, seed_ptr, seeds_ptr, G_ptr, Z_ptr, zs_ptr, vis_ptr, tmp_ptr, out_id, out_sc, nev_ptr, NW, T,
          R: tl.constexpr, DEG: tl.constexpr, E: tl.constexpr, BUF: tl.constexpr, L: tl.constexpr, NS: tl.constexpr, KO: tl.constexpr, DBG: tl.constexpr = 0, RC: tl.constexpr = 32):
    q = tl.program_id(0); NEW: tl.constexpr = E * DEG
    qc = tl.load(qc_ptr + q)
    idx = tl.arange(0, BUF); vb = vis_ptr + q.to(tl.int64) * NW
    # seeds (already scored by the caller): mark visited, fill the beam
    si = tl.load(seeds_ptr + q * NS + tl.arange(0, NS)); ss = tl.load(seed_ptr + q * NS + tl.arange(0, NS))
    tl.atomic_or(vb + (si >> 5), (1 << (si & 31)).to(tl.int32))
    tl.store(tmp_ptr + q * BUF + tl.arange(0, NS), _pack(ss, si, tl.zeros_like(si)))
    tl.debug_barrier()
    key = tl.load(tmp_ptr + q * BUF + idx, mask=idx < NS, other=-9223372036854775807)
    key = tl.sort(key, descending=True)
    nev = 0
    j = tl.arange(0, NEW); jn = j // DEG; jd = j % DEG
    for t in range(T):
        live = (idx < L) & (key > -9223372036854775807)
        unexp = live & ((key & 1) == 0)
        rank = tl.cumsum(unexp.to(tl.int32), 0) - 1
        sel = unexp & (rank < E)
        nid = ((key >> 1) & 0x7FFFFFFF).to(tl.int32)
        node = tl.zeros([NEW], dtype=tl.int32) - 1
        for e in tl.static_range(E):
            ne = tl.sum(tl.where(sel & (rank == e), nid + 1, 0)) - 1
            node = tl.where(jn == e, ne, node)
        key = key | sel.to(tl.int64)
        ok = node >= 0
        nb = tl.load(G_ptr + node.to(tl.int64) * DEG + jd, mask=ok, other=0)
        old = tl.atomic_or(vb + (nb >> 5), (1 << (nb & 31)).to(tl.int32), mask=ok)
        new = ok & (((old >> (nb & 31)) & 1) == 0)
        if DBG == 2: nb = tl.where(new, nb, 0)
        acc = tl.zeros([NEW], dtype=tl.float32); rc = tl.arange(0, RC); zrow = Z_ptr + nb.to(tl.int64)[:, None] * R
        for c0 in tl.static_range(0, R, RC):
            zc = tl.load(zrow + (c0 + rc)[None, :], mask=new[:, None], other=0).to(tl.float32)
            acc += tl.sum(zc * tl.load(qz_ptr + q * R + c0 + rc).to(tl.float32)[None, :], 1)
        sc = qc + tl.load(zs_ptr + nb, mask=new, other=0.0).to(tl.float32) * acc
        nev += tl.sum(new.to(tl.int32))
        kth = tl.min(tl.where(idx < L, key, 9223372036854775807))                                 # beam floor
        tail = _pack(sc, nb, tl.zeros_like(nb))
        tail = tl.where(new & (tail > kth), tail, -9223372036854775807)
        if tl.max(tail) > kth:                                                                       # sort only when the beam changes
            tl.store(tmp_ptr + q * BUF + j, tail)
            tl.debug_barrier()
            tl2 = tl.load(tmp_ptr + q * BUF + (idx - L), mask=idx >= L, other=-9223372036854775807)
            tl.debug_barrier()
            if DBG == 1: key = tl.where(idx < L, key, tl2)
            else: key = tl.sort(tl.where(idx < L, key, tl2), descending=True)
    o = tl.arange(0, KO)
    tl.store(tmp_ptr + q * BUF + idx, key)
    tl.debug_barrier()
    kk = tl.load(tmp_ptr + q * BUF + o)
    tl.store(out_id + q * KO + o, ((kk >> 1) & 0x7FFFFFFF).to(tl.int32))
    tl.store(out_sc + q * KO + o, _unpack_score(kk))
    tl.store(nev_ptr + q, nev)

class GraphS1:
    def __init__(self, Z, zs, W, c, G, nseed=8192, seed=0):
        """Z int8 [N,R], zs fp16 [N] scale, W [d,R] projection, c [d] centre, G int32 [N,DEG]."""
        self.Z, self.zs, self.W, self.c, self.G = Z, zs, W, c, G; self.N, self.R = Z.shape; self.DEG = G.shape[1]
        g = torch.Generator().manual_seed(seed); self.smp = torch.randperm(self.N, generator=g)[:nseed].to(dev).int()
        self.Zs = (Z[self.smp.long()].float() * zs[self.smp.long()].float()[:, None]).half()
        self.NW = (self.N + 31) // 32; self.bufs = {}
    def search(self, Qb, L=256, E=2, T=100, KO=128, NS=64, DBG=0, nw=4, RC=32):
        B = len(Qb); BUF = triton.next_power_of_2(L + E * self.DEG); L = BUF - E * self.DEG
        if B not in self.bufs:
            self.bufs[B] = dict(vis=torch.zeros(B, self.NW, dtype=torch.int32, device=dev), tmp=torch.empty(B, BUF * 2, dtype=torch.int64, device=dev))
        b = self.bufs[B]; b["vis"].zero_()
        qz = Qb @ self.W; qc = Qb @ self.c
        v, i = (qz.half() @ self.Zs.T).float().topk(NS, 1); seeds = self.smp[i].contiguous(); v = (v + qc[:, None]).contiguous()
        oid = torch.empty(B, KO, dtype=torch.int32, device=dev); osc = torch.empty(B, KO, device=dev); nev = torch.empty(B, dtype=torch.int32, device=dev)
        _walk[(B,)](qz.contiguous(), qc.contiguous(), v, seeds, self.G, self.Z, self.zs, b["vis"], b["tmp"], oid, osc, nev, self.NW, T,
                    R=self.R, DEG=self.DEG, E=E, BUF=BUF, L=L, NS=NS, KO=KO, DBG=DBG, RC=RC, num_warps=nw)
        return oid, osc, nev
