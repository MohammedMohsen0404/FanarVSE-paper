"""Stage 3 by zero-copy: full fp16 vectors in pinned host memory, read by the GPU over PCIe (UVA) only for ambiguous candidates."""
import torch, triton, triton.language as tl
class HostPtr:
    def __init__(self, t): self.t, self.dtype = t, t.dtype
    def data_ptr(self): return self.t.data_ptr()
def pin(host16, chunk=100000):
    H = torch.empty(host16.shape, dtype=host16.dtype, pin_memory=True)
    for j in range(0, len(host16), chunk): H[j:j + chunk].copy_(host16[j:j + chunk])
    return H

@triton.jit
def _stage3(h_ptr, ids_ptr, amb_ptr, q_ptr, score_ptr, KC, D: tl.constexpr, BD: tl.constexpr):
    """score[q, i] = <x_ids[q,i], q> for ambiguous candidates (one program per candidate; others exit)."""
    i = tl.program_id(0); q = tl.program_id(1)
    if tl.load(amb_ptr + q * KC + i) != 0:
        r = tl.load(ids_ptr + q * KC + i).to(tl.int64)
        acc = tl.zeros((BD,), dtype=tl.float32)
        for d0 in tl.static_range(0, D, BD):
            j = d0 + tl.arange(0, BD)
            jm = j < D
            acc += tl.load(h_ptr + r * D + j, mask=jm, other=0.0).to(tl.float32) * tl.load(q_ptr + q * D + j, mask=jm, other=0.0)
        tl.store(score_ptr + q * KC + i, tl.sum(acc, axis=0))

def stage3(H, ids, amb, Q, score):
    B, KC = ids.shape; D = Q.shape[1]
    _stage3[(KC, B)](HostPtr(H), ids.contiguous(), amb.to(torch.int8).contiguous(), Q.float().contiguous(), score, KC, D=D, BD=512)
    return score
