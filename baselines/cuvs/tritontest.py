import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); tl.store(x_ptr + i, tl.load(x_ptr + i, mask=i < n) + 1, mask=i < n)
x = torch.zeros(1000, device="cuda"); k[(4,)](x, 1000, B=256); torch.cuda.synchronize(); print("triton ok", float(x.sum()))
