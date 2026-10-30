"""E8 lattice nearest-point quantiser (Conway & Sloane, 1982). E8 = D8 U (D8 + 1/2)."""
import torch


def _nearest_d8(x):
    """Nearest point of D8 (integer vectors with even coordinate sum). x: (..., 8)."""
    f = torch.round(x)
    odd = (f.sum(-1) % 2) != 0
    if odd.any():
        err = x - f
        k = err.abs().argmax(-1, keepdim=True)
        step = torch.where(torch.gather(err, -1, k) >= 0, 1.0, -1.0)
        g = f.clone()
        g.scatter_add_(-1, k, step)
        f = torch.where(odd[..., None], g, f)
    return f


def nearest_e8(x):
    """Nearest E8 point to each 8-block. x: (..., 8) -> same shape."""
    a = _nearest_d8(x)
    b = _nearest_d8(x - 0.5) + 0.5
    da = ((x - a) ** 2).sum(-1, keepdim=True)
    db = ((x - b) ** 2).sum(-1, keepdim=True)
    return torch.where(da <= db, a, b)
