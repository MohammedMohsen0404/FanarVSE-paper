"""Bitshift-trellis quantizer (QTIP-style) for vector search: data-oblivious, k bits/dim, L-bit sliding state.
State after coordinate j = last L bits of the bitstream; coordinate value = alpha * T[state].  T: 2^L pseudo-random
Gaussian values (fixed table). Encoding = Viterbi over each segment (min squared error). Decoding is parallel:
value_j depends only on a window of L bits. Segments carry their own L-bit initial state."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import math, numpy as np, torch

dev = "cuda"


def make_table(L, seed=0):
    from scipy.stats import norm
    S = 2 ** L; perm = np.random.default_rng(seed).permutation(S)
    return torch.tensor(norm.ppf((perm + 0.5) / S), dtype=torch.float32, device=dev)


@torch.no_grad()
def viterbi(Xs, T, L, k, alpha):
    """Xs: (B, n) segment values. Returns reconstruction (B, n) and bits used (n*k + L per segment)."""
    B, n = Xs.shape; S = 2 ** L; K = 2 ** k; vals = alpha * T                    # value emitted on entering state s'
    sp = torch.arange(S, device=dev)
    preds = torch.stack([(sp >> k) | (u << (L - k)) for u in range(K)], 1)      # (S, K) predecessor states of s'
    cost = (Xs[:, :1] - vals[None, :]) ** 2                                      # first coordinate: any start state
    back = torch.empty(n - 1, B, S, dtype=torch.uint8, device=dev)
    for j in range(1, n):
        cand = cost[:, preds]                                                    # (B, S, K)
        best, arg = cand.min(2); back[j - 1] = arg.to(torch.uint8)
        cost = best + (Xs[:, j:j + 1] - vals[None, :]) ** 2
    s = cost.argmin(1); path = torch.empty(B, n, dtype=torch.long, device=dev); path[:, -1] = s
    for j in range(n - 1, 0, -1):
        s = preds[s, back[j - 1].gather(1, s[:, None])[:, 0].long()]; path[:, j - 1] = s
    return vals[path]


def lloyd_max(k):
    import faiss
    g = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32)
    km = faiss.Kmeans(1, 2 ** k, niter=60, seed=1); km.train(g); return torch.tensor(np.sort(km.centroids[:, 0]), device=dev)


if __name__ == "__main__":
    import json
    torch.manual_seed(0); res = []
    G = torch.randn(4096, 256, device=dev)                                       # 1M Gaussian coordinates, segments of 256
    for k in (1, 2):
        cb = lloyd_max(k); thr = (cb[1:] + cb[:-1]) / 2; dl = float(((G - cb[torch.bucketize(G, thr)]) ** 2).mean())
        shannon = 2 ** (-2 * k)
        r = {"k": k, "Shannon": shannon, "LloydMax(TurboQuant)": round(dl, 4), "LM_gap_dB": round(10 * math.log10(dl / shannon), 2)}
        for L in (8, 10, 12):
            if L <= k: continue
            T = make_table(L); best = None
            for a in np.linspace(0.6, 1.3, 15):
                Y = viterbi(G[:512], T, L, k, float(a)); m = float(((G[:512] - Y) ** 2).mean())
                if best is None or m < best[0]: best = (m, float(a))
            Y = viterbi(G[:2048], T, L, k, best[1]); m = float(((G[:2048] - Y) ** 2).mean()); m_eff = m
            rate = k + L / 256                                                     # include per-segment initial state
            r[f"trellis_L{L}"] = round(m, 4); r[f"trellis_L{L}_gap_dB"] = round(10 * math.log10(m / shannon), 2); r[f"trellis_L{L}_rate"] = round(rate, 3)
        print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(f"{FANAR}/results/trellis_gaussian.json", "w"), indent=1)


def make_table_v(L, V, seed=0):
    """2^L pseudo-random Gaussian V-dim codewords (fixed, data-oblivious)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(2 ** L, V, generator=g).to(dev)


@torch.no_grad()
def viterbi_v(Xs, T, L, k, alpha):
    """Bitshift trellis emitting a V-dim codeword per k bits (rate k/V bits per dim). Xs: (B, n*V)."""
    B = Xs.shape[0]; V = T.shape[1]; n = Xs.shape[1] // V; S = 2 ** L; K = 2 ** k; vals = alpha * T
    Xg = Xs.reshape(B, n, V); sp = torch.arange(S, device=dev)
    preds = torch.stack([(sp >> k) | (u << (L - k)) for u in range(K)], 1)
    emit = lambda j: ((Xg[:, j, None, :] - vals[None]) ** 2).sum(-1)               # (B, S)
    cost = emit(0); back = torch.empty(n - 1, B, S, dtype=torch.uint8, device=dev)
    for j in range(1, n):
        best, arg = cost[:, preds].min(2); back[j - 1] = arg.to(torch.uint8); cost = best + emit(j)
    s = cost.argmin(1); path = torch.empty(B, n, dtype=torch.long, device=dev); path[:, -1] = s
    for j in range(n - 1, 0, -1):
        s = preds[s, back[j - 1].gather(1, s[:, None])[:, 0].long()]; path[:, j - 1] = s
    return vals[path].reshape(B, n * V)
