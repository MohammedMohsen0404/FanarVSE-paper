"""Training-free lattice shell codebooks with 256 codewords (8-bit codes), for use as frozen PQ codebooks.
D4 (4-dim, 2 bits/dim), E8 (8-dim, 1 bit/dim), BW16 Barnes-Wall (16-dim, 0.5 bit/dim)."""
import itertools
import numpy as np


def d4_shell(k=256):
    pts = [v for v in itertools.product(range(-3, 4), repeat=4) if sum(v) % 2 == 0]
    P = np.array(pts, np.float32)
    return P[np.argsort((P ** 2).sum(1), kind="stable")][:k]


def e8_shell(k=256):
    pts = [v for v in itertools.product([-2, -1, 0, 1, 2], repeat=8) if sum(v) % 2 == 0 and sum(a * a for a in v) <= 8]
    pts += [v for v in itertools.product([-1.5, -0.5, 0.5, 1.5], repeat=8) if sum(v) % 2 == 0 and sum(a * a for a in v) <= 8]
    P = np.array(pts, np.float32)
    return P[np.argsort((P ** 2).sum(1), kind="stable")][:k]


def _rm14_weight8():
    """The 30 weight-8 codewords of the first-order Reed-Muller code RM(1,4), length 16."""
    words = set()
    for coef in itertools.product([0, 1], repeat=5):
        w = tuple((coef[0] + sum(coef[i + 1] * ((x >> i) & 1) for i in range(4))) % 2 for x in range(16))
        if sum(w) == 8: words.add(w)
    return sorted(words)


def bw16_shell(k=256, seed=0):
    """Origin + a well-spread subset of the 4320 minimal vectors of the Barnes-Wall lattice (farthest-point selection)."""
    mins = []
    for i, j in itertools.combinations(range(16), 2):
        for si, sj in itertools.product([2, -2], repeat=2):
            v = np.zeros(16, np.float32); v[i], v[j] = si, sj; mins.append(v)
    for w in _rm14_weight8():
        idx = [t for t in range(16) if w[t]]
        for signs in itertools.product([1, -1], repeat=8):
            if signs.count(-1) % 2 == 0:
                v = np.zeros(16, np.float32); v[idx] = signs; mins.append(v)
    M = np.array(mins, np.float32)
    assert len(M) == 4320, len(M)
    rng = np.random.default_rng(seed); chosen = [int(rng.integers(len(M)))]
    dmin = ((M - M[chosen[0]]) ** 2).sum(1)
    while len(chosen) < k - 1:
        c = int(dmin.argmax()); chosen.append(c); dmin = np.minimum(dmin, ((M - M[c]) ** 2).sum(1))
    return np.vstack([np.zeros((1, 16), np.float32), M[chosen]])


SHELLS = {4: d4_shell, 8: e8_shell, 16: bw16_shell}


def bw16_three_shells(k=65536):
    """Barnes-Wall BW16 points with norm^2 <= 12 in the scaling x in Z^16, x mod 2 in RM(1,4), sum(x) = 0 mod 4:
    origin (1) + norm^2 8 (4320) + norm^2 12 (61440) = 65761 points; keep the k nearest the origin (1 bit/dim at k=65536)."""
    pts = [np.zeros(16, np.float32)] + list(bw16_shell(4321)[1:])          # origin + all 4320 minimal vectors
    third = []
    for w in _rm14_weight8():
        sup = [t for t in range(16) if w[t]]; off = [t for t in range(16) if not w[t]]
        for signs in itertools.product([1, -1], repeat=8):
            if signs.count(-1) % 2 == 1:                                     # odd number of minus signs
                base = np.zeros(16, np.float32); base[sup] = signs
                for p in off:
                    for s2 in (2, -2):
                        v = base.copy(); v[p] = s2; third.append(v)
    P = np.vstack([np.array(pts, np.float32), np.array(third, np.float32)])
    assert len(P) == 65761, len(P)
    return P[np.argsort((P ** 2).sum(1), kind="stable")][:k]
