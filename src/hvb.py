"""Holographic Vector Base (HVB): core library.

A hologram stores m vectors in ONE complex spectrum of the same size as a single vector:
    H = sum_i  K_i * rfft(v_i)          (K_i: unit-modulus phase key of slot i)
Scoring a query against every slot of a hologram is a matched filter (unbind + dot):
    s_i = <q, irfft(conj(K_i) * H)>  =  v_i . q  +  crosstalk
Insert = add, delete = subtract (exact up to float error). Holograms of disjoint shards merge by addition.
"""
import numpy as np


def rfft_weights(d):
    """Parseval weights so that <a,b> = (1/d) * sum_f w_f Re(A_f conj(B_f)) for real signals."""
    F = d // 2 + 1
    w = np.full(F, 2.0, dtype=np.float32)
    w[0] = 1.0
    if d % 2 == 0:
        w[-1] = 1.0
    return w


class Prep:
    """Centre, optional partial PCA whitening (power alpha), L2-normalise. Cosine == inner product after prep."""

    def __init__(self, alpha=0.0, rank=None):
        self.alpha, self.rank = alpha, rank

    def fit(self, X):
        self.mu = X.mean(0, keepdims=True)
        if self.alpha > 0 or self.rank:
            C = np.cov((X - self.mu)[: min(len(X), 200_000)].T)
            ev, V = np.linalg.eigh(C)
            ev, V = ev[::-1], V[:, ::-1]
            r = self.rank or X.shape[1]
            self.V = V[:, :r].astype(np.float32)
            self.scale = (np.maximum(ev[:r], 1e-9) ** (-self.alpha / 2)).astype(np.float32)
        else:
            self.V = None
        return self

    def __call__(self, X):
        Z = X - self.mu
        if self.V is not None:
            Z = (Z @ self.V) * self.scale
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return Z.astype(np.float32)


def make_keys(m, d, seed=0):
    """m random phase keys (rfft domain). DC / Nyquist bins are real (+-1) to keep irfft consistent."""
    F = d // 2 + 1
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, size=(m, F)).astype(np.float32)
    K = np.exp(1j * ph).astype(np.complex64)
    K[:, 0] = np.sign(rng.standard_normal(m)).astype(np.float32)
    if d % 2 == 0:
        K[:, -1] = np.sign(rng.standard_normal(m)).astype(np.float32)
    return K


class HoloIndex:
    """Flat holographic store: items i -> (group i//m, slot i%m). All holograms share one key bank."""

    def __init__(self, d, m, seed=0, dtype=np.complex64):
        self.d, self.m, self.F = d, m, d // 2 + 1
        self.K = make_keys(m, d, seed)
        self.w = rfft_weights(d)
        self.dtype = dtype
        # real matrix for scoring: [Re K ; -Im K]
        self.Kcat = np.concatenate([self.K.real, -self.K.imag], axis=1).T.copy()  # (2F, m)
        self.n = 0
        self.H = np.zeros((0, self.F), dtype=np.complex64)

    # ---------- build / update ----------
    def add(self, X):
        """Append rows X (n,d) (already prepped). Returns ids."""
        n0 = self.n
        ids = np.arange(n0, n0 + len(X))
        G = (ids // self.m)
        S = (ids % self.m)
        need = G.max() + 1
        if need > len(self.H):
            self.H = np.concatenate([self.H, np.zeros((need - len(self.H), self.F), np.complex64)])
        Xf = np.fft.rfft(X, axis=1).astype(np.complex64) * self.K[S]
        np.add.at(self.H, G, Xf) if len(X) < 2000 else self._seg_add(G, Xf)
        self.n += len(X)
        return ids

    def _seg_add(self, G, Xf):
        order = np.argsort(G, kind="stable")
        Gs, Xs = G[order], Xf[order]
        cut = np.flatnonzero(np.diff(Gs)) + 1
        starts = np.r_[0, cut]
        self.H[Gs[starts]] += np.add.reduceat(Xs, starts, axis=0)

    def remove(self, ids, X):
        """Exact deletion by subtraction (caller supplies the stored vectors)."""
        ids = np.asarray(ids)
        Xf = np.fft.rfft(X, axis=1).astype(np.complex64) * self.K[ids % self.m]
        np.subtract.at(self.H, ids // self.m, Xf)

    # ---------- search ----------
    def scores(self, q):
        """Approximate <q, v_i> for ALL stored items. q: (d,) prepped."""
        Q = np.fft.rfft(q).astype(np.complex64) * self.w
        P = Q[None, :] * np.conj(self.H)  # (G, F)
        Pcat = np.concatenate([P.real, P.imag], axis=1)  # (G, 2F)
        S = (Pcat @ self.Kcat) / self.d  # (G, m)
        return S.reshape(-1)[: self.n]

    def bytes_per_vector(self, bits=32):
        return (2 * self.F * bits / 8) / self.m


# ======================================================================================
# Spectral (Wiener / LMMSE) holographic decoding on a PCA-reduced space
# ======================================================================================
def pca_basis(Xp, n_fit=200_000):
    C = np.cov(Xp[:n_fit].T)
    ev, V = np.linalg.eigh(C)
    return ev[::-1].astype(np.float32).clip(1e-9), V[:, ::-1].astype(np.float32)


def quantize_spectrum(H, bits):
    """Per-group symmetric uniform quantisation of Re/Im parts (simulates int-`bits` storage)."""
    if bits >= 32:
        return H
    if bits == 16:
        return (H.real.astype(np.float16).astype(np.float32) + 1j * H.imag.astype(np.float16).astype(np.float32)).astype(np.complex64)
    L = 2 ** (bits - 1) - 1
    sc = np.maximum(np.abs(np.concatenate([H.real, H.imag], 1)).max(1, keepdims=True), 1e-12) / L
    return (np.round(H.real / sc) * sc + 1j * np.round(H.imag / sc) * sc).astype(np.complex64)


class SpectralHolo:
    """PCA-reduce to r dims, superpose m items per hologram, decode with a Wiener-filtered query.
    wiener=False -> plain matched filter (the original HVB proposal)."""

    def __init__(self, ev, V, r, m, bits=32, wiener=True, seed=0, sigma_scale=1.0):
        self.ev, self.V, self.r, self.m, self.bits, self.wiener = ev[:r], V[:, :r], r, m, bits, wiener
        self.h = HoloIndex(r, m, seed)
        T = float(self.ev.sum())
        self.sig2 = sigma_scale * (m - 1) * T / r
        self.w = self.ev / (self.ev + self.sig2) if wiener else np.ones(r, np.float32)

    def add(self, Xp):
        self.h.add(Xp @ self.V)
        self.h.H = quantize_spectrum(self.h.H, self.bits)
        return self

    def scores(self, q):
        return self.h.scores((q @ self.V) * self.w)

    def bytes_per_vector(self):
        return self.h.bytes_per_vector(self.bits)
