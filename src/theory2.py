"""Theory round 2: direct tests of the paper's theoretical claims (Sec. 3, 5).
usage: theory2.py {law|rotpair|cond|cover|iso|ident2}
  law      Corollary 1 (exact law) across dimension D = 8..1536 and codebooks (sign, E8, 2/4-bit Lloyd-Max, trellis), on real
           OpenAI-1536 data projected to its top-D PCA dims. At small D the exact law (scaled Beta) and the Gaussian differ, so this
           separates the two. Also checks the variance formula (std Z = 1 for every codebook = design rule) and the top-100 pairs.
  rotpair  Theorem 1 as literally stated: one fixed (q, o) pair, randomness over 1,000 Haar rotations. Per-pair bias, std, KS test.
  cond     998k engine (stage 2, 4-bit): Z conditional on c, on the true rank of the candidate, on ||q_perp||; correlation of errors
           across the candidates of one query (matters for the union bound and the read count).
  cover    coverage lemma and union bound: per-query coverage failure vs K P(|Z|>z) and 1-(1-p)^K; exactness when covered;
           exactness of the block-max selection lemma.
  iso      error-isotropy principle, controlled: synthetic unbiased errors with prescribed covariance (equal MSE, different alignment
           with the query covariance). Prediction: recall is a function of tr(Sigma_q E), not of tr(E).
  ident2   identification estimate out of sample (GloVe-100) and its scaling with k (OpenAI and GloVe)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, numpy as np, torch
from scipy.special import betainc
from scipy import stats
sys.path.insert(0, f"{FANAR}/src")
from trellis import make_table, viterbi
from trellis_scan import viterbi_paths
from e8 import nearest_e8
dev = "cuda"; EXP = sys.argv[1]; torch.manual_seed(0); F64 = bool(os.environ.get("F64"))
if F64: torch.set_default_dtype(torch.float64)
OUT = f"{FANAR}/results/theory2_{EXP}" + ("_f64" if F64 else "") + ".json"; res = []
def emit(r):
    print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(OUT, "w"), indent=1)

def haar(D, g=None, B=None):
    """Haar-distributed orthogonal matrix (QR of a Gaussian with the sign of diag(R) fixed); B: batch."""
    shape = (D, D) if B is None else (B, D, D)
    A = torch.randn(*shape, device=dev, generator=g); Qm, Rm = torch.linalg.qr(A)
    return Qm * torch.sign(torch.diagonal(Rm, dim1=-2, dim2=-1)).unsqueeze(-2)

# exact law of Z = sqrt(m) u_1, m = D - 1 (Corollary 1)
def law_tail(z, D):
    m = D - 1; x = 1 - z ** 2 / m
    return 0.0 if x <= 0 else float(betainc((m - 1) / 2, 0.5, x))
def law_cdf(z, D):
    m = D - 1; z = np.asarray(z, np.float64); t = np.clip(z ** 2 / m, 0, 1)
    return 0.5 + 0.5 * np.sign(z) * betainc(0.5, (m - 1) / 2, t)
gauss_tail = lambda z: float(2 * stats.norm.sf(z))

def zstats(Z, D, tag=""):
    Z = Z.double(); n = Z.numel(); out = {f"{tag}pairs": n, f"{tag}mean": round(float(Z.mean()), 4), f"{tag}std": round(float(Z.std()), 4),
                                         f"{tag}excess_kurt": round(float(((Z - Z.mean()) ** 4).mean() / Z.var() ** 2 - 3), 4), f"{tag}kurt_law": round(-6 / (D + 1), 4)}
    for z in (1, 2, 3):
        out[f"{tag}P>{z}"] = round(float((Z.abs() > z).double().mean()), 5); out[f"{tag}P>{z}_law"] = round(law_tail(z, D), 5); out[f"{tag}P>{z}_gauss"] = round(gauss_tail(z), 5)
    out[f"{tag}max|Z|"] = round(float(Z.abs().max()), 3); out[f"{tag}support_sqrt(m)"] = round((D - 1) ** 0.5, 3)
    s = Z[torch.randperm(n, device=Z.device)[:200000]].cpu().numpy()
    out[f"{tag}KS_law"] = round(float(stats.kstest(s, lambda x: law_cdf(x, D)).statistic), 5)
    out[f"{tag}KS_gauss"] = round(float(stats.kstest(s, "norm").statistic), 5)
    out[f"{tag}KS_p_law"] = float(f"{stats.kstest(s, lambda x: law_cdf(x, D)).pvalue:.3g}"); out[f"{tag}KS_p_gauss"] = float(f"{stats.kstest(s, 'norm').pvalue:.3g}")
    return out

# ---------------------------------------------------------------- codebooks: rotated unit vectors F (n, D) -> code Y (n, D), nonzero
_lm = {}
def lloyd(bits):
    if bits not in _lm:
        import faiss
        g = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32); km = faiss.Kmeans(1, 2 ** bits, niter=60, seed=1); km.train(g)
        cb = torch.from_numpy(np.sort(km.centroids[:, 0])).to(dev); _lm[bits] = (cb, (cb[1:] + cb[:-1]) / 2)
    return _lm[bits]
_tt = {}
def trellis_code(F):
    D = F.shape[1]
    if D not in _tt:
        Tt = make_table(10); G = torch.randn(512, D, device=dev)
        _tt[D] = (Tt, min(((float(((G - viterbi(G, Tt, 10, 1, float(a))) ** 2).mean()), float(a)) for a in np.linspace(0.6, 1.3, 15)))[1])
    Tt, a = _tt[D]; Xs = F * D ** 0.5
    path = torch.cat([viterbi_paths(Xs[i:i + 2048], Tt, 10, a) for i in range(0, len(Xs), 2048)]); return (a * Tt)[path]
def code(name, F):
    D = F.shape[1]
    if name == "sign": return torch.sign(F) + (F == 0).float()
    if name.startswith("lm"):
        cb, thr = lloyd(int(name[2:])); return cb.to(F.dtype)[torch.bucketize(F * D ** 0.5, thr.to(F.dtype))]
    if name == "e8":                                                  # E8 lattice per 8-block at unit coordinate scale (about 1 bit/dim)
        Y = nearest_e8((F * D ** 0.5).reshape(len(F), -1, 8)).reshape(len(F), D)
        z = Y.abs().sum(1) == 0
        if z.any(): Y[z] = torch.sign(F[z])
        return Y
    if name == "trellis": return trellis_code(F)
    raise ValueError(name)

def load_oai(n=228000):
    X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r")
    T = torch.from_numpy(np.ascontiguousarray(X[:n])).to(dev).float(); Q = torch.from_numpy(np.load(f"{FANAR}/data/oai1m_q.npy")).to(dev)
    mu = T[:50000].mean(0); T = torch.nn.functional.normalize(T - mu, dim=1); Q = torch.nn.functional.normalize(Q - mu, dim=1)
    ev, V = torch.linalg.eigh(torch.cov(T[:50000].T)); return T, Q, ev.flip(0), V.flip(1)
def load_glove():
    import h5py
    f = h5py.File(f"{FANAR}/data/glove-100-angular.hdf5", "r"); X = torch.from_numpy(np.array(f["train"], np.float32)).to(dev); Q = torch.from_numpy(np.array(f["test"][:2000], np.float32)).to(dev)
    mu = X[:100000].mean(0); X = torch.nn.functional.normalize(X - mu, dim=1); Q = torch.nn.functional.normalize(Q - mu, dim=1)
    ev, V = torch.linalg.eigh(torch.cov(X[:100000].T)); return X, Q, ev.flip(0), V.flip(1)

def pair_errors(O, Qs, Y, P, top=None):
    """O: unit vectors (n, D) in the PCA frame, Qs: queries (nq, D), Y: codes of O @ P. Returns Z (nq, n) or on top pairs."""
    D = O.shape[1]; ob = (Y / Y.norm(dim=1, keepdim=True)) @ P.T; c = (O * ob).sum(1)
    true = Qs @ O.T; est = (Qs @ ob.T) / c[None]
    qperp = torch.sqrt((Qs.norm(dim=1, keepdim=True) ** 2 - true ** 2).clamp(min=1e-12))
    sig = qperp * torch.sqrt((1 - c ** 2).clamp(min=0))[None] / (c[None] * (D - 1) ** 0.5)
    Z = (est - true) / sig
    return Z, c, true

# ---------------------------------------------------------------- experiments
if EXP == "law":
    T, Q, ev, V = load_oai(); S = torch.randperm(len(T), device=dev)[:20000]; Qs0 = Q[:500]
    for D in (8, 16, 32, 64, 128, 256, 1536):
        O = torch.nn.functional.normalize(T[S] @ V[:, :D], dim=1); Qs = torch.nn.functional.normalize(Qs0 @ V[:, :D], dim=1)
        top = (Qs @ torch.nn.functional.normalize(T @ V[:, :D], dim=1).T).topk(100, 1).indices          # true top-100 of each query (D-dim scores)
        Otop = torch.nn.functional.normalize(T[top.flatten()] @ V[:, :D], dim=1)
        for name in ("sign", "e8", "lm2", "lm4", "trellis"):
            if name == "trellis" and D > 256: continue
            for seed in range(3):
                g = torch.Generator(device=dev).manual_seed(seed); P = haar(D, g)
                Y = code(name, O @ P); Z, c, _ = pair_errors(O, Qs, Y, P)
                Yt = code(name, Otop @ P); obt = (Yt / Yt.norm(dim=1, keepdim=True)) @ P.T; ct = (Otop * obt).sum(1)
                qrep = Qs.repeat_interleave(100, 0); tr = (qrep * Otop).sum(1); est = (qrep * obt).sum(1) / ct
                sg = torch.sqrt((1 - tr ** 2).clamp(min=1e-12)) * torch.sqrt((1 - ct ** 2).clamp(min=0)) / (ct * (D - 1) ** 0.5)
                Zt = (est - tr) / sg
                r = {"D": D, "codebook": name, "rot_seed": seed, "mean_c": round(float(c.mean()), 4), "std_c": round(float(c.std()), 5)}
                r.update(zstats(Z.flatten(), D)); r.update(zstats(Zt, D, "top100_"))
                emit(r)
                del Z, Y, Yt

elif EXP == "rotpair":
    T, Q, ev, V = load_oai(); M = 1000
    for D in ((1536,) if F64 else (16, 64, 256, 1536)):
        O_all = torch.nn.functional.normalize(T @ V[:, :D], dim=1); Qs = torch.nn.functional.normalize(Q[:20] @ V[:, :D], dim=1)
        sc = Qs @ O_all.T; nn1 = sc.topk(10, 1).indices; rnd = torch.randint(0, len(T), (20,), device=dev)
        pairs = [(i, int(nn1[i, 0]), "top1") for i in range(20)] + [(i, int(nn1[i, 9]), "top10") for i in range(20)] + [(i, int(rnd[i]), "random") for i in range(20)]
        O = O_all[torch.tensor([p[1] for p in pairs], device=dev)]; Qp = Qs[torch.tensor([p[0] for p in pairs], device=dev)]
        if F64: O, Qp = O.double(), Qp.double()
        for name in (("lm4",) if F64 else ("sign", "lm4", "trellis")):
            if name == "trellis" and D > 256: continue
            Zs = []; bs = (25 if F64 else 50) if D == 1536 else 200; g = torch.Generator(device=dev).manual_seed(7)
            for b0 in range(0, M, bs):
                Pb = haar(D, g, bs)                                                     # (bs, D, D)
                F = torch.einsum("pd,bde->bpe", O, Pb).reshape(-1, D)                    # each pair rotated by each of the bs rotations
                Y = code(name, F).reshape(bs, len(pairs), D); Yn = Y / Y.norm(dim=2, keepdim=True)
                ob = torch.einsum("bpe,bde->bpd", Yn, Pb); c = (ob * O[None]).sum(2); tr = (Qp * O).sum(1)
                est = (ob * Qp[None]).sum(2) / c; qperp = torch.sqrt((1 - tr ** 2).clamp(min=1e-12))
                Zs.append((est - tr[None]) / (qperp[None] * torch.sqrt((1 - c ** 2).clamp(min=0)) / (c * (D - 1) ** 0.5)))
                del Pb, F, Y
            Z = torch.cat(Zs).double().cpu().numpy()                                     # (M, pairs)
            tt = stats.ttest_1samp(Z, 0, axis=0).pvalue; ks = np.array([stats.kstest(Z[:, j], lambda x: law_cdf(x, D)).pvalue for j in range(Z.shape[1])])
            ksg = np.array([stats.kstest(Z[:, j], "norm").pvalue for j in range(Z.shape[1])])
            sd = Z.std(0, ddof=1); kinds = np.array([p[2] for p in pairs])
            r = {"D": D, "codebook": name, "rotations": M, "pairs": len(pairs), "mean_Z_abs_max": round(float(np.abs(Z.mean(0)).max()), 4),
                 "frac_bias_p<0.01": round(float((tt < 0.01).mean()), 3), "expected_frac": 0.01, "std_per_pair_min": round(float(sd.min()), 4), "std_per_pair_max": round(float(sd.max()), 4),
                 "std_se": round(float(1 / np.sqrt(2 * M)), 4), "frac_KS_law_p<0.01": round(float((ks < 0.01).mean()), 3), "frac_KS_gauss_p<0.01": round(float((ksg < 0.01).mean()), 3),
                 "fisher_p_bias": float(f"{stats.combine_pvalues(tt)[1]:.3g}"), "fisher_p_KS_law": float(f"{stats.combine_pvalues(ks)[1]:.3g}"), "fisher_p_KS_gauss": float(f"{stats.combine_pvalues(ksg)[1]:.3g}")}
            for kd in ("top1", "top10", "random"):
                m_ = kinds == kd; r[f"{kd}_mean_Z"] = round(float(Z[:, m_].mean()), 4); r[f"{kd}_std_Z"] = round(float(Z[:, m_].std()), 4)
            emit(r)

elif EXP in ("cond", "cover"):
    sys.argv = [sys.argv[0], "rot"]                                                     # import the engine without running an experiment
    import importlib.util
    spec = importlib.util.spec_from_file_location("rf", f"{FANAR}/src/rev_fanardb.py"); src = open(f"{FANAR}/src/rev_fanardb.py").read()
    src = src.split("# ---------------------------------------------------------------- experiments")[0]
    rf = type(sys)("rf"); rf.__dict__["__file__"] = f"{FANAR}/src/rev_fanardb.py"; exec(compile(src, "rev_fanardb.py", "exec"), rf.__dict__)
    s1 = rf.build_stage1(0); s2 = rf.build_stage2(s1, rf.rotation("haar")); Qg, host16, d, NQ = rf.Qg, rf.host16, rf.d, rf.NQ
    from trellis_scan import trellis_search_ivf
    from fused_scan import rescore_planes
    def pairs(Qb, K, nprobe=32):
        QC = Qb @ s1["c"]
        pos = trellis_search_ivf(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], 10, K, s1["cen"], s1["off"], nprobe)
        est = rescore_planes(Qb @ s2["Pf"], QC, pos, s2["planes"], rf.CB, s2["s2"]); sg = s2["sig"][pos].float()
        ids = s1["order"][pos]; Vv = host16.index_select(0, ids.cpu().reshape(-1)).to(dev).float().reshape(len(Qb), K, d)
        true = torch.einsum("qkd,qd->qk", Vv, Qb); r_ = Vv - s1["c"]; nf = r_.norm(dim=2); qo = (true - QC[:, None]) / nf
        qperp = torch.sqrt((Qb.norm(dim=1, keepdim=True) ** 2 - qo ** 2).clamp(min=0))
        return pos, ids, est, true, sg, qperp, s2["c2"][pos]

    if EXP == "cond":
        K = 200; Ze, Zu, C2, RK, QP, TR = [], [], [], [], [], []
        for s in range(0, 1000, 125):
            pos, ids, est, true, sg, qperp, c2 = pairs(Qg[s:s + 125], K)
            Ze.append((est - true) / (sg * qperp)); Zu.append((est - true) / sg); C2.append(c2.expand_as(est)); QP.append(qperp); TR.append(true)
            RK.append(true.argsort(1, descending=True).argsort(1))                     # rank of the candidate's true score among the K
        Ze, Zu, C2, RK, QP, TR = map(torch.cat, (Ze, Zu, C2, RK, QP, TR))
        emit({"what": "all pairs", **zstats(Ze.flatten(), d), "corr(Z, true score)": round(float(np.corrcoef(Ze.flatten().cpu(), TR.flatten().cpu())[0, 1]), 4),
              "corr(Z, c)": round(float(np.corrcoef(Ze.flatten().cpu(), C2.flatten().cpu())[0, 1]), 4)})
        qs = torch.quantile(C2.flatten()[:1_000_000], torch.linspace(0, 1, 6, device=dev))
        for i in range(5):
            m = (C2 >= qs[i]) & (C2 <= qs[i + 1]); z = Ze[m]
            emit({"what": "by c (quintile)", "c_lo": round(float(qs[i]), 4), "c_hi": round(float(qs[i + 1]), 4), "n": int(m.sum()), "mean": round(float(z.mean()), 4), "std": round(float(z.std()), 4),
                  "P>2": round(float((z.abs() > 2).float().mean()), 5), "P>3": round(float((z.abs() > 3).float().mean()), 5)})
        for lo, hi in ((0, 1), (1, 10), (10, 11), (11, 50), (50, 200)):
            m = (RK >= lo) & (RK < hi); z = Ze[m]
            emit({"what": "by true rank among candidates", "rank": f"{lo + 1}-{hi}", "n": int(m.sum()), "mean": round(float(z.mean()), 4), "std": round(float(z.std()), 4),
                  "P>2": round(float((z.abs() > 2).float().mean()), 5), "P>3": round(float((z.abs() > 3).float().mean()), 5), "std_used(engine sigma)": round(float(Zu[m].std()), 4)})
        qq = torch.quantile(QP.flatten()[:1_000_000], torch.linspace(0, 1, 5, device=dev))
        for i in range(4):
            m = (QP >= qq[i]) & (QP <= qq[i + 1]); z = Ze[m]
            emit({"what": "by ||q_perp|| (quartile)", "lo": round(float(qq[i]), 4), "hi": round(float(qq[i + 1]), 4), "mean": round(float(z.mean()), 4), "std": round(float(z.std()), 4), "std_used": round(float(Zu[m].std()), 4)})
        # within-query dependence: var of a query's mean Z across queries is 1/K if the K errors are uncorrelated
        mq = Ze.mean(1); ratio = float(mq.var() * K); rho = (ratio - 1) / (K - 1)
        emit({"what": "within-query error dependence", "K": K, "K*var(mean_q Z)": round(ratio, 3), "implied_mean_pairwise_corr": round(rho, 5),
              "note": "1 = uncorrelated; >1 positively correlated errors among a query's candidates"})

    else:  # cover
        Z_GRID = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.36, 5.0)
        for K in (200, 400):
            fails_e = {z: 0 for z in Z_GRID}; fails_u = {z: 0 for z in Z_GRID}; exact_cov = {z: [0, 0] for z in Z_GRID}; exact_unc = {z: [0, 0] for z in Z_GRID}
            nq = 0; pmiss = {z: 0.0 for z in Z_GRID}
            for s in range(0, NQ, 125):
                pos, ids, est, true, sg, qperp, c2 = pairs(Qg[s:s + 125], K); nq += len(pos)
                ze = ((est - true) / (sg * qperp)).abs(); zu = ((est - true) / sg).abs()
                topk_true = true.topk(10, 1).indices.sort(1).values                      # exact top-10 of the candidate set
                for z in Z_GRID:
                    lo, hi = est - z * sg, est + z * sg                                  # the engine's intervals (sigma with ||q||)
                    sure = lo > hi.topk(11, 1).values[:, -1:]; amb = (hi >= lo.topk(10, 1).values[:, -1:]) & ~sure
                    score = torch.where(sure, est + 10, torch.where(amb, true, torch.full_like(est, -1e9)))
                    got = score.topk(10, 1).indices.sort(1).values; ok = (got == topk_true).all(1)
                    cov = (zu <= z).all(1)                                               # engine intervals cover every candidate
                    fails_u[z] += int((~cov).sum()); fails_e[z] += int(((ze > z).any(1)).sum())
                    exact_cov[z][0] += int((ok & cov).sum()); exact_cov[z][1] += int(cov.sum()); exact_unc[z][0] += int((ok & ~cov).sum()); exact_unc[z][1] += int((~cov).sum())
            for z in Z_GRID:
                p = law_tail(z, d)
                emit({"K": K, "z": z, "queries": nq, "P_fail_query_exact_sigma": round(fails_e[z] / nq, 5), "P_fail_query_engine_sigma": round(fails_u[z] / nq, 5),
                      "union_bound_K*p": round(min(1, K * p), 5), "independent_1-(1-p)^K": round(1 - (1 - p) ** K, 5),
                      "exact_given_covered": f"{exact_cov[z][0]}/{exact_cov[z][1]}", "exact_given_not_covered": f"{exact_unc[z][0]}/{exact_unc[z][1]}"})
        # block-max selection lemma: candidate set vs exact top-K of the stage-1 estimates over the probed cells
        import triton
        from trellis_scan import _trellis_rescore
        words, tab, scale = s1["words"], s1["tab"], s1["scale1"]; N, W = words.shape; R = tab.shape[0] and s1["W1"].shape[1]; RP = triton.next_power_of_2(R)
        cell_of = torch.empty(N, dtype=torch.long, device=dev)
        for cc in range(len(s1["off"]) - 1): cell_of[int(s1["off"][cc]):int(s1["off"][cc + 1])] = cc
        for K in (200, 400):
            agree, tot, gap_viol = 0, 0, 0
            for s in range(0, 400, 20):
                Qb = Qg[s:s + 20]; B = len(Qb); QC = Qb @ s1["c"]; Qr = (Qb @ s1["W1"]).float().contiguous()
                pos = trellis_search_ivf(Qr, Qb, QC, words, tab, scale, 10, K, s1["cen"], s1["off"], 32)
                allpos = torch.arange(N, device=dev, dtype=pos.dtype).repeat(B, 1).contiguous(); est = torch.empty(B, N, device=dev)
                _trellis_rescore[(triton.cdiv(N, 32), B)](Qr, allpos, words, tab, scale, QC.float().contiguous(), est, B, N, W=W, R=R, RP=RP, L=10, BM=32)
                probe = (Qb @ s1["cen"].T).topk(32, 1).indices; inprobe = (cell_of[None, :] == probe[:, :, None]).any(1)
                est = torch.where(inprobe, est, torch.full_like(est, -1e30)); ex = est.topk(K, 1)
                for i in range(B):
                    a_, b_ = set(pos[i].tolist()), set(ex.indices[i].tolist()); agree += len(a_ & b_); tot += K
                    if a_ != b_:                                                        # are the disagreements ties at the K-th value (fp16 block maxima)?
                        kth = float(ex.values[i, -1]); miss = torch.tensor(sorted(b_ - a_), device=dev)
                        gap_viol += int((est[i, miss] - kth).abs().gt(2e-3).sum())
            emit({"what": "block-max lemma", "K": K, "queries": 400, "overlap_with_exact_topK": round(agree / tot, 6), "missed_items_not_within_2e-3_of_Kth": gap_viol})

elif EXP == "iso":
    T, Q, ev, V = load_oai(); N, d = T.shape
    GT = torch.cat([(Q[i:i + 500] @ T.T).topk(100, 1).indices for i in range(0, 2000, 500)])
    Sq = Q.T @ Q / len(Q); qv = (V.T @ Sq @ V).diagonal()                              # query second moment along each data PC
    lam = ev.clamp(min=1e-9)
    for gamma in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        w0 = lam ** gamma; w0 = w0 / w0.sum()
        for mse in (0.05, 0.1, 0.2, 0.4, 0.8):
            w = w0 * mse; trSE = float((w * qv).sum()); g = torch.Generator(device=dev).manual_seed(11)
            est = torch.empty(len(Q), N, device=dev)
            for j in range(0, N, 20000):
                Ech = (torch.randn(min(20000, N - j), d, device=dev, generator=g) * w.sqrt()[None]) @ V.T
                est[:, j:j + 20000] = Q @ (T[j:j + 20000] + Ech).T
            I = est.topk(100, 1).indices; del est
            r10 = float(np.mean([len(set(I[i, :10].tolist()) & set(GT[i, :10].tolist())) / 10 for i in range(len(Q))]))
            c100 = float(np.mean([len(set(I[i].tolist()) & set(GT[i, :10].tolist())) / 10 for i in range(len(Q))]))
            emit({"gamma(E ~ lambda^gamma)": gamma, "mse_trE": mse, "tr(SqE)": round(trSE, 7), "iso_ref tr(Sq)trE/d": round(float(Sq.trace()) * mse / d, 7),
                  "rho": round(trSE / (float(Sq.trace()) * mse / d), 3), "recall@10": round(r10, 4), "top10_in_top100": round(c100, 4)})
    # does recall collapse onto tr(Sigma_q E) but not onto tr(E)?  R^2 of a cubic fit of recall on log x
    rr = np.array([r["recall@10"] for r in res]); xe = np.log([r["mse_trE"] for r in res]); xs = np.log([r["tr(SqE)"] for r in res])
    def r2(x):
        p = np.polyfit(x, rr, 3); return float(1 - ((np.polyval(p, x) - rr) ** 2).sum() / ((rr - rr.mean()) ** 2).sum())
    emit({"summary": "recall@10 explained by", "R2_log_trE": round(r2(xe), 4), "R2_log_tr(SqE)": round(r2(xs), 4),
          "spearman_trE": round(float(stats.spearmanr(xe, rr).correlation), 4), "spearman_tr(SqE)": round(float(stats.spearmanr(xs, rr).correlation), 4)})

elif EXP == "ident2":
    for ds in ("glove", "oai"):
        T, Q, ev, V = load_glove() if ds == "glove" else load_oai(); N, d = T.shape
        lam = ev.clamp(min=0); pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        Rs = (8, 16, 24, 32, 48, 64, 100) if ds == "glove" else (32, 64, 96, 128, 192, 256, 384)
        scores = lambda Qp, Ob: torch.cat([(Qp[i:i + 250] @ Ob.T) for i in range(0, len(Qp), 250)])
        GTk = {k: torch.cat([(Q[i:i + 250] @ T.T).topk(k, 1) .indices for i in range(0, len(Q), 250)]) for k in (1, 10, 100)}
        taus = {k: torch.cat([(Q[i:i + 250] @ T.T).topk(k, 1).values[:, -1] for i in range(0, len(Q), 250)]) for k in (1, 10, 100)}
        pred = {k: {"tau_med": round(float(taus[k].median()), 4), "bits_pred_PR": round(pr * float(np.log2(1 / float(taus[k].median()))), 1),
                    "bits_pred_PR_p10_p90": [round(pr * float(np.log2(1 / float(torch.quantile(taus[k], q)))), 1) for q in (0.9, 0.1)]} for k in (1, 10, 100)}
        emit({"dataset": ds, "N": N, "d": d, "participation_ratio": round(pr, 1), "pred": pred})
        g = torch.Generator(device=dev).manual_seed(3)
        for Rr in Rs:
            if Rr > d: continue
            W = V[:, :Rr] @ haar(Rr, g); Zp = T @ W; O = Zp / Zp.norm(dim=1, keepdim=True); Qp = Q @ W
            for name, bits in (("sign", Rr), ("trellis", Rr + 9), ("lm2", 2 * Rr)):
                Y = torch.cat([code(name, O[i:i + 100000]) for i in range(0, N, 100000)]); ob = Y / Y.norm(dim=1, keepdim=True)
                c = (O * ob).sum(1); obs = ob / c[:, None] * Zp.norm(dim=1, keepdim=True)        # unbiased estimate of <q, x> in the R-dim projection
                r = {"dataset": ds, "R": Rr, "codebook": name, "bits": bits, "mean_c": round(float(c.mean()), 4)}
                for k in (1, 10, 100):
                    I = torch.cat([(Qp[i:i + 250] @ obs.T).topk(20 * k, 1).indices for i in range(0, len(Qp), 250)])
                    r[f"top{k}_in_top{20 * k}"] = round(float(np.mean([len(set(I[i].tolist()) & set(GTk[k][i].tolist())) / k for i in range(len(Q))])), 4)
                emit(r); del Y, ob, obs
            del Zp, O
        del T
elif EXP == "ident3":                                                                  # fair version: same N, fixed candidate budget
    NS = 228000
    for ds in ("glove", "oai"):
        T, Q, ev, V = load_glove() if ds == "glove" else load_oai(NS)
        if ds == "glove":
            T = T[torch.randperm(len(T), device=dev, generator=torch.Generator(device=dev).manual_seed(5))[:NS]]
            ev, V = torch.linalg.eigh(torch.cov(T[:100000].T)); ev, V = ev.flip(0), V.flip(1)
        N, d = T.shape; lam = ev.clamp(min=0); pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        GTk = {k: torch.cat([(Q[i:i + 250] @ T.T).topk(k, 1).indices for i in range(0, len(Q), 250)]) for k in (1, 10, 100)}
        taus = {k: torch.cat([(Q[i:i + 250] @ T.T).topk(k, 1).values[:, -1] for i in range(0, len(Q), 250)]) for k in (1, 10, 100)}
        emit({"dataset": ds, "N": N, "d": d, "participation_ratio": round(pr, 1),
              "bits_pred_PR": {k: round(pr * float(np.log2(1 / float(taus[k].median()))), 1) for k in (1, 10, 100)}})
        g = torch.Generator(device=dev).manual_seed(3)
        for Rr in ((8, 16, 24, 32, 48, 64, 100) if ds == "glove" else (32, 64, 96, 128, 192, 256, 384)):
            W = V[:, :Rr] @ haar(Rr, g); Zp = T @ W; O = Zp / Zp.norm(dim=1, keepdim=True); Qp = Q @ W
            for name, bits in (("sign", Rr), ("trellis", Rr + 9), ("lm2", 2 * Rr)):
                Y = torch.cat([code(name, O[i:i + 100000]) for i in range(0, N, 100000)]); ob = Y / Y.norm(dim=1, keepdim=True)
                c = (O * ob).sum(1); obs = ob / c[:, None] * Zp.norm(dim=1, keepdim=True)
                I = torch.cat([(Qp[i:i + 250] @ obs.T).topk(2000, 1).indices for i in range(0, len(Qp), 250)])
                r = {"dataset": ds, "N": N, "R": Rr, "codebook": name, "bits": bits}
                for k in (1, 10, 100):
                    for K in (200, 2000):
                        r[f"top{k}_in_{K}"] = round(float(np.mean([len(set(I[i, :K].tolist()) & set(GTk[k][i].tolist())) / k for i in range(len(Q))])), 4)
                emit(r); del Y, ob, obs, I
            del Zp, O
        del T
print("DONE", flush=True)
