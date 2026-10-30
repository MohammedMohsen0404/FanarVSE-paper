"""Reviewer round, FanarVSE side (998k OpenAI-1536, native Windows). Same engine as engine_v6_paper.py (R=256, 64 cells, 4-bit stage 2).
usage: rev_fanardb.py {rot|seeds|zsweep|k100|batch|encoder|scale|platform|grid}
  rot     stage-2 rotation: Haar vs block-FWHT(+perm) vs block-diagonal Haar vs none -> standardized-error stats, coverage, recall, reads
  seeds   5 rotation/IVF seeds (full rebuild) -> recall, reads, QPS spread
  zsweep  error-bar width: Gaussian z, Cantelli (distribution-free one-sided) and empirically calibrated z -> recall, read distribution, QPS
  k100    recall@100 (K up to 2000), candidate ceiling, reads, QPS
  batch   batch size 1..500: per-batch latency p50/p99, QPS
  encoder trellis encoder throughput; beam-search approximation (beam width vs c and time)
  platform same configs + micro-benchmarks, run once on native Windows and once in WSL2 (OS-overhead check)
  scale   stage-1 cost from 1M to 10M codes (replicated codes; cost model only, no recall)"""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, os, json, time, platform, numpy as np, torch, faiss
ROOT = os.environ.get("FANAR", f"{FANAR}")                                       # {FANAR} under WSL2
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ROOT + "/src")
from trellis import make_table, viterbi
from trellis_scan import viterbi_paths, paths_to_words, trellis_search, trellis_search_ivf
from fused_scan import rescore_planes
import trellis_scan; trellis_scan.return_est = False                      # trellis_search references this module global
dev = "cuda"; EXP = sys.argv[1]
X = np.load(ROOT + "/data/oai1m_base_fp16.npy", mmap_mode="r"); Qn = np.load(ROOT + "/data/oai1m_q.npy")
GT10 = np.load(ROOT + "/data/oai1m_gt.npy"); GT100 = np.load(ROOT + "/data/oai1m_gt100.npy")
N, d = X.shape; NQ = len(Qn); host16 = torch.from_numpy(np.ascontiguousarray(X)); Qg = torch.from_numpy(Qn).to(dev)
R, L, BITS, NLIST, REPS = int(os.environ.get("FANAR_R", 256)), 10, 4, 64, 3
OUT = f"{ROOT}/results/rev_fanardb_{EXP}" + (f"_R{R}" if R != 256 else "") + (f"_{platform.system().lower()}" if EXP == "platform" else "") + ".json"; res = []
def emit(r):
    print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(OUT, "w"), indent=1)
os.makedirs(ROOT + "/data/rev_cache", exist_ok=True)


# ---------------------------------------------------------------- index build
def build_stage1(seed):
    f = f"{ROOT}/data/rev_cache/s1_seed{seed}" + (f"_R{R}" if R != 256 else "") + ".pt"
    if os.path.exists(f): return torch.load(f, weights_only=False)                     # our own cache file
    torch.manual_seed(seed); t0 = time.perf_counter()
    S = host16[:50000].to(dev).float(); c = S.mean(0); ev, V = torch.linalg.eigh(torch.cov((S - c).T)); V = V.flip(1); del S
    W1 = V[:, :R] @ torch.linalg.qr(torch.randn(R, R, device=dev))[0]; _ = torch.randn(d, d, device=dev)   # keep the RNG stream of the paper run
    Tt = make_table(L); G = torch.randn(512, R, device=dev); best = None
    for a in np.linspace(0.6, 1.3, 15):
        m = float(((G - viterbi(G, Tt, L, 1, float(a))) ** 2).mean())
        if best is None or m < best[0]: best = (m, float(a))
    alpha = best[1]; tab = (alpha * Tt).contiguous()
    samp = host16[torch.randperm(N)[:200000]].to(dev).float(); cen = samp[torch.randperm(len(samp), device=dev)[:NLIST]].clone()
    for _ in range(15):
        a_ = (samp @ cen.T).argmax(1); cen = torch.nn.functional.normalize(torch.zeros_like(cen).index_add_(0, a_, samp) + 1e-9, dim=1)
    del samp
    Acell = torch.cat([(host16[j:j + 100000].to(dev).float() @ cen.T).argmax(1) for j in range(0, N, 100000)]); order = torch.argsort(Acell)
    off = torch.searchsorted(Acell[order], torch.arange(NLIST + 1, device=dev)).cpu().numpy(); oc = order.cpu()
    words = None; scale1 = torch.empty(N, dtype=torch.float16, device=dev)
    for j in range(0, N, 16384):
        Rr = host16[oc[j:j + 16384]].to(dev).float() - c; n = len(Rr)
        Z = Rr @ W1; nz = Z.norm(dim=1); Zs = Z / nz[:, None] * R ** 0.5
        path = torch.cat([viterbi_paths(Zs[i:i + 2048], Tt, L, alpha) for i in range(0, n, 2048)])
        wv = paths_to_words(path, L).to(torch.int32)
        if words is None: words = torch.empty(N, wv.shape[1], dtype=torch.int32, device=dev)
        words[j:j + n] = wv; Y = tab[path]; yn = Y.norm(dim=1); fip1 = (Zs * Y).sum(1) / (yn * R ** 0.5)
        scale1[j:j + n] = (nz / (fip1 * yn)).half()
    torch.cuda.synchronize()
    s1 = dict(c=c, W1=W1, tab=tab, alpha=alpha, cen=cen, off=off, order=order, words=words, scale1=scale1, build_s=time.perf_counter() - t0)
    torch.save(s1, f); return s1

def fwht_blocks(x, B=512):
    """orthonormal Walsh-Hadamard transform on consecutive blocks of B coordinates (d=1536 = 3 x 512)."""
    n = x.shape[0]; y = x.reshape(n, -1, B).clone(); h = 1
    while h < B:
        y = y.reshape(n, -1, B // (2 * h), 2, h); a, b = y[:, :, :, 0], y[:, :, :, 1]; y = torch.stack([a + b, a - b], 3); h *= 2
    return y.reshape(n, -1) / B ** 0.5

def rotation(kind, seed=0):
    """returns the d x d matrix P with x @ P = rotated x."""
    g = torch.Generator(device=dev).manual_seed(1000 + seed); I = torch.eye(d, device=dev)
    if kind == "haar": return torch.linalg.qr(torch.randn(d, d, device=dev, generator=g))[0]
    if kind == "none": return I
    if kind.startswith("bfwht"):                                                        # bfwht1: signs + block FWHT; bfwht3: 3 x (perm, signs, block FWHT)
        Y = I
        for _ in range(int(kind[-1])):
            p = torch.randperm(d, device=dev, generator=g) if kind != "bfwht1" else torch.arange(d, device=dev)
            s = torch.randint(0, 2, (d,), device=dev, generator=g).float() * 2 - 1
            Y = fwht_blocks(Y[:, p] * s)
        return Y
    if kind.startswith("blockhaar"):                                                     # block-diagonal Haar with blocks of b coordinates
        b = int(kind[9:]); P = torch.zeros(d, d, device=dev)
        for i in range(0, d, b): P[i:i + b, i:i + b] = torch.linalg.qr(torch.randn(b, b, device=dev, generator=g))[0]
        return P
    raise ValueError(kind)

g1 = np.random.default_rng(1).standard_normal((400000, 1)).astype(np.float32); km = faiss.Kmeans(1, 2 ** BITS, niter=60, seed=1); km.train(g1)
CB = torch.from_numpy(np.sort(km.centroids[:, 0])).to(dev) * d ** -0.5; THR = (CB[1:] + CB[:-1]) / 2
SH = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8); pack = lambda bit: ((bit.reshape(bit.shape[0], -1, 8).to(torch.uint8)) << SH).sum(2).to(torch.uint8)

def build_stage2(s1, Pf):
    planes = torch.empty(BITS, N, d // 8, dtype=torch.uint8, device=dev); s2 = torch.empty(N, dtype=torch.float16, device=dev)
    sig = torch.empty(N, dtype=torch.float16, device=dev); cc = torch.empty(N, dtype=torch.float32, device=dev); oc = s1["order"].cpu()
    for j in range(0, N, 16384):
        Rr = host16[oc[j:j + 16384]].to(dev).float() - s1["c"]; n = len(Rr)
        nf = Rr.norm(dim=1); F = (Rr / nf[:, None]) @ Pf; q = torch.bucketize(F, THR); Yf = CB[q]; ynf = Yf.norm(dim=1); fip = (F * Yf).sum(1) / ynf
        for p in range(BITS): planes[p, j:j + n] = pack(((q >> (BITS - 1 - p)) & 1).bool())
        s2[j:j + n] = (nf / (fip * ynf)).half(); sig[j:j + n] = (nf * torch.sqrt((1 - fip ** 2).clamp(min=0)) / fip / (d - 1) ** 0.5).half(); cc[j:j + n] = fip
    return dict(planes=planes, s2=s2, sig=sig, c2=cc, Pf=Pf)


# ---------------------------------------------------------------- search
class Engine:
    def __init__(self, s1, s2):
        self.s1, self.s2 = s1, s2; self.planes_host = None
    def gpu_stage(self, Qb, K, z, nprobe, k=10, lean=False):
        s1, s2 = self.s1, self.s2; Qr = Qb @ s1["W1"]; QC = Qb @ s1["c"]
        pos = trellis_search_ivf(Qr, Qb, QC, s1["words"], s1["tab"], s1["scale1"], L, K, s1["cen"], s1["off"], nprobe) if nprobe else trellis_search(Qr, QC, s1["words"], s1["tab"], s1["scale1"], L, K)[0]
        est = rescore_planes(Qb @ s2["Pf"], QC, pos, s2["planes"], CB, s2["s2"]); sg = s2["sig"][pos].float()
        lo, hi = est - z * sg, est + z * sg
        sure = lo > hi.topk(k + 1, 1).values[:, -1:]; amb = (hi >= lo.topk(k, 1).values[:, -1:]) & ~sure
        return s1["order"][pos].cpu().numpy(), est.cpu().numpy(), sure.cpu().numpy(), amb.cpu().numpy()
    @staticmethod
    def cpu_stage(Qb, ids, est, sure, amb, k=10):
        score = np.where(sure, est + 10.0, -1e9).astype(np.float32); bi, bk = np.nonzero(amb)
        if len(bi): score[bi, bk] = (host16.index_select(0, torch.from_numpy(ids[bi, bk])).numpy().astype(np.float32) * Qb[bi]).sum(1)
        return np.take_along_axis(ids, np.argsort(-score, 1)[:, :k], 1), amb.sum(1)
    def run(self, K, z, nprobe, k=10, B=500, nq=NQ):
        outs, reads, cands = [], [], []
        with ThreadPoolExecutor(2) as ex:
            futs = []
            for s in range(0, nq, B):
                g = self.gpu_stage(Qg[s:s + B], K, z, nprobe, k); cands.append(g[0])
                futs.append(ex.submit(self.cpu_stage, Qn[s:s + B], *g, k))
            for f in futs: I, r = f.result(); outs.append(I); reads.append(r)
        return np.vstack(outs), np.concatenate(reads), np.vstack(cands)
    def bench(self, K, z, nprobe, k=10, B=500, nq=NQ):
        self.run(K, z, nprobe, k, B, nq); qps = []
        for _ in range(REPS):
            torch.cuda.synchronize(); t0 = time.perf_counter(); I, reads, cands = self.run(K, z, nprobe, k, B, nq); qps.append(nq / (time.perf_counter() - t0))
        return I, reads, cands, float(np.mean(qps)), float(np.std(qps))

def recall(I, k=10, nq=NQ):
    G = GT10 if k == 10 else GT100[:, :k]
    return round(float(np.mean([len(set(I[i, :k]) & set(G[i])) / k for i in range(nq)])), 4)
def cand_recall(C, k=10, nq=NQ):
    G = GT10 if k == 10 else GT100[:, :k]
    return round(float(np.mean([len(set(C[i]) & set(G[i])) / k for i in range(nq)])), 4)
def read_stats(r):
    return {"reads_mean": round(float(r.mean()), 2), "reads_p50": float(np.percentile(r, 50)), "reads_p90": float(np.percentile(r, 90)),
            "reads_p99": float(np.percentile(r, 99)), "reads_p999": float(np.percentile(r, 99.9)), "reads_max": int(r.max()), "frac_zero_reads": round(float((r == 0).mean()), 4)}

def pair_stats(eng, nq=1000, K=200, nprobe=32, z_list=(1, 2, 3)):
    """standardized stage-2 errors on (query, candidate) pairs: z = (est - true) / sigma_pred, with sigma_pred from Theorem 1.
    sigma_used = per-vector sig (assumes ||q_perp|| = 1, what the engine uses); sigma_exact = sig * ||q_perp||."""
    s1, s2 = eng.s1, eng.s2; zs_used, zs_exact = [], []
    for s in range(0, nq, 250):
        Qb = Qg[s:s + 250]; QC = Qb @ s1["c"]
        pos = trellis_search_ivf(Qb @ s1["W1"], Qb, QC, s1["words"], s1["tab"], s1["scale1"], L, K, s1["cen"], s1["off"], nprobe)
        est = rescore_planes(Qb @ s2["Pf"], QC, pos, s2["planes"], CB, s2["s2"]); sg = s2["sig"][pos].float()
        ids = s1["order"][pos].cpu(); V = host16.index_select(0, ids.reshape(-1)).to(dev).float().reshape(len(Qb), K, d)
        true = torch.einsum("qkd,qd->qk", V, Qb); r = V - s1["c"]; nf = r.norm(dim=2); qo = (true - QC[:, None]) / nf
        qperp = torch.sqrt((Qb.norm(dim=1, keepdim=True) ** 2 - qo ** 2).clamp(min=0))
        zs_used.append(((est - true) / sg).flatten()); zs_exact.append(((est - true) / (sg * qperp)).flatten())
    zu, ze = torch.cat(zs_used), torch.cat(zs_exact)
    out = {"pairs": int(len(ze)), "z_mean": round(float(ze.mean()), 4), "z_std_exact": round(float(ze.std()), 4), "z_std_used": round(float(zu.std()), 4),
           "excess_kurtosis": round(float(((ze - ze.mean()) ** 4).mean() / ze.var() ** 2 - 3), 3)}
    for zz in z_list:
        out[f"P(|z|>{zz})_used"] = round(float((zu.abs() > zz).float().mean()), 5); out[f"P(|z|>{zz})_exact"] = round(float((ze.abs() > zz).float().mean()), 5)
    out["q999_abs_z_used"] = round(float(torch.quantile(zu.abs()[:5_000_000].float(), 0.999)), 3)
    out["mean_c_stage2"] = round(float(s2["c2"].mean()), 5)
    return out, zu.cpu().numpy()


# ---------------------------------------------------------------- experiments
if EXP == "rot":
    s1 = build_stage1(0)
    for kind in ("haar", "bfwht1", "bfwht3", "blockhaar64", "blockhaar8", "none"):
        eng = Engine(s1, build_stage2(s1, rotation(kind))); ps, _ = pair_stats(eng)
        for K in (200, 400):
            I, reads, C, m, s = eng.bench(K, 2.0, 32)
            emit({"rotation": kind, "K": K, "nprobe": 32, "z": 2.0, "recall@10": recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), **read_stats(reads), **ps})
        del eng; torch.cuda.empty_cache()

elif EXP == "seeds":
    for seed in range(5):
        s1 = build_stage1(seed); eng = Engine(s1, build_stage2(s1, rotation("haar", seed))); ps, _ = pair_stats(eng, nq=500)
        for K in (200, 400):
            I, reads, C, m, s = eng.bench(K, 2.0, 32)
            emit({"seed": seed, "K": K, "nprobe": 32, "recall@10": recall(I), "cand_recall@10": cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1),
                  "build_s": round(s1["build_s"], 1), **read_stats(reads), "z_std_used": ps["z_std_used"], "P(|z|>2)_used": ps["P(|z|>2)_used"]})
        del eng, s1; torch.cuda.empty_cache()

elif EXP == "zsweep":
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    # empirical calibration on queries 0..999 (pairs), evaluation reported on all queries and on the held-out half
    ps, zu = pair_stats(eng, nq=1000); z_cal = {a: float(np.quantile(np.abs(zu), 1 - a)) for a in (1e-3, 1e-4)}
    print(json.dumps({"calibration": ps, "z_cal": z_cal}), flush=True)
    zs = [("gauss", 1.0), ("gauss", 1.5), ("gauss", 2.0), ("gauss", 2.5), ("gauss", 3.0),
          ("cantelli d=0.05", (1 / 0.05 - 1) ** 0.5), ("cantelli d=0.01", (1 / 0.01 - 1) ** 0.5),
          ("calibrated 1e-3", z_cal[1e-3]), ("calibrated 1e-4", z_cal[1e-4])]
    for K in (200, 400):
        for name, z in zs:
            I, reads, C, m, s = eng.bench(K, z, 32)
            ho = slice(1000, NQ)
            emit({"bound": name, "z": round(z, 3), "K": K, "nprobe": 32, "recall@10": recall(I), "recall@10_heldout": round(float(np.mean([len(set(I[i]) & set(GT10[i])) / 10 for i in range(1000, NQ)])), 4),
                  "cand_ceiling": cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **read_stats(reads)})
            if name == "gauss" and z == 2.0:
                np.save(f"{ROOT}/results/rev_reads_K{K}_z2.npy", reads)
                # query difficulty: gap between the true 10th and 11th scores vs reads
                Gs = np.array([float((host16[GT100[i, 9]].float() @ torch.from_numpy(Qn[i])) - (host16[GT100[i, 10]].float() @ torch.from_numpy(Qn[i]))) for i in range(NQ)])
                np.save(f"{ROOT}/results/rev_gap10_11.npy", Gs)
                from scipy.stats import spearmanr
                print(json.dumps({"K": K, "spearman(reads, gap10_11)": round(float(spearmanr(reads, Gs).correlation), 3)}), flush=True)
    # exact rerank of all K=400 candidates = ceiling of the candidate set
    I, reads, C, m, s = eng.bench(400, 1e6, 32)
    emit({"bound": "exact rerank of all K (z=inf)", "z": "inf", "K": 400, "nprobe": 32, "recall@10": recall(I), "cand_ceiling": cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **read_stats(reads)})

elif EXP == "k100":
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    for nprobe, K in ((16, 1000), (32, 1000), (64, 1000), (32, 2000), (64, 2000)):          # nprobe 64 = all cells (full scan)
        I, reads, C, m, s = eng.bench(K, 2.0, nprobe, k=100, B=250)
        emit({"k": 100, "K": K, "nprobe": nprobe, "recall@100": recall(I, 100), "cand_ceiling@100": cand_recall(C, 100), "recall@10_of_these": recall(I, 10),
              "QPS": round(m, 1), "QPS_std": round(s, 1), **read_stats(reads)})

elif EXP == "batch":
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    for K in (200, 400):
        for B, nq in ((1, 300), (4, 400), (16, 800), (64, 1984), (100, 2000), (250, 2000), (500, 2000)):
            eng.run(K, 2.0, 32, B=B, nq=min(nq, 4 * B)); lat = []; I_all = []
            for _ in range(REPS):
                for st in range(0, nq, B):                                            # one batch at a time, synchronous: latency per batch
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    g = eng.gpu_stage(Qg[st:st + B], K, 2.0, 32); I, _ = eng.cpu_stage(Qn[st:st + B], *g)
                    lat.append(time.perf_counter() - t0); I_all.append(I)
            lat = np.array(lat) * 1000; I = np.vstack(I_all[: (nq + B - 1) // B])
            emit({"K": K, "nprobe": 32, "batch": B, "recall@10": recall(I, 10, nq), "latency_ms_p50": round(float(np.median(lat)), 3),
                  "latency_ms_p99": round(float(np.percentile(lat, 99)), 3), "QPS_sync": round(B / (lat.mean() / 1000), 1)})

elif EXP == "encoder":
    s1 = build_stage1(0); W1, Tt, alpha = s1["W1"], s1["tab"] / s1["alpha"], s1["alpha"]
    Rr = host16[:16384].to(dev).float() - s1["c"]; Z = Rr @ W1; Zs = Z / Z.norm(dim=1, keepdim=True) * R ** 0.5; tab = s1["tab"]
    def cval(path):
        Y = tab[path]; return float(((Zs * Y).sum(1) / (Y.norm(dim=1) * R ** 0.5)).mean()), float(((Zs - Y) ** 2).mean())
    @torch.no_grad()
    def beam_paths(Xs, Bm):
        """beam search over the bitshift trellis: keep the Bm cheapest partial paths (successor of s: ((s << 1) & (S-1)) | b)."""
        n_, r_ = Xs.shape; S = 2 ** L; c0 = (Xs[:, :1] - tab[None]) ** 2; cost, st = c0.topk(Bm, 1, largest=False)
        par = torch.empty(r_ - 1, n_, Bm, dtype=torch.int16, device=dev); sts = torch.empty(r_, n_, Bm, dtype=torch.int16, device=dev); sts[0] = st
        for j in range(1, r_):
            nxt = ((st[:, :, None] << 1) & (S - 1)) | torch.arange(2, device=dev); nc = cost[:, :, None] + (Xs[:, j, None, None] - tab[nxt]) ** 2
            cost, ix = nc.reshape(n_, -1).topk(Bm, 1, largest=False); st = nxt.reshape(n_, -1).gather(1, ix); par[j - 1] = (ix // 2).to(torch.int16); sts[j] = st.to(torch.int16)
        b = cost.argmin(1); path = torch.empty(n_, r_, dtype=torch.long, device=dev)
        for j in range(r_ - 1, -1, -1):
            path[:, j] = sts[j].gather(1, b[:, None])[:, 0].long() & (S - 1)
            if j: b = par[j - 1].gather(1, b[:, None])[:, 0].long()
        return path
    for chunk in (1024, 2048, 4096):
        torch.cuda.synchronize(); t0 = time.perf_counter(); P = torch.cat([viterbi_paths(Zs[i:i + chunk], Tt, L, alpha) for i in range(0, len(Zs), chunk)]); torch.cuda.synchronize()
        tt = time.perf_counter() - t0; cv, mse = cval(P)
        emit({"encoder": "viterbi (full, 1024 states)", "chunk": chunk, "vec_per_s": round(len(Zs) / tt, 1), "c_stage1": round(cv, 5), "mse": round(mse, 5),
              "proj_min_10M": round(1e7 / (len(Zs) / tt) / 60, 1), "proj_min_100M": round(1e8 / (len(Zs) / tt) / 60, 1)})
    for Bm in (1, 4, 16, 64, 256):
        torch.cuda.synchronize(); t0 = time.perf_counter(); P = torch.cat([beam_paths(Zs[i:i + 4096], Bm) for i in range(0, len(Zs), 4096)]); torch.cuda.synchronize()
        tt = time.perf_counter() - t0; cv, mse = cval(P)
        emit({"encoder": f"beam {Bm}", "chunk": 4096, "vec_per_s": round(len(Zs) / tt, 1), "c_stage1": round(cv, 5), "mse": round(mse, 5),
              "proj_min_10M": round(1e7 / (len(Zs) / tt) / 60, 1), "proj_min_100M": round(1e8 / (len(Zs) / tt) / 60, 1)})

elif EXP == "grid":                                                                     # FanarVSE parameter grid (baselines were swept widely too)
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    for nprobe in (12, 16, 24, 32, 48):
        for K in (150, 200, 300, 400, 600):
            I, reads, C, m, s = eng.bench(K, 2.0, nprobe)
            emit({"nprobe": nprobe, "K": K, "recall@10": recall(I), "cand_ceiling": cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1), **read_stats(reads)})

elif EXP == "highrecall":                                                               # the >= 0.999 regime, three timed runs (run with FANAR_R=384)
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    for nprobe, K in ((48, 400), (64, 400), (64, 600), (64, 800)):
        I, reads, C, m, s = eng.bench(K, 2.0, nprobe)
        emit({"R": R, "nprobe": nprobe, "K": K, "recall@10": recall(I), "cand_ceiling": cand_recall(C), "QPS": round(m, 1), "QPS_std": round(s, 1),
              "gpu_bytes": s1["words"].shape[1] * 4 + 2 + 4 * d // 8 + 4 + 10, "stage1_bytes": s1["words"].shape[1] * 4 + 2, **read_stats(reads)})

elif EXP == "platform":
    s1 = build_stage1(0); eng = Engine(s1, build_stage2(s1, rotation("haar")))
    for nprobe, K in ((8, 200), (32, 200), (32, 400)):
        I, reads, C, m, s = eng.bench(K, 2.0, nprobe)
        emit({"os": platform.system(), "what": f"FanarVSE np{nprobe} K{K}", "recall@10": recall(I), "QPS": round(m, 1), "QPS_std": round(s, 1), "reads_mean": round(float(reads.mean()), 2)})
    for B in (1, 16):
        lat = []
        for st in range(0, 200 * B, B):
            torch.cuda.synchronize(); t0 = time.perf_counter(); g = eng.gpu_stage(Qg[st % NQ:st % NQ + B], 200, 2.0, 32); eng.cpu_stage(Qn[st % NQ:st % NQ + B], *g); lat.append(time.perf_counter() - t0)
        emit({"os": platform.system(), "what": f"FanarVSE latency batch {B}", "ms_p50": round(1000 * float(np.median(lat[20:])), 3)})
    del eng; torch.cuda.empty_cache()
    x = torch.zeros(1, device=dev); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20000): x.add_(1)
    torch.cuda.synchronize(); emit({"os": platform.system(), "what": "kernel launch, async (us)", "value": round((time.perf_counter() - t0) / 20000 * 1e6, 2)})
    t0 = time.perf_counter()
    for _ in range(5000): x.add_(1); torch.cuda.synchronize()
    emit({"os": platform.system(), "what": "kernel launch + sync round trip (us)", "value": round((time.perf_counter() - t0) / 5000 * 1e6, 2)})
    h = torch.empty(256 * 2 ** 20, dtype=torch.uint8).pin_memory(); g_ = torch.empty_like(h, device=dev)
    for name, f in (("H2D pinned 256MB (GB/s)", lambda: g_.copy_(h, non_blocking=True)), ("D2H pinned 256MB (GB/s)", lambda: h.copy_(g_, non_blocking=True))):
        f(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): f()
        torch.cuda.synchronize(); emit({"os": platform.system(), "what": name, "value": round(10 * 256 / 1024 / (time.perf_counter() - t0), 2)})
    A = torch.randn(2000, d, device=dev, dtype=torch.float16); Bm = torch.randn(d, 100000, device=dev, dtype=torch.float16); A @ Bm; torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): A @ Bm
    torch.cuda.synchronize(); emit({"os": platform.system(), "what": "fp16 GEMM 2000x1536x100k (TFLOP/s)", "value": round(20 * 2 * 2000 * d * 1e5 / (time.perf_counter() - t0) / 1e12, 2)})
    ids = torch.from_numpy(np.random.default_rng(0).integers(0, N, 12000)); host16.index_select(0, ids); t0 = time.perf_counter()
    for _ in range(10): host16.index_select(0, ids)
    emit({"os": platform.system(), "what": "host gather of 12k random fp16 rows (ms)", "value": round((time.perf_counter() - t0) / 10 * 1000, 2)})

elif EXP == "scale":
    s1 = build_stage1(0); Qb = Qg[:500]; Qr = Qb @ s1["W1"]; QC = Qb @ s1["c"]
    for rep in (1, 2, 5, 10):
        words = s1["words"].repeat(rep, 1); scale = s1["scale1"].repeat(rep)
        f = lambda: trellis_search(Qr, QC, words, s1["tab"], scale, L, 400)
        f(); torch.cuda.synchronize(); ts = []
        for _ in range(REPS):
            t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        emit({"N": N * rep, "stage1_full_scan_ms_per_500q": round(1000 * float(np.mean(ts)), 2), "stage1_QPS_full_scan": round(500 / float(np.mean(ts)), 1),
              "stage1_gpu_MB": round((words.numel() * 4 + scale.numel() * 2) / 2 ** 20, 1)})
        del words, scale; torch.cuda.empty_cache()
print("DONE", flush=True)
