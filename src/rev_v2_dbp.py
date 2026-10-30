"""[reviewer round: 3 timed repeats, results to rev_v2_dbp.json] Pipelined FanarVSE: GPU fused 1-bit scan (codes only on GPU) || CPU exact rerank from fp16 host vectors.
GPU produces candidate ids for batch i+1 while CPU threads rerank batch i (only ids cross PCIe)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, glob, json, time, threading, queue, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, f"{FANAR}/src")
from fused_scan import search1
dev = "cuda"


class PipelinedDB:
    def __init__(self, X, dims=768, BN=32, seed=0):
        torch.manual_seed(seed); N, d = X.shape; self.N, self.d, self.BN = N, d, BN
        self.host16 = torch.from_numpy(X).half()                                   # rerank tier (host RAM, fp16)
        S = torch.from_numpy(X[:50000]).to(dev); self.c = S.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((S - self.c).T)); V = V.flip(1); del S
        self.W = V[:, :dims] @ torch.linalg.qr(torch.randn(dims, dims, device=dev))[0]
        self.codes = torch.empty(N, dims // 8, dtype=torch.uint8, device=dev); self.scale = torch.empty(N, dtype=torch.float16, device=dev)
        shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
        for j in range(0, N, 100000):
            Z = (torch.from_numpy(X[j:j + 100000]).to(dev) - self.c) @ self.W; nz = Z.norm(dim=1); O = Z / nz[:, None]
            self.codes[j:j + len(Z)] = (((O > 0).reshape(len(Z), -1, 8).to(torch.uint8)) << shifts).sum(2).to(torch.uint8)
            self.scale[j:j + len(Z)] = (nz / O.abs().sum(1)).half()

    def gpu_bytes_per_vector(self): return self.codes.shape[1] + 2

    def _rerank(self, Qc, cand, k):
        X = self.host16.index_select(0, cand.reshape(-1)).reshape(cand.shape[0], cand.shape[1], self.d).float()
        sc = torch.einsum("bkd,bd->bk", X, Qc)
        return cand.gather(1, sc.topk(k, 1).indices).numpy()

    def search(self, Q, k=10, kp=40, batch=500, workers=4):
        Qg = torch.as_tensor(Q, device=dev); Qc = torch.as_tensor(Q); QZ = Qg @ self.W; QC = Qg @ self.c
        futs = []
        with ThreadPoolExecutor(workers) as ex:
            for s in range(0, len(Q), batch):
                cand = search1(QZ[s:s + batch], QC[s:s + batch], self.codes, self.scale, kp, BN=self.BN).cpu()   # GPU stage
                futs.append(ex.submit(self._rerank, Qc[s:s + batch], cand, k))                                    # CPU stage (overlaps next GPU batch)
            return np.vstack([f.result() for f in futs])


if __name__ == "__main__":
    import pyarrow.parquet as pq_
    NQ = 2000; files = sorted(glob.glob(f"{FANAR}/data/dbp/shard*.parquet"))[:6]
    X = np.concatenate([np.array(pq_.read_table(p, columns=["openai"]).column("openai").to_pylist(), np.float32) for p in files])
    X = X[np.random.default_rng(0).permutation(len(X))]; X -= X[:50000].mean(0); X /= np.linalg.norm(X, axis=1, keepdims=True)
    T, Q = np.ascontiguousarray(X[:-NQ]), np.ascontiguousarray(X[-NQ:]); del X
    GT = np.load(f"{FANAR}/results/gt_dbp_wsl.npy")
    out = []
    for dims in (384, 768):
        db = PipelinedDB(T, dims)
        for kp in (40, 100):
            for batch, workers in ((500, 4), (500, 8)):
                db.search(Q[:500], kp=kp, batch=batch, workers=workers); torch.cuda.synchronize(); q = []
                for _ in range(3):
                    t0 = time.perf_counter(); I = db.search(Q, kp=kp, batch=batch, workers=workers); q.append(NQ / (time.perf_counter() - t0))
                out.append({"ds": "dbp229k", "dims": dims, "kp": kp, "batch": batch, "workers": workers, "gpu_bytes": db.gpu_bytes_per_vector(),
                            "recall@10": round(float(np.mean([len(set(I[i]) & set(GT[i])) / 10 for i in range(NQ)])), 4), "QPS": round(float(np.mean(q)), 1), "QPS_std": round(float(np.std(q)), 1)})
                print(json.dumps(out[-1]), flush=True); json.dump(out, open(f"{FANAR}/results/rev_v2_dbp.json", "w"), indent=1)
        del db; torch.cuda.empty_cache()
