"""Recall vs QPS at 998k with the final engine (v19 results; run from <repo root>). Same baselines as rev_figs.py."""
import os as _osf; _osf.makedirs("figures/out", exist_ok=True)   # output folder (run from the repo root)
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("figures/rev_figs.py").read().split("# ---------------- Fig: recall vs QPS at 998k")[0])
v1 = pareto([(x["recall@10"], x["QPS"]) for x in J("rev_fanardb_grid.json")] + [(x["recall@10"], x["QPS"]) for x in J("rev_fanardb_highrecall_R384.json")])
v3 = pareto([(x["recall@10"], x["QPS"]) for f in ("v19_sweep.json", "v19_sweep_R384.json") for x in J(f)])
cag = [(x["recall"], x["QPS"]) for x in J("rev_diag2.json") if x["engine"] == "CAGRA" and x["k"] == 10]
cag += [(x["recall@10"], x["QPS"]) for x in J("baselines_1m_rep.json") if x["engine"].startswith("CAGRA")]
cag = pareto(cag)
rb9 = sorted((x["recall"], x["QPS"]) for x in J("rev_rabitq_verify.json") if x["bits"] == 9 and x["n_lists"] == 4096)
rb5 = sorted((x["recall"], x["QPS"]) for x in J("rev_baselines.json") if x["engine"] == "IVF-RaBitQ" and x["cfg"]["bits"] == 5 and x["k"] == 10 and x["batch"] == 500)
pq = [(x["recall@10"], x["QPS"]) for x in J("baselines_1m_rep.json") if "IVF-PQ" in x["engine"]]
flat = [(x["recall@10"], x["QPS"]) for x in J("baselines_1m_rep.json") if "Flat" in x["engine"]]
lean = pareto([(x["recall@10"], x["QPS"]) for f in ("v19_lean.json", "v19_lean_R384.json") for x in J(f) if "recall@10" in x])
leanssd = pareto([(x["recall@10"], x["QPS"]) for f in ("v19_nvme.json", "v19_nvme_R384.json") for x in J(f) if "recall@10" in x])
fig, a = plt.subplots(figsize=(6.4, 4.3), facecolor=BG); style(a)
for name, col, m, pts, ls in (("FanarVSE (570–586 B/vec)", C[0], "o", v3, "-"), ("FanarVSE-v1 (820–836 B/vec)", C[0], "o", v1, ":"),
                              ("FanarVSE-lean, rest in host RAM (66–82 B/vec)", C[2], "s", lean, "-"), ("FanarVSE-lean, full vectors on NVMe (66–82 B/vec)", C[2], "s", leanssd, "--"),
                              ("CAGRA, best of deg 32/64 (3,200–3,328 B/vec)", C[1], "^", cag, "-"),
                              ("IVF-RaBitQ 9 bit (1,851 B/vec)", C[5], "D", rb9, "-"), ("IVF-RaBitQ 5 bit (1,101 B/vec)", C[5], "d", rb5, "--"),
                              ("cuVS IVF-PQ + refine (104 B/vec)", C[3], "v", pq, "-"), ("FAISS-GPU flat fp16 (3,072 B/vec)", C[4], "P", flat, "none")):
    x, y = zip(*pts); a.plot(x, y, marker=m, color=col, lw=1.4, ms=4 if ls != ":" else 2.5, label=name, ls=ls, alpha=1 if ls != ":" else 0.6)
a.set_yscale("log"); a.set_xlabel("Recall@10"); a.set_ylabel("Queries per second (batch 500)"); a.set_xlim(0.88, 1.0015)
a.legend(fontsize=8.68, frameon=False, loc="lower left"); a.set_title("OpenAI-1536, 998k vectors, RTX 3050 (6 GB)", color=INK)
save("fig_recall_qps_1m")
