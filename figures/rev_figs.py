"""Figures for the revised paper, read directly from the result JSON files (run from <repo root>)."""
import os as _osf; _osf.makedirs("figures/out", exist_ok=True)   # output folder (run from the repo root)
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc8"]; INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 12.60, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12.60})
plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.titleweight": "bold", "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0})

OUT = ["figures/out/", "figures/out/"]
J = lambda f: json.load(open("results/" + f))
def style(a):
    a.set_facecolor(BG); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
def save(name):
    plt.tight_layout()
    for o in OUT: plt.savefig(o + name + ".pdf"); plt.savefig(o + name + ".png", dpi=200)
def pareto(pts):
    """keep points not dominated in (recall up, QPS up); sorted by recall."""
    pts = sorted(set(pts), key=lambda p: (-p[0], -p[1])); out, best = [], -1
    for r, q in pts:
        if q > best: out.append((r, q)); best = q
    return sorted(out)

# ---------------- Fig: recall vs QPS at 998k (drawn by fig_1m_v4.py)
# ---------------- Fig: error bars (read distribution, z sweep)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), facecolor=BG)
a = ax[0]; style(a)
for K, col in ((200, C[0]), (400, C[1])):
    r = np.load(f"results/rev_reads_K{K}_z2.npy"); b = np.arange(0, r.max() + 2) - 0.5
    a.hist(r, bins=b, histtype="step", color=col, lw=1.4, label=f"K={K}: mean {r.mean():.1f}, p99 {np.percentile(r, 99):.0f}, max {r.max()}")
a.set_xlabel("Exact reads per query (z = 2)"); a.set_ylabel("Queries"); a.set_ylim(0, 430); a.legend(fontsize=8.82, frameon=False, loc="upper right")
a = ax[1]; style(a); zs = [x for x in J("rev_fanardb_zsweep.json") if x["K"] == 400 and x["z"] != "inf"]
zs = sorted(zs, key=lambda x: x["z"]); a2 = a.twinx()
a.plot([x["z"] for x in zs], [x["reads_mean"] for x in zs], "o-", color=C[0], ms=4, label="reads / query")
a2.plot([x["z"] for x in zs], [x["recall@10"] for x in zs], "s--", color=C[1], ms=4, label="recall@10")
for x in zs:
    if x["bound"].startswith("cantelli"): a.annotate("Cantelli\n$\\delta$=" + x["bound"].split("=")[1], (x["z"], x["reads_mean"]), fontsize=8.40, xytext=(-28, 8), textcoords="offset points")
a.set_xscale("log"); a.set_xlabel("Interval half-width z (log)"); a.set_ylabel("Exact reads / query", color=C[0]); a2.set_ylabel("Recall@10", color=C[1])
a2.set_ylim(0.989, 0.993); a2.axhline(0.9923, color=INK2, lw=0.6, ls=":"); a2.spines["top"].set_visible(False)
save("fig_errorbars")
print("figures written")
