"""Figure for the direct tests of the theory (results/theory2_*.json) -> figs/fig_theory.{pdf,png} in both paper folders."""
import os as _osf; _osf.makedirs("figures/out", exist_ok=True)   # output folder (run from the repo root)
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import betainc
from scipy import stats
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]; INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 11.20, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11.90})
plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.titleweight": "bold", "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0})

def style(a):
    a.set_facecolor(BG); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
R = f"{FANAR}/results/"
law = json.load(open(R + "theory2_law.json")); iso = [r for r in json.load(open(R + "theory2_iso.json")) if "mse_trE" in r]
cov = json.load(open(R + "theory2_cover.json"))
tail = lambda z, D: betainc((D - 2) / 2, 0.5, max(0.0, 1 - z ** 2 / (D - 1)))
CBS = [("sign", "sign", C[1], "o"), ("e8", "E8", C[3], "D"), ("lm4", "4-bit LM", C[0], "s"), ("trellis", "trellis", C[2], "^")]
fig, ax = plt.subplots(2, 2, figsize=(7.8, 6.3), facecolor=BG); [style(a) for a in ax.flat]
Ds = np.unique([r["D"] for r in law]); Dc = np.geomspace(8, 1536, 200)
# (a) P(|Z|>2) vs dimension
a = ax[0, 0]
for key, name, col, m in CBS:
    pts = [(r["D"], r["P>2"]) for r in law if r["codebook"] == key]; x, y = zip(*pts)
    a.plot(np.array(x) * (1 + 0.04 * CBS.index((key, name, col, m))), np.array(y) * 100, m, color=col, ms=3.5, alpha=0.8, label=name, ls="none")
a.plot(Dc, [100 * tail(2, D) for D in Dc], color=INK, lw=1.2, label="exact law (Cor. 1)"); a.axhline(100 * 2 * stats.norm.sf(2), color=INK2, ls="--", lw=1, label="Gaussian")
a.set_xscale("log"); a.set_xlabel("dimension $D$"); a.set_ylabel("$P(|Z|>2)$ (%)"); a.set_title("(a) tail mass, 3 rotations each"); a.legend(fontsize=8.40, frameon=False, loc="lower right")
# (b) excess kurtosis vs dimension
a = ax[0, 1]
for key, name, col, m in CBS:
    pts = [(r["D"], r["excess_kurt"]) for r in law if r["codebook"] == key]; x, y = zip(*pts)
    a.plot(np.array(x) * (1 + 0.04 * CBS.index((key, name, col, m))), y, m, color=col, ms=3.5, alpha=0.8, ls="none")
a.plot(Dc, -6 / (Dc + 1), color=INK, lw=1.2); a.axhline(0, color=INK2, ls="--", lw=1)
a.set_xscale("log"); a.set_xlabel("dimension $D$"); a.set_ylabel("excess kurtosis of $Z$"); a.set_title("(b) kurtosis: law $-6/(D+1)$")
# (c) isotropy: recall vs tr(E) and vs tr(Sigma_q E)
a = ax[1, 0]; gam = sorted({r["gamma(E ~ lambda^gamma)"] for r in iso}); cols = plt.cm.viridis(np.linspace(0, 0.9, len(gam)))
for g, col in zip(gam, cols):
    pts = sorted((r["tr(SqE)"], r["recall@10"]) for r in iso if r["gamma(E ~ lambda^gamma)"] == g); x, y = zip(*pts)
    a.plot(x, y, "o-", color=col, ms=3, lw=1, label=f"$\\gamma={g:g}$")
a.set_xscale("log"); a.set_xlabel(r"ranking error $\mathrm{tr}(\Sigma_q E)$"); a.set_ylabel("recall@10 (estimator)")
a.set_title("(c) error $E\\propto\\lambda^\\gamma$, MSE 0.05--0.8"); a.legend(fontsize=8.12, frameon=False, ncol=2, loc="upper right")
ins = a.inset_axes([0.1, 0.1, 0.36, 0.36]); ins.set_facecolor(BG)
for g, col in zip(gam, cols):
    pts = sorted((r["mse_trE"], r["recall@10"]) for r in iso if r["gamma(E ~ lambda^gamma)"] == g); x, y = zip(*pts); ins.plot(x, y, "o-", color=col, ms=1.8, lw=0.8)
ins.set_xscale("log"); ins.set_xticks([0.05, 0.2, 0.8]); ins.set_xticklabels(["0.05", "0.2", "0.8"]); ins.minorticks_off(); ins.set_xlabel("MSE tr$(E)$", fontsize=7.70, labelpad=1); ins.tick_params(labelsize=7.00, pad=1)
# (d) per-query coverage failure vs union bound
a = ax[1, 1]
for K, col in ((200, C[0]), (400, C[1])):
    rr = [r for r in cov if r.get("K") == K and "z" in r and r["z"] >= 3.0]; z = [r["z"] for r in rr]
    a.plot(z, [max(r["union_bound_K*p"], 1e-5) for r in rr], "-", color=col, lw=1.2, label=f"bound $K\\,P(|Z|>z)$, $K={K}$")
    a.plot(z, [max(r["P_fail_query_exact_sigma"], 1e-5) for r in rr], "o", color=col, ms=4, label="observed, exact $\\sigma$")
    a.plot(z, [max(r["P_fail_query_engine_sigma"], 1e-5) for r in rr], "x", color=col, ms=4, label="observed, engine $\\sigma$ ($\\|q\\|$)")
a.set_yscale("log"); a.set_ylim(3e-4, 1.2); a.set_xlabel("half-width $z$"); a.set_ylabel("share of queries uncovered")
a.set_title("(d) coverage, 2,000 queries"); a.legend(fontsize=6.44, frameon=False, loc="upper right")
a.text(4.95, 4.5e-4, "0 of 2,000\nat $z=5$", fontsize=7.70, color=INK2, ha="right")
plt.tight_layout()
for d in (f"{FANAR}/figures/out/", f"{FANAR}/figures/out/"):
    plt.savefig(d + "fig_theory.pdf"); plt.savefig(d + "fig_theory.png", dpi=220)
print("ok")
