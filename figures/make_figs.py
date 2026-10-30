import os as _osf; _osf.makedirs("figures/out", exist_ok=True)   # output folder (run from the repo root)
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]; INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 12.60, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12.60})
plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.titleweight": "bold", "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0})

def style(a):
    a.set_facecolor(BG); a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
# Fig 2: distortion vs rate
fig, a = plt.subplots(figsize=(5.4, 4.0), facecolor=BG); style(a)
R = np.linspace(0.8, 2.2, 50); a.plot(R, 10 * np.log10(2.0 ** (-2 * R)), color=INK2, lw=1.2, ls="--", label="Shannon bound $D(R)=2^{-2R}$")
a.plot([1, 2], 10 * np.log10([0.3647, 0.1200]), "o", color=C[1], ms=6, label="Lloyd-Max per coordinate (TurboQuant)")
for L, col, m in ((8, C[3], "D"), (10, C[2], "s"), (12, C[0], "^")):
    pts = {8: [(1.031, 0.2954), (2.031, 0.0857)], 10: [(1.039, 0.2779), (2.039, 0.0768)], 12: [(1.047, 0.2644), (2.047, 0.0710)]}[L]
    x, y = zip(*pts); a.plot(x, 10 * np.log10(y), m, color=col, ms=6, label=f"bitshift trellis L={L}")
a.set_xlabel("Rate (bits/coordinate, incl. overhead)"); a.set_ylabel("MSE (dB), N(0,1) source"); a.legend(fontsize=9.10, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig("figures/out/fig_distortion.pdf"); plt.savefig("figures/out/fig_distortion.png", dpi=200)
# Fig 3: identification sizing
fig, a = plt.subplots(figsize=(5.4, 4.0), facecolor=BG); style(a)
for name, col, m, pts in (("sign codes (1 b/dim)", C[1], "o", [(128, .9325), (192, .9799), (256, .9921), (384, .9983)]),
                          ("trellis 1 b/dim", C[0], "^", [(138, .9588), (202, .9891), (266, .9966), (394, .9989)]),
                          ("trellis 0.5 b/dim", C[2], "s", [(74, .8521), (106, .9366), (138, .9659), (202, .9871)])):
    x, y = zip(*pts); a.plot(x, y, marker=m, color=col, lw=1.6, ms=5, label=name)
a.axvspan(104, 169, color=GRID, alpha=0.7, lw=0); a.axvline(140, color=INK2, ls="--", lw=1)
a.text(205, 0.895, "identification estimate\n~140 bits (participation\nratio; band: 10th-90th\npercentile of $\\tau$)", fontsize=8.82, color=INK2)
a.set_xlabel("Stage-1 code size (bits per vector)"); a.set_ylabel("True top-10 in top-200"); a.legend(fontsize=9.10, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig("figures/out/fig_identification.pdf"); plt.savefig("figures/out/fig_identification.png", dpi=200)
print("ok")
