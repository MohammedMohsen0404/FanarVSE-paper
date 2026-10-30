import os as _osf; _osf.makedirs("figures/out", exist_ok=True)   # output folder (run from the repo root)
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [json.loads(l) for p in glob.glob("results/exp35b_*_r1.log") for l in open(p) if l.startswith("{")]
names = {"glove": "GloVe-100", "arxiv": "arXiv\nInstructorXL-768", "oai3": "OpenAI-3-large\n1536", "dbp": "OpenAI-ada\n1536"}
order = ["glove", "arxiv", "oai3", "dbp"]
val = lambda ds, cb, key: [r[key] for r in rows if r["ds"] == ds and r["codebook"] == cb and r["nprobe"] == 64][0]
colors = {"E8-256": "#2a78d6", "BW16-65536": "#eb6834"}; labels = {"E8-256": "E8 lattice (1982)", "BW16-65536": "Barnes-Wall BW16 (1959)"}
plt.rcParams.update({"font.size": 14.00, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "figure.titleweight": "bold", "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0})

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor="#fcfcfb")
x = np.arange(len(order)); w = 0.36
for a, key, title in ((ax[0], "R@10", "No rerank"), (ax[1], "R@10_rerank20", "Rerank top-20")):
    a.set_facecolor("#fcfcfb")
    for j, cb in enumerate(("E8-256", "BW16-65536")):
        g = [100 * (val(ds, cb, key) - val(ds, "hypercube", key)) for ds in order]
        bars = a.bar(x + (j - 0.5) * w, g, w - 0.04, color=colors[cb], label=labels[cb])
        for b, v in zip(bars, g): a.text(b.get_x() + b.get_width() / 2, v + 0.12, f"+{v:.1f}", ha="center", va="bottom", fontsize=11.20, color="#52514e")
    a.axhline(0, color="#52514e", lw=0.8); a.set_xticks(x); a.set_xticklabels([names[d] for d in order], fontsize=11.20)
    a.set_title(title, color="#0b0b0b"); a.grid(axis="y", color="#e4e3df", lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("Recall@10 gain over hypercube (points)"); ax[0].legend(frameon=False, fontsize=11.20, loc="upper right")
fig.suptitle("Same 1 bit/dim, same IVF, same estimator: only the codebook changes (nprobe 64)", color="#0b0b0b", fontsize=12.60)
plt.tight_layout(); plt.savefig("figures/out/fig_lattice_gain.png", dpi=160, facecolor="#fcfcfb"); print("ok")
