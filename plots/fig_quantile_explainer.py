#!/usr/bin/env python3
"""What the quantile-matching arm (`ftbqm`, row 3 of fig5) actually does to a weight tensor.

The operation, per 2-D weight tensor in blocks 0-8:

    sort the procedural tensor's values,
    sort the random tensor's values,
    write proc's k-th smallest value into the slot holding random's k-th smallest.

The result therefore carries proc's ENTIRE value multiset -- norm, variance, kurtosis, every
quantile -- while the question of *which* weight sits *where* is inherited from the random init.
It is the image-processing "histogram matching" operation: take the tonal distribution from one
image and the pixel ranking from another.

Run:  .venv/bin/python plots/fig_quantile_explainer.py
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / "cache" / "row3_tensors.npz"
OUT = Path(__file__).resolve().parent / "out"

C_RAND, C_PROC, C_OUT = "#8c8c8c", "#0173B2", "#CC78BC"


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(CACHE, allow_pickle=True)
    rand, proc, out, KEY = z["rand"], z["proc"], z["out"], str(z["key"])

    def stats(t):
        zz = (t - t.mean()) / t.std()
        return float(np.linalg.norm(t)), float(t.std()), float((zz ** 4).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1))

    # ---- panel 1: histograms ----
    ax = axes[0]
    lim = float(np.percentile(np.abs(proc), 99.98))
    bins = np.linspace(-lim, lim, 200)
    ax.hist(rand, bins=bins, color=C_RAND, alpha=0.55, label="random init", density=True)
    ax.hist(proc, bins=bins, histtype="step", lw=2.4, color=C_PROC,
            label="procedural", density=True)
    ax.hist(out, bins=bins, histtype="step", lw=1.6, color=C_OUT, ls="--",
            label="result (ftbqm)", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("weight value"); ax.set_ylabel("density (log)")
    ax.set_title("1. The result's histogram IS proc's")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.02, 0.03,
            "the dashed line sits exactly on the solid one:\nsame values, to the last element.\n"
            "note proc is ~3.7x WIDER than random init.",
            transform=ax.transAxes, fontsize=7.6, color="0.35", va="bottom")

    # ---- panel 2: result vs random, showing the monotone map ----
    ax = axes[1]
    rng = np.random.default_rng(0)
    idx = rng.choice(rand.size, 4000, replace=False)
    ax.scatter(rand[idx], out[idx], s=3, alpha=0.25, color=C_OUT, lw=0)
    ax.set_xlabel("value at that slot in random init")
    ax.set_ylabel("value at that slot in the result")
    ax.set_title("2. ...but WHERE each value goes\ncomes from the random init")
    ax.text(0.03, 0.95,
            "a monotone curve: the biggest weight in the\nrandom tensor becomes the biggest weight\n"
            "in proc's set, and so on down.\nNothing about proc's arrangement survives.",
            transform=ax.transAxes, fontsize=7.8, color="0.3", va="top")
    ax.grid(alpha=0.25, lw=0.5)

    # ---- panel 3: the toy example, spelled out ----
    ax = axes[2]
    ax.axis("off")
    R = [0.3, -0.1, 0.9, -0.5, 0.2]
    P = [2.0, -3.0, 0.1, 5.0, -0.4]
    Ps = sorted(P)
    ranks = [sorted(R).index(v) for v in R]
    res = [Ps[r] for r in ranks]
    rows = [("slot", "random", "its rank", "-> result")]
    rows += [(str(i), f"{R[i]:+.1f}", str(ranks[i]), f"{res[i]:+.1f}") for i in range(5)]
    ax.text(0.5, 1.02, "3. A five-weight example", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(0.02, 0.88, f"procedural values, sorted:  {[f'{v:+.1f}' for v in Ps]}",
            transform=ax.transAxes, fontsize=8.4, color=C_PROC, fontweight="bold")
    y = 0.76
    for j, row in enumerate(rows):
        for x, cell in zip([0.06, 0.30, 0.54, 0.80], row):
            ax.text(x, y, cell, transform=ax.transAxes, fontsize=9,
                    fontweight="bold" if j == 0 else "normal",
                    color="0.25" if j == 0 else "0.1", family="monospace")
        y -= 0.105
        if j == 0:
            ax.plot([0.04, 0.96], [y + 0.055, y + 0.055], transform=ax.transAxes,
                    color="0.8", lw=1)
    ax.text(0.02, 0.13,
            "The slot holding random's smallest weight (slot 3)\n"
            "receives proc's smallest value (-3.0), and so on.\n\n"
            "Result: proc's exact set of numbers, arranged by\nrandom init's ordering.",
            transform=ax.transAxes, fontsize=8.2, color="0.3", va="top")

    n1, s1, k1 = stats(rand); n2, s2, k2 = stats(proc); n3, s3, k3 = stats(out)
    fig.suptitle("Row 3 of fig5: what `ftbqm` does to every 2-D weight tensor in blocks 0-8",
                 y=1.06, fontsize=11.5, fontweight="bold")
    fig.text(0.0, -0.05,
             f"Tensor shown: {KEY} (ViT-B, {proc.size:,} weights).   "
             f"‖W‖  random {n1:.2f} / proc {n2:.2f} / result {n3:.2f}.   "
             f"kurtosis  random {k1:.2f} / proc {k2:.2f} / result {k3:.2f}.\n"
             "The result matches proc on every distributional statistic by construction, and keeps none "
             "of proc's structure. That arm scores +0.01 over random init — so the value distribution, "
             "on its own, carries nothing.",
             fontsize=7.8, color="0.35", va="top")
    fig.savefig(OUT / "fig_row3_quantile_explainer.png")
    print("wrote", OUT / "fig_row3_quantile_explainer.png")
    print(f"norms   random {n1:.3f} proc {n2:.3f} result {n3:.3f}")
    print(f"kurtosis random {k1:.3f} proc {k2:.3f} result {k3:.3f}")
    print("result multiset == proc's:", np.array_equal(np.sort(out), np.sort(proc)))


if __name__ == "__main__":
    main()
