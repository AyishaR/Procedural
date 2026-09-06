#!/usr/bin/env python3
"""rho per block at init: random baseline vs the checkpoint-free recipe (blocks 9-11 -> rho 1.4).

Data from plots/cache/init_rho.json (plots/measure_init_rho_arms.py, 256 real val images,
rho defined exactly as in engine.attention_residual_analysis).
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent / "out"
D = json.load(open(Path(__file__).resolve().parent / "cache" / "init_rho.json"))
C_R, C_RECIPE, TARGET = "#8c8c8c", "#DE8F05", 1.4


def series(arm, key):
    s = D[arm]
    return [ (s[str(i)] if str(i) in s else s[i])[key] for i in range(12) ]


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.arange(12)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    for ax, key, title in [(axes[0], "rho_attn", "Attention sublayer"),
                           (axes[1], "rho_mlp", "MLP sublayer")]:
        ax.axvspan(8.5, 11.5, color=C_RECIPE, alpha=0.08, lw=0, zorder=0)
        ax.axhline(TARGET, color=C_RECIPE, ls=":", lw=1.3, zorder=1)
        # random is drawn as a wide halo: blocks 0-8 are identical by construction, so the
        # recipe line sits inside it there and the two only separate at block 9.
        ax.plot(x, series("r", key), "-", color=C_R, lw=6.5, alpha=0.45,
                solid_capstyle="round", label="random init", zorder=2)
        ax.plot(x, series("r", key), "o", color=C_R, ms=5, zorder=3)
        ax.plot(x, series("ftbrho", key), "-o", color=C_RECIPE, lw=2.0, ms=5.5,
                label=r"recipe: blocks 9-11 set to $\rho$ = 1.4", zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel("block index")
        ax.set_xticks(x)
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.text(10, TARGET * 1.16, r"target $\rho$ = 1.4", ha="center", fontsize=8,
                color=C_RECIPE, fontweight="bold")
        ax.text(10, 0.108, "calibrated\nblocks", ha="center", fontsize=8, color=C_RECIPE)

    axes[0].set_ylabel(r"$\rho$  =  $\|\Delta_{\rm sublayer}\| \, / \, \|r_{\rm in}\|$   (log)")
    axes[0].legend(loc="lower left")
    axes[0].set_ylim(0.09, 3.0)

    fig.suptitle("What the checkpoint-free recipe does at init: the last three blocks write "
                 "6-9x more into the residual stream",
                 y=1.04, fontsize=11.5, fontweight="bold")
    fig.text(0.0, -0.05,
             "ViT-B/16, measured on 256 real ImageNet val images with no training; "
             r"$\rho$ defined as in engine.attention_residual_analysis." "\n"
             "Blocks 0-8 are untouched and identical to random init. Reaching the target needs "
             "COMPOUNDING factors, because scaling block 9 inflates the stream block 10 is then "
             "measured against:\n"
             "attention (v and proj each) x2.90, x4.63, x7.72 — so the write itself moves x8.4, "
             "x21.5, x59.5 — and mlp.fc2 x9.78, x28.42, x84.93.\n"
             "Random init's rho DECAYS with depth (attention 0.37 -> 0.15, MLP 0.61 -> 0.23); the recipe "
             "inverts that in the last quarter — a 9.2x jump for attention at block 11, 6.2x for the MLP.\n"
             "Blocks 0-8 coincide exactly, so the grey halo is the recipe line there. Worth +1.61 top-1 over "
             "random init, with no checkpoint involved.",
             fontsize=7.8, color="0.35", va="top")
    fig.savefig(OUT / "fig8_recipe_rho.png")
    print("wrote", OUT / "fig8_recipe_rho.png")
    for k in ["rho_attn", "rho_mlp"]:
        print(f"{k}: random {[round(v,3) for v in series('r',k)]}")
        print(f"{k}: recipe {[round(v,3) for v in series('ftbrho',k)]}")


if __name__ == "__main__":
    main()
