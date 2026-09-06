#!/usr/bin/env python3
"""Per-layer statistics DURING training, from wandb (plots/cache/wandb_epochwise.json).

The accuracy gap between ftbqmln (+0.78) and ftb4e3 (+2.08) does not open until epoch ~214
(docs 3.10.9.8). These panels ask which per-layer quantity separates the arms BEFORE that,
i.e. which is a precursor rather than a consequence.

Caveats carried on the figure:
  * grad_norm is the -1 sentinel for every arm except `r`, so it is not plotted.
  * acc_layerN is a per-layer read-out probe: near-zero for early blocks by construction,
    so it must be read as a DEPTH PROFILE, never averaged over blocks 0-8.
  * stats are logged ~every 10 epochs, and ftbqm has 26 points where the others have 33.
"""
import json
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent / "out"
D = json.load(open(Path(__file__).resolve().parent / "cache" / "wandb_epochwise.json"))
PAL = {"r": "#8c8c8c", "ftb4o": "#D55E00", "ftbqm": "#CC78BC", "ftbqmln": "#DE8F05",
       "ftbqm1d": "#B07AA1", "ftb4e3": "#029E73", "ftb3i": "#0173B2"}
LAB = {"r": "random init (78.08)", "ftb4o": r"random @ proc's $\rho$ (77.27)",
       "ftbqm": "proc weight values (78.16)", "ftbqmln": "+ LayerNorm tensors (78.86)",
       "ftbqm1d": "+ ALL 8, pooled qkv (running)",
       "ftb4e3": "+ ALL 8, sliced qkv (80.16)", "ftb3i": "proc, not permuted (79.99)"}
ARMS = ["r", "ftb4o", "ftbqm", "ftbqmln", "ftbqm1d", "ftb4e3", "ftb3i"]
CROSS = 214


def prof(arm, fam, epoch):
    row = D[arm].get(str(epoch))
    if row is None:
        return None
    return [row.get(f"Epoch-wise/{fam}_layer{l}") for l in range(12)]


def series(arm, fam, lo=1, hi=9):
    out = {}
    for e, row in D[arm].items():
        v = [row.get(f"Epoch-wise/{fam}_layer{l}") for l in range(lo, hi)]
        v = [x for x in v if x is not None]
        if v:
            out[int(e)] = float(np.mean(v))
    return dict(sorted(out.items()))


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "font.family": "DejaVu Sans"})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    # ---- (a) rho over training ----
    ax = axes[0]
    for a in ARMS:
        s = series(a, "delta_norm_ratio")
        ax.plot(list(s), list(s.values()), "-o", ms=3, color=PAL[a], lw=1.9, label=LAB[a])
    ax.axvline(CROSS, color="0.4", ls=":", lw=1.2)
    ax.text(CROSS + 5, ax.get_ylim()[1] * 0.97, "accuracy\ncrossover", fontsize=7.4,
            color="0.4", va="top")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$\rho$  (mean over blocks 1-8)")
    ax.set_title("(a) residual write magnitude")
    ax.legend(loc="upper left", fontsize=6.9, ncol=1)
    ax.grid(alpha=0.25, lw=0.5)

    # ---- (b) activation RMS over training ----
    ax = axes[1]
    for a in ARMS:
        s = series(a, "blk_act_rms")
        ax.plot(list(s), list(s.values()), "-o", ms=3, color=PAL[a], lw=1.9)
    ax.axvline(CROSS, color="0.4", ls=":", lw=1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("block activation RMS (blocks 1-8)")
    ax.set_ylim(0, 30)   # ftb3i spikes to ~900 at ep69 (the documented loss spike); clipped
    ax.set_title("(b) residual stream scale")
    ax.text(0.97, 0.96, "y clipped at 30;\nftb3i spikes to ~900 at ep69",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.45")
    ax.grid(alpha=0.25, lw=0.5)

    # ---- (c) acc_layer depth profile ----
    ax = axes[2]
    for a in ARMS:
        p = prof(a, "acc", 149)
        if p is None:
            continue
        ax.plot(range(12), p, "-o", ms=3.5, color=PAL[a], lw=1.9)
    ax.set_xlabel("block index"); ax.set_ylabel("read-out accuracy at that block (%)")
    ax.set_xticks(range(12))
    ax.set_title("(c) where accuracy lives, at epoch 149")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.03, 0.95, "epoch 149 = 65 epochs BEFORE\nthe accuracy curves cross",
            transform=ax.transAxes, fontsize=7.6, color="0.35", va="top")

    fig.suptitle("Per-layer statistics during training: which quantity separates the arms "
                 "BEFORE their test curves do?", y=1.04, fontsize=11.5, fontweight="bold")
    fig.text(0.0, -0.06,
             "ViT-B/16 ImageNet-1k, seed 0, logged ~every 10 epochs (26 points for `proc weight "
             "values`, 33 for the rest). Arms differ ONLY in initialisation.\n"
             "ftb4o is the arm that refutes rho as the mechanism: lowest rho of any arm, WORST final accuracy. "
             "`grad_norm_layerN` is the -1 sentinel for every arm except random init, so it is not "
             "plotted. `acc_layerN` is a per-block read-out probe and is near-zero for early blocks "
             "by construction:\nit must be read as a depth profile, never averaged over blocks 0-8 "
             "(an earlier draft of this analysis did exactly that and wrongly concluded the metric "
             "was unusable).",
             fontsize=7.8, color="0.35", va="top")
    fig.savefig(OUT / "fig9_training_dynamics.png")
    print("wrote", OUT / "fig9_training_dynamics.png")
    for fam in ["delta_norm_ratio", "blk_act_rms"]:
        A, B = series("ftbqmln", fam), series("ftb4e3", fam)
        pre = [e for e in sorted(set(A) & set(B)) if e <= 199]
        print(f"{fam}: mean ratio ftb4e3/ftbqmln before ep200 = "
              f"{np.mean([B[e]/A[e] for e in pre]):.2f}")
    for a in ARMS:
        p = prof(a, "acc", 149)
        print(f"  acc@149 {a:9} blk9 {p[9]:5.1f}  blk10 {p[10]:5.1f}  blk11 {p[11]:5.1f}")


if __name__ == "__main__":
    main()
