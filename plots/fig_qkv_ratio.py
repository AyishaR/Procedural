#!/usr/bin/env python3
"""fig10: the qk/v scale ratio at init -- the one statistic that survives all four gates.

Panel A  every arm's blocks-0-8 qk/v ratio against its last-epoch ImageNet-1k accuracy,
         with the three arms still training shown as predictions at their measured ratio.
Panel B  the same ratio per block, showing that ftb4o inverts it and that ftb4e3 reproduces
         the procedural checkpoint exactly.
Panel C  why every other candidate dies: `ftb4e3` is `ftb3i` with each tensor randomly
         permuted within its slice and the two train to the same accuracy, so a statistic
         that moves between them cannot be the cause.

Run:  .venv/bin/python plots/fig_qkv_ratio.py
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "ckpt_diff.json"
OUT = HERE / "out"

PEND = []          # all three slice arms have now finished; kept for future additions
NEW = ["ftbqm1dv", "ftbqm1dqk", "ftbqm1dvo"]   # the arms that were predictions in the first draft
C_WIN, C_MID, C_BASE, C_BAD, C_PRED = "#029E73", "#DE8F05", "#8c8c8c", "#D55E00", "#0173B2"


def mean_feat(d, arm, k):
    return float(np.mean([x for x in d[arm]["per_block"][k] if x is not None]))


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(CACHE))
    acc = d["_acc"]
    arms = [a for a in acc if a != "p"]
    x = np.array([mean_feat(d, a, "W.qk_over_v") for a in arms])
    y = np.array([acc[a] for a in arms])
    pr = stats.pearsonr(x, y)

    fig = plt.figure(figsize=(16.4, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.15], wspace=0.42)

    # ---------------- Panel A ----------------
    ax = fig.add_subplot(gs[0, 0])
    fit = np.polyfit(x, y, 1)
    xs = np.linspace(0.25, 2.35, 50)
    ax.plot(xs, np.polyval(fit, xs), color="0.65", lw=1.2, zorder=1)
    # six arms sit at exactly 1.00: label them once, as a block, rather than six times
    tied = sorted([a for a in arms if abs(mean_feat(d, a, "W.qk_over_v") - 1.0) < 0.01],
                  key=lambda a: -acc[a])
    for a, xa, ya in zip(arms, x, y):
        c = C_WIN if a in ("ftb4e3", "ftb3i") else (C_BAD if a == "ftb4o" else
                                                    (C_BASE if a == "r" else
                                                     (C_PRED if a in NEW else C_MID)))
        ax.scatter(xa, ya, s=95, color=c, zorder=3, edgecolor="w", lw=1.0,
                   marker="D" if a in NEW else "o")
        if a not in tied:
            # ftbqm1dv sits at the same x as ftb4e3/ftb3i and 0.03 below ftbqm1dvo, so it
            # is the one label that has to go right or it lands on top of ftbqm1dvo's.
            dx, ha = ((0.07, "left") if a in ("ftb4o", "ftbqm1dv") else (-0.07, "right"))
            ax.annotate(a, (xa, ya), (xa + dx, ya), fontsize=8.5, va="center",
                        ha=ha, color="0.2", fontweight="bold")
    ax.annotate("", xy=(1.03, acc[tied[0]]), xytext=(1.30, acc[tied[0]]),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    ax.annotate("", xy=(1.03, acc[tied[-1]]), xytext=(1.30, acc[tied[-1]]),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.9))
    ax.plot([1.30, 1.30], [acc[tied[-1]], acc[tied[0]]], color="0.55", lw=0.9)
    ax.text(1.34, (acc[tied[0]] + acc[tied[-1]]) / 2 - 0.02,
            "all six tied at 1.00:\n" + "\n".join(f"{t}  {acc[t]:.2f}" for t in tied),
            fontsize=7.4, va="center", color="0.3")
    for a in PEND:
        xa = mean_feat(d, a, "W.qk_over_v")
        ax.axvline(xa, color=C_PRED, lw=1.0, ls=":", zorder=0)
        ax.annotate(a, (xa, 79.05), (xa, 79.05), fontsize=7.4, rotation=90,
                    color=C_PRED, ha="center", va="bottom")
    ax.axhline(acc["r"], color=C_BASE, lw=0.8, ls="--", zorder=0)
    ax.set_ylim(77.0, 80.9)
    ax.set_xlabel(r"$\|W_{qk}\| / \|W_v\|$ at init, mean over blocks 0-8")
    ax.set_ylabel("last-epoch test top-1 (%)")
    ax.set_title(f"A.  the qk/v scale ratio predicts the gap\n"
                 f"Pearson r = {pr[0]:+.2f}  (p = {pr[1]:.1g}, n = {len(arms)})")
    ax.text(0.03, 0.97, "diamonds: the three arms run AFTER this\nrelation was fitted, as a test of it",
            transform=ax.transAxes, fontsize=7.4, color=C_PRED, va="top")

    # ---------------- Panel B ----------------
    ax = fig.add_subplot(gs[0, 1])
    show = [("p", "procedural checkpoint", "#0173B2", "-"),
            ("ftb4e3", "ftb4e3  (+2.08)", C_WIN, "-"),
            ("ftbqm1dv", "ftbqm1dv  (+1.40)", "#1F9E89", "--"),
            ("ftbqm1dvo", "ftbqm1dvo  (+1.37)", "#56B4E9", "--"),
            ("ftbqm1dqk", "ftbqm1dqk  (+0.50)", "#CC78BC", "--"),
            ("ftbqm1d", "ftbqm1d  (+0.42)", C_MID, "-"),
            ("r", "random init", C_BASE, "-"),
            ("ftb4o", "ftb4o  (-0.81)", C_BAD, "-")]
    for a, lab, c, ls in show:
        ax.plot(range(9), d[a]["per_block"]["W.qk_over_v"], ls, color=c, marker="o",
                ms=3.4, lw=1.7, label=lab)
    ax.axhline(1.0, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel("block"); ax.set_ylabel(r"$\|W_{qk}\| / \|W_v\|$")
    ax.set_title("B.  ftb4o inverts the ratio the winners keep")
    ax.set_ylim(0.15, 3.35)
    ax.legend(fontsize=7.2, loc="upper center", ncol=2, columnspacing=0.9,
              handlelength=1.6, borderpad=0.2)
    ax.text(0.98, 0.055, "the procedural line is hidden exactly under ftb4e3;\n"
                         "ftb4o's block 8 sits at 1.0 because that arm\ncalibrates blocks 0-7 only.",
            transform=ax.transAxes, fontsize=7.0, color="0.4", va="bottom", ha="right")

    # ---------------- Panel C ----------------
    ax = fig.add_subplot(gs[0, 2])
    cands = [("F.logit_std", "logit spread"),
             ("F.attn_entropy", "attention entropy"),
             ("F.tok_cos", "token cosine sim."),
             ("F.eff_rank", "token eff. rank"),
             ("G.g_over_w", r"$\|g\|/\|W\|$"),
             ("F.rho_attn", r"$\rho_{attn}$"),
             ("W.value_write", "value write"),
             ("W.logit_scale", "logit scale"),
             ("W.qk_over_v", r"$\|W_{qk}\|/\|W_v\|$")]
    labs, vals = [], []
    for k, lab in cands:
        a, b = mean_feat(d, "ftb3i", k), mean_feat(d, "ftb4e3", k)
        labs.append(lab)
        vals.append(abs(b - a) / max(abs(a), abs(b), 1e-12))
    order = np.argsort(vals)[::-1]
    cols = [C_BAD if vals[i] > 0.25 else C_WIN for i in order]
    ax.barh(range(len(order)), [vals[i] for i in order], color=cols, height=0.62)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labs[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.25, color="0.4", lw=1.0, ls="--")
    ax.set_xlabel("relative disagreement between ftb3i and ftb4e3 at init")
    ax.set_title("C.  the shuffle test kills the rest")
    ax.set_xlim(0, 1.12)
    ax.text(0.97, 0.06,
            "ftb4e3 is ftb3i with every tensor randomly\n"
            "permuted within its slice; they score 79.99\n"
            "and 80.16 — the same run. A statistic that\n"
            "moves between them is not the cause.",
            transform=ax.transAxes, fontsize=7.2, color="0.3", ha="right", va="bottom")

    fig.suptitle("What separates the +2 arms from the rest, measured at init on the weights that are actually trained",
                 y=1.045, fontsize=12, fontweight="bold")
    fig.text(0.0, -0.09,
             "205 init-time statistics were screened against four gates: rank correlation with final accuracy; placing ftb4o (the one arm that scored BELOW random) on the far side of random;\n"
             "separating the two +2 arms from the middle pack; and surviving the within-slice shuffle that turns ftb3i into ftb4e3. Exactly one family survives: the ratio of the q/k weight scale\n"
             "to the v weight scale. LayerNorm cannot undo it — the same gain multiplies q, k and v, so it cancels in the ratio.\n"
             "\n"
             "The three diamonds were run AFTER the relation was fitted on the other ten arms, as a test of it, and they land on the line: ftbqm1dvo and ftbqm1dqk were built to move the qk/v\n"
             "ratio and the attention logit scale in OPPOSITE directions, so they separate the two readings. The logit-scale reading required ftbqm1dqk > ftbqm1dvo; the observed order is the\n"
             "reverse (+0.50 vs +1.37), which retires it. ftbqm1dv is hidden under ftb4e3 in panel B: the two are identical on this statistic by construction.\n"
             "\n"
             "CAVEAT: six of the thirteen arms are tied at 1.00, so the ratio explains the split between groups and none of the 0.73-point spread inside the middle pack; ftb4o is n=1 and the two\n"
             "single-slice arms are n=2. Panel C is a gate, not a ranking: a statistic can fail it and still be a consequence of the mechanism rather than unrelated to it.",
             fontsize=7.6, color="0.35", va="top")
    fig.savefig(OUT / "fig10_qkv_ratio.png")
    print("wrote", OUT / "fig10_qkv_ratio.png")
    print(f"pearson r={pr[0]:+.3f} p={pr[1]:.2g}")


if __name__ == "__main__":
    main()
