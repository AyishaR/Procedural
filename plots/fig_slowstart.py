#!/usr/bin/env python3
"""fig11: does a slow-starting arm always catch up?  ftb11s tracks the winners, then stops.

Every arm that ends ahead on this project starts far BEHIND random init and crosses late
(docs 3.10.9.8): ftb4e3 is 9.4 points down at epoch 49 and does not overtake until ~214.  That
makes "it starts slow" useless as an early read -- and makes it tempting to read any slow start
as a winner in progress.  ftb11s is the counter-example: through epoch 74 it is inside the
winners' envelope, and then its recovery simply stops while theirs continues to the last epoch.

Run:  .venv/bin/python plots/fig_slowstart.py
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

R = Path(__file__).resolve().parent.parent / "results" / "imnet_base"
OUT = Path(__file__).resolve().parent / "out"

ARMS = {
    "r":      ("29384839", ["s0", "s1", "s2"]),
    "p":      ("29377576", ["s0", "s1", "s2"]),
    "ftb4e3": ("29451642", ["s0", "s1", "s2"]),
    "ftb3i":  ("29469072", ["s0", "s1", "s2"]),
    "ftb11i": ("29457108", ["s0", "s1", "s2"]),
    "ftb11d": ("29514777", ["s0", "s1", "s2"]),
    "ftb11s": ("29514778", ["s1", "s2"]),
}
STYLE = {
    "p":      ("procedural init (+2.01)",              "#0173B2", "-", 2.0),
    "ftb4e3": ("ftb4e3 (+2.08)",                       "#029E73", "-", 2.0),
    "ftb3i":  ("ftb3i (+1.91)",                        "#56B4E9", "-", 1.4),
    "ftb11i": ("ftb11i — proc block 0 intact (+0.70)", "#DE8F05", "-", 1.8),
    "ftb11d": ("ftb11d — block 0 scaled DOWN to random's rho", "#CA9161", "--", 1.8),
    "ftb11s": ("ftb11s — block 0 scaled UP to proc's rho",     "#D55E00", "-", 2.4),
}


def ev(sid, seeds):
    out = []
    for s in seeds:
        f = R / f"results_IMNET_BASE_{sid}" / s / "log.txt"
        if not f.exists():
            continue
        out.append({r["epoch"]: r["test_acc1"]
                    for r in (json.loads(l) for l in open(f)) if r.get("test_acc1")})
    return out


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "font.family": "DejaVu Sans"})
    OUT.mkdir(parents=True, exist_ok=True)
    D = {a: ev(sid, ss) for a, (sid, ss) in ARMS.items()}
    rb = {e: np.mean([d[e] for d in D["r"] if e in d])
          for e in sorted({e for d in D["r"] for e in d})}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})

    for a, (lab, c, ls, lw) in STYLE.items():
        eps = sorted({e for d in D[a] for e in d} & set(rb))
        y = [np.mean([d[e] for d in D[a] if e in d]) - rb[e] for e in eps]
        for axx in (ax, ax2):
            axx.plot(eps, y, ls, color=c, lw=lw, label=lab if axx is ax else None,
                     zorder=5 if a == "ftb11s" else 3)
    for axx in (ax, ax2):
        axx.axhline(0, color="0.4", lw=1.0)
        axx.grid(alpha=0.25, lw=0.5)
        axx.set_xlabel("epoch")
    ax.set_ylim(-14, 3.2)
    ax.set_ylabel("test top-1 minus random init (pp)")
    ax.set_title("A.  every winner starts far behind — so does ftb11s")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.axvspan(40, 80, color="#029E73", alpha=0.07, lw=0)
    ax.text(60, 2.6, "ftb11s inside the\nwinners' envelope", fontsize=7.4,
            ha="center", va="top", color="#1a7a5a")

    ax2.set_xlim(90, 300); ax2.set_ylim(-4.2, 2.6)
    ax2.set_title("B.  ...but only ftb11s stops recovering")
    ax2.axvline(214, color="#029E73", ls=":", lw=1.1)
    ax2.text(216, -3.9, "ftb4e3 overtakes here", fontsize=7.2, color="#029E73")
    ax2.text(97, 2.45, "ftb11s: flat from ~150, and its last point is its\n"
                       "worst. Recovery over the last 100 epochs:\n"
                       "+0.57/100ep, against +1.25 to +1.82 for the\n"
                       "three arms that end ahead.",
             fontsize=7.4, color="#D55E00", ha="left", va="top")

    fig.suptitle("A slow start does not imply a late win: rho-matching block 0 tracks the winners "
                 "for 75 epochs, then plateaus 2.5 points down",
                 y=1.045, fontsize=11.4, fontweight="bold")
    fig.text(0.0, -0.07,
             "Curves are the mean over 3 seeds (ftb11s: 2 — its third diverged at epoch 37 and was restarted). ftb11d and ftb11s are still training and stop at ~epoch 210-230; nothing here is a\n"
             "final number. ftb3i's spike at epoch 74 is a known loss spike, not a measurement error. ftb11i (proc's block 0 kept intact) is the upper bound for this pair: scaling that block DOWN to\n"
             "random's write magnitude removes the whole +0.70, and scaling a RANDOM block 0 UP to proc's costs 2.5 points — so neither direction of rho reproduces what proc's block-0 weights do.\n"
             "The instability reading does not hold: the two surviving ftb11s seeds have ZERO train-loss spikes (max epoch-to-epoch jump 0.013, against random init's 0.012) and no test dip over 0.53pp.",
             fontsize=7.6, color="0.35", va="top")
    fig.savefig(OUT / "fig11_slowstart.png")
    print("wrote", OUT / "fig11_slowstart.png")


if __name__ == "__main__":
    main()
