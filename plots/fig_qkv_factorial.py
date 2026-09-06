#!/usr/bin/env python3
"""fig12: the qkv slicing experiment is a 2x2 factorial, and it comes out clean.

The four arms were queued one at a time to answer different questions, but together they are a
complete 2x2: {qk slice pooled, matched} x {v slice pooled, matched}.  Reading them that way is
much stronger than reading them as four points on a ladder, because it separates a main effect
from an interaction -- and there is no interaction.

Panel B answers the obvious objection: why would matching one slice matter so much more than the
other?  Because the fused qkv tensor is 2/3 q and k, and proc's q and k are the WIDE rows.  Pooling
q, k and v into one multiset therefore hands every row roughly the q/k distribution -- which leaves
q and k nearly correct and inflates v by 76%.  Matching qk separately is close to a no-op; matching
v separately is the entire intervention.

Run:  .venv/bin/python plots/fig_qkv_factorial.py
"""
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "imnet_base"
OUT = Path(__file__).resolve().parent / "out"
CACHE = Path(__file__).resolve().parent / "cache" / "ckpt_diff.json"

SID = {"r": "29384839", "ftbqm1d": "29501773", "ftbqm1dqk": "29511670",
       "ftbqm1dvo": "29511673", "ftbqm1dv": "29507368"}
CELL = {("pooled", "pooled"): "ftbqm1d", ("matched", "pooled"): "ftbqm1dqk",
        ("pooled", "matched"): "ftbqm1dvo", ("matched", "matched"): "ftbqm1dv"}


def acc(sid):
    o = []
    for s in ["s0", "s1", "s2"]:
        f = R / f"results_IMNET_BASE_{sid}" / s / "log.txt"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        if len(rows) < 300:          # a live seed's last epoch is not a result
            continue
        a = [r.get("test_acc1") for r in rows if r.get("test_acc1")]
        if a:
            o.append(a[-1])
    return np.array(o)


def main():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "legend.frameon": False, "font.family": "DejaVu Sans"})
    OUT.mkdir(parents=True, exist_ok=True)
    A = {k: acc(v) for k, v in SID.items()}
    rb = A["r"].mean()
    M = lambda qk, v: A[CELL[(qk, v)]].mean() - rb
    S = lambda qk, v: (A[CELL[(qk, v)]].std(ddof=1) if len(A[CELL[(qk, v)]]) > 1 else 0.0)
    N = lambda qk, v: len(A[CELL[(qk, v)]])

    mv = ((M("pooled", "matched") - M("pooled", "pooled")) +
          (M("matched", "matched") - M("matched", "pooled"))) / 2
    mq = ((M("matched", "pooled") - M("pooled", "pooled")) +
          (M("matched", "matched") - M("pooled", "matched"))) / 2
    ix = ((M("matched", "matched") - M("pooled", "matched")) -
          (M("matched", "pooled") - M("pooled", "pooled")))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---------------- Panel A: the 2x2 ----------------
    CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        "g", ["#F2F2F2", "#B7E4D0", "#029E73"])
    for i, v in enumerate(["pooled", "matched"]):
        for j, qk in enumerate(["pooled", "matched"]):
            d, sd, n = M(qk, v), S(qk, v), N(qk, v)
            ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                                       color=CMAP(min(d / 1.6, 1.0)), zorder=1))
            ax.text(j, i + 0.13, f"{d:+.2f}", ha="center", va="center", fontsize=19,
                    fontweight="bold", color="#0d3d2c" if d > 0.9 else "#333", zorder=3)
            ax.text(j, i - 0.16, f"{CELL[(qk, v)]}\n{'±%.2f  ' % sd if n > 1 else ''}n={n}",
                    ha="center", va="center", fontsize=7.6, color="#3a3a3a", zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["qk slice\nPOOLED", "qk slice\nMATCHED"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["v slice\nPOOLED", "v slice\nMATCHED"], fontsize=9)
    ax.set_xlim(-0.62, 1.95); ax.set_ylim(-0.62, 1.62)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # main effects in the margin
    ax.annotate("", xy=(1.62, 1.0), xytext=(1.62, 0.0),
                arrowprops=dict(arrowstyle="->", color="#029E73", lw=2.2))
    ax.text(1.70, 0.5, f"matching v\n{mv:+.2f}", color="#029E73", fontsize=10.5,
            fontweight="bold", va="center")
    ax.annotate("", xy=(1.0, -0.56), xytext=(0.0, -0.56),
                arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=2.2))
    ax.text(0.5, -0.50, f"matching qk   {mq:+.2f}", color="#7a7a7a", fontsize=9.5,
            fontweight="bold", ha="center", va="bottom")
    ax.set_title("A.  a 2x2 factorial: matching the v slice is the whole effect\n"
                 f"main effect v {mv:+.2f}   |   main effect qk {mq:+.2f}   |   "
                 f"interaction {ix:+.2f}")

    # ---------------- Panel B: why ----------------
    d = json.load(open(CACHE))
    f = lambda a, k: float(np.mean(d[a]["per_block"][k]))
    labs = ["q", "k", "v"]
    x = np.arange(3); w = 0.26
    for off, (arm, lab, c) in zip([-w, 0, w],
                                  [("r", "random init", "#8c8c8c"),
                                   ("ftbqm1d", "pooled qkv (what the top row does)", "#DE8F05"),
                                   ("p", "procedural checkpoint", "#0173B2")]):
        ax2.bar(x + off, [f(arm, f"W.{n}_norm") for n in labs], w, color=c, label=lab)
    for i, n in enumerate(labs):
        r = f("ftbqm1d", f"W.{n}_norm") / f("p", f"W.{n}_norm")
        ax2.text(i, max(f("ftbqm1d", f"W.{n}_norm"), f("p", f"W.{n}_norm")) + 2.0,
                 f"pooled is\n{r:.0%} of proc", ha="center", fontsize=8,
                 color="#B00020" if r > 1.3 else "#2a7a5a", fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels([f"$W_{n}$" for n in labs], fontsize=12)
    ax2.set_ylabel(r"$\|W\|_F$ at init, mean over blocks 0-8")
    ax2.set_ylim(0, 74)
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    ax2.set_title("B.  why one slice and not the other")

    fig.suptitle("The qkv slicing result: proc's transferable content in blocks 0-8 is a NARROW v, "
                 "not a large q/k",
                 y=1.04, fontsize=12, fontweight="bold")
    fig.text(0.0, -0.07,
             "All four arms are otherwise identical: a random model whose blocks 0-8 receive proc's exact per-tensor value multisets by rank map, plus proc's eight 1-D parameters in a\n"
             "uniformly random permutation. They differ ONLY in whether the fused attn.qkv is matched as one pooled multiset or as its [0:2e] and [2e:3e] slices independently.\n"
             "n = 3 for the two diagonal cells and n = 2 for the off-diagonal ones (their third seed is still training and is excluded rather than read early), so the main effect of v is\n"
             "solid and the ZERO interaction is the weaker claim. The v main effect is +0.93 against a seed s.d. of 0.10-0.40; the qk main effect of +0.05 is far inside noise.\n"
             "\n"
             "WHY ONE SLICE AND NOT THE OTHER. The fused qkv tensor is 2/3 q and k, and proc's q and k are its WIDE rows. Pooling all three into one value multiset therefore hands every\n"
             "row roughly the q/k distribution: q and k come out nearly right (92% and 82% of proc's), and v comes out 76% too wide. 'Matching qk' is close to a no-op by construction; 'matching v'\n"
             "is the entire intervention. The factorial is not telling us that q and k are unimportant in general — only that the pooled baseline already had them almost right.\n"
             "\n"
             "This also reconciles the rho failures. rho is a FORWARD-PASS quantity, so it can be hit with the wrong weights: ftb4o matches proc's rho_attn (0.462 vs 0.471) while its\n"
             "weight-space write is 7.4x proc's, because its LayerNorm gains are 1.0 where proc's are 0.38. Matching rho and matching the qk/v configuration are not the same target,\n"
             "and only the latter transfers.",
             fontsize=7.6, color="0.35", va="top")
    fig.savefig(OUT / "fig12_qkv_factorial.png")
    print("wrote", OUT / "fig12_qkv_factorial.png")
    print(f"main v {mv:+.3f}  main qk {mq:+.3f}  interaction {ix:+.3f}")


if __name__ == "__main__":
    main()
