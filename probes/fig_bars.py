"""fig25: the arm comparison as bar charts -- classification on the left, the other tasks on the right.

Plot style follows plots/fig_sink_gate_paper.py (tueplots ICLR bundle with usetex, family serif, no top/right
spines, seaborn tab10 colours, legend above the axes, grid alpha 0.3, pdf + png through one `save`).

Bars are the difference from random, which is the quantity with a meaningful zero: bars of raw accuracy would
need a truncated axis to show 1-5 point differences on a 0-70 scale.  Random's own score is printed under each
dataset so the bars can be read back as absolutes.

The two halves are drawn differently on purpose.  Every classification column is the same metric (top-1 of the
model's own head), so those bars share one axis and their heights are comparable across datasets.  The other
tasks are not: mIoU, delta1 and PCK have different scales and different floors (VOC mIoU floors at 3.0,
NYU delta1 at 39.6), so a shared axis would invite a comparison that means nothing.  Each gets its own panel and
its own axis.  All four are higher-is-better, delta1 included -- it is the share of patches within 1.25x of the
true depth, so the negative bars there mean every arm is worse than random at depth.

Error bars are the standard deviation over seeds and appear only for arms that have more than one seed.

  python -m probes.fig_bars [--arms r,p,ftb3b,ftbanapermb7i,ftbc7lm] [--venue iclr2024] [--no-usetex]
"""
import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tueplots import bundles, figsizes

from probes import registry
from probes.fig_arms import arm_table, per_model_values

SLOT = {"p": 0, "ftb3b": 1, "ftbanapermb7i": 2, "ftbc7lm": 3}        # tab10 slots, as in fig_sink_gate_paper
CLS = [("imagenet_val", "val"), ("imagenet_v2", "V2"), ("imagenet_r", "R"), ("imagenet_sketch", "Sketch"),
       ("imagenet_a", "A"), ("objectnet", "ObjectNet"), ("imagenet_c", "C")]
OTHER = [("voc_miou", "VOC 2012, mIoU"), ("ade_miou", "ADE20K, mIoU"),
         ("nyu_delta1", r"NYU, $\delta_1$"), ("spair_pck", "SPair-71k, PCK@0.1")]


def latest_venue():
    return sorted(b for b in dir(bundles) if b.startswith("iclr"))[-1]


def draw(args):
    venue = args.venue or latest_venue()
    bundle, figsize = getattr(bundles, venue), getattr(figsizes, venue)
    usetex = (not args.no_usetex) and shutil.which("latex") is not None
    A = arm_table(per_model_values(), args.arms.split(","))
    arms = args.arms.split(",")
    base, comps = arms[0], arms[1:]
    colour = {a: sns.color_palette("tab10")[SLOT[a]] for a in comps}

    plt.rcParams.update(bundle(usetex=usetex, rel_width=1.0, nrows=1, ncols=1, family="serif"))
    plt.rcParams.update(figsize(rel_width=1.0, nrows=1, ncols=1, height_to_width_ratio=0.42))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    def bars(ax, keys, width=0.78):
        w = width / len(comps)
        x = np.arange(len(keys))
        for j, a in enumerate(comps):
            off = (j - (len(comps) - 1) / 2) * w
            d = [A[a][k][0] - A[base][k][0] for k in keys]
            e = [A[a][k][1] if A[a][k][2] > 1 else 0.0 for k in keys]
            ax.bar(x + off, d, w * 0.9, color=colour[a], zorder=2,
                   yerr=e, error_kw=dict(ecolor="0.3", lw=0.7, capsize=1.5, zorder=3))
        ax.axhline(0, color="black", linewidth=1, zorder=3)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    fig = plt.figure()
    left, right = fig.subfigures(1, 2, width_ratios=[1.62, 1], wspace=0.02)
    axc = left.subplots()
    keys = [k for k, _ in CLS if all(k in A[a] for a in arms)]
    lab = dict(CLS)
    bars(axc, keys)
    axc.set_xticks(np.arange(len(keys)))
    axc.set_xticklabels([f"{lab[k]}\n" + rf"\textit{{{A[base][k][0]:.1f}}}" for k in keys])
    axc.set_ylabel(r"Top-1 $-$ random")
    shown = [t for t in OTHER if all(t[0] in A[a] for a in arms)]
    axs = right.subplots(2, 2)
    for i, (ax, (k, tlab)) in enumerate(zip(axs.flat, shown)):
        bars(ax, [k], width=0.72)
        ax.set_xticks([])
        ax.set_title(rf"{tlab}" + "\n" + rf"\textit{{random {A[base][k][0]:.1f}}}",
                     loc="left", fontsize=plt.rcParams["axes.labelsize"] - 1.5)
        if i % 2 == 0:
            ax.set_ylabel(r"$-$ random", fontsize=plt.rcParams["axes.labelsize"] - 1)
    for ax in axs.flat[len(shown):]:
        ax.axis("off")
    handles = [Patch(facecolor=colour[a], label=rf"\texttt{{{a}}} ($n={A[a][keys[0]][2]}$)") for a in comps]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)

    out = registry.FIGS
    for ext, dpi in (("pdf", None), ("png", 1500)):
        fig.savefig(out / f"fig25_bars.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"wrote fig25_bars.pdf/.png  [{venue}, usetex={usetex}]  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r,p,ftb3b,ftbanapermb7i,ftbc7lm")
    ap.add_argument("--venue", default=None, help="tueplots bundle name (default: the newest iclr* installed)")
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    draw(args)


if __name__ == "__main__":
    main()
