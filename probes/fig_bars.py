"""fig25 / fig26: the arm comparison as bar charts -- classification (main text), the other tasks (appendix).

Plot style follows plots/fig_sink_gate_paper.py (tueplots ICLR bundle with usetex, family serif, no top/right
spines, seaborn tab10 colours, legend above the axes, grid alpha 0.3, pdf + png through one `save`).

Bars are the difference from random, which is the quantity with a meaningful zero: bars of raw accuracy would
need a truncated axis to show 1-5 point differences on a 0-70 scale.  Random's own score is on the top axis of
the classification figure and in the panel titles of the appendix one, so the bars read back as absolutes.

Two figures, because the two halves cannot share an axis.  Every classification column is the same metric
(top-1 of the model's own head), so those bars belong on one axis and their heights are comparable across
datasets.  The other tasks are not: mIoU, delta1 and PCK have different scales and different floors (VOC mIoU
floors at 3.0, NYU delta1 at 39.6), so each gets its own panel and its own axis.  All four are
higher-is-better, delta1 included -- it is the share of patches within 1.25x of the true depth, so the negative
bars there mean every arm is worse than random at depth.

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

# arm -> (tab10 slot, label in the paper's terms)
ARM = {"p":             (0, "k-Dyck D4"),
       "ftb3b":         (1, "Random init w/ PW residual write ratios"),
       "ftbanapermb7i": (2, "Random init w/ PW early-layer targets"),
       "ftbc7lm":       (3, "Combined")}
CLS = [("imagenet_val", "IN-val"), ("imagenet_v2", "IN-V2"), ("imagenet_r", "IN-R"),
       ("imagenet_sketch", "IN-Sketch"), ("imagenet_a", "IN-A"), ("objectnet", "ObjectNet"),
       ("imagenet_c", "IN-C")]
# the bars are differences of percentages, so the unit is percentage points; "vs." keeps "over" from reading
# as a ratio.  One line to change if a different wording is wanted.
YLABEL = r"$\Delta$ Top-1 vs.\ random init (pp)"
OTHER = [("voc_miou", "VOC 2012", "mIoU"), ("ade_miou", "ADE20K", "mIoU"),
         ("nyu_delta1", "NYU", r"$\delta_1$"), ("spair_pck", "SPair-71k", "PCK@0.1")]


def latest_venue():
    return sorted(b for b in dir(bundles) if b.startswith("iclr"))[-1]


def draw(args):
    venue = args.venue or latest_venue()
    bundle, figsize = getattr(bundles, venue), getattr(figsizes, venue)
    usetex = (not args.no_usetex) and shutil.which("latex") is not None
    arms = args.arms.split(",")
    base, comps = arms[0], arms[1:]
    A = arm_table(per_model_values(), arms)
    tab10 = sns.color_palette("tab10")
    colour = {a: tab10[ARM[a][0]] for a in comps}
    pct = r"\%" if usetex else "%"

    def style(rel_width, nrows, ncols, ratio):
        plt.rcParams.update(bundle(usetex=usetex, rel_width=rel_width, nrows=nrows, ncols=ncols, family="serif"))
        plt.rcParams.update(figsize(rel_width=rel_width, nrows=nrows, ncols=ncols, height_to_width_ratio=ratio))
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    def bars(ax, keys, width=0.78):
        w = width / len(comps)
        x = np.arange(len(keys))
        for j, a in enumerate(comps):
            off = (j - (len(comps) - 1) / 2) * w
            ax.bar(x + off, [A[a][k][0] - A[base][k][0] for k in keys], w * 0.9, color=colour[a], zorder=2)
        ax.axhline(0, color="black", linewidth=1, zorder=3)
        ax.set_xlim(-0.6, len(keys) - 0.4)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=colour[a], label=ARM[a][1]) for a in comps]

    def save(fig, name, ncol):
        fig.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False)
        for ext, dpi in (("pdf", None), ("png", 1500)):
            fig.savefig(registry.FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"wrote {name}.pdf/.png ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in before the legend)")

    # ------------------------------------------------------------------ main text: classification, full width
    style(1.0, 1, 1, 0.36)
    keys = [k for k, _ in CLS if all(k in A[a] for a in arms)]
    lab = dict(CLS)
    fig, ax = plt.subplots()
    bars(ax, keys)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([lab[k] for k in keys])
    ax.set_ylabel(YLABEL, fontsize=plt.rcParams["axes.labelsize"] + 1)
    ax.tick_params(axis="y", labelsize=plt.rcParams["ytick.labelsize"] + 1)
    # random's own score as a row inside the axes: an extra axis on top costs height the figure does not have
    hi = max(A[a][k][0] - A[base][k][0] for a in comps for k in keys)
    ax.set_ylim(top=hi * 1.26)
    row = hi * 1.15
    ax.set_xlim(-1.02, len(keys) - 0.42)
    ax.text(-0.70, row, r"\textit{random init}", ha="right", va="center",
            fontsize=plt.rcParams["xtick.labelsize"] - 0.5, color="0.25")
    for i, k in enumerate(keys):
        ax.text(i, row, rf"\textit{{{A[base][k][0]:.1f}}}", ha="center", va="center",
                fontsize=plt.rcParams["xtick.labelsize"] - 0.5, color="0.25")
    save(fig, "fig25_bars_classification", len(handles))

    # ------------------------------------------------------------------ appendix: the other tasks, full width
    shown = [t for t in OTHER if all(t[0] in A[a] for a in arms)]
    style(1.0, 1, len(shown), 1.05)
    fig, axs = plt.subplots(1, len(shown))
    for ax, (k, tlab, unit) in zip(np.atleast_1d(axs), shown):
        bars(ax, [k], width=0.72)
        ax.set_xticks([])
        ax.set_title(rf"{tlab}, {unit}" + "\n" + rf"\textit{{random {A[base][k][0]:.1f}}}",
                     loc="center", fontsize=plt.rcParams["axes.labelsize"])
    fig.supylabel(r"$\Delta$ vs.\ random init", fontsize=plt.rcParams["axes.labelsize"] + 1)
    save(fig, "fig26_bars_other_tasks", len(handles))


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
