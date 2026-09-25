"""fig25: the arm comparison as bar charts -- classification on the left, the other tasks on the right.

One text-width figure, styled with tueplots' newest ICLR bundle and the Paul Tol bright palette.

Bars are the difference from random, because that is the quantity with a meaningful zero: random is the
baseline, so a zero-based bar is honest, whereas bars of raw accuracy would need a truncated axis to show
1-5 point differences on a 0-70 scale.  Random's absolute score is printed under each task.

The two halves are drawn differently on purpose.  Every classification column is the same metric (top-1
accuracy of the model's own head), so those bars share one axis and their heights are comparable across
datasets.  The other tasks are not: mIoU, delta1 and PCK have different scales and different floors, so a
shared axis would invite a comparison that does not mean anything.  They get one small panel each, with its
own axis and its metric named, so bars are compared within a task only.

  python -m probes.fig_bars [--arms r,p,ftb3b,ftbanapermb7i,ftbc7lm]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots.constants.color import palettes

from probes import registry
from probes.fig_arms import arm_table, per_model_values

TOL = ["#" + str(c) for c in palettes.paultol_bright]          # blue, red, green, yellow, cyan, purple, grey
BAR_COLOR = {"p": TOL[0], "ftb3b": TOL[1], "ftbanapermb7i": TOL[2], "ftbc7lm": TOL[5]}

CLS = [("imagenet_val", "val"), ("imagenet_v2", "V2"), ("imagenet_r", "R"), ("imagenet_sketch", "Sketch"),
       ("imagenet_a", "A"), ("objectnet", "ObjectNet"), ("imagenet_c", "C")]
OTHER = [("voc_miou", "VOC 2012", "mIoU"), ("ade_miou", "ADE20K", "mIoU"),
         ("nyu_delta1", "NYU depth", r"$\delta_1$"), ("spair_pck", "SPair-71k", "PCK@0.1")]


def latest_iclr(**kw):
    names = sorted(b for b in dir(bundles) if b.startswith("iclr"))
    return getattr(bundles, names[-1])(**kw), names[-1]


def bars(ax, A, arms, keys, small=False):
    """Grouped bars of (arm - random) for `keys`, with random's +/-1 seed sd shaded."""
    base = arms[0]
    comps = arms[1:]
    x = np.arange(len(keys))
    w = 0.78 / len(comps)
    for i, k in enumerate(keys):
        sd = A[base][k][1]
        ax.add_patch(plt.Rectangle((x[i] - 0.47, -sd), 0.94, 2 * sd, color="0.85", lw=0, zorder=0))
    for j, a in enumerate(comps):
        off = (j - (len(comps) - 1) / 2) * w
        d = [A[a][k][0] - A[base][k][0] for k in keys]
        e = [A[a][k][1] if A[a][k][2] > 1 else 0.0 for k in keys]
        ax.bar(x + off, d, w * 0.92, color=BAR_COLOR[a], zorder=2,
               label=rf"\texttt{{{a}}} ($n={A[a][keys[0]][2]}$)",
               yerr=e, error_kw=dict(ecolor="0.25", lw=0.7, capsize=1.4, zorder=3))
    ax.axhline(0, color="black", lw=0.9, zorder=4)
    ax.set_xlim(-0.62, len(keys) - 0.38)
    ax.grid(alpha=0.3, axis="y", lw=0.4)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    ax.set_axisbelow(True)
    if small:
        ax.tick_params(axis="both", labelsize=6, length=2, pad=1.5)


def build(usetex, arms):
    """One text-width figure.  The ICLR bundle turns constrained layout on, so spacing is left to it:
    subfigure suptitles and an `outside` figure legend are the only placement hints given."""
    rc, name = latest_iclr(usetex=usetex, rel_width=1.0)
    A = arm_table(per_model_values(), arms)
    base = arms[0]
    with plt.rc_context(rc):
        W, H = rc["figure.figsize"]
        fig = plt.figure(figsize=(W, H * 0.95))
        left, right = fig.subfigures(1, 2, width_ratios=[1.62, 1], wspace=0.01)
        # ---- left: classification, one shared axis (every column is the same metric)
        axc = left.subplots()
        keys = [k for k, _ in CLS if all(k in A[a] for a in arms)]
        lab = dict(CLS)
        bars(axc, A, arms, keys)
        axc.set_xticks(np.arange(len(keys)))
        axc.set_xticklabels([f"{lab[k]}\n" + rf"\textit{{{A[base][k][0]:.1f}}}" for k in keys], fontsize=6.4)
        axc.set_ylabel(r"top-1 $-$ random (points)", fontsize=7.5)
        axc.tick_params(axis="y", labelsize=6.5)
        handles, labels = axc.get_legend_handles_labels()
        # ---- right: one small panel per task, each on its own axis and in its own metric
        shown = [t for t in OTHER if all(t[0] in A[a] for a in arms)]
        axs = right.subplots(2, 2)
        for ax, (k, tlab, unit) in zip(axs.flat, shown):
            bars(ax, A, arms, [k], small=True)
            ax.set_xticks([])
            ax.set_title(rf"{tlab}" + "\n" + rf"\textit{{{unit}, rand.\ {A[base][k][0]:.1f}}}", fontsize=5.8, pad=2)
        for ax in axs.flat[len(shown):]:
            ax.axis("off")
        fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False,
                   fontsize=6.4, handlelength=0.9, columnspacing=1.4, handletextpad=0.4)
        fig.savefig(registry.FIGS / "fig25_bars.png", dpi=400)
        fig.savefig(registry.FIGS / "fig25_bars.pdf")
        plt.close(fig)
    print(f"wrote fig25_bars.png/.pdf  [{name}, usetex={usetex}, width={W:.2f}in = textwidth]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r,p,ftb3b,ftbanapermb7i,ftbc7lm")
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()
    registry.ensure_dirs()
    arms = args.arms.split(",")
    try:
        build(not args.no_tex, arms)
    except Exception as e:
        print(f"[fig_bars] LaTeX rendering failed ({type(e).__name__}: {e}); falling back to usetex=False")
        build(False, arms)
    print(f"-> {registry.FIGS}")


if __name__ == "__main__":
    main()
