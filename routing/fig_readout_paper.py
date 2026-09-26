"""Paper figures for the last-3 readout: ImageNet validation accuracy over training, and the logit lens over layers.
Plot style follows plots/fig_sink_gate_paper.py and visualise/report_plots_clean.ipynb (origin/main): tueplots iclr2024
bundle with usetex, no top/right spines, seaborn tab10 colours in the same order (Random init 0, k-Dyck-D4 1,
k-Dyck-Shuffled-D98 2), 1 pt lines with 1.5 pt markers, grid alpha 0.3, legend above the axes, PDF + PNG preview.

Series (lines are means over the available seeds; no bands):
  Random init            r      results_IMNET_BASE_29384839 s0-2
  k-Dyck-D4              p      results_IMNET_BASE_29377576 s0-2   (procedural warm-up, all 12 blocks)
  k-Dyck-Shuffled-D98    pd98   results_IMNET_BASE_29547831 s0     (only in the *_ksd98 versions)
  Last-3 readout         F      kempfe/routing/runs/readout_F_29784360 s0 (--readout_routing fixed_last3)

Validation accuracy: test_acc1 from each run's log.txt (evaluated every 5 epochs on the 50k ImageNet val set).
Logit lens: top-1 of the model's own final LayerNorm + head applied to the CLS token after each layer, full ImageNet val;
cached probes/lens.py results for r / p / pd98, routing/readout_lens.py for F (same protocol; r_s0 reproduced to 0.04).
For the last-3 readout the value at layer 12 is the full residual stream, which its head never reads.

    python routing/fig_readout_paper.py [--rel_width 0.5]
output: kempfe/routing/figures/paper/{val_acc,logit_lens}{,_ksd98}.{pdf,png}
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

RESULTS = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/log.txt"
LENS = "/work/dlc2workfs3/schrodi-procedural/kempfe/probes/results/lens/{name}.json"
F_LOG = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/runs/readout_F_29784360/s0/log.txt"
F_LENS = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/readout_lens/lens_F.json"
OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/figures/paper"
LINKS = "/work/dlclarge1/kempfe-mats/Procedural/figures"      # every figure is symlinked here

# key, label, tab10 slot, log files, lens files
SERIES = [
    ("random", "Random init", 0, [RESULTS.format(sid=29384839, seed=s) for s in range(3)],
     [LENS.format(name=f"r_s{s}") for s in range(3)]),
    ("kdyck", "k-Dyck-D4", 1, [RESULTS.format(sid=29377576, seed=s) for s in range(3)],
     [LENS.format(name=f"p_s{s}") for s in range(3)]),
    ("ksd", "k-Dyck-Shuffled-D98", 2, [RESULTS.format(sid=29547831, seed=0)], [LENS.format(name="pd98_s0")]),
    ("last3", "Last-3 readout", 3, [F_LOG], [F_LENS]),
]


def val_curve(path):
    """{epoch: test_acc1} of a finished run (last record per epoch wins: resumes re-log)."""
    rec = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            rec[d["epoch"]] = d
    assert max(rec) == 299, f"{path} has not finished"
    return {e: d["test_acc1"] for e, d in rec.items() if d.get("test_acc1") is not None}


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from tueplots import bundles

    usetex = shutil.which("latex") is not None
    plt.rcParams.update(bundles.iclr2024(usetex=usetex, rel_width=a.rel_width, nrows=1, ncols=1, family="serif"))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    tab10 = sns.color_palette("tab10")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pct = r"\%" if usetex else "%"

    def save(fig, name):
        for extension, dpi in (("pdf", None), ("png", 600)):
            path = out / f"{name}.{extension}"
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            if a.link:
                link = Path(LINKS) / f"readout_paper_{name}.{extension}"
                if link.is_symlink() or not link.exists():
                    link.unlink(missing_ok=True)
                    link.symlink_to(path)
        plt.close(fig)
        print(f"wrote {out}/{name}.pdf ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in)")

    for with_ksd in (False, True):
        series = [s for s in SERIES if with_ksd or s[0] != "ksd"]
        suffix = "_ksd98" if with_ksd else ""
        handles = [Line2D([], [], color=tab10[slot], marker="o", markersize=1.5, linewidth=1, label=label)
                   for _, label, slot, _, _ in series]
        ncol = len(handles) if len(handles) <= 3 else 2

        # validation accuracy over training
        fig, ax = plt.subplots()
        for _, label, slot, logs, _ in series:
            curves = [val_curve(p) for p in logs]
            epochs = sorted(set.intersection(*[set(c) for c in curves]))
            mean = np.mean([[c[e] for e in epochs] for c in curves], axis=0)
            ax.plot(np.array(epochs) + 1, mean, linewidth=1, color=tab10[slot], zorder=2)   # 60 evals: no markers
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Validation accuracy ({pct})")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 85)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.grid(True, alpha=0.3)
        # the final accuracies differ by 1-2 points on a 0-80 axis: zoom on the last 100 epochs
        ins = ax.inset_axes([0.45, 0.24, 0.52, 0.42])
        for _, label, slot, logs, _ in series:
            curves = [val_curve(p) for p in logs]
            epochs = [e for e in sorted(set.intersection(*[set(c) for c in curves])) if e >= 199]
            mean = np.mean([[c[e] for e in epochs] for c in curves], axis=0)
            ins.plot(np.array(epochs) + 1, mean, linewidth=0.8, color=tab10[slot])
        ins.set_xlim(200, 300)
        ins.set_ylim(a.inset_ylim)
        ins.tick_params(labelsize=plt.rcParams["xtick.labelsize"] * 0.85, length=2)
        ins.grid(True, alpha=0.3)
        for s in ("top", "right"):
            ins.spines[s].set_visible(True)
        fig.legend(handles=[Line2D([], [], color=h.get_color(), linewidth=1, label=h.get_label()) for h in handles],
                   loc="outside upper center", ncol=ncol, frameon=False)
        save(fig, "val_acc" + suffix)

        # logit lens over layers
        fig, ax = plt.subplots()
        layers = np.arange(1, 13)                   # layer 0 (the embedding, before any block) is not shown
        for _, label, slot, _, lens in series:
            top1 = np.mean([json.load(open(p))["logit_lens"]["top1"] for p in lens], axis=0)[1:]
            ax.plot(layers, top1, marker="o", markersize=1.5, linewidth=1, color=tab10[slot], zorder=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Logit lens accuracy ({pct})")
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks(layers[::2] if a.rel_width < 0.45 else layers)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.legend(handles=handles, loc="outside upper center", ncol=ncol, frameon=False)
        save(fig, "logit_lens" + suffix)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rel_width", type=float, default=0.5, help="figure width as a fraction of the ICLR line width")
    p.add_argument("--inset_ylim", type=float, nargs=2, default=(75.0, 81.0))
    p.add_argument("--out", default=OUT)
    p.add_argument("--no_link", dest="link", action="store_false", help="do not symlink into the repo's figures/")
    main(p.parse_args())
