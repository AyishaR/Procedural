"""Last-3 readout (arm F, fixed_last3) against the reference arms over training: random init, procedural init
(k-Dyck in all 12 blocks) and late-layer rescaling (random init, blocks 9-11 rescaled to match procedural,
arm ftb3b). Reads log.txt only (no GPU); every run must have finished (epoch 299).

Left: test top-1 minus the random seed mean at matched epochs (3-seed arms: seed mean + min-max band).
Right: epoch-299 top-1 per seed. Also writes the plotted numbers as CSV / JSON next to the figure.

    python routing/readout_vs_arms.py --out /work/dlc2workfs3/schrodi-procedural/kempfe/routing/figures
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RESULTS = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/log.txt"
RUNS = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/runs/{name}/s{seed}/log.txt"
GRAY = "#8a8984"
# (display name, short name, colour, linestyle, log paths); hues in the validated categorical order
ARMS = [
    ("Random init", "Random", GRAY, "-", [RESULTS.format(sid=29384839, seed=s) for s in range(3)]),
    ("Last-3 readout", "Last-3 readout", "#2a78d6", "-",
     [RUNS.format(name="readout_F_29784360", seed=s) for s in range(3)]),
    ("Procedural init", "Procedural", "#eb6834", "-", [RESULTS.format(sid=29377576, seed=s) for s in range(3)]),
    ("Late-layer rescaling", "Late-layer rescaling",
     "#1baf7a", "-", [RESULTS.format(sid=sid, seed=s) for s, sid in enumerate((29388202, 29406778, 29406779))]),
]


def load(path):
    """{epoch: test_acc1} of a finished run; last record per epoch wins (resumes re-log)."""
    rec = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            rec[d["epoch"]] = d
    assert max(rec) == 299, f"{path} has not finished (last epoch {max(rec)})"
    return {e: d["test_acc1"] for e, d in rec.items() if d.get("test_acc1") is not None}


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = [[load(p) for p in paths] for *_, paths in ARMS]
    epochs = sorted(set.intersection(*[set(a) for runs in acc for a in runs]))
    rmean = {e: np.mean([a[e] for a in acc[0]]) for e in epochs}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.6), width_ratios=[2.3, 1], layout="constrained")
    rows, ends, clipped = [], [], []
    for (name, short, col, ls, paths), runs in zip(ARMS, acc):
        gaps = np.array([[a[e] - rmean[e] for e in epochs] for a in runs])
        mean = gaps.mean(0)
        seeds = f" ({len(runs)} seeds)" if len(runs) > 1 else " (1 seed)"
        if short == "Random":
            ax.fill_between(epochs, gaps.min(0), gaps.max(0), color=GRAY, alpha=0.25, lw=0,
                            label="Random init: seed range (3 seeds)")
            ax.axhline(0, color=GRAY, lw=1.2)
        else:
            if len(runs) > 1:
                ax.fill_between(epochs, gaps.min(0), gaps.max(0), color=col, alpha=0.12, lw=0)
            ax.plot(epochs, mean, color=col, ls=ls, lw=2.0, label=name + seeds)
            ends.append((mean[-1], short, col))
            lo, hi = int(np.argmin(mean)), int(np.argmax(mean))
            parts = ([f"lowest {mean[lo]:+.1f} (epoch {epochs[lo]})"] if mean[lo] < -1.2 else []) + \
                    ([f"highest {mean[hi]:+.1f} (epoch {epochs[hi]})"] if mean[hi] > 2.6 else [])
            if parts:
                clipped.append(f"{short}: " + ", ".join(parts))
        rows += [dict(arm=short, epoch=e, n_seeds=len(runs), gap_mean=round(float(mean[i]), 4),
                      gap_min=round(float(gaps[:, i].min()), 4), gap_max=round(float(gaps[:, i].max()), 4))
                 for i, e in enumerate(epochs)]
    # direct labels at the right edge, nudged apart
    ends.sort()
    placed = []
    for y, short, col in ends:
        yy = y if not placed or y - placed[-1] > 0.14 else placed[-1] + 0.14
        placed.append(yy)
        ax.annotate(short, (epochs[-1], y), xytext=(epochs[-1] + 6, yy), fontsize=9, color="#222222",
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.set_xlim(0, epochs[-1] + 75)
    ax.set_xticks(range(0, 301, 50))
    ax.set_ylim(-1.2, 2.6)
    ax.text(0.99, 0.02, "early values outside the plotted range (seed means):\n" + "\n".join(clipped),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#52514e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ImageNet top-1 minus random init (points)")
    ax.set_title("Accuracy gain over random init during training", fontsize=11)
    fig.text(0.01, -0.03, "Last-3 readout: the classifier head reads only the summed outputs of layers 10-12. "
             "Late-layer rescaling: random init with layers 10-12 rescaled to match procedural init.",
             fontsize=8.5, color="#52514e")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    finals = []
    for y, ((name, short, col, _, _), runs) in enumerate(zip(ARMS, acc)):
        vals = [a[299] for a in runs]
        bx.scatter(vals, [y] * len(vals), s=50, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        m = float(np.mean(vals))
        if len(vals) > 1:
            bx.plot([m, m], [y - 0.28, y + 0.28], color="#222222", lw=1.4, zorder=4)
        bx.annotate(f"{m:.2f}", (max(vals), y), xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8.5,
                    color="#222222")
        finals.append(dict(arm=short, n_seeds=len(vals), top1_mean=round(m, 3), top1=[round(v, 3) for v in vals]))
    bx.axvline(float(np.mean([a[299] for a in acc[0]])), color=GRAY, lw=1.2, zorder=1)
    bx.set_yticks(range(len(ARMS)))
    bx.set_yticklabels([a[1] for a in ARMS], fontsize=9.5)
    bx.set_ylim(-0.6, len(ARMS) - 0.4)
    bx.set_xlim(right=bx.get_xlim()[1] + 0.25)
    bx.set_xlabel("final ImageNet top-1 (epoch 299)")
    bx.set_title("Final accuracy (dots: seeds, tick: mean)", fontsize=11)
    bx.grid(axis="x", color="#e6e6e3", lw=0.8)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "readout_vs_arms.png", dpi=140, bbox_inches="tight")
    with open(out / "readout_vs_arms_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "readout_vs_arms_final.json").write_text(json.dumps(finals, indent=1))
    print("wrote", out / "readout_vs_arms.png")
    for r in finals:
        print(f"  {r['arm']:28s} {r['top1_mean']:.3f}  {r['top1']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/work/dlc2workfs3/schrodi-procedural/kempfe/routing/figures")
    main(p.parse_args().out)
