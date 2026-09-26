"""The readout-cut arms against the reference arms over training: random (r), procedural (p, k-Dyck
in all 12 blocks) and the two late-block rescaling arms (ftbrho: random, blocks 9-11 upscaled with
rho 1.4; ftb3b: random, blocks 9-11 rho-matched to proc). Reads log.txt only (no GPU); every run
must have finished (epoch 299).

Left panel: test top-1 minus the random seed mean at matched epochs (3-seed arms: seed mean plus
min-max band). Right panel: epoch-299 top-1 per seed. Also writes the plotted numbers as CSV.

    python routing/readout_vs_arms.py --out /work/dlc2workfs3/schrodi-procedural/kempfe/routing/figures
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RESULTS = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/log.txt"
RUNS = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/runs/{name}/s0/log.txt"

# group -> (label, colour, [(run label, log path, linestyle)]). One hue per group, in the validated
# categorical order; within a group, line style separates the members (with c = dashed).
GRAY = "#8a8984"
GROUPS = {
    "r": ("random (r, 3 seeds)", GRAY,
          [(f"r s{s}", RESULTS.format(sid=29384839, seed=s), "-") for s in range(3)]),
    "last3": ("readout: last 3 blocks (F; D = + c)", "#2a78d6",
              [("F fixed_last3", RUNS.format(name="readout_F_29784360"), "-"),
               ("D fixed_emb_last3", RUNS.format(name="readout_D_29768024"), "--")]),
    "p": ("procedural (p, 3 seeds)", "#eb6834",
          [(f"p s{s}", RESULTS.format(sid=29377576, seed=s), "-") for s in range(3)]),
    "rescale": ("late rescaling (ftbrho / ftb3b, 3 seeds each)", "#1baf7a",
                [(f"ftbrho s{s}", RESULTS.format(sid=sid, seed=s), "-")
                 for s, sid in enumerate((29453944, 29461252, 29461253))] +
                [(f"ftb3b s{s}", RESULTS.format(sid=sid, seed=s), "--")
                 for s, sid in enumerate((29388202, 29406778, 29406779))]),
    "last4": ("readout: last 4 blocks (A; G = + c)", "#eda100",
              [("A fixed_last4", RUNS.format(name="readout_A_29725255"), "-"),
               ("G fixed_emb_last4", RUNS.format(name="readout_G_29784361"), "--")]),
}


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

    acc = {g: {lab: load(p) for lab, p, _ in members} for g, (_, _, members) in GROUPS.items()}
    epochs = sorted(set.intersection(*[set(a) for g in acc.values() for a in g.values()]))
    rmean = {e: np.mean([a[e] for a in acc["r"].values()]) for e in epochs}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.6), width_ratios=[2.3, 1], layout="constrained")
    rows = []
    ends = []
    for g, (label, col, members) in GROUPS.items():
        # sub-arms of a group are separated by line style (rescaling: ftbrho / ftb3b, 3 seeds each)
        styles = {}
        for lab, _, ls in members:
            styles.setdefault(ls, []).append(lab)
        for ls, labs in styles.items():
            gaps = np.array([[acc[g][l][e] - rmean[e] for e in epochs] for l in labs])
            mean = gaps.mean(axis=0)
            name = labs[0].split(" s")[0] if len(labs) > 1 else labs[0]
            if g == "r":
                ax.fill_between(epochs, gaps.min(axis=0), gaps.max(axis=0), color=GRAY, alpha=0.25, lw=0,
                                label="random: seed range")
                ax.axhline(0, color=GRAY, lw=1.2)
            else:
                if len(labs) > 1:
                    ax.fill_between(epochs, gaps.min(axis=0), gaps.max(axis=0), color=col, alpha=0.12, lw=0)
                ax.plot(epochs, mean, color=col, lw=1.8, ls=ls,
                        label=f"{name} ({len(labs)} seeds)" if len(labs) > 1 else name)
                ends.append((mean[-1], name, col))
            for i, e in enumerate(epochs):
                rows.append(dict(group=g, arm=name, epoch=e, n_seeds=len(labs),
                                 gap_mean=round(float(mean[i]), 4),
                                 gap_min=round(float(gaps[:, i].min()), 4), gap_max=round(float(gaps[:, i].max()), 4)))
    # direct labels at the right edge (relief for the low-contrast slots), nudged apart
    ends.sort()
    placed = []
    for y, name, col in ends:
        yy = y if not placed or y - placed[-1] > 0.13 else placed[-1] + 0.13
        placed.append(yy)
        ax.annotate(name, (epochs[-1], y), xytext=(epochs[-1] + 6, yy), fontsize=8.5, color="#222222",
                    va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.set_xlim(0, epochs[-1] + 62)
    ax.set_xticks(range(0, 301, 50))
    ax.set_ylim(-1.2, 2.6)
    # the y-range is zoomed on the late phase; say what falls outside it
    clipped = []
    for g in ("p", "rescale"):
        for lab_prefix in sorted({l.split(" s")[0] for l in acc[g]}):
            labs = [l for l in acc[g] if l.startswith(lab_prefix + " s")]
            m_ = [np.mean([acc[g][l][e] for l in labs]) - rmean[e] for e in epochs]
            i_lo, i_hi = int(np.argmin(m_)), int(np.argmax(m_))
            clipped.append(f"{lab_prefix}: min {m_[i_lo]:+.1f} @ ep {epochs[i_lo]}, max {m_[i_hi]:+.1f} @ ep {epochs[i_hi]}")
    ax.text(0.99, 0.02, "outside the axis range (seed means):\n" + "\n".join(clipped), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="#52514e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test top-1 minus random seed mean (points)")
    ax.set_title("Accuracy gap to random over training (lines: seed mean; bands: seed range)", fontsize=11)
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # right: final top-1 per seed
    order = [("random r", "r", None), ("readout A (last 4)", "last4", "A fixed_last4"),
             ("readout G (last 4 + c)", "last4", "G fixed_emb_last4"),
             ("readout F (last 3)", "last3", "F fixed_last3"), ("readout D (last 3 + c)", "last3", "D fixed_emb_last3"),
             ("rescale ftbrho", "rescale", "ftbrho"), ("rescale ftb3b", "rescale", "ftb3b"),
             ("procedural p", "p", None)]
    finals = []
    for y, (lab, g, key) in enumerate(order):
        col = GROUPS[g][1]
        vals = [a[299] for l, a in acc[g].items() if key is None or l.startswith(key)]
        bx.scatter(vals, [y] * len(vals), s=46, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        m = float(np.mean(vals))
        if len(vals) > 1:
            bx.plot([m, m], [y - 0.28, y + 0.28], color="#222222", lw=1.4, zorder=4)
        bx.annotate(f"{m:.2f}", (m, y), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8,
                    color="#222222")
        finals.append(dict(arm=lab, n_seeds=len(vals), top1_mean=round(m, 3), top1=[round(v, 3) for v in vals]))
    rm = np.mean([a[299] for a in acc["r"].values()])
    bx.axvline(rm, color=GRAY, lw=1.2, zorder=1)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([o[0] for o in order], fontsize=9)
    bx.set_ylim(-0.6, len(order) - 0.4)
    bx.set_xlabel("epoch-299 test top-1 (%)")
    bx.set_title("Final accuracy (dots: seeds; tick: mean)", fontsize=11)
    bx.grid(axis="x", color="#e6e6e3", lw=0.8)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "readout_vs_arms.png", dpi=140)
    with open(out / "readout_vs_arms_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "readout_vs_arms_final.json").write_text(json.dumps(finals, indent=1))
    print("wrote", out / "readout_vs_arms.png")
    for r in finals:
        print(f"  {r['arm']:26s} {r['top1_mean']:.3f}  {r['top1']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/work/dlc2workfs3/schrodi-procedural/kempfe/routing/figures")
    main(p.parse_args().out)
