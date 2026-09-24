"""Results of the progressive-unfreezing arms on ViT-S / ImageNet-100 (docs/unfreeze_vit_handoff.md,
"Logging and comparison"). Reads log.txt per run; no GPU.

Primary metric: the paired delta vs the same-seed base (schrodi's i100_playground r_random/s<seed>)
at epoch 299, top-1 and val loss, next to the 6-run base spread (r_random + r_random_k4, seeds
0-2, identical recipe). Only runs whose log reaches epoch 299 enter the table; running arms are
drawn in the curves (marked "running") but never tabulated.

Figure: test top-1 / test loss per epoch and the gap to the base-seed mean with the per-epoch
seed band, unfreeze epochs marked per arm.

    python unfreeze/curves.py                    # all arms found under kempfe/unfreeze/runs
    python unfreeze/curves.py --seed 1 --arms u4f-one0.2
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import unfreeze as uf  # noqa: E402

OUT_ROOT = Path("/work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze")
BASE_ROOT = Path("/work/dlc2workfs3/schrodi-procedural/results/i100_playground")
BASE_RUNS = [f"{r}/s{s}" for r in ("r_random", "r_random_k4") for s in range(3)]
ARMS = ["u4f-lin0.2", "u4b-lin0.2", "u4f-one0.2", "u4b-one0.2"]
COLORS = {"u4f-lin0.2": "C0", "u4b-lin0.2": "C3", "u4f-one0.2": "C2", "u4b-one0.2": "C1"}
LAST, T, NITER = 299, 74100, 247


def read_log(p):
    """{epoch: record}, last record per epoch wins (a requeue re-logs its resume epoch)."""
    out = {}
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            out[d["epoch"]] = d
    return out


def series(log, key):
    eps = sorted(log)
    return np.array(eps), np.array([log[e][key] for e in eps])


def unfreeze_epochs(arm):
    s = uf.parse_arm(arm)
    return [u / NITER for u in uf.unfreeze_iters(s["sched"], s["chunks"], s["end"], T) if u > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default=str(OUT_ROOT / "figures"))
    ap.add_argument("--prefix", default="unfreeze")
    args = ap.parse_args()

    base = {r: read_log(BASE_ROOT / r / "log.txt") for r in BASE_RUNS}
    assert all(LAST in b for b in base.values()), "a base run did not reach epoch 299"
    pair = base[f"r_random/s{args.seed}"]
    fin = {k: np.array([b[LAST][k] for b in base.values()]) for k in ("test_acc1", "test_loss")}
    print(f"base (6 runs) epoch {LAST}: top-1 {fin['test_acc1'].mean():.2f} +- "
          f"{fin['test_acc1'].std(ddof=1):.2f}, loss {fin['test_loss'].mean():.4f} +- "
          f"{fin['test_loss'].std(ddof=1):.4f}; paired base r_random/s{args.seed}: "
          f"{pair[LAST]['test_acc1']:.2f} / {pair[LAST]['test_loss']:.4f}")

    runs, table = {}, []
    for arm in args.arms:
        p = OUT_ROOT / "runs" / arm / f"s{args.seed}" / "log.txt"
        if not p.exists():
            print(f"{arm}: no log yet")
            continue
        log = read_log(p)
        done = LAST in log
        runs[arm] = (log, done)
        if not done:
            print(f"{arm}: running, last epoch {max(log)} (not tabulated)")
            continue
        a, lo = log[LAST]["test_acc1"], log[LAST]["test_loss"]
        row = dict(arm=arm, top1=a, loss=lo, d_top1_paired=a - pair[LAST]["test_acc1"],
                   d_loss_paired=lo - pair[LAST]["test_loss"],
                   d_top1_vs_mean=a - fin["test_acc1"].mean(),
                   z_top1=(a - fin["test_acc1"].mean()) / fin["test_acc1"].std(ddof=1),
                   unfreeze_epochs=unfreeze_epochs(arm))
        table.append(row)
    if table:
        print(f"\n{'arm':12s} {'top-1':>7s} {'d paired':>9s} {'d mean':>7s} {'z':>6s} "
              f"{'loss':>7s} {'d loss':>8s}  unfreeze epochs")
        for r in table:
            print(f"{r['arm']:12s} {r['top1']:7.2f} {r['d_top1_paired']:+9.2f} "
                  f"{r['d_top1_vs_mean']:+7.2f} {r['z_top1']:+6.1f} {r['loss']:7.4f} "
                  f"{r['d_loss_paired']:+8.4f}  {r['unfreeze_epochs']}")
    os.makedirs(args.out, exist_ok=True)
    with open(Path(args.out) / f"{args.prefix}_s{args.seed}_table.json", "w") as f:
        json.dump(dict(seed=args.seed, base_top1=fin["test_acc1"].tolist(),
                       base_loss=fin["test_loss"].tolist(), rows=table), f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained", sharex=True)
    for j, key in enumerate(("test_acc1", "test_loss")):
        eps = sorted(set.intersection(*(set(b) for b in base.values())))
        B = np.array([[b[e][key] for e in eps] for b in base.values()])
        mu, sd = B.mean(0), B.std(0, ddof=1)
        ax, gx = axes[0, j], axes[1, j]
        ax.fill_between(eps, B.min(0), B.max(0), color="0.8", label="base, 6 runs (min-max)")
        ax.plot(eps, [pair[e][key] for e in eps], color="k", lw=1, label=f"base r_random s{args.seed}")
        gx.fill_between(eps, -sd, sd, color="0.8", label="base +- 1 sd")
        gx.fill_between(eps, -2 * sd, 2 * sd, color="0.9")
        gx.axhline(0, color="k", lw=0.6)
        for arm, (log, done) in runs.items():
            x, y = series(log, key)
            c = COLORS.get(arm, None)
            lab = arm + ("" if done else " (running)")
            ax.plot(x, y, color=c, lw=1.2, label=lab)
            m = np.isin(x, eps)
            idx = np.searchsorted(eps, x[m])
            gx.plot(x[m], y[m] - mu[idx], color=c, lw=1.2, label=lab)
            for ue in unfreeze_epochs(arm):
                for a_ in (ax, gx):
                    a_.axvline(ue, color=c, lw=0.7, ls=":")
        ax.set_ylabel("test top-1 (%)" if key == "test_acc1" else "test loss")
        gx.set_ylabel(("top-1" if key == "test_acc1" else "loss") + " - base mean")
        gx.set_xlabel("epoch")
        if key == "test_acc1":
            ax.set_ylim(max(0, min(B.min(), 30)), None)
        ax.legend(fontsize=8)
        ax.set_title(("top-1" if key == "test_acc1" else "loss") + " (dotted: unfreeze epochs)")
    # zoomed insets are left to the reader: the gap panel carries the comparison
    path = Path(args.out) / f"{args.prefix}_curves_s{args.seed}.png"
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
