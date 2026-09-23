"""Post-hoc figures for the readout-only routing arms: the readout row at init / mid / final,
per-source trajectories with the gradient-sign consistency, and the accuracy trajectories against
the base arm. Reads readout.jsonl and log.txt per run; no GPU.

    python routing/readout_curves.py --run A=<dir> --run B=<dir> --run C=<dir> --out kempfe/routing/figures
    # the skip follow-up (fixed rows, so only the curves figure), with arm A for reference:
    python routing/readout_curves.py --run A=<dir> --run D=<dir> --run E=<dir> --prefix readout_skip --out ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.residual_routing import source_labels  # noqa: E402
from routing.curves import BASE, load_jsonl, pick  # noqa: E402

ARM_NAMES = {"A": "A fixed_last4", "B": "B learn_last4", "C": "C learn_ones",
             "D": "D fixed_emb_last3", "E": "E fixed_b0_last3"}
COLORS = {"A": "C0", "B": "C1", "C": "C2", "D": "C3", "E": "C4"}


def load_readout(p):
    """{epoch: record}; last record per epoch wins (a requeue re-logs its resume epoch)."""
    out = {}
    for d in load_jsonl(p):
        out[d["epoch"]] = d
    return out


def _labels(n):
    return source_labels((n - 1) // 2)


def plot_rows(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 3.2 * len(runs)), layout="constrained",
                             sharex=True)
    for ax, (arm, rec) in zip(np.atleast_1d(axes), runs.items()):
        eps = pick(rec)
        n = len(rec[eps[0]]["w"])
        xs = np.arange(n)
        width = 0.8 / len(eps)
        for i, e in enumerate(eps):
            w = np.array(rec[e]["w"])
            ax.bar(xs + (i - (len(eps) - 1) / 2) * width, w, width=width,
                   color=plt.cm.viridis(i / max(1, len(eps) - 1)), label=f"epoch {e}")
        ax.axhline(0, color="k", lw=0.6)
        ax.axhline(1, color="0.6", lw=0.6, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(_labels(n), rotation=70, fontsize=7)
        ax.set_ylabel("readout weight")
        wf = np.array(rec[eps[-1]]["w"])
        ax.set_title(f"{ARM_NAMES[arm]}: final min {wf.min():.2f}  max {wf.max():.2f}  "
                     f"emb {wf[0]:.2f}  mean(0..16) {wf[:17].mean():.2f}  mean(17..24) {wf[17:].mean():.2f}",
                     fontsize=10)
        ax.legend(fontsize=8, ncol=len(eps))
    fig.suptitle("readout-only routing: the readout row (linear weights; dashed = 1 = vanilla)")
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_trajectories(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    learn = {a: r for a, r in runs.items() if any(d.get("n_steps") for d in r.values())}
    if not learn:
        print("no learnable arm with gradient statistics; skipping", path)
        return
    fig, axes = plt.subplots(2, len(learn), figsize=(7.5 * len(learn), 8.5), layout="constrained",
                             squeeze=False)
    for col, (arm, rec) in enumerate(learn.items()):
        eps = sorted(rec)
        W = np.array([rec[e]["w"] for e in eps])                 # (epochs, n)
        n = W.shape[1]
        labels = _labels(n)
        ax = axes[0, col]
        cmap = plt.cm.plasma
        for j in range(n):
            late = j >= 2 * ((n - 1) // 2 - 4) + 1
            ax.plot(eps, W[:, j], lw=1.6 if late else 0.9,
                    color=cmap(j / (n - 1)) if late else (0.55, 0.55, 0.55, 0.7),
                    label=labels[j] if late or j == 0 else None)
        ax.axhline(1, color="k", lw=0.6, ls="--"); ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("epoch"); ax.set_ylabel("readout weight")
        ax.set_title(f"{ARM_NAMES[arm]}: per-source weight (grey = emb / blocks 0-7)")
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        # sign consistency heatmap: rows = sources, cols = epochs
        S = np.array([rec[e]["sign_consistency"] if rec[e].get("sign_consistency") else [np.nan] * n
                      for e in eps]).T
        ax = axes[1, col]
        im = ax.imshow(S, aspect="auto", cmap="magma", vmin=0.5, vmax=1.0, origin="upper",
                       extent=[eps[0] - 0.5, eps[-1] + 0.5, n - 0.5, -0.5])
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("epoch")
        ax.set_title("gradient sign consistency per epoch (0.5 = coin flip, 1 = every step agrees)")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_curves(logs, path, routed=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bases = [load_jsonl(p) for p in BASE if p.exists()]
    # per-epoch base mean and seed range at the eval epochs every seed has
    acc = {}
    for b in bases:
        for d in b:
            if d.get("test_acc1") is not None:
                acc.setdefault(d["epoch"], []).append(d["test_acc1"])
    bmean = {e: float(np.mean(v)) for e, v in acc.items() if len(v) == len(bases)}
    brange = {e: (min(v) - bmean[e], max(v) - bmean[e]) for e, v in acc.items() if e in bmean}
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), layout="constrained")
    for i, b in enumerate(bases):
        be = [(d["epoch"], d["test_acc1"]) for d in b if d.get("test_acc1") is not None]
        ax[0].plot(*zip(*be), color="0.5", lw=1, label="base (3 seeds)" if i == 0 else None)
        ax[2].plot([d["epoch"] for d in b], [d["train_loss"] for d in b], color="0.5", lw=1,
                   label="base (3 seeds)" if i == 0 else None)
    es = sorted(bmean)
    ax[1].fill_between(es, [brange[e][0] for e in es], [brange[e][1] for e in es], color="0.85",
                       label="base seed range (3 seeds)")
    if len(bases) == 3:
        fin = [[d for d in b if d["epoch"] == 299][-1]["test_acc1"] for b in bases]
        ax[1].axhline(float(np.std(fin, ddof=1)), color="0.6", lw=0.8, ls=":")
        ax[1].axhline(-float(np.std(fin, ddof=1)), color="0.6", lw=0.8, ls=":",
                      label=f"±1 sd of base at 299 (±{np.std(fin, ddof=1):.2f})")
    ax[1].axhline(0, color="k", lw=0.8)

    def gap_series(r):
        last = {d["epoch"]: d for d in r}
        re = [(e, last[e]["test_acc1"]) for e in sorted(last) if last[e].get("test_acc1") is not None]
        return re, [(e, a - bmean[e]) for e, a in re if e in bmean]

    if routed is not None:
        re, de = gap_series(load_jsonl(routed))
        ax[0].plot(*zip(*re), color="0.3", lw=1, ls="--", label="free router (all 325 edges)")
        ax[1].plot(*zip(*de), color="0.3", lw=1, ls="--", label="free router (all 325 edges)")
    for arm, r in logs.items():
        re, de = gap_series(r)
        last = {d["epoch"]: d for d in r}
        rr = [last[e] for e in sorted(last)]
        if re:
            ax[0].plot(*zip(*re), color=COLORS[arm], lw=1.6, label=ARM_NAMES[arm])
        if de:
            ax[1].plot(*zip(*de), "o-", color=COLORS[arm], ms=2.5, lw=1.2, label=ARM_NAMES[arm])
        ax[2].plot([d["epoch"] for d in rr], [d["train_loss"] for d in rr], color=COLORS[arm], lw=1.4,
                   label=ARM_NAMES[arm])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test top-1"); ax[0].legend(); ax[0].set_title("test accuracy")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("arm − base mean (top-1)"); ax[1].legend(fontsize=8)
    ax[1].set_title("accuracy gap to the base seed mean at matched epochs")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("train loss"); ax[2].legend(); ax[2].set_title("train loss")
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="ARM=<run dir holding log.txt and readout.jsonl>, ARM in A-E; repeatable")
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="readout", help="figure file-name prefix")
    p.add_argument("--routed", default=None, help="log.txt of the free-router run, drawn as a reference")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = dict(kv.split("=", 1) for kv in a.run)
    assert set(dirs) <= set(ARM_NAMES), dirs
    # a fixed arm's row never moves, so rows / trajectories are drawn only when it changed
    runs = {arm: rec for arm, rec in ((arm, load_readout(Path(d) / "readout.jsonl"))
                                      for arm, d in dirs.items() if (Path(d) / "readout.jsonl").exists())
            if any(r["n_steps"] for r in rec.values())}
    logs = {arm: load_jsonl(Path(d) / "log.txt") for arm, d in dirs.items()
            if (Path(d) / "log.txt").exists()}
    if runs:
        plot_rows(runs, out / f"{a.prefix}_rows.png")
        plot_trajectories(runs, out / f"{a.prefix}_trajectories.png")
    if logs:
        plot_curves(logs, out / f"{a.prefix}_curves.png", routed=a.routed)
