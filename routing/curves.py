"""Post-hoc figures for a routed run: coefficient heatmaps at init / mid / final and the
accuracy trajectory against the base arm. Reads routing.jsonl and log.txt; no GPU.

    python routing/curves.py --run kempfe/routing/runs/route_29590809/s0 --out kempfe/routing/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models.residual_routing import consumer_labels, source_labels  # noqa: E402

BASE = [REPO / f"results/imnet_base/results_IMNET_BASE_29384839/s{s}/log.txt" for s in range(3)]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_routing(p):
    """{epoch: matrix}; last row per epoch wins (a requeue re-logs its resume epoch)."""
    out = {}
    for d in load_jsonl(p):
        out[d["epoch"]] = np.array(d["a"])
    return out


def pick(epochs):
    e = sorted(epochs)
    mid = min(e, key=lambda x: abs(x - (e[0] + e[-1]) / 2))
    return list(dict.fromkeys([e[0], mid, e[-1]]))


def plot_heatmaps(mats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    eps = pick(mats)
    n = len(mats[eps[0]])
    depth = (n - 1) // 2
    rows, cols = consumer_labels(depth), source_labels(depth)
    vals = np.concatenate([m[np.tril(np.ones_like(m)) > 0] for m in (mats[e] for e in eps)])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    vmax = max(vals.max(), 1 / vals.min(), 1.01)
    norm = LogNorm(vmin=1 / vmax, vmax=vmax)             # symmetric log: white = 1
    fig, axes = plt.subplots(1, len(eps), figsize=(6.2 * len(eps), 5.6), layout="constrained")
    for ax, e in zip(np.atleast_1d(axes), eps):
        C = mats[e]
        im = ax.imshow(np.where(np.tril(np.ones_like(C)) > 0, C, np.nan), cmap="RdBu_r",
                       norm=norm, origin="upper", aspect="auto")
        step = max(1, n // 12)
        ax.set_xticks(range(0, n, step)); ax.set_xticklabels(cols[::step], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(0, n, step)); ax.set_yticklabels(rows[::step], fontsize=7)
        ax.set_xlabel("source"); ax.set_ylabel("consumer")
        tril = C[np.tril(np.ones_like(C)) > 0]
        ax.set_title(f"epoch {e}   min {tril.min():.3f}  max {tril.max():.2f}  "
                     f"emb col {C[1:, 0].mean():.2f}  diag {np.diag(C).mean():.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("residual routing -- coefficient matrices (log colour, white = 1, upper triangle invalid)")
    fig.savefig(path, dpi=130)
    print("wrote", path)


def plot_curves(run_log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = load_jsonl(run_log)
    bases = [load_jsonl(p) for p in BASE if p.exists()]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), layout="constrained")
    # test acc
    re = [(d["epoch"], d["test_acc1"]) for d in r if d.get("test_acc1") is not None]
    for i, b in enumerate(bases):
        be = [(d["epoch"], d["test_acc1"]) for d in b if d.get("test_acc1") is not None]
        ax[0].plot(*zip(*be), color="0.5", lw=1, label="base (3 seeds)" if i == 0 else None)
    ax[0].plot(*zip(*re), color="C1", lw=1.8, label="routed s0")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test top-1"); ax[0].legend(); ax[0].set_title("test accuracy")
    # delta vs base s0 at matched epochs
    b0 = {d["epoch"]: d for d in bases[0]} if bases else {}
    de = [(e, a - b0[e]["test_acc1"]) for e, a in re if e in b0 and b0[e].get("test_acc1") is not None]
    ax[1].axhline(0, color="k", lw=0.8); ax[1].plot(*zip(*de), "o-", color="C1", ms=3)
    if len(bases) == 3:
        fin = [ [d for d in b if d["epoch"] == 299][-1]["test_acc1"] for b in bases]
        sd = float(np.std(fin, ddof=1))
        ax[1].axhspan(-sd, sd, color="0.85", label=f"base seed sd at 299 (±{sd:.2f})"); ax[1].legend()
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("routed − base s0 (top-1)"); ax[1].set_title("accuracy gap at matched epochs")
    # train loss + coefficient drift
    ax[2].plot([d["epoch"] for d in r], [d["train_loss"] for d in r], color="C1", label="routed train loss")
    if bases:
        ax[2].plot([d["epoch"] for d in bases[0]], [d["train_loss"] for d in bases[0]], color="0.5", label="base s0 train loss")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("train loss"); ax[2].legend(loc="upper right")
    ax2 = ax[2].twinx()
    ax2.plot([d["epoch"] for d in r], [d["routing/max"] for d in r], color="C3", lw=1, ls="--", label="routing/max")
    ax2.plot([d["epoch"] for d in r], [d["routing/emb_col_mean"] for d in r], color="C0", lw=1, ls="--", label="emb column mean")
    ax2.set_ylabel("coefficient"); ax2.legend(loc="center right")
    ax[2].set_title("train loss (lower for routed) and coefficient drift")
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir holding log.txt and routing.jsonl")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(load_routing(run / "routing.jsonl"), out / "routing_heatmaps.png")
    plot_curves(run / "log.txt", out / "routing_curves.png")
