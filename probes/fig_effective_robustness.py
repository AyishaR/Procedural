"""fig21: effective-robustness view of the layerwise probes.

x = ImageNet linear-probe top-1 at a depth (in-distribution), y = a transfer / OOD readout at the same depth; each
arm is a trajectory from the embedding to block 11 (large marker); random's trajectory is the reference line.
Also writes the per-arm mean offset from random's line (piecewise-linear interpolation in x) over blocks 3-8 and
9-11 to effective_robustness.csv.  Axes logit-scaled for percentage metrics.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probes import data as pdata
from probes import registry
from probes.figutils import arm_stats, by_arm, load_json_dir


def logit(p):
    q = np.clip(np.asarray(p, dtype=float), 0.2, 99.8)
    return np.log(q / (100 - q))


def per_seed_mean(recs, sets, key):
    out = {}
    for arm in registry.ARM_ORDER:
        per = {}
        for ds in sets:
            for s, v in by_arm([r for r in recs if r["dataset"] == ds], key).get(arm, {}).items():
                per.setdefault(s, []).append(v)
        d = {s: np.mean(v, 0) for s, v in per.items() if len(v) == len(sets)}
        if d:
            out[arm] = d
    return out


def main():
    registry.ensure_dirs()
    probes = load_json_dir("probes"); dense = load_json_dir("dense")
    import json
    knn = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__knn.json")) if "__limit" not in f.name]
    low = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__lowlevel.json")) if "__limit" not in f.name]
    T = [d for d in pdata.CLASSIFICATION if d != "imnet"]
    X = by_arm([r for r in probes if r["dataset"] == "imnet"], lambda r: r["ftypes"]["concat"]["test_acc"])
    Y = {
        "transfer mean (7 sets), top-1 %": (per_seed_mean(probes, T, lambda r: r["ftypes"]["concat"]["test_acc"]), True),
        "ImageNet-R zero-shot, top-1 %": (by_arm([r for r in probes if r["dataset"] == "imnet"], lambda r: r["ftypes"]["concat"]["imagenet_r_acc"]), True),
        "VOC patch mIoU %": (by_arm([r for r in dense if r["dataset"] == "voc"], lambda r: r["miou"]), True),
        "NYUv2 patch delta1 %": (by_arm([r for r in dense if r["dataset"] == "nyu"], lambda r: r["delta1"]), True),
        "k-NN transfer mean, top-1 %": (per_seed_mean(knn, T, lambda r: r["knn_top1"]), True),
        "low-level statistics, mean R2": (by_arm([r for r in low if r["dataset"] == "imnet"], lambda r: np.mean(r["r2"], 1)), False),
    }
    xm = {a: arm_stats(X[a])[0] for a in X}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    rows = []
    for ax, (lab, (S, pct)) in zip(axes.flat, Y.items()):
        ym = {a: arm_stats(S[a])[0] for a in S}
        tx = logit if pct else (lambda v: np.asarray(v, dtype=float))
        for arm in ["r"] + [a for a in registry.ARM_ORDER if a != "r"]:
            if arm not in ym or arm not in xm:
                continue
            x, y = xm[arm], ym[arm]
            c = registry.ARM_COLOR[arm]; n1 = arm in ("ftb4o", "ftbrhos")
            ax.plot(logit(x), tx(y), color=c, lw=2.6 if arm == "r" else 1.6, ls="--" if n1 else "-", alpha=0.9, label=f"{arm} (n={arm_stats(S[arm])[2]})")
            ax.scatter(logit(x), tx(y), s=10, color=c, zorder=3)
            ax.scatter(logit(x[12:]), tx(y[12:]), s=42, color=c, edgecolor="white", linewidth=0.8, zorder=4)
            if arm != "r":
                inside = (x >= xm["r"].min()) & (x <= xm["r"].max())
                off = np.where(inside, y - np.interp(x, xm["r"], ym["r"]), np.nan)
                rows.append(dict(readout=lab, arm=arm, mid_b3_b8=round(float(np.nanmean(off[4:10])), 3), top_b9_b11=round(float(np.nanmean(off[10:13])), 3)))
        xt = [2, 5, 10, 20, 40, 60, 80]; ax.set_xticks(logit(xt)); ax.set_xticklabels(xt)
        if pct:
            lo, hi = min(v.min() for v in ym.values()), max(v.max() for v in ym.values())
            yt = [v for v in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90] if lo <= v <= hi]; ax.set_yticks(logit(yt)); ax.set_yticklabels(yt)
        ax.set_xlabel("ImageNet linear-probe top-1 at the same depth (%), logit axis", fontsize=8); ax.set_ylabel(lab + (" (logit axis)" if pct else ""), fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("fig21: effective robustness of the layerwise probes -- each arm's (ImageNet probe, OOD readout) trajectory from the embedding to block 11 (large marker); random = reference line", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(registry.FIGS / "fig21_effective_robustness.png", dpi=300); plt.close(fig)
    with open(registry.FIGS / "effective_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["readout", "arm", "mid_b3_b8", "top_b9_b11"]); w.writeheader(); w.writerows(rows)
    print(f"wrote fig21 and effective_robustness.csv ({len(rows)} rows) to {registry.FIGS}")


if __name__ == "__main__":
    main()
