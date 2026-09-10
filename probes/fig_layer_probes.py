"""fig18: layerwise linear-probe accuracy vs depth per dataset and arm (+ fig18b per feature type) and
probes_summary.csv.  Reads $PROBE_ROOT/results/probes/<model>__<dataset>.json, writes to $PROBE_ROOT/figures."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probes import data as pdata
from probes import registry
from probes.figutils import arm_stats, by_arm, load_json_dir, missing_report, plot_arms, style_depth_axis

DATASETS = pdata.CLASSIFICATION
TRANSFER = [d for d in DATASETS if d != "imnet"]


def main():
    registry.ensure_dirs()
    recs = load_json_dir("probes")
    missing_report(recs, DATASETS, "fig18")
    acc = {ft: {ds: by_arm([r for r in recs if r["dataset"] == ds], lambda r, ft=ft: r["ftypes"].get(ft, {}).get("test_acc")) for ds in DATASETS}
           for ft in ("cls", "mean", "concat")}
    inr = by_arm([r for r in recs if r["dataset"] == "imnet"], lambda r: r["ftypes"].get("concat", {}).get("imagenet_r_acc_final_step"))

    # ---- fig18: concat headline
    n = len(DATASETS) + 1
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.6), squeeze=False)
    for j, ds in enumerate(DATASETS):
        pa = acc["concat"][ds]
        ax = axes[0, j]; plot_arms(ax, pa); ax.set_title(ds, fontsize=9); style_depth_axis(ax)
        if j == 0: ax.set_ylabel("linear-probe test top-1 (%)", fontsize=8)
        ax = axes[1, j]
        norm = {a: {s: v / v.max() * 100 for s, v in d.items()} for a, d in pa.items()}
        plot_arms(ax, norm, label_n=False); style_depth_axis(ax)
        if j == 0: ax.set_ylabel("% of own best depth", fontsize=8)
        ax = axes[2, j]
        if "r" in pa:
            rm, _, _ = arm_stats(pa["r"])
            diff = {a: {s: v - rm for s, v in d.items()} for a, d in pa.items() if a != "r"}
            plot_arms(ax, diff, label_n=False); ax.axhline(0, color="0.15", lw=0.8)
        style_depth_axis(ax)
        if j == 0: ax.set_ylabel("difference to random (points)", fontsize=8)
    # last column: mean over transfer sets (rows 0-1: mean acc / mean normalised; row 2: mean difference to r), plus IN-R
    def transfer_mean(fn):
        out = {}
        for a in registry.ARM_ORDER:
            per_seed = {}
            for ds in TRANSFER:
                for s, v in acc["concat"][ds].get(a, {}).items():
                    per_seed.setdefault(s, []).append(fn(ds, a, s, v))
            if per_seed:
                out[a] = {s: np.mean(vs, 0) for s, vs in per_seed.items() if len(vs) == len(TRANSFER)}
        return {a: d for a, d in out.items() if d}
    ax = axes[0, -1]; plot_arms(ax, transfer_mean(lambda ds, a, s, v: v)); ax.set_title("mean over transfer sets", fontsize=9); style_depth_axis(ax)
    ax = axes[1, -1]; plot_arms(ax, transfer_mean(lambda ds, a, s, v: v / v.max() * 100), label_n=False); style_depth_axis(ax)
    ax = axes[2, -1]
    rmeans = {ds: arm_stats(acc["concat"][ds]["r"])[0] for ds in TRANSFER if "r" in acc["concat"][ds]}
    dm = transfer_mean(lambda ds, a, s, v: v - rmeans[ds]) if len(rmeans) == len(TRANSFER) else {}
    plot_arms(ax, {a: d for a, d in dm.items() if a != "r"}, label_n=False); ax.axhline(0, color="0.15", lw=0.8); style_depth_axis(ax)
    ax.set_title("mean difference to random", fontsize=9)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle("fig18: layerwise linear probes (CLS+mean-patch features, standardised; wd/step selected on val); depth 0 = embedding, d = output of block d-1", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(registry.FIGS / "fig18_layer_probes.png", dpi=300); plt.close(fig)

    # ---- fig18b: per ftype
    fig, axes = plt.subplots(3, len(DATASETS), figsize=(3.2 * len(DATASETS), 8.2), squeeze=False)
    for i, ft in enumerate(("cls", "mean", "concat")):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]; plot_arms(ax, acc[ft][ds], label_n=(i == 0 and j == 0)); style_depth_axis(ax)
            if i == 0: ax.set_title(ds, fontsize=9)
            if j == 0: ax.set_ylabel(f"{ft}: test top-1 (%)", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle("fig18b: probe accuracy by feature type (CLS token / mean patch / concat)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(registry.FIGS / "fig18b_layer_probes_ftypes.png", dpi=300); plt.close(fig)

    # ---- summary csv (per arm x dataset x ftype) + per-seed rows
    rows = []
    for ft in ("cls", "mean", "concat"):
        for ds in DATASETS:
            for a in registry.ARM_ORDER:
                d = acc[ft][ds].get(a)
                if not d:
                    continue
                m, s, nn = arm_stats(d)
                row = dict(ftype=ft, dataset=ds, arm=a, n=nn, best_depth=int(m.argmax()), best_acc=round(float(m.max()), 2),
                           acc_b4=round(float(m[5]), 2), acc_b6=round(float(m[7]), 2), acc_b8=round(float(m[9]), 2), acc_b11=round(float(m[12]), 2),
                           norm_auc=round(float(np.mean([v / v.max() for v in d.values()])), 4),
                           sd_b8=round(float(s[9]), 2), per_seed_b8=";".join(f"{v[9]:.2f}" for v in d.values()))
                if ds == "imnet" and ft == "concat" and a in inr:
                    im, _, _ = arm_stats(inr[a]); row.update(inr_b8=round(float(im[9]), 2), inr_b11=round(float(im[12]), 2))
                rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("ftype", "dataset", "arm", "n"), k))
    with open(registry.FIGS / "probes_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote fig18, fig18b and probes_summary.csv ({len(rows)} rows) to {registry.FIGS}")


if __name__ == "__main__":
    main()
