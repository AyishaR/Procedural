"""fig20: dense probes (VOC / ADE20K patch mIoU, NYU depth), low-level image-statistics R^2 and k-NN accuracy vs depth."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probes import data as pdata
from probes import registry
from probes.figutils import by_arm, load_json_dir, plot_arms, style_depth_axis


def main():
    registry.ensure_dirs()
    dense = load_json_dir("dense")
    knn = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__knn.json"))]
    low = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__lowlevel.json"))]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    panels = [
        (axes[0, 0], by_arm([r for r in dense if r["dataset"] == "voc"], lambda r: r["miou"]), "VOC 2012: patch-level mIoU (%)"),
        (axes[0, 1], by_arm([r for r in dense if r["dataset"] == "ade20k"], lambda r: r["miou"]), "ADE20K: patch-level mIoU (%)"),
        (axes[0, 2], by_arm([r for r in dense if r["dataset"] == "nyu"], lambda r: r["delta1"]), "NYUv2 depth: patch-level delta1 (%)"),
        (axes[0, 3], by_arm([r for r in dense if r["dataset"] == "nyu"], lambda r: r["rmse"]), "NYUv2 depth: patch-level RMSE (m)"),
        (axes[1, 0], by_arm([r for r in low if r["dataset"] == "imnet"], lambda r: np.mean(r["r2"], 1)), "low-level statistics: mean R^2 over 10 targets (ImageNet)"),
        (axes[1, 1], by_arm([r for r in low if r["dataset"] == "imnet"], lambda r: np.asarray(r["r2"])[:, r["targets"].index("grad_mag")]), "low-level: gradient-energy R^2 (ImageNet)"),
        (axes[1, 2], by_arm([r for r in knn if r["dataset"] == "imnet"], lambda r: r["knn_top1"]), "k-NN (k=20, cosine) top-1: ImageNet"),
    ]
    for ax, d, title in panels:
        plot_arms(ax, d, label_n=(ax is axes[0, 0])); ax.set_title(title, fontsize=9); style_depth_axis(ax)
    # kNN mean over transfer sets
    per = {}
    for r in knn:
        if r["dataset"] in pdata.CLASSIFICATION and r["dataset"] != "imnet":
            per.setdefault(r["arm"], {}).setdefault(r["seed"], []).append(np.asarray(r["knn_top1"]))
    nT = len(pdata.CLASSIFICATION) - 1
    km = {a: {s: np.mean(v, 0) for s, v in d.items() if len(v) == nT} for a, d in per.items()}
    ax = axes[1, 3]; plot_arms(ax, {a: d for a, d in km.items() if d}); ax.set_title("k-NN top-1: mean over transfer sets", fontsize=9); style_depth_axis(ax)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("fig20: non-classification readouts vs depth (dense probes on the 14x14 patch grid; ridge R^2 of image statistics; k-NN on CLS+mean features)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(registry.FIGS / "fig20_dense_probes.png", dpi=300); plt.close(fig)
    print(f"wrote fig20 to {registry.FIGS} (dense {len(dense)}, knn {len(knn)}, lowlevel {len(low)} records)")


if __name__ == "__main__":
    main()
