"""Print markdown tables for docs section 0e from the probe / lens / dense results (mean +/- sd over seeds).

  python -m probes.summarize            # all tables to stdout
"""
import json

import numpy as np

from probes import data as pdata
from probes import registry
from probes.figutils import arm_stats, by_arm, load_json_dir

IDX = {"b2": 3, "b4": 5, "b6": 7, "b8": 9, "b10": 11, "b11": 12}


def fmt(m, s, n, d=1):
    return f"{m:.{d}f}" if n < 2 else f"{m:.{d}f} ± {s:.{d}f}"


def table(title, per_arm, cols, d=1, note=""):
    """per_arm {arm: {seed: [13]}}; cols = [(label, index or callable)]."""
    print(f"\n**{title}**{('  ' + note) if note else ''}\n")
    print("| arm | n | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|---|" + "|".join("---" for _ in cols) + "|")
    for arm in registry.ARM_ORDER:
        if arm not in per_arm or not per_arm[arm]:
            continue
        vals = np.stack(list(per_arm[arm].values()))
        n = len(vals)
        cells = []
        for _, ix in cols:
            v = ix(vals) if callable(ix) else vals[:, ix]
            cells.append(fmt(v.mean(), v.std(ddof=1) if n > 1 else 0.0, n, d))
        print(f"| {arm} | {n} | " + " | ".join(cells) + " |")


def main():
    probes = load_json_dir("probes")
    lens = load_json_dir("lens")
    dense = load_json_dir("dense")
    knn = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__knn.json")) if "__limit" not in f.name]
    low = [json.load(open(f)) for f in sorted((registry.RES / "probes").glob("*__lowlevel.json")) if "__limit" not in f.name]
    transfer = [d for d in pdata.CLASSIFICATION if d != "imnet"]
    acc = {ds: by_arm([r for r in probes if r["dataset"] == ds], lambda r: r["ftypes"].get("concat", {}).get("test_acc")) for ds in pdata.CLASSIFICATION}
    depth_cols = [(k, IDX[k]) for k in ("b2", "b4", "b6", "b8", "b10", "b11")] + [("best depth", lambda v: v.argmax(1).astype(float)), ("best", lambda v: v.max(1))]
    table("ImageNet-1k linear probe (concat features), test top-1 by depth", acc["imnet"], depth_cols)
    # transfer mean: per seed mean over transfer sets (only seeds with every set)
    tm = {}
    for arm in registry.ARM_ORDER:
        per_seed = {}
        for ds in transfer:
            for s, v in acc[ds].get(arm, {}).items():
                per_seed.setdefault(s, []).append(v)
        tm[arm] = {s: np.mean(v, 0) for s, v in per_seed.items() if len(v) == len(transfer)}
    table("Transfer sets, mean over the 7 sets of test top-1 by depth", {a: d for a, d in tm.items() if d}, depth_cols)
    for ds in transfer:
        table(f"{ds}: test top-1 by depth", acc[ds], depth_cols)
    # relative-to-random at b6/b8 on each set (mean over seeds)
    print("\n**Difference to random at b6 / b8 (points, arm mean - r mean)**\n")
    print("| arm | " + " | ".join(pdata.CLASSIFICATION) + " |")
    print("|---|" + "|".join("---" for _ in pdata.CLASSIFICATION) + "|")
    for arm in registry.ARM_ORDER:
        if arm == "r":
            continue
        cells = []
        for ds in pdata.CLASSIFICATION:
            if arm in acc[ds] and "r" in acc[ds]:
                m, _, _ = arm_stats(acc[ds][arm]); rm, _, _ = arm_stats(acc[ds]["r"])
                cells.append(f"{m[7] - rm[7]:+.1f} / {m[9] - rm[9]:+.1f}")
            else:
                cells.append("—")
        if any(c != "—" for c in cells):
            print(f"| {arm} | " + " | ".join(cells) + " |")
    inr = by_arm([r for r in probes if r["dataset"] == "imnet"], lambda r: r["ftypes"].get("concat", {}).get("imagenet_r_acc"))
    table("ImageNet-R zero-shot with the ImageNet probes (200-class mask)", inr, depth_cols)
    # lenses
    ll = by_arm(lens, lambda r: r["logit_lens"]["top1"]); tl = by_arm(lens, lambda r: r["tuned_lens"]["top1"])
    table("Logit lens top-1 (ImageNet val)", ll, [(k, IDX[k]) for k in ("b6", "b8", "b10", "b11")] + [("first depth >= 90% of final", lambda v: np.array([int(np.argmax(x >= 0.9 * x[12])) for x in v], dtype=float))])
    table("Tuned lens top-1 (ImageNet val)", tl, [(k, IDX[k]) for k in ("b6", "b8", "b10", "b11")] + [("first depth >= 90% of final", lambda v: np.array([int(np.argmax(x >= 0.9 * x[12])) for x in v], dtype=float))])
    pdm = by_arm(lens, lambda r: [r["tuned_lens"]["pred_depth_mean"]] * 13)
    fb8 = by_arm(lens, lambda r: [100 * r["tuned_lens"]["frac_decided_by_b8"]] * 13)
    table("Tuned lens: mean prediction depth / % of images decided by b8", {a: {s: np.array([pdm[a][s][0], fb8[a][s][0]] + [0] * 11) for s in pdm[a]} for a in pdm},
          [("mean prediction depth (0=emb)", 0), ("% decided by b8", 1)], d=2)
    # dense
    for ds, key, lab in (("voc", "miou", "VOC patch mIoU"), ("ade20k", "miou", "ADE20K patch mIoU"), ("nyu", "delta1", "NYUv2 patch delta1 (%)"), ("nyu", "rmse", "NYUv2 patch RMSE (m)")):
        table(lab, by_arm([r for r in dense if r["dataset"] == ds], lambda r, key=key: r[key]), depth_cols, d=2 if key == "rmse" else 1)
    base = [r for r in dense if r["dataset"] == "nyu"]
    if base:
        print(f"\n(constant-prediction baselines: nyu delta1 {base[0]['baseline']['delta1']:.1f}, rmse {base[0]['baseline']['rmse']:.2f}; "
              + "; ".join(f"{d} mIoU {[r for r in dense if r['dataset'] == d][0]['baseline']['miou']:.1f}" for d in ("voc", "ade20k") if any(r['dataset'] == d for r in dense)) + ")")
    table("Low-level image statistics: mean ridge R^2 over 10 targets (ImageNet)", by_arm([r for r in low if r["dataset"] == "imnet"], lambda r: np.mean(r["r2"], 1)), depth_cols, d=3)
    table("k-NN top-1, ImageNet", by_arm([r for r in knn if r["dataset"] == "imnet"], lambda r: r["knn_top1"]), depth_cols)
    km = {}
    for arm in registry.ARM_ORDER:
        per_seed = {}
        for r in knn:
            if r["arm"] == arm and r["dataset"] in transfer:
                per_seed.setdefault(r["seed"], []).append(np.asarray(r["knn_top1"]))
        km[arm] = {s: np.mean(v, 0) for s, v in per_seed.items() if len(v) == len(transfer)}
    table("k-NN top-1, mean over the 7 transfer sets", {a: d for a, d in km.items() if d}, depth_cols)


if __name__ == "__main__":
    main()
