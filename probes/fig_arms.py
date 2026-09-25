"""fig24: the three-arm picture -- random vs proc vs the checkpoint-free composed init, across every readout.

Collects one scalar per (arm, readout): the model's own-head accuracy on ImageNet and the six shifted test sets,
and the frozen-feature readouts at their best depth (linear probes, transfer, segmentation, depth, k-NN,
SPair-71k correspondence, TokenCut object discovery).  Panel 1 shows each arm's distance from random in points
with random's seed spread as the noise scale; panel 2 asks whether the intervention reproduces proc by plotting
(proc - random) against (composed - random), where a point on the diagonal means "fully reproduced".  For the
shifted sets the accuracy-matched version is used as well (offset from random's slope-1 probit line, fig22), so
that the composed arm's higher ImageNet accuracy is not read as robustness.

  python -m probes.fig_arms [--arms r,p,ftbc7lm]
"""
import argparse
import csv
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probes import data as pdata
from probes import registry

T7 = [d for d in pdata.CLASSIFICATION if d != "imnet"]


def _load(sub, pat="*.json", skip=("__limit",)):
    out = []
    for f in sorted(glob.glob(str(registry.RES / sub / pat))):
        if any(s in f for s in skip):
            continue
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            print(f"[fig_arms] unreadable {f}")
    return out


def per_model_values():
    """-> {model: {readout: value}} plus the readout order/labels."""
    probes = [r for r in _load("probes") if "__knn" not in r.get("dataset", "") ]
    probes = [r for r in _load("probes", "*_s*.json", ("__limit", "__knn", "__lowlevel"))]
    knn = _load("probes", "*__knn.json")
    dense = _load("dense")
    ood = [r for r in _load("ood") if r.get("complete") or r.get("sets")]
    disc = _load("discovery")
    corr = _load("correspondence")
    V = {}

    def put(model, key, val):
        if val is not None and np.isfinite(val):
            V.setdefault(model, {})[key] = float(val)

    for r in ood:
        s = r["sets"]
        put(r["model"], "imagenet_val", s.get("imnet_val", {}).get("top1"))
        put(r["model"], "imagenet_v2", s.get("imagenet_v2", {}).get("top1"))
        put(r["model"], "imagenet_r", s.get("imagenet_r", {}).get("top1_masked"))
        put(r["model"], "imagenet_sketch", s.get("imagenet_sketch", {}).get("top1"))
        put(r["model"], "imagenet_a", s.get("imagenet_a", {}).get("top1_masked"))
        put(r["model"], "objectnet", s.get("objectnet", {}).get("top1_masked"))
        put(r["model"], "imagenet_c", s.get("imagenet_c", {}).get("top1"))
    for r in probes:
        if r["dataset"] == "imnet":
            put(r["model"], "imnet_probe_b11", r["ftypes"]["concat"]["test_acc"][12])
    for m in {r["model"] for r in probes}:
        per = [np.asarray(r["ftypes"]["concat"]["test_acc"]) for r in probes if r["model"] == m and r["dataset"] in T7]
        if len(per) == len(T7):
            put(m, "transfer_best", np.mean(per, 0).max())
            put(m, "transfer_best__b11", np.mean(per, 0)[12])
    for m in {r["model"] for r in knn}:
        per = [np.asarray(r["knn_top1"]) for r in knn if r["model"] == m and r["dataset"] in T7]
        if len(per) == len(T7):
            put(m, "knn_best", np.mean(per, 0).max())
            put(m, "knn_best__b11", np.mean(per, 0)[12])
    for r in dense:
        key = {"voc": ("voc_miou", "miou"), "ade20k": ("ade_miou", "miou"), "nyu": ("nyu_delta1", "delta1")}.get(r["dataset"])
        if key:
            put(r["model"], key[0], max(r[key[1]]))
            put(r["model"], key[0] + "__b11", r[key[1]][12])
    for r in [x for x in corr if x["dataset"].endswith("_test")]:
        put(r["model"], "spair_pck", max(r["pck@0.1"]))
        put(r["model"], "spair_pck__b11", r["pck@0.1"][12])
    for r in disc:
        put(r["model"], "corloc", max(r["corloc"]))
        put(r["model"], "corloc__b11", r["corloc"][12])
        put(r["model"], "corloc_keys", max(r["corloc_keys"]))
        put(r["model"], "corloc_keys__b11", r["corloc_keys"][11])
    return V


READOUTS = [
    ("imagenet_val", "ImageNet val (own head)", "own head"),
    ("imagenet_v2", "ImageNet-V2", "own head"),
    ("imagenet_r", "ImageNet-R (200)", "own head"),
    ("imagenet_sketch", "ImageNet-Sketch", "own head"),
    ("imagenet_a", "ImageNet-A (200)", "own head"),
    ("objectnet", "ObjectNet (113)", "own head"),
    ("imagenet_c", "ImageNet-C (mean)", "own head"),
    ("imnet_probe_b11", "ImageNet probe b11", "frozen features"),
    ("transfer_best", "transfer mean of 7 (best)", "frozen features"),
    ("knn_best", "k-NN transfer (best)", "frozen features"),
    ("voc_miou", "VOC seg mIoU (best)", "frozen features"),
    ("ade_miou", "ADE20K seg mIoU (best)", "frozen features"),
    ("nyu_delta1", "NYU depth delta1 (best)", "frozen features"),
    ("spair_pck", "SPair-71k PCK@0.1 (best)", "training-free"),
    ("corloc", "TokenCut CorLoc, stream (best)", "training-free"),
    ("corloc_keys", "TokenCut CorLoc, keys (best)", "training-free"),
]


B11 = {k: k + "__b11" for k in ("transfer_best", "knn_best", "voc_miou", "ade_miou", "nyu_delta1",
                                "spair_pck", "corloc", "corloc_keys")}


def arm_table(V, arms):
    """-> {arm: {readout: (mean, sd, n)}}"""
    out = {}
    for a in arms:
        names = [m["name"] for m in registry.MODELS if m["arm"] == a and m["name"] in V]
        rec = {}
        for k in [x for x, _, _ in READOUTS] + list(B11.values()):
            vals = [V[n][k] for n in names if k in V[n]]
            if vals:
                rec[k] = (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals))
        out[a] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r,p,ftbc7lm")
    args = ap.parse_args()
    registry.ensure_dirs()
    arms = args.arms.split(",")
    V = per_model_values()
    A = arm_table(V, arms)
    base = arms[0]
    keys = [k for k, _, _ in READOUTS if k in A[base] and all(k in A[a] for a in arms)]
    labels = {k: l for k, l, _ in READOUTS}
    group = {k: g for k, _, g in READOUTS}

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    y = np.arange(len(keys))[::-1]
    span = max(abs(A[a][k][0] - A[base][k][0]) + A[a][k][1] for a in arms[1:] for k in keys)
    xlim = span * 1.12
    for i, k in enumerate(keys):
        s0 = A[base][k][1]
        ax.add_patch(plt.Rectangle((-s0, y[i] - 0.22), 2 * s0, 0.44, color="0.88", zorder=1))
    for j, a in enumerate(arms[1:]):
        c = registry.ARM_COLOR[a]
        d = [A[a][k][0] - A[base][k][0] for k in keys]
        e = [A[a][k][1] for k in keys]
        off = 0.17 if j == 0 else -0.17
        ax.errorbar(d, y + off, xerr=e, fmt="o", color=c, ms=5.5, capsize=2.5, lw=1.1, zorder=3,
                    label=f"{a} (n={A[a][keys[0]][2]})")
    ax.axvline(0, color=registry.ARM_COLOR[base], lw=1.6, zorder=2)
    # numeric columns on the right, one per arm, outside the data range
    colx = [xlim * (1.12 + 0.17 * j) for j in range(len(arms) - 1)]
    for j, a in enumerate(arms[1:]):
        ax.text(colx[j], y[0] + 0.85, a, ha="center", va="bottom", fontsize=7.5, color=registry.ARM_COLOR[a], fontweight="bold")
        for i, k in enumerate(keys):
            ax.text(colx[j], y[i], f"{A[a][k][0] - A[base][k][0]:+.2f}", ha="center", va="center",
                    fontsize=7, color=registry.ARM_COLOR[a], family="monospace")
    ax.set_xlim(-xlim, colx[-1] + xlim * 0.12)
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.3)
    ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in keys], fontsize=8)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    ax.set_xticks([t for t in ax.get_xticks() if -xlim <= t <= xlim])
    ax.set_xlabel(f"difference from {base}, in points of each readout's own metric\n(grey band = {base}'s ±1 seed sd)", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title("Every readout, as a distance from random", fontsize=9.5)

    ax = axes[1]
    tgt, comp = arms[1], arms[-1]
    pts = [k for k in keys if k != "imagenet_val"]          # the ID axis is the control variable, not a readout
    dx = {k: (A[tgt][k][0] - A[base][k][0]) / max(A[base][k][1], 1e-6) for k in pts}
    dy = {k: (A[comp][k][0] - A[base][k][0]) / max(A[base][k][1], 1e-6) for k in pts}
    lim = max(2.5, max(abs(v) for v in list(dx.values()) + list(dy.values())) * 1.2)
    ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=1, ls="--", zorder=1, label=f"{comp} reproduces {tgt} exactly")
    ax.fill_between([-lim, lim], [-1, -1], [1, 1], color="0.9", zorder=0)
    ax.fill_betweenx([-lim, lim], [-1, -1], [1, 1], color="0.9", zorder=0)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1); ax.axvline(0, color="0.6", lw=0.8, zorder=1)
    GC = {"own head": "#1f78b4", "frozen features": "#e6ab02", "training-free": "#7570b3"}
    order = sorted(pts, key=lambda k: dx[k])
    for i, k in enumerate(order):
        col = GC[group[k]]
        ax.scatter(dx[k], dy[k], s=48, color=col, edgecolor="white", linewidth=0.9, zorder=4)
        dxo, dyo = (7, 5) if i % 2 == 0 else (-7, -11)
        ax.annotate(labels[k], (dx[k], dy[k]), xytext=(dxo, dyo), textcoords="offset points", fontsize=6.3,
                    color="0.2", ha="left" if dxo > 0 else "right", zorder=5)
    for g, col in GC.items():
        ax.scatter([], [], s=48, color=col, label=g)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{tgt} − {base}, in units of {base}'s seed sd", fontsize=8)
    ax.set_ylabel(f"{comp} − {base}, in units of {base}'s seed sd", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    ax.set_title(f"Does {comp} reproduce {tgt}?   (grey = within one seed sd of {base}; ImageNet val excluded)", fontsize=9.5)
    fig.suptitle(f"fig24: {' vs '.join(arms)} across every readout — own-head accuracy on ImageNet and six shifted sets, "
                 f"frozen-feature probes at their best depth, and training-free correspondence / object discovery", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(registry.FIGS / "fig24_arms.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------- tables
    with open(registry.FIGS / "arms_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["readout", "group"] + [f"{a}_{x}" for a in arms for x in ("mean", "sd", "n")]
                   + [f"{a}_at_b11" for a in arms] + [f"d_{a}" for a in arms[1:]] + [f"d_{a}_in_sd" for a in arms[1:]])
        for k in keys:
            row = [labels[k], group[k]]
            for a in arms:
                row += [round(A[a][k][0], 3), round(A[a][k][1], 3), A[a][k][2]]
            for a in arms:
                b = B11.get(k)
                row += [round(A[a][b][0], 3) if b and b in A[a] else ""]
            s0 = max(A[base][k][1], 1e-6)
            row += [round(A[a][k][0] - A[base][k][0], 3) for a in arms[1:]]
            row += [round((A[a][k][0] - A[base][k][0]) / s0, 2) for a in arms[1:]]
            w.writerow(row)
    md = ["value = best depth (b11 in brackets where the readout is depth-resolved)", "",
          "| readout | " + " | ".join(f"{a} (n={A[a][keys[0]][2]})" for a in arms) + " | " +
          " | ".join(f"{a} - {base}" for a in arms[1:]) + " |",
          "|---" * (1 + len(arms) + len(arms) - 1) + "|"]
    for k in keys:
        b = B11.get(k)
        cells = [f"{A[a][k][0]:.2f}" + (f" ± {A[a][k][1]:.2f}" if A[a][k][2] > 1 else "")
                 + (f" ({A[a][b][0]:.1f})" if b and b in A[a] else "") for a in arms]
        diffs = [f"{A[a][k][0] - A[base][k][0]:+.2f}" for a in arms[1:]]
        md.append(f"| {labels[k]} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
    (registry.FIGS / "arms_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote fig24_arms.png, arms_summary.csv, arms_summary.md to {registry.FIGS}")
    missing = [(a, k) for a in arms for k, _, _ in READOUTS if k not in A[a]]
    if missing:
        print(f"[fig_arms] missing {len(missing)} (arm, readout) pairs: {missing[:8]}")


if __name__ == "__main__":
    main()
