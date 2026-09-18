"""Figures for docs/training_dynamics.md from the stage results (seed mean per arm, thin per-seed lines,
`probes.registry.ARM_COLOR` identities, n = 1 arms labelled).

  python -m dynamics.figures [--only rawlens,bypass,funcchange,grads]
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dynamics import registry as dreg  # noqa: E402
from dynamics import summarize as S  # noqa: E402
from dynamics.rawlens import HALF  # noqa: E402
from probes import registry as preg  # noqa: E402

BLOCK_SLOTS = [HALF.index(f"b{l}") for l in range(12)]
ARM_ROLE = dict(dreg.ROLE_OF_ARM)
ARMS = [a for a in preg.ARM_ORDER if a in ARM_ROLE]


def arms_by_role():
    out = defaultdict(list)
    for a in ARMS:
        out[ARM_ROLE[a]].append(a)
    return out


def runs_of(arm):
    """Seed runs of an arm: the dynamics runs, or the history-only canonical seeds."""
    dyn = [r["name"] for r in dreg.RUNS if r["name"].rsplit("_s", 1)[0] == arm]
    return dyn or [n for n in dreg.HISTORY_RUNS if n.rsplit("_s", 1)[0] == arm]


def color(arm):
    return preg.ARM_COLOR.get(arm, "0.5")


def savefig(fig, name):
    dreg.FIGS.mkdir(parents=True, exist_ok=True)
    p = dreg.FIGS / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}", flush=True)


# ------------------------------------------------------------------------------------------------- experiment 1
def mean_by_epoch(series):
    """[(E [n_i], V [n_i, ...])] -> (E_common, mean over seeds joined by epoch identity, n) ; None if no common epochs."""
    if not series:
        return None
    common = sorted(set.intersection(*[set(np.asarray(E).tolist()) for E, _ in series]))
    if not common:
        return None
    rows = []
    for E, V in series:
        idx = [np.asarray(E).tolist().index(c) for c in common]
        rows.append(np.asarray(V)[idx])
    return np.array(common, float), np.mean(rows, 0), len(series)


def rawlens_curve(run, split="test", metric="top1"):
    """(E [n], values [n, 25]) merging the historical training-time lens (canonical arms) with recomputed epochs;
    recomputed values take precedence where both exist.  Init (E = 0) is excluded."""
    parts = {}
    h = S.load_hist(run) if (split == "test" and metric == "top1") else None
    if h is not None:
        for E, row in zip(h["E"], h["top1"]):
            if E > 0:
                parts[float(E)] = np.asarray(row, float)
    rl = S.load_rawlens(run, split)
    if rl is not None:
        for E, row in zip(rl["E"], rl[metric]):
            if E > 0:
                parts[float(E)] = np.asarray(row, float)
    if not parts:
        return None
    E = np.array(sorted(parts))
    return E, np.stack([parts[e] for e in E])


def rawlens_grid(run, split="test"):
    """(E, top1 over the 12 after-block depths, source tag) for the heatmaps."""
    c = rawlens_curve(run, split, "top1")
    if c is None:
        return None
    src = "rawlens" if S.load_rawlens(run, split) is not None else "hist"
    if src == "rawlens" and load_hist_ok(run):
        src = "hist+rawlens"
    return c[0], c[1][:, BLOCK_SLOTS], src


def load_hist_ok(run):
    return S.load_hist(run) is not None


def fig_rawlens_heatmaps(split="test"):
    per_arm = {}
    for arm in ARMS:
        grids = [g for g in (rawlens_grid(n, split) for n in runs_of(arm)) if g is not None]
        m = mean_by_epoch([(g[0], g[1]) for g in grids])
        if m is None:
            continue
        per_arm[arm] = (m[0], m[1], m[2], grids[0][2])
    if not per_arm:
        return
    n = len(per_arm)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), squeeze=False)
    for ax, (arm, (E, M, k, src)) in zip(axes.flat, per_arm.items()):
        im = ax.pcolormesh(np.arange(len(E) + 1), np.arange(13), M.T, vmin=0, vmax=85, cmap="viridis", shading="flat")
        ax.set_xticks(np.arange(len(E)) + 0.5)
        ax.set_xticklabels([f"{int(e)}" if i % max(1, len(E) // 8) == 0 else "" for i, e in enumerate(E)], fontsize=6, rotation=90)
        ax.set_yticks(np.arange(12) + 0.5)
        ax.set_yticklabels([f"b{l}" for l in range(12)], fontsize=6)
        ax.set_title(f"{arm} ({ARM_ROLE[arm]}, n={k}, {src})", fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="raw-lens top-1 (%)")
    fig.suptitle(f"Raw lens (checkpoint's own norm+head on CLS) over training, {split} set; x = epochs trained E, y = after-block depth", fontsize=9)
    savefig(fig, f"fig1_rawlens_heatmap_{split}.png")


def fig_rawlens_curves(split="test", depths=("b6", "b8", "b9", "b10", "b11"), metric="top1"):
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.2), sharey=metric == "top1")
    for ax, d in zip(axes, depths):
        for arm in ARMS:
            curves = []
            for n in runs_of(arm):
                c = rawlens_curve(n, split, metric)
                if c is None:
                    continue
                E, v = c[0], c[1][:, HALF.index(d)]
                curves.append((E, v))
                ax.plot(E, v, color=color(arm), lw=0.6, alpha=0.35)
            m = mean_by_epoch(curves)
            if m is not None:
                ax.plot(m[0], m[1], color=color(arm), lw=1.6, label=f"{arm} (n={m[2]})")
        ax.set_xscale("log")
        ax.set_title(d, fontsize=9)
        ax.set_xlabel("epochs trained E")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"raw-lens {metric} ({split})")
    axes[-1].legend(fontsize=6, ncol=2)
    savefig(fig, f"fig2_rawlens_{metric}_{split}.png")


# ------------------------------------------------------------------------------------------------- experiment 2
def bypass_series(run, split="test", gain="0.0", key="C"):
    recs = [r for r in S.load_stage("bypass", run) if r["split"] == split]
    if not recs:
        return None
    E = np.array([r["E"] for r in recs], float)
    M = np.array([r["by_gain"][gain][key] for r in recs], float)       # [n, 12]
    last3 = np.array([r["multi"]["last3"][key] for r in recs], float)
    return E, M, last3


def fig_bypass(split="test", gain="0.0"):
    blocks = [8, 9, 10, 11]
    for key, lab in (("C", "C_l = L_bypass - L_full (hard CE)"), ("dacc", "top-1 lost when bypassing block l (pts)")):
        fig, axes = plt.subplots(1, len(blocks) + 1, figsize=(3.4 * (len(blocks) + 1), 3.2))
        for ax, l in zip(axes, blocks + ["last3"]):
            for arm in ARMS:
                series = [s for s in (bypass_series(n, split, gain, key) for n in runs_of(arm)) if s is not None]
                if not series:
                    continue
                curves = []
                for E, M, last3 in series:
                    v = last3 if l == "last3" else M[:, l]
                    curves.append((E, v))
                    ax.plot(E, v, color=color(arm), lw=0.6, alpha=0.35)
                m = mean_by_epoch(curves)
                if m is not None:
                    ax.plot(m[0], m[1], color=color(arm), lw=1.6, label=f"{arm} (n={m[2]})")
            ax.set_xscale("log")
            ax.set_title(f"block {l}" if l != "last3" else "blocks 9-11 together", fontsize=9)
            ax.set_xlabel("epochs trained E")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(lab)
        axes[-1].legend(fontsize=6, ncol=2)
        savefig(fig, f"fig3_bypass_{key}_g{gain}_{split}.png")


# ------------------------------------------------------------------------------------------------- experiment 3
def funcchange_series(run, set_name="diag_train2k", key="U_frob"):
    """(E_to [n], U [n,12] with censored fp16 values as NaN, floor [12]) using the summary's censoring rule."""
    cu = S.censored_U(run, set_name, key)
    if cu is None:
        return None
    e, E_to, U, mask, floor, src = cu
    return E_to, np.where(mask, np.nan, U), floor


def fig_funcchange(set_name="diag_train2k", key="U_frob"):
    per_arm = {}
    for arm in ARMS:
        series = [s for s in (funcchange_series(nm, set_name, key) for nm in runs_of(arm)) if s is not None]
        if series:
            per_arm[arm] = series
    if not per_arm:
        return
    lr = dreg.lr_epoch_means()
    cmap = plt.get_cmap("viridis")
    n = len(per_arm)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False, sharey=True)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for k, (ax, (arm, series)) in enumerate(zip(axes.flat, per_arm.items())):
        m = mean_by_epoch([(E, U) for E, U, _ in series])
        E, U, nn = m
        floor = np.max([f for _, _, f in series], 0)
        for l in range(12):
            ax.plot(E, U[:, l], color=cmap(l / 11), lw=0.9, label=f"b{l}" if k == 0 else None)
        ax.fill_between(E, 1e-8, 3 * floor.max(), color="0.8", alpha=0.5, lw=0, label="3x fp16 floor (censored below)" if k == 0 else None)
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, 301), lr, color="k", lw=0.6, ls=":")
        ax2.set_yscale("log")
        ax2.set_yticks([])
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"{arm} ({ARM_ROLE[arm]}, n={nn})", fontsize=8)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=5, ncol=3)
    fig.suptitle(f"Block-function change per epoch pair (x = end of the interval), {key} on {set_name}; censored fp16 values omitted; dotted: lr", fontsize=9)
    savefig(fig, f"fig4_funcchange_{key}_{set_name}.png")
    # late share (lower bound: censored terms = 0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for arm, series in per_arm.items():
        curves = []
        for E, U, _ in series:
            U0 = np.nan_to_num(U, nan=0.0)
            share = U0[:, 9:].sum(1) / np.maximum(U0.sum(1), 1e-30)
            curves.append((E, share))
            ax.plot(E, share, color=color(arm), lw=0.6, alpha=0.35)
        m = mean_by_epoch(curves)
        if m is not None:
            ax.plot(m[0], m[1], color=color(arm), lw=1.6, label=f"{arm} (n={m[2]})")
    ax.set_xscale("log")
    ax.set_xlabel("epochs trained E (end of interval)")
    ax.set_ylabel("late share  sum_{l>=9} U_l / sum_l U_l  (censored = 0)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, f"fig4b_funcchange_lateshare_{key}_{set_name}.png")


# ------------------------------------------------------------------------------------------------- experiment 4
def grads_series(run, set_name="diag_train4k", objective="smooth"):
    recs = [r for r in S.load_stage("grads", run) if set_name in r["sets"]]
    if not recs:
        return None
    E = np.array([r["E"] for r in recs], float)
    o = [r["sets"][set_name]["objectives"][objective] for r in recs]
    return dict(E=E, per_block=np.array([x["per_block"] for x in o]), rel=np.array([x["rel_block"] for x in o]),
                late=np.array([x["late_share"] for x in o]), total=np.array([x["total"] for x in o]),
                p_minus_q=np.array([x["stats"]["mean_p_y_minus_q_y"] for x in o]), frac=np.array([x["stats"]["frac_p_y_gt_q_y"] for x in o]),
                loss=np.array([x["loss"] for x in o]))


def fig_grads(set_name="diag_train4k", objective="smooth"):
    panels = [("late", "late share  sum_{l>=9}|g_l|^2 / sum|g_l|^2"), ("total", "|g| total"), ("p_minus_q", "mean p_y - q_y"), ("frac", "frac p_y > q_y")]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.4 * (len(panels) + 1), 3.2))
    for ax, (k, lab) in zip(axes, panels):
        for arm in ARMS:
            series = [s for s in (grads_series(n, set_name, objective) for n in runs_of(arm)) if s is not None]
            if not series:
                continue
            for s in series:
                ax.plot(s["E"], s[k], color=color(arm), lw=0.6, alpha=0.35)
            m = mean_by_epoch([(s["E"], s[k]) for s in series])
            if m is not None:
                ax.plot(m[0], m[1], color=color(arm), lw=1.6, label=f"{arm} (n={m[2]})")
        ax.set_xscale("log")
        if k == "total":
            ax.set_yscale("log")
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("epochs trained E")
        ax.grid(alpha=0.3)
    ax = axes[-1]                                                        # relative gradient of the late blocks
    for arm in ARMS:
        series = [s for s in (grads_series(n, set_name, objective) for n in runs_of(arm)) if s is not None]
        if not series:
            continue
        m = mean_by_epoch([(s["E"], s["rel"][:, 9:].mean(1)) for s in series])
        if m is not None:
            ax.plot(m[0], m[1], color=color(arm), lw=1.6, label=f"{arm} (n={m[2]})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("mean_{l>=9} |g_l| / |theta_l|", fontsize=8)
    ax.set_xlabel("epochs trained E")
    ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"Gradient snapshots on {set_name}, objective {objective}", fontsize=9)
    savefig(fig, f"fig5_grads_{set_name}_{objective}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="rawlens,bypass,funcchange,grads")
    args = ap.parse_args()
    os.umask(0o002)
    want = args.only.split(",")
    if "rawlens" in want:
        for split in ("test", "val"):
            fig_rawlens_heatmaps(split)
            fig_rawlens_curves(split)
        fig_rawlens_curves("test", metric="ce")
        fig_rawlens_curves("test", metric="margin_median")
    if "bypass" in want:
        for split in ("test", "val"):
            fig_bypass(split, "0.0")
        fig_bypass("test", "0.9")
    if "funcchange" in want:
        for key in ("U_frob", "U_cls"):
            fig_funcchange("diag_train2k", key)
        fig_funcchange("diag_test2k", "U_frob")
    if "grads" in want:
        fig_grads("diag_train4k", "smooth")
        fig_grads("diag_test4k", "smooth")
        fig_grads("diag_train4k", "hard")


if __name__ == "__main__":
    main()
