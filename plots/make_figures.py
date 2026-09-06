#!/usr/bin/env python3
"""
Figures for the ImageNet-1k late-block calibration results (ViT-B/16, 300 epochs).

Data source
-----------
Per-run `log.txt` under results/imnet_base/results_IMNET_BASE_<slurm_id>/s<seed>/.
This is the authoritative source: train_loss for all 300 epochs and
test_loss / test_acc1 at every eval epoch, for every seed, with no gaps.

Why not wandb: jobs are `--requeue`d and each resume opens a NEW wandb run, so a
single (slurm_id, seed) is split across several wandb runs that must be stitched by
epoch. log.txt is written by the training loop itself and is already continuous.
wandb remains the only source for per-layer rho during training
(`Epoch-wise/delta_norm_ratio_layer*`) -- see fig_rho_evolution.py.

CONVENTIONS
  * every reported accuracy is LAST-EPOCH test top-1, never max-over-epochs.
  * train_loss is computed on mixup(0.8) + cutmix(1.0) + label-smoothing(0.1)
    targets, so it is NOT on the same scale as test_loss.  Comparing train and test
    loss to each other is meaningless here; comparing the SAME quantity ACROSS ARMS
    is fine, and that is all these figures do.

Usage
  python plots/make_figures.py                # all figures -> plots/out/
  python plots/make_figures.py --only 1,2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "imnet_base"
OUT = Path(__file__).resolve().parent / "out"

# --------------------------------------------------------------------------------------
# Arms.  Several arms have their seeds split across job ids, so each arm maps to a
# list of (slurm_id, seed) -- grouping by directory alone would silently undercount.
# --------------------------------------------------------------------------------------
ARMS: dict[str, list[tuple[str, str]]] = {
    "r":         [("29384839", "s0"), ("29384839", "s1"), ("29384839", "s2")],
    "p":         [("29377576", "s0"), ("29377576", "s1"), ("29377576", "s2")],
    "ftbrho":    [("29453944", "s0"), ("29461252", "s1"), ("29461253", "s2")],
    "ftbcomp11": [("29472870", "s0"), ("29472870", "s1"), ("29472870", "s2")],
    "ftbcomp1":  [("29498148", "s0"), ("29498148", "s1"), ("29498148", "s2")],
    "ftbqm":     [("29498141", "s0"), ("29498141", "s1"), ("29498141", "s2")],
    "ftbqm1dpar":[("29502416", "s0"), ("29502416", "s1"), ("29502416", "s2")],
    "ftbqm1d":   [("29501773", "s0"), ("29501773", "s1"), ("29501773", "s2")],
    "ftbqm1dv":  [("29507368", "s0"), ("29507368", "s1"), ("29507368", "s2")],
    "ftbqm1dqk": [("29511670", "s0"), ("29511670", "s1"), ("29511670", "s2")],
    "ftbqm1dvo": [("29511673", "s0"), ("29511673", "s1"), ("29511673", "s2")],
    "ftbqmln":   [("29504032", "s0"), ("29504032", "s1"), ("29504032", "s2")],
    "ftbqmbias": [("29501780", "s0"), ("29501780", "s1"), ("29501780", "s2")],
    "ftb4o":     [("29451652", "s0")],
    "ftbnorm":   [("29482212", "s0"), ("29482212", "s1"), ("29482212", "s2")],
    "ftb4e3":    [("29451642", "s0"), ("29451642", "s1"), ("29451642", "s2")],
    "ftb3i":     [("29469072", "s0"), ("29469072", "s1"), ("29469072", "s2")],
    "ftb11i":    [("29457108", "s0"), ("29457108", "s1"), ("29457108", "s2")],
    "ftb11is":   [("29472466", "s0"), ("29472466", "s1"), ("29472466", "s2")],
    "ftb1i":     [("29469074", "s0"), ("29469074", "s1"), ("29469074", "s2")],
    "ftb7i":     [("29457109", "s0"), ("29469067", "s1"), ("29469068", "s2")],
    "ftb8i":     [("29457107", "s0"), ("29469063", "s1"), ("29469064", "s2")],
    "ftbclip01": [("29485866", "s0"), ("29485866", "s1"), ("29485866", "s2")],
    # --- multi-seed arms added after review: none of these were in the first draft ---
    "a1":        [("29388202", "s0"), ("29406778", "s1"), ("29406779", "s2")],
    "a2":        [("29407014", "s0"), ("29407014", "s1"), ("29407014", "s2")],
    "ftb4jd":    [("29465210", "s0"), ("29469065", "s1"), ("29469066", "s2")],
    "ftbcomp25": [("29465211", "s0"), ("29469061", "s1"), ("29469062", "s2")],
    "ftbclip1":  [("29485869", "s0"), ("29485869", "s1"), ("29485869", "s2")],
    "ftbclip5":  [("29485872", "s0"), ("29485872", "s1"), ("29485872", "s2")],
    "ftb2i":     [("29469073", "s0")],
    "ftb4i":     [("29448854", "s0")],
    "ftb5i":     [("29451645", "s0")],
    "ftb6i":     [("29451646", "s0")],
    "ftb9i":     [("29462317", "s0")],
    "ftb10i":    [("29462316", "s0")],
    # full late-block (h) sweep: ftbKh = K procedural blocks at the END
    "ftb1h":     [("29484973", "s0")],
    "ftb2h":     [("29484974", "s0")],
    "ftb3h":     [("29469075", "s0")],
    "ftb4h":     [("29448853", "s0")],
    "ftb5h":     [("29457110", "s0")],
    "ftb6h":     [("29457111", "s0")],
    "ftb7h":     [("29469076", "s0")],
    "ftb8h":     [("29484975", "s0")],
    "ftb9h":     [("29484976", "s0")],
    "ftb10h":    [("29484977", "s0")],
    "ftb11h":    [("29484978", "s0")],
    "ftb0h":     [("29435780", "s0")],   # all 12 blocks procedural -- shared endpoint
}

LABEL = {
    "r":         "random init",
    "p":         "procedural init",
    "ftbrho":    r"recipe: calibrate blocks 9-11 ($\rho{\approx}1.4$)",
    "ftbcomp11": "proc 0-10 + calibrated block 11",
    "ftbcomp1":  "proc block 0 + calibrated 9-11",
    "ftbqm":     "quantile-matched values, blocks 0-8",
    "ftbnorm":   "per-tensor norms only",
    "ftb4e3":    "proc values shuffled, blocks 0-8",
    "ftb4o":     r"$\rho$ only",
    "ftbclip01": "proc, extreme tail clipped",
    "ftb3i":     "proc blocks 0-8 intact",
    "a1":        "calibrate 9-11 to measured ratios",
    "ftb4jd":    "proc 0-7 + calibrated 8-11",
    "ftbcomp25": "proc 0-3 + calibrated 9-11",
    "ftb1i":     "proc 0-10, block 11 left random",
}

C = {
    "r":         "#8c8c8c",
    "p":         "#0173B2",
    "ftbrho":    "#DE8F05",
    "ftbcomp11": "#029E73",
    "ftbcomp1":  "#56B4E9",
    "ftbqm":     "#CC78BC",
    "ftbqm1dpar":"#CC78BC",
    "ftbnorm":   "#CA9161",
    "ftb4e3":    "#949494",
    "a1":        "#E69F00",
    "ftb4jd":    "#9BD0C0",
    "ftbcomp25": "#7FC3A9",
    "ftb1i":     "#3E9BD1",
    "pos":       "#029E73",
    "neg":       "#D55E00",
    "null":      "#9a9a9a",
}


def setup():
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "font.family": "DejaVu Sans",
    })
    OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
def load_run(sid, seed) -> pd.DataFrame:
    p = RESULTS / f"results_IMNET_BASE_{sid}" / seed / "log.txt"
    return pd.DataFrame([json.loads(l) for l in open(p)])


def load_arm(arm) -> pd.DataFrame:
    out = []
    for sid, seed in ARMS[arm]:
        df = load_run(sid, seed)[["epoch", "train_loss", "test_loss", "test_acc1"]].copy()
        df["seed"], df["arm"] = seed, arm
        out.append(df)
    return pd.concat(out, ignore_index=True)


def summary() -> pd.DataFrame:
    """Per (arm, seed): last-epoch acc, final train loss, min/final test loss."""
    recs = []
    for arm in ARMS:
        for sid, seed in ARMS[arm]:
            df = load_run(sid, seed)
            ev = df[df["test_acc1"].notna() & (df["test_acc1"] > 0)]
            if ev.empty:
                continue
            tl = df[df["test_loss"].notna() & (df["test_loss"] > 0)]
            d = df["train_loss"].diff()
            recs.append(dict(
                arm=arm, seed=seed,
                acc=float(ev.iloc[-1]["test_acc1"]),
                train_loss=float(df["train_loss"].iloc[-1]),
                tl_min=float(tl["test_loss"].min()),
                tl_final=float(tl["test_loss"].iloc[-1]),
                tl_min_epoch=int(tl.loc[tl["test_loss"].idxmin(), "epoch"]),
                spikes=int((d > 0.25).sum()),
                max_jump=float(d.max()),
            ))
    s = pd.DataFrame(recs)
    s["overfit"] = s["tl_final"] - s["tl_min"]
    return s


def ms(S, arm, col="acc"):
    v = S[S["arm"] == arm][col].values
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def band(ax, df, ycol, color, label, lw=1.9, ls="-"):
    d = df[df[ycol].notna() & (df[ycol] > 0)]
    g = d.groupby("epoch")[ycol]
    m, lo, hi = g.mean(), g.min(), g.max()
    ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label, zorder=3)
    if d["seed"].nunique() > 1:
        ax.fill_between(m.index, lo.values, hi.values, color=color, alpha=0.16, lw=0, zorder=2)


def caption(fig, text, color="0.35"):
    fig.text(0.0, -0.045, text, fontsize=7.8, color=color, va="top", ha="left")


# ======================================================================================
# FIG 1 (requested): random init vs late-block calibration -- train loss, test loss, acc
# ======================================================================================
def fig1_loss_curves(S):
    arms = ["r", "ftbrho", "p"]
    D = {a: load_arm(a) for a in arms}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    ax = axes[0]
    for a in arms:
        band(ax, D[a], "train_loss", C[a], LABEL[a])
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
    ax.set_title("Training loss  (lower = fits train set better)")
    ax.legend(loc="upper right")
    ax.set_ylim(2.0, 7.1)
    axi = ax.inset_axes([0.44, 0.34, 0.52, 0.34])
    for a in arms:
        band(axi, D[a], "train_loss", C[a], None, lw=1.4)
    axi.set_xlim(240, 300); axi.set_ylim(2.15, 2.60)
    axi.tick_params(labelsize=6.5); axi.set_title("final epochs", fontsize=7, fontweight="normal")
    axi.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    for a in arms:
        band(ax, D[a], "test_loss", C[a], LABEL[a])
    ax.set_xlabel("epoch"); ax.set_ylabel("test loss")
    ax.set_title("Test loss")
    ax.set_ylim(0.85, 7.1)
    axi = ax.inset_axes([0.44, 0.40, 0.52, 0.36])
    for a in arms:
        band(axi, D[a], "test_loss", C[a], None, lw=1.4)
    axi.set_xlim(150, 300); axi.set_ylim(0.93, 1.22)
    axi.tick_params(labelsize=6.5)
    axi.set_title("random init turns UP", fontsize=7, color="#B00020", fontweight="normal")
    axi.grid(alpha=0.25, lw=0.4)

    ax = axes[2]
    for a in arms:
        band(ax, D[a], "test_acc1", C[a], LABEL[a])
    ax.set_xlabel("epoch"); ax.set_ylabel("test top-1 (%)")
    ax.set_title("Test accuracy")
    ax.set_ylim(35, 82)
    rows = [f"{LABEL[a].split(':')[0]}: {ms(S,a)[0]:.2f} $\\pm$ {ms(S,a)[1]:.2f}" for a in arms]
    ax.text(0.97, 0.06, "\n".join(rows), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(fc="white", ec="0.8", boxstyle="round,pad=0.4"))

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Late-block calibration improves generalisation without improving the training fit\n"
                 "ImageNet-1k, ViT-B/16, 300 epochs, 3 seeds (band = min-max over seeds)",
                 y=1.07, fontsize=11.5, fontweight="bold")
    caption(fig, "Train loss is on mixup+cutmix+smoothed targets, so it is not on the same scale as test loss; "
                 "only across-arm comparison of the same quantity is meaningful.\n"
                 "The recipe tracks random init almost exactly on TRAIN loss, yet ends 1.6 points higher in "
                 "test accuracy. Spikes in the procedural curves are real training instabilities (see fig6).")
    fig.savefig(OUT / "fig1_loss_curves.png")
    plt.close(fig)
    print("fig1_loss_curves.png")


# ======================================================================================
# FIG 2: the mechanism -- better init means WORSE train fit and less overfitting
# ======================================================================================
def fig2_generalisation(S):
    """Across EVERY arm in the study, not just the headline four."""
    A = (S.groupby("arm")
           .agg(n=("acc", "size"), acc=("acc", "mean"), train_loss=("train_loss", "mean"),
                overfit=("overfit", "mean"), tl_min_epoch=("tl_min_epoch", "mean"))
           .reset_index())
    HL = {"r": "random", "ftbrho": "recipe", "p": "procedural", "ftbcomp11": "composition"}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
    fig.subplots_adjust(wspace=0.34)

    for ax, xcol, xlab, ttl in [
            (axes[0], "train_loss", "final train loss",
             "Better training fit $\\rightarrow$ worse generalisation"),
            (axes[1], "overfit", "test-loss rise from its minimum",
             "Overfitting predicts final accuracy")]:
        for _, row in A.iterrows():
            a = row["arm"]
            hl = a in HL
            ax.scatter(row[xcol], row["acc"],
                       s=95 if hl else 34,
                       color=C.get(a, "#b9b9b9") if hl else "#b9b9b9",
                       edgecolor="0.25" if hl else "none",
                       linewidth=0.9, zorder=5 if hl else 3, alpha=1.0 if hl else 0.75)
        r_, p_ = stats.pearsonr(A[xcol], A["acc"])
        xs = np.linspace(A[xcol].min(), A[xcol].max(), 50)
        ax.plot(xs, np.poly1d(np.polyfit(A[xcol], A["acc"], 1))(xs),
                ls="--", lw=1.3, color="0.45", zorder=2)
        ax.set_xlabel(xlab); ax.set_ylabel("test top-1 (%), last epoch")
        ax.set_title(ttl, fontsize=10.2, pad=16)
        ax.text(0.5, 1.012, f"Pearson r = {r_:+.2f}   (n={len(A)} arms, p={p_:.0e})",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6, color="0.3")
        # per-panel label offsets, chosen so no two labels collide
        OFF = {"train_loss": {"r": (0, 15, "center"), "ftbrho": (0, 15, "center"),
                              "p": (0, 15, "center"), "ftbcomp11": (0, 15, "center")},
               "overfit":    {"r": (-8, 8, "right"),  "ftbrho": (10, -4, "left"),
                              "p": (12, -10, "left"), "ftbcomp11": (10, 6, "left")}}
        for a, lab in HL.items():
            row = A[A["arm"] == a].iloc[0]
            dx, dy, ha = OFF[xcol][a]
            ax.annotate(lab, (row[xcol], row["acc"]), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=8.5,
                        color=C[a], fontweight="bold", zorder=6)
        ax.margins(x=0.10, y=0.13)

    ax = axes[2]
    arms4 = ["r", "ftbrho", "p", "ftbcomp11"]
    for i, a in enumerate(arms4):
        m, s, n = ms(S, a, "tl_min_epoch")
        ax.bar(i, m, color=C[a], alpha=0.88, width=0.6, zorder=3)
        ax.errorbar(i, m, yerr=s, color="0.2", capsize=4, lw=1.3, zorder=4)
        ax.text(i, m + s + 5, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(arms4)))
    ax.set_xticklabels([HL[a] for a in arms4], fontsize=8.5)
    ax.set_ylabel("epoch of best test loss")
    ax.set_ylim(0, 330)
    ax.set_title("Best test loss arrives later\nfor the better initialisations", fontsize=10.2, pad=16)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].grid(alpha=0.25, lw=0.5); axes[1].grid(alpha=0.25, lw=0.5)

    fig.suptitle("Initialisation acts as a regulariser here, not as better optimisation",
                 y=1.10, fontsize=11.5, fontweight="bold")
    caption(fig, "Every point is one arm (mean over its seeds); all 27 arms share an identical training "
                 "recipe and differ only in initialisation. Grey = the other arms in the study.\n"
                 "The arm that reaches the LOWEST training loss (random init) is the one that generalises "
                 "worst. Correlations are across arm means and are descriptive, not causal — but they hold "
                 "over the whole study, not just the headline arms.")
    fig.savefig(OUT / "fig2_generalisation.png")
    plt.close(fig)
    print("fig2_generalisation.png")


# ======================================================================================
# FIG 3: headline numbers
# ======================================================================================
def fig3_headline(S):
    """Every n=3 arm on the main ladder, grouped by what the method actually requires."""
    GROUPS = [
        ("no procedural weights\nin the trained model", "#DE8F05",
         [("r",        "random\ninit"),
          ("ftbrho",   r"calib 9-11" + "\n" + r"to $\rho$=1.4"),
          ("a1",       "calib 9-11 to\nmeasured ratios")]),
        ("procedural weights\n+ late-block calibration", "#029E73",
         [("ftbcomp1",  "proc blk 0\n+ calib 9-11"),
          ("ftb4jd",    "proc 0-7\n+ calib 8-11"),
          ("ftbcomp25", "proc 0-3\n+ calib 9-11"),
          ("ftb1i",     "proc 0-10,\nblk 11 random"),
          ("ftbcomp11", "proc 0-10\n+ calib blk 11")]),
    ]
    rb = ms(S, "r")[0]; pb = ms(S, "p")[0]
    order, spans, x = [], [], 0
    for title, gcol, items in GROUPS:
        spans.append((x, x + len(items) - 1, title, gcol))
        for a, lab in items:
            order.append((x, a, lab)); x += 1
        x += 1.25

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axhline(rb, color=C["r"], ls=":", lw=1.2, zorder=1)
    ax.axhline(pb, color=C["p"], ls=":", lw=1.4, zorder=1)
    xmax = order[-1][0] + 0.75
    ax.text(xmax + 0.05, rb, "random\n78.08", fontsize=8, color=C["r"], va="center", ha="left")
    ax.text(xmax + 0.05, pb, "procedural init\n80.09", fontsize=8, color=C["p"],
            va="center", ha="left", fontweight="bold")

    for xi, a, lab in order:
        m, s, n = ms(S, a)
        col = C.get(a, "#666")
        pts = S[S["arm"] == a]["acc"].values
        jit = np.linspace(-0.13, 0.13, len(pts)) if len(pts) > 1 else np.zeros(1)
        ax.scatter(xi + jit - 0.30, pts, s=28, color=col, alpha=0.5, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.errorbar(xi, m, yerr=s if n > 1 else None, fmt="o", ms=10, color=col,
                    capsize=4, lw=1.8, zorder=4)
        ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points", xytext=(0, 24),
                    ha="center", fontsize=9.5, fontweight="bold", color=col)
        if a != "r":
            ax.text(xi, 77.45, f"{m - rb:+.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="0.35")

    for x0, x1, title, gcol in spans:
        ax.add_patch(plt.Rectangle((x0 - 0.55, 77.32), (x1 - x0) + 1.1, 4.2,
                                   fc=gcol, alpha=0.05, ec=gcol, lw=1.0,
                                   linestyle="--", zorder=0))
        ax.text((x0 + x1) / 2, 81.36, title, ha="center", va="top", fontsize=8.8,
                color=gcol, fontweight="bold")

    ax.set_xticks([xi for xi, _, _ in order])
    ax.set_xticklabels([lab for _, _, lab in order], fontsize=8.2)
    ax.set_ylabel("test top-1 (%), last epoch")
    ax.set_ylim(77.3, 81.55)
    ax.set_xlim(-0.75, xmax + 1.35)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.text(-0.7, 77.45, "vs random:", fontsize=8, color="0.35", ha="left", va="bottom")
    ax.set_title("ImageNet-1k, ViT-B/16 — every arm with n=3 seeds on the main ladder\n"
                 "small dots are seeds, large dot is mean $\\pm$ s.d.", loc="left")
    caption(fig, "Left group keeps NO procedural weights. Only the $\\rho$=1.4 arm is fully "
                 "checkpoint-free (absolute target); the 'measured ratios' arm still reads its target off a "
                 "procedural checkpoint,\nthough it keeps none of its weights — and already matches procedural "
                 "init (80.00 vs 80.09). Right group keeps procedural blocks and calibrates the rest.\n"
                 "'proc 0-10, blk 11 random' carries no calibration at all and still beats procedural init, "
                 "which is why block 11 is the one worth calibrating.\n"
                 "Welch: recipe vs random +1.61 (7.9σ); proc 0-10 + calibrated blk 11 vs procedural init "
                 "+0.54 (4.4σ).")
    fig.savefig(OUT / "fig3_headline.png")
    plt.close(fig)
    print("fig3_headline.png")


# ======================================================================================
# FIG 4: where do procedural weights help -- early or late?
# ======================================================================================
def fig4_depth(S):
    iser = [("ftb11i", 1), ("ftb10i", 2), ("ftb9i", 3), ("ftb8i", 4), ("ftb7i", 5),
            ("ftb6i", 6), ("ftb5i", 7), ("ftb4i", 8), ("ftb3i", 9), ("ftb2i", 10), ("ftb1i", 11)]
    hser = [("ftb1h", 1), ("ftb2h", 2), ("ftb3h", 3), ("ftb4h", 4), ("ftb5h", 5), ("ftb6h", 6),
            ("ftb7h", 7), ("ftb8h", 8), ("ftb9h", 9), ("ftb10h", 10), ("ftb11h", 11)]
    rb, pb = ms(S, "r")[0], ms(S, "p")[0]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    txt = []
    for ser, col, lab, mk in [(iser, "#0173B2", "at the START  (blocks 0..k-1)", "o"),
                              (hser, "#D55E00", "at the END  (blocks 12-k..11)", "s")]:
        xs, ys, es, big = [], [], [], []
        for arm, k in ser:
            m, s, n = ms(S, arm)
            xs.append(k); ys.append(m); es.append(s if n > 1 else 0.0); big.append(n > 1)
        ax.errorbar(xs, ys, yerr=es, marker="none", lw=2, color=col, capsize=3, label=lab, zorder=3)
        ax.scatter([x for x, b in zip(xs, big) if b], [y for y, b in zip(ys, big) if b],
                   marker=mk, s=58, color=col, zorder=4, edgecolor="white", linewidth=0.8)
        ax.scatter([x for x, b in zip(xs, big) if not b], [y for y, b in zip(ys, big) if not b],
                   marker=mk, s=28, facecolor="white", edgecolor=col, linewidth=1.4, zorder=4)
        rho, pv = stats.spearmanr(xs, ys)
        txt.append(f"{lab.split('  ')[0]}: Spearman $\\rho$={rho:+.2f} (p={pv:.3f})")

    # shared endpoint: k = 12 is the same model either way (all blocks procedural)
    m12 = ms(S, "ftb0h")[0]
    ax.scatter([12], [m12], marker="*", s=190, color=C["p"], zorder=6,
               edgecolor="white", linewidth=0.9)
    ax.annotate("k=12: both series meet\n(all blocks procedural)", (12, m12),
                textcoords="offset points", xytext=(4, -34), ha="center",
                fontsize=7.4, color=C["p"])

    ax.axhline(rb, color=C["r"], ls=":", lw=1.3)
    ax.annotate("random init", (0.7, rb + 0.04), fontsize=8, color=C["r"], va="bottom")
    ax.set_xlabel("number of procedurally-initialised blocks $k$  (the remaining $12-k$ are random)")
    ax.set_ylabel("test top-1 (%), last epoch")
    ax.set_xticks(range(1, 13)); ax.set_xlim(0.55, 12.9); ax.set_ylim(78.0, 80.8)
    ax.grid(alpha=0.25, lw=0.5)
    leg = ax.legend(loc="upper left", title="procedural blocks placed:", title_fontsize=8.5)
    leg._legend_box.align = "left"
    ax.text(0.985, 0.03, "\n".join(txt) + "\nfilled = 3 seeds,  hollow = 1 seed",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.8, bbox=dict(fc="white", ec="0.85", boxstyle="round,pad=0.4"))
    ax.set_title("Both placements help; the early one helps more at almost every depth", loc="left")

    # ---- paired difference at matched k: the actual statistical claim ----
    ax = axes[1]
    ks = [k for _, k in iser]
    diff = [ms(S, a)[0] - ms(S, f"ftb{k}h")[0] for a, k in iser]
    cols = ["#0173B2" if d > 0 else "#D55E00" for d in diff]
    ax.bar(ks, diff, color=cols, alpha=0.85, width=0.66, zorder=3)
    md = float(np.mean(diff))
    ax.axhline(md, color="0.25", ls="--", lw=1.3, zorder=4)
    ax.axhline(0, color="0.3", lw=1)
    w, pw = stats.wilcoxon(diff)
    ax.text(0.5, 0.965,
            f"mean +{md:.2f},  positive in {sum(1 for d in diff if d > 0)}/11\n"
            f"Wilcoxon signed-rank  p = {pw:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            bbox=dict(fc="white", ec="0.85", boxstyle="round,pad=0.4"))
    ax.set_xlabel("number of procedural blocks $k$")
    ax.set_ylabel("early $-$ late  (percentage points)")
    ax.set_xticks(range(1, 12)); ax.set_ylim(-0.45, 2.05)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Paired advantage of placing them early", loc="left")

    caption(fig, "Both series now span the full sweep k=1..11 (they were partial in an earlier draft) and "
                 "meet at k=12, where the two are the same model — a useful consistency check:\n"
                 "that arm scores 80.09, identical to the separately-trained procedural baseline. The early "
                 "series climbs steeply (Spearman +0.97); the late series drifts up weakly and not "
                 "significantly (+0.43, p=0.19).\n"
                 "The defensible claim is the PAIRED one on the right: at matched k, placing the blocks early "
                 "is worth +0.66 on average and wins at 10 of 11 depths. Late-block arms are single seed "
                 "(ViT-B seed s.d. ~0.3),\nso no individual pair should be read alone.")
    fig.savefig(OUT / "fig4_depth.png")
    plt.close(fig)
    print("fig4_depth.png")


# ======================================================================================
# FIG 5: mechanism -- which property of procedural blocks 0-8 carries the benefit?
# ======================================================================================
def fig5_mechanism(S):
    """Elimination matrix: what each arm preserves in blocks 0-8, next to what it scores.

    Every arm here shares the SAME base: a model whose patch/pos/cls embeddings, head,
    final norm and blocks 9-11 are all randomly initialised.  (pr_load_model deletes
    cls_token / pos_embed / patch_embed.* from any `pr_*` checkpoint, and --skip_norm
    deletes norm.weight/bias, so NO arm inherits procedural embeddings or a procedural
    final norm -- verified in the run logs.)  They differ only in what is put into
    blocks 0-8, which is what the matrix on the left enumerates.
    """
    # 0 = not taken from the checkpoint, 1 = taken, 2 = partial (see caption)
    COLS = ["weight\nSIZES", "weight\nVALUES",
            "LayerNorm gains\n& biases", "learned\nARRANGEMENT",
            "Q,K vs V\nkept apart"]
    # marker codes: 0 none, 1 from checkpoint, 2 norm only, 3 moments only, 4 half,
    #               5 INVERTED (ftb4o pushes the qk/v ratio to the far side of random)
    #
    # The fifth column is the discriminating variable.  The fused attn.qkv is one tensor, so
    # "proc's weight values" can be honoured with q, k and v pooled into a single multiset --
    # which systematically WIDENS v, because proc's v is ~2.2x narrower than its q and k -- or
    # with the [0:2e] and [2e:3e] slices matched independently, which preserves the asymmetry.
    # The last four rows separate the two slices to find out which one carries the benefit.
    ROWS = [
        ("r",           "Random init   (baseline)",                                  [0, 0, 0, 0, 0]),
        ("ftb4o",       "Random, rescaled to proc's WRITE STRENGTH ($\\rho$)",       [0, 0, 0, 0, 5]),
        ("ftbnorm",     "Random, rescaled to proc's weight SIZES",                   [1, 0, 2, 0, 0]),
        ("ftbqm",       "Proc's weight VALUES, randomly shuffled",                   [1, 1, 0, 0, 0]),
        ("ftbqmbias",   "   |- plus its biases",                                     [1, 1, 4, 0, 0]),
        ("ftbqm1dpar",  "   |- plus 1-D params, MOMENTS ONLY",                       [1, 1, 3, 0, 0]),
        ("ftbqmln",     "   |- plus its LayerNorm gains",                            [1, 1, 4, 0, 0]),
        ("ftbqm1d",     "   |- plus ALL 1-D params  (Q,K,V pooled together)",        [1, 1, 1, 0, 0]),
        ("ftbqm1dqk",   "      |- ...with Q,K kept at their own scale",              [1, 1, 1, 0, 4]),
        ("ftbqm1dvo",   "      |- ...with V kept at its own scale",                  [1, 1, 1, 0, 4]),
        ("ftbqm1dv",    "      |- ...with both kept separate",                       [1, 1, 1, 0, 1]),
        ("ftb4e3",      "PROC'S WEIGHTS, shuffled within each slice",                [1, 1, 1, 0, 1]),
        ("ftb3i",       "PROC'S WEIGHTS, intact",                                    [1, 1, 1, 1, 1]),
    ]
    rb = ms(S, "r")[0]
    fig, (axm, axb, axc) = plt.subplots(1, 3, figsize=(19.0, 5.4),
                                        gridspec_kw={"width_ratios": [1.15, 0.75, 1.0]})
    fig.subplots_adjust(wspace=0.30)
    ys = np.arange(len(ROWS))[::-1]

    # ---------------- left: the preserved-property matrix ----------------
    for y, (arm, lab, marks) in zip(ys, ROWS):
        for j, mk in enumerate(marks):
            fc = {0: "#EFEFEF", 1: "#029E73", 2: "#A8DCC9", 3: "#A8DCC9", 4: "#7FC9AE",
                  5: "#D55E00"}[mk]
            axm.scatter(j, y, s=340, marker="s", color=fc,
                        edgecolor="#cfcfcf" if mk == 0 else "none", linewidth=0.8, zorder=3)
            axm.text(j, y, {0: "\u2013", 1: "\u2713", 2: "~", 3: "\u03bc\u03c3", 4: "\u00bd",
                            5: "\u2717"}[mk], ha="center", va="center",
                     fontsize=11, color="white" if mk in (1, 5) else "#7a7a7a",
                     fontweight="bold", zorder=4)
    axm.set_xticks(range(len(COLS)))
    axm.set_xticklabels(COLS, fontsize=8.2)
    axm.xaxis.set_ticks_position("top")
    axm.set_yticks(ys)
    axm.set_yticklabels([l for _, l, _ in ROWS], fontsize=8.8)
    PAL0 = {"r": "#8c8c8c", "ftb4o": "#D55E00", "ftbnorm": "#CA9161", "ftbqm": "#CC78BC",
            "ftbqmbias": "#B07AA1", "ftbqm1dpar": "#9C6BAF", "ftbqmln": "#DE8F05",
            "ftbqm1d": "#E8A33D", "ftbqm1dqk": "#CC78BC", "ftbqm1dvo": "#56B4E9",
            "ftbqm1dv": "#1F9E89", "ftb4e3": "#029E73", "ftb3i": "#0173B2"}
    for t, (arm, _, _) in zip(axm.get_yticklabels(), ROWS):
        t.set_color(PAL0[arm])
        m, _, _ = ms(S, arm)
        t.set_fontweight("bold" if (m - ms(S, "r")[0]) > 1.0 else "normal")
    axm.set_xlim(-0.6, len(COLS) - 0.4); axm.set_ylim(-0.6, len(ROWS) - 0.4)
    for sp in axm.spines.values():
        sp.set_visible(False)
    axm.tick_params(length=0)
    axm.set_title("what blocks 0-8 take from the procedural checkpoint",
                  fontsize=9.6, pad=34, loc="left")

    # the step where the benefit appears.
    # NOTE: keep this index in sync with the axb divider below -- both sit under the last
    # row that stays small (currently ftbqm1dqk, ROWS index 8).
    DIV = 8
    axm.axhline(ys[DIV] - 0.5, color="#B00020", ls="--", lw=1.4, zorder=2)
    # right-aligned inside the matrix: left-aligned it collided with the row labels
    axm.text(len(COLS) - 0.45, ys[DIV] - 0.44,
             "at most +0.78 above          at least +1.37 below",
             color="#B00020", fontsize=8.2, va="bottom", ha="right", fontweight="bold")

    # ---------------- middle (5a): final accuracy ----------------
    rb = ms(S, "r")[0]
    PAL = {"r": "#8c8c8c", "ftb4o": "#D55E00", "ftbnorm": "#CA9161", "ftbqm": "#CC78BC",
           "ftbqmbias": "#B07AA1", "ftbqm1dpar": "#9C6BAF", "ftbqmln": "#DE8F05",
           "ftbqm1d": "#E8A33D", "ftbqm1dqk": "#CC78BC", "ftbqm1dvo": "#56B4E9",
           "ftbqm1dv": "#1F9E89", "ftb4e3": "#029E73", "ftb3i": "#0173B2"}
    for y, (arm, lab, _) in zip(ys, ROWS):
        m, sd, n = ms(S, arm)
        d = m - rb
        axb.barh(y, d, color=PAL[arm], alpha=0.9, height=0.55, zorder=3)
        axb.errorbar(d, y, xerr=sd if n > 1 else None, color="0.25", capsize=3, lw=1.2, zorder=4)
        off = (sd if n > 1 else 0) + 0.07
        axb.text(d + (off if d >= 0 else -off), y, f"{d:+.2f}", va="center",
                 ha="left" if d >= 0 else "right", fontsize=9, fontweight="bold", zorder=5)
    axb.axvline(0, color="0.3", lw=1)
    axb.axhline(ys[DIV] - 0.5, color="#B00020", ls="--", lw=1.4, zorder=2)
    axb.set_yticks(ys); axb.set_yticklabels([])
    axb.set_ylim(-0.6, len(ROWS) - 0.4); axb.set_xlim(-1.15, 3.0)
    axb.set_xlabel("final test top-1 vs random init (pp)")
    axb.grid(axis="x", alpha=0.25, lw=0.5)
    axb.set_title("(a) where each arm ends up", fontsize=9.6, pad=34, loc="left")

    # ---------------- right (5b): test curves over training ----------------
    curves, bands = {}, {}
    for arm, _, _ in ROWS:
        d = load_arm(arm)
        d = d[d["test_acc1"].notna() & (d["test_acc1"] > 0)]
        g = d.groupby("epoch")["test_acc1"]
        curves[arm] = g.mean()
        bands[arm] = g.std() if d["seed"].nunique() > 1 else None
    for arm, _, _ in ROWS:
        m, _, _ = ms(S, arm)
        big = (m - rb) > 1.0
        if bands[arm] is not None:
            axc.fill_between(curves[arm].index, curves[arm] - bands[arm],
                             curves[arm] + bands[arm], color=PAL[arm],
                             alpha=0.18 if big else 0.10, lw=0, zorder=2)
        axc.plot(curves[arm].index, curves[arm].values, color=PAL[arm],
                 lw=2.2 if big else 1.4, alpha=1.0 if big else 0.85, zorder=5 if big else 3)
    axc.axvspan(214, 300, color="#029E73", alpha=0.06, lw=0, zorder=0)
    axc.set_xlabel("epoch"); axc.set_ylabel("test top-1 (%)")
    axc.set_xlim(0, 300); axc.set_ylim(20, 82)
    axc.grid(alpha=0.25, lw=0.5)
    axc.set_title("(b) how they get there   (band = $\\pm$1 s.d. over seeds)",
                  fontsize=9.6, pad=34, loc="left")

    axi = axc.inset_axes([0.36, 0.11, 0.61, 0.48])
    for arm, _, _ in ROWS:
        m, _, _ = ms(S, arm)
        big = (m - rb) > 1.0
        if bands[arm] is not None:
            axi.fill_between(curves[arm].index, curves[arm] - bands[arm],
                             curves[arm] + bands[arm], color=PAL[arm],
                             alpha=0.20 if big else 0.11, lw=0, zorder=2)
        axi.plot(curves[arm].index, curves[arm].values, color=PAL[arm],
                 lw=2.0 if big else 1.2, alpha=1.0 if big else 0.85, zorder=5 if big else 3)
        if big or arm in ("r", "ftbqmln", "ftb4o"):
            axi.annotate(f"{m:.2f}", (299, curves[arm].iloc[-1]), textcoords="offset points",
                         xytext=(4, -2), fontsize=7.0, color=PAL[arm],
                         fontweight="bold", annotation_clip=False)
    axi.axvline(214, color="#029E73", ls=":", lw=1.1)
    axi.set_xlim(150, 300); axi.set_ylim(76.5, 80.9)
    axi.tick_params(labelsize=6.5); axi.grid(alpha=0.25, lw=0.4)
    axi.set_title("final 150 epochs", fontsize=7.2, fontweight="normal")

    fig.suptitle("Which properties of the procedural weights actually transfer?",
                 y=1.10, fontsize=11.3, fontweight="bold")
    caption(fig,
            "Every arm shares one base: randomly initialised patch/pos/cls embeddings, head, final norm "
            "and blocks 9-11. `pr_load_model` deletes the embeddings from any `pr_*` checkpoint and\n"
            "`--skip_norm` deletes the final norm, so no arm inherits either (verified in the run logs).\n"
            "Markers: check = taken from the checkpoint, dash = left at random init, ~ = only the tensor NORM "
            "matched, mu-sigma = a Gaussian matched to the checkpoint's mean and std, 1/2 = half the 1-D "
            "tensors\n(biases only, or LayerNorm only -- the two partition the 72 exactly, 36 + 36), or in the "
            "last column ONE of the two qkv slices. The X marks the one arm that pushes the qk/v ratio to the "
            "FAR side of random (0.35 vs 1.00).\n"
            "RANDOMLY PERMUTED = every entry of the tensor keeps proc's exact value, reordered by a uniform "
            "random permutation, so the value multiset, norm, variance and kurtosis are preserved exactly and "
            "only the arrangement is destroyed.\nThe indented rows are variants off the base above them, NOT a "
            "cumulative chain. n = 3 seeds for every row except the rho-matched row (n=1).\n"
            "\n"
            "WHAT THE LAST FOUR ROWS SETTLE. The fused attn.qkv is a single tensor, so 'proc's weight values' "
            "is ambiguous: pooling q, k and v into one multiset WIDENS v, because proc's v is ~2.2x narrower "
            "than its q and k.\nSplitting the two slices apart isolates which one matters. Matching the v slice "
            "alone gives +1.37; matching the qk slice alone gives +0.50, which is not distinguishable from the "
            "pooled row it is built on (+0.42).\nMatching BOTH gives +1.40 -- no more than v alone. Across all four cells the main effect of matching v is +0.88 and of matching qk is +0.10. So the attention logit scale, which the qk slice sets, contributes nothing here, and "
            "the write magnitude set by v carries the effect.\n"
            "\n"
            "WHAT ORDERS THEM. Everything that helps does so by lowering ONE number: the attention write magnitude at init, gamma*||W_v||*||W_proj||/d. The LayerNorm gains act only as a multiplier on it\n"
            "(2.237 x 0.384 = 0.859); keeping V at its own scale lowers it again (0.869 x 28.8/50.9 = 0.492). Grouped by that scalar the arms fall into three bands, +1.69 / +0.51 / +0.07, F = 64.9, p = 5e-11, R^2 = 0.83.\n"
            "\n"
            "HOW BIG IS THE RESIDUAL? The bottom four rows span +1.37 to +2.08 and NO pair among them differs "
            "significantly (Welch p = 0.073 to 0.50); pooled they are 79.80 +/- 0.42 (n = 11), i.e. +1.72, and "
            "they separate from the\nrows above at p < 1e-4. The two green rows are also constructed to be "
            "statistically equivalent -- a rank map onto an i.i.d. tensor IS a uniform permutation -- so their "
            "0.68 gap is most likely seed noise, not a missing mechanism.\nThat reading is provisional: the "
            "ordering within the group is monotone, which n = 3 cannot resolve either way at this effect size.",
            color="#333333")
    fig.savefig(OUT / "fig5_mechanism.png")
    plt.close(fig)
    print("fig5_mechanism.png")


# ======================================================================================
# FIG 6: training stability
# ======================================================================================
def fig6_stability(S):
    arms = ["r", "ftbrho", "p", "ftbcomp11"]
    D = {a: load_arm(a) for a in arms}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for a in arms:
        d = D[a]
        for sd, g in d.groupby("seed"):
            ax.plot(g["epoch"], g["train_loss"], color=C[a], lw=1.0, alpha=0.85,
                    label=LABEL[a].split(":")[0].split(" +")[0] if sd == "s0" else None)
    ax.set_xlim(30, 130); ax.set_ylim(3.2, 7.1)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
    ax.set_title("Every seed, epochs 30-130 (loss spikes)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    # spikes against the number of INTACT procedural blocks at the start of the network
    iser = [("ftb11i", 1), ("ftb10i", 2), ("ftb9i", 3), ("ftb8i", 4), ("ftb7i", 5),
            ("ftb6i", 6), ("ftb5i", 7), ("ftb4i", 8), ("ftb3i", 9), ("ftb2i", 10),
            ("ftb1i", 11), ("p", 12)]
    xs = [k for _, k in iser]
    ys = [ms(S, a, "spikes")[0] for a, _ in iser]
    ax.plot(xs, ys, "-o", color="#0173B2", lw=1.8, ms=6, zorder=3,
            label="procedural blocks intact")
    for a, mk, col, lab in [("ftb4e3", "D", "#B00020", "same 9 blocks, SHUFFLED"),
                            ("ftb11is", "D", "#B00020", None)]:
        k = 9 if a == "ftb4e3" else 1
        ax.scatter([k], [ms(S, a, "spikes")[0]], marker=mk, s=70, color=col,
                   zorder=5, label=lab, edgecolor="white", linewidth=0.8)
    ax.scatter([0], [ms(S, "r", "spikes")[0]], marker="o", s=55, color=C["r"],
               zorder=5, label="random / recipe (0 blocks)")
    ax.scatter([0], [ms(S, "ftbrho", "spikes")[0]], marker="o", s=55, color=C["ftbrho"], zorder=6)
    ax.annotate("shuffle removes\nthe instability", (9, 0), textcoords="offset points",
                xytext=(-6, 26), ha="right", fontsize=7.8, color="#B00020", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#B00020", lw=1.1))
    ax.set_xlabel("number of intact procedural blocks (from block 0)")
    ax.set_ylabel("loss spikes per run")
    ax.set_ylim(-0.35, 5.9); ax.set_xlim(-0.9, 13.0)
    ax.set_title("Instability tracks intact structure,\nnot the weight values")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.5)

    fig.suptitle("Procedural weights destabilise training; shuffling them — or using the recipe — does not",
                 y=1.03, fontsize=11.5, fontweight="bold")
    caption(fig, "A spike is an epoch-over-epoch train-loss jump > 0.25. Right panel, controlled pair: 9 "
                 "procedural blocks INTACT (ftb3i) spike 4.0 times per run; the SAME 9 blocks\nwith weights "
                 "shuffled (ftb4e3) — identical value multiset, identical norms — never spike, and score the "
                 "SAME: 80.16 +/- 0.10 vs 79.99 +/- 0.36 (Welch p = 0.50), and also\nindistinguishable from "
                 "full procedural init (80.09, p = 0.50). So at this depth the instability is removable at no "
                 "cost in accuracy. Random init and the recipe never\nspike either (largest jump 0.01 over 6 "
                 "runs). Runs do recover from spikes, but this is a practical cost of the checkpoint. "
                 "See fig7: at ONE block the shuffle is not free.")
    fig.savefig(OUT / "fig6_stability.png")
    plt.close(fig)
    print("fig6_stability.png")


# ======================================================================================
# FIG 7: structure vs statistics at a single block
# ======================================================================================
def fig7_single_block(S):
    """Does shuffling cost anything? Depends entirely on how many blocks are procedural."""
    PANELS = [
        ("ONE procedural block (block 0)",
         [("r", "random\ninit", C["r"]),
          ("ftb11is", "block 0 proc,\nSHUFFLED", C["neg"]),
          ("ftb11i", "block 0 proc,\nintact", C["pos"])],
         ("ftb11is", "ftb11i"), (77.4, 79.35)),
        ("NINE procedural blocks (0-8)",
         [("r", "random\ninit", C["r"]),
          ("ftb4e3", "blocks 0-8 proc,\nSHUFFLED", C["neg"]),
          ("ftb3i", "blocks 0-8 proc,\nintact", C["pos"])],
         ("ftb4e3", "ftb3i"), (77.4, 80.95)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, (title, rows, (sh, it), ylim) in zip(axes, PANELS):
        base = ylim[0]
        for i, (a, lab, col) in enumerate(rows):
            m, s, n = ms(S, a)
            pts = S[S["arm"] == a]["acc"].values
            ax.bar(i, m - base, bottom=base, color=col, alpha=0.85, width=0.55, zorder=3)
            ax.scatter([i] * len(pts), pts, s=20, color="0.15", zorder=5)
            ax.errorbar(i, m, yerr=s, color="0.15", capsize=4, lw=1.4, zorder=4)
            ax.text(i, m + s + 0.05, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([l for _, l, _ in rows], fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel("test top-1 (%), last epoch")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_title(title, loc="left")

        msh, ssh, _ = ms(S, sh); mit, sit, _ = ms(S, it)
        t, p = stats.ttest_ind(S[S.arm == sh]["acc"], S[S.arm == it]["acc"], equal_var=False)
        top = max(msh + ssh, mit + sit)
        y = top + (ylim[1] - ylim[0]) * 0.11
        ax.annotate("", xy=(2, y), xytext=(1, y),
                    arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.3))
        verdict = "n.s. — shuffling is FREE" if p > 0.05 else f"{abs(t):.1f}$\\sigma$ — shuffling COSTS"
        ax.text(1.5, y + (ylim[1] - ylim[0]) * 0.022,
                f"{msh - mit:+.2f}   (p = {p:.2f})\n{verdict}", ha="center", fontsize=8.6,
                fontweight="bold", color="#B00020" if p <= 0.05 else "#1a7a4c")
        # spikes annotation
        ax.text(0.015, 0.965, f"loss spikes per run\n  shuffled: {ms(S, sh, 'spikes')[0]:.1f}\n"
                              f"  intact:   {ms(S, it, 'spikes')[0]:.1f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color="0.35",
                bbox=dict(fc="white", ec="0.88", boxstyle="round,pad=0.35"))

    fig.suptitle("Does destroying the arrangement cost anything? Only when there is one procedural block",
                 y=1.02, fontsize=11.5, fontweight="bold")
    caption(fig, "Shuffling (`randperm` per tensor) preserves every per-tensor statistic exactly — norm, "
                 "variance, kurtosis, the whole histogram — and destroys only the arrangement. n=3 seeds "
                 "per arm, Welch t-test.\n"
                 "At one block the arrangement is worth a full point. At nine blocks it is worth nothing "
                 "measurable (and the shuffled arm is also indistinguishable from full procedural init, "
                 "80.16 vs 80.09, p=0.50)\n"
                 "while removing all four loss spikes per run. This pair is the sharpest open puzzle in the "
                 "study: whatever the arrangement contributes at one block appears to be recoverable from "
                 "the other eight.")
    fig.savefig(OUT / "fig7_shuffle.png")
    plt.close(fig)
    print("fig7_shuffle.png")


FIGS = {1: fig1_loss_curves, 2: fig2_generalisation, 3: fig3_headline, 4: fig4_depth,
        5: fig5_mechanism, 6: fig6_stability, 7: fig7_single_block}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    setup()
    S = summary()
    S.to_csv(OUT / "per_seed_summary.csv", index=False)
    agg = (S.groupby("arm")
             .agg(n=("acc", "size"), acc=("acc", "mean"), sd=("acc", "std"),
                  train_loss=("train_loss", "mean"), overfit=("overfit", "mean"),
                  spikes=("spikes", "mean"))
             .round(3).sort_values("acc"))
    agg.to_csv(OUT / "arm_summary.csv")
    print(agg.to_string(), "\n")
    for k in ([int(x) for x in a.only.split(",") if x] or sorted(FIGS)):
        FIGS[k](S)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
