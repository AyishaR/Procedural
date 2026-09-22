"""fig22 / fig23: zero-shot effective robustness of the final checkpoints (own head, no probes).

x = own ImageNet val top-1, y = OOD top-1 with the same head, both on a probit axis (Taori et al. 2020).  Stage-1
reference line = slope 1 in probit space through random's mean (the slope will come from the ftblrm trajectory in
stage 2; over the 5-point ID span of the finals the slope changes offsets by < 0.3 pt); the OLS line over all
finals is drawn for comparison.  Effective robustness of a model = its vertical distance from the reference line at
its own x, reported in accuracy points (and probit units in the CSV).  Writes fig22 (V2 / R / C-mean scatter +
per-arm offsets), fig23 (per-corruption-group x severity and per-corruption offsets), ood_offsets.csv,
ood_models.csv and ood_summary.md.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from probes import ood_data
from probes import registry

MAIN_SETS = [("imagenet_v2", "ImageNet-V2 (10k)"), ("imagenet_r", "ImageNet-R (30k, 200-class mask)"), ("imagenet_c", "ImageNet-C mean (15 x 5, 5k/cell)")]
N1 = ("ftb4o", "ftbrhos", "ftblrm")       # single-seed arms (dashed / hollow)


def probit(p):
    return norm.ppf(np.clip(np.asarray(p, dtype=float), 0.05, 99.95) / 100.0)


def inv_probit(z):
    return 100.0 * norm.cdf(np.asarray(z, dtype=float))


def load():
    recs = []
    for f in sorted((registry.RES / "ood").glob("*.json")):
        if "__limit" in f.name:
            continue
        r = json.load(open(f))
        if r.get("complete"):
            recs.append(r)
    return recs


def readouts(r):
    """Flat {key: top-1} for one model: the three main sets, C per group / severity / corruption / cell."""
    s = r["sets"]
    c = s["imagenet_c"]
    out = {"imnet_val": s["imnet_val"]["top1"], "imagenet_v2": s["imagenet_v2"]["top1"], "imagenet_r": s["imagenet_r"]["top1_masked"],
           "imagenet_r_unmasked": s["imagenet_r"]["top1"], "imagenet_c": c["top1"], "imnet_val_c_draw": s["imnet_val_c_draw"]["top1"]}
    out.update({f"c_group/{g}": v for g, v in c["per_group"].items()})
    out.update({f"c_sev/{k}": v for k, v in c["per_severity"].items()})
    out.update({f"c_corr/{k}": v for k, v in c["per_corruption"].items()})
    out.update({f"c_cell/{k}/{sv}": c["cells"][k][sv]["top1"] for k in c["cells"] for sv in c["cells"][k]})
    return out


class RefLine:
    """Slope-1 probit line through random's mean (x0, y0) for one readout, plus the OLS over all finals."""

    def __init__(self, x_all, y_all, x_r, y_r):
        self.x0, self.y0 = probit(np.mean(x_r)), probit(np.mean(y_r))
        zx, zy = probit(x_all), probit(y_all)
        self.ols = np.polyfit(zx, zy, 1) if len(zx) > 2 else (np.nan, np.nan)

    def predict(self, x):
        return inv_probit(probit(x) - self.x0 + self.y0)

    def offset_pts(self, x, y):
        return np.asarray(y, dtype=float) - self.predict(x)

    def offset_probit(self, x, y):
        return probit(y) - (probit(x) - self.x0 + self.y0)


def analyse(recs):
    """Everything the figures, tables and the results page share: per-model readouts, reference lines, arm offsets."""
    R = {r["model"]: readouts(r) for r in recs}
    arm_of = {r["model"]: r["arm"] for r in recs}
    names = [m["name"] for m in registry.MODELS if m["name"] in R]
    keys = [k for k in R[names[0]] if k != "imnet_val"]
    x_all = np.array([R[n]["imnet_val"] for n in names])
    r_names = [n for n in names if arm_of[n] == "r"]
    if not r_names:
        raise SystemExit("random's finals are missing; the reference line needs them")
    lines = {k: RefLine(x_all, [R[n][k] for n in names], [R[n]["imnet_val"] for n in r_names], [R[n][k] for n in r_names]) for k in keys}
    arms = [a for a in registry.ARM_ORDER if any(arm_of[n] == a for n in names)]
    per_arm = {a: [n for n in names if arm_of[n] == a] for a in arms}

    def arm_off(a, k):
        """(mean, sd, n) of the arm's offsets from random's line on readout k, in top-1 points."""
        v = np.array([lines[k].offset_pts(R[n]["imnet_val"], R[n][k]) for n in per_arm[a]])
        return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0), len(v)

    def group_sev_off(a, g, sv):
        """Per-seed offsets of the arm on one (group, severity): mean over the group's corruptions of the cell offsets."""
        return np.array([np.mean([lines[f"c_cell/{c}/{sv}"].offset_pts(R[n]["imnet_val"], R[n][f"c_cell/{c}/{sv}"]) for c in ood_data.C_GROUPS[g]]) for n in per_arm[a]])

    return dict(R=R, arm_of=arm_of, names=names, keys=keys, x_all=x_all, lines=lines, arms=arms, per_arm=per_arm, arm_off=arm_off, group_sev_off=group_sev_off)


def main():
    registry.ensure_dirs()
    recs = load()
    if not recs:
        raise SystemExit("no complete results under results/ood")
    A = analyse(recs)
    R, arm_of, names, keys, x_all, lines, arms, per_arm, arm_off = (A[k] for k in ("R", "arm_of", "names", "keys", "x_all", "lines", "arms", "per_arm", "arm_off"))

    # ---------------------------------------------------------------- fig22: scatter + per-arm offsets
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, (k, lab) in enumerate(MAIN_SETS):
        ax = axes[0, j]
        L = lines[k]
        xs = np.linspace(x_all.min() - 1.5, x_all.max() + 1.0, 50)
        ax.plot(probit(xs), probit(L.predict(xs)), color="0.15", lw=1.4, label="random's line (slope 1, probit)")
        ax.plot(probit(xs), np.polyval(L.ols, probit(xs)), color="0.5", lw=1, ls=":", label=f"OLS over {len(names)} finals (slope {L.ols[0]:.2f})")
        for a in arms:
            xa = [R[n]["imnet_val"] for n in per_arm[a]]; ya = [R[n][k] for n in per_arm[a]]
            c = registry.ARM_COLOR[a]
            ax.scatter(probit(xa), probit(ya), s=34, color=c if a not in N1 else "white", edgecolor=c, linewidth=1.2, zorder=4, label=f"{a} (n={len(xa)})")
            if len(xa) > 1:
                ax.scatter(probit(np.mean(xa)), probit(np.mean(ya)), s=110, marker="+", color=c, linewidth=1.4, zorder=5)
        xt = [74, 76, 78, 80, 82]; ax.set_xticks(probit(xt)); ax.set_xticklabels(xt)
        lo, hi = min(R[n][k] for n in names), max(R[n][k] for n in names)
        yt = [v for v in [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] if lo - 3 <= v <= hi + 3]
        ax.set_yticks(probit(yt)); ax.set_yticklabels(yt)
        ax.set_xlim(probit(x_all.min() - 1.2), probit(x_all.max() + 0.8))
        ax.set_xlabel("own ImageNet val top-1 (%), probit axis", fontsize=8); ax.set_ylabel(f"{lab} top-1 (%), probit axis", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax = axes[1, j]
        for i, a in enumerate(arms):
            m, s, n = arm_off(a, k)
            c = registry.ARM_COLOR[a]
            ax.errorbar(i, m, yerr=s if n > 1 else 0, fmt="o" if a not in N1 else "o", color=c, mfc=c if a not in N1 else "white", mec=c, ms=6, capsize=3, lw=1.2)
            ax.scatter([i] * n, [lines[k].offset_pts(R[nn]["imnet_val"], R[nn][k]) for nn in per_arm[a]], s=9, color=c, alpha=0.6, zorder=3)
            ax.annotate(f"{m:+.1f}", (i, m + (s if n > 1 else 0)), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color="0.2")
        ax.axhline(0, color="0.15", lw=1)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("offset from random's line (top-1 pts)", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3, axis="y")
        ax.set_title(lab, fontsize=8)
    fig.suptitle("fig22: zero-shot effective robustness of the 21 final checkpoints (own head) -- ID = ImageNet val, y = OOD top-1; "
                 "random's finals define the reference line (+ = arm mean; hollow = n=1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(registry.FIGS / "fig22_ood_effective_robustness.png", dpi=300); plt.close(fig)

    # ---------------------------------------------------------------- fig23: ImageNet-C groups x severity, per corruption
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    sev = ood_data.SEVERITIES
    for j, (g, cs) in enumerate(ood_data.C_GROUPS.items()):
        ax = fig.add_subplot(gs[0, j])
        for a in arms:
            ms, ss = [], []
            for sv in sev:
                v = A["group_sev_off"](a, g, sv)
                ms.append(v.mean()); ss.append(v.std(ddof=1) if len(v) > 1 else 0.0)
            ms, ss = np.array(ms), np.array(ss)
            col = registry.ARM_COLOR[a]
            ax.plot(sev, ms, color=col, lw=2.2 if a == "r" else 1.5, ls="--" if a in N1 else "-", marker="o", ms=3.5, label=f"{a} (n={len(per_arm[a])})")
            if len(per_arm[a]) > 1:
                ax.fill_between(sev, ms - ss, ms + ss, color=col, alpha=0.12, lw=0)
        ax.axhline(0, color="0.15", lw=1)
        ax.set_title(f"{g}: {', '.join(cs)}", fontsize=7.5)
        ax.set_xticks(sev); ax.set_xlabel("severity", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("offset from random's line (top-1 pts)", fontsize=8); ax.legend(fontsize=6, ncol=2)
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(len(ood_data.CORRUPTIONS))
    for i, a in enumerate(arms):
        ms, ss = [], []
        for c in ood_data.CORRUPTIONS:
            m, s, n = arm_off(a, f"c_corr/{c}")
            ms.append(m); ss.append(s)
        col = registry.ARM_COLOR[a]
        ax.errorbar(xs + (i - len(arms) / 2) * 0.075, ms, yerr=ss, fmt="o", color=col, mfc=col if a not in N1 else "white", mec=col, ms=4, capsize=2, lw=1, label=a)
    for b in np.cumsum([len(v) for v in ood_data.C_GROUPS.values()])[:-1]:
        ax.axvline(b - 0.5, color="0.7", lw=0.8)
    ax.axhline(0, color="0.15", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(ood_data.CORRUPTIONS, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("offset from random's line, mean over severities (top-1 pts)", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6.5, ncol=len(arms))
    fig.suptitle("fig23: ImageNet-C effective robustness by corruption group x severity (top) and per corruption (bottom); "
                 "offsets from random's slope-1 probit line at the model's own ImageNet accuracy", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(registry.FIGS / "fig23_ood_imagenet_c.png", dpi=300); plt.close(fig)

    # ---------------------------------------------------------------- tables
    with open(registry.FIGS / "ood_models.csv", "w", newline="") as f:
        cols = ["model", "arm", "seed", "imnet_val"] + keys
        w = csv.writer(f); w.writerow(cols + [f"off_{k}" for k in ("imagenet_v2", "imagenet_r", "imagenet_c")])
        for n in names:
            w.writerow([n, arm_of[n], registry.BY_NAME[n]["seed"], R[n]["imnet_val"]] + [round(R[n][k], 3) for k in keys]
                       + [round(float(lines[k].offset_pts(R[n]["imnet_val"], R[n][k])), 3) for k in ("imagenet_v2", "imagenet_r", "imagenet_c")])
    rows = []
    for a in arms:
        xa = np.array([R[n]["imnet_val"] for n in per_arm[a]])
        for k in keys:
            if k.startswith("c_cell/"):
                continue
            ya = np.array([R[n][k] for n in per_arm[a]])
            m, s, n = arm_off(a, k)
            zp = np.mean([lines[k].offset_probit(R[nn]["imnet_val"], R[nn][k]) for nn in per_arm[a]])
            rows.append(dict(arm=a, n=n, set=k, id_mean=round(xa.mean(), 3), ood_mean=round(ya.mean(), 3), ood_sd=round(ya.std(ddof=1) if n > 1 else 0.0, 3),
                             offset_pts=round(float(m), 3), offset_pts_sd=round(float(s), 3), offset_probit=round(float(zp), 4),
                             ols_slope=round(float(lines[k].ols[0]), 3)))
    with open(registry.FIGS / "ood_offsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| arm | n | ImageNet val | V2 | R (200) | C mean | noise | blur | weather | digital | sev 1 | sev 5 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        xa = np.mean([R[n]["imnet_val"] for n in per_arm[a]])
        cells = []
        for k in ["imagenet_v2", "imagenet_r", "imagenet_c", "c_group/noise", "c_group/blur", "c_group/weather", "c_group/digital", "c_sev/1", "c_sev/5"]:
            m, s, n = arm_off(a, k)
            ya = np.mean([R[n][k] for n in per_arm[a]])
            cells.append(f"{ya:.1f} ({m:+.1f})")
        md.append(f"| {a} | {len(per_arm[a])} | {xa:.2f} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("cell = arm-mean OOD top-1 (mean per-model offset from random's slope-1 probit line, each at that model's own ImageNet accuracy, in top-1 points); "
              "OLS slopes over the finals: " + ", ".join(f"{k} {lines[k].ols[0]:.2f}" for k, _ in MAIN_SETS))
    (registry.FIGS / "ood_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    diffs = [r["sets"]["imnet_val"]["diff_vs_log"] for r in recs if r["sets"]["imnet_val"].get("diff_vs_log") is not None]
    chk = f"max |diff| {max(abs(d) for d in diffs):.3f} over {len(diffs)} models" if diffs else "log comparison unavailable"
    print(f"\nval top-1 vs log.txt: {chk}; wrote fig22, fig23, ood_*.csv, ood_summary.md to {registry.FIGS}")


if __name__ == "__main__":
    main()
