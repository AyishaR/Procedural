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

MAIN_SETS = [("imagenet_v2", "ImageNet-V2 (10k)"), ("imagenet_r", "ImageNet-R (30k, 200-class mask)"),
             ("imagenet_sketch", "ImageNet-Sketch (50k)"), ("imagenet_a", "ImageNet-A (7.5k, 200-class mask)"),
             ("objectnet", "ObjectNet (18.6k, 113 classes)"), ("imagenet_c", "ImageNet-C mean (15 x 5, 5k/cell)")]
N1 = ("ftb4o", "ftbrhos", "ftblrm")       # single-seed arms (dashed / hollow)


def probit(p):
    return norm.ppf(np.clip(np.asarray(p, dtype=float), 0.05, 99.95) / 100.0)


def inv_probit(z):
    return 100.0 * norm.cdf(np.asarray(z, dtype=float))


def load():
    """Every stored model record that has the ImageNet val anchor; sets missing for some model are dropped later.

    `complete` means "every set in ood_data.SETS is stored", which is False while a set is still being added,
    so it must not be the filter here.
    """
    recs = []
    for f in sorted((registry.RES / "ood").glob("*.json")):
        if "__limit" in f.name:
            continue
        r = json.load(open(f))
        if "imnet_val" in r.get("sets", {}):
            recs.append(r)
    return recs


# key -> (record name, field); a set missing from a model's JSON is simply absent from its readouts
PLAIN = {"imnet_val": ("imnet_val", "top1"), "imagenet_v2": ("imagenet_v2", "top1"),
         "imagenet_r": ("imagenet_r", "top1_masked"), "imagenet_r_unmasked": ("imagenet_r", "top1"),
         "imagenet_sketch": ("imagenet_sketch", "top1"), "imagenet_a": ("imagenet_a", "top1_masked"),
         "imagenet_a_unmasked": ("imagenet_a", "top1"), "objectnet": ("objectnet", "top1_masked"),
         "objectnet_unmasked": ("objectnet", "top1"), "imnet_val_c_draw": ("imnet_val_c_draw", "top1")}


def readouts(r):
    """Flat {key: top-1} for one model: every stored set, plus C per group / severity / corruption / cell."""
    s = r["sets"]
    out = {k: s[name][field] for k, (name, field) in PLAIN.items() if name in s and field in s[name]}
    c = s.get("imagenet_c")
    if c:
        out["imagenet_c"] = c["top1"]
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

    def predict_ols(self, x):
        """Same anchor, but the slope fitted over all finals -- the sensitivity check when that slope is not 1."""
        return inv_probit(self.ols[0] * (probit(x) - self.x0) + self.y0)

    def offset_pts(self, x, y):
        return np.asarray(y, dtype=float) - self.predict(x)

    def offset_probit(self, x, y):
        return probit(y) - (probit(x) - self.x0 + self.y0)


def analyse(recs):
    """Everything the figures, tables and the results page share: per-model readouts, reference lines, arm offsets."""
    R = {r["model"]: readouts(r) for r in recs}
    arm_of = {r["model"]: r["arm"] for r in recs}
    names = [m["name"] for m in registry.MODELS if m["name"] in R]
    common = set(R[names[0]])
    for n in names[1:]:
        common &= set(R[n])
    dropped = sorted((set(R[names[0]]) | set().union(*(set(R[n]) for n in names))) - common)
    if dropped:
        print(f"[fig_ood] {len(dropped)} readouts are missing for at least one model and are skipped: {dropped[:6]}")
    keys = [k for k in R[names[0]] if k != "imnet_val" and k in common]
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
    sets = [(k, lab) for k, lab in MAIN_SETS if k in lines]
    nr = int(np.ceil(len(sets) / 3))
    fig, axf = plt.subplots(nr, 3, figsize=(14, 4.3 * nr), squeeze=False)
    fig2, axo = plt.subplots(nr, 3, figsize=(14, 3.4 * nr), squeeze=False)
    for j, (k, lab) in enumerate(sets):
        ax = axf[j // 3][j % 3]
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
        ax = axo[j // 3][j % 3]
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
    for j in range(len(sets), nr * 3):
        axf[j // 3][j % 3].axis("off"); axo[j // 3][j % 3].axis("off")
    fig.suptitle(f"fig22: zero-shot effective robustness of the {len(names)} final checkpoints (own head) -- ID = ImageNet val, "
                 "y = OOD top-1; random's finals define the reference line (+ = arm mean; hollow = n=1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.04 / nr)); fig.savefig(registry.FIGS / "fig22_ood_effective_robustness.png", dpi=300); plt.close(fig)
    fig2.suptitle("fig22b: effective robustness per arm -- offset from random's slope-1 probit line at the arm's own ImageNet accuracy", fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 1 - 0.05 / nr)); fig2.savefig(registry.FIGS / "fig22b_ood_offsets.png", dpi=300); plt.close(fig2)

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
        w = csv.writer(f); w.writerow(cols + [f"off_{k}" for k, _ in MAIN_SETS if k in lines])
        for n in names:
            w.writerow([n, arm_of[n], registry.BY_NAME[n]["seed"], R[n]["imnet_val"]] + [round(R[n][k], 3) for k in keys]
                       + [round(float(lines[k].offset_pts(R[n]["imnet_val"], R[n][k])), 3) for k, _ in MAIN_SETS if k in lines])
    rows = []
    for a in arms:
        xa = np.array([R[n]["imnet_val"] for n in per_arm[a]])
        for k in keys:
            if k.startswith("c_cell/"):
                continue
            ya = np.array([R[n][k] for n in per_arm[a]])
            m, s, n = arm_off(a, k)
            zp = np.mean([lines[k].offset_probit(R[nn]["imnet_val"], R[nn][k]) for nn in per_arm[a]])
            mo = float(np.mean([R[nn][k] - lines[k].predict_ols(R[nn]["imnet_val"]) for nn in per_arm[a]]))
            rnoise = float(np.std([lines[k].offset_pts(R[nn]["imnet_val"], R[nn][k]) for nn in per_arm["r"]], ddof=1)) if len(per_arm["r"]) > 1 else 0.0
            rows.append(dict(arm=a, n=n, set=k, id_mean=round(xa.mean(), 3), ood_mean=round(ya.mean(), 3), ood_sd=round(ya.std(ddof=1) if n > 1 else 0.0, 3),
                             offset_pts=round(float(m), 3), offset_pts_sd=round(float(s), 3), offset_probit=round(float(zp), 4),
                             offset_pts_ols_slope=round(mo, 3), ols_slope=round(float(lines[k].ols[0]), 3),
                             random_seed_sd=round(rnoise, 3)))
    with open(registry.FIGS / "ood_offsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    shown = [k for k, _ in MAIN_SETS if k in lines] + ["c_group/noise", "c_group/blur", "c_group/weather", "c_group/digital", "c_sev/1", "c_sev/5"]
    shown = [k for k in shown if k in lines]
    hdr = {"imagenet_v2": "V2", "imagenet_r": "R (200)", "imagenet_sketch": "Sketch", "imagenet_a": "A (200)",
           "objectnet": "ObjectNet", "imagenet_c": "C mean"}
    md = ["| arm | n | ImageNet val | " + " | ".join(hdr.get(k, k.split("/")[-1]) for k in shown) + " |",
          "|---" * (3 + len(shown)) + "|"]
    for a in arms:
        xa = np.mean([R[n]["imnet_val"] for n in per_arm[a]])
        cells = []
        for k in shown:
            m, s, n = arm_off(a, k)
            ya = np.mean([R[n][k] for n in per_arm[a]])
            cells.append(f"{ya:.1f} ({m:+.1f})")
        md.append(f"| {a} | {len(per_arm[a])} | {xa:.2f} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("cell = arm-mean OOD top-1 (mean per-model offset from random's slope-1 probit line, each at that model's own ImageNet accuracy, in top-1 points).")
    md.append("")
    md.append("Sensitivity of the offset to the reference slope (a slope far from 1 makes the slope-1 offset unreliable "
              "for the arms furthest from random's accuracy):")
    md.append("")
    md.append("| set | OLS slope | random's seed sd | " + " | ".join(f"{a}: slope 1 / OLS" for a in arms if a != "r") + " |")
    md.append("|---" * (4 + len(arms) - 2) + "|")
    for k, _ in MAIN_SETS:
        if k not in lines:
            continue
        rn = np.std([lines[k].offset_pts(R[n]["imnet_val"], R[n][k]) for n in per_arm["r"]], ddof=1) if len(per_arm["r"]) > 1 else 0.0
        cells = []
        for a in arms:
            if a == "r":
                continue
            o1 = np.mean([lines[k].offset_pts(R[n]["imnet_val"], R[n][k]) for n in per_arm[a]])
            o2 = np.mean([R[n][k] - lines[k].predict_ols(R[n]["imnet_val"]) for n in per_arm[a]])
            cells.append(f"{o1:+.2f} / {o2:+.2f}")
        md.append(f"| {k} | {lines[k].ols[0]:.2f} | {rn:.2f} | " + " | ".join(cells) + " |")
    (registry.FIGS / "ood_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    diffs = [r["sets"]["imnet_val"]["diff_vs_log"] for r in recs if r["sets"]["imnet_val"].get("diff_vs_log") is not None]
    chk = f"max |diff| {max(abs(d) for d in diffs):.3f} over {len(diffs)} models" if diffs else "log comparison unavailable"
    print(f"\nval top-1 vs log.txt: {chk}; wrote fig22, fig22b, fig23, ood_*.csv, ood_summary.md to {registry.FIGS}")


if __name__ == "__main__":
    main()
