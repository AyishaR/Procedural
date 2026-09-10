"""fig19: logit lens vs tuned lens at epoch 299 (ImageNet val, ImageNet-R) per arm, and lens_summary.csv."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probes import registry
from probes.figutils import arm_stats, by_arm, load_json_dir, plot_arms, style_depth_axis


def main():
    registry.ensure_dirs()
    recs = load_json_dir("lens")
    probes = [r for r in load_json_dir("probes") if r["dataset"] == "imnet"]
    ll = by_arm(recs, lambda r: r["logit_lens"]["top1"])
    tl = by_arm(recs, lambda r: r["tuned_lens"]["top1"])
    tk = by_arm(recs, lambda r: r["tuned_lens"]["kl"])
    lk = by_arm(recs, lambda r: r["logit_lens"]["kl"])
    ag = by_arm(recs, lambda r: r["tuned_lens"]["agree"])
    rl = by_arm(recs, lambda r: r.get("imagenet_r", {}).get("logit_lens", {}).get("top1"))
    rt = by_arm(recs, lambda r: r.get("imagenet_r", {}).get("tuned_lens", {}).get("top1"))
    pc = by_arm(probes, lambda r: r["ftypes"].get("cls", {}).get("test_acc"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    ax = axes[0, 0]; plot_arms(ax, ll); ax.set_title("logit lens: top-1 vs truth (ImageNet val)", fontsize=9); style_depth_axis(ax); ax.legend(fontsize=6)
    ax = axes[0, 1]; plot_arms(ax, tl); plot_arms(ax, pc, band=False, ls="--", lw=1.0, label_n=False, alpha=0.7)
    ax.set_title("tuned lens: top-1 (solid) vs 1000-class CLS probe (dashed)", fontsize=9); style_depth_axis(ax)
    ax = axes[0, 2]; plot_arms(ax, ag, label_n=False); ax.set_title("tuned lens: agreement with final prediction (%)", fontsize=9); style_depth_axis(ax)
    ax = axes[1, 0]
    plot_arms(ax, {a: {s: np.maximum(v, 1e-4) for s, v in d.items()} for a, d in tk.items()}, label_n=False)
    plot_arms(ax, {a: {s: np.maximum(v, 1e-4) for s, v in d.items()} for a, d in lk.items()}, band=False, ls=":", lw=1.0, label_n=False, alpha=0.7)
    ax.set_yscale("log"); ax.set_title("KL(final || lens): tuned (solid), logit (dotted)", fontsize=8); style_depth_axis(ax)
    ax = axes[1, 1]; plot_arms(ax, rt); plot_arms(ax, rl, band=False, ls=":", lw=1.0, label_n=False, alpha=0.7)
    ax.set_title("ImageNet-R top-1 (200-class mask): tuned (solid), logit (dotted)", fontsize=8); style_depth_axis(ax)
    ax = axes[1, 2]
    for arm in registry.ARM_ORDER:
        hs = [r["tuned_lens"]["pred_depth_hist"] for r in recs if r["arm"] == arm]
        if hs:
            h = np.mean(hs, 0); h = h / h.sum() * 100
            ax.plot(np.arange(13), h, color=registry.ARM_COLOR[arm], lw=1.4, label=f"{arm} (n={len(hs)})")
    ax.set_title("tuned lens: prediction depth (% of images)", fontsize=9); style_depth_axis(ax); ax.legend(fontsize=6)
    fig.suptitle("fig19: logit lens and tuned lens at epoch 299 (CLS features; tuned lens = identity + affine into the frozen norm/head, trained with KL to the final distribution)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(registry.FIGS / "fig19_lens.png", dpi=300); plt.close(fig)

    rows = []
    for arm in registry.ARM_ORDER:
        rs = [r for r in recs if r["arm"] == arm]
        if not rs:
            continue
        lm, _, n = arm_stats(ll[arm]); tm, _, _ = arm_stats(tl[arm])
        first90_l = int(np.argmax(lm >= 0.9 * lm[12])); first90_t = int(np.argmax(tm >= 0.9 * tm[12]))
        rows.append(dict(arm=arm, n=n, logit_top1_b8=round(float(lm[9]), 2), logit_top1_b9=round(float(lm[10]), 2), logit_top1_b10=round(float(lm[11]), 2), final=round(float(lm[12]), 2),
                         tuned_top1_b8=round(float(tm[9]), 2), tuned_top1_b9=round(float(tm[10]), 2), tuned_top1_b10=round(float(tm[11]), 2),
                         first_depth_logit_90pct=first90_l, first_depth_tuned_90pct=first90_t,
                         pred_depth_mean=round(float(np.mean([r["tuned_lens"]["pred_depth_mean"] for r in rs])), 3),
                         frac_decided_by_b8=round(float(np.mean([r["tuned_lens"]["frac_decided_by_b8"] for r in rs])), 4),
                         final_vs_logtxt=round(float(np.mean([r["logit_lens"]["top1"][12] - (r["final_top1_logtxt"] or np.nan) for r in rs])), 3)))
    with open(registry.FIGS / "lens_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["arm"]); w.writeheader(); w.writerows(rows)
    print(f"wrote fig19 and lens_summary.csv ({len(rows)} arms) to {registry.FIGS}")


if __name__ == "__main__":
    main()
