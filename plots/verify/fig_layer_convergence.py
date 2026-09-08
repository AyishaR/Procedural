"""fig17: per-layer training dynamics from the wandb traces (plots/verify/wandb_layerwise.py).
NOTE: for every run launched after ~2026-08-15 the row logged as epoch 299 is an artefact (the
end-of-training pass in attention_analyse_final re-loads checkpoint-299.pth and logs a model with
uniform attention and chance probe accuracy), so epochs <= 289 are used throughout.  The per-block
head-probe accuracy (`acc_layer`: the trained head applied to block i's output) is valid up to 289.
Rows = quantity, columns = arm, lines = blocks (colour = depth).  Bottom: convergence epoch per
block = first logged epoch after which rho stays within 10% of its value at epoch 289 (the last valid row)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = json.load(open("/home/schrodi/Procedural/plots/cache/verify/wandb_layerwise.json"))
LABEL = {"r": "random", "p": "proc (all 12)", "ftb3i": "proc 0-8, random 9-11", "ftb3h": "random 0-8, proc 9-11 (flipped)",
         "ftbrho": "random, 9-11 upscaled (rho 1.4)", "ftb3b": "random, 9-11 rho-matched to proc", "ftb4o": "random, 0-7 upscaled to proc rho",
         "ftbqmlnvo": "proc marginals rank-mapped 0-8"}
ORDER = [a for a in ["r", "ftb3i", "ftb3h", "ftbrho", "ftb3b", "ftb4o", "p", "ftbqmlnvo"] if a in C]
FAMS = [("acc", "head-probe top-1 of block output (%)", False), ("delta_norm_ratio", "rho = |sublayer out| / |stream|", True), ("attn_entropy", "attention entropy (nats; uniform = 5.28)", False)]   # grad_norm is only logged (not the -1 sentinel) for r, p, ftb3b -> omitted
def seed_mean(arm, fam):
    per = []
    for s, d in C[arm].items():
        m = {}
        for e, row in d.items():
            v = np.array([row.get(f"{fam}_layer{l}", np.nan) for l in range(12)], float)
            v[v == -1.0] = np.nan   # -1 is the "not logged" sentinel
            if not np.all(np.isnan(v)): m[int(e)] = v
        if m: per.append(m)
    if not per: return {}, 0
    eps = sorted(set.intersection(*[set(m) for m in per]))
    # the epoch-299 row of every run launched after ~2026-08-15 is an artefact (entropy jumps to
    # uniform, rho jumps, the probe reads chance): the end-of-training analysis pass measures a
    # different model state. Use epochs <= 289 for every arm so they are comparable.
    return {e: np.nanmean([m[e] for m in per], 0) for e in eps if 0 <= e <= 289}, len(per)
fams = [f for f in FAMS if any(seed_mean(a, f[0])[0] for a in ORDER)]
n = len(ORDER); R = len(fams) + 1
fig, axes = plt.subplots(R, n, figsize=(3.6 * n, 3.1 * R), squeeze=False); cmap = plt.get_cmap("viridis")
for j, a in enumerate(ORDER):
    for i, (fam, ylab, logy) in enumerate(fams):
        ax = axes[i, j]; d, ns = seed_mean(a, fam); eps = sorted(d)
        for l in range(12):
            ax.plot(eps, [d[e][l] for e in eps], color=cmap(l / 11), lw=1.2, label=f"b{l}" if (i == 0 and j == 0) else None)
        if logy: ax.set_yscale("log")
        if i == 0: ax.set_title(f"{a}: {LABEL.get(a, a)} (n={ns})", fontsize=8.5)
        if j == 0: ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        if i == len(fams) - 1: ax.set_xlabel("epoch", fontsize=8)
    if n and fams: axes[0, 0].legend(fontsize=5.5, ncol=3, loc="upper right")
# convergence epoch per block, from the head probe: first logged epoch at which the probe reaches 90% of
# its epoch-289 value; blocks whose probe never exceeds 20% have no meaningful convergence epoch
for j, a in enumerate(ORDER):
    ax = axes[R - 1, j]; d, ns = seed_mean(a, "acc"); eps = sorted(d)
    if not eps: continue
    final = d[eps[-1]]; conv = [next((e for e in eps if d[e][l] >= 0.9 * final[l]), eps[-1]) if final[l] >= 20 else np.nan for l in range(12)]
    ax.bar(range(12), [0 if np.isnan(c) else c for c in conv], color=[cmap(l / 11) for l in range(12)]); ax.set_ylim(0, 300); ax.set_xlim(-0.6, 11.6); ax.set_xticks(range(12)); ax.set_xlabel("block", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=.3, axis="y")
    for l in range(12):
        if np.isnan(conv[l]): ax.text(l, 6, "<20%", rotation=90, fontsize=5.5, ha="center", va="bottom", color="0.45")
    if j == 0: ax.set_ylabel("epoch at which the probe reaches\n90% of its epoch-289 value", fontsize=8)
plt.tight_layout(); out = "/home/schrodi/Procedural/plots/out/fig17_layer_convergence.png"; plt.savefig(out, dpi=140); print("saved", out)
print("\nhead-probe convergence epoch per block (90% of the epoch-289 value; '-' = probe below 20%):")
print("arm        " + " ".join(f"b{l:<3d}" for l in range(12)) + "   probe @289 per block")
for a in ORDER:
    d, ns = seed_mean(a, "acc"); eps = sorted(d); final = d[eps[-1]]
    conv = [next((e for e in eps if d[e][l] >= 0.9 * final[l]), eps[-1]) if final[l] >= 20 else None for l in range(12)]
    print(f"{a:10s} " + " ".join(f"{c:<4d}" if c is not None else "  - " for c in conv) + "   " + " ".join(f"{v:.0f}" for v in final))
