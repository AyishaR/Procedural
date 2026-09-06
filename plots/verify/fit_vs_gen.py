"""Does the final TRAINING loss (fit) predict final test accuracy across arms?  Separates
'regularised' inits (fit worse, test better) from 'damaged' ones (fit worse, test worse)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
S = "/home/schrodi/Procedural/plots/cache/verify"
T = json.load(open(f"{S}/trajectories.json"))
G = {  # group by construction (from the verified signature table)
 "proc values in blocks 0-8 (tail random)": ["ftb3i","ftbqmlnvo","ftb4e3fix","ftbqm1dvo","ftbqmvo","ftbqmln","ftbqm","ftbqm1d","ftbsb","ftbsv"],
 "no checkpoint values, blocks 0-8 rescaled": ["r","ftbnorm","ftbcfg","ftbqu","ftbvu","ftbvd","ftbslice","ftbclip01","ftbclip1","ftbclip5","ftb4o"],
 "block 0 only": ["ftb11i","ftb11isfix","ftb11d","ftb11s"],
 "proc prefix, intact (ftbKi) / full proc": ["p","ftb1i","ftb2i","ftb4i","ftb5i","ftb6i","ftb7i","ftb8i","ftb9i","ftb10i","ftb4m","ftb4l","ftb4g","ftb0a","ftb0m"],
 "late-block recipe / proc suffix (h, e, b, rho, comp)": ["ftbrho","ftbrho07","ftb4k","ftb4n","ftb9e","ftb10e","ftb7e","ftb9b","ftb10b","ftb11b","ftb1h","ftb2h","ftb3h","ftb4h","ftb5h","ftb6h","ftb7h","ftb8h","ftb9h","ftb10h","ftb11h","ftb0h","ftb0g","ftbcomp1","ftbcomp11","ftbcomp25","ftb4jd","ftb3es1",
     "ftb1b","ftb2b","ftb3b","ftb4b","ftb5b","ftb6b","ftb7b","ftb8b","ftb1e","ftb2e","ftb4e","ftb5e","ftb6e","ftb8e","pds2","pds3","pds4","pds5","pds12"],
 "attention-only suffix (pattn / rattn / pattn*d)": ["pattn1","pattn2","pattn3","pattn4","pattn5","pattn6","rattn1","rattn2","rattn3","rattn4","rattn5","rattn6","pattn1d","pattn2d","pattn3d","pattn4d","pattn5d","pattn6d"],
}
col = {list(G)[0]: "C3", list(G)[1]: "C0", list(G)[2]: "C2", list(G)[3]: "C1", list(G)[4]: "0.5", list(G)[5]: "C8"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
allx, ally, alltl = [], [], []
for grp, arms in G.items():
    xs, ys, tls, labels = [], [], [], []
    for a in arms:
        k = f"{a}|clean"
        if k not in T: continue
        xs.append(T[k]["train_loss"]); ys.append(T[k]["acc"]["299"] if "299" in T[k]["acc"] else T[k]["acc"][299]); tls.append(T[k]["test_loss"]); labels.append(a)
    if not xs: continue
    axes[0].scatter(xs, ys, c=col[grp], label=grp, s=28 + 10 * np.array([T[f'{a}|clean']['n'] for a in labels]), alpha=0.85, edgecolor="k", lw=0.4)
    axes[1].scatter(xs, tls, c=col[grp], s=28, alpha=0.85, edgecolor="k", lw=0.4)
    for x, y, t, l in zip(xs, ys, tls, labels):
        axes[0].annotate(l, (x, y), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
        axes[1].annotate(l, (x, t), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
    allx += xs; ally += ys; alltl += tls
    if len(xs) >= 5:
        r, p = stats.pearsonr(xs, ys); print(f"{grp:52s} n={len(xs):2d}  r(trainL, acc) = {r:+.2f} (p={p:.3f})")
r, p = stats.pearsonr(allx, ally); print(f"{'ALL':52s} n={len(allx)}  r(trainL, acc) = {r:+.2f} (p={p:.1e})")
axes[0].set_xlabel("final training loss (epoch 299, mixup/cutmix soft targets)"); axes[0].set_ylabel("final test top-1"); axes[0].legend(fontsize=7, loc="lower right")
axes[0].set_title(f"fit vs accuracy across {len(allx)} arms (clean seeds; marker size = n)", fontsize=10)
axes[1].set_xlabel("final training loss"); axes[1].set_ylabel("final test loss"); axes[1].set_title("fit vs test loss", fontsize=10)
axes[0].grid(alpha=0.3); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{S}/fig_fit_vs_gen.png", dpi=160); print("saved fig_fit_vs_gen.png")
# within the blocks-0-8 families only
sub = G[list(G)[0]] + G[list(G)[1]]
xs = [T[f"{a}|clean"]["train_loss"] for a in sub if f"{a}|clean" in T]; ys = [T[f"{a}|clean"]["acc"]["299"] for a in sub if f"{a}|clean" in T]
r, p = stats.pearsonr(xs, ys); print(f"blocks-0-8 arms only (value + checkpoint-free): n={len(xs)} r = {r:+.2f} p={p:.4f}")
rs, ps = stats.spearmanr(xs, ys); print(f"   spearman {rs:+.2f} p={ps:.4f}")
