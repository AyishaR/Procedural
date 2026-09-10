"""Paper version of fig18: two initialisation interventions on ViT-B/16 (ImageNet-1k, 300 epochs)
that act on different depths but produce the same training dynamics.

Conditions (seed means, 3 seeds each):
  standard init                 timm trunc-normal init (r)
  early-block attenuation       blocks 0-8 rescaled to a depth profile: block 0 amplified, blocks 1-8
                                attenuated; blocks 9-11 standard (ftbqmlnvo)
  late-block amplification      blocks 9-11 residual writes amplified x1.4 (ftbrho)
  late-block amplification*     blocks 9-11 residual writes matched to a proc network (ftb3b)
Panels: (a) residual-write ratio at init; (b) per-block linear-probe accuracy at the end of training;
(c) probe accuracy of the last block and of block 7 during training; (d) residual-write ratio at the end.
Output: plots/out/fig18_two_levers_paper.{png,pdf}"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = "/home/schrodi/Procedural"
C = json.load(open(f"{ROOT}/plots/cache/verify/wandb_layerwise.json"))
F = json.load(open(f"{ROOT}/results/init_dumps/init_forward_stats.json"))
LAST = 289   # last epoch with a valid per-block measurement
import seaborn as sns
T10 = sns.color_palette("tab10")
COND = [("r", "standard init", T10[7]),
        ("ftb3i", "blocks 0–8: proc weights, blocks 9–11 standard", T10[1]),
        ("ftbqmlnvo", "blocks 0–8: per-matrix weight std from proc (zero mean), LN gains 0.3–0.5", T10[3]),
        ("ftb3h", "blocks 9–11: proc weights, blocks 0–8 standard (1 seed)", T10[4]),
        ("ftbrho", "blocks 9–11: v, proj, fc2 scaled to ‖f(x)‖/‖x‖ = 1.4", T10[0]),
        ("ftb3b", "blocks 9–11: v, proj, fc2 scaled to proc’s ‖f(x)‖/‖x‖", T10[9])]
FINAL = {"r": 78.1, "ftb3i": 80.0, "ftbqmlnvo": 79.9, "ftb3h": 78.9, "ftbrho": 79.7, "ftb3b": 80.0}

def seed_mean(arm, fam):
    per = []
    for s, d in C[arm].items():
        m = {}
        for e, row in d.items():
            if int(e) > LAST: continue
            v = np.array([row.get(f"{fam}_layer{l}", np.nan) for l in range(12)], float); v[v == -1.0] = np.nan
            if not np.all(np.isnan(v)): m[int(e)] = v
        if fam == "acc":
            # drop artefact rows: a resumed job's end-of-segment analysis pass measures a random model
            # (probe at chance in every block), e.g. ftb3i seeds 0 and 2 at epoch 69 and every run's epoch 299
            eps_s = sorted(m); m = {e: v for i, e in enumerate(eps_s) for v in [m[e]]
                                    if not (i > 0 and e > 20 and np.nanmax(v) < 0.5 * np.nanmax(m[eps_s[i - 1]]))}
        per.append(m)
    eps = sorted(set.union(*[set(m) for m in per]))
    eps = [e for e in eps if sum(e in m for m in per) >= max(1, len(per) - 1)]   # keep epochs present in all but one seed
    return eps, np.array([np.nanmean([m[e] for m in per if e in m], 0) for e in eps])

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6)); blocks = np.arange(12)

# (a) residual-write ratio at initialisation, relative to the standard init
r0 = F["r_s0"]
for arm, lab, c in COND:
    key = f"{arm}_s0"
    if key not in F: continue
    ra = [F[key][str(b)]["rho_attn"] / r0[str(b)]["rho_attn"] for b in blocks]
    rm = [F[key][str(b)]["rho_mlp"] / r0[str(b)]["rho_mlp"] for b in blocks]
    ax[0].plot(blocks, ra, "-o", color=c, ms=3, lw=1.4); ax[0].plot(blocks, rm, "--s", color=c, ms=3, lw=1.2, alpha=0.8)
ax[0].set_yscale("log"); ax[0].axhline(1, color="k", lw=0.5)
ax[0].set_xlabel("block"); ax[0].set_ylabel(r"$\|f_\ell(x)\|\,/\,\|x\|$, relative to standard init")
ax[0].set_title("(a) Residual-write ratio at initialisation")
ax[0].legend([Line2D([], [], color="k", ls="-", marker="o", ms=3), Line2D([], [], color="k", ls="--", marker="s", ms=3)],
             ["attention sublayer", "MLP sublayer"], loc="lower right", frameon=False)

# (b) class decodability per block at the end of training
for arm, lab, c in COND:
    eps, A = seed_mean(arm, "acc"); ax[1].plot(blocks, A[eps.index(LAST)], "-o", color=c, ms=3, lw=1.4)
ax[1].set_xlabel("block"); ax[1].set_ylabel("linear-probe top-1 accuracy (%)")
ax[1].set_title(f"(b) Class decodability per block, epoch {LAST}")

# (c) probe accuracy during training: last block and block 7
for arm, lab, c in COND:
    eps, A = seed_mean(arm, "acc")
    ax[2].plot(eps, A[:, 11], "-", color=c, lw=1.4); ax[2].plot(eps, A[:, 7], "--", color=c, lw=1.2, alpha=0.85)
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("linear-probe top-1 accuracy (%)")
ax[2].set_title("(c) Probe accuracy during training")
ax[2].legend([Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")], ["block 11 (last)", "block 7"], loc="center right", frameon=False)

# (d) residual-write ratio at the end of training, absolute (the standard init is not flat: training
# itself builds a loud block 0, quiet middle and loud top from every init)
for arm, lab, c in COND:
    eps, A = seed_mean(arm, "delta_norm_ratio"); ax[3].plot(blocks, A[eps.index(LAST)], "-o", color=c, ms=3, lw=1.4)
ax[3].set_yscale("log"); ax[3].set_xlabel("block"); ax[3].set_ylabel(r"$\|f_\ell(x)\|\,/\,\|x\|$, MLP sublayer")
ax[3].set_title(f"(d) Residual-write ratio at epoch {LAST}")

for a in ax: a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
handles = [Line2D([], [], color=c, lw=2) for _, _, c in COND]
labels = [f"{lab}  ({FINAL[arm]:.1f}%)" for arm, lab, _ in COND]
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.09))
plt.tight_layout(rect=(0, 0, 1, 0.88))
for ext in ("png", "pdf"):
    plt.savefig(f"{ROOT}/plots/out/fig18_two_levers_paper.{ext}", dpi=200, bbox_inches="tight")
print("wrote plots/out/fig18_two_levers_paper.png / .pdf")
