"""fig18: one goal, two levers. Random (r), the early lever (ftbqmlnvo: blocks 0-8 quiet behind a
loud block 0) and the late lever (ftbrho / ftb3b: blocks 9-11 upscaled) side by side.
 (a) write profile at init, rho = |sublayer out| / |stream| relative to random, from the init dumps
 (b) where the readout sits at the end: head-probe top-1 per block at epoch 289 (seed mean)
 (c) how the readout forms: probe of a mid block (7) and the top block (11) over training
 (d) the write profile at epoch 289 (delta_norm_ratio from wandb, seed mean), relative to random
Output: plots/out/fig18_two_levers.png and the numbers on stdout."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/home/schrodi/Procedural"
C = json.load(open(f"{ROOT}/plots/cache/verify/wandb_layerwise.json"))
F = json.load(open(f"{ROOT}/results/init_dumps/init_forward_stats.json"))
ARMS = [("r", "random", "0.4"), ("ftbqmlnvo", "early lever: 0-8 quiet, block 0 loud (79.9)", "C3"),
        ("ftbrho", "late lever: 9-11 upscaled x1.4 (79.7)", "C0"), ("ftb3b", "late lever: 9-11 rho-matched to proc (80.0)", "C9")]

def seed_mean(arm, fam):
    per = []
    for s, d in C[arm].items():
        m = {}
        for e, row in d.items():
            if int(e) > 289: continue
            v = np.array([row.get(f"{fam}_layer{l}", np.nan) for l in range(12)], float); v[v == -1.0] = np.nan
            if not np.all(np.isnan(v)): m[int(e)] = v
        per.append(m)
    eps = sorted(set.intersection(*[set(m) for m in per]))
    return eps, np.array([np.nanmean([m[e] for m in per], 0) for e in eps])

fig, ax = plt.subplots(1, 4, figsize=(17, 3.9)); blocks = np.arange(12)
# (a) init profile relative to random
r0 = F["r_s0"]
for arm, lab, c in ARMS:
    key = f"{arm}_s0"
    if key not in F: continue
    ra = [F[key][str(b)]["rho_attn"] / r0[str(b)]["rho_attn"] for b in blocks]
    rm = [F[key][str(b)]["rho_mlp"] / r0[str(b)]["rho_mlp"] for b in blocks]
    ax[0].plot(blocks, ra, "-o", color=c, ms=3, label=lab); ax[0].plot(blocks, rm, "--s", color=c, ms=3, alpha=0.7)
    print(f"init rho/random {arm:10s} attn " + " ".join(f"{x:4.2f}" for x in ra) + " | mlp " + " ".join(f"{x:4.2f}" for x in rm))
ax[0].set_yscale("log"); ax[0].axhline(1, color="k", lw=0.5); ax[0].set_xlabel("block"); ax[0].set_ylabel("write / random at init (solid attn, dashed MLP)")
ax[0].set_title("(a) init: what each lever changes", fontsize=9); ax[0].legend(fontsize=6, loc="lower left")
# (b) readout location at 289
print("\nprobe top-1 per block at epoch 289 (seed mean):")
for arm, lab, c in ARMS:
    eps, A = seed_mean(arm, "acc"); a289 = A[eps.index(289)]
    ax[1].plot(blocks, a289, "-o", color=c, ms=3, label=lab)
    print(f"{arm:10s} " + " ".join(f"{x:5.1f}" for x in a289) + f"   | blocks 6-9 mean {np.nanmean(a289[6:10]):.1f}, block 11 {a289[11]:.1f}")
ax[1].set_xlabel("block"); ax[1].set_ylabel("head-probe top-1 at epoch 289 (%)"); ax[1].set_title("(b) end: where the readout sits", fontsize=9); ax[1].legend(fontsize=6)
# (c) probe over training for blocks 7 and 11
for arm, lab, c in ARMS:
    eps, A = seed_mean(arm, "acc")
    ax[2].plot(eps, A[:, 11], "-", color=c, label=f"{arm} block 11"); ax[2].plot(eps, A[:, 7], "--", color=c, alpha=0.8, label=f"{arm} block 7")
    pk = int(np.nanargmax(A[:, 7])); print(f"{arm:10s} block-7 probe peak {A[pk,7]:.1f} at epoch {eps[pk]}, at 289: {A[-1,7]:.1f};  block 11 at 289: {A[-1,11]:.1f}")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("head-probe top-1 (%)"); ax[2].set_title("(c) training: top block (solid) vs block 7 (dashed)", fontsize=9); ax[2].legend(fontsize=5, ncol=2)
# (d) end profile relative to random
epsr, Rr = seed_mean("r", "delta_norm_ratio"); rend = Rr[epsr.index(289)]
print("\nrho at epoch 289 relative to random:")
for arm, lab, c in ARMS[1:]:
    eps, A = seed_mean(arm, "delta_norm_ratio"); a = A[eps.index(289)] / rend
    ax[3].plot(blocks, a, "-o", color=c, ms=3, label=lab)
    print(f"{arm:10s} " + " ".join(f"{x:4.2f}" for x in a))
ax[3].axhline(1, color="k", lw=0.5); ax[3].set_xlabel("block"); ax[3].set_ylabel("rho at epoch 289 / random"); ax[3].set_title("(d) end: the profile that survives", fontsize=9); ax[3].legend(fontsize=6)
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{ROOT}/plots/out/fig18_two_levers.png", dpi=150); print("\nwrote plots/out/fig18_two_levers.png")
