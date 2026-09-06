"""Figure: end-of-training depth profiles (MLP LayerNorm gain, rho_mlp) for key arms, and the
depth-allocation scalar against final accuracy across all arms."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/home/schrodi/Procedural/plots/cache/verify"
rows = {r["arm"]: r for r in json.load(open(f"{S}/end_state_rows.json"))}
KEY = ["r", "ftbqmlnvo", "ftb4e3fix", "ftb3i", "ftbqmln", "ftbnorm", "ftbqmvo", "ftbvd", "ftbcfg", "ftbrho", "p", "ftb11isfix"]
col = {"r": "k", "ftbqmlnvo": "C3", "ftb4e3fix": "C1", "ftb3i": "C4", "ftbqmln": "C0", "ftbnorm": "C9", "ftbqmvo": "C2", "ftbvd": "C5", "ftbcfg": "C7", "ftbrho": "C8", "p": "C6", "ftb11isfix": "0.6"}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for a in KEY:
    if a not in rows: continue
    r = rows[a]; lab = f"{a} ({r['delta']:+.2f})"
    axes[0].plot(range(12), r["g2"], marker="o", ms=3.5, color=col[a], label=lab, lw=1.6 if a in ("r", "ftbqmlnvo") else 1.1)
    axes[1].plot(range(12), r["rho_mlp"], marker="o", ms=3.5, color=col[a], label=lab, lw=1.6 if a in ("r", "ftbqmlnvo") else 1.1)
axes[0].set_title("end of training: MLP LayerNorm gain (mean) per block", fontsize=10); axes[0].set_xlabel("block"); axes[0].set_ylabel("gamma2 mean"); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
axes[1].set_title("end of training: rho_mlp = ||MLP output|| / ||stream|| per block", fontsize=10); axes[1].set_xlabel("block"); axes[1].set_ylabel("rho_mlp"); axes[1].grid(alpha=.3)
xs, ys, labs, cs = [], [], [], []
for a, r in rows.items():
    if a.endswith("_PRE"): continue
    xs.append(np.mean(r["g2"][6:9])); ys.append(r["acc"]); labs.append(a)
    cs.append("C3" if a in ("ftb3i","ftbqmlnvo","ftb4e3fix","ftbqm1dvo","ftbqmvo","ftbqmln","ftbqm","ftbqm1d") else "C0" if a in ("r","ftbnorm","ftbcfg","ftbqu","ftbvu","ftbvd","ftbslice","ftbclip01","ftbclip1","ftbclip5","ftb4o") else "0.5")
axes[2].scatter(xs, ys, c=cs, s=26, edgecolor="k", lw=.4)
for x, y, l in zip(xs, ys, labs): axes[2].annotate(l, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
axes[2].set_xlabel("end-state gamma2, mean over blocks 6-8"); axes[2].set_ylabel("final test top-1"); axes[2].set_title("depth allocation of MLP work vs accuracy (all arms)", fontsize=10); axes[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{S}/fig_end_state.png", dpi=160); print("saved")
