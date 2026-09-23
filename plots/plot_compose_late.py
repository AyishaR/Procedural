"""Late-lever part of the compose arm's reconstruction figure: write ratios (||branch|| / ||stream||) of the attention and the MLP
branch per block for timm random, the committed early lever C, the compose arm ftbc7l and the ftbrho init (checkpoint blocks 9-11),
from plots/out/compose_late_ratios.json (written by plots/verify/verify_compose.py on 256 training images).
usage: .venv/bin/python plots/plot_compose_late.py  ->  plots/out/reconstruction_ftbc7l_late.png / .pdf"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("/home/schrodi/Procedural/plots/out/compose_late_ratios.json"))
COL = {"timm": "#898781", "C": "#2a78d6", "compose": "#eb6834", "ftbrho": "#1baf7a"}
LAB = {"timm": "timm random", "C": "early lever C (blocks 0-7)", "compose": "compose ftbc7l (C + late lever 9-11)", "ftbrho": "ftbrho init (checkpoint blocks 9-11)"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8)); blocks = list(range(12))
for ax, key, title in zip(axes, ("attn", "mlp"), ("attention write ratio  ||attn(x)|| / ||x||", "MLP write ratio  ||mlp(x)|| / ||x||")):
    for name in ("timm", "C", "compose", "ftbrho"):
        if name not in R: continue
        ys = [R[name][str(b)][key] for b in blocks]; ax.plot(blocks, ys, "-o", ms=4, lw=1.6, color=COL[name], label=LAB[name])
        if name == "compose": [ax.annotate(f"{ys[b]:.2f}", (b, ys[b]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=COL[name]) for b in (9, 10, 11)]
    ax.axvspan(8.5, 11.5, color="#eb6834", alpha=0.07, lw=0); ax.axvspan(-0.5, 7.5, color="#2a78d6", alpha=0.05, lw=0)
    ax.set_xticks(blocks); ax.set_xlabel("block"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.25)
axes[0].set_ylabel("write ratio at initialisation (256 training images)"); axes[1].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Compose arm ftbc7l at init: early lever on blocks 0-7 (blue band) + late lever on blocks 9-11 (orange band); block 8 timm", fontsize=10)
fig.tight_layout(); fig.savefig("/home/schrodi/Procedural/plots/out/reconstruction_ftbc7l_late.png", dpi=160); fig.savefig("/home/schrodi/Procedural/plots/out/reconstruction_ftbc7l_late.pdf"); print("wrote plots/out/reconstruction_ftbc7l_late.png/.pdf")
