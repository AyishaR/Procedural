"""Figures for Phase 0 of docs/early_lever_mechanism_plan.md. Reads plots/out/phase0/*.json and plots/out/mechanism_autopsy.json,
writes plots/out/phase0/figs/*.png (+ .pdf). Login-node safe (no GPU).
Colour follows the ROLE of an arm, the same in every figure (validated with the dataviz skill's palette checker, light surface):
  committed design = blue, procedural prefix = orange, scale-only winner = aqua, no-gain comparison = yellow, active-unit twin = magenta,
  random-level reference = gray (context). Scatter plots use two hues + gray and encode the task by marker shape.
usage: .venv/bin/python plots/verify/mechanism_phase0_plots.py"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
HALO = [patheffects.withStroke(linewidth=2.6, foreground='#fcfcfb')]
from scipy.stats import spearmanr, rankdata
ROOT = "/home/schrodi/Procedural"; OUT = f"{ROOT}/plots/out/phase0/figs"; os.makedirs(OUT, exist_ok=True)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = {"committed": "#2a78d6", "prefix": "#eb6834", "scale-only winner": "#1baf7a", "no gain": "#eda100", "active-unit twin": "#e87ba4", "random-level": MUTED}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                     "axes.labelcolor": INK2, "text.color": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
                     "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.markersize": 6.5, "axes.axisbelow": True})
def merged(mode):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/plots/out/phase0/{mode}_shard*.json")): out.update(json.load(open(f)))
    return out
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); plt.close(fig); print("wrote", f"{OUT}/{name}.png")
def headline(fig, title, subtitle, y=0.995):
    """title + wrapped subtitle inside the figure width; returns nothing, the caller reserves the top margin"""
    import textwrap
    w = fig.get_size_inches()[0]; t = textwrap.fill(title, int(w * 9.3)); sub = textwrap.fill(subtitle, int(w * 13.2)); h = fig.get_size_inches()[1]
    fig.text(0.01, y, t, fontsize=12.5, fontweight="bold", color=INK, va="top", ha="left", linespacing=1.15)
    fig.text(0.01, y - (0.30 * (t.count("\n") + 1)) / h, sub, fontsize=9.5, color=INK2, va="top", ha="left", linespacing=1.25)
    header = 0.30 * (t.count("\n") + 1) + 0.19 * (sub.count("\n") + 1) + 0.48        # inches: title lines + subtitle lines + air above the panel titles
    fig.subplots_adjust(top=1.0 - header / h)
def end_labels(ax, ends, pad, min_gap):
    """direct labels at the line ends; when two ends are closer than min_gap the labels are spread and tied back with a hairline leader"""
    ends = sorted(ends, key=lambda t: t[1]); ys = [e[1] for e in ends]
    for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (x, y, text, col), yy in zip(ends, ys):
        ax.plot([x, x + pad * 0.85], [y, yy], color=AXIS, lw=0.8, clip_on=False, zorder=1)
        ax.plot([x + pad * 0.85], [yy], marker="o", ms=4, color=col, mec=SURFACE, mew=0.8, clip_on=False, zorder=3)
        ax.text(x + pad, yy, text, va="center", ha="left", fontsize=8.5, color=INK2, clip_on=False)
m17 = lambda f, n: float(np.mean([f[str(b)][n] for b in range(1, 8)]))

Z, DEPTH, FIT, CLS = merged("zoo"), merged("depth"), merged("fitgap"), merged("clspath")
AUT = json.load(open(f"{ROOT}/plots/out/mechanism_autopsy.json"))
rows = []
for key, r in Z.items():
    d, p, f = r["drift"], r["probe"], r["functional"]
    if "29" not in p or "0" not in d: continue
    task = "ksd" if r["arm"].startswith("ftbanak") else r["task"]
    late_timm = all(abs(np.mean([d["0"][str(b)][n] for b in range(9, 12)]) - 1.0) < 0.04 for n in ("rms_q", "rms_k", "rms_fc1"))
    rows.append({"arm": r["arm"], "task": task, "final": r["final"], "train_loss": r["train_loss"], "lens7": p["29"]["lens7"], "probe7": p["29"]["probe7"], "model29": p["29"]["model"],
                 "probe7_end": p.get("299", {}).get("probe7", np.nan), "late_sig": p["29"]["lens7"] < 3.0 and p["29"]["probe7"] > 8, "family": late_timm,
                 "drift_k9": 100 * m17(d["9"], "k") if "9" in d else np.nan, "drift_q9": 100 * m17(d["9"], "q") if "9" in d else np.nan,
                 "gelu29": m17(f["29"], "gelu_rms"), "D9": m17(f["9"], "D"), "att_spec9": m17(f["9"], "att_specific"), "att_common9": m17(f["9"], "att_common"),
                 "entropy0": m17(f["0"], "entropy"), "drift_in9": 100 * np.mean([m17(d["9"], n) for n in ("q", "k", "fc1")]) if "9" in d else np.nan,
                 "drift_in19": 100 * np.mean([m17(d["19"], n) for n in ("q", "k", "fc1")]) if "19" in d else np.nan, "rms_q": m17(d["0"], "rms_q")})
for r in rows: r["family"] = r["family"] and not r["late_sig"]
NAMES = {"ftbanapermb7i": "committed, kdyck", "ftbanakpermb7i": "committed, ksd", "ftbana": "ftbana (profile, fast fc1)", "ftbanap": "scale-only winner", "ftbrhosl": "slow LATE blocks",
         "ftbqu": "random-level", "ftbanak": "ksd scale-only", "ftbanaperab7w": "write-matched"}

# ---------------- F1: the lens transient against the final accuracy
fig, ax = plt.subplots(figsize=(8.6, 5.6)); fig.subplots_adjust(top=0.84, right=0.97)
groups = [("other arms (procedural or rescaled late blocks)", lambda r: not r["family"] and not r["late_sig"], MUTED, 0.55), ("loud late blocks (lens suppressed by construction)", lambda r: r["late_sig"], C["prefix"], 0.95),
          ("early-lever arms (blocks 9-11 untouched)", lambda r: r["family"], C["committed"], 0.95)]
for label, sel, col, alpha in groups:
    for task, mk in (("kdyck", "o"), ("ksd", "^")):
        pts = [r for r in rows if sel(r) and r["task"] == task]
        ax.scatter([r["lens7"] for r in pts], [r["final"] for r in pts], s=58, marker=mk, color=col, alpha=alpha, edgecolor=SURFACE, linewidth=1.2, zorder=3 if col != MUTED else 2,
                   label=f"{label}, {task}" if task == "kdyck" else None)
ax.axvline(30, color=AXIS, lw=1.0); ax.text(30.6, 75.25, "lens = 30", fontsize=8.5, color=MUTED, va="bottom")
for arm, (dx, dy) in {"ftbanapermb7i": (2.5, 0.42), "ftbanakpermb7i": (-1.5, -0.62), "ftbana": (3.0, -0.45), "ftbanap": (3.5, 0.45), "ftbrhosl": (-2.0, 0.45), "ftbqu": (4.5, 0.55)}.items():
    pts = [r for r in rows if r["arm"] == arm]
    if pts:
        x, y = np.mean([p["lens7"] for p in pts]), np.mean([p["final"] for p in pts])
        ax.annotate(NAMES[arm], (x, y), xytext=(x + dx, y + dy), fontsize=8.5, color=INK2, ha="left" if dx > 0 else "right", va="center", path_effects=HALO, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=4))
fam = [r for r in rows if r["family"]]
ax.set_xlabel("head lens at block 7, epoch 29: top-1 (%) of final norm + head on the block-7 class token"); ax.set_ylabel("last-epoch top-1 (%)")
ax.scatter([], [], s=58, marker="^", color=INK2, edgecolor=SURFACE, label="triangles: ksd runs"); ax.legend(loc="lower left", fontsize=8.5, handletextpad=0.4, borderaxespad=0.2)
headline(fig, "A high lens transient at block 7 marks the runs that do not gain", f"{len(rows)} finished runs with per-epoch checkpoints. Spearman with the final: all runs {spearmanr([r['lens7'] for r in rows], [r['final'] for r in rows])[0]:+.2f}, "
         f"early-lever family {spearmanr([r['lens7'] for r in fam], [r['final'] for r in fam])[0]:+.2f} (n = {len(fam)}); among winners it does not rank.")
save(fig, "f1_lens_vs_final")

# ---------------- F2: where the class is computed (depth profiles)
LINES = [("ftbqu", "random-level", "random-level"), ("ftbana", "no gain", "ftbana 76.6"), ("ftbanap", "scale-only winner", "scale-only 80.2"), ("ftb4i_kdyck", "prefix", "kdyck prefix 79.9"), ("ftbanapermb7i", "committed", "committed 80.4")]
fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True); fig.subplots_adjust(top=0.84, right=0.97, hspace=0.28, wspace=0.16)
for ci, e in enumerate((29, 299)):
    for ri, (kind, ylabel) in enumerate((("probe", "trained linear probe, top-1 (%)"), ("lens", "head lens, top-1 (%)"))):
        ax = axes[ri, ci]; ends = []
        for arm, role, label in LINES:
            r = DEPTH.get(f"{arm}@{e}")
            if r is None: continue
            y = [r[kind][str(b)] for b in range(12)]; ax.plot(range(12), y, color=C[role], marker="o", mec=SURFACE, mew=1.0, lw=2.6 if role == "random-level" else 2.0, zorder=2 if role == "random-level" else 3, label=label)
            ends.append((11, y[-1], label, C[role]))
        ax.axvspan(-0.5, 7.5, color=GRID, alpha=0.35, lw=0, zorder=0); ax.set_xlim(-0.5, 11.5); ax.set_ylim(0, 85); ax.set_xticks(range(12))
        ax.set_title(f"{'probe' if kind == 'probe' else 'lens'}, epoch {e}"); ax.set_ylabel(ylabel if ci == 0 else "")
        if ri == 1: ax.set_xlabel("block")
axes[0, 0].text(3.5, 78, "blocks 0-7 (early lever)", fontsize=8.5, color=MUTED, ha="center"); axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=8.5)
headline(fig, "Winners with sink and gate compute the class late",
         "The middle of a random-init network is already class-decodable, also in the final network; the scale-only winner keeps a random-like profile. kdyck arms. Probe: class token + mean patch token, 20 training / 10 validation images per class. Lens: final norm + head on the class token.")
save(fig, "f2_depth_profiles")

# ---------------- F3: the gain is a smaller generalisation gap
fig, ax = plt.subplots(figsize=(8.2, 5.4)); fig.subplots_adjust(top=0.84, right=0.95)
ROLE_OF = {"ftbanapermb7i": "committed", "ftbanakpermb7i": "committed", "ftb4i_kdyck": "prefix", "ftb4i": "prefix", "ftbanap": "scale-only winner", "ftbanal": "scale-only winner",
           "ftbanaperab7i": "active-unit twin", "ftbanakperab7i": "active-unit twin", "ftbana": "no gain", "ftbanak": "no gain", "ftbanapermb7w": "no gain"}
LAB = {"ftbanapermb7i": "committed kdyck", "ftbanakpermb7i": "committed ksd", "ftb4i_kdyck": "kdyck prefix", "ftb4i": "ksd prefix", "ftbanap": "scale-only", "ftbanal": "scale-only + lr", "ftbanaperab7i": "twin kdyck",
       "ftbanakperab7i": "twin ksd", "ftbana": "ftbana", "ftbanak": "ksd scale-only", "ftbanapermb7w": "write-matched", "r": "timm random", "ftbqu": "random-level (q,k x2.2)", "ftbrhosl": "random-level (slow late)"}
seen = set()
for arm, r in FIT.items():
    role = ROLE_OF.get(arm, "random-level"); x, y = r["val_ce"] - r["train_clean_ce"], r["val_acc"]
    ax.scatter([x], [y], s=70, color=C[role], edgecolor=SURFACE, linewidth=1.4, zorder=3, label=None if role in seen else role); seen.add(role)
    off = {"ftb4i_kdyck": (-4, 9), "ftb4i": (8, -3), "ftbanal": (8, 7), "ftbanakperab7i": (-4, -13), "ftbanaperab7i": (9, -4), "ftbanakpermb7i": (-70, -13), "ftbqu": (-118, 7), "ftbanak": (8, -11), "r": (8, 2), "ftbrhosl": (-70, 9)}.get(arm, (8, 5))
    ax.annotate(LAB.get(arm, arm), (x, y), xytext=off, textcoords="offset points", fontsize=8.3, color=INK2, path_effects=HALO)
xs, ys = [r["val_ce"] - r["train_clean_ce"] for r in FIT.values()], [r["val_acc"] for r in FIT.values()]
ax.set_xlabel("generalisation gap at the end: validation cross-entropy minus cross-entropy on clean training images"); ax.set_ylabel("validation top-1 (%), 25 images per class"); ax.legend(loc="lower left", fontsize=8.5)
headline(fig, "The gain is regularisation: winners fit the clean training images less and generalise better",
         f"Final checkpoints, 25k training images without augmentation and 25k validation images. Spearman(gap, accuracy) = {spearmanr(xs, ys)[0]:+.2f}; clean-train top-1 is 99.3-99.4% for random-level and losers, 96.6-98.4% for prefix-like winners.")
save(fig, "f3_fit_gap")

# ---------------- F4: realised relative weight change per epoch
EP = [0, 4, 9, 19, 49, 99, 199]
PANELS = [("kdyck", [("ftbrhosl", "random-level", "random-level (timm early blocks)"), ("ftbana", "no gain", "ftbana 76.6 (raw fc1 0.36x timm)"), ("ftbanal", "scale-only winner", "ftbana + lr scales 79.8"), ("ftbanapermb7i", "committed", "committed 80.4")]),
          ("ksd", [("ftbrhosl", "random-level", "random-level (timm early blocks)"), ("ftbanak", "no gain", "ksd scale-only 77.9 (= random)"), ("ftbanakpermb7i", "committed", "committed 79.9")])]
fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), sharex=True, sharey=True); fig.subplots_adjust(top=0.83, right=0.80, hspace=0.30, wspace=0.10)
for ri, (task, lines) in enumerate(PANELS):
    for ci, t in enumerate(("q", "fc1", "fc2")):
        ax = axes[ri, ci]; ends = []
        for arm, role, label in lines:
            d = AUT["drift"][arm]; y = [100 * float(np.mean([d[str(e)][str(b)][t] for b in range(1, 8)])) for e in EP]
            ax.plot(range(len(EP)), y, color=C[role], marker="o", mec=SURFACE, mew=1.0, lw=2.6 if role == "random-level" else 2.0, zorder=2 if role == "random-level" else 3); ends.append((len(EP) - 1, y[-1], label, C[role]))
        ax.set_xticks(range(len(EP))); ax.set_xticklabels([f"{e}" for e in EP]); ax.set_ylim(0, 40); ax.set_title(f"{task}: {t}, blocks 1-7")
        if ci == 0: ax.set_ylabel("relative change per epoch (%)")
        if ri == 1: ax.set_xlabel("epoch e (change from e to e+1)")
        if ci == 2:
            first = [(len(EP) - 1, 34 - 5.0 * i, lab, C[role]) for i, (_, role, lab) in enumerate(lines)]
            for x, yy, lab, col in first:
                ax.plot([x + 0.35], [yy], marker="o", ms=5, color=col, mec=SURFACE, mew=0.8, clip_on=False); ax.text(x + 0.6, yy, lab, va="center", ha="left", fontsize=8.5, color=INK2, clip_on=False)
headline(fig, "Slow steps are real where the raw matrix is large, last ~50 epochs, and are not sufficient on ksd",
         "||W(e+1) - W(e)|| / ||W(e)||, mean over blocks 1-7. All arms converge by epoch ~100 (AdamW norm equilibrium). The ksd scale-only arm is as slow as the committed arm and ends at random level; the arm below random is the fast one.")
save(fig, "f4_realised_drift")

# ---------------- F5: functional state of blocks 1-7 over training
EPF = [0, 1, 2, 4, 9, 19, 29, 49, 99, 199, 299]
SETS = [("kdyck", [("ftbrhosl", "random-level", "random-level"), ("ftbana", "no gain", "ftbana 76.6"), ("ftbanap", "scale-only winner", "scale-only 80.2"), ("ftbanaperab7i", "active-unit twin", "active-unit twin 79.6"),
                   ("ftb4i_kdyck", "prefix", "kdyck prefix 79.9"), ("ftbanapermb7i", "committed", "committed 80.4")]),
        ("ksd", [("ftbrhosl", "random-level", "random-level"), ("ftbanak", "no gain", "ksd scale-only 77.9"), ("ftbanakperab7i", "active-unit twin", "active-unit twin 79.7"), ("ftb4i", "prefix", "ksd prefix 80.1"),
                 ("ftbanakpermb7i", "committed", "committed 79.9")])]
METR = [("D", "D: token differences changed per block"), ("att_specific", "token-specific attention write"), ("gelu_rms", "GELU output rms (MLP activity)")]
fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.4), sharex=True); fig.subplots_adjust(top=0.88, right=0.80, hspace=0.24, wspace=0.14)
for ci, (task, lines) in enumerate(SETS):
    for ri, (metric, ylabel) in enumerate(METR):
        ax = axes[ri, ci]
        for arm, role, label in lines:
            f = AUT["functional"][arm]; y = [float(np.mean([f[str(e)][str(b)][metric] for b in range(1, 8)])) for e in EPF]
            ax.plot(range(len(EPF)), y, color=C[role], marker="o", ms=5, mec=SURFACE, mew=0.9, lw=2.6 if role == "random-level" else 2.0, zorder=2 if role == "random-level" else 3, label=label)
        ax.set_xticks(range(len(EPF))); ax.set_xticklabels([str(e) for e in EPF]); ax.set_ylim(0, None)
        ax.set_title(f"{ylabel} | {task}")
        if ri == 2: ax.set_xlabel("epoch")
axes[0, 1].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, title="left column (kdyck)", title_fontsize=8.5, alignment="left")
axes[1, 1].legend(*axes[1, 1].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, title="right column (ksd)", title_fontsize=8.5, alignment="left")
headline(fig, "Random-init early blocks scramble tokens from step 0; every winner starts transparent and opens over 10-30 epochs",
         "Blocks 1-7, 256 fixed training images, per-epoch checkpoints. On ksd the MLPs wake within ~2 epochs and only the token-specific attention stays low for ~10.", y=0.985)
save(fig, "f5_functional_state")

# ---------------- F6: mediator screen (kdyck early-lever family)
famk = [r for r in rows if r["family"] and r["task"] == "kdyck"]
CAND = [("lens7", "head lens at block 7, epoch 29"), ("probe7", "trained probe at block 7, epoch 29"), ("drift_k9", "realised change of k, epoch 9"), ("drift_in19", "realised change of q, k, fc1, epoch 19"),
        ("drift_in9", "realised change of q, k, fc1, epoch 9"), ("entropy0", "attention entropy after epoch 0"), ("gelu29", "GELU rms of blocks 1-7, epoch 29"), ("D9", "change of token differences D, epoch 9"),
        ("att_spec9", "token-specific attention write, epoch 9"), ("att_common9", "common attention write, epoch 9"), ("train_loss", "final training loss (fit deficit)")]
y = np.array([r["final"] for r in famk]); base = rankdata([r["lens7"] for r in famk]); res = []
for k, label in CAND:
    x = np.array([r[k] for r in famk], float); rho = spearmanr(x, y)[0]; rx, ry = rankdata(x), rankdata(y)
    part = np.nan if k == "lens7" else float(np.corrcoef(rx - np.polyval(np.polyfit(base, rx, 1), base), ry - np.polyval(np.polyfit(base, ry, 1), base))[0, 1]); res.append((label, rho, part))
res.sort(key=lambda t: abs(t[1])); fig, ax = plt.subplots(figsize=(9.2, 5.6)); fig.subplots_adjust(top=0.84, left=0.40, right=0.96)
yy = np.arange(len(res)); ax.barh(yy + 0.19, [t[1] for t in res], height=0.32, color=C["committed"], label="Spearman with the last-epoch accuracy", zorder=3)
ax.barh(yy - 0.19, [0 if np.isnan(t[2]) else t[2] for t in res], height=0.32, color=C["prefix"], label="partial, given the lens at block 7", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in res]); ax.axvline(0, color=AXIS, lw=1.0); ax.set_xlim(-1, 1); ax.grid(axis="y", visible=False); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8.5)
for i, t in enumerate(res):
    ax.text(t[1] + (0.02 if t[1] >= 0 else -0.02), i + 0.19, f"{t[1]:+.2f}", va="center", ha="left" if t[1] >= 0 else "right", fontsize=8, color=INK2)
headline(fig, "kdyck early-lever arms: slow realised steps of q, k, fc1 predict the final beyond what the lens explains", f"{len(famk)} runs whose blocks 9-11 are untouched. Rank correlations of early-phase quantities with the last-epoch accuracy; mean over blocks 1-7.")
save(fig, "f6_mediator_screen")

# ---------------- F7: realised drift against the final, early-lever family
fig, ax = plt.subplots(figsize=(8.2, 5.4)); fig.subplots_adjust(top=0.84, right=0.96)
fam_all = [r for r in rows if r["family"]]
for low, col, lab in ((True, C["committed"], "lens at block 7, epoch 29 <= 30"), (False, MUTED, "lens > 30")):
    for task, mk in (("kdyck", "o"), ("ksd", "^")):
        pts = [r for r in fam_all if (r["lens7"] <= 30) == low and r["task"] == task]
        ax.scatter([r["drift_k9"] for r in pts], [r["final"] for r in pts], s=62, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.3, zorder=3, label=lab if task == "kdyck" else None)
ax.scatter([], [], s=58, marker="^", color=INK2, edgecolor=SURFACE, label="triangles: ksd runs")
for arm, (dx, dy) in {"ftbanapermb7i": (1.6, 0.42), "ftbana": (1.6, -0.45), "ftbanak": (2.2, -0.55), "ftbanakpermb7i": (-0.3, -0.75), "ftbqu": (1.8, 0.45)}.items():
    pts = [r for r in fam_all if r["arm"] == arm]
    if pts:
        x, y = np.mean([p["drift_k9"] for p in pts]), np.mean([p["final"] for p in pts])
        ax.annotate(NAMES[arm], (x, y), xytext=(x + dx, y + dy), fontsize=8.5, color=INK2, va="center", ha="left" if dx > 0 else "right", path_effects=HALO, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=4))
ax.set_xlabel("realised relative change of k in blocks 1-7 from epoch 9 to 10 (%)"); ax.set_ylabel("last-epoch top-1 (%)"); ax.legend(loc="lower left", fontsize=8.5)
headline(fig, "Slow early k goes with a gain on kdyck, but slow steps alone do not make a ksd run win", f"Early-lever family, {len(fam_all)} runs. Spearman on kdyck {spearmanr([r['drift_k9'] for r in famk], [r['final'] for r in famk])[0]:+.2f} (n = {len(famk)}). The slow ksd runs at 77.6-78.3 are scale-only recipes without sink.")
save(fig, "f7_drift_vs_final")

# ---------------- F8: is the class information at block 7 in the class token or in the patches?
ORDER = ["ftbrhosl", "ftbvd", "ftbqu", "ftbana", "ftbanak", "ftbqmln", "ftbanaperab7w", "ftbqmlnvo", "ftbanap", "ftbanaperab7i", "ftbrhop", "ftbanakpermb7i", "ftbanapermb7i", "ftb4i_kdyck", "ftb4i"]
LAB8 = {"ftbrhosl": "random-level (slow late) 78.3", "ftbvd": "random, v x0.46  78.7", "ftbqu": "random-level (q,k x2.2) 78.1", "ftbana": "ftbana 76.6", "ftbanak": "ksd scale-only 77.9", "ftbqmln": "ftbqmln 78.2",
        "ftbanaperab7w": "write-matched 78.9", "ftbqmlnvo": "ftbqmlnvo 79.9", "ftbanap": "scale-only 80.2", "ftbanaperab7i": "active-unit twin 79.6", "ftbrhop": "late lever 79.9", "ftbanakpermb7i": "committed ksd 79.9",
        "ftbanapermb7i": "committed kdyck 80.4", "ftb4i_kdyck": "kdyck prefix 79.9", "ftb4i": "ksd prefix 80.1"}
arms8 = [a for a in ORDER if f"{a}@29" in CLS]; fig, ax = plt.subplots(figsize=(8.8, 6.2)); fig.subplots_adjust(top=0.85, left=0.30, right=0.96)
yy = np.arange(len(arms8))[::-1]
ax.barh(yy + 0.19, [CLS[f"{a}@29"]["probe_cls"]["7"] for a in arms8], height=0.32, color=C["committed"], label="probe on the class token", zorder=3)
ax.barh(yy - 0.19, [CLS[f"{a}@29"]["probe_patch"]["7"] for a in arms8], height=0.32, color=C["prefix"], label="probe on the mean patch token", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([LAB8[a] for a in arms8]); ax.grid(axis="y", visible=False); ax.set_xlabel("trained linear probe at block 7, epoch 29: top-1 (%), 10 training / 5 validation images per class"); ax.legend(loc="lower right", fontsize=8.5)
headline(fig, "At epoch 29 the class is in the patch tokens as much as in the class token", "So the block-7 transient is early class alignment of the whole token stream, not a shortcut through the class token alone. Arms sorted from high to low transient.")
save(fig, "f8_cls_vs_patch")
