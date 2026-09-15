"""Consistency check of the mechanism across ALL arms at the level of training dynamics (2026-09-15).
Per arm: (1) training-loss deficit vs random and test-acc gap vs random at matched epochs (log.txt);
(2) the mid-block readout transient: head-probe accuracy of block 7's output, its peak over training and
value at epochs 99/289; block-11 probe at 49/99/149/289; (3) mean attention entropy of blocks 1-8 at epochs
9/49/149; (4) mean MLP / attention write ratio of blocks 1-8 at epochs 9/19/49/99 (wandb per-layer cache).
Reads plots/cache/verify/wandb_layerwise.json (plots/verify/wandb_layerwise.py) and the run logs.
usage: .venv/bin/python plots/verify/dynamics_consistency.py"""
import json, os, re, numpy as np
ROOT = "/home/schrodi/Procedural"
C = json.load(open(f"{ROOT}/plots/cache/verify/wandb_layerwise.json")); LAST = 289
src = open(f"{ROOT}/plots/verify/wandb_layerwise.py").read()
IDS = {m.group(1): [(int(a), int(b)) for a, b in re.findall(r'\((\d+), (\d)', m.group(2))]
       for m in re.finditer(r'"(\w+)": \[((?:\(\d+, \d(?:, SHUF)?\),? ?)+)\]', src)}
ORDER = ["r", "p", "ftb3i", "ftb11h", "ftbqmlnvo", "ftbqmlnvog", "ftbqmlnvot", "ftbanag", "ftbanap", "ftbanal", "ftbanai", "ftbanau", "ftbanab", "ftbanaf",
         "ftbana", "ftblrm", "ftbrhos", "ftb7h", "ftb3h", "ftb3b", "ftbrho", "ftbrhop", "ftbrhopl", "ftbrhosl", "ftbcomp11", "ftbanac",
         "ftb4", "ftb4i", "ftb4h", "ftbanak", "ftbanakw", "ftbanakg", "ftbqmlnvok", "ftbanakb"]
def log(sid, seed):
    p = f"{ROOT}/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/log.txt"
    if not os.path.exists(p): return {}
    return {x["epoch"]: (x["test_acc1"], x["train_loss"]) for x in (json.loads(l) for l in open(p)) if "test_acc1" in x}
def seed_mean(arm, fam, layers=range(12)):
    per = []
    for s, d in C.get(arm, {}).items():
        m = {}
        for e, row in d.items():
            if int(e) > LAST: continue
            v = np.array([row.get(f"{fam}_layer{l}", np.nan) for l in layers], float); v[v == -1] = np.nan
            if not np.all(np.isnan(v)): m[int(e)] = v
        if fam == "acc":   # drop the end-of-segment artefact rows (a resumed job measures a random model)
            es = sorted(m); m = {e: v for i, e in enumerate(es) for v in [m[e]] if not (i > 0 and e > 20 and np.nanmax(v) < 0.5 * np.nanmax(m[es[i - 1]]))}
        per.append(m)
    if not per: return None, None
    eps = sorted(set.union(*[set(m) for m in per])); eps = [e for e in eps if sum(e in m for m in per) >= max(1, len(per) - 1)]
    return eps, np.array([np.nanmean([m[e] for m in per if e in m], 0) for e in eps])
EPS = [49, 99, 149, 199, 249, 299]
R = [l for l in (log(s, d) for s, d in IDS["r"]) if l]
rm = {e: (np.mean([l[e][0] for l in R if e in l]), np.mean([l[e][1] for l in R if e in l])) for e in EPS}
print("random (n=%d): " % len(R) + "  ".join(f"{e}: {rm[e][0]:.2f}/{rm[e][1]:.3f}" for e in EPS))
print("\n(1) train-loss deficit vs random (+ = fits worse) / test-acc gap vs random")
print(f"{'arm':11s} n | " + " | ".join(f"{'ep' + str(e):>13s}" for e in EPS) + " | final (vs r)")
for a in ORDER:
    if a == "r" or a not in IDS: continue
    L = [l for l in (log(s, d) for s, d in IDS[a]) if l]
    if not L: continue
    cells = []
    for e in EPS:
        vals = [l[e] for l in L if e in l]
        cells.append(f"{np.mean([v[1] for v in vals]) - rm[e][1]:+.3f}/{np.mean([v[0] for v in vals]) - rm[e][0]:+5.2f}" if vals else "      -      ")
    fin = [l[299][0] for l in L if 299 in l]
    print(f"{a:11s} {len(L)} | " + " | ".join(cells) + (f" | {np.mean(fin):.2f} ({np.mean(fin) - rm[299][0]:+.2f})" if fin else " | running"))
print("\n(2-4) per-layer traces: block-7 probe peak (epoch) / at 99 / at 289 | block-11 probe 49/99/149/289 | attn entropy blocks 1-8 at 9/49/149 | MLP write ratio blocks 1-8 at 9/19/49/99 | attn write ratio at 9/19/49/99")
for a in ORDER:
    eps, A = seed_mean(a, "acc")
    if eps is None: print(f"{a:11s} (no trace)"); continue
    b7 = A[:, 7]; pk = int(np.nanargmax(b7)); g = lambda e, col: (f"{A[eps.index(e), col]:5.1f}" if e in eps else "  -  ")
    ee, E = seed_mean(a, "attn_entropy", range(1, 9)); ge = lambda e: (f"{np.nanmean(E[ee.index(e)]):4.2f}" if (ee and e in ee) else " -  ")
    em, Mw = seed_mean(a, "delta_norm_ratio", range(1, 9)); gm = lambda e: (f"{np.nanmean(Mw[em.index(e)]):4.2f}" if (em and e in em) else " -  ")
    ea, Aw = seed_mean(a, "attn_delta_norm_ratio", range(1, 9)); ga = lambda e: (f"{np.nanmean(Aw[ea.index(e)]):4.2f}" if (ea and e in ea) else " -  ")
    print(f"{a:11s} | {b7[pk]:5.1f} ({eps[pk]:3d}) / {g(99, 7)} / {g(289, 7)} | {g(49, 11)} / {g(99, 11)} / {g(149, 11)} / {g(289, 11)} | {ge(9)} / {ge(49)} / {ge(149)} | {gm(9)} {gm(19)} {gm(49)} {gm(99)} | {ga(9)} {ga(19)} {ga(49)} {ga(99)}")
