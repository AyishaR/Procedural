"""Phase-0 look at EXISTING training traces for the early-lever mechanism plan (docs/early_lever_mechanism_plan.md).
No training, no GPU: reads plots/cache/verify/wandb_layerwise.json (per-block lens accuracy, attention entropy, write ratios,
per (arm, seed, epoch)), the runs' log.txt (test_acc1, train_loss) and grad_norms json.
usage: .venv/bin/python plots/verify/mechanism_phase0.py [section ...]   sections: lens profile persist early grad"""
import json, os, re, sys
import numpy as np
ROOT = "/home/schrodi/Procedural"
D = json.load(open(f"{ROOT}/plots/cache/verify/wandb_layerwise.json"))
src = open(f"{ROOT}/plots/verify/wandb_layerwise.py").read()
PROJECT, SHUF = "p", "s"
ARMS = eval("{" + src.split("ARMS = {", 1)[1].split("\n}\n", 1)[0] + "\n}")
def log_of(sid, seed):
    out = {}; lf = f"{ROOT}/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/log.txt"
    if not os.path.exists(lf): return out
    for l in open(lf):
        try: d = json.loads(l)
        except Exception: continue
        if "epoch" in d: out[int(d["epoch"])] = d
    return out
def final(arm):
    vals = []
    for t in ARMS.get(arm, []):
        lg = log_of(t[0], t[1])
        if 299 in lg and "test_acc1" in lg[299]: vals.append(lg[299]["test_acc1"])
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)
def series(arm, key, seed=None):
    """{epoch: mean over seeds} of one cached metric key"""
    out = {}
    for s, eps in D.get(arm, {}).items():
        if seed is not None and s != str(seed): continue
        for e, row in eps.items():
            if key in row and row[key] is not None: out.setdefault(int(e), []).append(row[key])
    return {e: float(np.mean(v)) for e, v in sorted(out.items())}
KSD = {a for a, ts in ARMS.items() if any(len(t) > 2 and t[2] == SHUF for t in ts)} | {"ftb4i"}
sections = sys.argv[1:] or ["lens", "profile", "persist", "early"]

if "lens" in sections:
    print("== A. block-7 lens transient (max of acc_layer7 over epochs <= 99) against the last-epoch accuracy, every cached arm with a final")
    rows = []
    for arm in sorted(D):
        f, n = final(arm); s = {e: v for e, v in series(arm, "acc_layer7").items() if e <= 99}
        if n == 0 or not s: continue
        pe = max(s, key=s.get); rows.append((arm, "ksd" if arm in KSD else "kdyck", f, n, s[pe], pe, s.get(19, float("nan")), s.get(29, float("nan"))))
    rows.sort(key=lambda r: -r[2])
    print(f"   {'arm':16s} task   final (n)   lens7 peak (epoch)   lens7@19  lens7@29")
    for r in rows: print(f"   {r[0]:16s} {r[1]:5s}  {r[2]:5.2f} ({r[3]})   {r[4]:5.1f} ({r[5]:3d})          {r[6]:5.1f}    {r[7]:5.1f}")
    from scipy.stats import spearmanr, pearsonr
    for name, sel in (("all", rows), ("kdyck", [r for r in rows if r[1] == "kdyck"]), ("ksd", [r for r in rows if r[1] == "ksd"]),
                      ("all, final >= 77.5 (undamaged)", [r for r in rows if r[2] >= 77.5])):
        x = np.array([r[4] for r in sel]); y = np.array([r[2] for r in sel])
        print(f"   {name}: n = {len(sel)}  Spearman(lens7 peak, final) = {spearmanr(x, y)[0]:+.2f}   Pearson = {pearsonr(x, y)[0]:+.2f}   | lens7@29: Spearman = {spearmanr([r[7] for r in sel], y)[0]:+.2f}")

if "profile" in sections:
    print("== B. where in depth the class becomes decodable: lens accuracy of blocks 3, 5, 7, 8, 9, 10, 11 at epochs 9, 19, 29, 49, 99, 299")
    for arm in ("r", "ftb4i_kdyck", "ftbanapermb7i", "ftbanaperab7i", "ftbanap", "ftbana", "ftb4i", "ftbanakpermb7i", "ftbanakperab7i", "ftbanak", "ftbanakperab7w"):
        if arm not in D: continue
        f, n = final(arm); print(f"   {arm} (final {f:.2f}, n={n})")
        for e in (9, 19, 29, 49, 99, 299):
            vals = [series(arm, f"acc_layer{b}").get(e, float("nan")) for b in (3, 5, 7, 8, 9, 10, 11)]
            print(f"      epoch {e:3d}: " + "  ".join(f"b{b} {v:5.1f}" for b, v in zip((3, 5, 7, 8, 9, 10, 11), vals)))

if "persist" in sections:
    print("== C. how long the installed state lasts: mean over blocks 1-7 of attention entropy (nats) and of the MLP / attention write ratios")
    for arm in ("r", "ftb4i_kdyck", "ftbanapermb7i", "ftbanaperab7i", "ftbanap", "ftbana", "ftbanal", "ftb4i", "ftbanakpermb7i", "ftbanakperab7i", "ftbanak"):
        if arm not in D: continue
        print(f"   {arm}")
        for fam, label in (("attn_entropy", "entropy   "), ("delta_norm_ratio", "MLP write "), ("attn_delta_norm_ratio", "attn write")):
            line = []
            for e in (4, 9, 19, 29, 49, 99, 199, 299):
                v = [series(arm, f"{fam}_layer{b}").get(e) for b in range(1, 8)]; v = [x for x in v if x is not None]
                line.append(f"{e}: {np.mean(v):6.3f}" if v else f"{e}:    -  ")
            print(f"      {label} " + "  ".join(line))

if "early" in sections:
    print("== D. slow starters: test accuracy at epochs 9 / 19 / 49 / 99 and train loss at 19 / 99 / 299, against the final")
    for arm in ("r", "ftb4i_kdyck", "ftbanapermb7i", "ftbanapermb7", "ftbanaperab7i", "ftbanaperab7", "ftbanap", "ftbanal", "ftbana", "ftb4i", "ftbanakpermb7i", "ftbanakperab7i", "ftbanaksg", "ftbanaks", "ftbanak"):
        ts = ARMS.get(arm, []);
        if not ts: continue
        lg = log_of(ts[0][0], ts[0][1])
        if 299 not in lg: continue
        g = lambda e, k: lg.get(e, {}).get(k, float("nan"))
        print(f"   {arm:16s} acc@9 {g(9,'test_acc1'):5.1f}  @19 {g(19,'test_acc1'):5.1f}  @49 {g(49,'test_acc1'):5.1f}  @99 {g(99,'test_acc1'):5.1f}  final {g(299,'test_acc1'):5.2f} | train loss @19 {g(19,'train_loss'):.3f}  @99 {g(99,'train_loss'):.3f}  @299 {g(299,'train_loss'):.3f}")
