"""Generalisation at matched fit, over training: for each arm epoch e, find the epoch of the random baseline
(3-seed mean trajectory) with the same train loss (linear interpolation) and report test loss(arm, e) minus
test loss(r at that train loss). Negative = the arm generalises better than random at the same fit.
Also the plain train-loss deficit vs r at the same epoch. Sources: results/.../log.txt of every seed."""
import json, numpy as np, statistics as st, sys
ROOT = "/home/schrodi/Procedural"
S = json.load(open(f"{ROOT}/plots/cache/verify/synthesis.json"))
AT = json.load(open(f"{ROOT}/plots/cache/verify/arm_truth.json")); items = AT if isinstance(AT, list) else list(AT.values())
OLD = json.load(open(f"{ROOT}/plots/cache/verify/old_runs_named.json"))
def dirs(arm):
    out = [x["output_dir"] for x in items if x["jobnames"][0] == arm and x.get("max_epoch") == 299 and "pr_6463456" not in (x["sig"].get("initialize") or "")]
    if not out: out = [x["ckpt"].rsplit("/", 1)[0] for x in OLD if x["label"] == arm]
    if not out:   # still running: take the output_dir from the newest slurm log
        import glob, re
        for f in sorted(glob.glob(f"{ROOT}/logs/ft_*_{arm}.out"), reverse=True):
            m = re.search(r"output_dir='([^']+)'", open(f, errors="ignore").read(20000))
            if m: out = [m.group(1)]; break
    return out
def traj(arm):
    per = []
    for d in dirs(arm):
        rows = {}
        for l in open(f"{ROOT}/{d}/log.txt"):
            try: r = json.loads(l); rows[r["epoch"]] = r
            except Exception: pass
        per.append(rows)
    eps = sorted(set.intersection(*[set(p) for p in per]))
    tl = np.array([st.mean(p[e]["train_loss"] for p in per) for e in eps])
    te = np.array([st.mean(p[e]["test_loss"] for p in per) if all(p[e].get("test_loss") is not None for p in per) else np.nan for e in eps])
    acc = np.array([st.mean(p[e]["test_acc1"] for p in per) if all(p[e].get("test_acc1") is not None for p in per) else np.nan for e in eps])
    return np.array(eps), tl, te, acc, len(per)
er, tlr, ter, accr, nr = traj("r")
ok = ~np.isnan(ter); tlr_o, ter_o = tlr[ok], ter[ok]; order = np.argsort(tlr_o)
def r_test_at(train_loss):   # interpolate r's test loss at a given train loss (r's train loss decreases monotonically)
    if train_loss < tlr_o.min() or train_loss > tlr_o.max(): return np.nan
    return float(np.interp(train_loss, tlr_o[order], ter_o[order]))
ARMS = sys.argv[1:] or ["p", "ftb3i", "ftb1i", "ftbcomp11", "ftbqmlnvo", "ftbqmlnvog", "ftbqmlnvot", "ftb4e3fix", "ftbrho", "ftb3b", "ftb2b", "ftb11h", "ftb7h", "ftb3h", "ftbvd", "ftbvu", "ftbqmln", "ftbrhos", "ftblrm", "ftbana", "ftbanaf", "ftbanab", "ftbanag"]
EPS = [29, 49, 99, 149, 199, 249, 299]
MD = ["\n## T7. Generalisation at matched fit over training (plots/verify/matched_fit.py)\n", "Left: test loss minus r's test loss at the same train loss (negative = better generalisation at matched fit). Right: train-loss deficit vs r at the same epoch.\n", "| arm | n | " + " | ".join(f"{e}" for e in EPS) + " | | " + " | ".join(f"{e}" for e in EPS) + " |", "|---|---|" + "---|" * len(EPS) + "---|" + "---|" * len(EPS)]
print("test loss minus r's test loss at the SAME train loss (negative = better generalisation at matched fit) | train-loss deficit vs r at the same epoch")
print(f"{'arm':11s} n  " + "".join(f"{e:>7d}" for e in EPS) + "   |  " + "".join(f"{e:>7d}" for e in EPS))
for arm in ARMS:
    try: e, tl, te, acc, n = traj(arm)
    except Exception as ex: print(arm, "no data", ex); continue
    row = []; row2 = []
    for E in EPS:
        if E in e:
            i = list(e).index(E); j = list(er).index(E) if E in er else None
            row.append(te[i] - r_test_at(tl[i]) if not np.isnan(te[i]) else np.nan); row2.append(tl[i] - tlr[j] if j is not None else np.nan)
        else: row.append(np.nan); row2.append(np.nan)
    print(f"{arm:11s} {n}  " + "".join(f"{v:+7.3f}" if not np.isnan(v) else f"{'-':>7s}" for v in row) + "   |  " + "".join(f"{v:+7.3f}" if not np.isnan(v) else f"{'-':>7s}" for v in row2))
    MD.append(f"| `{arm}` | {n} | " + " | ".join(f"{v:+.3f}" if not np.isnan(v) else "-" for v in row) + " | | " + " | ".join(f"{v:+.3f}" if not np.isnan(v) else "-" for v in row2) + " |")
open(f"{ROOT}/docs/synthesis_data.md", "a").write("\n".join(MD) + "\n")
