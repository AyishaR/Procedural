"""Per-arm training trajectories from log.txt: acc at checkpoints, final train loss, and the
pre-fix vs clean comparison for the arms that have both."""
import json, os, numpy as np, collections
S = "/home/schrodi/Procedural/plots/cache/verify"; ROOT = "/home/schrodi/Procedural"
R = json.load(open(f"{S}/arm_truth.json"))
def rng_edit(sig):
    return bool(sig.get("weight_shuffle")) or (sig.get("quantile_1d_mode") not in (None, "skip")) or sig.get("quantile_source") == "parametric" or sig.get("quantile_1d_source") == "parametric"
EP = [9, 29, 49, 99, 149, 199, 249, 299]
per = collections.defaultdict(list)   # (arm, status) -> list of dict
for r in R:
    if r["max_epoch"] != 299 or not r["output_dir"]: continue
    rows = {}
    for l in open(os.path.join(ROOT, r["output_dir"], "log.txt")):
        try: rr = json.loads(l)
        except Exception: continue
        if "epoch" in rr: rows[rr["epoch"]] = rr
    if 299 not in rows or "test_acc1" not in rows[299]: continue
    status = "PRE" if (r["pre_fix"] and rng_edit(r["sig"])) else "clean"
    d = {"acc": {e: rows[e].get("test_acc1") for e in EP if e in rows}, "train_loss": rows[299]["train_loss"],
         "train_loss_min": min(v["train_loss"] for v in rows.values()), "test_loss": rows[299].get("test_loss")}
    for jn in r["jobnames"]: per[(jn, status)].append(d)
# baselines
for sid, name in [("29384839", "r"), ("29377576", "p")]:
    for s in range(3):
        rows = {}
        for l in open(f"{ROOT}/results/imnet_base/results_IMNET_BASE_{sid}/s{s}/log.txt"):
            try: rr = json.loads(l)
            except Exception: continue
            if "epoch" in rr: rows[rr["epoch"]] = rr
        per[(name, "clean")].append({"acc": {e: rows[e].get("test_acc1") for e in EP if e in rows}, "train_loss": rows[299]["train_loss"],
                                    "train_loss_min": min(v["train_loss"] for v in rows.values()), "test_loss": rows[299].get("test_loss")})
def m(lst, f): 
    v = [f(x) for x in lst if f(x) is not None]; return np.mean(v) if v else float("nan")
print(f"{'arm':12s} {'st':5s} {'n':>2s} " + " ".join(f"ep{e:>3d}" for e in EP) + "   trainL  testL")
out = {}
for (arm, st), lst in sorted(per.items(), key=lambda kv: -m(kv[1], lambda x: x['acc'].get(299))):
    accs = [m(lst, lambda x, e=e: x["acc"].get(e)) for e in EP]
    tl = m(lst, lambda x: x["train_loss"]); te = m(lst, lambda x: x["test_loss"])
    out[f"{arm}|{st}"] = dict(n=len(lst), acc=dict(zip(EP, accs)), train_loss=tl, test_loss=te)
    print(f"{arm:12s} {st:5s} {len(lst):>2d} " + " ".join(f"{a:5.1f}" for a in accs) + f"   {tl:.3f}  {te:.3f}")
json.dump(out, open(f"{S}/trajectories.json", "w"), indent=1)
print("\n--- pre-fix (contaminated) vs clean, same flags ---")
for arm in ["ftb4e3", "ftbqm1d", "ftbqmln", "ftbqm1dvo", "ftb11is"]:
    pre = per.get((arm, "PRE")) or per.get((arm + "fix", "PRE")); cl = per.get((arm, "clean")) or per.get((arm + "fix", "clean"))
    if not pre or not cl: continue
    print(f"{arm:10s} acc299 PRE {m(pre, lambda x: x['acc'][299]):.2f} clean {m(cl, lambda x: x['acc'][299]):.2f} | trainL PRE {m(pre, lambda x: x['train_loss']):.3f} clean {m(cl, lambda x: x['train_loss']):.3f} | ep49 PRE {m(pre, lambda x: x['acc'].get(49)):.2f} clean {m(cl, lambda x: x['acc'].get(49)):.2f}")
