"""Summarise end_state_stats.json: per-arm (seed-mean) end-of-training depth profiles of the
MLP LayerNorm gain, rho_mlp, rho_attn, attention entropy; and correlate depth-allocation
scalars with final accuracy across all arms."""
import json, numpy as np, collections, sys
from scipy import stats
S = "/home/schrodi/Procedural/plots/cache/verify"
ES = json.load(open("/home/schrodi/Procedural/results/init_dumps/end_state_stats.json"))
T = json.load(open(f"{S}/trajectories.json"))
# accuracy per label (arm or arm_PRE), from trajectories.json (already seed-mean, clean/PRE split)
def acc_of(label):
    arm, st = (label[:-4], "PRE") if label.endswith("_PRE") else (label, "clean")
    k = f"{arm}|{st}"
    if k in T: return T[k]["acc"]["299"], T[k]["train_loss"], T[k]["n"]
    return None, None, 0
by_arm = collections.defaultdict(list)
for lab, d in ES.items(): by_arm[lab.rsplit("_s", 1)[0]].append(d)
def prof(ds, section, key): return np.mean([[d[section][str(b)][key] for b in range(12)] for d in ds], 0)
rows = []
for arm, ds in by_arm.items():
    a, tl, n = acc_of(arm)
    if a is None: continue
    g2 = prof(ds, "weights", "gamma2_mean"); g1 = prof(ds, "weights", "gamma1_mean"); rm = prof(ds, "forward", "rho_mlp"); ra = prof(ds, "forward", "rho_attn"); en = prof(ds, "forward", "attn_entropy")
    lg = prof(ds, "weights", "logit_eff"); mw = prof(ds, "weights", "mlp_write_eff"); aw = prof(ds, "weights", "attn_write_eff")
    rows.append(dict(arm=arm, acc=a, delta=a - 78.079, trainL=tl, n=n, nck=len(ds), g2=g2, g1=g1, rho_mlp=rm, rho_attn=ra, ent=en, logit=lg, mlpw=mw, attnw=aw))
rows.sort(key=lambda r: -r["acc"])
def fmt(v): return " ".join(f"{x:4.2f}" for x in v)
print("END STATE (epoch 299), seed-mean.  gamma2 = MLP LayerNorm gain mean per block 0..11")
print(f"{'arm':14s} {'delta':>6s} {'n':>2s}  gamma2 per block")
for r in rows: print(f"{r['arm']:14s} {r['delta']:+6.2f} {r['nck']:>2d}  {fmt(r['g2'])}")
print(f"\n{'arm':14s} {'delta':>6s}  rho_mlp per block (64 val images)")
for r in rows: print(f"{r['arm']:14s} {r['delta']:+6.2f}  {fmt(r['rho_mlp'])}")
print(f"\n{'arm':14s} {'delta':>6s}  rho_attn per block")
for r in rows: print(f"{r['arm']:14s} {r['delta']:+6.2f}  {fmt(r['rho_attn'])}")
print(f"\n{'arm':14s} {'delta':>6s}  attention entropy per block (uniform = 5.28)")
for r in rows: print(f"{r['arm']:14s} {r['delta']:+6.2f}  {fmt(r['ent'])}")
# depth-allocation scalars vs accuracy
print("\ncorrelations with final accuracy across arms (Spearman):")
cands = {"gamma2 mean blocks 6-8": lambda r: r["g2"][6:9].mean(), "gamma2 mean blocks 9-10": lambda r: r["g2"][9:11].mean(),
         "gamma2 ratio (9-10)/(6-8)": lambda r: r["g2"][9:11].mean() / r["g2"][6:9].mean(),
         "rho_mlp mean blocks 6-8": lambda r: r["rho_mlp"][6:9].mean(), "rho_mlp mean blocks 9-11": lambda r: r["rho_mlp"][9:12].mean(),
         "rho_mlp share of blocks 9-11": lambda r: r["rho_mlp"][9:12].sum() / r["rho_mlp"].sum(),
         "rho_attn mean blocks 0-8": lambda r: r["rho_attn"][0:9].mean(), "rho_attn mean blocks 9-11": lambda r: r["rho_attn"][9:12].mean(),
         "entropy mean blocks 0-8": lambda r: r["ent"][0:9].mean(), "entropy mean blocks 9-11": lambda r: r["ent"][9:12].mean(),
         "gamma1 mean blocks 0-8": lambda r: r["g1"][0:9].mean(), "sum rho_mlp all": lambda r: r["rho_mlp"].sum(), "sum rho_attn all": lambda r: r["rho_attn"].sum(),
         "final train loss": lambda r: r["trainL"]}
sub = [r for r in rows if not r["arm"].endswith("_PRE")]
for name, f in cands.items():
    x = np.array([f(r) for r in sub]); y = np.array([r["acc"] for r in sub])
    rs, p = stats.spearmanr(x, y); print(f"  {name:32s} rho={rs:+.2f} p={p:.1e} (n={len(x)})")
json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows], open(f"{S}/end_state_rows.json", "w"), indent=1)
