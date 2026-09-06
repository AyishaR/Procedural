"""Join verified accuracies (arm_truth.json) with init statistics (init_dist_stats.json,
init_forward_stats.json).  Prints the arm table with effective scales RELATIVE TO RANDOM INIT
(r_s0), block-averaged over 1-8 and separately for block 0, plus shape distances to proc."""
import json, sys, numpy as np, collections
S = "/home/schrodi/Procedural/plots/cache/verify"
D = "/home/schrodi/Procedural/results/init_dumps"
R = json.load(open(f"{S}/arm_truth.json"))
ST = json.load(open(f"{D}/init_dist_stats.json"))
try: FW = json.load(open(f"{D}/init_forward_stats.json"))
except Exception: FW = None
RBASE = 78.079
def acc(name):
    """clean seeds first; fall back to contaminated (flagged)."""
    cl = [r["acc_last"] for r in R if name in r["jobnames"] and r["max_epoch"] == 299 and r["acc_last"] and not r["pre_fix"]]
    pre = [r["acc_last"] for r in R if name in r["jobnames"] and r["max_epoch"] == 299 and r["acc_last"] and r["pre_fix"]]
    if cl: return np.mean(cl), len(cl), "clean"
    if pre: return np.mean(pre), len(pre), "PRE"
    return None, 0, "-"
# which arms draw from the torch RNG after DDP (contaminated when pre-fix)
def rng_edit(name):
    sig = [r["sig"] for r in R if name in r["jobnames"]][0]
    return bool(sig.get("weight_shuffle")) or (sig.get("quantile_1d_mode") not in (None, "skip")) or sig.get("quantile_source") == "parametric" or sig.get("quantile_1d_source") == "parametric"
MANUAL = {"r": (78.079, 3, "clean"), "p": (80.092, 3, "clean")}
ref = ST["r_s0"]
def rel(arm, key, blocks):
    return np.mean([ST[arm][str(b)][key] / ref[str(b)][key] for b in blocks])
def absavg(arm, key, blocks): return np.mean([ST[arm][str(b)][key] for b in blocks])
rows = []
for dump in sorted(ST):
    arm = dump.rsplit("_s", 1)[0]
    if arm in MANUAL: a, n, tag = MANUAL[arm]
    else:
        a, n, tag = acc(arm)
        if tag == "PRE" and not rng_edit(arm): tag = "clean*"   # pre-fix but no RNG edit -> unaffected
    d = None if a is None else a - RBASE
    B18 = range(1, 9); B0 = [0]
    rows.append(dict(dump=dump, arm=arm, delta=d, n=n, tag=tag,
        logit18=rel(dump, "logit_eff", B18), veff18=rel(dump, "v_eff", B18), write18=rel(dump, "attn_write_eff", B18),
        mlppre18=rel(dump, "fc1_eff", B18), mlpw18=rel(dump, "mlp_write_eff", B18), proj18=rel(dump, "proj_norm", B18),
        g1=absavg(dump, "gamma1_mean", B18), g2=absavg(dump, "gamma2_mean", B18),
        logit0=rel(dump, "logit_eff", B0), write0=rel(dump, "attn_write_eff", B0), mlpw0=rel(dump, "mlp_write_eff", B0),
        kurt_fc1=absavg(dump, "fc1.kurt", B18), kurt_v=absavg(dump, "v.kurt", B18),
        shape_v=absavg(dump, "v.w1_proc_shape", B18), shape_fc1=absavg(dump, "fc1.w1_proc_shape", B18),
        raw_v=absavg(dump, "v.w1_proc_raw", B18), raw_q=absavg(dump, "q.w1_proc_raw", B18), raw_fc1=absavg(dump, "fc1.w1_proc_raw", B18),
        srank_v=absavg(dump, "v.stable_rank", B18), srank_fc1=absavg(dump, "fc1.stable_rank", B18),
        bias_fc1=absavg(dump, "fc1bias_std", B18), qbias=absavg(dump, "qbias_std", B18)))
rows.sort(key=lambda r: (-(r["delta"] if r["delta"] is not None else -9)))
hdr = f"{'arm':16s} {'delta':>6s} {'n':>2s} {'tag':6s} | {'logit':>6s} {'v_eff':>6s} {'write':>6s} {'mlpPre':>6s} {'mlpW':>6s} {'proj':>5s} {'g1':>5s} {'g2':>5s} | {'lg0':>5s} {'wr0':>5s} {'mw0':>5s} | {'kFc1':>5s} {'kV':>4s} {'shV':>6s} {'shF1':>6s} {'rawV':>6s} {'rawQ':>6s} {'srV':>5s} {'srF1':>5s}"
print("effective scales are multiples of random init (r_s0); blocks 1-8 averaged; lg0/wr0/mw0 = block 0")
print(hdr)
for r in rows:
    d = f"{r['delta']:+.2f}" if r["delta"] is not None else "   -  "
    print(f"{r['dump']:16s} {d:>6s} {r['n']:>2d} {r['tag']:6s} | {r['logit18']:6.2f} {r['veff18']:6.2f} {r['write18']:6.2f} {r['mlppre18']:6.2f} {r['mlpw18']:6.2f} {r['proj18']:5.2f} {r['g1']:5.2f} {r['g2']:5.2f} | {r['logit0']:5.2f} {r['write0']:5.2f} {r['mlpw0']:5.2f} | {r['kurt_fc1']:5.2f} {r['kurt_v']:4.2f} {r['shape_v']:6.3f} {r['shape_fc1']:6.3f} {r['raw_v']:6.4f} {r['raw_q']:6.4f} {r['srank_v']:5.0f} {r['srank_fc1']:5.0f}")
json.dump(rows, open(f"{S}/join_rows.json", "w"), indent=1)
if FW:
    print("\nforward pass at init (64 val images): rho_attn / rho_mlp per block 0..11, attention entropy (max 5.28)")
    for r in rows:
        f = FW.get(r["dump"]); 
        if not f: continue
        d = f"{r['delta']:+.2f}" if r["delta"] is not None else "   -  "
        ra = " ".join(f"{f[str(i)]['rho_attn']:.2f}" for i in range(12)); rm = " ".join(f"{f[str(i)]['rho_mlp']:.2f}" for i in range(12))
        en = " ".join(f"{f[str(i)]['attn_entropy']:.1f}" for i in range(12))
        print(f"{r['dump']:16s} {d:>6s} rho_a {ra}\n{'':23s} rho_m {rm}\n{'':23s} ent   {en}   |r_out| b8 {f['8']['rout']:.0f} b11 {f['11']['rout']:.0f}")
