"""Gather every number the synthesis document (docs/i100_synthesis.md) cites, from primary sources only:
  * plots/cache/verify/arm_truth.json  -- every run's init flags parsed from its slurm Namespace dump,
                                          last-epoch accuracy, train loss (refreshed by arm_truth.py)
  * results/imnet_base/<dir>/log.txt    -- per-epoch train loss / test acc / test loss
  * results/init_dumps/init_forward_stats.json -- forward-pass write ratios at init (64 val images)
  * plots/cache/verify/wandb_layerwise.json    -- per-block head-probe accuracy and write ratios per epoch
Writes docs/synthesis_data.md (tables) and plots/cache/verify/synthesis.json (the same numbers).
Arms are grouped by construction (from the Namespace flags), not by name."""
import json, glob, re, os, statistics as st, collections
import numpy as np
ROOT = "/home/schrodi/Procedural"
AT = json.load(open(f"{ROOT}/plots/cache/verify/arm_truth.json"))
items = AT if isinstance(AT, list) else list(AT.values())
F = json.load(open(f"{ROOT}/results/init_dumps/init_forward_stats.json"))
try:
    C = json.load(open(f"{ROOT}/plots/cache/verify/wandb_layerwise.json"))
except Exception:
    C = {}

# ---------------------------------------------------------------- construction from the Namespace flags
def describe(sig, name):
    s = sig
    init = s.get("initialize") or ""
    proc = "proc ckpt" if "pr_vitb_n" in init else ("none" if init == "" else os.path.basename(init))
    parts = []
    rb = s.get("random_blocks")
    if rb not in (None, "", []):
        parts.append(f"random blocks {rb}")
    im = s.get("init_method") or "default"
    if im != "default":
        parts.append(im)
    sb = s.get("init_method_scaled_blocks")
    if sb not in (None, "", []):
        parts.append(f"scaled blocks {sb}")
    for k in ("init_method_copied_blocks", "target_ratio_absolute", "target_ratio_scale", "target_ratio_flatten",
              "quantile_source", "quantile_1d_mode", "quantile_qkv_mode", "quantile_1d_source",
              "slice_scale_qk", "slice_scale_v", "slice_scale_proj", "weight_shuffle", "custom_init_type", "weight_init"):
        v = s.get(k)
        if v not in (None, "", [], -1, -1.0, 1.0, False, "skip", "empirical", "pooled"):
            parts.append(f"{k}={v}")
    return f"init={proc}; " + "; ".join(parts)

OVERRIDE = {  # arms whose defining flags are not in arm_truth's KEYS (verified from their run scripts)
    "ftblrm": "init=none; timm random; per-tensor lr scales rms(random)/rms(proc) on q/k/v/proj/fc1/fc2 of blocks 0-8 (--lr_match_ckpt)",
    "ftbana": "init=none; timm random; analytic profile: per-slice std multipliers, block 0 own + linear ramp blocks 1-8 (profile_ftbana.json), LN gains 1, biases 0",
    "ftbanaf": "ftbana + blocks 9-11 fc2 x0.30 (flat MLP top)",
    "ftbanab": "ftbana + proc LN biases (permuted) in blocks 0-8",
    "ftbanag": "ftbana + proc LN gains (permuted) in blocks 0-8, q/k/v/fc1 multipliers / rms(gamma)",
    "ftbanap": "ftbana + Gaussian-sampled LN gains and biases (proc per-block mean/std) in blocks 0-8",
}

# ---------------------------------------------------------------- per-run last-epoch numbers
def logrows(d):
    try:
        rows = {}
        for l in open(f"{ROOT}/{d}/log.txt"):
            try:
                r = json.loads(l); rows[r["epoch"]] = r
            except Exception:
                pass
        return rows
    except FileNotFoundError:
        return {}

runs = []
for x in items:
    if x.get("max_epoch") is None or x["max_epoch"] < 299 or x.get("acc_last") is None:
        continue
    if "pr_6463456" in (x["sig"].get("initialize") or ""):   # a coworker's ksd-checkpoint runs share results/, not ours
        continue
    rows = logrows(x["output_dir"]); r299 = rows.get(299, {})
    # the pre-2026-08-31 DDP bug (docs: ddp-rank-sync) only bites inits that draw from the torch RNG after the
    # DDP wrap: weight shuffles and quantile matching. Deterministic inits (block copies, delta-norm scaling) are valid.
    sg = x["sig"]; rng_edit = bool(sg.get("weight_shuffle")) or (sg.get("init_method") == "quantile_match_target_blocks") or (sg.get("quantile_1d_mode") not in (None, "", "skip"))
    contaminated = bool(x.get("pre_fix")) and rng_edit
    runs.append(dict(arm=x["jobnames"][0], seed=x["seed"], sid=x["slurm_id"], pre_fix=contaminated,
                     acc=x["acc_last"], train_loss=r299.get("train_loss", x.get("train_loss_last")), test_loss=r299.get("test_loss"),
                     desc=OVERRIDE.get(x["jobnames"][0], describe(x["sig"], x["jobnames"][0])), out=x["output_dir"], rows=rows))
print(f"{len(runs)} finished runs (epoch 299) in arm_truth")
# runs launched before the ft_* log convention (r, p, ftb3b, ...) live in old_runs_named.json (inventory_old.py)
have = {r["arm"] for r in runs}
for x in json.load(open(f"{ROOT}/plots/cache/verify/old_runs_named.json")):
    if x["label"] in have or x.get("acc") is None:
        continue
    d = os.path.dirname(x["ckpt"]); rows = logrows(d); r299 = rows.get(299, {})
    if 299 not in rows:
        continue
    runs.append(dict(arm=x["label"], seed=int(x["seed"].lstrip("s")), sid=x["slurm_id"], pre_fix=False, era_old=True,
                     acc=x["acc"], train_loss=r299.get("train_loss", x.get("train_loss")), test_loss=r299.get("test_loss"),
                     desc=OVERRIDE.get(x["label"], describe(x["sig"], x["label"])), out=d, rows=rows))
print(f"{len(runs)} runs after adding the older inventory")

# ---------------------------------------------------------------- arm table (clean era only, unless an arm has no clean seeds)
arms = collections.OrderedDict()
for r in sorted(runs, key=lambda r: (r["arm"], r["pre_fix"], r["seed"])):
    arms.setdefault((r["arm"], r["pre_fix"]), []).append(r)
R = [a for (n, p), a in arms.items() if n == "r" and not p]
if not R:
    R = [a for (n, p), a in arms.items() if n == "r"]
r_runs = R[0]; r_mean = st.mean(x["acc"] for x in r_runs)
table = []
for (name, pre), rs in arms.items():
    accs = [x["acc"] for x in rs]; tl = [x["train_loss"] for x in rs if x["train_loss"] is not None]; te = [x["test_loss"] for x in rs if x["test_loss"] is not None]
    table.append(dict(arm=name, era="contaminated" if pre else ("old" if rs[0].get("era_old") else "clean"), n=len(accs), acc=st.mean(accs), sd=(st.stdev(accs) if len(accs) > 1 else float("nan")),
                      d=st.mean(accs) - r_mean, train_loss=(st.mean(tl) if tl else float("nan")), test_loss=(st.mean(te) if te else float("nan")), desc=rs[0]["desc"]))
table.sort(key=lambda t: -t["acc"])

# ---------------------------------------------------------------- fit vs generalisation over all clean arms
clean = [t for t in table if t["era"] in ("clean", "old") and not np.isnan(t["train_loss"])]
x = np.array([t["train_loss"] for t in clean]); y = np.array([t["acc"] for t in clean]); z = np.array([t["test_loss"] for t in clean])
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
fitgen = dict(n=len(clean), pearson_acc_trainloss=float(np.corrcoef(x, y)[0, 1]), spearman_acc_trainloss=float(spearman(x, y)),
              pearson_testloss_trainloss=float(np.corrcoef(x, z)[0, 1]))
# residual of test loss on train loss: which arms are "damaged" (test loss higher than the line predicts)
A = np.vstack([x, np.ones_like(x)]).T; coef, *_ = np.linalg.lstsq(A, z, rcond=None); resid = z - A @ coef
fitgen["testloss_line"] = dict(slope=float(coef[0]), intercept=float(coef[1]), resid_sd=float(resid.std()))
for t, rz in zip(clean, resid):
    t["resid_testloss"] = float(rz)

# ---------------------------------------------------------------- trajectories for the representative arms
REP = ["r", "p", "ftb3i", "ftb1i", "ftbqmlnvo", "ftbqmlnvog", "ftbqmlnvot", "ftbrho", "ftb3b", "ftbcomp11", "ftb4o",
       "ftb3h", "ftb7h", "ftb11h", "ftb4e3fix", "ftbvd", "ftbvu", "ftbqu", "ftbqmln", "ftbnorm", "ftbrhos", "ftblrm", "ftbana", "ftbanaf", "ftbanab", "ftbanag", "ftbanap"]
EPS = [9, 29, 49, 99, 149, 199, 249, 299]
traj = {}
for name in REP:
    rs = arms.get((name, False)) or arms.get((name, True))
    if not rs:   # running arms: read their partial log directly
        for d in glob.glob(f"{ROOT}/logs/ft_*_{name}.out"):
            m = re.search(r"output_dir='([^']+)'", open(d, errors="ignore").read(20000))
            if m:
                rows = logrows(m.group(1))
                if rows: rs = [dict(rows=rows, seed=0)]; break
    if not rs:
        continue
    t = {}
    for e in EPS:
        acc = [x["rows"][e]["test_acc1"] for x in rs if e in x["rows"] and x["rows"][e].get("test_acc1") is not None]
        tl = [x["rows"][e]["train_loss"] for x in rs if e in x["rows"]]
        te = [x["rows"][e]["test_loss"] for x in rs if e in x["rows"] and x["rows"][e].get("test_loss") is not None]
        t[e] = dict(acc=(st.mean(acc) if acc else None), train_loss=(st.mean(tl) if tl else None), test_loss=(st.mean(te) if te else None))
    traj[name] = dict(n=len(rs), max_epoch=max(max(x["rows"]) for x in rs), at=t)

# ---------------------------------------------------------------- init profiles (forward stats) per arm with a dump
def prof(key):
    ra = np.array([F[key][str(b)]["rho_attn"] for b in range(12)]); rm = np.array([F[key][str(b)]["rho_mlp"] for b in range(12)])
    ent = np.array([F[key][str(b)]["attn_entropy"] for b in range(12)])
    return dict(attn_b0=float(ra[0]), attn_mid=float(ra[1:9].mean()), attn_top=float(ra[9:].mean()),
                mlp_b0=float(rm[0]), mlp_mid=float(rm[1:9].mean()), mlp_top=float(rm[9:].mean()), entropy_mid=float(ent[1:9].mean()),
                stream_at_9=float(F[key]["9"]["rin"]))
init = {k[:-3]: prof(k) for k in F if k.endswith("_s0")}

# ---------------------------------------------------------------- dynamics from wandb (per-block probe)
def seed_mean(arm, fam, last=289):
    per = []
    for s, d in C.get(arm, {}).items():
        m = {}
        for e, row in d.items():
            if int(e) > last: continue
            v = np.array([row.get(f"{fam}_layer{l}", np.nan) for l in range(12)], float); v[v == -1.0] = np.nan
            if not np.all(np.isnan(v)): m[int(e)] = v
        if fam == "acc":
            es = sorted(m); m = {e: v for i, e in enumerate(es) for v in [m[e]] if not (i > 0 and e > 20 and np.nanmax(v) < 0.5 * np.nanmax(m[es[i - 1]]))}
        if m: per.append(m)   # seeds whose wandb runs returned no rows (ftb1i s1/s2) are skipped
    if not per: return None, None
    eps = sorted(set.union(*[set(m) for m in per])); eps = [e for e in eps if sum(e in m for m in per) >= max(1, len(per) - 1)]
    return eps, np.array([np.nanmean([m[e] for m in per if e in m], 0) for e in eps])
dyn = {}
for arm in C:
    eps, A = seed_mean(arm, "acc")
    if eps is None or not eps: continue
    last = A[-1]; pk = int(np.nanargmax(A[:, 7]))
    d = dict(last_epoch=eps[-1], probe_b7_peak=float(A[pk, 7]), probe_b7_peak_epoch=eps[pk], probe_b7_last=float(last[7]),
             probe_b6to9_last=float(np.nanmean(last[6:10])), probe_b11_last=float(last[11]), probe_b10_last=float(last[10]))
    e2, M = seed_mean(arm, "delta_norm_ratio")
    if e2: d["mlp_write_last"] = dict(b0=float(M[-1][0]), mid=float(M[-1][1:9].mean()), top=float(M[-1][9:].mean()))
    e3, Aa = seed_mean(arm, "attn_delta_norm_ratio")
    if e3: d["attn_write_last"] = dict(b0=float(Aa[-1][0]), mid=float(Aa[-1][1:9].mean()), top=float(Aa[-1][9:].mean()))
    dyn[arm] = d

# ---------------------------------------------------------------- split series (from the parsed flags)
def blocks_of(v):
    return [int(b) for b in str(v).split(",") if b.strip().isdigit()] if v not in (None, "", []) else []
series = collections.defaultdict(list)
byname = {t["arm"]: t for t in table if t["era"] in ("clean", "old")}
for t in byname.values():
    rs = (arms.get((t["arm"], False)) or arms.get((t["arm"], True)))
    sig = None
    for x in items:
        if x["jobnames"][0] == t["arm"]: sig = x["sig"]; break
    if sig is None:
        for x in json.load(open(f"{ROOT}/plots/cache/verify/old_runs_named.json")):
            if x["label"] == t["arm"]: sig = x["sig"]; break
    if sig is None: continue
    rb, sb, im = blocks_of(sig.get("random_blocks")), blocks_of(sig.get("init_method_scaled_blocks")), sig.get("init_method") or "default"
    proc = "pr_vitb_n" in (sig.get("initialize") or "")
    tra = sig.get("target_ratio_absolute"); trs = sig.get("target_ratio_scale")
    key = None
    if proc and im == "default" and rb and not sig.get("weight_shuffle"):
        key = "proc prefix, random top (i-series)" if rb == list(range(min(rb), 12)) and min(rb) > 0 else ("random bottom, proc top (h-series)" if rb == list(range(0, max(rb) + 1)) else None)
    elif proc and im == "upscale_random_match_delta_norms" and rb and sb and (tra in (None, -1, -1.0)) and (trs in (None, 1.0)) and not sig.get("init_method_copied_blocks"):
        key = "random bottom, top upscaled to proc write ratios (b-series)" if sb == list(range(min(sb), 12)) else None
    elif proc and im == "downscale_pr_match_delta_norms" and sb and not sig.get("init_method_copied_blocks"):
        key = "random bottom, proc top downscaled to random write ratios (e-series)" if rb else "full proc, top blocks downscaled (pds)"
    elif proc and im == "upscale_random_match_attn_delta_norms" and rb and sb:
        key = "random bottom, top attention upscaled only (rattn)"
    if key:
        series[key].append(dict(arm=t["arm"], n=t["n"], acc=t["acc"], sd=t["sd"], train_loss=t["train_loss"], random=rb, scaled=sb, k=(len(rb) if "top" in key.split(",")[0] or key.startswith("proc prefix") else 12 - len(rb))))
for k in series: series[k].sort(key=lambda d: (len(d["random"]), d["arm"]))

json.dump(dict(r_mean=r_mean, table=table, fitgen=fitgen, traj=traj, init=init, dyn=dyn, series=series), open(f"{ROOT}/plots/cache/verify/synthesis.json", "w"), indent=1)

# ---------------------------------------------------------------- markdown
L = []
L.append("# Synthesis data (generated by plots/verify/gather_synthesis.py)\n")
L.append(f"Random baseline r = {r_mean:.2f} (n = {len(r_runs)}). Last-epoch top-1, seed means; train loss = epoch-299 mixup train loss; test loss = epoch-299 CE.\n")
L.append("## T1. Every finished arm (construction from the Namespace flags)\n")
L.append("| arm | era | n | acc | sd | vs r | train loss | test loss | construction |\n|---|---|---|---|---|---|---|---|---|")
for t in table:
    L.append(f"| `{t['arm']}` | {t['era']} | {t['n']} | {t['acc']:.2f} | {t['sd']:.2f} | {t['d']:+.2f} | {t['train_loss']:.3f} | {t['test_loss']:.3f} | {t['desc']} |")
L.append(f"\n## T2. Fit vs generalisation over the {fitgen['n']} clean arms\n")
L.append(f"Pearson(acc, train loss) = {fitgen['pearson_acc_trainloss']:+.2f}, Spearman = {fitgen['spearman_acc_trainloss']:+.2f}; Pearson(test loss, train loss) = {fitgen['pearson_testloss_trainloss']:+.2f}; "
         f"test loss = {fitgen['testloss_line']['slope']:.2f} x train loss + {fitgen['testloss_line']['intercept']:.2f}, residual sd {fitgen['testloss_line']['resid_sd']:.3f}.\n")
L.append("Arms more than 2 residual sd above the line (test loss worse than their fit predicts):\n")
for t in sorted(clean, key=lambda t: -t["resid_testloss"]):
    if t["resid_testloss"] > 2 * fitgen["testloss_line"]["resid_sd"]:
        L.append(f"- `{t['arm']}` {t['acc']:.2f}, train loss {t['train_loss']:.3f}, test loss {t['test_loss']:.3f} (+{t['resid_testloss']:.3f})")
L.append("\n## T3. Trajectories of representative arms (test acc / train loss at epoch)\n")
L.append("| arm | n | " + " | ".join(str(e) for e in EPS) + " |\n|---|---|" + "---|" * len(EPS))
for name, t in traj.items():
    L.append(f"| `{name}` acc | {t['n']} | " + " | ".join(f"{t['at'][e]['acc']:.2f}" if t['at'][e]['acc'] is not None else "-" for e in EPS) + " |")
    L.append(f"| `{name}` train loss | | " + " | ".join(f"{t['at'][e]['train_loss']:.3f}" if t['at'][e]['train_loss'] is not None else "-" for e in EPS) + " |")
L.append("\n## T4. Forward-pass write ratios at init (64 val images), per arm with a dump\n")
L.append("| arm | attn b0 | attn 1-8 | attn 9-11 | MLP b0 | MLP 1-8 | MLP 9-11 | entropy 1-8 | stream norm at block 9 |\n|---|---|---|---|---|---|---|---|---|")
for k, v in sorted(init.items()):
    L.append(f"| `{k}` | {v['attn_b0']:.2f} | {v['attn_mid']:.3f} | {v['attn_top']:.3f} | {v['mlp_b0']:.2f} | {v['mlp_mid']:.3f} | {v['mlp_top']:.3f} | {v['entropy_mid']:.2f} | {v['stream_at_9']:.0f} |")
L.append("\n## T5. Training dynamics from wandb (seed means, epochs <= 289)\n")
L.append("| arm | last ep | block-7 probe peak (epoch) | block-7 probe at end | blocks 6-9 probe at end | block 10 / 11 probe at end | MLP write end: b0 / 1-8 / 9-11 | attn write end: b0 / 1-8 / 9-11 |\n|---|---|---|---|---|---|---|---|")
for k, v in sorted(dyn.items()):
    mw = v.get("mlp_write_last", {}); aw = v.get("attn_write_last", {})
    L.append(f"| `{k}` | {v['last_epoch']} | {v['probe_b7_peak']:.1f} ({v['probe_b7_peak_epoch']}) | {v['probe_b7_last']:.1f} | {v['probe_b6to9_last']:.1f} | {v['probe_b10_last']:.1f} / {v['probe_b11_last']:.1f} | "
             + (f"{mw['b0']:.2f} / {mw['mid']:.2f} / {mw['top']:.2f}" if mw else "-") + " | " + (f"{aw['b0']:.2f} / {aw['mid']:.2f} / {aw['top']:.2f}" if aw else "-") + " |")
L.append("\n## T6. Split series (clean + old runs; k = number of blocks in the named segment)\n")
for k, rows in series.items():
    L.append(f"**{k}**\n\n| arm | random blocks | scaled blocks | n | acc | sd | vs r | train loss |\n|---|---|---|---|---|---|---|---|")
    for d in rows:
        L.append(f"| `{d['arm']}` | {','.join(map(str, d['random'])) or '-'} | {','.join(map(str, d['scaled'])) or '-'} | {d['n']} | {d['acc']:.2f} | {d['sd']:.2f} | {d['acc']-r_mean:+.2f} | {d['train_loss']:.3f} |")
    L.append("")
open(f"{ROOT}/docs/synthesis_data.md", "w").write("\n".join(L) + "\n")
print("wrote docs/synthesis_data.md and plots/cache/verify/synthesis.json")
