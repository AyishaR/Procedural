"""Ground-truth table of every ViT-B run: parsed from slurm .out Namespace dumps + log.txt.
No arm names assumed; the job name is only a label. Init flags come from the Namespace."""
import re, json, glob, os, sys, datetime as dt
from collections import defaultdict
ROOT = "/home/schrodi/Procedural"
OUT = os.path.join("/home/schrodi/Procedural/plots/cache/verify", "arm_truth.json")
FIX_TIME = dt.datetime.fromisoformat("2026-08-31 20:56:24+02:00")  # commit 1ad0e0b
KEYS = ["initialize","random_blocks","weight_shuffle","init_method","init_method_scaled_blocks",
        "init_method_scaled_attributes","quantile_source","quantile_1d_mode","quantile_qkv_mode",
        "quantile_1d_source","custom_init_type","custom_init_blocks","slice_scale_qk","slice_scale_v",
        "slice_scale_proj","spectral_values","spectral_basis","target_ratio_absolute","target_ratio_scale",
        "target_ratio_flatten","clip_outlier_blocks","outlier_clip_frac","weight_init","layer_11_scale_method",
        "layer_11_scale_ln","layer_11_scale_attn_qk","layer_11_scale_attn_v","layer_11_scale_attn_proj",
        "layer_11_target_qkvp_ln1_norm_ratio","skip_norm","shuffle_load","skip_load_blocks","freeze_blocks",
        "attention_residual_scaling","attention_out_scaling","learning_rate_scaling","init_method_copied_blocks",
        "simultaneous_init_scaling","init_method_bias_scaling","notes","epochs","lr","warmup_epochs","model","drop_path","skip_load_block_attributes","custom_pr_load","hold_back_blocks","delete_blocks","skip_attn_segments","initialize_as_pr","head_init_scale"]
def parse_ns(line):
    body = line[line.index("Namespace(")+len("Namespace("):].rstrip()
    if body.endswith(")"): body = body[:-1]
    try:
        return eval("dict(" + body + ")", {"__builtins__": {}, "dict": dict})
    except Exception as e:
        return None
runs = {}
for f in sorted(glob.glob(f"{ROOT}/logs/ft_*_*.out")):
    m = re.match(r".*/ft_(\d+)_(.+)\.out$", f)
    jobid, jobname = m.group(1), m.group(2)
    started, sid, ns = None, None, None
    with open(f, errors="replace") as fh:
        for line in fh:
            if line.startswith("Started at") and started is None:
                try: started = dt.datetime.strptime(line.strip()[len("Started at "):], "%a %b %d %I:%M:%S %p %Z %Y")
                except Exception:
                    try: started = dt.datetime.strptime(line.strip()[len("Started at "):], "%a %d %b %Y %I:%M:%S %p %Z")
                    except Exception: started = line.strip()
            elif line.startswith("Running with ID"):
                sid = line.split()[-1]
            elif line.startswith("Namespace(") and ns is None:
                ns = parse_ns(line)
            if ns is not None and sid is not None and started is not None:
                break
    if ns is None or sid is None:
        continue
    key = (sid, ns.get("seed"))
    rec = runs.setdefault(key, {"slurm_id": sid, "seed": ns.get("seed"), "jobnames": set(), "jobids": [],
                                "starts": [], "sig": {k: ns.get(k) for k in KEYS}, "output_dir": ns.get("output_dir")})
    rec["jobnames"].add(jobname); rec["jobids"].append(jobid)
    rec["starts"].append(str(started))
    # earliest .out defines the init-time code version
    if isinstance(started, dt.datetime):
        if rec.get("first_start") is None or started < rec["first_start"]:
            rec["first_start"] = started; rec["sig_first"] = {k: ns.get(k) for k in KEYS}
for key, rec in runs.items():
    od = rec["output_dir"]; lf = os.path.join(ROOT, od, "log.txt") if od else None
    rec["max_epoch"] = None; rec["acc_last"] = None; rec["acc_at"] = {}
    if lf and os.path.exists(lf):
        rows = {}
        for line in open(lf):
            try: r = json.loads(line)
            except Exception: continue
            if "epoch" in r: rows[r["epoch"]] = r
        if rows:
            me = max(rows); rec["max_epoch"] = me
            rec["acc_last"] = rows[me].get("test_acc1")
            for e in (249, 274, 284, 289, 299):
                if e in rows and "test_acc1" in rows[e]: rec["acc_at"][e] = rows[e]["test_acc1"]
            rec["train_loss_last"] = rows[me].get("train_loss")
    fs = rec.get("first_start")
    rec["first_start"] = str(fs) if fs else None
    rec["pre_fix"] = (fs.replace(tzinfo=dt.timezone(dt.timedelta(hours=2))) < FIX_TIME) if isinstance(fs, dt.datetime) else None
    rec["jobnames"] = sorted(rec["jobnames"])
json.dump(list(runs.values()), open(OUT, "w"), indent=1, default=str)
print(f"{len(runs)} (slurm_id, seed) runs -> {OUT}")
# compact table
by_name = defaultdict(list)
for rec in runs.values():
    for jn in rec["jobnames"]: by_name[jn].append(rec)
for jn in sorted(by_name):
    recs = sorted(by_name[jn], key=lambda r: (r["slurm_id"], r["seed"]))
    for r in recs:
        acc = f"{r['acc_last']:.2f}" if r["acc_last"] is not None else "  -  "
        print(f"{jn:14s} {r['slurm_id']} s{r['seed']} ep={str(r['max_epoch']):>4s} acc={acc} prefix={r['pre_fix']} start={r['first_start']} notes={r['sig']['notes']!r}")
