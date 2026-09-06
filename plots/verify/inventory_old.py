"""Inventory of the result dirs NOT covered by logs/ft_*.out: recover each run's args from its
full checkpoint (checkpoint-*.pth stores `args`), name it via the b_vitb_* chain launchers
(SLURM_ID -> run script), and record epochs/accuracy from log.txt."""
import json, os, glob, re, torch
S = "/home/schrodi/Procedural/plots/cache/verify"
# 1) chain launchers: SLURM_ID -> script name
launch = {}
for f in glob.glob("vitbase_runs/b_vitb_*.sh"):
    t = open(f).read()
    m = re.search(r"^SLURM_ID=(\d+)", t, re.M); s = re.search(r'^SCRIPT="?([\w./]+)"?', t, re.M)
    if m and s and m.group(1) != "0":
        launch.setdefault(m.group(1), set()).add(os.path.basename(s.group(1)).replace("run_train_", "").replace(".sh", ""))
KEYS = ["initialize","random_blocks","weight_shuffle","init_method","init_method_scaled_blocks","init_method_scaled_attributes",
        "quantile_source","quantile_1d_mode","quantile_qkv_mode","quantile_1d_source","custom_init_type","custom_init_blocks",
        "slice_scale_qk","slice_scale_v","slice_scale_proj","target_ratio_absolute","target_ratio_scale","target_ratio_flatten",
        "clip_outlier_blocks","outlier_clip_frac","weight_init","layer_11_scale_method","layer_11_scale_ln","layer_11_scale_attn_qk",
        "layer_11_scale_attn_v","layer_11_scale_attn_proj","layer_11_target_qkvp_ln1_norm_ratio","skip_norm","shuffle_load",
        "skip_load_blocks","skip_load_block_attributes","freeze_blocks","init_method_copied_blocks","simultaneous_init_scaling",
        "init_method_bias_scaling","notes","epochs","lr","warmup_epochs","model","seed","slurm_id","custom_pr_load","hold_back_blocks",
        "attention_residual_scaling","attention_out_scaling","learning_rate_scaling","head_init_scale","drop_path"]
def norm(v):
    if isinstance(v, (list, tuple)): return ",".join(str(x) for x in v)
    if isinstance(v, dict):
        if not v: return ""
        return ";".join(f"{k}[{','.join(map(str, vv)) if isinstance(vv, (list, tuple)) else vv}]" for k, vv in v.items())
    return v
R = json.load(open(f"{S}/arm_truth.json")); known = set(r["slurm_id"] for r in R)
out = []
for d in sorted(glob.glob("results/imnet_base/results_IMNET_BASE_*")):
    sid = d.split("_")[-1]
    if sid in known or not sid.isdigit(): continue
    for sd in (sorted(glob.glob(d + "/s*")) or [d]):
        cks = sorted(glob.glob(sd + "/checkpoint-*[0-9].pth"), key=lambda p: int(re.search(r"checkpoint-(\d+)\.pth", p).group(1)))
        args = None
        if cks:
            try:
                ck = torch.load(cks[-1], map_location="cpu", weights_only=False, mmap=True)
                a = ck.get("args"); args = vars(a) if a is not None and not isinstance(a, dict) else a
            except Exception as e:
                try:
                    ck = torch.load(cks[-1], map_location="cpu", weights_only=False); a = ck.get("args"); args = vars(a) if a is not None and not isinstance(a, dict) else a
                except Exception as e2: print("ERR", cks[-1], e2)
        rows = {}
        lf = os.path.join(sd, "log.txt")
        if os.path.exists(lf):
            for l in open(lf):
                try: r = json.loads(l)
                except Exception: continue
                if "epoch" in r: rows[r["epoch"]] = r
        me = max(rows) if rows else None
        rec = {"slurm_id": sid, "seed_dir": os.path.basename(sd), "launcher_names": sorted(launch.get(sid, [])),
               "max_epoch": me, "acc_last": rows[me].get("test_acc1") if me is not None else None,
               "train_loss_last": rows[me].get("train_loss") if me is not None else None,
               "sig": {k: norm(args.get(k)) for k in KEYS} if args else None, "ckpt": cks[-1] if cks else None}
        out.append(rec)
json.dump(out, open(f"{S}/old_runs.json", "w"), indent=1, default=str)
print(len(out), "runs inventoried;", sum(1 for o in out if o["sig"]), "with args;", sum(1 for o in out if o["max_epoch"] == 299), "completed")
DEF = {"initialize":"","random_blocks":"","weight_shuffle":"","init_method":"default","init_method_scaled_blocks":"","init_method_scaled_attributes":"",
 "quantile_source":"empirical","quantile_1d_mode":"skip","quantile_qkv_mode":"pooled","quantile_1d_source":"empirical","custom_init_type":"","custom_init_blocks":"",
 "slice_scale_qk":1.0,"slice_scale_v":1.0,"slice_scale_proj":1.0,"target_ratio_absolute":-1.0,"target_ratio_scale":1.0,"target_ratio_flatten":False,"clip_outlier_blocks":"",
 "outlier_clip_frac":0.001,"weight_init":"","layer_11_scale_method":"","layer_11_scale_ln":1.0,"layer_11_scale_attn_qk":1.0,"layer_11_scale_attn_v":1.0,"layer_11_scale_attn_proj":1.0,
 "layer_11_target_qkvp_ln1_norm_ratio":-1.0,"skip_norm":True,"shuffle_load":False,"skip_load_blocks":"","freeze_blocks":"","init_method_copied_blocks":"","simultaneous_init_scaling":False,
 "init_method_bias_scaling":False,"epochs":300,"lr":0.002,"warmup_epochs":50,"model":"vit_base","custom_pr_load":"","hold_back_blocks":"","attention_residual_scaling":"","attention_out_scaling":"",
 "learning_rate_scaling":False,"head_init_scale":1.0,"drop_path":0,"skip_load_block_attributes":"","notes":""}
def short(sig):
    return "; ".join(f"{k}={v}" for k, v in sig.items() if k not in ("seed","slurm_id") and v is not None and not (k in DEF and v == DEF[k]))
for o in sorted(out, key=lambda o: (o["slurm_id"], o["seed_dir"])):
    if o["max_epoch"] is None and not o["sig"]: continue
    acc = f"{o['acc_last']:.2f}" if o["acc_last"] else "  -  "
    print(f"{o['slurm_id']} {o['seed_dir']} ep={str(o['max_epoch']):>4s} acc={acc} name={'/'.join(o['launcher_names']) or '?':10s} :: {short(o['sig']) if o['sig'] else 'NO ARGS'}")
