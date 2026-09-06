"""Emit one dump_init command per arm, with init flags taken VERBATIM from the arm's own
slurm Namespace dump (arm_truth.json), not from memory. Adds a few reference / hypothetical arms."""
import json, os
S = "/home/schrodi/Procedural/plots/cache/verify"
R = json.load(open(f"{S}/arm_truth.json"))
PROC = "results/pr_vitb_n/pr_6066174_final.pth"
STR_KEYS = ["initialize","random_blocks","weight_shuffle","init_method","init_method_scaled_blocks",
            "init_method_scaled_attributes","quantile_source","quantile_1d_mode","quantile_qkv_mode",
            "quantile_1d_source","custom_init_type","custom_init_blocks","spectral_values","spectral_basis",
            "clip_outlier_blocks","weight_init","skip_load_blocks","skip_load_block_attributes",
            "init_method_copied_blocks","custom_pr_load"]
NUM_KEYS = ["slice_scale_qk","slice_scale_v","slice_scale_proj","target_ratio_absolute","target_ratio_scale",
            "outlier_clip_frac","head_init_scale"]
BOOL_STR = ["skip_norm","target_ratio_flatten","initialize_as_pr"]      # str2bool flags
BOOL_RAW = ["simultaneous_init_scaling","init_method_bias_scaling"]     # type=bool: only emit when True
COMMON = ('--model vit_base --data_set IMNET --data_path /data/datasets/ILSVRC2012 --input_size 224 '
          '--batch_size 128 --total_batch_size 128 --update_freq 1 --epochs 300 --warmup_epochs 50 --lr 2e-3 '
          '--use_amp true --num_workers 6 --procedural_data kdyck --procedural_order standard --pr_notes "" '
          '--enable_wandb true --auto_resume false --save_ckpt false --slurm_id 0 --dist_eval true')
def flags_from_sig(sig):
    out = []
    for k in STR_KEYS:
        v = sig.get(k)
        if v is None: continue
        out.append(f'--{k} "{v}"')
    for k in NUM_KEYS:
        v = sig.get(k)
        if v is None: continue
        out.append(f'--{k} {v}')
    for k in BOOL_STR:
        v = sig.get(k)
        if v is None: continue
        out.append(f'--{k} {"true" if v else "false"}')
    for k in BOOL_RAW:
        if sig.get(k): out.append(f'--{k} True')
    return " ".join(out)
WANT = ["ftb3i","ftb4e3fix","ftbqks","ftbqmlnvo","ftbqm1dvo","ftbqm1dv","ftbqm1dqk","ftbqmvo","ftbqmln","ftbqm",
        "ftbqm1d","ftbqm1dpar","ftbqmbias",
        "ftbnorm","ftbcfg","ftbqu","ftbvu","ftbvd","ftbslice","ftbclip01","ftbclip5","ftb4o",
        "ftb11i","ftb11isfix","ftb11d","ftb11s",
        "ftb4i","ftb4h","ftb4m","ftb4l","ftb4k","ftb4n","ftb4g","ftb0a","ftb0m","ftb9e","ftb9b","ftbrho","ftb0h","ftb0g"]
cmds = []
for arm in WANT:
    recs = [r for r in R if arm in r["jobnames"] and r["seed"] == 0]
    if not recs:
        print("MISSING", arm); continue
    recs.sort(key=lambda r: (r["pre_fix"] is True, r["slurm_id"]))   # prefer clean (post-fix) record
    rec = recs[0]
    cmds.append((arm, 0, flags_from_sig(rec["sig"]), f"{rec['slurm_id']}"))
# reference / hypothetical arms (flags written by hand; nothing in logs for these)
cmds.append(("r", 0, '--initialize "" --skip_norm true', "manual"))
cmds.append(("r", 1, '--initialize "" --skip_norm true', "manual"))
cmds.append(("r", 2, '--initialize "" --skip_norm true', "manual"))
cmds.append(("p", 0, f'--initialize "{PROC}" --skip_norm true', "manual"))
cmds.append(("ftbsb", 0, f'--initialize "{PROC}" --skip_norm true --random_blocks "9,10,11" --custom_init_type spectral --custom_init_blocks "0,1,2,3,4,5,6,7,8" --spectral_values mp --spectral_basis keep', "script"))
cmds.append(("ftbsv", 0, f'--initialize "{PROC}" --skip_norm true --random_blocks "9,10,11" --custom_init_type spectral --custom_init_blocks "0,1,2,3,4,5,6,7,8" --spectral_values keep --spectral_basis random', "script"))
# hypothetical: Student-t (kurtosis+norm) donor instead of proc's values. H2 = pooled (code path works);
# Hlnvo_asis = ftbqmlnvo + parametric AS THE CODE CURRENTLY RUNS IT (qkv is skipped by the v_only branch)
cmds.append(("Hpar_pooled", 0, f'--initialize "{PROC}" --skip_norm true --init_method quantile_match_target_blocks --init_method_scaled_blocks "0,1,2,3,4,5,6,7,8" --quantile_source parametric', "hypothetical"))
cmds.append(("Hpar_lnvo_asis", 0, f'--initialize "{PROC}" --skip_norm true --init_method quantile_match_target_blocks --init_method_scaled_blocks "0,1,2,3,4,5,6,7,8" --quantile_source parametric --quantile_1d_mode layernorm --quantile_qkv_mode v_only', "hypothetical"))
with open(f"{S}/dump_cmds.txt", "w") as f:
    for arm, seed, fl, src in cmds:
        f.write(f"{arm}_s{seed}\t{src}\t{COMMON} --seed {seed} {fl}\n")
print(len(cmds), "commands")
for arm, seed, fl, src in cmds: print(f"{arm}_s{seed:<3} [{src}] {fl[:150]}")
