import json, collections, sys
R = json.load(open("/home/schrodi/Procedural/plots/cache/verify/arm_truth.json"))
DEF = {"initialize":"","random_blocks":"","weight_shuffle":"","init_method":"default","init_method_scaled_blocks":"",
 "init_method_scaled_attributes":"","quantile_source":"empirical","quantile_1d_mode":"skip","quantile_qkv_mode":"pooled",
 "quantile_1d_source":"empirical","custom_init_type":"","custom_init_blocks":"","slice_scale_qk":1.0,"slice_scale_v":1.0,
 "slice_scale_proj":1.0,"spectral_values":"keep","spectral_basis":"keep","target_ratio_absolute":-1.0,"target_ratio_scale":1.0,
 "target_ratio_flatten":False,"clip_outlier_blocks":"","outlier_clip_frac":0.001,"weight_init":"","layer_11_scale_method":"",
 "layer_11_scale_ln":1.0,"layer_11_scale_attn_qk":1.0,"layer_11_scale_attn_v":1.0,"layer_11_scale_attn_proj":1.0,
 "layer_11_target_qkvp_ln1_norm_ratio":-1.0,"skip_norm":True,"shuffle_load":False,"skip_load_blocks":"","freeze_blocks":"",
 "attention_residual_scaling":"","attention_out_scaling":"","learning_rate_scaling":False,"init_method_copied_blocks":"",
 "simultaneous_init_scaling":False,"init_method_bias_scaling":False,"epochs":300,"lr":0.002,"warmup_epochs":50,"model":"vit_base","drop_path":0}
def short(sig):
    out=[]
    for k,v in sig.items():
        if k=="notes": continue
        if v is None: continue   # flag did not exist in that code version
        if k in DEF and v==DEF[k]: continue
        if k=="weight_shuffle" and v and "[" in v:
            segs=[s for s in v.split(";") if s]
            blocks=[s.split("[")[0] for s in segs]; names=set(s.split("[",1)[1].rstrip("]") for s in segs)
            v=f"blocks {','.join(blocks)} :: " + (" | ".join(sorted(names)) if len(names)>1 else names.pop())
        out.append(f"{k}={v}")
    return "; ".join(out)
byname=collections.defaultdict(list)
for r in R:
    for jn in r["jobnames"]: byname[jn].append(r)
only = sys.argv[1:] 
for jn in sorted(byname):
    if only and jn not in only: continue
    sigs=collections.OrderedDict()
    for r in byname[jn]:
        s=short(r["sig"]); sigs.setdefault(s,[]).append(r)
    for s,recs in sigs.items():
        recs=sorted(recs,key=lambda r:(r['slurm_id'],r['seed']))
        accs=[f"{r['acc_last']:.2f}" if r['acc_last'] is not None and r['max_epoch']==299 else f"({r['max_epoch']})" for r in recs]
        fix=set("clean" if not r["pre_fix"] else "PRE" for r in recs)
        print(f"### {jn}  n={len(recs)} ids={sorted(set(r['slurm_id'] for r in recs))} {'/'.join(sorted(fix))} accs={accs}")
        print("    "+s)
