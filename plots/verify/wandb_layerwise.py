"""Pull the per-layer, per-epoch training traces (head-probe accuracy of each block's output,
delta-norm ratio rho, attention entropy) from wandb for a set of arms and cache them.
wandb logs one row per (epoch, layer), every ~10 epochs, from every rank; rank 0 is used, and
resumed segments (one wandb run each) are merged with later segments winning.
Output: plots/cache/verify/wandb_layerwise.json  {arm: {seed: {epoch: {metric_layerI: value}}}}"""
import json, os, sys, time, wandb
OUT = "/home/schrodi/Procedural/plots/cache/verify/wandb_layerwise.json"
PROJECT = "procedural_pretraining/vit base kdyck"
ARMS = {  # arm -> [(slurm_id, seed)]
    "r": [(29384839, 0), (29384839, 1), (29384839, 2)],
    "p": [(29377576, 0), (29377576, 1), (29377576, 2)],
    "ftb3i": [(29469072, 0), (29469072, 1), (29469072, 2)],
    "ftb3h": [(29469075, 0)],
    "ftbrho": [(29453944, 0), (29461252, 1), (29461253, 2)],
    "ftb3b": [(29388202, 0), (29406778, 1), (29406779, 2)],
    "ftb4o": [(29451652, 0)],
    "ftbqmlnvo": [(29523316, 0), (29523316, 1), (29523316, 2)],
}
FAMS = ["acc", "delta_norm_ratio", "attn_entropy", "grad_norm", "blk_act_rms"]
api = wandb.Api(timeout=120)
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
for arm, ids in ARMS.items():
    for sid, seed in ids:
        have = cache.get(arm, {}).get(str(seed), {})
        missing = [f for f in FAMS if not any(k.startswith(f + "_layer") for e in have.values() for k in e)]
        if not missing:
            continue
        runs = sorted(api.runs(PROJECT, filters={"config.slurm_id": sid, "config.seed": seed, "display_name": "GPU 0"}),
                      key=lambda r: r.created_at)
        merged = dict(have)
        t0 = time.time()
        for run in runs:
            for fam in missing:
                for l in range(12):
                    k = f"Epoch-wise/{fam}_layer{l}"
                    try:
                        rows = run.history(keys=["Epoch-wise/epoch", k], samples=5000, pandas=False)
                    except Exception as e:
                        print("  ERR", run.id, k, e, flush=True); continue
                    for row in rows:
                        e = row.get("Epoch-wise/epoch"); v = row.get(k)
                        if e is None or v is None: continue
                        merged.setdefault(str(int(e)), {})[f"{fam}_layer{l}"] = float(v)
        cache.setdefault(arm, {})[str(seed)] = merged
        json.dump(cache, open(OUT, "w"))
        eps = sorted(int(e) for e in merged)
        print(f"{arm} s{seed}: {len(runs)} segments, {len(eps)} epochs [{eps[:2]}..{eps[-2:]}] in {time.time()-t0:.0f}s", flush=True)
print("done ->", OUT)
