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
    "ftb7h": [(29469076, 0)],     # random 0-4, proc 5-11 (79.67)
    "ftb11h": [(29484978, 0)],    # random block 0, proc 1-11 (79.85)
    "ftb8h": [(29484975, 0)],     # random 0-3, proc 4-11 (79.11)
    "ftb9h": [(29484976, 0)],     # random 0-2, proc 3-11 (78.82)
    "ftbqmlnvog": [(29538122, 0), (29538122, 1), (29538122, 2)],   # Gaussian twin
    "ftbqmlnvot": [(29543647, 0)],                                 # Student-t twin
    "ftbrhos": [(29538140, 0)],    # write budgets on random q/k/fc1 (75.07)
    "ftblrm": [(29545846, 0)],     # step-size matched random (77.73)
    "ftbana": [(29572321, 0)],     # analytic profile (76.61)
    "ftbanaf": [(29578396, 0)],    # analytic, flat top (76.41)
    "ftbanag": [(29581216, 0)],    # analytic + proc LN gains (running)
    "ftbanab": [(29581214, 0)],    # analytic + proc LN biases (running)
    "ftbanap": [(29592459, 0)],    # analytic + sampled LN gains and biases (running)
    "ftbcomp11": [(29472870, 0), (29472870, 1), (29472870, 2)],   # both levers combined (80.63)
    "ftb1i": [(29469074, 0), (29472868, 1), (29472869, 2)],         # proc 0-10, random block 11 (80.37)
}
FAMS = ["acc", "delta_norm_ratio", "attn_entropy", "grad_norm", "blk_act_rms", "attn_delta_norm_ratio"]
# `Epoch-wise/delta_norm_ratio_layer{l}` is logged twice per epoch and layer (engine.py:563 attention row,
# engine.py:599 MLP row, same key). The last row wins in `delta_norm_ratio` (= MLP write ratio); the
# family `attn_delta_norm_ratio` re-reads the same key and keeps the FIRST row per epoch (= attention).
WANDB_KEY = {"attn_delta_norm_ratio": "delta_norm_ratio"}
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
                    k = f"Epoch-wise/{WANDB_KEY.get(fam, fam)}_layer{l}"
                    try:
                        rows = run.history(keys=["Epoch-wise/epoch", k], samples=5000, pandas=False)
                    except Exception as e:
                        print("  ERR", run.id, k, e, flush=True); continue
                    seen = set()
                    for row in rows:
                        e = row.get("Epoch-wise/epoch"); v = row.get(k)
                        if e is None or v is None: continue
                        if fam in WANDB_KEY:            # keep the first row per epoch (attention sublayer)
                            if (int(e), l) in seen: continue
                            seen.add((int(e), l))
                        merged.setdefault(str(int(e)), {})[f"{fam}_layer{l}"] = float(v)
        cache.setdefault(arm, {})[str(seed)] = merged
        json.dump(cache, open(OUT, "w"))
        eps = sorted(int(e) for e in merged)
        print(f"{arm} s{seed}: {len(runs)} segments, {len(eps)} epochs [{eps[:2]}..{eps[-2:]}] in {time.time()-t0:.0f}s", flush=True)
print("done ->", OUT)
