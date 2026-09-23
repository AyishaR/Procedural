"""Pull the per-layer, per-epoch training traces (head-probe accuracy of each block's output,
delta-norm ratio rho, attention entropy) from wandb for a set of arms and cache them.
wandb logs one row per (epoch, layer), every ~10 epochs, from every rank; rank 0 is used, and
resumed segments (one wandb run each) are merged with later segments winning.
Output: plots/cache/verify/wandb_layerwise.json  {arm: {seed: {epoch: {metric_layerI: value}}}}"""
import json, os, sys, time, wandb
OUT = "/home/schrodi/Procedural/plots/cache/verify/wandb_layerwise.json"
PROJECT = "procedural_pretraining/vit base kdyck"
SHUF = "procedural_pretraining/vit base kdyck shuffle"
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
    # 2026-09-15: the decomposition arms and the ksd generality arms (third tuple element = wandb project when not the default)
    "ftbanau": [(29602226, 0)],    # ftbana weights + isotropic-statistics gains (77.35)
    "ftbanal": [(29609180, 0)],    # slow steps without anisotropy (79.78)
    "ftbanai": [(29602228, 0)],    # input side removed (78.11)
    "ftbanac": [(29620684, 0)],    # early + late lever
    "ftbrhop": [(29626634, 0)],    # late lever via proj/fc2 only (79.93)
    "ftbrhopl": [(29632673, 0)],   # loud not slow, bf16 (80.13)
    "ftbrhosl": [(29626866, 0)],   # slow not loud (78.31)
    "ftb4": [(29547831, 0, SHUF)],       # ksd all blocks (80.21)
    "ftb4i": [(29547835, 0, SHUF)],      # ksd blocks 0-7, random 8-11 (80.05)
    "ftb4h": [(29547834, 0, SHUF)],      # random 0-7, ksd 8-11 (77.88)
    "ftbanak": [(29626632, 0, SHUF)],    # ksd second-moment recipe (77.86)
    "ftbanakw": [(29626868, 0, SHUF)],   # write profile matched (76.65)
    "ftbanakb": [(29637998, 0, SHUF)],   # + MLP gate
    "ftbanakg": [(29634997, 0, SHUF)],   # exact scales + permuted LN vectors
    "ftbqmlnvok": [(29634995, 0, SHUF)], # twin recipe on ksd
    "ftbanapx": [(29723558, 0)],   # kdyck control: exact fold, q/k separate, ramp
    "ftbanape": [(29723598, 0)],   # kdyck control: exact fold, per block, no corrections
    "ftbanaks": [(29720235, 0, SHUF)],   # sink
    "ftbanaksw": [(29720203, 0, SHUF)],  # weak sink
    "ftbanaksg": [(29720205, 0, SHUF)],  # sink + persistent gate
    "ftbanakbs": [(29720207, 0, SHUF)],  # persistent gate
    "ftbanakd": [(29720209, 0, SHUF)],   # diffuse q/k + persistent gate
    # 2026-09-17: kdyck prefix 0-7 (the reference of the blocks-0-7 family) and the blocks-0-7 recipe family
    "ftb4i_kdyck": [(29448854, 0), (29448854, 1), (29448854, 2)],   # kdyck blocks 0-7, random 8-11 (79.89)
    "ftbanaperab7": [(29729545, 0)],            # scales (all six) + entropy + active-unit gate
    "ftbanaperab7w": [(29729541, 0)],           # + v, proj, fc2 write-matched
    "ftbanaperab7i": [(29733656, 0)],           # v, proj, fc2 at timm (fresh start on the group partition; the shared-partition job 29729560 never ran)
    "ftbanapermb7": [(29729553, 0)],            # mean-gate control
    "ftbanapermb7w": [(29729556, 0)],           # mean-gate control, write-matched
    "ftbanapermb7i": [(29737095, 0)],           # mean gate, v / proj / fc2 at timm (kdyck); resumed on the group partition as 29737711, same results id
    "ftbanakpermb7i": [(29736861, 0, SHUF)],         # mean gate, v / proj / fc2 at timm (ksd)
    "ftbanakpermb7": [(29737097, 0, SHUF)],          # mean gate, checkpoint effective scales on the write side (ksd)
    "ftbanakperab7": [(29729543, 0, SHUF)],     # ksd
    "ftbanakperab7w": [(29729547, 0, SHUF)],
    "ftbanakperab7i": [(29729558, 0, SHUF)],
    # 2026-09-20: mechanism study (docs/early_lever_mechanism_plan.md): plain random with per-epoch checkpoints, and wave 1
    "r0": [(29744580, 0)],
    "ftbc7sg": [(29745254, 0)], "ftbck7sgb": [(29746035, 0, SHUF)],      # structure only (sink + mean gate on timm); the ksd cell in bf16
    "ftbck7sg": [(29745248, 0, SHUF)],                                   # its fp16 original, dead at epoch 10 (non-finite loss)
    "ftbc7a1": [(29745240, 0)], "ftbck7a1": [(29745242, 0, SHUF)],       # committed init, compensated steps on q, k rows and fc1
    "ftbck7ps": [(29745244, 0, SHUF)], "ftbck7pg": [(29745246, 0, SHUF)], "ftbck7p": [(29745250, 0, SHUF)], "ftbc7p": [(29745252, 0)],
    "ftbc7ps": [(29745319, 0)], "ftbc7pg": [(29745320, 0)], "ftbc7a1g": [(29745321, 0)], "ftbck7a1g": [(29745322, 0, SHUF)],   # 40-epoch screens (8 L40S)
    "ftbc7a2": [(29745327, 0)], "ftbck7a3qk": [(29745324, 0, SHUF)], "ftbck7a3f": [(29745325, 0, SHUF)],
    # 2026-09-21, wave 2: forced readability (C1), its two converses on plain random, single-component cells
    "ftbc7c1": [(29751052, 0)], "ftbck7c1": [(29751067, 0, SHUF)], "r0sup": [(29751150, 0)], "r0frz": [(29751165, 0)], "r0frzl": [(29760130, 0)],
    "ftbc7s": [(29751055, 0)], "ftbc7g": [(29751058, 0)], "ftbck7s": [(29751061, 0, SHUF)], "ftbck7g": [(29751064, 0, SHUF)],
    "ftb4c1": [(29754117, 0)], "ftb4kc1": [(29754123, 0, SHUF)],      # C1 on the FULL procedural checkpoints (2026-09-22)
    "ftbc7l": [(29756597, 0)],                                          # compose arm: C (0-7) + late lever (9-11), v+proj+fc2
}
# arms whose cache is re-pulled even if present (running arms): REFRESH=arm1,arm2 ; ONLY=arm1,arm2 restricts the loop
REFRESH = set(x for x in os.environ.get("REFRESH", "").split(",") if x)
ONLY = set(x for x in os.environ.get("ONLY", "").split(",") if x)
FAMS = ["acc", "delta_norm_ratio", "attn_entropy", "attn_delta_norm_ratio"] + ([] if os.environ.get("FAST") else ["grad_norm", "blk_act_rms"])
FAMS += [f for f in os.environ.get("EXTRA_FAMS", "").split(",") if f]      # e.g. EXTRA_FAMS=attn_mad,cls_attn_entropy (other per-block keys of engine.model_analyse)
# `Epoch-wise/delta_norm_ratio_layer{l}` is logged twice per epoch and layer (engine.py:563 attention row,
# engine.py:599 MLP row, same key). The last row wins in `delta_norm_ratio` (= MLP write ratio); the
# family `attn_delta_norm_ratio` re-reads the same key and keeps the FIRST row per epoch (= attention).
WANDB_KEY = {"attn_delta_norm_ratio": "delta_norm_ratio"}
api = wandb.Api(timeout=120)
cache = json.load(open(OUT)) if os.path.exists(OUT) else {}
for arm, ids in ARMS.items():
    if ONLY and arm not in ONLY:
        continue
    for entry in ids:
        sid, seed = entry[0], entry[1]; project = entry[2] if len(entry) > 2 else PROJECT
        have = cache.get(arm, {}).get(str(seed), {})
        missing = list(FAMS) if arm in REFRESH else [f for f in FAMS if not any(k.startswith(f + "_layer") for e in have.values() for k in e)]
        if not missing:
            continue
        runs = sorted(api.runs(project, filters={"config.slurm_id": sid, "config.seed": seed, "display_name": "GPU 0"}),
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
