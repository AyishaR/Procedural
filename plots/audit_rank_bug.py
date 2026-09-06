#!/usr/bin/env python3
"""Which runs are affected by the post-DDP rank-divergence bug?

A run is affected iff it is DISTRIBUTED (world_size > 1) and performs a weight edit AFTER
utils.pr_load_model constructs DDP using the per-rank torch RNG (seed = args.seed + rank):

  SEVERE   --weight_shuffle / --target_model_weight_shuffle
           -> utils.shuffle_weights uses torch.randperm on EVERY listed tensor, so all of
              blocks' weights differ across ranks.
  PARTIAL  --quantile_1d_mode != "skip"  (torch.randperm, main.py:983)
           --quantile_1d_source parametric (torch.randn, main.py:976)
           -> only the 1-D parameters differ across ranks.
  CLEAN    everything else: the 2-D quantile rank map (argsort of the DDP-synced tensor),
           rho-matching (scale factors broadcast from rank 0), norm matching, outlier
           clipping, slice_scale -- all deterministic given synced weights.

Usage:  python plots/audit_rank_bug.py            (writes plots/cache/rank_bug_audit.json)
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "plots" / "cache" / "rank_bug_audit.json"
RES = ROOT / "results" / "imnet_base"
KEYS = ["slurm_id", "seed", "world_size", "notes", "weight_shuffle", "target_model_weight_shuffle",
        "quantile_1d_mode", "quantile_1d_source", "init_method", "init_method_scaled_blocks",
        "random_blocks", "epochs"]


def has_fix_marker(sid):
    """main.py prints "[init-sync] broadcast N tensors" once the re-sync runs.

    This is the ONLY marker that proves the fix actually executed, rather than that the
    script or the config intended it -- so it outranks the flags below.
    """
    for f in (ROOT / "logs").glob(f"ft_*_*.out"):
        if f"_{sid}_" in f.name or f.name.startswith(f"ft_{sid}_"):
            try:
                if "[init-sync] broadcast" in f.read_text(errors="ignore"):
                    return True
            except OSError:
                pass
    return False


def classify(c, sid=None):
    if sid is not None and has_fix_marker(sid):
        return "clean", "FIXED: [init-sync] marker present in the run log"
    if (c.get("notes") or "") == "rank-sync-fix":
        return "clean", "FIXED: tagged rank-sync-fix"
    if (c.get("world_size") or 1) <= 1:
        return "clean", "single process"
    if c.get("weight_shuffle") or c.get("target_model_weight_shuffle"):
        return "SEVERE", "weight_shuffle -> randperm on every listed tensor"
    q = c.get("quantile_1d_mode") or ""
    if q and q != "skip":
        return "PARTIAL", f"quantile_1d_mode={q} -> randperm on 1-D params"
    if (c.get("quantile_1d_source") or "") == "parametric":
        return "PARTIAL", "parametric 1-D -> torch.randn"
    return "clean", "no post-DDP RNG edit"


def final_acc(sid, seed):
    p = RES / f"results_IMNET_BASE_{sid}" / f"s{seed}" / "log.txt"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if len(rows) < 299:
        return None
    a = [r.get("test_acc1") for r in rows if r.get("test_acc1")]
    return a[-1] if a else None


def fetch():
    import wandb
    api = wandb.Api(timeout=120)
    out = {}
    for r in api.runs("procedural_pretraining/vit base kdyck", per_page=500):
        c = r.config
        sid, seed = c.get("slurm_id"), c.get("seed")
        if sid is None or seed is None:
            continue
        k = f"{sid}|{seed}"
        if k not in out:
            out[k] = {kk: c.get(kk) for kk in KEYS}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


if __name__ == "__main__":
    cfg = fetch() if ("--refresh" in sys.argv or not CACHE.exists()) else json.load(open(CACHE))
    rows = []
    for k, c in cfg.items():
        sid, seed = k.split("|")
        verdict, why = classify(c, sid)
        rows.append((verdict, sid, seed, c.get("world_size"), why, final_acc(sid, seed)))
    order = {"SEVERE": 0, "PARTIAL": 1, "clean": 2}
    rows.sort(key=lambda r: (order[r[0]], r[1], r[2]))
    n_fin = lambda v: sum(1 for r in rows if r[0] == v and r[5] is not None)
    print(f"{len(cfg)} (slurm_id, seed) pairs in wandb\n")
    for v in ["SEVERE", "PARTIAL", "clean"]:
        tot = sum(1 for r in rows if r[0] == v)
        print(f"  {v:8} {tot:4} runs   ({n_fin(v)} reached 300 epochs locally)")
    print(f"\n{'verdict':9}{'slurm_id':>10}{'seed':>5}{'gpus':>5}{'top-1':>9}   why")
    for v, sid, seed, ws, why, acc in rows:
        if v == "clean" or acc is None:
            continue
        print(f"{v:9}{sid:>10}{seed:>5}{str(ws):>5}{acc:9.2f}   {why}")
