"""Historical (training-time) per-layer records -> the rawlens summary schema, and their validation.

`results/imnet_base/accuracy_IMNET_BASE_<sid>[_s<seed>].json` holds, for every analysed epoch, the raw CLS
logit lens through the checkpoint's own norm+head on the full official val set (engine.model_analyse, bf16),
per block after attention (`layer_sub = L - 0.5`, our `aL`) and after the block (`layer_sub = L`, our `bL`),
plus activation norms, residual write ratios and (older runs) the epoch-averaged training gradient norm per
block.  Conventions: `epoch` null = init (E = 0), file index e = E - 1; the first row per (epoch, layer,
layer_sub) wins (a second epoch-299 pass with shuffled block order sits at chance); the embedding slot has no
historical lens.  `log.txt` gives train loss per epoch and test acc/loss every 5th epoch (last line wins per
epoch); `grad_norms_IMNET_BASE_*.json` gives epoch-averaged per-tensor gradient norms.

  python -m dynamics.hist [--runs all] [--validate]
"""
import argparse
import glob
import json
import os

import numpy as np

from dynamics import registry as dreg
from dynamics.rawlens import HALF
from probes import registry as preg

BLOCK_TENSORS = ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2")   # rsplit-level keys, no q/k/v slices


def find_json(e):
    base = preg.RESULTS / "imnet_base"
    for cand in (base / f"accuracy_IMNET_BASE_{e['slurm_id']}_s{e['seed']}.json", base / f"accuracy_IMNET_BASE_{e['slurm_id']}.json"):
        if cand.exists():
            return cand
    return None


def find_ft(path, seed):
    d = json.load(open(path))
    fts = [ft for rec in d for ft in rec.get("ft", [])]
    if len(fts) == 1:
        return fts[0]
    m = [ft for ft in fts if ft.get("seed") == seed]
    return m[0] if m else None


def parse_stats(stats):
    """-> dict with files (sorted, -1 = init), top1 [n_files, 25] (NaN where absent), and per-block series."""
    rows = {}
    for r in stats:
        if "acc" not in r or r.get("head") is not None or r.get("layer") is None or r.get("layer_sub") is None:
            continue
        ep = -1 if r.get("epoch") is None else int(r["epoch"])
        key = (ep, int(r["layer"]), float(r["layer_sub"]))
        rows.setdefault(key, r)                                            # first row wins
    files = sorted({k[0] for k in rows})
    n = len(files)
    top1 = np.full((n, len(HALF)), np.nan)
    blk = {k: np.full((n, 12), np.nan) for k in ("blk_act_norm", "blk_act_rms", "mlp_delta_norm_ratio", "attn_delta_norm_ratio",
                                                  "grad_norm", "attn_delta_norm", "mlp_delta_norm")}
    for (ep, L, sub), r in rows.items():
        i = files.index(ep)
        if sub == L:                                                        # after the block -> bL
            top1[i, HALF.index(f"b{L}")] = r["acc"]
            for k in ("blk_act_norm", "blk_act_rms", "attn_delta_norm", "mlp_delta_norm"):
                if r.get(k) is not None:
                    blk[k][i, L] = r[k]
            if r.get("delta_norm_ratio") is not None:
                blk["mlp_delta_norm_ratio"][i, L] = r["delta_norm_ratio"]
            g = r.get("grad_norm")
            if g is not None and g != -1:
                blk["grad_norm"][i, L] = g
        elif abs(sub - (L - 0.5)) < 1e-6:                                   # after attention -> aL
            top1[i, HALF.index(f"a{L}")] = r["acc"]
            if r.get("delta_norm_ratio") is not None:
                blk["attn_delta_norm_ratio"][i, L] = r["delta_norm_ratio"]
    return dict(files=files, top1=top1, **blk)


def parse_log(e):
    recs = {}
    for r in preg._log_lines(e):
        if r.get("epoch") is not None:
            recs[int(r["epoch"])] = r                                       # last line wins per epoch
    eps = sorted(recs)
    return dict(epoch=eps, **{k: [recs[i].get(k) for i in eps] for k in ("train_loss", "test_acc1", "test_loss", "train_lr", "train_grad_norm")})


def parse_grad_norms(e):
    base = preg.RESULTS / "imnet_base"
    for cand in (base / f"grad_norms_IMNET_BASE_{e['slurm_id']}_s{e['seed']}.json", base / f"grad_norms_IMNET_BASE_{e['slurm_id']}.json"):
        if not cand.exists():
            continue
        try:
            d = json.load(open(cand))
        except json.JSONDecodeError:
            continue
        recs = {}
        for r in d:
            if r.get("epoch") is not None and (r.get("seed") is None or r.get("seed") == e["seed"]):
                recs[int(r["epoch"])] = r
        if not recs:
            continue
        eps = sorted(recs)
        per_block = []
        for ep in eps:
            r = recs[ep]
            per_block.append([float(np.sqrt(sum(r.get(f"module.blocks.{l}.{t}", 0.0) ** 2 for t in BLOCK_TENSORS))) for l in range(12)])
        return dict(source=str(cand), epoch=eps, per_block=per_block,
                    note="sqrt of the sum of squared epoch-averaged per-tensor norms (norm1, qkv, proj, norm2, fc1, fc2); a proxy, not the norm of the mean gradient")
    return None


def build(name):
    e = preg.BY_NAME[name]
    out = dict(run=name, arm=e["arm"], seed=e["seed"], role=dreg.ROLE.get(name, "history"), slurm_id=e["slurm_id"],
               halfdepth=HALF, log=parse_log(e), grad_norms_json=parse_grad_norms(e), **dreg.stamp())
    p = find_json(e)
    ft = find_ft(p, e["seed"]) if p is not None else None
    if ft is None:
        out.update(source=None, files=[], E=[], top1=[])
        return out
    st = parse_stats(ft["stats"])
    out.update(source=str(p), files=st["files"], E=[0 if f < 0 else dreg.epoch_of(f) for f in st["files"]],
               top1=np.where(np.isnan(st["top1"]), None, st["top1"]).tolist())
    for k in ("blk_act_norm", "blk_act_rms", "mlp_delta_norm_ratio", "attn_delta_norm_ratio", "grad_norm", "attn_delta_norm", "mlp_delta_norm"):
        arr = st[k]
        out[k] = None if np.isnan(arr).all() else np.where(np.isnan(arr), None, arr).tolist()
    return out


def validate(names):
    """Historical lens vs the recomputed rawlens (test split) at shared epochs, and vs log.txt test_acc1."""
    report = {}
    for name in names:
        hp = dreg.RES / "hist" / f"{name}.json"
        if not hp.exists():
            continue
        h = json.load(open(hp))
        rows = []
        for f, top1 in zip(h["files"], h["top1"]):
            rp = dreg.RES / "rawlens" / f"{name}@e{f}__test.json"
            if f < 0 or not rp.exists():
                continue
            r = json.load(open(rp))
            hv = np.array([np.nan if v is None else v for v in top1], float)
            rv = np.array(r["top1"], float)
            ok = ~np.isnan(hv)
            d = np.abs(hv[ok] - rv[ok])
            rows.append(dict(file=f, n_slots=int(ok.sum()), max_abs=float(d.max()), mean_abs=float(d.mean()),
                             hist_b11=float(hv[-1]), raw_b11=float(rv[-1]), log_test_acc1=r.get("test_acc1_log")))
        logchk = []
        log = h["log"]
        for f, top1 in zip(h["files"], h["top1"]):
            if f >= 0 and f in log["epoch"] and top1[-1] is not None:
                la = log["test_acc1"][log["epoch"].index(f)]
                if la is not None:
                    logchk.append(dict(file=f, hist_b11=top1[-1], log_test_acc1=la, diff=abs(top1[-1] - la)))
        report[name] = dict(vs_rawlens=rows, vs_log=logchk)
        if rows:
            print(f"{name:14s} vs rawlens: " + "  ".join(f"e{r['file']}: max {r['max_abs']:.2f} mean {r['mean_abs']:.2f}" for r in rows), flush=True)
        if logchk:
            print(f"{name:14s} vs log:     max |b11 - test_acc1| = {max(r['diff'] for r in logchk):.2f} over {len(logchk)} epochs", flush=True)
    dreg.write_json(dreg.RES / "hist" / "validation.json", report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all", help="all = the 12 canonical history runs + every dynamics run")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    names = dreg.HISTORY_RUNS + [r["name"] for r in dreg.RUNS] if args.runs == "all" else args.runs.split(",")
    for name in names:
        outp = dreg.RES / "hist" / f"{name}.json"
        if outp.exists() and not args.force:
            continue
        rec = build(name)
        dreg.write_json(outp, rec)
        ne = len(rec["files"])
        last = rec["top1"][-1] if ne else None
        print(f"{name:14s} source={'none' if rec['source'] is None else os.path.basename(rec['source'])} epochs={ne}"
              + (f" b6/b8/b10/b11@e{rec['files'][-1]}=" + "/".join('nan' if last[HALF.index(k)] is None else f"{last[HALF.index(k)]:.1f}" for k in ('b6', 'b8', 'b10', 'b11')) if ne else "")
              + f" log_epochs={len(rec['log']['epoch'])} gradjson={'yes' if rec['grad_norms_json'] else 'no'}", flush=True)
    if args.validate:
        validate(names)


if __name__ == "__main__":
    main()
