"""Experiment 1b: tuned lens and independently fitted linear probes at selected epochs (the probes pipeline
run on epoch entries).

Runs `probes.extract_features` (bf16, the section-0e cache layout [N,13,2,768] = CLS and mean-patch at 13
depths, on the frozen 100k / 20k / 50k ImageNet splits), `probes.lens` (raw + tuned lens on CLS, KL to the
checkpoint's own final distribution, step selected on val) and `probes.train_probes` (13 depths x 6 weight
decays, (wd, step) selected per depth on val; cls, mean and concat features) for `<run>@e<file>` entries.
Must run with PROBE_ROOT = DYN_ROOT so the caches and results land under the dynamics root.

  PROBE_ROOT=$DYN_ROOT python -m dynamics.lens_probe --runs all --files lens+matched [--stage all] [--limit 512]
"""
import argparse
import json
import os
from datetime import datetime

import torch

from dynamics import registry as dreg
from dynamics.rawlens import parse_files
from probes import extract_features as pex
from probes import features as feat
from probes import lens as plens
from probes import registry as preg
from probes import train_probes as ptp


def files_for(spec, e):
    out = set()
    for part in spec.split("+"):
        out |= set(parse_files(part, e))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--files", default="lens+matched")
    ap.add_argument("--stage", default="all", choices=["all", "extract", "lens", "probes"])
    ap.add_argument("--ftypes", default="cls,mean,concat")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lens_steps", type=int, default=3000)
    ap.add_argument("--probe_steps", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    assert preg.PROBE_ROOT.resolve() == dreg.DYN_ROOT.resolve(), f"run with PROBE_ROOT={dreg.DYN_ROOT} (got {preg.PROBE_ROOT})"
    preg.ensure_dirs()
    dreg.ensure_dirs()
    dreg.check_subsets()
    torch.backends.cudnn.benchmark = True
    for e in dreg.resolve_runs(args.runs):
        ents = [dreg.epoch_entry(e, i) for i in files_for(args.files, e)]
        print(f"{e['name']}: {[ee['name'] for ee in ents]}", flush=True)
        if args.stage in ("all", "extract"):
            for split in ("train", "val", "test"):
                pex.extract(ents, "imnet", split, args.batch_size, args.workers, args.limit)
        if args.stage in ("all", "lens"):
            for ee in ents:
                outp = preg.RES / "lens" / f"{ee['name']}{feat.result_suffix(ee['name'], 'imnet')}.json"
                if outp.exists() and not args.force:
                    print(f"skip {outp.name} (exists)", flush=True)
                    continue
                if not all(feat.is_done(ee["name"], "imnet", s) for s in ("train", "val", "test")):
                    print(f"skip {ee['name']}: imnet features not cached", flush=True)
                    continue
                res = plens.run(ee, args.lens_steps, 4096)
                res.update(run=ee["base"], file=ee["epoch"], E=dreg.epoch_of(ee["epoch"]), role=dreg.ROLE[ee["base"]])
                print(f"{ee['name']:22s} logit lens top-1: " + " ".join(f"{a:5.1f}" for a in res["logit_lens"]["top1"]), flush=True)
                print(f"{ee['name']:22s} tuned lens top-1: " + " ".join(f"{a:5.1f}" for a in res["tuned_lens"]["top1"]) + f"  (best step {res['tuned_lens']['best_step']})", flush=True)
                dreg.write_json(outp, res)
        if args.stage in ("all", "probes"):
            for ee in ents:
                outp = preg.RES / "probes" / f"{ee['name']}__imnet{feat.result_suffix(ee['name'], 'imnet')}.json"
                if outp.exists() and not args.force:
                    print(f"skip {outp.name} (exists)", flush=True)
                    continue
                if not all(feat.is_done(ee["name"], "imnet", s) for s in ("train", "val", "test")):
                    print(f"skip {ee['name']}: imnet features not cached", flush=True)
                    continue
                rec = dict(model=ee["name"], run=ee["base"], file=ee["epoch"], E=dreg.epoch_of(ee["epoch"]), role=dreg.ROLE[ee["base"]],
                           arm=ee["arm"], seed=ee["seed"], ckpt=str(preg.ckpt_path(ee)), dataset="imnet", depths=preg.DEPTHS,
                           wd_grid=ptp.WD_GRID, eval_steps=[(i + 1) * ptp.EVAL_EVERY for i in range(args.probe_steps // ptp.EVAL_EVERY)],
                           config=dict(steps=args.probe_steps, batch=4096, lr=1e-3, warmup=100, probe_seed=0),
                           git=preg.git_commit(), date=datetime.now().isoformat(timespec="seconds"), ftypes={})
                for ft in args.ftypes.split(","):
                    res, sizes = ptp.train_one(ee["name"], "imnet", ft, args.probe_steps, 4096)
                    rec["ftypes"][ft] = res
                    rec.update(sizes)
                    print(f"{ee['name']:22s} {ft:6s} probe test top-1 by depth: " + " ".join(f"{a:5.1f}" for a in res["test_acc"]) + f"  ({res['seconds']:.0f}s)", flush=True)
                dreg.write_json(outp, rec)


if __name__ == "__main__":
    main()
