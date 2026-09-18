"""Experiment 1a: the checkpoint's own readout applied to intermediate CLS states, over training.

For every (run, epoch file) the model is run once on the two fixed sets (val = training-split images, test =
official val); at 25 half-depths (emb, then after-attention `aL` and after-block `bL` of every block L) the
checkpoint's own final norm + head is applied to the CLS token.  Per image and half-depth: top-1/5, CE (hard and
label-smoothed), correct-class margin, confidence, agreement with the final prediction.  Slot `b11` IS the
model's own output (checked against the model forward on the first batch).  fp32 stream by default with TF32
matmuls (`--precision`); many checkpoints stay resident so JPEG decoding is amortised.

  python -m dynamics.rawlens --runs all --files dense [--init] [--precision tf32] [--limit 512]
"""
import argparse
import os
import time

import numpy as np
import torch

from dynamics import common
from dynamics import data as ddata
from dynamics import registry as dreg
from probes import registry as preg

HALF = ["emb"] + [f"{t}{l}" for l in range(12) for t in ("a", "b")]      # 25 half-depths
FINAL = len(HALF) - 1                                                     # slot b11 = model output
SPLITS = ("val", "test")


def parse_files(spec, e):
    avail = set(dreg.available_files(e))
    if spec == "dense":
        want = dreg.DENSE_FILES
    elif spec == "core":
        want = dreg.CORE_FILES
    elif spec == "lens":
        want = dreg.LENS_FILES
    elif spec == "matched":
        want = dreg.matched_files(e["name"])
    elif spec == "transitions":
        want = dreg.transition_files(e["name"])
    elif spec == "avail":
        want = sorted(avail)
    else:
        want = [int(x) for x in spec.split(",") if x]
    return [i for i in want if i in avail]


def init_entry(e):
    p = dreg.init_dump(e)
    if p is None:
        return None
    return dict(e, name=f"{e['name']}@init", base=e["name"], epoch=-1, ckpt=str(p.relative_to(preg.RESULTS)),
                log=str(preg.log_path(e).relative_to(preg.RESULTS)))


def save_npz(path, **arrays):
    """Atomic NPZ write (temporary file, then rename); the JSON summary written afterwards is the completion marker."""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def out_paths(stage, ee, split, limit, tag=""):
    suf = (f"__limit{limit}" if limit else "") + (f"__{tag}" if tag else "")
    j = dreg.RES / stage / f"{ee['name']}__{split}{suf}.json"
    return j, j.with_suffix(".npz")


@torch.no_grad()
def half_states(m, img):
    hs, mids = common.states(m, img, mid=True)
    cls = [hs[0][:, 0]]
    for l in range(len(m.blocks)):
        cls += [mids[l][:, 0], hs[l + 1][:, 0]]
    return torch.stack(cls, 1), hs[-1]                                     # [B,25,D], final full state


def run_split(models, split, args):
    ds, meta = ddata.dataset(split, args.limit)
    N = len(ds)
    acc = {}
    for name in models:
        acc[name] = dict(pred=np.zeros((N, len(HALF)), np.int16), ce=np.zeros((N, len(HALF)), np.float16),
                         ce_smooth=np.zeros((N, len(HALF)), np.float16), margin=np.zeros((N, len(HALF)), np.float16),
                         maxp=np.zeros((N, len(HALF)), np.float16), top5=np.zeros((N, len(HALF)), bool),
                         sums=torch.zeros(7, len(HALF), dtype=torch.float64, device="cuda"), check=None)
    labels = np.zeros(N, np.int16)
    seen, t0 = 0, time.time()
    for img, y in ddata.loader(ds, args.batch, args.workers):
        img = img.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        B = img.shape[0]
        labels[seen:seen + B] = y.cpu().numpy()
        for name, m in models.items():
            with torch.no_grad(), common.precision(args.precision):
                z, h_final = half_states(m, img)
                logits = m.head(m.fc_norm(m.norm(z.reshape(B * len(HALF), -1)))).float().view(B, len(HALF), -1)
                if acc[name]["check"] is None:                              # slot b11 == the model's own forward
                    ref = m(img).float()
                pm = common.per_image(logits.reshape(B * len(HALF), -1), y.repeat_interleave(len(HALF)))
            pm = {k: v.view(B, len(HALF)) for k, v in pm.items()}
            a = acc[name]
            if a["check"] is None:
                a["check"] = dict(agree_top1=float((ref.argmax(-1) == pm["pred"][:, FINAL]).float().mean()),
                                  max_abs_logit_diff=float((ref - logits[:, FINAL]).abs().max()), n=int(B))
            agree_final = pm["pred"] == pm["pred"][:, FINAL:FINAL + 1]
            a["sums"] += torch.stack([pm["correct"].double().sum(0), pm["top5"].double().sum(0), pm["ce"].double().sum(0),
                                      pm["ce_smooth"].double().sum(0), pm["margin"].double().sum(0), pm["maxp"].double().sum(0),
                                      agree_final.double().sum(0)])
            sl = slice(seen, seen + B)
            a["pred"][sl] = pm["pred"].cpu().numpy(); a["ce"][sl] = pm["ce"].cpu().numpy()
            a["ce_smooth"][sl] = pm["ce_smooth"].cpu().numpy(); a["margin"][sl] = pm["margin"].cpu().numpy()
            a["maxp"][sl] = pm["maxp"].cpu().numpy(); a["top5"][sl] = pm["top5"].cpu().numpy()
        seen += B
        if seen % (args.batch * 20) == 0 or seen == N:
            print(f"  [{split}] {seen}/{N} imgs, {seen / (time.time() - t0):.0f} img/s x {len(models)} ckpts", flush=True)
    assert seen == N
    return acc, labels, meta


def summary(ee, split, a, labels, meta, args):
    s = (a["sums"] / len(labels)).cpu().numpy()
    med_margin = np.median(a["margin"].astype(np.float32), 0)
    out = dict(run=ee["base"], arm=ee["arm"], seed=ee["seed"], role=dreg.ROLE[ee["base"]], file=ee["epoch"],
               E=0 if ee["epoch"] < 0 else dreg.epoch_of(ee["epoch"]), split=split, n=int(len(labels)), limit=args.limit,
               precision=args.precision, tag=args.tag, halfdepth=HALF, ckpt=str(preg.ckpt_path(ee)),
               top1=(100 * s[0]).tolist(), top5=(100 * s[1]).tolist(), ce=s[2].tolist(), ce_smooth=s[3].tolist(),
               margin_mean=s[4].tolist(), margin_median=med_margin.tolist(), maxp=s[5].tolist(), agree_final=(100 * s[6]).tolist(),
               final_check=a["check"], k_bias_removed=a.get("k_bias_removed"),
               test_acc1_log=preg.final_test_acc(ee) if (split == "test" and ee["epoch"] >= 0) else None,
               subset_sha256=meta.get("subset_sha256"), lr_mean=None if ee["epoch"] < 0 else float(dreg.lr_epoch_means()[ee["epoch"]]),
               **dreg.stamp())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--files", default="dense", help="dense | core | lens | matched | transitions | avail | comma list of epoch indices")
    ap.add_argument("--init", action="store_true", help="also evaluate the verified init dump (E = 0) where present")
    ap.add_argument("--precision", default="tf32", choices=["fp32", "tf32", "bf16"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resident", type=int, default=16, help="checkpoints evaluated per pass over the images")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for result names (smoke precision comparisons)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    dreg.check_subsets()
    torch.backends.cudnn.benchmark = True
    splits = args.splits.split(",")
    for e in dreg.resolve_runs(args.runs):
        ents = [dreg.epoch_entry(e, i) for i in parse_files(args.files, e)]
        if args.init and init_entry(e) is not None:
            ents.append(init_entry(e))
        todo = [ee for ee in ents if args.force or not all(out_paths("rawlens", ee, s, args.limit, args.tag)[0].exists() for s in splits)]
        print(f"{e['name']}: {len(todo)} of {len(ents)} checkpoints to do", flush=True)
        for c in range(0, len(todo), args.resident):
            chunk = todo[c:c + args.resident]
            models = {}
            for ee in chunk:
                m = preg.load_model(ee, "cuda")
                models[ee["name"]] = m
            print(f"  resident: {[ee['name'].split('@')[1] for ee in chunk]}", flush=True)
            for split in splits:
                todo_split = [ee for ee in chunk if args.force or not out_paths("rawlens", ee, split, args.limit, args.tag)[0].exists()]
                if not todo_split:
                    continue
                acc, labels, meta = run_split({ee["name"]: models[ee["name"]] for ee in todo_split}, split, args)
                for ee in todo_split:
                    a = acc[ee["name"]]
                    a["k_bias_removed"] = getattr(models[ee["name"]], "k_bias_removed", None)
                    jp, npz = out_paths("rawlens", ee, split, args.limit, args.tag)
                    if jp.exists():
                        jp.unlink()                                             # invalidate before rebuilding
                    save_npz(npz, pred=a["pred"], ce=a["ce"], ce_smooth=a["ce_smooth"], margin=a["margin"], maxp=a["maxp"],
                             top5=a["top5"], labels=labels, halfdepth=np.array(HALF))
                    summ = summary(ee, split, a, labels, meta, args)
                    dreg.write_json(jp, summ)                                   # the JSON is the completion marker
                    print(f"  {ee['name']:22s} {split:4s} top1 b5/b8/b10/b11: " + "/".join(f"{summ['top1'][HALF.index(k)]:5.1f}" for k in ("b5", "b8", "b10", "b11"))
                          + f"  check agree={summ['final_check']['agree_top1']:.4f} maxdiff={summ['final_check']['max_abs_logit_diff']:.2e}"
                          + (f"  log={summ['test_acc1_log']}" if summ['test_acc1_log'] is not None else ""), flush=True)
                del acc
            del models
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
