"""Experiment 2: how useful is each block at each training stage?

For every (run, epoch file) and each fixed set: the residual stream is computed once per batch; block l is then
replaced by a gain-scaled version (gain 0 = bypass, i.e. the identity; 0.9 = mild attenuation of both residual
branches) and blocks l+1..11 are recomputed from the cached prefix state, so prefix arithmetic cancels exactly
in every loss difference.  C_l(e) = L_bypass - L_full (hard-label CE; also the label-smoothed CE), the
accuracy lost, and the counts of images corrected / harmed by including the block.  Bypassing the last 2 / 3
blocks together = the checkpoint's own readout on b9 / b8 (recorded here as well).  In addition the exact local
sensitivity dL/dg of every image's loss to each of the 24 branch gains at g = 1 (one forward + backward per
objective).  fp32 stream with TF32 matmuls by default.

  python -m dynamics.bypass --runs all --files core [--gains 0,0.9] [--precision tf32] [--limit 512]
"""
import argparse
import os
import time

import numpy as np
import torch

from dynamics import common
from dynamics import data as ddata
from dynamics import registry as dreg
from dynamics.rawlens import out_paths, parse_files, save_npz
from probes import registry as preg

MULTI = {"last2": 10, "last3": 9}          # readout on h_10 / h_9 == bypass of blocks 10-11 / 9-11


def files_for(spec, e):
    """'core+matched+transitions' style unions of the parse_files specs."""
    out = set()
    for part in spec.split("+"):
        out |= set(parse_files(part, e))
    return sorted(out)


def run_split(m, split, args, gains):
    ds, meta = ddata.dataset(split, args.limit)
    N, L, G = len(ds), len(m.blocks), len(gains)
    per = dict(ce_full=np.zeros(N, np.float16), pred_full=np.zeros(N, np.int16), labels=np.zeros(N, np.int16),
               ce=np.zeros((N, L, G), np.float16), ce_smooth=np.zeros((N, L, G), np.float16), pred=np.zeros((N, L, G), np.int16),
               ce_multi=np.zeros((N, len(MULTI)), np.float16), pred_multi=np.zeros((N, len(MULTI)), np.int16))
    objs = ("hard", "smooth") if args.gain_grad else ()
    for o in objs:
        per[f"gain_grad_{o}"] = np.zeros((N, L, 2), np.float16)
    # sums: [variant, stat] with stats = ce, ce_smooth, correct, corrected, harmed ; variants = L*G + len(MULTI)
    V = L * G + len(MULTI)
    sums = torch.zeros(V, 5, dtype=torch.float64, device="cuda")
    full_sums = torch.zeros(3, dtype=torch.float64, device="cuda")
    gg_sum = {o: torch.zeros(L, 2, dtype=torch.float64, device="cuda") for o in objs}
    gg_sq = {o: torch.zeros(L, 2, dtype=torch.float64, device="cuda") for o in objs}
    gg_abs = {o: torch.zeros(L, 2, dtype=torch.float64, device="cuda") for o in objs}
    seen, t0 = 0, time.time()
    for img, y in ddata.loader(ds, args.batch, args.workers):
        img = img.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        B = img.shape[0]
        sl = slice(seen, seen + B)
        per["labels"][sl] = y.cpu().numpy()
        with torch.no_grad(), common.precision(args.precision):
            hs = common.states(m, img)
            pf = common.per_image(common.head_on(m, hs[-1]), y)
            full_sums += torch.stack([pf["ce"].double().sum(), pf["ce_smooth"].double().sum(), pf["correct"].double().sum()])
            per["ce_full"][sl] = pf["ce"].cpu().numpy(); per["pred_full"][sl] = pf["pred"].cpu().numpy()

            def record(v, pm):
                sums[v] += torch.stack([pm["ce"].double().sum(), pm["ce_smooth"].double().sum(), pm["correct"].double().sum(),
                                        (~pm["correct"] & pf["correct"]).double().sum(), (pm["correct"] & ~pf["correct"]).double().sum()])

            for gi, g in enumerate(gains):
                for l in range(L):
                    h = common.block_forward(m.blocks[l], hs[l], g, g)
                    pm = common.per_image(common.head_on(m, common.suffix(m, h, l + 1)), y)
                    record(l * G + gi, pm)
                    per["ce"][sl, l, gi] = pm["ce"].cpu().numpy(); per["ce_smooth"][sl, l, gi] = pm["ce_smooth"].cpu().numpy()
                    per["pred"][sl, l, gi] = pm["pred"].cpu().numpy()
            for k, (name, d) in enumerate(MULTI.items()):
                pm = common.per_image(common.head_on(m, hs[d]), y)
                record(L * G + k, pm)
                per["ce_multi"][sl, k] = pm["ce"].cpu().numpy(); per["pred_multi"][sl, k] = pm["pred"].cpu().numpy()
        for o in objs:
            g = torch.ones(B, L, 2, device="cuda", requires_grad=True)
            with common.precision(args.precision):
                logits = common.head_on(m, common.forward_with_gains(m, img, g))
            (common.loss_fn(logits, y, o) * B).backward()                     # sum of per-image losses -> per-image dL_i/dg_i
            gr = g.grad.detach()
            gg_sum[o] += gr.double().sum(0); gg_sq[o] += gr.double().pow(2).sum(0); gg_abs[o] += gr.double().abs().sum(0)
            per[f"gain_grad_{o}"][sl] = gr.cpu().numpy()
        seen += B
        if seen % (args.batch * 10) == 0 or seen == N:
            print(f"  [{split}] {seen}/{N} imgs, {seen / (time.time() - t0):.0f} img/s", flush=True)
    assert seen == N
    s = (sums / N).cpu().numpy()
    f = (full_sums / N).cpu().numpy()
    out = dict(split=split, n=N, subset_sha256=meta.get("subset_sha256"), gains=gains,
               full=dict(ce=float(f[0]), ce_smooth=float(f[1]), top1=float(100 * f[2])), by_gain={})
    for gi, g in enumerate(gains):
        rows = s[[l * G + gi for l in range(L)]]
        out["by_gain"][str(g)] = dict(ce=rows[:, 0].tolist(), C=(rows[:, 0] - f[0]).tolist(), C_smooth=(rows[:, 1] - f[1]).tolist(),
                                      top1=(100 * rows[:, 2]).tolist(), dacc=(100 * (f[2] - rows[:, 2])).tolist(),
                                      corrected=np.rint(N * rows[:, 3]).astype(int).tolist(), harmed=np.rint(N * rows[:, 4]).astype(int).tolist())
    out["multi"] = {}
    for k, name in enumerate(MULTI):
        r = s[L * G + k]
        out["multi"][name] = dict(ce=float(r[0]), C=float(r[0] - f[0]), C_smooth=float(r[1] - f[1]), top1=float(100 * r[2]),
                                  dacc=float(100 * (f[2] - r[2])), corrected=int(round(N * r[3])), harmed=int(round(N * r[4])))
    if objs:
        out["gain_grad"] = {}
        for o in objs:
            mean = (gg_sum[o] / N).cpu().numpy()
            sd = ((gg_sq[o] / N).cpu().numpy() - mean ** 2).clip(0) ** 0.5
            out["gain_grad"][o] = dict(mean=mean.tolist(), sd=sd.tolist(), mean_abs=(gg_abs[o] / N).cpu().numpy().tolist(),
                                       note="dL/dg at g=1 per branch [block][attn, mlp]; mean/sd over images; loss = mean CE ->" " per-image gradient of its own loss")
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--files", default="core+matched+transitions", help="'+'-union of: core | matched | transitions | dense | avail | comma list")
    ap.add_argument("--gains", default="0,0.9")
    ap.add_argument("--precision", default="tf32", choices=["fp32", "tf32", "bf16"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--no-gain-grad", dest="gain_grad", action="store_false")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    dreg.check_subsets()
    torch.backends.cudnn.benchmark = True
    gains = [float(g) for g in args.gains.split(",")]
    splits = args.splits.split(",")
    for e in dreg.resolve_runs(args.runs):
        ents = [dreg.epoch_entry(e, i) for i in files_for(args.files, e)]
        todo = [ee for ee in ents if args.force or not all(out_paths("bypass", ee, s, args.limit, args.tag)[0].exists() for s in splits)]
        print(f"{e['name']}: {len(todo)} of {len(ents)} checkpoints to do", flush=True)
        for ee in todo:
            m = preg.load_model(ee, "cuda")
            for p in m.parameters():
                p.requires_grad_(False)
            for split in splits:
                jp, npz = out_paths("bypass", ee, split, args.limit, args.tag)
                if jp.exists() and not args.force:
                    continue
                t0 = time.time()
                out, per = run_split(m, split, args, gains)
                out.update(run=ee["base"], arm=ee["arm"], seed=ee["seed"], role=dreg.ROLE[ee["base"]], file=ee["epoch"], E=dreg.epoch_of(ee["epoch"]),
                           precision=args.precision, tag=args.tag, limit=args.limit, ckpt=str(preg.ckpt_path(ee)), k_bias_removed=m.k_bias_removed,
                           lr_mean=float(dreg.lr_epoch_means()[ee["epoch"]]), seconds=time.time() - t0, **dreg.stamp())
                if jp.exists():
                    jp.unlink()
                save_npz(npz, **per)
                dreg.write_json(jp, out)
                c0 = out["by_gain"][str(gains[0])]["C"]
                print(f"  {ee['name']:22s} {split:4s} L_full={out['full']['ce']:.3f} top1={out['full']['top1']:.2f}  C(g={gains[0]}) b0..b11: "
                      + " ".join(f"{c:.2f}" for c in c0) + f"  last3 C={out['multi']['last3']['C']:.2f}  ({out['seconds']:.0f}s)", flush=True)
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
