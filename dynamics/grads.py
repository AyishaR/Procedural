"""Experiment 4 (supplement): gradient snapshots at saved weights on fixed diagnostic batches.

For every (run, epoch file), set and objective, the exact full-batch gradient of the mean loss is accumulated
over chunks (fp32, TF32 off, no autocast): per-block and per-tensor gradient norms (plain, plus the training
loop's own double-counting grouping for parity with grad_norms_IMNET_BASE_*.json), per-block parameter norms
without the k-bias (-> relative gradient), the late share, the loss, and the output-side quantities that
drive the gradient: mean p_y, p_y - q_y, ||p - q||_1 and the fraction of images with p_y > q_y (where the
true-class gradient flips sign).  Objectives: 'smooth' = CE against the label-smoothed target (the training
criterion without mixup/cutmix, q_y = 1 - s + s/C) and 'hard' = plain CE.  These are newly evaluated
gradients at saved weights, not the historical training gradients.

  python -m dynamics.grads --runs all --files dense
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from dynamics import common
from dynamics import data as ddata
from dynamics import registry as dreg
from dynamics.rawlens import parse_files
from probes import registry as preg


class Stats:
    def __init__(self, C, smoothing=common.SMOOTHING):
        self.C, self.s = C, smoothing
        self.sums = torch.zeros(8, dtype=torch.float64, device="cuda")
        self.n = 0

    def __call__(self, logits, y):
        p = F.softmax(logits.float(), -1)
        p_y = p.gather(1, y[:, None])[:, 0]
        q_y = 1 - self.s + self.s / self.C
        q = torch.full_like(p, self.s / self.C).scatter_(1, y[:, None], q_y)
        l1_smooth = (p - q).abs().sum(-1)
        l1_hard = 2 * (1 - p_y)
        self.sums += torch.stack([(logits.argmax(-1) == y).double().sum(), p_y.double().sum(), (p_y - q_y).double().sum(),
                                  l1_smooth.double().sum(), l1_hard.double().sum(), (p_y > q_y).double().sum(),
                                  p.max(-1).values.double().sum(), (-torch.log(p_y.clamp_min(1e-12))).double().sum()])
        self.n += int(y.shape[0])

    def result(self):
        s = (self.sums / self.n).cpu().tolist()
        return dict(top1=100 * s[0], mean_p_y=s[1], mean_p_y_minus_q_y=s[2], mean_l1_p_minus_q=s[3], mean_l1_p_minus_q_smooth=s[3] if self.s > 0 else None,
                    mean_l1_p_minus_q_hard=s[4], frac_p_y_gt_q_y=s[5], mean_maxp=s[6], ce_hard=s[7], q_y=1 - self.s + self.s / self.C,
                    smoothing=self.s, n=self.n, note="q is the objective's own target: smoothed (q_y = 1 - s + s/C) or one-hot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--files", default="dense", help="dense | core | comma list of epoch indices")
    ap.add_argument("--sets", default="diag_train4k,diag_test4k")
    ap.add_argument("--objectives", default="smooth,hard")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    dreg.check_subsets()
    sets = {s: ddata.load_resident(s, "cuda", torch.float16, limit=args.limit) for s in args.sets.split(",")}
    objectives = args.objectives.split(",")
    lr = dreg.lr_epoch_means()
    suf = f"__limit{args.limit}" if args.limit else ""
    m = preg.new_model().cuda().eval()
    with common.precision("fp32"):
        for e in dreg.resolve_runs(args.runs):
            files = parse_files(args.files, e)
            todo = [i for i in files if args.force or not (dreg.RES / "grads" / f"{e['name']}@e{i}{suf}.json").exists()]
            print(f"{e['name']}: {len(todo)} of {len(files)} checkpoints to do", flush=True)
            for i in todo:
                t0 = time.time()
                ee = dreg.epoch_entry(e, i)
                kb = preg.load_state(m, ee)
                pn = common.param_norms(m)
                rec = dict(run=e["name"], arm=e["arm"], seed=e["seed"], role=dreg.ROLE[e["name"]], file=i, E=dreg.epoch_of(i),
                           lr_mean=float(lr[i]), k_bias_removed=kb, param_norm_block=[pn["per_block"][l] for l in range(12)],
                           param_norm_other=pn["other"], chunk=args.chunk, limit=args.limit, sets={}, **dreg.stamp())
                for name, (X, y, meta) in sets.items():
                    rs = dict(n=int(X.shape[0]), subset_sha256=meta.get("subset_sha256"), objectives={})
                    for obj in objectives:
                        st = Stats(1000, smoothing=common.SMOOTHING if obj == "smooth" else 0.0)   # q of the objective itself
                        loss, chunk_norms = common.accumulate_grads(m, X, y, obj, args.chunk, forward=lambda xb: m(xb.float()), on_logits=st)
                        gn = common.grad_norms(m)
                        ht, hg, hpb = common.hist_style_grad_norms(m)
                        pb = [gn["per_block"][l] for l in range(12)]
                        late = sum(v * v for v in pb[9:]) / max(1e-30, sum(v * v for v in pb))
                        rs["objectives"][obj] = dict(loss=loss, total=gn["total"], per_block=pb, other=gn["other"], late_share=late,
                                                     rel_block=[g / p for g, p in zip(pb, rec["param_norm_block"])],
                                                     per_tensor=gn["per_tensor"], qkv_split=gn["qkv_split"], hist_total=ht,
                                                     hist_per_block=[hpb[l] for l in range(12)], hist_groups=hg,
                                                     chunk_norms=chunk_norms, stats=st.result())
                    rec["sets"][name] = rs
                rec["seconds"] = time.time() - t0
                dreg.write_json(dreg.RES / "grads" / f"{e['name']}@e{i}{suf}.json", rec)
                r0 = next(iter(rec["sets"].values()))["objectives"][objectives[0]]
                print(f"  e{i:3d} loss={r0['loss']:.3f} |g| total={r0['total']:.3e} b0..b11: " + " ".join(f"{v:.1e}" for v in r0["per_block"])
                      + f"  late={r0['late_share']:.2f} p_y>q_y={r0['stats']['frac_p_y_gt_q_y']:.2f} ({rec['seconds']:.0f}s)", flush=True)
            m.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
