"""Layerwise linear probes on cached features: one batched logistic regression per (depth, weight decay).

For every (model, dataset, ftype) all 13 depths x 6 weight decays are trained simultaneously with a single
einsum (Adam's element-wise update makes the summed loss exactly equivalent to independent runs).  Features are
standardised per depth with train statistics; the (wd, step) pair is selected per depth on the held-out val
split and the test accuracy at that point is reported.  For ImageNet the selected 1000-class probes are also
evaluated zero-shot on ImageNet-R (logits masked to its 200 classes).

  python -m probes.train_probes --models all --datasets all --ftypes cls,mean,concat [--steps 2000]
"""
import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from probes import data as pdata
from probes import features as feat
from probes import registry

WD_GRID = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
EVAL_EVERY = 100


def lr_at(step, total, base=1e-3, warmup=100, final=1e-5):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(X, y, mu, sd, W, b, mask=None, chunk=2048, per_class=False):
    """X [N,13,D] fp16, W [13,n_wd,D,C] -> top-1 [13,n_wd] (and mean per-class acc if requested)."""
    K, nw, _, C = W.shape
    correct = torch.zeros(K, nw, device=W.device)
    if per_class:
        cls_correct = torch.zeros(K, nw, C, device=W.device)
        cls_count = torch.zeros(C, device=W.device)
    for s in range(0, X.shape[0], chunk):
        x = (X[s:s + chunk].float() - mu) / sd                       # [B,13,D]
        yy = y[s:s + chunk]
        logits = torch.einsum("bkd,kwdc->kwbc", x, W) + b[:, :, None]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        pred = logits.argmax(-1)                                     # [13,nw,B]
        hit = (pred == yy[None, None]).float()
        correct += hit.sum(-1)
        if per_class:
            cls_correct.index_add_(2, yy, hit)
            cls_count.index_add_(0, yy, torch.ones_like(yy, dtype=torch.float))
    acc = 100 * correct / X.shape[0]
    if per_class:
        present = cls_count > 0
        return acc, 100 * (cls_correct[:, :, present] / cls_count[present]).mean(-1)
    return acc


def train_one(model, dataset, ftype, steps, batch, device="cuda", seed=0):
    assert steps >= EVAL_EVERY, f"steps must be >= {EVAL_EVERY} (one validation point)"
    tr, va, te = (feat.load(model, dataset, s) for s in ("train", "val", "test"))
    C = int(tr["meta"]["n_classes"])
    Xtr, Xva, Xte = (feat.select(d["X"], ftype, device) for d in (tr, va, te))
    ytr, yva, yte = (torch.from_numpy(d["y"].astype(np.int64)).to(device) for d in (tr, va, te))
    mu, sd = feat.standardise_stats(Xtr)
    N, K, D = Xtr.shape
    nw = len(WD_GRID)
    wd = torch.tensor(WD_GRID, device=device).view(1, nw, 1, 1)
    W = torch.zeros(K, nw, D, C, device=device, requires_grad=True)
    b = torch.zeros(K, nw, C, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-3, betas=(0.9, 0.999))
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = min(batch, N)
    perm, ptr = torch.randperm(N, generator=g), 0
    n_eval = steps // EVAL_EVERY
    val_curve = torch.zeros(n_eval, K, nw)
    test_curve = torch.zeros(n_eval, K, nw)
    # running best per depth over (wd, step) by val accuracy: the selected probe's weights are snapshotted so
    # every reported number (test, per-class, train, ImageNet-R) comes from exactly the selected probe
    best_val = torch.full((K,), -1.0)
    best_wd_i = torch.zeros(K, dtype=torch.long)
    best_step = torch.zeros(K, dtype=torch.long)
    best_W = torch.zeros(K, 1, D, C, device=device)
    best_b = torch.zeros(K, 1, C, device=device)
    t0 = time.time()
    for step in range(steps):
        if ptr + B > N:
            perm, ptr = torch.randperm(N, generator=g), 0
        idx = perm[ptr:ptr + B].to(device)
        ptr += B
        x = (Xtr[idx].float() - mu) / sd
        yy = ytr[idx]
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, steps)
        logits = torch.einsum("bkd,kwdc->kwbc", x, W) + b[:, :, None]
        ce = F.cross_entropy(logits.reshape(-1, C), yy.repeat(K * nw), reduction="none").view(K, nw, B).mean(-1).sum()
        loss = ce + 0.5 * (wd * W * W).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            i = (step + 1) // EVAL_EVERY - 1
            va_acc = evaluate(Xva, yva, mu, sd, W.detach(), b.detach()).cpu()
            val_curve[i] = va_acc
            test_curve[i] = evaluate(Xte, yte, mu, sd, W.detach(), b.detach()).cpu()
            cur_best, cur_w = va_acc.max(1)                            # [K]
            improved = cur_best > best_val
            for k in torch.nonzero(improved).flatten().tolist():
                best_val[k], best_wd_i[k], best_step[k] = cur_best[k], cur_w[k], step + 1
                best_W[k, 0] = W.detach()[k, cur_w[k]]
                best_b[k, 0] = b.detach()[k, cur_w[k]]
    del W, b, opt
    test_sel, pc = evaluate(Xte, yte, mu, sd, best_W, best_b, per_class=True)
    n_tr_eval = min(20000, N)
    tr_idx = torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(seed + 1))[:n_tr_eval].to(device)
    train_sel = evaluate(Xtr[tr_idx], ytr[tr_idx], mu, sd, best_W, best_b)
    out = dict(
        test_acc=test_sel[:, 0].cpu().tolist(), val_acc=best_val.tolist(),
        best_wd=[WD_GRID[int(i)] for i in best_wd_i], best_step=best_step.tolist(),
        val_curve=val_curve.permute(1, 2, 0).tolist(), test_curve=test_curve.permute(1, 2, 0).tolist(),
        train_acc=train_sel[:, 0].cpu().tolist(), train_acc_n=int(n_tr_eval), train_acc_sampling="seeded uniform subset of the train split",
        mean_class_acc=pc[:, 0].cpu().tolist(),
        seconds=time.time() - t0,
    )
    if dataset == "imnet" and feat.is_done(model, "imagenet_r", "test"):
        r = feat.load(model, "imagenet_r", "test")
        Xr = feat.select(r["X"], ftype, device)
        yr = torch.from_numpy(r["y"].astype(np.int64)).to(device)
        mask = torch.from_numpy(pdata.imagenet_r_class_mask()).to(device)
        out["imagenet_r_acc"] = evaluate(Xr, yr, mu, sd, best_W, best_b, mask=mask)[:, 0].cpu().tolist()
        del Xr
    del Xtr, Xva, Xte, best_W, best_b
    torch.cuda.empty_cache()
    return out, dict(n_train=N, n_val=int(va["y"].shape[0]), n_test=int(te["y"].shape[0]), n_classes=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--ftypes", default="cls,mean,concat")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    models = registry.resolve_models(args.models)
    names = pdata.CLASSIFICATION if args.datasets == "all" else args.datasets.split(",")
    ftypes = args.ftypes.split(",")
    for e in models:
        for dataset in names:
            outp = registry.RES / "probes" / f"{e['name']}__{dataset}{feat.result_suffix(e['name'], dataset)}.json"
            if outp.exists() and not args.force:
                print(f"skip {outp.name} (exists)", flush=True)
                continue
            if not all(feat.is_done(e["name"], dataset, s) for s in ("train", "val", "test")):
                print(f"skip {e['name']} {dataset}: features not cached", flush=True)
                continue
            rec = dict(model=e["name"], arm=e["arm"], seed=e["seed"], ckpt=str(registry.ckpt_path(e)), dataset=dataset,
                       depths=registry.DEPTHS, wd_grid=WD_GRID, eval_steps=[(i + 1) * EVAL_EVERY for i in range(args.steps // EVAL_EVERY)],
                       config=dict(steps=args.steps, batch=args.batch, lr=1e-3, warmup=100, probe_seed=0),
                       git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"), ftypes={})
            for ft in ftypes:
                res, sizes = train_one(e["name"], dataset, ft, args.steps, args.batch)
                rec["ftypes"][ft] = res
                rec.update(sizes)
                print(f"{e['name']:14s} {dataset:10s} {ft:6s} test top-1 by depth: " + " ".join(f"{a:5.1f}" for a in res["test_acc"]) + f"  ({res['seconds']:.0f}s)", flush=True)
            tmp = outp.with_suffix(".tmp")
            json.dump(rec, open(tmp, "w"))
            os.replace(tmp, outp)


if __name__ == "__main__":
    main()
