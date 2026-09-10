"""Per-patch linear probes for semantic segmentation (VOC, ADE20K) and monocular depth (NYUv2), trained online.

Dataset-major with every requested model resident on the GPU.  Per batch each frozen model runs one bf16
forward that collects the patch-token maps at all 13 depths; its heads W[13, n_wd, 768, C] then take one Adam
step (all depths and weight decays at once).  Targets live on the 14x14 patch grid (majority label with an
ignore rule / mean log-depth), so all metrics are patch-level.  The (wd, epoch) pair is chosen per depth on the
held-out 10 % of train; the test split is the official val (VOC, ADE20K) / test (NYU) set.

  python -m probes.dense_probes --datasets voc,ade20k,nyu --models all [--limit 64] [--epochs 20]
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from probes import data as pdata
from probes import registry
from probes.train_probes import lr_at

WD_GRID = [0.0, 1e-4, 1e-2]
EPOCHS = {"voc": 20, "ade20k": 10, "nyu": 30}
EVAL_EVERY_EPOCHS = {"voc": 4, "ade20k": 2, "nyu": 5}
D = 768
N_DEPTH = 13


@torch.no_grad()
def token_maps(m, img):
    """-> [B,13,196,768] fp32 patch tokens after the embedding and after every block."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = m.patch_embed(img)
        x = m._pos_embed(x)
        x = m.norm_pre(x)
        out = [x[:, 1:].float()]
        for blk in m.blocks:
            x = blk(x)
            out.append(x[:, 1:].float())
    return torch.stack(out, 1)


class Heads:
    def __init__(self, C, device):
        self.C = C
        self.W = torch.zeros(N_DEPTH, len(WD_GRID), D, C, device=device, requires_grad=True)
        self.b = torch.zeros(N_DEPTH, len(WD_GRID), C, device=device, requires_grad=True)
        self.opt = torch.optim.Adam([self.W, self.b], lr=1e-3)
        self.wd = torch.tensor(WD_GRID, device=device).view(1, -1, 1, 1)
        self.mu = None
        self.sd = None

    def logits(self, f):                                            # f [B,13,196,768] -> [13,nw,B,196,C]
        x = (f - self.mu) / self.sd
        return torch.einsum("bkpd,kwdc->kwbpc", x, self.W) + self.b[:, :, None, None]


def seg_loss(logits, y):                                            # y [B,14,14] long (-1 ignore)
    K, nw, B, P, C = logits.shape
    t = y.view(B, P)
    valid = t >= 0
    tt = t.clamp_min(0)
    lg = logits.reshape(K * nw, B * P, C)
    ce = F.cross_entropy(lg.transpose(1, 2), tt.view(1, -1).expand(K * nw, -1), reduction="none")   # [K*nw, B*P]
    ce = (ce * valid.view(1, -1)).sum(1) / valid.sum().clamp_min(1)
    return ce.sum()


def depth_loss(logits, y):                                          # y [B,14,14] float (nan ignore)
    K, nw, B, P, _ = logits.shape
    t = y.view(B, P)
    valid = torch.isfinite(t)
    tt = torch.where(valid, t, torch.zeros_like(t))
    err = (logits[..., 0] - tt.view(1, 1, B, P)) ** 2
    return ((err * valid.view(1, 1, B, P)).sum((2, 3)) / valid.sum().clamp_min(1)).sum()


@torch.no_grad()
def seg_metrics(conf):
    """conf [13,nw,C,C] (rows = truth) -> mIoU [13,nw], patch acc [13,nw]."""
    tp = torch.diagonal(conf, dim1=-2, dim2=-1)
    fn = conf.sum(-1) - tp
    fp = conf.sum(-2) - tp
    iou = tp / (tp + fn + fp).clamp_min(1)
    present = conf.sum(-1) > 0
    miou = (iou * present).sum(-1) / present.sum(-1).clamp_min(1)
    acc = tp.sum(-1) / conf.sum((-1, -2)).clamp_min(1)
    return 100 * miou, 100 * acc


@torch.no_grad()
def evaluate(nets, heads, loader, task, C):
    stats = {n: dict(conf=torch.zeros(N_DEPTH, len(WD_GRID), C, C, device="cuda")) if task == "seg" else
             dict(se=torch.zeros(N_DEPTH, len(WD_GRID), device="cuda"), ae=torch.zeros(N_DEPTH, len(WD_GRID), device="cuda"),
                  d1=torch.zeros(N_DEPTH, len(WD_GRID), device="cuda"), n=0, ss_tot=0.0, tsum=0.0, tsq=0.0)
             for n in nets}
    for img, y in loader:
        img, y = img.cuda(non_blocking=True), y.cuda(non_blocking=True)
        for n, m in nets.items():
            lg = heads[n].logits(token_maps(m, img))                  # [13,nw,B,196,C]
            st = stats[n]
            if task == "seg":
                t = y.view(-1)
                valid = t >= 0
                pred = lg.argmax(-1).reshape(N_DEPTH, len(WD_GRID), -1)[:, :, valid]
                tt = t[valid]
                idx = (tt[None, None] * C + pred).reshape(N_DEPTH, len(WD_GRID), -1)
                for k in range(N_DEPTH):
                    for w in range(len(WD_GRID)):
                        st["conf"][k, w] += torch.bincount(idx[k, w], minlength=C * C).view(C, C).float()
            else:
                t = y.view(-1)
                valid = torch.isfinite(t)
                p = lg[..., 0].reshape(N_DEPTH, len(WD_GRID), -1)[:, :, valid]
                tt = t[valid]
                st["se"] += ((p.exp() - tt.exp()) ** 2).sum(-1)
                st["ae"] += ((p.exp() - tt.exp()).abs() / tt.exp()).sum(-1)
                ratio = torch.maximum((p - tt).exp(), (tt - p).exp())
                st["d1"] += (ratio < 1.25).float().sum(-1)
                st["n"] += int(valid.sum())
                st["tsum"] += float(tt.sum()); st["tsq"] += float((tt * tt).sum())
                st.setdefault("se_log", torch.zeros_like(st["se"]))
                st["se_log"] += ((p - tt) ** 2).sum(-1)
    out = {}
    for n, st in stats.items():
        if task == "seg":
            miou, acc = seg_metrics(st["conf"])
            out[n] = dict(miou=miou.cpu(), acc=acc.cpu())
        else:
            N = max(st["n"], 1)
            var_t = st["tsq"] / N - (st["tsum"] / N) ** 2
            out[n] = dict(rmse=(st["se"] / N).sqrt().cpu(), absrel=(st["ae"] / N).cpu(), delta1=(100 * st["d1"] / N).cpu(),
                          r2_log=(1 - (st["se_log"] / N) / max(var_t, 1e-8)).cpu())
    return out


def run(models, dataset, limit, epochs, batch, workers):
    task = "depth" if dataset == "nyu" else "seg"
    todo = [e for e in models if not (registry.RES / "dense" / f"{e['name']}__{dataset}.json").exists()]
    if not todo:
        print(f"[{dataset}] all models done", flush=True); return
    ds_tr, C, meta = pdata.build_dense_dataset(dataset, "train")
    ds_va, _, _ = pdata.build_dense_dataset(dataset, "val")
    ds_te, _, _ = pdata.build_dense_dataset(dataset, "test")
    if limit:
        ds_tr, ds_va, ds_te = (torch.utils.data.Subset(d, list(range(min(limit, len(d))))) for d in (ds_tr, ds_va, ds_te))
    mk = lambda d, sh: DataLoader(d, batch_size=batch, shuffle=sh, num_workers=workers, pin_memory=True, drop_last=False)
    ltr, lva, lte = mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)
    nets = {e["name"]: registry.load_model(e, "cuda") for e in todo}
    heads = {n: Heads(C, "cuda") for n in nets}
    # per-depth standardisation statistics from the first ~256 training images
    with torch.no_grad():
        acc = {n: [torch.zeros(N_DEPTH, D, device="cuda"), torch.zeros(N_DEPTH, D, device="cuda"), 0] for n in nets}
        seen = 0
        for img, _ in ltr:
            img = img.cuda()
            for n, m in nets.items():
                f = token_maps(m, img)
                acc[n][0] += f.sum((0, 2)); acc[n][1] += (f * f).sum((0, 2)); acc[n][2] += f.shape[0] * f.shape[2]
            seen += img.shape[0]
            if seen >= 256:
                break
        for n in nets:
            s1, s2, cnt = acc[n]
            mu = s1 / cnt
            heads[n].mu = mu[None, :, None, :]
            heads[n].sd = ((s2 / cnt - mu * mu).clamp_min(0).sqrt().clamp_min(1e-6))[None, :, None, :]
    steps_per_epoch = len(ltr)
    total = epochs * steps_per_epoch
    step = 0
    hist = {n: [] for n in nets}
    best = {n: dict(score=torch.full((N_DEPTH, len(WD_GRID)), -1e9), epoch=torch.zeros(N_DEPTH, len(WD_GRID), dtype=torch.long),
                    test=None) for n in nets}
    t0 = time.time()
    for ep in range(epochs):
        for img, y in ltr:
            img, y = img.cuda(non_blocking=True), y.cuda(non_blocking=True)
            lr = lr_at(step, total)
            for n, m in nets.items():
                h = heads[n]
                for grp in h.opt.param_groups:
                    grp["lr"] = lr
                lg = h.logits(token_maps(m, img))
                loss = (seg_loss(lg, y) if task == "seg" else depth_loss(lg, y)) + 0.5 * (h.wd * h.W * h.W).sum()
                h.opt.zero_grad(set_to_none=True); loss.backward(); h.opt.step()
            step += 1
        if (ep + 1) % EVAL_EVERY_EPOCHS[dataset] == 0 or ep + 1 == epochs:
            ev_va = evaluate(nets, heads, lva, task, C)
            ev_te = evaluate(nets, heads, lte, task, C)
            key = "miou" if task == "seg" else "delta1"
            for n in nets:
                hist[n].append(dict(epoch=ep + 1, val={k: v.tolist() for k, v in ev_va[n].items()}, test={k: v.tolist() for k, v in ev_te[n].items()}))
                s = ev_va[n][key]
                better = s > best[n]["score"]
                best[n]["score"] = torch.where(better, s, best[n]["score"])
                best[n]["epoch"] = torch.where(better, torch.full_like(best[n]["epoch"], ep + 1), best[n]["epoch"])
                if best[n]["test"] is None:
                    best[n]["test"] = {k: v.clone() for k, v in ev_te[n].items()}
                else:
                    for k, v in ev_te[n].items():
                        best[n]["test"][k] = torch.where(better, v, best[n]["test"][k])
            print(f"[{dataset}] epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s) " + " | ".join(
                f"{n}: {key} b11 {float(ev_te[n][key][12].max()):.1f}" for n in list(nets)[:4]), flush=True)
    for e in todo:
        n = e["name"]
        b = best[n]
        wsel = b["score"].argmax(1)                                   # [13] best wd per depth by val
        rec = dict(model=n, arm=e["arm"], seed=e["seed"], ckpt=str(registry.ckpt_path(e)), dataset=dataset, task=task, n_classes=C,
                   depths=registry.DEPTHS, wd_grid=WD_GRID, epochs=epochs, n_train=len(ds_tr), n_val=len(ds_va), n_test=len(ds_te),
                   best_wd=[WD_GRID[int(i)] for i in wsel], best_epoch=b["epoch"].gather(1, wsel[:, None]).squeeze(1).tolist(),
                   history=hist[n], git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"), limit=limit)
        for k, v in b["test"].items():
            rec[k] = v.gather(1, wsel[:, None]).squeeze(1).tolist()
            rec[k + "_grid"] = v.tolist()
        outp = registry.RES / "dense" / f"{n}__{dataset}.json"
        tmp = outp.with_suffix(".tmp"); json.dump(rec, open(tmp, "w")); os.replace(tmp, outp)
        key = "miou" if task == "seg" else "delta1"
        print(f"{n:14s} {dataset:7s} {key} by depth: " + " ".join(f"{a:5.1f}" for a in rec[key]), flush=True)
    del nets; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="voc,ade20k,nyu")
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="override the per-dataset default")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    for dataset in args.datasets.split(","):
        run(models, dataset, args.limit, args.epochs or EPOCHS[dataset], args.batch, args.workers)


if __name__ == "__main__":
    main()
