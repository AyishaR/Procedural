"""Extract pooled residual-stream features at 13 depths for every (model, dataset, split) into the feature cache.

Dataset-major: each decoded batch is pushed through every requested model (all resident on the GPU), which
amortises JPEG decoding, the actual bottleneck.  No hooks: the ViT forward loop is mirrored so the pooled
stream is read after the embedding and after every block, and the final logits come for free (end-to-end check).

  python -m probes.extract_features --datasets imnet,cifar100 --models all --batch_size 256 --workers 16 [--limit 512]
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from probes import data as pdata
from probes import registry
from probes.features import LOWLEVEL_TARGETS, cache_dir

N_DEPTH = 13
D = 768


def lowlevel_stats(img):
    """[B,3,224,224] normalised batch -> [B,10] low-level image statistics from the de-normalised RGB crop."""
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=img.device).view(1, 3, 1, 1)
    x = (img.float() * std + mean).clamp(0, 1)
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gx = lum[:, :, 1:] - lum[:, :, :-1]
    gy = lum[:, 1:, :] - lum[:, :-1, :]
    grad = (gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2).sqrt().mean((1, 2))
    lap = (lum[:, 1:-1, 1:-1] * 4 - lum[:, :-2, 1:-1] - lum[:, 2:, 1:-1] - lum[:, 1:-1, :-2] - lum[:, 1:-1, 2:])
    mx, mn = x.max(1).values, x.min(1).values
    sat = ((mx - mn) / mx.clamp_min(1e-3)).mean((1, 2))
    return torch.stack([*x.mean((2, 3)).unbind(1), *x.std((2, 3)).unbind(1), lum.std((1, 2)), grad, lap.var((1, 2)), sat], 1)


@torch.no_grad()
def forward_depths(m, img):
    """-> pooled [B,13,2,768] fp32 and final logits [B,1000]."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = m.patch_embed(img)
        x = m._pos_embed(x)
        x = m.norm_pre(x)
        feats = [torch.stack([x[:, 0].float(), x[:, 1:].float().mean(1)], 1)]
        for blk in m.blocks:
            x = blk(x)
            feats.append(torch.stack([x[:, 0].float(), x[:, 1:].float().mean(1)], 1))
        logits = m.head(m.fc_norm(m.norm(x)[:, 0])).float()
    return torch.stack(feats, 1), logits


def extract(models, dataset, split, batch_size, workers, limit=None):
    def cached(e):
        d = cache_dir(e["name"], dataset, split)
        if not (d / "_DONE").exists():
            return False
        meta = json.load(open(d / "meta.json"))
        if meta.get("limit") != limit:                    # a limited (smoke) cache never counts as the full one
            return False
        mt = meta.get("ckpt_mtime")                       # provenance: the cache must come from the current checkpoint file
        return mt is None or abs(mt - os.path.getmtime(registry.ckpt_path(e))) < 1.0
    todo = [e for e in models if not cached(e)]
    if not todo:
        print(f"[{dataset}/{split}] all {len(models)} models cached", flush=True)
        return
    ds, C, meta = pdata.build_probe_dataset(dataset, split)
    N = len(ds) if limit is None else min(limit, len(ds))
    if limit is not None:
        ds = torch.utils.data.Subset(ds, list(range(N)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False, persistent_workers=False)
    ents = {e["name"]: e for e in todo}
    nets = {e["name"]: registry.load_model(e, "cuda") for e in todo}
    want_lowlevel = dataset in ("imnet", "places365")
    writers = {}
    for name in nets:
        d = cache_dir(name, dataset, split)
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
        writers[name] = dict(
            feats=np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16, shape=(N, N_DEPTH, 2, D)),
            labels=np.zeros(N, dtype=np.int16), preds=np.zeros(N, dtype=np.int16), index=np.arange(N, dtype=np.int32),
            maxabs=np.zeros(N_DEPTH), sumnorm=np.zeros(N_DEPTH), correct=0,
            lowlevel=np.zeros((N, 10), dtype=np.float32) if want_lowlevel else None,
        )
    t0, seen = time.time(), 0
    for img, y in loader:
        img = img.cuda(non_blocking=True)
        B = img.shape[0]
        ll = lowlevel_stats(img).cpu().numpy() if want_lowlevel else None
        for name, m in nets.items():
            f, logits = forward_depths(m, img)
            w = writers[name]
            fa = f.abs().amax((0, 2, 3)).cpu().numpy()
            w["maxabs"] = np.maximum(w["maxabs"], fa)
            w["sumnorm"] += f[:, :, 0].norm(dim=-1).sum(0).cpu().numpy()
            w["feats"][seen:seen + B] = f.half().cpu().numpy()
            w["labels"][seen:seen + B] = y.numpy()
            pred = logits.argmax(1).cpu()
            w["preds"][seen:seen + B] = pred.numpy()
            w["correct"] += int((pred == y).sum())
            if ll is not None:
                w["lowlevel"][seen:seen + B] = ll
        seen += B
        if seen % (batch_size * 20) == 0 or seen == N:
            print(f"[{dataset}/{split}] {seen}/{N} imgs, {seen / (time.time() - t0):.0f} img/s x {len(nets)} models", flush=True)
    assert seen == N, (seen, N)
    for name, w in writers.items():
        d = cache_dir(name, dataset, split)
        assert np.isfinite(w["maxabs"]).all() and w["maxabs"].max() < 6e4, (name, w["maxabs"])
        w["feats"].flush()
        del w["feats"]
        np.save(d / "labels.npy", w["labels"])
        np.save(d / "preds.npy", w["preds"])
        np.save(d / "index.npy", w["index"])
        if w["lowlevel"] is not None:
            np.save(d / "lowlevel.npy", w["lowlevel"])
        e = ents[name]
        info = dict(model=name, arm=e["arm"], seed=e["seed"], ckpt=str(registry.ckpt_path(e)), epoch=e.get("epoch", 299),
                    k_bias_removed=getattr(nets[name], "k_bias_removed", {}),
                    ckpt_mtime=os.path.getmtime(registry.ckpt_path(e)), dataset=dataset, split=split, N=N, n_classes=C,
                    limit=limit, subset_sha256=meta["subset_sha256"], crop_pct=meta["crop_pct"],
                    depth_maxabs=w["maxabs"].tolist(), depth_mean_cls_norm=(w["sumnorm"] / N).tolist(),
                    final_top1=100.0 * w["correct"] / N if C == 1000 else None,
                    lowlevel_targets=LOWLEVEL_TARGETS if w["lowlevel"] is not None else None,
                    git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"))
        json.dump(info, open(d / "meta.json", "w"), indent=1)
        (d / "_DONE").touch()
        print(f"[{dataset}/{split}] {name}: done, final top-1 {info['final_top1']}, max|x| {w['maxabs'].max():.0f}", flush=True)
    del nets
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="comma list or 'all' (= every cached set)")
    ap.add_argument("--models", default="all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images per split")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    print(f"models ({len(models)}): {[m['name'] for m in models]}", flush=True)
    names = pdata.CACHED if args.datasets == "all" else args.datasets.split(",")
    for dataset in names:
        splits = ("test",) if dataset in pdata.EVAL_ONLY else pdata.SPLITS
        for split in splits:
            extract(models, dataset, split, args.batch_size, args.workers, args.limit)


if __name__ == "__main__":
    main()
