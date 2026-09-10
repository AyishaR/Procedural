"""Reader for the feature cache written by extract_features.py.

Layout: $PROBE_ROOT/features/<model>/<dataset>/<split>/{feats.npy, labels.npy, index.npy, preds.npy,
[lowlevel.npy], meta.json, _DONE}.  feats.npy is fp16 [N, 13, 2, 768]: axis 1 = depth (0 = embedding,
d = output of block d-1), axis 2 = {CLS token, mean over patch tokens}.
"""
import json
import os

import numpy as np
import torch

from probes import registry

FTYPES = ("cls", "mean", "concat")
ALLOW_LIMITED = os.environ.get("PROBE_ALLOW_LIMITED") == "1"    # smoke tests only: accept --limit caches
LOWLEVEL_TARGETS = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "lum_std", "grad_mag", "laplacian_var", "saturation"]


def cache_dir(model, dataset, split):
    return registry.FEATURES / model / dataset / split


def cache_limit(model, dataset, split):
    """None for a complete cache, the --limit value for a smoke cache, False if there is no finished cache."""
    d = cache_dir(model, dataset, split)
    if not (d / "_DONE").exists() or not (d / "meta.json").exists():
        return False
    return json.load(open(d / "meta.json")).get("limit")


def is_done(model, dataset, split):
    lim = cache_limit(model, dataset, split)
    if lim is False:
        return False
    return lim is None or ALLOW_LIMITED


def result_suffix(model, dataset, split="train"):
    """'' for complete caches, '__limitN' when a smoke cache is (explicitly) consumed, so results never mix."""
    lim = cache_limit(model, dataset, split)
    return f"__limit{lim}" if lim else ""


def load(model, dataset, split):
    d = cache_dir(model, dataset, split)
    if not (d / "_DONE").exists():
        raise FileNotFoundError(f"no finished feature cache at {d}")
    lim = json.load(open(d / "meta.json")).get("limit")
    if lim is not None and not ALLOW_LIMITED:
        raise RuntimeError(f"{d} is a limited smoke cache (limit={lim}); set PROBE_ALLOW_LIMITED=1 to use it")
    out = dict(
        X=np.load(d / "feats.npy", mmap_mode="r"),
        y=np.load(d / "labels.npy"),
        index=np.load(d / "index.npy"),
        preds=np.load(d / "preds.npy"),
        meta=json.load(open(d / "meta.json")),
    )
    if (d / "lowlevel.npy").exists():
        out["lowlevel"] = np.load(d / "lowlevel.npy")
    return out


def select(X, ftype, device="cuda", chunk=8192):
    """[N,13,2,768] memmap -> fp16 torch tensor [N,13,D] on `device`, read in chunks (D = 768 or 1536)."""
    if ftype not in FTYPES:
        raise ValueError(ftype)
    N = X.shape[0]
    D = 1536 if ftype == "concat" else 768
    out = torch.empty((N, X.shape[1], D), dtype=torch.float16, device=device)
    for s in range(0, N, chunk):
        blk = torch.from_numpy(np.array(X[s:s + chunk]))   # copy: memmap is read-only
        if ftype == "cls":
            v = blk[:, :, 0]
        elif ftype == "mean":
            v = blk[:, :, 1]
        else:
            v = torch.cat([blk[:, :, 0], blk[:, :, 1]], dim=-1)
        out[s:s + chunk] = v.to(device, non_blocking=True)
    return out


def standardise_stats(Xtr, eps=1e-6):
    """Per-depth, per-dim mean/std from a [N,13,D] fp16 tensor, computed in fp32 chunks."""
    N = Xtr.shape[0]
    s1 = torch.zeros(Xtr.shape[1:], dtype=torch.float64, device=Xtr.device)
    s2 = torch.zeros_like(s1)
    for i in range(0, N, 8192):
        b = Xtr[i:i + 8192].double()
        s1 += b.sum(0)
        s2 += (b * b).sum(0)
    mu = s1 / N
    sd = (s2 / N - mu * mu).clamp_min(0).sqrt().clamp_min(eps)
    return mu.float(), sd.float()
