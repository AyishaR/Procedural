"""Image sets for the dynamics stages.

`val` / `test` are the frozen probes ImageNet splits (imnet_val = 20/class held out of the train split, i.e.
training-distribution images the models were trained on; imnet_test = the official 50k val set).  The
diagnostic sets are seeded per-class draws from those frozen splits, frozen in turn under $PROBE_ROOT/subsets
(diag_<x>_all.json) with a sha256 like every probes subset.  One eval transform everywhere (Resize 256 bicubic,
CenterCrop 224, ImageNet mean/std = the training eval transform).
"""
import torch
from torch.utils.data import DataLoader, Subset

from probes import data as pdata

DIAG = {  # name: (source dataset, source split, images per class)
    "diag_train2k": ("imnet", "train", 2), "diag_test2k": ("imnet", "test", 2),
    "diag_train4k": ("imnet", "train", 4), "diag_test4k": ("imnet", "test", 4),
}
SPLIT_NOTE = {"val": "train-split images (20/class held out of the probe train subset)",
              "test": "official ImageNet val (50k)",
              "diag_train2k": "2/class of the probe train subset", "diag_test2k": "2/class of the official val",
              "diag_train4k": "4/class of the probe train subset", "diag_test4k": "4/class of the official val"}


def diag_items(name):
    """Frozen (path, label) list of a diagnostic set; drawn once per class from the frozen source split."""
    rec = pdata._load_subset(name, "all")
    if rec is not None:
        return rec
    src, split, k = DIAG[name]
    base = pdata.subset_items(src, split)
    (sel,) = pdata._per_class_subset([(p, y) for p, y in base["items"]], (k,), pdata._rng(name))
    return pdata._save_subset(name, "all", sel, base["n_classes"], kind="path")


def dataset(name, limit=None):
    """'val' / 'test' -> the probes imnet split, 'diag_*' -> a diagnostic set.  Returns (ds, meta)."""
    if name in DIAG:
        rec = diag_items(name)
        ds = pdata.PathDataset([(p, y) for p, y in rec["items"]], pdata.eval_transform())
        meta = dict(dataset=name, n=len(ds), n_classes=rec["n_classes"], subset_sha256=rec["sha256"], crop_pct=224 / 256)
    else:
        ds, _, meta = pdata.build_probe_dataset("imnet", name)
        meta = dict(meta, dataset=name)
    meta["note"] = SPLIT_NOTE.get(name, "")
    if limit is not None:
        n = min(int(limit), len(ds))
        ds = Subset(ds, list(range(n)))
        meta = dict(meta, n=n, limit=int(limit))
    return ds, meta


def loader(ds, batch_size, workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)


def load_resident(name, device="cuda", dtype=torch.float16, batch_size=256, workers=8, limit=None):
    """Decode a whole (small) set once -> (x [N,3,224,224] `dtype` on `device`, y [N] long on `device`, meta)."""
    ds, meta = dataset(name, limit)
    xs, ys = [], []
    for img, y in loader(ds, batch_size, workers):
        xs.append(img.to(device=device, dtype=dtype, non_blocking=True))
        ys.append(y.to(device))
    return torch.cat(xs), torch.cat(ys), meta
