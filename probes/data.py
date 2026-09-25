"""Datasets for the probe pipeline.

Cached classification sets (one fixed eval transform, seeded per-class subsets shared by every model), the
ImageNet-R evaluation set (labels in ImageNet-1k index space), and the dense sets (VOC, ADE20K, NYUv2) with
paired image/label transforms and per-patch (14x14) targets.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True   # a handful of ObjectNet PNGs are truncated; decode what is there
from torch.utils.data import Dataset

from probes import registry

sys.path.insert(0, str(registry.ROOT))

CLASSIFICATION = ["imnet", "places365", "cifar100", "dtd", "flowers", "cars", "aircraft", "eurosat"]
EVAL_ONLY = ["imagenet_r"]
DENSE = ["voc", "ade20k", "nyu"]
CACHED = CLASSIFICATION + EVAL_ONLY
SPLITS = ("train", "val", "test")
GRID = 14
PATCH = 16
IMG = 224
SEED = 0
IMG_EXT = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG")

# per-class counts for the seeded subsets of the big sets: (train, val)
SUBSET_PER_CLASS = {"imnet": (100, 20), "places365": (100, 20)}
SPLIT_FRACTIONS = {"cars": (0.85, 0.15), "eurosat": (0.6, 0.2, 0.2)}


# ----------------------------------------------------------------------------------------------- transforms
def eval_transform(crop_pct=None):
    from datasets import build_transform
    args = registry.build_args()
    if crop_pct is not None:
        args.crop_pct = crop_pct
    return build_transform(False, args)


def _rng(dataset, split=""):
    h = int(hashlib.sha256(f"{dataset}:{split}:{SEED}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


# ----------------------------------------------------------------------------------------------- base datasets
class PathDataset(Dataset):
    """items = [(path, label)], returns (tensor, label).  `pre` is an optional PIL->PIL hook (aircraft banner crop)."""

    def __init__(self, items, transform, pre=None):
        self.items = list(items)
        self.transform = transform
        self.pre = pre

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.pre is not None:
            img = self.pre(img)
        return self.transform(img), int(y)


class IndexedDataset(Dataset):
    """Subset of a torchvision dataset object (PIL images), with our own transform."""

    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return self.transform(img.convert("RGB")), int(y)


def _list_folder(root, class_key=None):
    """ImageFolder-style listing -> (items [(path, label)], classes).  class_key orders class dirs (default: name)."""
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")], key=class_key)
    cidx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        files = sorted(f for f in (root / c).iterdir() if f.suffix in IMG_EXT and not f.name.startswith("._"))
        items += [(str(f), cidx[c]) for f in files]
    return items, classes


def _per_class_subset(items, n_per_class, rng):
    """Seeded per-class draw over sorted items -> list of (train_items, val_items, ...) by the tuple n_per_class."""
    by_c = {}
    for p, y in items:
        by_c.setdefault(y, []).append((p, y))
    outs = [[] for _ in n_per_class]
    for y in sorted(by_c):
        lst = by_c[y]
        perm = rng.permutation(len(lst))
        s = 0
        for k, n in enumerate(n_per_class):
            take = perm[s:s + n] if isinstance(n, int) else perm[s:s + int(round(n * len(lst)))]
            outs[k] += [lst[j] for j in sorted(take)]
            s += len(take)
    return outs


# ----------------------------------------------------------------------------------------------- subset registry
def _subset_file(dataset, split):
    return registry.SUBSETS / f"{dataset}_{split}.json"


def _save_subset(dataset, split, items, n_classes, kind="path"):
    registry.SUBSETS.mkdir(parents=True, exist_ok=True)
    def ser(p):                      # path -> str, torchvision index -> int, dense (image, label) pair -> list
        if isinstance(p, (list, tuple)):
            return [str(q) for q in p]
        return int(p) if isinstance(p, (int, np.integer)) else str(p)
    rec = dict(dataset=dataset, split=split, seed=SEED, kind=kind, n=len(items), n_classes=n_classes,
               items=[[ser(p), int(y)] for p, y in items])
    rec["sha256"] = hashlib.sha256(json.dumps(rec["items"]).encode()).hexdigest()
    tmp = _subset_file(dataset, split).with_suffix(f".{os.getpid()}.tmp")   # unique per writer: concurrent jobs may race
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, _subset_file(dataset, split))
    return rec


def _load_subset(dataset, split):
    f = _subset_file(dataset, split)
    return json.load(open(f)) if f.exists() else None


# ----------------------------------------------------------------------------------------------- per-dataset builders
def _imnet_classes():
    return sorted(d.name for d in (registry.IMNET / "train").iterdir() if d.is_dir())


def _build_items(dataset):
    """Returns {split: (items, kind)} and n_classes for every cached classification set; deterministic."""
    D = registry.DATASETS_ROOT
    if dataset == "imnet":
        tr, _ = _list_folder(registry.IMNET / "train")
        te, _ = _list_folder(registry.IMNET / "val")
        a, b = _per_class_subset(tr, SUBSET_PER_CLASS["imnet"], _rng(dataset))
        return {"train": a, "val": b, "test": te}, 1000
    if dataset == "places365":
        tr, cl = _list_folder(D / "places365" / "train", class_key=int)
        te, cl2 = _list_folder(D / "places365" / "val", class_key=int)
        assert cl == cl2 and len(cl) == 365, (len(cl), len(cl2))
        a, b = _per_class_subset(tr, SUBSET_PER_CLASS["places365"], _rng(dataset))
        return {"train": a, "val": b, "test": te}, 365
    if dataset == "flowers":
        out = {}
        for s, sub in (("train", "train"), ("val", "val"), ("test", "test")):
            it, cl = _list_folder(D / "flowers-102" / sub, class_key=int)
            assert len(cl) == 102, len(cl)
            out[s] = it
        return out, 102
    if dataset == "cars":
        tr, cl = _list_folder(D / "stanford-cars" / "train", class_key=int)
        te, cl2 = _list_folder(D / "stanford-cars" / "val", class_key=int)
        assert cl == cl2 and len(cl) == 196
        a, b = _per_class_subset(tr, SPLIT_FRACTIONS["cars"], _rng(dataset))
        return {"train": a, "val": b, "test": te}, 196
    if dataset == "aircraft":
        base = D / "fgvc-aircraft-2013b" / "data"
        variants = [l.strip() for l in open(base / "variants.txt") if l.strip()]
        cidx = {v: i for i, v in enumerate(variants)}
        assert len(variants) == 100
        out = {}
        for s in SPLITS:
            items = []
            for line in open(base / f"images_variant_{s}.txt"):
                line = line.strip()
                if not line:
                    continue
                iid, var = line.split(" ", 1)
                items.append((str(base / "images" / f"{iid}.jpg"), cidx[var]))
            out[s] = items
        return out, 100
    if dataset == "eurosat":
        root = registry.DATA / "eurosat" / "eurosat" / "2750"
        it, cl = _list_folder(root)
        assert len(cl) == 10, cl
        a, b, c = _per_class_subset(it, SPLIT_FRACTIONS["eurosat"], _rng(dataset))
        return {"train": a, "val": b, "test": c}, 10
    if dataset == "imagenet_r":
        root = D / "ImageNet" / "imagenet-r"
        wn = {w: i for i, w in enumerate(_imnet_classes())}
        items = []
        dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        for d in dirs:
            wnid = d.name.split("_")[0]
            assert wnid in wn, d.name
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("._"))
            items += [(str(f), wn[wnid]) for f in files]
        assert len(dirs) == 200, len(dirs)
        return {"test": items}, 1000
    if dataset in ("cifar100", "dtd"):
        raise ValueError("torchvision-object datasets are built in build_probe_dataset")
    raise KeyError(dataset)


def _torchvision_base(dataset, split):
    """(base torchvision dataset with PIL output, n_classes) for cifar100 / dtd."""
    import torchvision.datasets as tvd
    if dataset == "cifar100":
        base = tvd.CIFAR100(str(registry.DATA / "cifar100"), train=(split != "test"), download=False)
        return base, 100
    if dataset == "dtd":
        base = tvd.DTD(str(registry.DATA / "dtd"), split=split, partition=1, download=False)
        return base, 47
    raise KeyError(dataset)


def subset_items(dataset, split):
    """The frozen (path|index, label) list of a split; built once with the seed, then always re-read from disk."""
    rec = _load_subset(dataset, split)
    if rec is not None:
        return rec
    if dataset == "cifar100":
        base, C = _torchvision_base(dataset, "train")
        if split == "test":
            baset, _ = _torchvision_base(dataset, "test")
            items = [(i, int(baset.targets[i])) for i in range(len(baset))]
        else:
            labels = np.asarray(base.targets)
            items_all = [(i, int(labels[i])) for i in range(len(labels))]
            a, b = _per_class_subset(items_all, (450, 50), _rng(dataset))
            items = a if split == "train" else b
        return _save_subset(dataset, split, items, C, kind="index")
    if dataset == "dtd":
        base, C = _torchvision_base(dataset, split)
        items = [(i, int(base._labels[i])) for i in range(len(base))]
        return _save_subset(dataset, split, items, C, kind="index")
    per_split, C = _build_items(dataset)
    for s, it in per_split.items():          # write every split of this dataset at once (one seeded draw)
        if _load_subset(dataset, s) is None:
            _save_subset(dataset, s, it, C, kind="path")
    return _load_subset(dataset, split)


def build_probe_dataset(dataset, split):
    """-> (torch Dataset yielding (img[3,224,224], label), n_classes, meta dict)."""
    rec = subset_items(dataset, split)
    crop = 1.0 if dataset in ("cifar100", "eurosat") else None
    tf = eval_transform(crop_pct=crop)
    items = [(p, y) for p, y in rec["items"]]
    if rec["kind"] == "index":
        base, _ = _torchvision_base(dataset, "test" if (dataset == "cifar100" and split == "test") else ("train" if dataset == "cifar100" else split))
        ds = IndexedDataset(base, [int(p) for p, _ in items], tf)
    else:
        pre = (lambda im: im.crop((0, 0, im.width, im.height - 20))) if dataset == "aircraft" else None
        ds = PathDataset(items, tf, pre=pre)
    meta = dict(dataset=dataset, split=split, n=len(ds), n_classes=rec["n_classes"], subset_sha256=rec["sha256"],
                seed=SEED, crop_pct=crop if crop is not None else 224 / 256)
    return ds, rec["n_classes"], meta


def imagenet_r_class_mask():
    """Boolean [1000] mask of the 200 ImageNet-1k classes present in ImageNet-R."""
    rec = subset_items("imagenet_r", "test")
    m = np.zeros(1000, dtype=bool)
    m[np.unique([y for _, y in rec["items"]])] = True
    assert m.sum() == 200, m.sum()
    return m


# ----------------------------------------------------------------------------------------------- dense datasets
def patch_majority(mask, n_classes, ignore=255):
    """[H,W] long label map -> [14,14] long per-patch majority label, -1 where >50% of the cell is `ignore`."""
    m = mask.view(GRID, PATCH, GRID, PATCH).permute(0, 2, 1, 3).reshape(GRID * GRID, PATCH * PATCH)
    m = torch.where(m == ignore, torch.full_like(m, n_classes), m)
    counts = torch.zeros(GRID * GRID, n_classes + 1, dtype=torch.long)
    counts.scatter_add_(1, m, torch.ones_like(m))
    out = counts[:, :n_classes].argmax(1)
    out[counts[:, n_classes] * 2 > PATCH * PATCH] = -1
    return out.view(GRID, GRID)


def patch_mean_logdepth(depth):
    """[H,W] float metres (<=0 invalid) -> [14,14] mean log-depth per cell, NaN where <50% valid."""
    d = depth.view(GRID, PATCH, GRID, PATCH).permute(0, 2, 1, 3).reshape(GRID * GRID, PATCH * PATCH)
    valid = d > 0
    ld = torch.where(valid, d.clamp_min(1e-3).log(), torch.zeros_like(d))
    n = valid.sum(1)
    out = ld.sum(1) / n.clamp_min(1)
    out[n * 2 < PATCH * PATCH] = float("nan")
    return out.view(GRID, GRID)


class DenseTransform:
    """Paired transform: Resize(256 short side) -> RandomCrop/CenterCrop 224 (+flip) -> tensors."""

    def __init__(self, train):
        import torchvision.transforms.functional as F
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.F, self.train = F, train
        self.mean, self.std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    def __call__(self, img, label, label_mode):
        F = self.F
        img = F.resize(img, 256, interpolation=F.InterpolationMode.BICUBIC)
        label = F.resize(label, 256, interpolation=F.InterpolationMode.NEAREST)
        W, H = img.size
        if self.train:
            i = int(torch.randint(0, H - IMG + 1, (1,)))
            j = int(torch.randint(0, W - IMG + 1, (1,)))
        else:
            i, j = (H - IMG) // 2, (W - IMG) // 2
        img = F.crop(img, i, j, IMG, IMG)
        label = F.crop(label, i, j, IMG, IMG)
        if self.train and torch.rand(1).item() < 0.5:
            img, label = F.hflip(img), F.hflip(label)
        x = F.normalize(F.to_tensor(img), self.mean, self.std)
        if label_mode == "seg":
            y = torch.from_numpy(np.array(label, dtype=np.int64))
        else:
            y = torch.from_numpy(np.array(label, dtype=np.float32))
        return x, y


class SegDataset(Dataset):
    def __init__(self, items, n_classes, train, remap=None):
        self.items, self.C, self.tf, self.remap = list(items), n_classes, DenseTransform(train), remap

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, lp = self.items[i]
        img = Image.open(ip).convert("RGB")
        lab = Image.open(lp)
        if lab.mode != "L" and lab.mode != "P":
            lab = lab.convert("L")
        x, y = self.tf(img, lab, "seg")
        if self.remap is not None:
            y = self.remap(y)
        return x, patch_majority(y, self.C)


class NYUDataset(Dataset):
    """NYU Depth v2 labeled subset read lazily from the HDF5 .mat (images [N,3,640,480] -> HWC 480x640)."""

    def __init__(self, path, indices, train):
        import h5py
        self.path, self.indices, self.tf = str(path), list(indices), DenseTransform(train)
        self._f = None
        with h5py.File(self.path, "r") as f:
            assert f["images"].shape[1:] == (3, 640, 480), f["images"].shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import h5py
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        k = self.indices[i]
        img = np.asarray(self._f["images"][k]).transpose(2, 1, 0)      # -> H=480, W=640, 3
        dep = np.asarray(self._f["depths"][k]).T                       # -> H=480, W=640
        x, y = self.tf(Image.fromarray(img), Image.fromarray(dep.astype(np.float32), mode="F"), "depth")
        return x, patch_mean_logdepth(y)


def _dense_items(dataset):
    """{split: [(image_path, label_path)]} for voc / ade20k, or {split: [indices]} for nyu; seeded once."""
    rec = _load_subset(dataset, "train")
    if rec is not None:
        return {s: [tuple(x) if isinstance(x, list) else int(x) for x, _ in _load_subset(dataset, s)["items"]] for s in SPLITS}
    rng = _rng(dataset)
    if dataset == "voc":
        root = registry.DATA / "voc" / "VOCdevkit" / "VOC2012"
        def lst(name):
            ids = [l.strip() for l in open(root / "ImageSets" / "Segmentation" / f"{name}.txt") if l.strip()]
            return [(str(root / "JPEGImages" / f"{i}.jpg"), str(root / "SegmentationClass" / f"{i}.png")) for i in ids]
        tr_all, te = lst("train"), lst("val")
        perm = rng.permutation(len(tr_all))
        n_val = int(round(0.1 * len(tr_all)))
        splits = {"train": [tr_all[j] for j in sorted(perm[n_val:])], "val": [tr_all[j] for j in sorted(perm[:n_val])], "test": te}
        C = 21
    elif dataset == "ade20k":
        root = registry.DATA / "ade20k" / "ADEChallengeData2016"
        def lst(sub):
            imgs = sorted((root / "images" / sub).glob("*.jpg"))
            return [(str(p), str(root / "annotations" / sub / (p.stem + ".png"))) for p in imgs]
        tr_all, te = lst("training"), lst("validation")
        perm = rng.permutation(len(tr_all))[:4000]
        chosen = [tr_all[j] for j in perm]
        splits = {"train": sorted(chosen[400:]), "val": sorted(chosen[:400]), "test": te}
        C = 150
    elif dataset == "nyu":
        root = registry.DATA / "nyu"
        sp = root / "splits.mat"
        if sp.exists():
            from scipy.io import loadmat
            m = loadmat(sp)
            tr_all = [int(v) - 1 for v in np.asarray(m["trainNdxs"]).ravel()]
            te = [int(v) - 1 for v in np.asarray(m["testNdxs"]).ravel()]
        else:
            perm = rng.permutation(1449)
            tr_all, te = sorted(perm[:795].tolist()), sorted(perm[795:].tolist())
        perm = rng.permutation(len(tr_all))
        n_val = int(round(0.1 * len(tr_all)))
        splits = {"train": sorted(tr_all[j] for j in perm[n_val:]), "val": sorted(tr_all[j] for j in perm[:n_val]), "test": sorted(te)}
        C = 1
    else:
        raise KeyError(dataset)
    for s, it in splits.items():
        _save_subset(dataset, s, [(x, 0) for x in it], C, kind="dense")
    return splits


def build_dense_dataset(dataset, split):
    """-> (Dataset yielding (img, patch_target[14,14]), n_classes (1 for depth), meta)."""
    items = _dense_items(dataset)[split]
    train = split == "train"
    if dataset == "voc":
        ds, C = SegDataset(items, 21, train), 21
    elif dataset == "ade20k":
        # ADE labels: 0 = other/ignore, 1..150 -> 0..149
        remap = lambda y: torch.where(y == 0, torch.full_like(y, 255), y - 1)
        ds, C = SegDataset(items, 150, train, remap=remap), 150
    elif dataset == "nyu":
        ds, C = NYUDataset(registry.DATA / "nyu" / "nyu_depth_v2_labeled.mat", items, train), 1
    else:
        raise KeyError(dataset)
    meta = dict(dataset=dataset, split=split, n=len(ds), n_classes=C, seed=SEED, grid=GRID)
    return ds, C, meta


if __name__ == "__main__":
    for name in sys.argv[1:] or CACHED:
        for split in SPLITS:
            if name in EVAL_ONLY and split != "test":
                continue
            ds, C, meta = build_probe_dataset(name, split)
            x, y = ds[0]
            print(f"{name:11s} {split:5s} n={len(ds):6d} C={C:4d} x={tuple(x.shape)} y0={y} sha={meta['subset_sha256'][:8]}", flush=True)
