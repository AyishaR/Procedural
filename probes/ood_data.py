"""OOD test sets for the zero-shot effective-robustness pass (ImageNet-1k label space, test sets only).

ImageNet-V2 (matched-frequency), ImageNet-R (200-class logit mask, wnid mapping from probes/data.py) and ImageNet-C
(the 15 extracted corruptions x 5 severities; one seeded 5-per-class draw of val file names shared by every cell, so
cells are paired and the same 5k clean images give the reference).  V2 / R / ImageNet val use the repo's eval
transform (Resize 256 / CenterCrop 224); the ImageNet-C images are already 224x224 (Hendrycks' protocol:
CenterCrop 224 only, no resize).  Every list is frozen as a (path, label) subset under $PROBE_ROOT/subsets.
"""
import os

import numpy as np
from torchvision import transforms

from probes import data as pdata
from probes import registry

V2_ROOT = registry.DATASETS_ROOT / "ILSVRC2012" / "imagenetv2-matched-frequency-format-val"
C_ROOT = registry.DATASETS_ROOT / "ILSVRC2012" / "Corrupted"
C_GROUPS = {                    # Hendrycks & Dietterich (2019) groups; all 15 are extracted on disk
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
CORRUPTIONS = [c for g in C_GROUPS.values() for c in g]
SEVERITIES = [1, 2, 3, 4, 5]
C_PER_CLASS = 5                 # 5 x 1000 = 5k images per cell, 375k over the 75 cells
SETS = ["imnet_val", "imagenet_v2", "imagenet_r", "imagenet_c"]


def imagenet_c_transform():
    """ImageNet-C images are the 224x224 centre crops already; only normalise (CenterCrop is a no-op guard)."""
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    return transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])


def _imagenet_v2_items():
    dirs = sorted((d for d in V2_ROOT.iterdir() if d.is_dir()), key=lambda d: int(d.name))
    assert len(dirs) == 1000 and [int(d.name) for d in dirs] == list(range(1000)), len(dirs)
    items = []
    for d in dirs:
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpeg", ".jpg", ".png"))
        items += [(str(f), int(d.name)) for f in files]
    assert len(items) == 10000, len(items)
    return items


def imagenet_c_file_draw():
    """The seeded per-class draw of ImageNet-val file names (same 5 per class in every cell), frozen once."""
    rec = pdata._load_subset("imagenet_c", "files")
    if rec is not None:
        return rec
    wnids = pdata._imnet_classes()
    ref = C_ROOT / CORRUPTIONS[0] / "1"
    rng = pdata._rng("imagenet_c", "files")
    items = []
    for y, w in enumerate(wnids):
        files = sorted(f.name for f in (ref / w).iterdir() if f.suffix.upper() == ".JPEG")
        assert len(files) == 50, (w, len(files))
        take = sorted(rng.permutation(len(files))[:C_PER_CLASS])
        items += [(files[j], y) for j in take]
    return pdata._save_subset("imagenet_c", "files", items, 1000, kind="filename")


def _imagenet_c_items(corruption, severity):
    """Paths of the frozen draw inside one (corruption, severity) cell; every file must exist."""
    wnids = pdata._imnet_classes()
    cell = C_ROOT / corruption / str(severity)
    items = [(str(cell / wnids[y] / f), y) for f, y in imagenet_c_file_draw()["items"]]
    for p, _ in (items[0], items[len(items) // 2], items[-1]):     # a full stat pass costs ~40 s/cell on NFS
        assert os.path.exists(p), (corruption, severity, p)
    return items


def subset_items(name):
    """Frozen (path, label) record of an OOD set; 'imagenet_c/<corruption>/<sev>' names one cell."""
    if name == "imnet_val":
        return pdata.subset_items("imnet", "test")                    # official 50k val, sorted by wnid then file
    if name == "imagenet_r":
        return pdata.subset_items("imagenet_r", "test")
    if name == "imagenet_v2":
        rec = pdata._load_subset("imagenet_v2", "test")
        return rec if rec is not None else pdata._save_subset("imagenet_v2", "test", _imagenet_v2_items(), 1000)
    if name.startswith("imagenet_c/"):
        _, corruption, sev = name.split("/")
        assert corruption in CORRUPTIONS and int(sev) in SEVERITIES, name
        tag = f"{corruption}_{sev}"
        rec = pdata._load_subset("imagenet_c", tag)
        return rec if rec is not None else pdata._save_subset("imagenet_c", tag, _imagenet_c_items(corruption, int(sev)), 1000)
    raise KeyError(name)


def build_ood_dataset(name):
    """-> (Dataset yielding (img[3,224,224], label in 0..999), meta)."""
    rec = subset_items(name)
    tf = imagenet_c_transform() if name.startswith("imagenet_c/") else pdata.eval_transform()
    ds = pdata.PathDataset([(p, y) for p, y in rec["items"]], tf)
    meta = dict(dataset=name, n=len(ds), subset_sha256=rec["sha256"], seed=pdata.SEED,
                transform="centercrop224" if name.startswith("imagenet_c/") else "resize256_centercrop224")
    return ds, meta


def imagenet_c_clean_index():
    """Row indices of the ImageNet-C draw inside the 50k imnet_val list (same files, clean) -> the paired clean reference."""
    val = subset_items("imnet_val")["items"]
    pos = {os.path.basename(p): i for i, (p, _) in enumerate(val)}
    idx = np.array([pos[f] for f, _ in imagenet_c_file_draw()["items"]], dtype=np.int64)
    assert len(idx) == C_PER_CLASS * 1000 and len(np.unique(idx)) == len(idx)
    return idx


def all_set_names():
    return ["imnet_val", "imagenet_v2", "imagenet_r"] + [f"imagenet_c/{c}/{s}" for c in CORRUPTIONS for s in SEVERITIES]


if __name__ == "__main__":
    for name in all_set_names()[:4] + ["imagenet_c/zoom_blur/5"]:
        ds, meta = build_ood_dataset(name)
        x, y = ds[0]
        print(f"{name:30s} n={len(ds):6d} x={tuple(x.shape)} y0={y} sha={meta['subset_sha256'][:8]} tf={meta['transform']}", flush=True)
    print("imagenet_c clean index:", imagenet_c_clean_index()[:5], "mask200:", int(pdata.imagenet_r_class_mask().sum()))
