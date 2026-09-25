"""OOD test sets for the zero-shot effective-robustness pass (ImageNet-1k label space, test sets only).

ImageNet-V2 (matched-frequency), ImageNet-R and ImageNet-A (each a 200-class logit mask, wnid mapping from
probes/data.py), ImageNet-Sketch (all 1000 classes), ObjectNet (113 classes -> a set of ImageNet classes each,
logits masked to their union and the winner mapped back to the ObjectNet class; 2-px red border cropped) and ImageNet-C
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
SKETCH_ROOT = registry.DATA / "sketch"                                   # downloaded, 1000 wnid dirs
A_ROOT = registry.DATA / "imagenet-a"                                    # downloaded, 200 wnid dirs
ON_ROOTS = [registry.DATASETS_ROOT / "ObjectNet" / "objectnet-1.0" / "images",   # partial local copy (91 overlap dirs)
            registry.DATA / "objectnet" / "images"]                      # the remaining 22, extracted from the zip
ON_MAPPINGS = registry.DATASETS_ROOT / "ObjectNet" / "objectnet-1.0" / "mappings"
ON_BORDER = 2                   # ObjectNet ships a 2-px red border; the protocol removes it before the transform
SETS = ["imnet_val", "imagenet_v2", "imagenet_r", "imagenet_sketch", "imagenet_a", "objectnet", "imagenet_c"]


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
    """The seeded per-class draw of ImageNet-val file names (same 5 per class in every cell), frozen once.

    The per-cell subsets below are derived from this draw but cached independently: changing the draw would
    need every `imagenet_c_<corruption>_<sev>.json` deleted as well, or the cells would keep the old files.
    """
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


def _wnid_folder_items(root, n_dirs, exts=(".jpg", ".jpeg", ".png", ".JPEG")):
    """<root>/<wnid>/*.img -> [(path, ImageNet-1k index)] using the sorted-wnid order of probes/data.py."""
    wn = {w: i for i, w in enumerate(pdata._imnet_classes())}
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name in wn)
    assert len(dirs) == n_dirs, (root, len(dirs), n_dirs)
    items = []
    for d in dirs:
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("._"))
        items += [(str(f), wn[d.name]) for f in files]
    return items


def _class_mask(indices):
    """Boolean [1000] mask over a set of ImageNet-1k indices."""
    m = np.zeros(1000, dtype=bool)
    m[np.asarray(sorted(set(indices)), dtype=int)] = True
    return m


def objectnet_mapping():
    """-> (folders [113], in2on int[1000] with -1 outside the overlap, mask bool[1000] over the union).

    folder -> ObjectNet label -> 'name; name; ...' of ImageNet classes -> rows of imagenet_to_label_2012_v2 ->
    PyTorch (sorted-wnid) indices via pytorch_to_imagenet_2012_id.json, which is 0-based into that file.
    """
    import json
    labels = [l.strip() for l in open(ON_MAPPINGS / "imagenet_to_label_2012_v2")]
    p2i = json.load(open(ON_MAPPINGS / "pytorch_to_imagenet_2012_id.json"))
    assert len(labels) == 1000 and len(p2i) == 1000
    name2idx = {}
    for k, v in p2i.items():
        name2idx.setdefault(labels[v].strip(), []).append(int(k))
    o2i = json.load(open(ON_MAPPINGS / "objectnet_to_imagenet_1k.json"))
    f2o = json.load(open(ON_MAPPINGS / "folder_to_objectnet_label.json"))
    folders = sorted(f for f, lab in f2o.items() if lab in o2i)
    assert len(folders) == 113, len(folders)
    in2on = np.full(1000, -1, dtype=np.int64)
    for c, f in enumerate(folders):
        for n in (x.strip() for x in o2i[f2o[f]].split(";")):
            assert n in name2idx, (f, n)
            for i in name2idx[n]:
                assert in2on[i] == -1, (f, n, i, "ImageNet class claimed by two ObjectNet classes")
                in2on[i] = c
    return folders, in2on, _class_mask(np.flatnonzero(in2on >= 0))


def _objectnet_items():
    """[(path, ObjectNet class 0..112)] over both roots; every overlap folder must exist in exactly one."""
    folders, _, _ = objectnet_mapping()
    items = []
    for c, f in enumerate(folders):
        found = [r / f for r in ON_ROOTS if (r / f).is_dir()]
        assert len(found) == 1, (f, [str(x) for x in found])
        files = sorted(x for x in found[0].iterdir() if x.suffix.lower() == ".png" and not x.name.startswith("._"))
        assert files, found[0]
        items += [(str(x), c) for x in files]
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
    if name == "imagenet_sketch":
        rec = pdata._load_subset("imagenet_sketch", "test")
        return rec if rec is not None else pdata._save_subset("imagenet_sketch", "test", _wnid_folder_items(SKETCH_ROOT, 1000), 1000)
    if name == "imagenet_a":
        rec = pdata._load_subset("imagenet_a", "test")
        return rec if rec is not None else pdata._save_subset("imagenet_a", "test", _wnid_folder_items(A_ROOT, 200), 1000)
    if name == "objectnet":
        rec = pdata._load_subset("objectnet", "test")
        return rec if rec is not None else pdata._save_subset("objectnet", "test", _objectnet_items(), 113)
    if name.startswith("imagenet_c/"):
        _, corruption, sev = name.split("/")
        assert corruption in CORRUPTIONS and int(sev) in SEVERITIES, name
        tag = f"{corruption}_{sev}"
        rec = pdata._load_subset("imagenet_c", tag)
        return rec if rec is not None else pdata._save_subset("imagenet_c", tag, _imagenet_c_items(corruption, int(sev)), 1000)
    raise KeyError(name)


def eval_spec(name):
    """How a set is scored: (mask bool[1000] or None, pred_map int[1000] or None).

    mask  -> argmax is taken over those ImageNet classes only (label metadata, not tuning).
    pred_map -> the winning ImageNet class is mapped to the set's own label space before comparison (ObjectNet).
    """
    if name == "imagenet_r":
        return pdata.imagenet_r_class_mask(), None
    if name == "imagenet_a":
        rec = subset_items("imagenet_a")
        m = _class_mask([y for _, y in rec["items"]])
        assert m.sum() == 200, m.sum()
        return m, None
    if name == "objectnet":
        _, in2on, mask = objectnet_mapping()
        return mask, in2on
    return None, None


def build_ood_dataset(name):
    """-> (Dataset yielding (img[3,224,224], label), meta).  Labels are ImageNet-1k indices except ObjectNet (0..112)."""
    rec = subset_items(name)
    tf = imagenet_c_transform() if name.startswith("imagenet_c/") else pdata.eval_transform()
    pre = None
    if name == "objectnet":     # remove the red border before Resize 256 / CenterCrop 224
        pre = lambda im: im.crop((ON_BORDER, ON_BORDER, im.width - ON_BORDER, im.height - ON_BORDER))
    ds = pdata.PathDataset([(p, y) for p, y in rec["items"]], tf, pre=pre)
    meta = dict(dataset=name, n=len(ds), subset_sha256=rec["sha256"], seed=pdata.SEED,
                transform=("centercrop224" if name.startswith("imagenet_c/") else
                           "crop_border2_resize256_centercrop224" if name == "objectnet" else "resize256_centercrop224"))
    return ds, meta


def imagenet_c_clean_index():
    """Row indices of the ImageNet-C draw inside the 50k imnet_val list (same files, clean) -> the paired clean reference."""
    val = subset_items("imnet_val")["items"]
    pos = {os.path.basename(p): i for i, (p, _) in enumerate(val)}
    idx = np.array([pos[f] for f, _ in imagenet_c_file_draw()["items"]], dtype=np.int64)
    assert len(idx) == C_PER_CLASS * 1000 and len(np.unique(idx)) == len(idx)
    return idx


def all_set_names(extra_only=False):
    """Every named set; extra_only drops the stage-1 sets (val / V2 / R / C) so a rerun only adds the new ones."""
    new = ["imagenet_sketch", "imagenet_a", "objectnet"]
    if extra_only:
        return new
    return ["imnet_val", "imagenet_v2", "imagenet_r"] + new + [f"imagenet_c/{c}/{s}" for c in CORRUPTIONS for s in SEVERITIES]


if __name__ == "__main__":
    for name in all_set_names()[:4] + ["imagenet_c/zoom_blur/5"]:
        ds, meta = build_ood_dataset(name)
        x, y = ds[0]
        print(f"{name:30s} n={len(ds):6d} x={tuple(x.shape)} y0={y} sha={meta['subset_sha256'][:8]} tf={meta['transform']}", flush=True)
    print("imagenet_c clean index:", imagenet_c_clean_index()[:5], "mask200:", int(pdata.imagenet_r_class_mask().sum()))
