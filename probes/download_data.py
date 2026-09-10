"""Download the datasets that are not on /data/datasets into $PROBE_ROOT/data (idempotent).

Archives are fetched with curl (20-40 MB/s here; torchvision's own downloader crawls at ~130 kB/s through the
proxy) into the exact locations torchvision expects, then torchvision verifies the md5 and extracts.
CIFAR-100, DTD, EuroSAT (zenodo mirror; the dfki host times out), Pascal VOC 2012 segmentation,
ADE20K (ADEChallengeData2016.zip), NYU Depth v2 labeled (.mat, HDF5) + the official train/test splits.
"""
import os
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

PROBE_ROOT = Path(os.environ.get("PROBE_ROOT", "/work/dlc2workfs3/schrodi-procedural/kempfe/probes"))
DATA = PROBE_ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
os.umask(0o002)


def fetch(url, dst, min_bytes=1000):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"  have {dst.name} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
        return dst
    tmp = dst.with_name(dst.name + ".part")
    print(f"  curl {url}", flush=True)
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-C", "-", "-o", str(tmp), url], check=True)
    assert tmp.stat().st_size >= min_bytes, f"{tmp} too small"
    tmp.rename(dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
    return dst


def done(name):
    return (DATA / name / "_DONE").exists()


def mark(name):
    (DATA / name / "_DONE").touch()


def cifar100():
    from torchvision.datasets import CIFAR100
    root = DATA / "cifar100"
    fetch("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", root / "cifar-100-python.tar.gz", 100 << 20)
    CIFAR100(str(root), train=True, download=True)
    CIFAR100(str(root), train=False, download=True)


def dtd():
    from torchvision.datasets import DTD
    root = DATA / "dtd"
    fetch("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", root / "dtd" / "dtd-r1.0.1.tar.gz", 500 << 20)
    for split in ("train", "val", "test"):
        DTD(str(root), split=split, partition=1, download=True)


def eurosat():
    root = DATA / "eurosat"
    tgt = root / "eurosat" / "2750"
    if not tgt.exists():
        z = fetch("https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1", root / "EuroSAT_RGB.zip", 50 << 20)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        src = root / "EuroSAT_RGB"
        assert src.exists(), list(root.iterdir())
        tgt.parent.mkdir(parents=True, exist_ok=True)
        src.rename(tgt)
    classes = sorted(d.name for d in tgt.iterdir() if d.is_dir())
    n = sum(len(list((tgt / c).glob("*.jpg"))) for c in classes)
    print(f"  eurosat classes {len(classes)} images {n}", flush=True)
    assert len(classes) == 10 and n == 27000, (classes, n)


def voc():
    from torchvision.datasets import VOCSegmentation
    root = DATA / "voc"
    fetch("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", root / "VOCtrainval_11-May-2012.tar", 1 << 30)
    VOCSegmentation(str(root), year="2012", image_set="train", download=True)
    VOCSegmentation(str(root), year="2012", image_set="val", download=False)


def ade20k():
    root = DATA / "ade20k"
    z = fetch("http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip", root / "ADEChallengeData2016.zip", 500 << 20)
    if not (root / "ADEChallengeData2016" / "images" / "validation").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    n_tr = len(list((root / "ADEChallengeData2016" / "images" / "training").glob("*.jpg")))
    n_va = len(list((root / "ADEChallengeData2016" / "images" / "validation").glob("*.jpg")))
    print(f"  ade20k images: train {n_tr} val {n_va}", flush=True)
    assert n_tr == 20210 and n_va == 2000, (n_tr, n_va)


def nyu():
    root = DATA / "nyu"
    fetch("http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", root / "nyu_depth_v2_labeled.mat", 2 << 30)
    try:
        fetch("http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat", root / "splits.mat", 1000)
    except Exception as e:
        print(f"  splits.mat failed ({e!r}); a seeded split will be used", flush=True)
    import h5py
    with h5py.File(root / "nyu_depth_v2_labeled.mat", "r") as f:
        print("  nyu keys:", list(f.keys())[:8], "images", f["images"].shape, "depths", f["depths"].shape, flush=True)


JOBS = {"cifar100": cifar100, "dtd": dtd, "eurosat": eurosat, "voc": voc, "ade20k": ade20k, "nyu": nyu}

if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS)
    status = {}
    for n in names:
        if done(n):
            print(f"[{n}] already done", flush=True)
            status[n] = "done"
            continue
        print(f"[{n}] downloading ...", flush=True)
        try:
            JOBS[n]()
            mark(n)
            status[n] = "ok"
        except Exception:
            traceback.print_exc()
            status[n] = "FAILED"
        print(f"[{n}] {status[n]}", flush=True)
    print("SUMMARY", status, flush=True)
