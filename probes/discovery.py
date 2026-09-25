"""Training-free single-object discovery on VOC 2012 trainval with TokenCut, at every depth.

No labels are used to produce the box.  The patch tokens at one depth give a cosine-similarity graph, edges are
thresholded (>= TAU keeps weight 1, everything else EPS), and the second-smallest eigenvector of the normalised
Laplacian bipartitions the patches; the side holding the largest-magnitude entry is the object and the connected
component containing that seed becomes the box (Wang et al. 2022).  A normalised cut always returns a proper
bipartition, so the box cannot collapse to the whole frame -- LOST's degree rule does exactly that on these
supervised ViTs (every patch comes out foreground, mean box area 1.00), which is why TokenCut is used here.
The mean box area is reported next to CorLoc regardless, because a detector returning most of the image scores
well on CorLoc.

Both the residual stream (13 depths, what the rest of this package probes) and the attention keys of each block
(12 depths, the features TokenCut was built on) are scored.  CorLoc = share of images whose box has IoU >= 0.5
with any non-difficult ground-truth box.  Images are squashed to 224x224 so no object leaves the frame; boxes
are scored in original image coordinates.

  python -m probes.discovery --models r_s0,p_s0,ftbc7lm_s0 [--limit 256] [--workers 12]
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from probes import data as pdata
from probes import registry
from probes.dense_probes import token_maps

VOC_ROOT = registry.DATA / "voc" / "VOCdevkit" / "VOC2012"
GRID = 14
IMG = 224
D = 768
N_DEPTH = 13
TAU = 0.2               # TokenCut's edge threshold on cosine similarity
EPS = 1e-5
OUT = registry.RES / "discovery"
VARIANTS = (("stream", N_DEPTH), ("keys", N_DEPTH - 1))


def _boxes(xml_path):
    """Ground-truth boxes [(x1,y1,x2,y2)] of one VOC annotation, skipping 'difficult', plus the image size."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W, H = int(size.find("width").text), int(size.find("height").text)
    out = []
    for obj in root.findall("object"):
        d = obj.find("difficult")
        if d is not None and d.text.strip() == "1":
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((x1 - 1, y1 - 1, x2, y2))     # VOC ships 1-based inclusive corners; TokenCut's loader does the same
    return out, (W, H)


def voc_items():
    """Frozen list of VOC12 trainval images that have at least one non-difficult box."""
    rec = pdata._load_subset("discovery_voc12", "trainval")
    if rec is not None:
        return rec
    ids = [l.strip() for l in open(VOC_ROOT / "ImageSets" / "Main" / "trainval.txt") if l.strip()]
    items = []
    for iid in sorted(ids):
        boxes, _ = _boxes(VOC_ROOT / "Annotations" / f"{iid}.xml")
        if boxes:
            items.append((str(VOC_ROOT / "JPEGImages" / f"{iid}.jpg"), len(items)))
    return pdata._save_subset("discovery_voc12", "trainval", items, 20, kind="path")


class SquashDataset(Dataset):
    """Whole image squashed to 224x224 (no crop) + the GT boxes in original coordinates."""

    MAXB = 64        # VOC12 trainval tops out at 39 non-difficult boxes; truncating any of them would fake misses

    def __init__(self, items):
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        from torchvision import transforms
        self.items = list(items)
        self.tf = transforms.Compose([
            transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, _ = self.items[i]
        img = Image.open(p).convert("RGB")
        boxes, (W, H) = _boxes(VOC_ROOT / "Annotations" / f"{os.path.basename(p)[:-4]}.xml")
        assert len(boxes) <= self.MAXB, (p, len(boxes))
        b = np.zeros((self.MAXB, 4), dtype=np.float32)
        n = len(boxes)
        b[:n] = np.asarray(boxes, dtype=np.float32)
        return self.tf(img), torch.from_numpy(b), n, W, H


@torch.no_grad()
def key_maps(m, img):
    """-> [B,12,196,768] attention keys (concatenated over heads) read at every block."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = m.patch_embed(img)
        x = m._pos_embed(x)
        x = m.norm_pre(x)
        out = []
        for blk in m.blocks:
            qkv = blk.attn.qkv(blk.norm1(x))                  # [B,N,3D] laid out as q | k | v
            out.append(qkv[:, 1:, D:2 * D].float())
            x = blk(x)
    return torch.stack(out, 1)


@torch.no_grad()
def tokencut_masks(feats):
    """[B,196,768] patch tokens -> (foreground mask [B,196] bool, seed patch [B])."""
    f = torch.nn.functional.normalize(feats.float(), dim=-1)
    A = f @ f.transpose(1, 2)
    W = torch.where(A >= TAU, torch.ones_like(A), torch.full_like(A, EPS))
    d = W.sum(-1).clamp_min(1e-8)
    dm = d.pow(-0.5)
    B, N = d.shape
    L = torch.eye(N, device=f.device).expand(B, N, N) - dm.unsqueeze(-1) * W * dm.unsqueeze(-2)
    L = 0.5 * (L + L.transpose(1, 2))                         # symmetrise against round-off
    _, vec = torch.linalg.eigh(L.double())
    x = (dm.double() * vec[..., 1]).float()                   # Fiedler vector of the normalised cut
    part = x > x.mean(-1, keepdim=True)
    seed = x.abs().argmax(-1)                                 # TokenCut's rule: the extreme entry is on the object
    high = part.gather(1, seed.unsqueeze(1)).squeeze(1)
    mask = torch.where(high.unsqueeze(1), part, ~part)
    return mask, seed


def boxes_from_masks(mask, seed):
    """[B,196] mask + seed -> [B,4] boxes in 14x14 grid cells (the component containing the seed)."""
    from scipy import ndimage
    m = mask.view(-1, GRID, GRID).cpu().numpy()
    s = seed.cpu().numpy()
    out = np.zeros((len(m), 4), dtype=np.int64)
    for i in range(len(m)):
        lab, _ = ndimage.label(m[i])
        sy, sx = s[i] // GRID, s[i] % GRID
        comp = (lab == lab[sy, sx]) if lab[sy, sx] > 0 else m[i]
        ys, xs = np.nonzero(comp)
        out[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return out


def corloc_hit(box_grid, gt, n, W, H):
    """Box in grid cells -> original image pixels; True if IoU >= 0.5 with any of the n ground-truth boxes."""
    x1, y1, x2, y2 = box_grid
    sx, sy = W / GRID, H / GRID
    bx = np.array([x1 * sx, y1 * sy, (x2 + 1) * sx, (y2 + 1) * sy])
    for g in gt[:n]:
        ix1, iy1 = max(bx[0], g[0]), max(bx[1], g[1])
        ix2, iy2 = min(bx[2], g[2]), min(bx[3], g[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
        if union > 0 and inter / union >= 0.5:
            return True
    return False


def run(models, batch_size, workers, limit=None):
    rec = voc_items()
    items = [(p, y) for p, y in rec["items"]]
    if limit:
        items = items[:limit]
    ds = SquashDataset(items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    nets = {e["name"]: registry.load_model(e, "cuda") for e in models}
    hits = {n: {v: np.zeros(nd, dtype=np.int64) for v, nd in VARIANTS} for n in nets}
    area = {n: {v: np.zeros(nd) for v, nd in VARIANTS} for n in nets}
    seen, t0 = 0, time.time()
    for img, gt, n, W, H in loader:
        img = img.cuda(non_blocking=True)
        gt, n, W, H = gt.numpy(), n.numpy(), W.numpy(), H.numpy()
        for name, m in nets.items():
            maps = {"stream": token_maps(m, img), "keys": key_maps(m, img)}
            for v, nd in VARIANTS:
                for d in range(nd):
                    mask, seed = tokencut_masks(maps[v][:, d])
                    boxes = boxes_from_masks(mask, seed)
                    area[name][v][d] += float(((boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)).sum()) / (GRID * GRID)
                    for i in range(len(boxes)):
                        hits[name][v][d] += corloc_hit(boxes[i], gt[i], n[i], W[i], H[i])
        seen += img.shape[0]
        if seen % (batch_size * 10) == 0 or seen == len(ds):
            print(f"[discovery] {seen}/{len(ds)} imgs, {seen / (time.time() - t0):.1f} img/s x {len(nets)} models", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in nets:
        e = registry.BY_NAME[name]
        cor = {v: (100.0 * hits[name][v] / seen).round(4).tolist() for v, _ in VARIANTS}
        ar = {v: (area[name][v] / seen).round(4).tolist() for v, _ in VARIANTS}
        out = dict(model=name, arm=e["arm"], seed=e["seed"], dataset="voc12_trainval", task="tokencut_object_discovery",
                   metric="corloc_iou50", n=int(seen), limit=limit, tau=TAU, grid=GRID, transform="squash224",
                   corloc=cor["stream"], best_depth=int(np.argmax(cor["stream"])), mean_box_area=ar["stream"],
                   corloc_keys=cor["keys"], best_depth_keys=int(np.argmax(cor["keys"])), mean_box_area_keys=ar["keys"],
                   note="corloc = residual stream at 13 depths; corloc_keys = attention keys of the 12 blocks",
                   subset_sha256=rec["sha256"], git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"))
        tag = name + (f"__limit{limit}" if limit else "")
        json.dump(out, open(OUT / f"{tag}__voc12.json", "w"), indent=1)
        bk, bs = out["best_depth_keys"], out["best_depth"]
        print(f"[discovery] {name}: keys CorLoc best {cor['keys'][bk]:.2f} at b{bk} (area {ar['keys'][bk]:.2f}) | "
              f"stream best {cor['stream'][bs]:.2f} at depth {bs} (area {ar['stream'][bs]:.2f}) "
              f"-> {OUT / f'{tag}__voc12.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    print(f"models ({len(models)}): {[m['name'] for m in models]}", flush=True)
    run(models, args.batch_size, args.workers, args.limit)


if __name__ == "__main__":
    main()
