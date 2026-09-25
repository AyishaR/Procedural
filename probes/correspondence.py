"""Semantic correspondence on SPair-71k with frozen features, at every depth (no training, no matching head).

For each annotated source keypoint the source patch-token map is sampled bilinearly at that point, the cosine
similarity against every target patch token gives a 14x14 score map, the map is bilinearly upsampled to 224x224
and its argmax is the predicted match.  A match counts at PCK@alpha when it lands within alpha * max(w, h) of
the target object's bounding box from the ground-truth keypoint.  Both images are squashed to 224x224 so no
keypoint leaves the frame; predictions are scaled back to original image coordinates before scoring.

  python -m probes.correspondence --models all [--limit 200] [--split test]
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from probes import registry
from probes.dense_probes import token_maps

ROOT = registry.DATA / "SPair-71k"
GRID = 14
IMG = 224
N_DEPTH = 13
ALPHAS = (0.05, 0.1)
OUT = registry.RES / "correspondence"


def pair_files(split="test"):
    return sorted((ROOT / "PairAnnotation" / split).glob("*.json"))


class PairDataset(Dataset):
    """One item = one annotated pair: both images squashed to 224x224 plus keypoints/bbox in original pixels."""

    MAXK = 30

    def __init__(self, files):
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        from torchvision import transforms
        self.files = list(files)
        self.tf = transforms.Compose([
            transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        a = json.load(open(self.files[i]))
        cat = a["category"]
        si = Image.open(ROOT / "JPEGImages" / cat / a["src_imname"]).convert("RGB")
        ti = Image.open(ROOT / "JPEGImages" / cat / a["trg_imname"]).convert("RGB")
        skp = np.asarray(a["src_kps"], dtype=np.float32).reshape(-1, 2)
        tkp = np.asarray(a["trg_kps"], dtype=np.float32).reshape(-1, 2)
        k = min(len(skp), self.MAXK)
        S = np.zeros((self.MAXK, 2), dtype=np.float32); S[:k] = skp[:k]
        T = np.zeros((self.MAXK, 2), dtype=np.float32); T[:k] = tkp[:k]
        tb = np.asarray(a["trg_bndbox"], dtype=np.float32)          # x1, y1, x2, y2
        thr = float(max(tb[2] - tb[0], tb[3] - tb[1]))
        return (self.tf(si), self.tf(ti), torch.from_numpy(S), torch.from_numpy(T), k,
                si.size[0], si.size[1], ti.size[0], ti.size[1], thr)


@torch.no_grad()
def match(src_tok, trg_tok, kps, k, sw, sh, tw, th):
    """Per depth, predicted target points for the k source keypoints.  -> [13,k,2] in target pixel coordinates."""
    Dp = src_tok.shape[-1]
    s = F.normalize(src_tok.float(), dim=-1).permute(0, 2, 1).reshape(N_DEPTH, Dp, GRID, GRID)
    t = F.normalize(trg_tok.float(), dim=-1)                                   # [13,196,768]
    # sample the source map at the keypoints (normalised grid coords on the squashed 224x224 image)
    gx = 2 * (kps[:k, 0] + 0.5) / max(sw, 1e-6) - 1      # pixel centres, matching align_corners=False
    gy = 2 * (kps[:k, 1] + 0.5) / max(sh, 1e-6) - 1
    grid = torch.stack([gx, gy], -1).view(1, 1, k, 2).expand(N_DEPTH, 1, k, 2).to(s.device)
    q = F.grid_sample(s, grid, mode="bilinear", align_corners=False).squeeze(2)   # [13,768,k]
    q = F.normalize(q, dim=1)
    sim = torch.einsum("dnc,dck->dkn", t, q).reshape(N_DEPTH * k, 1, GRID, GRID)  # [13*k,1,14,14]
    up = F.interpolate(sim, size=(IMG, IMG), mode="bilinear", align_corners=False).view(N_DEPTH, k, IMG * IMG)
    idx = up.argmax(-1)
    y = ((idx // IMG).float() + 0.5) * th / IMG - 0.5                            # back to target pixels, same convention
    x = ((idx % IMG).float() + 0.5) * tw / IMG - 0.5
    return torch.stack([x, y], -1)


def run(models, batch_size, workers, split, limit=None):
    files = pair_files(split)
    if not files:
        raise SystemExit(f"no pair annotations under {ROOT / 'PairAnnotation' / split}")
    if limit:
        files = files[:limit]
    ds = PairDataset(files)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    nets = {e["name"]: registry.load_model(e, "cuda") for e in models}
    hit = {n: np.zeros((len(ALPHAS), N_DEPTH), dtype=np.int64) for n in nets}
    total, seen, t0 = 0, 0, time.time()
    for si, ti, S, T, k, sw, sh, tw, th, thr in loader:
        B = si.shape[0]
        both = torch.cat([si, ti]).cuda(non_blocking=True)
        for name, m in nets.items():
            tm = token_maps(m, both)                                             # [2B,13,196,768]
            for i in range(B):
                ki = int(k[i])
                if ki == 0:
                    continue
                pred = match(tm[i], tm[B + i], S[i].cuda(), ki, float(sw[i]), float(sh[i]), float(tw[i]), float(th[i]))
                gt = T[i, :ki].cuda()
                dist = (pred - gt.unsqueeze(0)).norm(dim=-1)                     # [13,k]
                for a, alpha in enumerate(ALPHAS):
                    hit[name][a] += (dist <= alpha * float(thr[i])).sum(1).cpu().numpy()
        total += int(k.sum())
        seen += B
        if seen % (batch_size * 10) == 0 or seen == len(ds):
            print(f"[correspondence] {seen}/{len(ds)} pairs, {seen / (time.time() - t0):.1f} pair/s x {len(nets)} models", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in nets:
        e = registry.BY_NAME[name]
        pck = {f"pck@{a}": (100.0 * hit[name][i] / max(total, 1)).round(4).tolist() for i, a in enumerate(ALPHAS)}
        out = dict(model=name, arm=e["arm"], seed=e["seed"], dataset=f"spair71k_{split}", task="semantic_correspondence",
                   n_pairs=int(seen), n_keypoints=int(total), limit=limit, grid=GRID, transform="squash224",
                   metric="pck_bbox_max_side", best_depth=int(np.argmax(pck["pck@0.1"])), git=registry.git_commit(),
                   date=datetime.now().isoformat(timespec="seconds"), **pck)
        tag = name + (f"__limit{limit}" if limit else "")
        json.dump(out, open(OUT / f"{tag}__spair_{split}.json", "w"), indent=1)
        p = pck["pck@0.1"]
        print(f"[correspondence] {name}: PCK@0.1 best {max(p):.2f} at depth {out['best_depth']} (b11 {p[12]:.2f}) "
              f"-> {OUT / f'{tag}__spair_{split}.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    print(f"models ({len(models)}): {[m['name'] for m in models]}", flush=True)
    run(models, args.batch_size, args.workers, args.split, args.limit)


if __name__ == "__main__":
    main()
