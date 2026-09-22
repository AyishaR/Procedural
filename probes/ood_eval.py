"""Zero-shot OOD evaluation of a final checkpoint with its OWN head (no probes, nothing tuned on shifted data).

Per model: ImageNet val (50k, x of the effective-robustness plot; checked against log.txt), ImageNet-V2, ImageNet-R
(logits masked to its 200 classes; the unmasked number is kept too) and the 75 ImageNet-C cells on the shared 5k
draw.  Writes results/ood/<model>.json (+ <model>__preds.npz with every per-image prediction) and skips a model
whose JSON already exists for the same checkpoint and limit.

  python -m probes.ood_eval --models r_s0 --batch_size 256 --workers 16 [--limit 512]
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
from probes import ood_data
from probes import registry

OUT = registry.RES / "ood"


def out_paths(entry, limit):
    tag = entry["name"] + (f"__limit{limit}" if limit is not None else "")
    return OUT / f"{tag}.json", OUT / f"{tag}__preds.npz"


@torch.no_grad()
def predict(model, names, batch_size, workers, limit, mask=None):
    """One DataLoader over the concatenated sets (one worker start-up instead of one per ImageNet-C cell) ->
    {name: (pred [N] int16 argmax over all 1000 logits, pred_masked [N] or None, labels [N] int16, meta)}."""
    parts, metas = [], []
    for name in names:
        ds, meta = ood_data.build_ood_dataset(name)
        N = len(ds) if limit is None else min(limit, len(ds))
        if limit is not None:
            ds = torch.utils.data.Subset(ds, list(range(N)))
        parts.append(ds); metas.append(dict(meta, n_eval=N))
    loader = DataLoader(torch.utils.data.ConcatDataset(parts), batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, drop_last=False)
    m = torch.as_tensor(mask, device="cuda") if mask is not None else None
    preds, preds_m, labels = [], [], []
    t0 = time.time()
    for img, y in loader:
        img = img.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img).float()
        preds.append(logits.argmax(1).cpu())
        if m is not None:
            preds_m.append(logits.masked_fill(~m, float("-inf")).argmax(1).cpu())
        labels.append(y)
    pred = torch.cat(preds).numpy().astype(np.int16)
    pred_m = torch.cat(preds_m).numpy().astype(np.int16) if m is not None else None
    lab = torch.cat(labels).numpy().astype(np.int16)
    total = sum(mt["n_eval"] for mt in metas)
    assert len(pred) == total, (len(pred), total)
    rate = round(total / (time.time() - t0), 1)
    out, s = {}, 0
    for name, mt in zip(names, metas):
        n = mt["n_eval"]
        out[name] = (pred[s:s + n], pred_m[s:s + n] if pred_m is not None else None, lab[s:s + n], dict(mt, img_per_s=rate))
        s += n
    return out


def top1(pred, lab):
    return round(float(100.0 * (pred == lab).mean()), 4)


def evaluate(entry, batch_size, workers, limit=None):
    jpath, ppath = out_paths(entry, limit)
    ck_mtime = os.path.getmtime(registry.ckpt_path(entry))
    if jpath.exists():
        old = json.load(open(jpath))
        if (old.get("ckpt") == str(registry.ckpt_path(entry)) and old.get("ckpt_mtime") == ck_mtime and old.get("limit") == limit
                and old.get("complete") and ppath.exists()):
            print(f"[{entry['name']}] already evaluated ({jpath.name}); skipping", flush=True)
            return old
    model = registry.load_model(entry, "cuda")
    log_acc = registry.final_test_acc(entry)
    res = dict(model=entry["name"], arm=entry["arm"], seed=entry["seed"], ckpt=str(registry.ckpt_path(entry)),
               ckpt_mtime=ck_mtime, limit=limit, batch_size=batch_size, workers=workers, git=registry.git_commit(),
               date=datetime.now().isoformat(timespec="seconds"), complete=False, sets={})
    npz = {}

    def run(names, mask=None):
        """-> {name: (rec, pred, lab)} for the sets evaluated in one pass."""
        res_names = {}
        for name, (pred, pred_m, lab, meta) in predict(model, names, batch_size, workers, limit, mask).items():
            key = name.replace("/", "__")
            npz[key + "__pred"] = pred
            if pred_m is not None:
                npz[key + "__pred_masked"] = pred_m
            npz[key + "__label"] = lab
            rec = dict(top1=top1(pred, lab), n=int(meta["n_eval"]), n_full=int(meta["n"]), subset_sha256=meta["subset_sha256"],
                       transform=meta["transform"], img_per_s=meta["img_per_s"])
            if pred_m is not None:
                rec["top1_masked"] = top1(pred_m, lab)
            print(f"[{entry['name']}] {name:32s} top-1 {rec['top1']:6.2f}" + (f" masked {rec['top1_masked']:6.2f}" if pred_m is not None else "")
                  + f"  ({rec['n']} imgs, pass rate {rec['img_per_s']:.0f} img/s)", flush=True)
            res_names[name] = (rec, pred, lab)
        return res_names

    # ImageNet val: the x axis; must agree with log.txt (same head, same eval transform, one centre crop)
    rec, pred, lab = run(["imnet_val"])["imnet_val"]
    rec["log_top1"] = log_acc
    rec["diff_vs_log"] = round(rec["top1"] - log_acc, 4) if (log_acc is not None and limit is None) else None
    res["sets"]["imnet_val"] = rec
    if limit is None:
        idx = ood_data.imagenet_c_clean_index()
        res["sets"]["imnet_val_c_draw"] = dict(top1=top1(pred[idx], lab[idx]), n=int(len(idx)),
                                              note="clean accuracy on the 5k val images used in every ImageNet-C cell")
    res["sets"]["imagenet_v2"] = run(["imagenet_v2"])["imagenet_v2"][0]
    res["sets"]["imagenet_r"] = run(["imagenet_r"], mask=pdata.imagenet_r_class_mask())["imagenet_r"][0]
    cell_names = [f"imagenet_c/{c}/{s}" for c in ood_data.CORRUPTIONS for s in ood_data.SEVERITIES]
    got = run(cell_names)
    cells = {c: {str(s): got[f"imagenet_c/{c}/{s}"][0] for s in ood_data.SEVERITIES} for c in ood_data.CORRUPTIONS}
    per_sev = {str(s): round(float(np.mean([cells[c][str(s)]["top1"] for c in ood_data.CORRUPTIONS])), 4) for s in ood_data.SEVERITIES}
    per_corr = {c: round(float(np.mean([cells[c][str(s)]["top1"] for s in ood_data.SEVERITIES])), 4) for c in ood_data.CORRUPTIONS}
    per_group = {g: round(float(np.mean([per_corr[c] for c in cs])), 4) for g, cs in ood_data.C_GROUPS.items()}
    res["sets"]["imagenet_c"] = dict(top1=round(float(np.mean(list(per_corr.values()))), 4), per_severity=per_sev,
                                     per_corruption=per_corr, per_group=per_group, cells=cells,
                                     draw_sha256=ood_data.imagenet_c_file_draw()["sha256"], per_class=ood_data.C_PER_CLASS)
    res["complete"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ppath, **npz)
    tmp = jpath.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, jpath)
    s = res["sets"]
    print(f"[{entry['name']}] DONE  val {s['imnet_val']['top1']:.2f} (log {log_acc})  v2 {s['imagenet_v2']['top1']:.2f}  "
          f"r {s['imagenet_r']['top1_masked']:.2f}  c {s['imagenet_c']['top1']:.2f}  -> {jpath}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images per set (results tagged __limitN)")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    print(f"models ({len(models)}): {[m['name'] for m in models]}", flush=True)
    for e in models:
        evaluate(e, args.batch_size, args.workers, args.limit)


if __name__ == "__main__":
    main()
