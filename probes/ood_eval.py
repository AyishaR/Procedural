"""Zero-shot OOD evaluation of a final checkpoint with its OWN head (no probes, nothing tuned on shifted data).

Per model: ImageNet val (50k, x of the effective-robustness plot; checked against log.txt), ImageNet-V2,
ImageNet-R and ImageNet-A (logits masked to their 200 classes; the unmasked number is kept too), ImageNet-Sketch,
ObjectNet (masked to the union of the 113 overlap classes, winner mapped back to the ObjectNet class) and the 75
ImageNet-C cells on the shared 5k draw.  Writes results/ood/<model>.json (+ <model>__preds.npz with every
per-image prediction); evaluation is INCREMENTAL -- a set already in the JSON with the same subset hash and the
same checkpoint is kept, so adding a set never recomputes the others.

Concurrency: run at most ONE process per model at a time.  Saves merge whatever is on disk, which repairs the
common case, but the read-merge-write is not locked, so two jobs racing on the same model can still drop a set.
Splitting by --sets across jobs is only safe for different models.

  python -m probes.ood_eval --models r_s0 --sets imagenet_sketch,objectnet --batch_size 256 --workers 16
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
EVAL_VERSION = 2        # bump when a transform, mask, remapping or metric changes; records without it are pre-versioning


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


def current_sha(name):
    """The hash a stored record must carry to count as up to date for `name`."""
    if name == "imagenet_c":
        return ood_data.imagenet_c_file_draw()["sha256"]
    return ood_data.subset_items(name)["sha256"]


def evaluate(entry, batch_size, workers, limit=None, sets=None, force=False):
    jpath, ppath = out_paths(entry, limit)
    ck_mtime = os.path.getmtime(registry.ckpt_path(entry))
    want = list(sets) if sets else list(ood_data.SETS)
    fresh = dict(model=entry["name"], arm=entry["arm"], seed=entry["seed"], ckpt=str(registry.ckpt_path(entry)),
                 ckpt_mtime=ck_mtime, limit=limit, batch_size=batch_size, workers=workers, git=registry.git_commit(),
                 date=datetime.now().isoformat(timespec="seconds"), complete=False, sets={})
    res, npz = fresh, {}
    if jpath.exists():
        old = json.load(open(jpath))
        reusable = (old.get("ckpt") == str(registry.ckpt_path(entry)) and old.get("ckpt_mtime") == ck_mtime
                    and old.get("limit") == limit and ppath.exists())
        if reusable and not force:
            res = old
            res.update(git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"))
            npz = {k: v for k, v in np.load(ppath).items()}
        elif not reusable:
            print(f"[{entry['name']}] existing {jpath.name} is stale (checkpoint or predictions changed); recomputing", flush=True)
    def have(name):
        """A stored set counts only if its subset hash matches and it was produced by this evaluator version.

        Records written before EVAL_VERSION existed are accepted (they were produced by the protocol in force
        at the time); bump EVAL_VERSION to force everything after a protocol change.
        """
        r = res["sets"].get(name)
        key = "draw_sha256" if name == "imagenet_c" else "subset_sha256"
        if not r or r.get(key) != current_sha(name):
            return False
        return r.get("eval_version", EVAL_VERSION) == EVAL_VERSION
    todo = [n for n in want if force or not have(n)]
    if not todo:
        print(f"[{entry['name']}] all {len(want)} sets already evaluated ({jpath.name}); skipping", flush=True)
        return res
    print(f"[{entry['name']}] evaluating {todo} (keeping {[n for n in want if n not in todo]})", flush=True)
    model = registry.load_model(entry, "cuda")
    log_acc = registry.final_test_acc(entry)
    OUT.mkdir(parents=True, exist_ok=True)

    def save():
        """Persist after every set, merging anything another process stored meanwhile.

        Two jobs may legitimately work on the same model at once (different --sets).  Each holds only its own
        results in memory, so without this merge the second writer would drop the first one's sets.
        """
        if jpath.exists():
            try:
                disk = json.load(open(jpath))
            except json.JSONDecodeError:
                disk = None
            same_ckpt = (disk and disk.get("ckpt") == str(registry.ckpt_path(entry))
                         and disk.get("ckpt_mtime") == ck_mtime and disk.get("limit") == limit)
            if same_ckpt:
                for k, v in disk.get("sets", {}).items():
                    res["sets"].setdefault(k, v)
                if ppath.exists():
                    for k, v in np.load(ppath).items():
                        npz.setdefault(k, v)
        res["complete"] = all(have(n) for n in ood_data.SETS)
        ptmp = ppath.with_suffix(f".{os.getpid()}.tmp.npz")      # publish the archive atomically
        np.savez_compressed(ptmp, **npz)
        os.replace(ptmp, ppath)
        tmp = jpath.with_suffix(f".{os.getpid()}.tmp")
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, jpath)

    def run(names, mask=None, pred_map=None):
        """-> {name: (rec, pred, lab)}.  mask restricts the argmax; pred_map relabels the winner (ObjectNet)."""
        res_names = {}
        for name, (pred, pred_m, lab, meta) in predict(model, names, batch_size, workers, limit, mask).items():
            key = name.replace("/", "__")
            npz[key + "__pred"] = pred
            if pred_m is not None:
                npz[key + "__pred_masked"] = pred_m
            npz[key + "__label"] = lab
            if pred_map is None:
                t1, t1m = top1(pred, lab), (top1(pred_m, lab) if pred_m is not None else None)
            else:                                  # ObjectNet: compare in the set's own 113-class space
                mp = np.asarray(pred_map)
                t1 = top1(mp[pred.astype(np.int64)], lab)
                t1m = top1(mp[pred_m.astype(np.int64)], lab) if pred_m is not None else None
            rec = dict(top1=t1, n=int(meta["n_eval"]), n_full=int(meta["n"]), subset_sha256=meta["subset_sha256"],
                       transform=meta["transform"], img_per_s=meta["img_per_s"], eval_version=EVAL_VERSION)
            if t1m is not None:
                rec["top1_masked"] = t1m
            print(f"[{entry['name']}] {name:32s} top-1 {rec['top1']:6.2f}" + (f" masked {rec['top1_masked']:6.2f}" if t1m is not None else "")
                  + f"  ({rec['n']} imgs, pass rate {rec['img_per_s']:.0f} img/s)", flush=True)
            res_names[name] = (rec, pred, lab)
        return res_names

    def simple(name):
        """One masked/plain set from ood_data.eval_spec."""
        mask, pmap = ood_data.eval_spec(name)
        return run([name], mask=mask, pred_map=pmap)[name][0]

    # ImageNet val: the x axis; must agree with log.txt (same head, same eval transform, one centre crop)
    if "imnet_val" in todo:
        rec, pred, lab = run(["imnet_val"])["imnet_val"]
        rec["log_top1"] = log_acc
        rec["diff_vs_log"] = round(rec["top1"] - log_acc, 4) if (log_acc is not None and limit is None) else None
        res["sets"]["imnet_val"] = rec
        if limit is None:
            idx = ood_data.imagenet_c_clean_index()
            res["sets"]["imnet_val_c_draw"] = dict(top1=top1(pred[idx], lab[idx]), n=int(len(idx)),
                                                  note="clean accuracy on the 5k val images used in every ImageNet-C cell")
        save()
    for name in ("imagenet_v2", "imagenet_r", "imagenet_sketch", "imagenet_a", "objectnet"):
        if name in todo:
            res["sets"][name] = simple(name)
            save()
    if "imagenet_c" in todo:
        cell_names = [f"imagenet_c/{c}/{s}" for c in ood_data.CORRUPTIONS for s in ood_data.SEVERITIES]
        got = run(cell_names)
        cells = {c: {str(s): got[f"imagenet_c/{c}/{s}"][0] for s in ood_data.SEVERITIES} for c in ood_data.CORRUPTIONS}
        per_sev = {str(s): round(float(np.mean([cells[c][str(s)]["top1"] for c in ood_data.CORRUPTIONS])), 4) for s in ood_data.SEVERITIES}
        per_corr = {c: round(float(np.mean([cells[c][str(s)]["top1"] for s in ood_data.SEVERITIES])), 4) for c in ood_data.CORRUPTIONS}
        per_group = {g: round(float(np.mean([per_corr[c] for c in cs])), 4) for g, cs in ood_data.C_GROUPS.items()}
        res["sets"]["imagenet_c"] = dict(top1=round(float(np.mean(list(per_corr.values()))), 4), per_severity=per_sev,
                                         per_corruption=per_corr, per_group=per_group, cells=cells,
                                         draw_sha256=ood_data.imagenet_c_file_draw()["sha256"], per_class=ood_data.C_PER_CLASS,
                                     eval_version=EVAL_VERSION)
    save()
    S = res["sets"]
    def g(k, f="top1"):
        return f"{S[k][f]:.2f}" if k in S and f in S[k] else "--"
    print(f"[{entry['name']}] DONE  val {g('imnet_val')} (log {log_acc})  v2 {g('imagenet_v2')}  r {g('imagenet_r','top1_masked')}  "
          f"sketch {g('imagenet_sketch')}  a {g('imagenet_a','top1_masked')}  objnet {g('objectnet','top1_masked')}  "
          f"c {g('imagenet_c')}  complete={res['complete']}  -> {jpath}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images per set (results tagged __limitN)")
    ap.add_argument("--sets", default="all", help=f"comma list or 'all' (= {','.join(ood_data.SETS)})")
    ap.add_argument("--force", action="store_true", help="recompute the requested sets even if they are already stored")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    models = registry.resolve_models(args.models)
    print(f"models ({len(models)}): {[m['name'] for m in models]}", flush=True)
    sets = None if args.sets == "all" else [x for x in args.sets.split(",") if x]
    for name in sets or []:
        if name not in ood_data.SETS:
            raise SystemExit(f"unknown set {name!r}; known: {ood_data.SETS}")
    for e in models:
        evaluate(e, args.batch_size, args.workers, args.limit, sets=sets, force=args.force)


if __name__ == "__main__":
    main()
