"""Two cheap readouts on the cached features.

k-NN: per dataset and depth (ftype concat), cosine k=20 weighted vote (DINO recipe) from the train subset to
the test split.  Low-level: closed-form ridge regression (batched over depths and lambdas) from standardised
ImageNet-train features to the 10 per-image image statistics stored at extraction; lambda selected on val,
R^2 per target on the ImageNet test split (and Places, which also stores the statistics).

  python -m probes.knn_lowlevel --models all --datasets all
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from probes import data as pdata
from probes import features as feat
from probes import registry

LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
K_NN = 20
TEMP = 0.07


@torch.no_grad()
def knn(Xtr, ytr, Xte, yte, C, k=K_NN, chunk=512):
    """Xtr/Xte [N,13,D] fp16 -> top-1 [13] with cosine similarity and temperature-weighted votes."""
    K = Xtr.shape[1]
    tr = torch.nn.functional.normalize(Xtr.float(), dim=-1).permute(1, 0, 2)        # [13,Ntr,D]
    correct = torch.zeros(K, device=Xtr.device)
    for s in range(0, Xte.shape[0], chunk):
        q = torch.nn.functional.normalize(Xte[s:s + chunk].float(), dim=-1).permute(1, 0, 2)   # [13,B,D]
        sim = torch.einsum("kbd,knd->kbn", q, tr)                                         # [13,B,Ntr]
        top, idx = sim.topk(min(k, tr.shape[1]), dim=-1)
        lab = ytr[idx]                                                                    # [13,B,k]
        w = (top / TEMP).exp()
        votes = torch.zeros(K, q.shape[1], C, device=Xtr.device).scatter_add_(2, lab, w)
        correct += (votes.argmax(-1) == yte[s:s + chunk][None]).float().sum(1)
    return (100 * correct / Xte.shape[0]).cpu().tolist()


@torch.no_grad()
def ridge_r2(Xtr, Ttr, Xva, Tva, Xte, Tte, mu, sd):
    """Batched ridge over 13 depths x lambdas; returns test R^2 [13, n_targets] at the val-selected lambda per depth/target."""
    D = Xtr.shape[2]
    dev = Xtr.device
    tmu, tsd = Ttr.mean(0), Ttr.std(0).clamp_min(1e-8)
    def prep(X):
        return ((X.float() - mu) / sd).permute(1, 0, 2)                # [13,N,D]
    A = prep(Xtr)
    T = ((Ttr - tmu) / tsd)                                            # [N,t]
    G = torch.einsum("knd,kne->kde", A, A) / A.shape[1]                # [13,D,D]
    R = torch.einsum("knd,nt->kdt", A, T) / A.shape[1]                 # [13,D,t]
    eye = torch.eye(D, device=dev)
    Wl = torch.stack([torch.linalg.solve(G + lam * eye, R) for lam in LAMBDAS], 1)   # [13,L,D,t]
    def r2(X, Tt):
        P = torch.einsum("knd,kldt->klnt", prep(X), Wl) * tsd + tmu     # [13,L,N,t]
        ss_res = ((P - Tt[None, None]) ** 2).sum(2)
        ss_tot = ((Tt - Tt.mean(0)) ** 2).sum(0)
        return 1 - ss_res / ss_tot                                     # [13,L,t]
    r2_va, r2_te = r2(Xva, Tva), r2(Xte, Tte)
    best = r2_va.argmax(1, keepdim=True)                               # [13,1,t]
    return r2_te.gather(1, best).squeeze(1).cpu().tolist(), [[LAMBDAS[int(i)] for i in row] for row in best.squeeze(1).cpu().tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    names = pdata.CLASSIFICATION if args.datasets == "all" else args.datasets.split(",")
    for e in registry.resolve_models(args.models):
        for dataset in names:
            if not all(feat.is_done(e["name"], dataset, s) for s in ("train", "val", "test")):
                print(f"skip {e['name']} {dataset}: features not cached", flush=True); continue
            outk = registry.RES / "probes" / f"{e['name']}__{dataset}__knn.json"
            if not outk.exists() or args.force:
                tr, te = feat.load(e["name"], dataset, "train"), feat.load(e["name"], dataset, "test")
                C = int(tr["meta"]["n_classes"])
                Xtr, Xte = feat.select(tr["X"], "concat"), feat.select(te["X"], "concat")
                ytr, yte = (torch.from_numpy(d["y"].astype(np.int64)).cuda() for d in (tr, te))
                acc = knn(Xtr, ytr, Xte, yte, C)
                json.dump(dict(model=e["name"], arm=e["arm"], seed=e["seed"], dataset=dataset, ftype="concat", k=K_NN, temp=TEMP,
                               knn_top1=acc, git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds")), open(outk, "w"))
                print(f"{e['name']:14s} {dataset:10s} kNN top-1 by depth: " + " ".join(f"{a:5.1f}" for a in acc), flush=True)
                del Xtr, Xte; torch.cuda.empty_cache()
            outl = registry.RES / "probes" / f"{e['name']}__{dataset}__lowlevel.json"
            if dataset in ("imnet", "places365") and (not outl.exists() or args.force):
                tr, va, te = (feat.load(e["name"], dataset, s) for s in ("train", "val", "test"))
                if "lowlevel" not in tr:
                    continue
                Xtr, Xva, Xte = (feat.select(d["X"], "concat") for d in (tr, va, te))
                Ttr, Tva, Tte = (torch.from_numpy(d["lowlevel"]).float().cuda() for d in (tr, va, te))
                mu, sd = feat.standardise_stats(Xtr)
                r2, lam = ridge_r2(Xtr, Ttr, Xva, Tva, Xte, Tte, mu, sd)
                json.dump(dict(model=e["name"], arm=e["arm"], seed=e["seed"], dataset=dataset, targets=feat.LOWLEVEL_TARGETS, lambdas=LAMBDAS,
                               r2=r2, best_lambda=lam, git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds")), open(outl, "w"))
                print(f"{e['name']:14s} {dataset:10s} low-level mean R2 by depth: " + " ".join(f"{np.mean(r):5.2f}" for r in r2), flush=True)
                del Xtr, Xva, Xte; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
