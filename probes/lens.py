"""Logit lens and tuned lens at epoch 299, from the cached ImageNet CLS features.

Per depth d the lens is h' = h + A_d h + c_d (A, c zero-initialised, so the untrained lens IS the logit lens),
followed by the checkpoint's frozen final LayerNorm and head.  The tuned lens is trained on the 100k-image
ImageNet train subset with KL(p_final || p_lens) summed over depths; the step is selected by val KL.  Metrics
on the ImageNet val cache: top-1/top-5 vs truth, agreement with the model's final prediction, mean KL, and
per-image prediction depth.  Both lenses are also evaluated on ImageNet-R (200-class mask).

  python -m probes.lens --models all [--steps 3000]
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from probes import data as pdata
from probes import features as feat
from probes import registry
from probes.train_probes import lr_at

EVAL_EVERY = 100


class Lens(torch.nn.Module):
    def __init__(self, norm_w, norm_b, head_w, head_b, K=13, D=768, eps=1e-6):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(K, D, D))
        self.c = torch.nn.Parameter(torch.zeros(K, D))
        for n, t in (("norm_w", norm_w), ("norm_b", norm_b), ("head_w", head_w), ("head_b", head_b)):
            self.register_buffer(n, t.float())
        self.eps = eps

    def forward(self, h, depths=None):
        """h [B,K,D] fp32 -> logits [K,B,C] for the given depth indices (default all)."""
        if depths is not None:
            h, A, c = h[:, depths], self.A[depths], self.c[depths]
        else:
            A, c = self.A, self.c
        x = h.permute(1, 0, 2)                                      # [K,B,D]
        x = x + torch.einsum("kbd,kde->kbe", x, A) + c[:, None]
        x = F.layer_norm(x, (x.shape[-1],), self.norm_w, self.norm_b, self.eps)
        return torch.einsum("kbd,cd->kbc", x, self.head_w) + self.head_b


@torch.no_grad()
def metrics(lens, X, y, p_final_pred, p_final_logp, cols=None, chunk=2048):
    """-> dict of per-depth lists: top1, top5, agree (with final argmax), kl(p_final||p_d); plus prediction depth.

    `cols` restricts every distribution to a class subset (ImageNet-R): logits are column-indexed and re-normalised,
    and `y` / `p_final_pred` must already be indices into that subset (see `final_dist`)."""
    K = X.shape[1]
    top1 = torch.zeros(K); top5 = torch.zeros(K); agree = torch.zeros(K); kl = torch.zeros(K)
    pred_depth = []
    for s in range(0, X.shape[0], chunk):
        h = X[s:s + chunk].float()
        yy = y[s:s + chunk]
        logits = lens(h)                                             # [K,B,C]
        if cols is not None:
            logits = logits[..., cols]
        logp = F.log_softmax(logits, -1)
        pred = logits.argmax(-1)                                    # [K,B]
        top1 += (pred == yy[None]).float().sum(1).cpu()
        top5 += (logits.topk(5, -1).indices == yy[None, :, None]).any(-1).float().sum(1).cpu()
        fp = p_final_pred[s:s + chunk]
        eq = pred == fp[None]                                       # [K,B]
        agree += eq.float().sum(1).cpu()
        lp = p_final_logp[s:s + chunk]                              # [B,C']  same support as logp
        kl += (lp.exp() * (lp[None] - logp)).sum(-1).sum(1).cpu()
        # prediction depth: first k such that eq[k:] all True (K-1 if only the last depth agrees)
        suffix_all = torch.flip(torch.cumprod(torch.flip(eq.int(), [0]), 0), [0]).bool()   # [K,B]
        first = torch.where(suffix_all.any(0), suffix_all.int().argmax(0), torch.full_like(yy, K - 1))
        pred_depth.append(first.cpu())
    N = X.shape[0]
    pd_all = torch.cat(pred_depth).float()
    return dict(top1=(100 * top1 / N).tolist(), top5=(100 * top5 / N).tolist(), agree=(100 * agree / N).tolist(),
                kl=(kl / N).tolist(), pred_depth_mean=float(pd_all.mean()), frac_decided_by_b8=float((pd_all <= 9).float().mean()),
                pred_depth_hist=torch.bincount(pd_all.long(), minlength=K).tolist())


def run(entry, steps, batch, device="cuda"):
    name = entry["name"]
    sd = torch.load(registry.ckpt_path(entry), map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd and isinstance(sd["model"], dict) else sd
    lens = Lens(sd["norm.weight"], sd["norm.bias"], sd["head.weight"], sd["head.bias"]).to(device)
    tr, va, te = (feat.load(name, "imnet", s) for s in ("train", "val", "test"))
    Xtr, Xva, Xte = (feat.select(d["X"], "cls", device) for d in (tr, va, te))
    ytr, yva, yte = (torch.from_numpy(d["y"].astype(np.int64)).to(device) for d in (tr, va, te))

    @torch.no_grad()
    def final_dist(X, cols=None, chunk=4096):
        """Final-depth distribution and argmax; with `cols`, both restricted to (and indexed within) that class subset."""
        lp, pr = [], []
        for s in range(0, X.shape[0], chunk):
            lg = lens(X[s:s + chunk].float(), depths=[12])[0]
            if cols is not None:
                lg = lg[:, cols]
            lp.append(F.log_softmax(lg, -1)); pr.append(lg.argmax(-1))
        return torch.cat(lp), torch.cat(pr)
    lp_tr, pred_tr = final_dist(Xtr)
    lp_va, pred_va = final_dist(Xva)
    lp_te, pred_te = final_dist(Xte)
    cached_pred = torch.from_numpy(te["preds"].astype(np.int64)).to(device)
    agree_cached = float((pred_te == cached_pred).float().mean())
    assert agree_cached >= 0.995, f"{name}: final-depth lens disagrees with cached predictions ({agree_cached:.4f})"

    out = dict(model=name, arm=entry["arm"], seed=entry["seed"], ckpt=str(registry.ckpt_path(entry)), depths=registry.DEPTHS,
               final_top1_logtxt=registry.final_test_acc(entry), agree_with_cached_preds=agree_cached,
               config=dict(steps=steps, batch=batch, lr=1e-3, warmup=100), git=registry.git_commit(), date=datetime.now().isoformat(timespec="seconds"))
    out["logit_lens"] = metrics(lens, Xte, yte, pred_te, lp_te)
    r = feat.load(name, "imagenet_r", "test") if feat.is_done(name, "imagenet_r", "test") else None
    if r is not None:
        Xr = feat.select(r["X"], "cls", device)
        mask = torch.from_numpy(pdata.imagenet_r_class_mask()).to(device)
        cols = mask.nonzero().flatten()                              # the 200 ImageNet classes of ImageNet-R
        remap = torch.full((1000,), -1, dtype=torch.long, device=device)
        remap[cols] = torch.arange(cols.numel(), device=device)
        yr = remap[torch.from_numpy(r["y"].astype(np.int64)).to(device)]
        assert int((yr < 0).sum()) == 0
        lp_r, pred_r = final_dist(Xr, cols=cols)                     # reference restricted to the same 200 classes
        out["imagenet_r"] = dict(n_classes=int(cols.numel()), logit_lens=metrics(lens, Xr, yr, pred_r, lp_r, cols=cols))

    # tuned lens
    assert steps >= EVAL_EVERY, f"steps must be >= {EVAL_EVERY} (one validation point)"
    opt = torch.optim.Adam(lens.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(0)
    N = Xtr.shape[0]
    B = min(batch, N)
    perm, ptr = torch.randperm(N, generator=g), 0
    curve, best_val, best_state, best_step = [], float("inf"), None, 0
    t0 = time.time()
    for step in range(steps):
        if ptr + B > N:
            perm, ptr = torch.randperm(N, generator=g), 0
        idx = perm[ptr:ptr + B].to(device); ptr += B
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, steps)
        logits = lens(Xtr[idx].float())                                 # [K,B,C]
        logp = F.log_softmax(logits, -1)
        lp = lp_tr[idx]
        kl = (lp.exp() * (lp[None] - logp)).sum(-1).mean(-1)         # [K]
        loss = kl[:12].sum()                                          # depth 12 is the identity (KL 0) by construction
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                vk = torch.zeros(13, device=device)
                for s in range(0, Xva.shape[0], 4096):
                    lg = F.log_softmax(lens(Xva[s:s + 4096].float()), -1)
                    lpv = lp_va[s:s + 4096]
                    vk += (lpv.exp() * (lpv[None] - lg)).sum(-1).sum(-1)
                vk = (vk / Xva.shape[0]).cpu()
            curve.append(dict(step=step + 1, train_kl=kl.detach().cpu().tolist(), val_kl=vk.tolist()))
            v = float(vk[:12].sum())
            if v < best_val:
                best_val, best_step = v, step + 1
                best_state = {k: t.detach().clone() for k, t in lens.state_dict().items()}
    assert best_state is not None
    lens.load_state_dict(best_state)
    out["tuned_lens"] = metrics(lens, Xte, yte, pred_te, lp_te)
    out["tuned_lens"].update(best_step=best_step, best_val_kl_sum=best_val, train_curve=curve, seconds=time.time() - t0)
    assert out["tuned_lens"]["kl"][12] < 1e-4, out["tuned_lens"]["kl"][12]
    if r is not None:
        out["imagenet_r"]["tuned_lens"] = metrics(lens, Xr, yr, pred_r, lp_r, cols=cols)
    del Xtr, Xva, Xte
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    registry.ensure_dirs()
    for e in registry.resolve_models(args.models):
        outp = registry.RES / "lens" / f"{e['name']}{feat.result_suffix(e['name'], 'imnet')}.json"
        if outp.exists() and not args.force:
            print(f"skip {outp.name} (exists)", flush=True); continue
        if not all(feat.is_done(e["name"], "imnet", s) for s in ("train", "val", "test")):
            print(f"skip {e['name']}: imnet features not cached", flush=True); continue
        res = run(e, args.steps, args.batch)
        print(f"{e['name']:14s} logit lens top-1: " + " ".join(f"{a:5.1f}" for a in res["logit_lens"]["top1"]), flush=True)
        print(f"{e['name']:14s} tuned lens top-1: " + " ".join(f"{a:5.1f}" for a in res["tuned_lens"]["top1"]) + f"  (best step {res['tuned_lens']['best_step']})", flush=True)
        tmp = outp.with_suffix(".tmp"); json.dump(res, open(tmp, "w")); os.replace(tmp, outp)


if __name__ == "__main__":
    main()
