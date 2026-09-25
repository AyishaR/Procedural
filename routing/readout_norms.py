"""What the head actually reads, per finished readout arm: norms at the CLS row and cheap eval-only
ablations of the constant c = cls_token + pos_embed[0] (docs/readout_intervention_math.md).

The head reads LN_f(z)_cls with z = sum_j w_j s_j. This script runs the full ImageNet val set once
through every model (one pass over the data, all models per batch), recomputes the forward block by
block exactly as Block.forward does (under the same fp16 autocast as engine.evaluate), and keeps
only the CLS row. Per model it records

  * how much each block writes: mean |attn_l| and |mlp_l| at the CLS row (and averaged over the
    patch tokens), and the stream norm |x_l| at the CLS row, l = 0..12;
  * the head input's composition: |c|, |v| with v = z - w_0 c (everything but the constant), |z|,
    cos(c, v), and the same after LN_f's feature-centring. It also records the LN denominator sigma
    per image and its coefficient of variation over images: if c dominates, sigma is nearly
    constant and LN_f acts almost linearly on v;
  * top-1 of eval-only variants computed from the stored z: as trained; c toggled (its term
    removed where w_0 != 0, added with coefficient 1 where w_0 = 0); v + alpha c for a few alpha
    (as trained is alpha = w_0); sigma frozen to its dataset mean.

Only finished runs (epoch-299 checkpoints). Base = the three r seeds, read in place.
Writes <out>/norms.json, <out>/z_<tag>.pt (CLS-row z, c, labels) and <out>/readout_norms.png.
    sbatch routing/sbatch/readout_norms.sh            # GPU, alldlc2_gpu-h200
    python routing/readout_norms.py --from-json        # redraw the figure
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=FutureWarning)

import datasets as ds  # noqa: E402
import utils  # noqa: E402
from routing.gate import build_args  # noqa: E402

RUNS = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/runs"
BASE = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_29384839"
MODELS = {   # tag -> (checkpoint dir, --readout_routing mode)
    "A": (f"{RUNS}/readout_A_29725255/s0", "fixed_last4"),
    "B": (f"{RUNS}/readout_B_29725256/s0", "learn_last4"),
    "C": (f"{RUNS}/readout_C_29725257/s0", "learn_ones"),
    "D": (f"{RUNS}/readout_D_29768024/s0", "fixed_emb_last3"),
    "E": (f"{RUNS}/readout_E_29768025/s0", "fixed_b0_last3"),
    "r0": (f"{BASE}/s0", "off"),
    "r1": (f"{BASE}/s1", "off"),
    "r2": (f"{BASE}/s2", "off"),
}
DEFAULT_OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/readout_norms"
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def finished(d):
    """Only epoch-299 runs count (experiment-output conventions)."""
    log = Path(d) / "log.txt"
    if not (Path(d) / "checkpoint-299.pth").exists() or not log.exists():
        return False
    last = [json.loads(l) for l in open(log) if l.strip()][-1]
    return last["epoch"] == 299


def load(tag, device):
    d, mode = MODELS[tag]
    assert finished(d), f"{tag}: {d} has not finished training -- refusing"
    a = build_args(["--readout_routing", mode])
    m = utils.build_model(a)
    ck = torch.load(f"{d}/checkpoint-299.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    test_acc1 = [json.loads(l) for l in open(Path(d) / "log.txt") if l.strip()][-1]["test_acc1"]
    return m.to(device).eval(), test_acc1


@torch.no_grad()
def cls_forward(m, x):
    """Block.forward unrolled; returns (sources at the CLS row [2L+1, B, d] fp32,
    per-branch mean patch-token norms [2L, B], stream norms at the CLS row [L+1, B])."""
    h = m.norm_pre(m._pos_embed(m.patch_embed(x)))
    src, patch, stream = [h[:, 0].float()], [], [h[:, 0].float().norm(dim=-1)]
    for blk in m.blocks:
        a = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(h))))
        h = h + a
        mm = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
        h = h + mm
        for t in (a, mm):
            src.append(t[:, 0].float())
            patch.append(t[:, m.num_prefix_tokens:].float().norm(dim=-1).mean(dim=1))
        stream.append(h[:, 0].float().norm(dim=-1))
    return torch.stack(src), torch.stack(patch), torch.stack(stream)


def head(m, z):
    """LN_f -> fc_norm -> head on CLS-row vectors (LN is per token, so this is the model's head)."""
    return m.head(m.fc_norm(m.norm(z)))


def ln_parts(m, z):
    """(centred z, sigma) exactly as LN_f computes them."""
    zc = z - z.mean(dim=-1, keepdim=True)
    sigma = torch.sqrt(zc.pow(2).mean(dim=-1) + m.norm.eps)
    return zc, sigma


def run(a):
    device = torch.device("cuda")
    args = build_args()
    args.crop_pct = None
    dataset, _ = ds.build_dataset(is_train=False, args=args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=a.batch, shuffle=False,
                                         num_workers=a.workers, pin_memory=True)
    tags = [t for t in MODELS if t in a.models]
    models, logged = {}, {}
    for t in tags:
        models[t], logged[t] = load(t, device)
        print(f"[load] {t}: {MODELS[t][1]}  logged test_acc1 {logged[t]:.3f}", flush=True)
    acc = {t: dict(z=[], src_norm=[], patch=[], stream=[], y=[], sanity=0.0) for t in tags}
    t0 = time.time()
    for bi, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        for t in tags:
            m = models[t]
            with torch.cuda.amp.autocast():
                src, patch, stream = cls_forward(m, x)
            w = m.readout_mixer.weight.float() if hasattr(m, "readout_mixer") else torch.ones(src.shape[0], device=device)
            z = src[0] * w[0]
            for j in range(1, src.shape[0]):          # the mixer's order, fp32 accumulation
                z = z + w[j] * src[j]
            if bi == 0:                               # our CLS-row head == the model's own forward
                with torch.cuda.amp.autocast():
                    ref = m(x).float()
                    mine = head(m, z).float()
                acc[t]["sanity"] = (ref - mine).abs().max().item()
                acc[t]["c"] = src[0][0].cpu()
                acc[t]["c_spread"] = (src[0] - src[0][0]).abs().max().item()
            acc[t]["z"].append(z.cpu())
            acc[t]["src_norm"].append(src.norm(dim=-1).cpu())
            acc[t]["patch"].append(patch.cpu())
            acc[t]["stream"].append(stream.cpu())
            acc[t]["y"].append(y)
        if bi % 20 == 0:
            print(f"[pass] batch {bi}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"meta": dict(n_images=len(dataset), models={t: MODELS[t] for t in tags}, alphas=ALPHAS,
                        crop_pct=args.crop_pct, autocast="fp16 (as engine.evaluate)")}
    for t in tags:
        m = models[t]
        w = (m.readout_mixer.weight.float().cpu() if hasattr(m, "readout_mixer")
             else torch.ones(2 * len(m.blocks) + 1))
        z = torch.cat(acc[t]["z"]); y = torch.cat(acc[t]["y"])
        src_norm = torch.cat(acc[t]["src_norm"], dim=1)       # [25, N]
        patch = torch.cat(acc[t]["patch"], dim=1)             # [24, N]
        stream = torch.cat(acc[t]["stream"], dim=1)           # [13, N]
        c = acc[t]["c"]
        torch.save({"z": z, "c": c, "y": y, "w": w}, out / f"z_{t}.pt")
        v = z - w[0] * c
        zc, sigma = ln_parts(m.cpu(), z)
        cc = c - c.mean()
        vc = v - v.mean(dim=-1, keepdim=True)
        mcpu = m.cpu().float()

        def top1(zz, sig=None):
            with torch.no_grad():
                if sig is None:
                    logits = head(mcpu, zz)
                else:                                          # LN_f with a frozen denominator
                    zzc = zz - zz.mean(dim=-1, keepdim=True)
                    n = zzc / sig * mcpu.norm.weight + mcpu.norm.bias
                    logits = mcpu.head(mcpu.fc_norm(n))
            return (logits.argmax(-1) == y).float().mean().item() * 100

        w0 = float(w[0])                  # the c coefficient: 1 (base, D, E), 0 (A), learned (B, C)
        has_c = w0 != 0.0
        r = dict(
            mode=MODELS[t][1], logged_test_acc1=logged[t], sanity_max_logit_dev=acc[t]["sanity"],
            c_spread_over_images=acc[t]["c_spread"], w=w.tolist(),
            c_norm=float(c.norm()), c_centred_norm=float(cc.norm()),
            v_norm_mean=float(v.norm(dim=-1).mean()), v_norm_sd=float(v.norm(dim=-1).std()),
            z_norm_mean=float(z.norm(dim=-1).mean()),
            cos_c_v_mean=float(F.cosine_similarity(v, c.expand_as(v), dim=-1).mean()),
            v_centred_norm_mean=float(vc.norm(dim=-1).mean()),
            c_share_of_centred_z=float(((w0 * cc).pow(2).sum() / zc.pow(2).sum(dim=-1)).mean()),
            sigma_mean=float(sigma.mean()), sigma_cv=float(sigma.std() / sigma.mean()),
            branch_cls_norm=src_norm[1:].mean(dim=1).tolist(),     # attn_0, mlp_0, ..., mlp_11
            branch_patch_norm=patch.mean(dim=1).tolist(),
            stream_cls_norm=stream.mean(dim=1).tolist(),           # x_0 .. x_12
            top1=dict(as_trained=top1(z),
                      c_toggled=top1(v if has_c else v + c),     # c term removed / added
                      sigma_frozen=top1(z, sig=sigma.mean()),
                      c_scaled={str(al): top1(v + al * c) for al in ALPHAS}),
            c_in_readout=has_c, c_coefficient=w0,
        )
        res[t] = r
        print(f"[{t}] logged {logged[t]:.3f} recomputed {r['top1']['as_trained']:.3f} (sanity {r['sanity_max_logit_dev']:.1e}); "
              f"|c| {r['c_norm']:.1f} |v| {r['v_norm_mean']:.1f} cos {r['cos_c_v_mean']:+.2f} "
              f"c-share {r['c_share_of_centred_z']:.2f} sigma CV {r['sigma_cv']:.3f}; "
              f"c toggled {r['top1']['c_toggled']:.2f} sigma frozen {r['top1']['sigma_frozen']:.2f}", flush=True)
        models[t] = None
    (out / "norms.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "norms.json")
    plot(res, out / "readout_norms.png")


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from models.residual_routing import source_labels

    tags = [t for t in res if t != "meta"]
    col = {"A": "C0", "B": "C1", "C": "C2", "D": "C3", "E": "C4"}
    fig, ax = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    labels = source_labels(12)[1:]
    xs = np.arange(len(labels))
    for t in tags:
        r = res[t]
        kw = dict(color=col.get(t, "0.5"), lw=1.8 if t in col else 1.0,
                  ls="-" if t in col else "--", label=f"{t} {r['mode']}" if t in col else (f"base r (3 seeds)" if t == "r0" else None))
        ax[0, 0].plot(xs, r["branch_cls_norm"], marker="o", ms=3, **kw)
        ax[0, 1].plot(range(13), r["stream_cls_norm"], marker="o", ms=3, **kw)
        ax[1, 0].plot(xs, r["branch_patch_norm"], marker="o", ms=3, **kw)
        al = [float(k) for k in r["top1"]["c_scaled"]]
        ax[1, 1].plot(al, list(r["top1"]["c_scaled"].values()), marker="o", ms=4, **kw)
    ax[0, 0].set_xticks(xs); ax[0, 0].set_xticklabels(labels, rotation=70, fontsize=7)
    ax[0, 0].set_ylabel("mean |branch output| at the CLS row"); ax[0, 0].set_title("how much each branch writes (CLS row)")
    ax[0, 1].set_xlabel("stream x_l (l = blocks applied)"); ax[0, 1].set_ylabel("mean |x_l| at the CLS row")
    ax[0, 1].set_title("residual stream norm at the CLS row")
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(labels, rotation=70, fontsize=7)
    ax[1, 0].set_ylabel("mean |branch output|, patch tokens"); ax[1, 0].set_title("how much each branch writes (patch tokens)")
    ax[1, 1].set_xlabel("alpha in head input v + alpha c"); ax[1, 1].set_ylabel("val top-1")
    ax[1, 1].set_title("eval-only: scaling the constant c in the head input (trained at alpha = 1 or 0)")
    for a_ in ax.flat:
        a_.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("READOUT_NORMS_OUT", DEFAULT_OUT))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--from-json", action="store_true")
    a = p.parse_args()
    if a.from_json:
        plot(json.loads((Path(a.out) / "norms.json").read_text()), Path(a.out) / "readout_norms.png")
    else:
        run(a)
