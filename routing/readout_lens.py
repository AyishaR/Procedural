"""Logit lens over depth for the last-3 readout arms (F, D), on the protocol of the cached probes/lens.py results.

The reference arms' logit lens already exists (kempfe/probes/results/lens/<model>.json, probes/lens.py):
full 50k ImageNet val, CLS row of the stream after 0..12 blocks, bf16 autocast, the checkpoint's final
LayerNorm + head; metrics top-1 / top-5 vs truth, agreement with and KL(p_final || p_depth) to the final
prediction, prediction depth (first depth from which the lens agrees with the final prediction at every
later depth). This script computes the same for the readout arms, plus r_s0 as a reproduction check against
its cached JSON.

For the readout arms the model's output is NOT the stream lens at depth 12 (the head reads
z = w_0 c + x_12 - x_k, not x_12), so two lenses are reported:

  stream lens   head(LN_f(x_l)_cls), l = 0..12                -- comparable with the cached curves
  window lens   head(LN_f(w_0 c + x_l - x_k)_cls), l = k..12  -- what the head reads if the window
                                                                stopped after block l; = output at 12
"final" (agree, KL, prediction depth) is always the model's actual output.

    sbatch -J readout-lens routing/sbatch/readout_lens.sh
    python routing/readout_lens.py --from-json      # redraw from lens_<tag>.json + the cached JSONs
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
from routing.readout_norms import cls_forward, head  # noqa: E402

RES = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}"
RUNS = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/runs/{name}/s0"
CACHED = Path("/work/dlc2workfs3/schrodi-procedural/kempfe/probes/results/lens")
MODELS = {   # tag -> (run dir, --readout_routing mode)
    "F": (RUNS.format(name="readout_F_29784360"), "fixed_last3"),
    "D": (RUNS.format(name="readout_D_29768024"), "fixed_emb_last3"),
    "r_s0": (RES.format(sid=29384839, seed=0), "off"),          # reproduction check vs the cached JSON
}
WINDOW = {"fixed_last3": 9, "fixed_emb_last3": 9, "fixed_last4": 8, "fixed_emb_last4": 8}
DEFAULT_OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/readout_lens"
K = 13


def load(tag, device):
    d, mode = MODELS[tag]
    log = [json.loads(l) for l in open(Path(d) / "log.txt") if l.strip()]
    assert log[-1]["epoch"] == 299, f"{tag}: not finished"
    m = utils.build_model(build_args(["--readout_routing", mode]))
    sd = torch.load(Path(d) / "checkpoint-299-model.pth", map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd and isinstance(sd["model"], dict) else sd
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (tag, missing[:5], unexpected[:5])
    return m.to(device).eval(), log[-1]["test_acc1"]


class Acc:
    """Per-depth sums of the probes/lens.py metrics against truth y and the final distribution."""

    def __init__(self, n_depth):
        self.n = 0
        self.top1 = torch.zeros(n_depth); self.top5 = torch.zeros(n_depth)
        self.agree = torch.zeros(n_depth); self.kl = torch.zeros(n_depth)
        self.pd = []

    def add(self, logits, y, final_pred, final_logp):          # logits [k, B, C] fp32
        logp = F.log_softmax(logits, -1)
        pred = logits.argmax(-1)
        self.top1 += (pred == y[None]).float().sum(1).cpu()
        self.top5 += (logits.topk(5, -1).indices == y[None, :, None]).any(-1).float().sum(1).cpu()
        eq = pred == final_pred[None]
        self.agree += eq.float().sum(1).cpu()
        self.kl += (final_logp.exp() * (final_logp[None] - logp)).sum(-1).sum(1).cpu()
        suffix = torch.flip(torch.cumprod(torch.flip(eq.int(), [0]), 0), [0]).bool()
        k = logits.shape[0]
        self.pd.append(torch.where(suffix.any(0), suffix.int().argmax(0), torch.full_like(y, k - 1)).cpu())
        self.n += len(y)

    def result(self, offset=0):
        pd = torch.cat(self.pd).float() + offset
        n = self.n
        return dict(top1=(100 * self.top1 / n).tolist(), top5=(100 * self.top5 / n).tolist(),
                    agree=(100 * self.agree / n).tolist(), kl=(self.kl / n).tolist(),
                    pred_depth_mean=float(pd.mean()), frac_decided_by_b8=float((pd <= 9).float().mean()),
                    pred_depth_hist=torch.bincount(pd.long(), minlength=K).tolist())


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
    acc = {t: dict(stream=Acc(K), window=Acc(K - WINDOW.get(MODELS[t][1], K)) if MODELS[t][1] in WINDOW else None,
                   final=0, sanity=None) for t in tags}
    t0 = time.time()
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device)
        for t in tags:
            m, mode = models[t], MODELS[t][1]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # as probes/extract_features
                src, _, _ = cls_forward(m, x)
                xs = [src[0]]
                for b in range(12):
                    xs.append(xs[-1] + src[2 * b + 1] + src[2 * b + 2])
                stream = torch.stack([head(m, h).float() for h in xs])            # [13, B, C]
                if mode in WINDOW:
                    k = WINDOW[mode]
                    w0 = float(m.readout_mixer.weight[0])
                    window = torch.stack([head(m, w0 * src[0] + (xs[l] - xs[k])).float() for l in range(k, K)])
                    final = window[-1]
                else:
                    window, final = None, stream[-1]
                if acc[t]["sanity"] is None:
                    acc[t]["sanity"] = (final - m(x).float()).abs().max().item()
            fpred, flogp = final.argmax(-1), F.log_softmax(final, -1)
            acc[t]["stream"].add(stream, y, fpred, flogp)
            if window is not None:
                acc[t]["window"].add(window, y, fpred, flogp)
            acc[t]["final"] += (fpred == y).sum().item()
        if bi % 20 == 0:
            print(f"[pass] batch {bi}/{len(loader)} {time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in tags:
        mode = MODELS[t][1]
        n = acc[t]["stream"].n
        r = dict(model=t, mode=mode, logged_test_acc1=logged[t], final_top1=100 * acc[t]["final"] / n,
                 sanity_max_logit_dev_bf16=acc[t]["sanity"], n_images=n, autocast="bf16",
                 logit_lens=acc[t]["stream"].result())
        if mode in WINDOW:
            r["window_lens"] = dict(start=WINDOW[mode], **acc[t]["window"].result(offset=WINDOW[mode]))
        (out / f"lens_{t}.json").write_text(json.dumps(r, indent=1))
        msg = f"[{t}] final {r['final_top1']:.3f} (logged {logged[t]:.3f}); stream top-1 " + \
              " ".join(f"{v:.1f}" for v in r["logit_lens"]["top1"])
        if t == "r_s0" and (CACHED / "r_s0.json").exists():
            c = json.load(open(CACHED / "r_s0.json"))["logit_lens"]
            dev = {m_: max(abs(u - v) for u, v in zip(r["logit_lens"][m_], c[m_])) for m_ in ("top1", "agree", "kl")}
            r["reproduction_vs_cached"] = dev
            (out / f"lens_{t}.json").write_text(json.dumps(r, indent=1))
            msg += f"  | max |new - cached|: {dev}"
        print(msg, flush=True)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # hues as in routing/readout_vs_arms.py (validated categorical order; random neutral gray)
    ref = {"random (r)": ("r", "#8a8984", "-"), "procedural (p)": ("p", "#eb6834", "-"),
           "rescale ftb3b": ("ftb3b", "#1baf7a", "-"), "rescale ftbrho": ("ftbrho", "#1baf7a", "--")}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6), layout="constrained")
    depth = np.arange(K)
    metrics = (("top1", "top-1 vs truth (percent)"), ("agree", "agreement with the model's output (percent)"),
               ("kl", "KL divergence: model output to lens"))
    for label, (arm, col, ls) in ref.items():
        runs = [json.load(open(CACHED / f"{arm}_s{s}.json"))["logit_lens"] for s in range(3)]
        for (mk, _), a_ in zip(metrics, ax):
            v = np.array([r[mk] for r in runs])
            a_.plot(depth, v.mean(0), color=col, ls=ls, lw=1.8, marker="o", ms=3.5, label=f"{label}, 3 seeds")
            a_.fill_between(depth, v.min(0), v.max(0), color=col, alpha=0.15, lw=0)
    for t, lab, ls in (("F", "readout F (last 3)", "-"), ("D", "readout D (last 3 + c)", "--")):
        f = out / f"lens_{t}.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        for (mk, _), a_ in zip(metrics, ax):
            a_.plot(depth, r["logit_lens"][mk], color="#2a78d6", ls=ls, lw=1.8, marker="o", ms=3.5,
                    label=f"{lab}: stream lens")
            w = r["window_lens"]
            a_.plot(depth[w["start"]:], w[mk], color="#2a78d6", ls=ls, lw=3.2, marker="s", ms=6, alpha=0.45,
                    label=f"{lab}: window lens (the head's input)")
    for (mk, yl), a_ in zip(metrics, ax):
        a_.set_ylabel(yl)
        a_.set_xlabel("residual stream after l blocks (CLS row)")
        a_.set_xticks(depth)
        a_.grid(color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    ax[0].set_title("logit lens: accuracy by depth")
    ax[1].set_title("logit lens: agreement with the final prediction")
    ax[2].set_title("logit lens: KL to the final prediction")
    ax[0].legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.savefig(out / "readout_lens.png", dpi=140)
    print("wrote", out / "readout_lens.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("READOUT_LENS_OUT", DEFAULT_OUT))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--from-json", action="store_true")
    a = p.parse_args()
    if a.from_json:
        plot(Path(a.out))
    else:
        run(a)
