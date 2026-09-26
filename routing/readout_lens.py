"""Logit lens over depth for the readout-cut arms and the reference arms, epoch-299 checkpoints.

Per model and depth l = 0..12 (x_l = the residual stream after l blocks, CLS row), the model's own
final LayerNorm + head is applied:

  stream lens   logits_l = head(LN_f(x_l)_cls)               -- the usual logit lens; at l = 12 it
                                                                is the model's output for r / p /
                                                                rescaling, but NOT for the readout
                                                                arms, whose head never reads x_12
  window lens   logits_l = head(LN_f(w_0 c + x_l - x_k)_cls)  -- readout arms only, l >= k (k = 9
                                                                for last 3, 8 for last 4): what the
                                                                head reads if the window stopped
                                                                after block l; at l = 12 it IS the
                                                                model's output

Metrics per depth on the full ImageNet val set (50k): top-1 vs truth, cross-entropy vs truth, and
top-1 agreement with the model's own final prediction. One pass over the data, every model per
batch, under the same fp16 autocast as engine.evaluate. Weights: the checkpoint-299-model.pth
analysis dumps (fp16, the readout mixer fp32), loaded strictly, the same files probes/ uses.

    sbatch -J readout-lens routing/sbatch/readout_lens.sh
    python routing/readout_lens.py --from-json      # redraw
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
MODELS = {   # tag -> (run dir, --readout_routing mode, group)
    **{f"r_s{s}": (RES.format(sid=29384839, seed=s), "off", "random") for s in range(3)},
    **{f"p_s{s}": (RES.format(sid=29377576, seed=s), "off", "procedural") for s in range(3)},
    **{f"ftb3b_s{s}": (RES.format(sid=sid, seed=s), "off", "rescale ftb3b")
       for s, sid in enumerate((29388202, 29406778, 29406779))},
    **{f"ftbrho_s{s}": (RES.format(sid=sid, seed=s), "off", "rescale ftbrho")
       for s, sid in enumerate((29453944, 29461252, 29461253))},
    "F": (RUNS.format(name="readout_F_29784360"), "fixed_last3", "readout F (last 3)"),
    "D": (RUNS.format(name="readout_D_29768024"), "fixed_emb_last3", "readout D (last 3 + c)"),
    "A": (RUNS.format(name="readout_A_29725255"), "fixed_last4", "readout A (last 4)"),
    "G": (RUNS.format(name="readout_G_29784361"), "fixed_emb_last4", "readout G (last 4 + c)"),
}
WINDOW = {"fixed_last3": 9, "fixed_emb_last3": 9, "fixed_last4": 8, "fixed_emb_last4": 8}
DEFAULT_OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/readout_lens"


def load(tag, device):
    d, mode, _ = MODELS[tag]
    log = [json.loads(l) for l in open(Path(d) / "log.txt") if l.strip()]
    assert log[-1]["epoch"] == 299, f"{tag}: not finished"
    m = utils.build_model(build_args(["--readout_routing", mode]))
    sd = torch.load(Path(d) / "checkpoint-299-model.pth", map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd and isinstance(sd["model"], dict) else sd
    missing, unexpected = m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (tag, missing[:5], unexpected[:5])
    return m.to(device).eval(), log[-1]["test_acc1"]


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
    L = 13
    st = {t: dict(correct=torch.zeros(L), ce=torch.zeros(L), agree=torch.zeros(L),
                  wcorrect=torch.zeros(L), wce=torch.zeros(L), wagree=torch.zeros(L), final=0.0,
                  sanity=None) for t in tags}
    n, t0 = 0, time.time()
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device)
        n += len(y)
        for t in tags:
            m = models[t]
            mode = MODELS[t][1]
            with torch.no_grad(), torch.cuda.amp.autocast():
                src, _, _ = cls_forward(m, x)                       # [25, B, d] fp32, CLS row
                xs = [src[0]]
                for b in range(12):
                    xs.append(xs[-1] + src[2 * b + 1] + src[2 * b + 2])
                out = m(x).float()
                pred = out.argmax(-1)
                if st[t]["sanity"] is None and mode == "off":
                    st[t]["sanity"] = (head(m, xs[12]).float() - out).abs().max().item()
                for l in range(L):
                    lg = head(m, xs[l]).float()
                    st[t]["correct"][l] += (lg.argmax(-1) == y).sum().item()
                    st[t]["ce"][l] += F.cross_entropy(lg, y, reduction="sum").item()
                    st[t]["agree"][l] += (lg.argmax(-1) == pred).sum().item()
                    if mode in WINDOW and l >= WINDOW[mode]:
                        w0 = float(m.readout_mixer.weight[0])
                        z = w0 * src[0] + (xs[l] - xs[WINDOW[mode]])
                        lw = head(m, z).float()
                        if l == 12 and st[t]["sanity"] is None:
                            st[t]["sanity"] = (lw - out).abs().max().item()
                        st[t]["wcorrect"][l] += (lw.argmax(-1) == y).sum().item()
                        st[t]["wce"][l] += F.cross_entropy(lw, y, reduction="sum").item()
                        st[t]["wagree"][l] += (lw.argmax(-1) == pred).sum().item()
                st[t]["final"] += (pred == y).sum().item()
        if bi % 20 == 0:
            print(f"[pass] batch {bi}/{len(loader)} {time.time() - t0:.0f}s", flush=True)
    res = {"meta": dict(n_images=n, window=WINDOW, models={t: MODELS[t] for t in tags})}
    for t in tags:
        s, mode = st[t], MODELS[t][1]
        r = dict(group=MODELS[t][2], mode=mode, logged_test_acc1=logged[t],
                 final_top1=100 * s["final"] / n, sanity_max_logit_dev=s["sanity"],
                 stream=dict(top1=(100 * s["correct"] / n).tolist(), ce=(s["ce"] / n).tolist(),
                             agree=(100 * s["agree"] / n).tolist()))
        if mode in WINDOW:
            k = WINDOW[mode]
            r["window"] = dict(start=k, top1=(100 * s["wcorrect"][k:] / n).tolist(),
                               ce=(s["wce"][k:] / n).tolist(), agree=(100 * s["wagree"][k:] / n).tolist())
        res[t] = r
        print(f"[{t}] final {r['final_top1']:.3f} (logged {logged[t]:.3f}, sanity {s['sanity']:.1e}) stream top-1 "
              + " ".join(f"{v:.1f}" for v in r["stream"]["top1"])
              + (("  window " + " ".join(f"{v:.1f}" for v in r["window"]["top1"])) if "window" in r else ""), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lens.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "lens.json")
    plot(res, out)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # one hue per group, validated categorical order (routing/readout_vs_arms.py); r neutral gray
    col = {"random": "#8a8984", "readout F (last 3)": "#2a78d6", "readout D (last 3 + c)": "#2a78d6",
           "procedural": "#eb6834", "rescale ftb3b": "#1baf7a", "rescale ftbrho": "#1baf7a",
           "readout A (last 4)": "#eda100", "readout G (last 4 + c)": "#eda100"}
    ls = {"readout D (last 3 + c)": "--", "readout G (last 4 + c)": "--", "rescale ftbrho": "--"}
    groups = {}
    for t, r in res.items():
        if t != "meta":
            groups.setdefault(r["group"], []).append(r)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), layout="constrained")
    depth = np.arange(13)
    for g, rs in groups.items():
        c, st_ = col[g], ls.get(g, "-")
        for k, (metric, a_) in enumerate((("top1", ax[0]), ("ce", ax[1]), ("agree", ax[2]))):
            v = np.array([r["stream"][metric] for r in rs])
            lab = f"{g} ({len(rs)} seeds)" if len(rs) > 1 else g
            a_.plot(depth, v.mean(0), color=c, ls=st_, lw=1.8, marker="o", ms=3.5, label=lab + " - stream lens")
            if len(rs) > 1:
                a_.fill_between(depth, v.min(0), v.max(0), color=c, alpha=0.15, lw=0)
            if "window" in rs[0]:
                k0 = rs[0]["window"]["start"]
                w = np.array([r["window"][metric] for r in rs])
                a_.plot(depth[k0:], w.mean(0), color=c, ls=st_, lw=2.6, marker="s", ms=5, alpha=0.55,
                        label=lab + " - window lens (what the head reads)")
    ax[0].set_ylabel("top-1 vs truth (%)"); ax[0].set_title("logit lens: accuracy by depth")
    ax[1].set_ylabel("cross-entropy vs truth"); ax[1].set_title("logit lens: loss by depth")
    ax[1].set_ylim(0, 8)
    ax[2].set_ylabel("agreement with the model's final prediction (%)")
    ax[2].set_title("logit lens: agreement with the model's output")
    for a_ in ax:
        a_.set_xlabel("x_l: residual stream after l blocks (CLS row)")
        a_.set_xticks(depth)
        a_.grid(color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
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
        plot(json.loads((Path(a.out) / "lens.json").read_text()), Path(a.out))
    else:
        run(a)
