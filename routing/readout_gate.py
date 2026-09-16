"""Pre-training gate for --readout_routing on the ViT (models/readout_routing.py).

Nothing trains until this passes. It builds a baseline and the three readout arms from the SAME
seed through the training code path (utils.build_model) and hard-fails unless:

  1  weights      every arm adds exactly readout_mixer.weight (25 floats) and shares every other
                  tensor bit for bit; the init vectors are the arm's definition;
  2  fp32         arm C (learn_ones) reproduces the baseline logits under torch.equal;
                  arm A (fixed_last4) equals the hand computation head(fc_norm(pool(norm(
                  x_after_block_11 - x_after_block_7)))) from the baseline's own stream to 1e-5;
                  arm B (learn_last4) equals arm A;
  3  fp16 AMP     the same under torch.cuda.amp.autocast, which is what this repo trains with;
  4  gradients    fp32: arm C's backbone gradients agree with the baseline's to summation order
                  (the graph differs -- a direct edge per branch output -- so not bit for bit);
                  arm B/C have finite non-zero weight gradients; one real step through
                  NativeScalerWithGradNormCount leaves the GradScaler in the same state on every
                  arm and buckets readout_mixer in get_layer_grad_norms;
  5  collectors   utils.HookCollector runs on arm C (Block.forward still runs) and its block
                  activations equal the baseline's -- model_analyse's blk_acc / ece stay defined;
  6  cost         median fwd+bwd time and peak memory at the real per-GPU batch for base / A / B / C.

Writes <out>/gate.json and <out>/gate.png and prints GATE PASSED as its last line.
Run it on a GPU (the login node has none):
    sbatch -J readout-gate --export=ALL,GATE_SCRIPT=routing/readout_gate.py routing/sbatch/gate.sh
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=FutureWarning)

import utils  # noqa: E402
from models.readout_routing import readout_init  # noqa: E402
from models.residual_routing import source_labels  # noqa: E402
from routing.gate import (  # noqa: E402
    BASE_ARGV, _forward, _maxdev, amp_step_grads, branch_outputs, build_args, check_function,
    plain_grads, probe_batches, timing)

DEFAULT_OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/gate_readout"
ARMS = {"A": "fixed_last4", "B": "learn_last4", "C": "learn_ones"}


def build(seed, mode="off"):
    """Exactly the training construction path: seed, then utils.build_model."""
    a = build_args(["--readout_routing", mode])
    torch.manual_seed(seed)
    return utils.build_model(a)


def check_weights(base, arms, depth):
    sb = base.state_dict()
    out = {}
    for tag, m in arms.items():
        sd = m.state_dict()
        extra = set(sd) - set(sb)
        assert extra == {"readout_mixer.weight"}, f"{tag}: unexpected keys {sorted(extra)}"
        assert set(sb) <= set(sd), f"{tag}: a baseline key is missing"
        bad = [k for k in sb if not torch.equal(sb[k], sd[k])]
        assert not bad, f"{tag}: same-seed backbone differs in {bad[:5]} -- do not train"
        w = sd["readout_mixer.weight"]
        ref, trainable = readout_init(ARMS[tag], depth)
        assert w.shape == (2 * depth + 1,) and torch.equal(w.cpu(), ref), f"{tag}: init vector"
        assert m.readout_mixer.trainable == trainable
        n_params = sum(p.numel() for n, p in m.named_parameters() if n.startswith("readout_mixer."))
        assert n_params == (2 * depth + 1 if trainable else 0), (tag, n_params)
        out[tag] = dict(n_sources=2 * depth + 1, trainable=trainable, n_params=n_params,
                        weight=w.cpu().tolist())
    return out


def stream_states(base, x, amp):
    """[x entering block 0, x after block 0, ..., x after block L-1] of the vanilla loop."""
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            h = base.norm_pre(base._pos_embed(base.patch_embed(x)))
            states = [h]
            for blk in base.blocks:
                h = blk(h)
                states.append(h)
    return states


def readout(base, mixed, amp):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            z = base.norm(mixed)
            z = z[:, base.num_prefix_tokens:].mean(dim=1) if base.global_pool == "avg" else z[:, 0]
            return base.head(base.fc_norm(z))


def check_fixed_arm(base, A, B, batches, amp, tol, label):
    """A == head(fc_norm(pool(norm(x12 - x8)))) built from the baseline's stream; B == A."""
    depth = len(base.blocks)
    dev, dev_hist, dev_ab = 0.0, 0.0, 0.0
    for x, _ in batches:
        states = stream_states(base, x, amp)
        ref = readout(base, states[depth] - states[depth - 4], amp)
        hist = branch_outputs(base, x, amp)
        ref2 = readout(base, sum(h.float() for h in hist[2 * (depth - 4):]).to(states[0].dtype), amp)
        with torch.no_grad():
            la, lb = _forward(A, x, amp), _forward(B, x, amp)
        assert torch.isfinite(la).all(), f"{label}: non-finite arm A output"
        dev, dev_hist = max(dev, _maxdev(la, ref)), max(dev_hist, _maxdev(la, ref2))
        dev_ab = max(dev_ab, _maxdev(la, lb))
        assert torch.equal(la, lb), f"{label}: B must be A at init (max dev {dev_ab:.3e})"
    assert dev <= tol, f"{label}: arm A deviates from the hand computation by {dev:.3e} > {tol:.1e}"
    assert dev_hist <= tol, f"{label}: arm A deviates from the summed branch outputs by {dev_hist:.3e}"
    return dict(max_dev_vs_stream_diff=dev, max_dev_vs_summed_branches=dev_hist, B_equals_A=True)


def grad_dev(ga, gb, names):
    worst, where = 0.0, None
    for n in names:
        d = (ga[n] - gb[n]).abs().max().item() / (ga[n].abs().max().item() + 1e-12)
        if d > worst:
            worst, where = d, n
    return worst, where


def check_collector(base, C, x):
    """HookCollector on the mixed model: Block.forward runs, so the per-block stream is there."""
    states = stream_states(base, x, False)
    C.eval()
    with utils.HookCollector(C) as acts:
        with torch.no_grad():
            out = C(x)
    with torch.no_grad():
        assert torch.equal(out, base(x)), "arm C under HookCollector no longer equals base"
    for i in range(len(C.blocks)):
        assert torch.equal(acts[i]["blk"], states[i + 1]), f"block {i} activation differs"
        assert torch.equal(acts[i]["inp"], states[i]), f"block {i} input differs"
    from models.residual_routing import has_hooks
    assert not has_hooks(C), "hooks left behind"
    return dict(layers=len(acts), keys=sorted(acts[0]))


# ---------------------------------------------------------------------------------------
def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = res["meta"]["depth"]
    labels = source_labels(depth)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), layout="constrained")
    xs = np.arange(len(labels))
    for i, (tag, col) in enumerate((("A", "C0"), ("B", "C1"), ("C", "C2"))):
        w = np.array(res["weights"][tag]["weight"])
        ax[0].bar(xs + (i - 1) * 0.27, w, width=0.27, color=col, label=f"{tag} = {ARMS[tag]}")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, rotation=70, fontsize=7)
    ax[0].set_ylabel("readout weight at init"); ax[0].legend(); ax[0].set_title("readout rows at init")
    t = res.get("timing", {})
    if t:
        tags = list(t)
        ax[1].bar(tags, [t[k]["ms_micro"] for k in tags], color=["0.5", "C0", "C1", "C2"][:len(tags)])
        for i, k in enumerate(tags):
            ax[1].text(i, t[k]["ms_micro"], f"{t[k]['ms_micro']:.1f}", ha="center", va="bottom", fontsize=8)
        ax[1].set_ylabel("ms per fwd+bwd micro-step (batch %d)" % res["meta"]["batch"])
        ax[1].set_title("cost, eager fp16 AMP")
        ax[2].bar(tags, [t[k]["peak_gb"] for k in tags], color=["0.5", "C0", "C1", "C2"][:len(tags)])
        ax[2].set_ylabel("peak GB allocated"); ax[2].set_title("memory")
    fig.suptitle("readout-routing gate -- " + res["meta"]["device"])
    fig.savefig(path, dpi=130)
    print("wrote", path)


def main_gate(a):
    if a.from_json:
        out = Path(a.out)
        plot(json.loads((out / "gate.json").read_text()), out / "gate.png")
        return
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "the gate needs a GPU; submit routing/sbatch/gate.sh"
    args = build_args()
    args.crop_pct = None
    batches = probe_batches(args, a.n_batches, a.batch, device, a.probe)
    print(f"[gate] {len(batches)} probe batches of {a.batch} ({a.probe}) on {torch.cuda.get_device_name(0)}")

    base = build(a.seed).to(device).eval()
    arms = {tag: build(a.seed, mode).to(device).eval() for tag, mode in ARMS.items()}
    A, B, C = arms["A"], arms["B"], arms["C"]
    depth = len(base.blocks)
    res = {"meta": dict(seed=a.seed, batch=a.batch, n_batches=a.n_batches, probe=a.probe,
                        device=torch.cuda.get_device_name(0), torch=torch.__version__,
                        depth=depth, argv=" ".join(BASE_ARGV), arms=ARMS)}

    # 1 -----------------------------------------------------------------------------
    res["weights"] = check_weights(base, arms, depth)
    print(f"[1/6] weights   OK  " + "; ".join(f"{t}: {v['n_params']} params, trainable={v['trainable']}"
                                             for t, v in res["weights"].items()))

    # 2 -----------------------------------------------------------------------------
    res["fp32"] = {
        "C": check_function(base, C, batches, False, True, 0.0, 0.0, "fp32 C"),
        "A": check_fixed_arm(base, A, B, batches, False, a.tol_fixed, "fp32 A"),
    }
    print(f"[2/6] fp32      OK  C exact; A vs hand {res['fp32']['A']['max_dev_vs_stream_diff']:.2e}")

    # 3 -----------------------------------------------------------------------------
    res["fp16"] = {
        "C": check_function(base, C, batches, True, True, 0.0, 0.0, "fp16 C"),
        "A": check_fixed_arm(base, A, B, batches, True, a.tol_fixed_amp, "fp16 A"),
    }
    print(f"[3/6] fp16 AMP  OK  C exact under autocast; A vs hand "
          f"{res['fp16']['A']['max_dev_vs_stream_diff']:.2e}")

    # 4 -----------------------------------------------------------------------------
    x, y = batches[0]
    lb32, gb32 = plain_grads(build(a.seed).to(device), x, y)
    grads32, losses32 = {}, {}
    for tag, mode in ARMS.items():
        losses32[tag], grads32[tag] = plain_grads(build(a.seed, mode).to(device), x, y)
    shared = [n for n in gb32 if n in grads32["C"]]
    dC, wC = grad_dev(gb32, grads32["C"], shared)
    assert dC <= a.tol_grad, f"fp32: arm C gradients differ from base by {dC:.2e} at {wC} -- do not train"
    assert abs(losses32["C"] - lb32) == 0.0, "arm C loss differs from base in fp32"
    wgrad = {}
    for tag in ("B", "C"):
        g = grads32[tag]["readout_mixer.weight"]
        assert torch.isfinite(g).all() and (g != 0).all(), f"{tag}: weight grad not finite / has zeros"
        wgrad[tag] = g.cpu().tolist()
    assert "readout_mixer.weight" not in grads32["A"]
    # a step through the trainer's machinery (fp16 autocast + GradScaler)
    lb, gbse, sb, nb, lnb = amp_step_grads(build(a.seed).to(device), x, y)
    step = {}
    for tag, mode in ARMS.items():
        l, g, s, n, ln = amp_step_grads(build(a.seed, mode).to(device), x, y)
        assert s == sb, f"GradScaler state diverged: base {sb}, {tag} {s}"
        assert np.isfinite(float(n)), f"{tag}: non-finite grad norm"
        buckets = [k for k in (ln or {}) if "readout_mixer" in k]
        if tag != "A":
            assert buckets, f"{tag}: readout_mixer missing from get_layer_grad_norms buckets"
        d, w = grad_dev(gbse, g, [k for k in gbse if k in g])
        step[tag] = dict(loss=l, scale=s, grad_norm=float(n), dev_vs_base=d, worst=w,
                         buckets=buckets,
                         weight_grad_norm=float(ln[buckets[0]]) if buckets else None)
    assert step["C"]["dev_vs_base"] <= a.tol_step, \
        f"fp16 step: arm C {step['C']['dev_vs_base']:.2e} exceeds the fp16 noise band"
    res["grads"] = dict(fp32=dict(base_loss=lb32, losses=losses32, C_dev=dC, C_worst=wC,
                                  weight_grad=wgrad),
                        fp16_step=dict(base_loss=lb, base_scale=sb, base_grad_norm=float(nb),
                                       arms=step))
    print(f"[4/6] grads     OK  fp32 C dev {dC:.2e} ({wC}); fp16 step C dev "
          f"{step['C']['dev_vs_base']:.2e}; scale {sb:g} on all arms; "
          f"|grad w| B {step['B']['weight_grad_norm']:.3e} C {step['C']['weight_grad_norm']:.3e} "
          f"vs total {float(nb):.3e}")

    # 5 -----------------------------------------------------------------------------
    res["collector"] = check_collector(base, C, x)
    print(f"[5/6] collector OK  HookCollector on arm C: {res['collector']['layers']} blocks, "
          f"block activations equal to base")

    # 6 -----------------------------------------------------------------------------
    res["timing"] = {}
    if not a.no_timing:
        makers = [("base", lambda: build(a.seed))] + \
                 [(tag, (lambda m=mode: build(a.seed, m))) for tag, mode in ARMS.items()]
        for tag, mk in makers:
            res["timing"][tag] = timing(mk, batches, a.n_timed, device)
            print(f"[6/6] timing    {tag:5s} {res['timing'][tag]['ms_micro']:7.1f} ms   "
                  f"peak {res['timing'][tag]['peak_gb']:6.2f} GB")
        b = res["timing"]["base"]["ms_micro"]
        for tag in ARMS:
            res["timing"][tag]["overhead_frac"] = res["timing"][tag]["ms_micro"] / b - 1
        print("        overhead: " + "  ".join(f"{t} +{100 * res['timing'][t]['overhead_frac']:.1f}%"
                                              for t in ARMS))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'gate.json'}")
    plot(res, out / "gate.png")
    print("GATE PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get("ROUTING_GATE_OUT", DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=128, help="the real per-GPU batch")
    p.add_argument("--n-batches", type=int, default=2)
    p.add_argument("--probe", choices=("imnet", "random"), default="imnet")
    p.add_argument("--n-timed", type=int, default=10)
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--from-json", action="store_true",
                   help="redraw gate.png from an existing gate.json (no GPU needed)")
    p.add_argument("--tol-fixed", type=float, default=1e-4, help="fp32 arm A vs hand computation")
    p.add_argument("--tol-fixed-amp", type=float, default=5e-2, help="autocast arm A vs hand")
    p.add_argument("--tol-grad", type=float, default=1e-5, help="fp32 arm C backbone grads (rel)")
    p.add_argument("--tol-step", type=float, default=5e-2, help="fp16 step noise band")
    main_gate(p.parse_args())
