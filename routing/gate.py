"""Pre-training gate for --resid_routing on the ViT (models/residual_routing.py).

Nothing trains until this passes. It builds a baseline and a routed vit_base from the SAME seed
through the training code path (utils.build_model) and hard-fails unless:

  1  weights      the routed model adds exactly the router keys and shares every other tensor
                  bit for bit, with all 325 coefficients exactly 1;
  2  fp32         routing_impl="loop" reproduces the baseline logits under torch.equal (same
                  summation order), "buf" to 1e-5;
  3  fp16 AMP     the same under torch.cuda.amp.autocast, which is what this repo trains with;
  4  gradients    in fp32 the routed backward reproduces the baseline's gradients (only the
                  summation order differs); under fp16 autocast the matmul backward itself
                  runs in half precision, so that deviation is measured, not gated, and one
                  real step through NativeScalerWithGradNormCount must leave the GradScaler
                  in the same state for every arm;
  5  loop vs buf  with random coefficients and a masked edge -- the ONLY test the Triton kernels
                  ever get, since the CPU suite exercises the eager fallbacks;
  6  init profile the 24 branch outputs are equal to the baseline's tensor for tensor, and the
                  per-consumer ||branch|| / ||mix|| profile is recorded;
  7  fp16 range   the coefficient-weighted products stay far inside fp16 range and no arm skips
                  more optimizer steps than the other, at init and at 2x the init coefficients;
  8  cost         median fwd+bwd time and peak memory at the real per-GPU batch, eager (this
                  repo never calls torch.compile), for base / loop / buf -- routing_impl for the
                  training run is chosen from these numbers.

Writes <out>/gate.json and <out>/gate.png and prints GATE PASSED as its last line.
Run it on a GPU (the login node has none): see routing/sbatch/gate.sh.
"""

import argparse
import json
import os
import statistics
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
import main  # noqa: E402
import utils  # noqa: E402
from models.residual_routing import consumer_labels  # noqa: E402
from utils import NativeScalerWithGradNormCount  # noqa: E402

DEFAULT_OUT = "/work/dlc2workfs3/schrodi-procedural/kempfe/routing/gate"
# the base arm's recipe (results/imnet_base/results_IMNET_BASE_29384839, seeds 0-2)
BASE_ARGV = ("--model vit_base --seed 0 --lr 2e-3 --min_lr 1e-6 --epochs 300 --warmup_epochs 50 "
             "--batch_size 128 --total_batch_size 4096 --update_freq 8 --use_amp true "
             "--data_set IMNET --data_path /data/datasets/ILSVRC2012 --weight_decay 0.05").split()


def build_args(extra=()):
    p = argparse.ArgumentParser("routing-gate", parents=[main.get_args_parser()])
    return p.parse_args(BASE_ARGV + list(extra))


def build(seed, routed, impl="loop", bound=0.0):
    """Exactly the training construction path: seed, then utils.build_model."""
    a = build_args(["--resid_routing", "true" if routed else "false",
                    "--routing_impl", impl, "--routing_bound", str(bound)])
    torch.manual_seed(seed)
    return utils.build_model(a)


def probe_batches(args, n_batches, batch, device, kind):
    """Fixed, seeded probe batches. Real val images by default -- the fp16 checks are about
    realistic activation magnitudes -- with a random fallback."""
    if kind == "random":
        g = torch.Generator().manual_seed(1234)
        return [(torch.randn(batch, 3, args.input_size, args.input_size, generator=g).to(device),
                 torch.randint(0, 1000, (batch,), generator=g).to(device))
                for _ in range(n_batches)]
    dataset, _ = ds.build_dataset(is_train=False, args=args)
    g = torch.Generator().manual_seed(1234)
    idx = torch.randperm(len(dataset), generator=g)[: n_batches * batch].tolist()
    out = []
    for b in range(n_batches):
        items = [dataset[i] for i in idx[b * batch:(b + 1) * batch]]
        out.append((torch.stack([it[0] for it in items]).to(device),
                    torch.tensor([it[1] for it in items]).to(device)))
    return out


def _maxdev(a, b):
    return (a.float() - b.float()).abs().max().item()


def _forward(model, x, amp):
    if amp:
        with torch.cuda.amp.autocast():
            return model(x)
    return model(x)


# ---------------------------------------------------------------------------------------
def check_weights(base, routed, depth, tol=0.0):
    sb, sr = base.state_dict(), routed.state_dict()
    router_keys = {k for k in sr if k.startswith("router.")}
    n = 2 * depth + 1
    assert router_keys == {"router.mask"} | {f"router.theta.{k}" for k in range(n)}, \
        f"unexpected router key set: {sorted(router_keys)}"
    assert set(sr) - router_keys == set(sb), "routed model changed a non-router key"
    bad = [k for k in sb if not torch.equal(sb[k], sr[k])]
    assert not bad, f"same-seed backbone differs in {bad[:5]} -- do not train"
    C = routed.router.coefficient_matrix()
    tril = torch.tril(torch.ones_like(C)) > 0
    dev = (C[tril] - 1).abs().max().item()
    assert dev <= tol, f"coefficients are not 1 at init (max dev {dev})"
    n_params = sum(p.numel() for p in routed.router.parameters())
    assert n_params == n * (n + 1) // 2, n_params
    return dict(n_consumers=n, n_router_params=n_params, n_router_keys=len(router_keys),
                max_coef_dev=dev)


def check_function(base, routed, batches, amp, exact, tol_logit, tol_loss, label):
    dl, dloss, losses = 0.0, 0.0, []
    for x, y in batches:
        with torch.no_grad():
            lb, lr = _forward(base, x, amp), _forward(routed, x, amp)
            cb = F.cross_entropy(lb.float(), y)
            cr = F.cross_entropy(lr.float(), y)
        assert torch.isfinite(lr).all() and torch.isfinite(cr), f"{label}: non-finite output"
        if exact:
            assert torch.equal(lb, lr), (
                f"{label}: the loop path must reproduce the baseline logits exactly "
                f"(max dev {_maxdev(lb, lr):.3e}) -- the summation order has changed")
        dl = max(dl, _maxdev(lb, lr))
        dloss = max(dloss, abs(cb.item() - cr.item()))
        losses.append((cb.item(), cr.item()))
    assert dl <= tol_logit, f"{label}: max logit dev {dl:.3e} > {tol_logit:.1e} -- do not train"
    assert dloss <= tol_loss, f"{label}: max loss dev {dloss:.3e} > {tol_loss:.1e} -- do not train"
    return dict(max_logit_dev=dl, max_loss_dev=dloss, exact=bool(exact),
                base_loss=losses[0][0], routed_loss=losses[0][1])


def plain_grads(model, x, y, amp=False):
    """One forward/backward, no GradScaler -- used for the fp32 correctness statement."""
    model.train()
    model.zero_grad(set_to_none=True)
    if amp:
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(model(x), y)
    else:
        loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (loss.detach().float().item(),
            {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None})


def amp_step_grads(model, x, y, clip_grad=None):
    """One optimizer step through the trainer's own machinery (fp16 autocast + GradScaler)."""
    model.train()
    model.zero_grad(set_to_none=True)
    scaler = NativeScalerWithGradNormCount()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-8)
    with torch.cuda.amp.autocast():
        loss = F.cross_entropy(model(x), y)
    norm, layer_norms = scaler(loss, opt, clip_grad=clip_grad,
                               parameters=model.named_parameters(), update_grad=True)
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return loss.detach().float().item(), grads, float(scaler._scaler.get_scale()), norm, layer_norms


def grad_dev(ga, gb, names):
    """Max relative deviation over shared parameters, thetas normalised by their shared scale
    (consumer 1's entry is structurally ~0 -- see tests/test_residual_routing.py)."""
    tscale = max((g.abs().max().item() for n, g in ga.items() if n.startswith("router.theta.")),
                 default=0.0)
    worst, where = 0.0, None
    for n in names:
        scale = tscale if n.startswith("router.theta.") else ga[n].abs().max().item()
        d = (ga[n] - gb[n]).abs().max().item() / (scale + 1e-12)
        if d > worst:
            worst, where = d, n
    return worst, where


def timing(make, batches, n_timed, device):
    """Median fwd+bwd ms per micro-step and peak memory, eager fp16 -- what training does."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = make().to(device)
    model.train()
    scaler = NativeScalerWithGradNormCount()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-8)
    x, y = batches[0]
    times, first_loss, last_loss = [], None, None
    for it in range(n_timed + 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(model(x), y)
        scaler(loss, opt, clip_grad=None, parameters=model.named_parameters(), update_grad=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        if it == 0:
            first_loss = loss.item()
            assert np.isfinite(first_loss), "non-finite loss on the first compiled step"
            for n, p in model.named_parameters():
                assert p.grad is None or torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
        if it >= 3:
            times.append(dt)
        last_loss = loss.item()
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    del model, opt, scaler
    torch.cuda.empty_cache()
    return dict(ms_micro=statistics.median(times), ms_micro_min=min(times),
                peak_gb=peak, reserved_gb=reserved, first_loss=first_loss,
                last_loss=last_loss, n_timed=len(times))


def branch_outputs(model, x, amp=False):
    """The 2*depth branch outputs, hooked on drop_path1 / drop_path2 (the last module of each
    branch) -- identical hook points for the vanilla and the routed model."""
    out, handles = [], []
    for blk in model.blocks:
        for mod in (blk.drop_path1, blk.drop_path2):
            handles.append(mod.register_forward_hook(lambda _m, _a, o: out.append(o.detach())))
    with torch.no_grad():
        _forward(model, x, amp)
    for h in handles:
        h.remove()
    return out


def init_profile(base, routed, x):
    """Branch outputs must match the baseline tensor for tensor at init; record the routed
    ||branch|| / ||mix|| profile per consumer (the well-defined analogue of the norm ratios)."""
    import utils as U
    with U.RoutedHookCollector(routed) as acts:
        rb = branch_outputs(routed, x)
        ratios, mix_norms = [], []
        for layer in range(len(routed.blocks)):
            for bkey, mkey in (("attn_out", "attn_mix"), ("mlp_out", "mlp_mix")):
                z = torch.norm(acts[layer][bkey].float(), dim=-1)
                m = torch.norm(acts[layer][mkey].float(), dim=-1)
                ratios.append((z / (m + 1e-8)).mean().item())
                mix_norms.append(m.mean().item())
    bb = branch_outputs(base, x)
    bad = [k for k, (a, b) in enumerate(zip(bb, rb)) if not torch.equal(a, b)]
    assert not bad, f"branch outputs {bad[:5]} differ from the baseline at init -- do not train"
    return dict(ratio=ratios, mix_norm=mix_norms, n_branches=len(rb))


def fp16_range_probe(routed, batches, n_steps, scale_coeffs=None):
    """fp16 headroom and GradScaler behaviour. In `loop` the only new fp16 op is the per-edge
    product a[j] * history[j]; `buf` accumulates the mix in fp32 and never performs it."""
    model = routed
    has_router = hasattr(model, "router")     # the baseline arm runs through here as the control
    if scale_coeffs is not None:
        assert has_router
        C = model.router.coefficient_matrix() * scale_coeffs
        model.router.set_coefficient_matrix(torch.tril(C))
    model.train()
    scaler = NativeScalerWithGradNormCount()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-8)
    scales, skips, worst_mix = [], 0, 0.0
    mixes = []
    h = (model.router.register_forward_hook(
        lambda _m, _a, o: mixes.append(o.detach().float().abs().max().item()))
        if has_router else None)
    for i in range(n_steps):
        x, y = batches[i % len(batches)]
        before = scaler._scaler.get_scale()
        model.zero_grad(set_to_none=True)
        mixes.clear()
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(model(x), y)
        assert np.isfinite(loss.item()), f"non-finite loss at step {i}"
        scaler(loss, opt, clip_grad=None, parameters=model.named_parameters(), update_grad=True)
        after = scaler._scaler.get_scale()
        skips += int(after < before)
        scales.append(after)
        worst_mix = max(worst_mix, max(mixes) if mixes else 0.0)
        for n, p in model.named_parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
    if h is not None:
        h.remove()
    return dict(steps=n_steps, skipped=skips, final_scale=scales[-1], min_scale=min(scales),
                max_abs_mix=worst_mix, frac_of_fp16_max=worst_mix / 65504.0)


# ---------------------------------------------------------------------------------------
def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # constrained layout: fig.colorbar below is incompatible with a later tight_layout()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4), layout="constrained")
    depth = (res["weights"]["n_consumers"] - 1) // 2
    labels = consumer_labels(depth)

    r = res["init_profile"]["ratio"]
    ax[0].plot(range(len(r)), r, "o-", ms=3)
    ax[0].set_xticks(range(0, len(r), max(1, len(r) // 8)))
    ax[0].set_xticklabels(labels[:len(r):max(1, len(r) // 8)], rotation=45, ha="right", fontsize=7)
    ax[0].set_title("init ||branch|| / ||mix|| per consumer")
    ax[0].set_ylabel("ratio")

    C = np.array(res["coefficient_matrix"])
    im = ax[1].imshow(np.where(np.tril(np.ones_like(C)) > 0, C, np.nan),
                      cmap="RdBu_r", vmin=0.0, vmax=2.0, origin="upper", aspect="auto")
    ax[1].set_title("coefficients at init (white = 1)")
    ax[1].set_xlabel("source")
    ax[1].set_ylabel("consumer")
    fig.colorbar(im, ax=ax[1])

    arms = [a for a in ("base", "loop", "buf") if a in res["timing"]]
    ms = [res["timing"][a]["ms_micro"] for a in arms]
    ax[2].bar(arms, ms, color=["0.4", "C0", "C1"][:len(arms)])
    for i, (a, v) in enumerate(zip(arms, ms)):
        over = "" if a == "base" else f"\n+{100 * (v / ms[0] - 1):.0f}%"
        ax[2].text(i, v, f"{v:.0f} ms{over}", ha="center", va="bottom", fontsize=8)
    ax[2].set_title(f"fwd+bwd per micro-step (batch {res['meta']['batch']}, eager fp16)")
    ax[2].set_ylabel("ms")
    ax[2].set_ylim(0, max(ms) * 1.35)

    gb = [res["timing"][a]["peak_gb"] for a in arms]
    ax[3].bar(arms, gb, color=["0.4", "C0", "C1"][:len(arms)])
    for i, v in enumerate(gb):
        ax[3].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax[3].axhline(141, ls=":", c="r", lw=1)
    ax[3].set_title("peak allocated (H200 = 141 GB)")
    ax[3].set_ylabel("GB")
    ax[3].set_ylim(0, max(max(gb) * 1.35, 5))

    fig.suptitle("residual routing gate -- vit_base, " + res["meta"]["probe"] + " probe")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


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

    base = build(a.seed, False).to(device).eval()
    loop = build(a.seed, True, "loop").to(device).eval()
    buf = build(a.seed, True, "buf").to(device).eval()
    depth = len(base.blocks)

    res = {"meta": dict(seed=a.seed, batch=a.batch, n_batches=a.n_batches, probe=a.probe,
                        device=torch.cuda.get_device_name(0), torch=torch.__version__,
                        depth=depth, argv=" ".join(BASE_ARGV))}

    # 1 -----------------------------------------------------------------------------
    res["weights"] = check_weights(base, loop, depth)
    res["coefficient_matrix"] = loop.router.coefficient_matrix().cpu().tolist()
    print(f"[1/8] weights   OK  {res['weights']}")

    # 2 -----------------------------------------------------------------------------
    res["fp32"] = {
        "loop": check_function(base, loop, batches, False, True, 0.0, 0.0, "fp32 loop"),
        "buf": check_function(base, buf, batches, False, False, a.tol_buf_logit, a.tol_buf_loss, "fp32 buf"),
    }
    print(f"[2/8] fp32      OK  loop exact; buf logit {res['fp32']['buf']['max_logit_dev']:.2e}")

    # 3 -----------------------------------------------------------------------------
    res["fp16"] = {
        "loop": check_function(base, loop, batches, True, False, a.tol_amp_loop, 1e-6, "fp16 loop"),
        "buf": check_function(base, buf, batches, True, False, a.tol_amp_buf_logit, a.tol_amp_buf_loss, "fp16 buf"),
    }
    with torch.no_grad():
        exact16 = all(torch.equal(_forward(base, x, True), _forward(loop, x, True))
                      for x, _ in batches[:1])
    res["fp16"]["loop"]["exact_under_autocast"] = bool(exact16)
    print(f"[3/8] fp16 AMP  OK  loop dev {res['fp16']['loop']['max_logit_dev']:.2e} "
          f"(exact: {exact16}); buf {res['fp16']['buf']['max_logit_dev']:.2e}")

    # 4 -----------------------------------------------------------------------------
    # Two statements, deliberately separated. In fp32 the routed backward must reproduce the
    # baseline's gradients: the forward is bit-identical and every coefficient is exactly 1, so
    # only the summation ORDER differs. Under fp16 autocast the matmul backward itself runs in
    # half precision (eps = 9.8e-4), so a differently-ordered but mathematically identical
    # backward lands ~1 ulp apart -- that is fp16 noise both arms are subject to, not a
    # correctness property, so it is measured and reported rather than tightly gated.
    x, y = batches[0]
    lb32, gb32 = plain_grads(build(a.seed, False).to(device), x, y)
    ll32, gl32 = plain_grads(build(a.seed, True, "loop").to(device), x, y)
    lf32, gf32 = plain_grads(build(a.seed, True, "buf").to(device), x, y)
    shared = [n for n in gb32 if n in gl32]
    d_loop32, w_loop32 = grad_dev(gb32, gl32, shared)
    d_buf32, w_buf32 = grad_dev(gb32, gf32, shared)
    assert d_loop32 <= a.tol_grad_loop, (
        f"fp32: loop gradients differ from base by {d_loop32:.2e} at {w_loop32} -- do not train")
    assert d_buf32 <= a.tol_grad_buf, (
        f"fp32: buf gradients differ from base by {d_buf32:.2e} at {w_buf32} -- do not train")
    print(f"[4/8] fp32 grad OK  loop {d_loop32:.2e} ({w_loop32}); buf {d_buf32:.2e} ({w_buf32})")

    lb, gbse, sb, nb, lnb = amp_step_grads(build(a.seed, False).to(device), x, y)
    ll, gl, sl, nl, _ = amp_step_grads(build(a.seed, True, "loop").to(device), x, y)
    lf, gf, sf, nf, lnf = amp_step_grads(build(a.seed, True, "buf").to(device), x, y)
    shared16 = [n for n in gbse if n in gl]
    d_loop, w_loop = grad_dev(gbse, gl, shared16)
    d_buf, w_buf = grad_dev(gbse, gf, shared16)
    assert d_loop <= a.tol_step_loop, f"fp16 step: loop {d_loop:.2e} at {w_loop} exceeds fp16 noise"
    assert d_buf <= a.tol_step_buf, f"fp16 step: buf {d_buf:.2e} at {w_buf} exceeds fp16 noise"
    assert sb == sl == sf, f"GradScaler state diverged: base {sb}, loop {sl}, buf {sf}"
    assert np.isfinite(float(nb)) and np.isfinite(float(nf)), "non-finite grad norm"
    theta_buckets = [k for k in (lnf or {}) if "router" in k]
    assert theta_buckets, "router.theta missing from get_layer_grad_norms buckets"
    res["grads"] = dict(
        fp32=dict(base_loss=lb32, loop_loss=ll32, buf_loss=lf32,
                  loop_dev=d_loop32, loop_worst=w_loop32, buf_dev=d_buf32, buf_worst=w_buf32),
        fp16=dict(base_loss=lb, loop_loss=ll, buf_loss=lf, scale=sb,
                  loop_dev=d_loop, loop_worst=w_loop, buf_dev=d_buf, buf_worst=w_buf,
                  fp16_eps=2 ** -10, base_grad_norm=float(nb), buf_grad_norm=float(nf),
                  theta_buckets=theta_buckets))
    print(f"[4/8] fp16 step OK  loop {d_loop:.2e} ({w_loop}); buf {d_buf:.2e} ({w_buf}); "
          f"fp16 eps {2 ** -10:.1e}; scale {sb:g} on all arms")

    # 5 -----------------------------------------------------------------------------
    g = torch.Generator().manual_seed(5)
    n = 2 * depth + 1
    C = torch.tril(torch.rand(n, n, generator=g) * 1.5 + 0.25)
    cl = build(a.seed, True, "loop").to(device)
    cf = build(a.seed, True, "buf").to(device)
    for m in (cl, cf):
        m.router.set_coefficient_matrix(C.to(device))
        m.router.set_edge_mask(n, 0, 0.0)
    outs = {}
    for tag, m in (("loop", cl), ("buf", cf)):
        m.train()
        m.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x), y)      # fp32, real device kernels
        loss.backward()
        outs[tag] = ({k: p.grad.detach().clone() for k, p in m.named_parameters()}, loss.item())
    dx, wx = grad_dev(outs["loop"][0], outs["buf"][0], list(outs["loop"][0]))
    dlogit = abs(outs["loop"][1] - outs["buf"][1])
    assert dx <= a.tol_crosscheck, (
        f"loop vs buf gradients differ by {dx:.2e} at {wx} -- this is the only check the Triton "
        f"kernels get, do not slacken it")
    res["crosscheck"] = dict(max_rel_grad_dev=dx, worst=wx, loss_dev=dlogit,
                             masked_theta_grad=float(outs["buf"][0][f"router.theta.{n-1}"][0]))
    print(f"[5/8] loop~buf  OK  max rel grad dev {dx:.2e} ({wx}), fp32 real kernels")

    # 6 -----------------------------------------------------------------------------
    res["init_profile"] = init_profile(base, loop, x)
    print(f"[6/8] profile   OK  {res['init_profile']['n_branches']} branch outputs equal to base; "
          f"ratio {min(res['init_profile']['ratio']):.3f}..{max(res['init_profile']['ratio']):.3f}")

    # 7 -----------------------------------------------------------------------------
    base_probe = fp16_range_probe(build(a.seed, False).to(device), batches, a.n_overflow)
    res["fp16_range"] = {
        "base_like": base_probe,
        "loop": fp16_range_probe(build(a.seed, True, "loop").to(device), batches, a.n_overflow),
        "buf": fp16_range_probe(build(a.seed, True, "buf").to(device), batches, a.n_overflow),
        "loop_2x": fp16_range_probe(build(a.seed, True, "loop").to(device), batches,
                                    a.n_overflow, scale_coeffs=2.0),
    }
    for arm in ("loop", "buf"):
        assert res["fp16_range"][arm]["skipped"] <= base_probe["skipped"], (
            f"{arm} skipped {res['fp16_range'][arm]['skipped']} steps vs base "
            f"{base_probe['skipped']} -- a systematic difference would confound the comparison")
    print(f"[7/8] fp16 range OK  skips base/loop/buf = {base_probe['skipped']}/"
          f"{res['fp16_range']['loop']['skipped']}/{res['fp16_range']['buf']['skipped']}; "
          f"max |mix| = {res['fp16_range']['loop']['frac_of_fp16_max']:.2%} of fp16 max")

    # 8 -----------------------------------------------------------------------------
    res["timing"] = {}
    if not a.no_timing:
        for tag, mk in (("base", lambda: build(a.seed, False)),
                        ("loop", lambda: build(a.seed, True, "loop")),
                        ("buf", lambda: build(a.seed, True, "buf"))):
            res["timing"][tag] = timing(mk, batches, a.n_timed, device)
            print(f"[8/8] timing    {tag:5s} {res['timing'][tag]['ms_micro']:7.1f} ms   "
                  f"peak {res['timing'][tag]['peak_gb']:6.2f} GB")
        b = res["timing"]["base"]["ms_micro"]
        for tag in ("loop", "buf"):
            res["timing"][tag]["overhead_frac"] = res["timing"][tag]["ms_micro"] / b - 1
        print(f"        overhead: loop +{100 * res['timing']['loop']['overhead_frac']:.0f}%  "
              f"buf +{100 * res['timing']['buf']['overhead_frac']:.0f}%")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'gate.json'}")
    if res["timing"]:
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
    p.add_argument("--n-overflow", type=int, default=30)
    p.add_argument("--no-timing", action="store_true")
    p.add_argument("--from-json", action="store_true",
                   help="redraw gate.png from an existing gate.json (no GPU needed)")
    p.add_argument("--tol-buf-logit", type=float, default=1e-5)
    p.add_argument("--tol-buf-loss", type=float, default=1e-6)
    p.add_argument("--tol-amp-loop", type=float, default=1e-6)
    p.add_argument("--tol-amp-buf-logit", type=float, default=5e-2)
    p.add_argument("--tol-amp-buf-loss", type=float, default=1e-3)
    p.add_argument("--tol-grad-loop", type=float, default=1e-5)   # fp32, the real claim
    p.add_argument("--tol-grad-buf", type=float, default=1e-4)    # fp32
    p.add_argument("--tol-step-loop", type=float, default=5e-2)   # fp16 noise band
    p.add_argument("--tol-step-buf", type=float, default=5e-2)
    p.add_argument("--tol-crosscheck", type=float, default=1e-4)
    main_gate(p.parse_args())
