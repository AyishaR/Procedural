"""The --readout_routing flag: a weighted mix over the 2L+1 sources feeds the classifier while
every block connection stays vanilla (models/readout_routing.py, docs/residual_routing.md).

Regression guards: the flag off is bit-identical to vanilla; the mixer draws no RNG (same-seed
backbone shared bit for bit); 'learn_ones' reproduces the vanilla logits EXACTLY in fp32 (same
summation order) and its backbone gradients up to fp32 summation order (fp64 agrees to 1e-14);
'fixed_last4' equals the hand-computed head(fc_norm(pool(norm(x_after_last - x_after_last-4))));
Block.forward still runs, so utils.HookCollector works on the mixed model; the weight lands in
optim_factory's no_decay group; the isolation guard, the strict=False resume hazard, the half
dump and the gradient-sign counters are covered.

Toy depth is 6 so that "the last 4 blocks" (sources 5..12 of 13) is a proper subset.
"""

import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore", category=FutureWarning)

from timm.models import create_model  # noqa: E402

import models.vision_transformer as mvt  # noqa: E402  (populates the timm registry)
from models.readout_routing import READOUT_MODES, ReadoutMixer, readout_init  # noqa: E402
from models.residual_routing import has_hooks, source_labels  # noqa: E402

DEPTH, IMG, NCLS = 6, 32, 8
ARGS = dict(pretrained=False, num_classes=NCLS, drop_path_rate=0.0, depth=DEPTH, img_size=IMG)
N = 2 * DEPTH + 1                      # sources: emb + 2*depth branch outputs
FIRST_LATE = 2 * (DEPTH - 4) + 1       # first source of the last 4 blocks (17 at depth 12)
ARMS = {"A": "fixed_last4", "B": "learn_last4", "C": "learn_ones"}


def _model(seed, **kw):
    torch.manual_seed(seed)
    return create_model("vit_b_test", **ARGS, **kw).eval()


def _batch(seed=7, n=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, IMG, IMG, generator=g),
            torch.randint(0, NCLS, (n,), generator=g))


def _stream_states(m, x):
    """[x entering block 0, x after block 0, ..., x after block L-1] of the vanilla loop."""
    with torch.no_grad():
        h = m.norm_pre(m._pos_embed(m.patch_embed(x)))
        states = [h]
        for blk in m.blocks:
            h = blk(h)
            states.append(h)
    return states


def _history(model, x):
    """The 2L branch outputs hooked on drop_path1 / drop_path2 (see tests/test_residual_routing)."""
    out, handles = [], []
    for blk in model.blocks:
        for mod in (blk.drop_path1, blk.drop_path2):
            handles.append(mod.register_forward_hook(lambda _m, _a, o: out.append(o.detach())))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    assert len(out) == 2 * DEPTH
    return out


def _readout(m, mixed):
    z = m.norm(mixed)
    z = z[:, m.num_prefix_tokens:].mean(dim=1) if m.global_pool == "avg" else z[:, 0]
    return m.head(m.fc_norm(z))


def _grads(model, x, y):
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    model.eval()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def _max_rel_dev(ga, gb):
    return max((ga[n] - gb[n]).abs().max().item() / (ga[n].abs().max().item() + 1e-12) for n in ga)


# --------------------------------------------------------------------------------------
# init vectors and the flag off
# --------------------------------------------------------------------------------------
def test_readout_init_vectors():
    w, t = readout_init("fixed_last4", 12)
    assert not t and w.tolist() == [0.0] * 17 + [1.0] * 8
    w, t = readout_init("learn_last4", 12)
    assert t and w.tolist() == [0.0] * 17 + [1.0] * 8
    w, t = readout_init("learn_ones", 12)
    assert t and w.tolist() == [1.0] * 25
    w, _ = readout_init("fixed_last4", DEPTH)
    assert w.tolist() == [0.0] * FIRST_LATE + [1.0] * 8
    assert source_labels(12)[17] == "attn_8" and source_labels(12)[24] == "mlp_11"
    with pytest.raises(AssertionError, match="last 4"):
        readout_init("fixed_last4", 3)
    for bad in ("off", "learn_last3"):
        with pytest.raises(AssertionError):
            readout_init(bad, 12)
    assert READOUT_MODES == ("off", "fixed_last4", "learn_last4", "learn_ones")


def test_default_off_bit_identical():
    x, _ = _batch()
    v = _model(0)
    assert v.readout_routing == "off" and not hasattr(v, "readout_mixer")
    assert not any(k.startswith("readout_mixer.") for k in v.state_dict())
    with torch.no_grad():
        a, b = v(x), _model(0)(x)
    assert torch.equal(a, b)


def test_rng_neutral_keys_and_buffer_vs_parameter():
    sv = _model(0).state_dict()
    for arm, mode in ARMS.items():
        m = _model(0, readout_routing=mode)
        sd = m.state_dict()
        assert set(sd) - set(sv) == {"readout_mixer.weight"}, arm
        assert all(torch.equal(sv[k], sd[k]) for k in sv), f"{arm}: same-seed backbone differs"
        params = dict(m.named_parameters())
        if mode == "fixed_last4":
            assert "readout_mixer.weight" not in params and not m.readout_mixer.trainable
            assert "readout_mixer.weight" in dict(m.named_buffers())
        else:
            assert "readout_mixer.weight" in params and m.readout_mixer.trainable
        assert m.readout_mixer.labels() == source_labels(DEPTH)
        assert m.readout_mixer.weight.dtype == torch.float32


# --------------------------------------------------------------------------------------
# function at init
# --------------------------------------------------------------------------------------
def test_learn_ones_is_vanilla_bit_for_bit():
    x, y = _batch()
    v, c = _model(0), _model(0, readout_routing="learn_ones")
    with torch.no_grad():
        assert torch.equal(v(x), c(x))
    v.train(); c.train()
    with torch.no_grad():                       # train mode too (dropout rates are 0)
        assert torch.equal(v(x), c(x))
    # gradients: same maths, different accumulation order for dL/dh_j (direct edge + stream)
    _, gv = _grads(v, x, y)
    _, gc = _grads(c, x, y)
    gc_backbone = {n: g for n, g in gc.items() if not n.startswith("readout_mixer.")}
    assert set(gc_backbone) == set(gv)
    assert _max_rel_dev(gv, gc_backbone) < 1e-5
    v64, c64 = _model(0).double(), _model(0, readout_routing="learn_ones").double()
    _, gv = _grads(v64, x.double(), y)
    _, gc = _grads(c64, x.double(), y)
    assert _max_rel_dev(gv, {n: g for n, g in gc.items() if n in gv}) < 1e-13


def test_fixed_last4_is_the_last_four_blocks_delta():
    x, _ = _batch()
    v, a, b = _model(0), _model(0, readout_routing="fixed_last4"), _model(0, readout_routing="learn_last4")
    states = _stream_states(v, x)
    with torch.no_grad():
        la, lb, lv = a(x), b(x), v(x)
        ref = _readout(v, states[DEPTH] - states[DEPTH - 4])      # x_after_last - x_after_last-4
    assert torch.equal(la, lb), "B is A at init"
    assert not torch.equal(la, lv)
    assert torch.allclose(la, ref, atol=1e-5, rtol=1e-5), (la - ref).abs().max()
    # and the sources are the vanilla branch outputs: hand-mixed sum over hist[FIRST_LATE-1:]
    hist = _history(v, x)
    with torch.no_grad():
        ref2 = _readout(v, sum(hist[FIRST_LATE - 1:]))
    assert torch.allclose(la, ref2, atol=1e-5, rtol=1e-5)


def test_mixer_reference_and_dtype():
    """Hand-written weighted sum pins the source indexing; fp16 sources are accumulated in fp32
    and the result comes back in the stream's dtype."""
    x, _ = _batch()
    v = _model(0)
    hist = _history(v, x)
    with torch.no_grad():
        emb = v.norm_pre(v._pos_embed(v.patch_embed(x)))
    g = torch.Generator().manual_seed(3)
    w = torch.rand(N, generator=g) * 2 - 0.5                      # negatives included
    mx = ReadoutMixer(DEPTH, "learn_ones")
    with torch.no_grad():
        mx.weight.copy_(w)
        got = mx([emb] + hist)
        ref = sum(w[j] * s for j, s in enumerate([emb] + hist))
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        got16 = mx([emb] + [h.half() for h in hist])           # fp32 stream, fp16 branches
    assert got16.dtype == torch.float32
    assert torch.allclose(got16, ref, atol=1e-2, rtol=1e-2)
    with pytest.raises(AssertionError):
        mx(hist)                                                # wrong number of sources


# --------------------------------------------------------------------------------------
# gradients, sign counters, summary
# --------------------------------------------------------------------------------------
def test_gradients_and_sign_counters():
    x, y = _batch()
    for mode in ("learn_last4", "learn_ones"):
        m = _model(0, readout_routing=mode)
        _, g = _grads(m, x, y)
        gw = g["readout_mixer.weight"]
        assert gw.shape == (N,) and torch.isfinite(gw).all() and (gw != 0).all()
        mx = m.readout_mixer
        assert mx.grad_stats()["n_steps"] == 0
        mx.record_grad_sign()                   # grad still attached from _grads
        mx.record_grad_sign()
        st = mx.grad_stats()
        assert st["n_steps"] == 2 and st["n_nonfinite"] == 0
        assert st["pos_frac"] == [1.0 if v > 0 else 0.0 for v in gw.tolist()]
        assert st["sign_consistency"] == [1.0] * N
        mx.weight.grad[0] = float("nan")        # a GradScaler-skipped step: counted, not signed
        mx.record_grad_sign()
        st = mx.grad_stats()
        assert st["n_steps"] == 2 and st["n_nonfinite"] == 1
        mx.reset_grad_stats()
        st = mx.grad_stats()
        assert st["n_steps"] == 0 and st["pos_frac"] is None and st["sign_consistency"] is None
    a = _model(0, readout_routing="fixed_last4")
    _, g = _grads(a, x, y)
    assert "readout_mixer.weight" not in g
    a.readout_mixer.record_grad_sign()          # no-op for the fixed arm
    assert a.readout_mixer.grad_stats()["n_steps"] == 0


def test_summary_at_init():
    s = _model(0, readout_routing="learn_last4").readout_mixer.summary()
    assert s["emb"] == 0.0 and s["early_mean"] == 0.0 and s["late_mean"] == 1.0
    assert s["min"] == 0.0 and s["max"] == 1.0 and s["n_neg"] == 0 and s["n_zero"] == FIRST_LATE
    assert s["late_early_ratio"] is None          # undefined -> JSON null, never Infinity
    import json
    assert "Infinity" not in json.dumps(s) and "NaN" not in json.dumps(s)
    s = _model(0, readout_routing="learn_ones").readout_mixer.summary()
    assert s["mean"] == s["min"] == s["max"] == s["emb"] == 1.0 and s["n_zero"] == 0
    assert s["late_early_ratio"] == 1.0
    assert set(s) == {"mean", "min", "max", "emb", "early_mean", "late_mean",
                      "late_early_ratio", "n_neg", "n_zero"}


# --------------------------------------------------------------------------------------
# the vanilla stream still exists: hook collectors, no leftover hooks
# --------------------------------------------------------------------------------------
def test_hook_collector_works_and_hooks_are_removed():
    import utils
    x, _ = _batch()
    v, c = _model(9), _model(9, readout_routing="learn_last4")
    states = _stream_states(v, x)
    hist = _history(v, x)
    with utils.HookCollector(c) as acts:
        with torch.no_grad():
            out = c(x)
    for i in range(DEPTH):
        assert torch.equal(acts[i]["blk"], states[i + 1])
        assert torch.equal(acts[i]["inp"], states[i])
        assert torch.equal(acts[i]["attn_out"], hist[2 * i])      # the re-run inside the hook
    assert not has_hooks(c), "capture hooks must be gone after the forward"
    with torch.no_grad():                        # the collector's re-run did not change the output
        assert torch.equal(out, c(x))
    with pytest.raises(AssertionError, match="use HookCollector"):
        utils.RoutedHookCollector(c)
    # a failed forward must not leave the capture hooks behind either
    c.blocks[2].mlp.fc2.register_forward_hook(lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        with torch.no_grad():
            c(x)
    assert sum(len(blk.drop_path1._forward_hooks) + len(blk.drop_path2._forward_hooks)
               for blk in c.blocks) == 0


def test_incompatible_configs_rejected():
    with pytest.raises(AssertionError, match="one intervention at a time"):
        _model(0, resid_routing=True, readout_routing="learn_ones")
    with pytest.raises(AssertionError, match="readout_routing must be one of"):
        _model(0, readout_routing="learn_last3")
    with pytest.raises(AssertionError, match="stochastic depth"):
        torch.manual_seed(0)
        create_model("vit_b_test", **{**ARGS, "drop_path_rate": 0.1}, readout_routing="learn_ones")
    with pytest.raises(AssertionError, match="LayerScale"):
        _model(0, readout_routing="learn_ones", init_values=1e-5)
    m = _model(0, readout_routing="learn_ones")
    m.set_grad_checkpointing(True)
    with pytest.raises(AssertionError, match="gradient checkpointing"):
        m(_batch()[0])


# --------------------------------------------------------------------------------------
# repo integration: optimizer groups, the isolation guard, resume, checkpoints, the half dump
# --------------------------------------------------------------------------------------
def _args(argv):
    import argparse

    import main
    p = argparse.ArgumentParser("t", parents=[main.get_args_parser()])
    return p.parse_args(argv), p


BASE_ARGV = ("--model vit_base --seed 0 --lr 2e-3 --epochs 300 --warmup_epochs 50 "
             "--batch_size 128 --total_batch_size 4096 --update_freq 8 --use_amp true "
             "--data_set IMNET --data_path /d --output_dir /o").split()


def test_weight_lands_in_the_no_decay_group():
    import optim_factory
    args, _ = _args(BASE_ARGV + ["--readout_routing", "learn_ones"])
    m = _model(0, readout_routing="learn_ones")
    opt = optim_factory.create_optimizer(args, m)
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    nodecay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    w = id(m.readout_mixer.weight)
    assert w in nodecay and w not in decay
    assert len(decay | nodecay) == len(list(m.parameters()))
    for g in opt.param_groups:
        assert g.get("lr_scale", 1) == 1.0


def test_build_model_and_isolation_guard():
    import utils
    for mode in ARMS.values():
        args, p = _args(BASE_ARGV + ["--readout_routing", mode])
        utils.assert_routing_isolated(args, p)          # the real arm command lines pass
        args.model, args.nb_classes, args.drop_path = "vit_b_test", NCLS, 0.0
        torch.manual_seed(0)
        m = utils.build_model(args)
        assert m.readout_routing == mode and m.readout_mixer.mode == mode
    for extra in (["--drop_path", "0.1"], ["--initialize", "x.pth"], ["--freeze_blocks", "0,1"],
                  ["--attention_residual_scaling", "11[0.5]"]):
        args, p = _args(BASE_ARGV + ["--readout_routing", "learn_ones"] + extra)
        with pytest.raises(AssertionError, match="composes with no other intervention"):
            utils.assert_routing_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--readout_routing", "learn_ones", "--resid_routing", "true"])
    with pytest.raises(AssertionError, match="do not compose"):
        utils.assert_routing_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--freeze_blocks", "0,1"])       # both off -> no-op
    utils.assert_routing_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--resid_routing", "true", "--routing_impl", "buf"])
    utils.assert_routing_isolated(args, p)                        # the routed line still passes


def test_auto_load_model_guards_the_mode(tmp_path):
    """auto_load_model loads strict=False; a mode mismatch (vanilla <-> mixed, or A <-> B) would
    corrupt the run in silence, and the chain driver resubmits on every TIMEOUT."""
    import argparse

    import main
    import utils

    def mkargs(mode, out):
        p = argparse.ArgumentParser("t", parents=[main.get_args_parser()])
        a = p.parse_args(["--model", "vit_b_test", "--readout_routing", mode,
                          "--output_dir", str(out)])
        a.auto_resume, a.resume, a.model_ema, a.start_epoch = True, "", False, 0
        return a

    class Stub:
        def load_state_dict(self, sd):
            pass

    def build(mode):
        return _model(0) if mode == "off" else _model(0, readout_routing=mode)

    dirs = {}
    for mode in ("off", "fixed_last4", "learn_last4"):
        d = tmp_path / mode
        d.mkdir()
        m = build(mode)
        if mode == "learn_last4":
            with torch.no_grad():
                m.readout_mixer.weight[0] = 0.25          # a trained value must round-trip
        torch.save({"model": m.state_dict(), "args": mkargs(mode, d), "epoch": 3,
                    "optimizer": torch.optim.AdamW(m.parameters()).state_dict(), "scaler": {}},
                   d / "checkpoint-3.pth")
        dirs[mode] = d

    def run(ckpt, mode):
        a = mkargs(mode, dirs[ckpt])
        m = build(mode)
        utils.auto_load_model(args=a, model=m, model_without_ddp=m,
                              optimizer=torch.optim.AdamW(m.parameters()),
                              loss_scaler=Stub(), model_ema=None)
        return a, m

    a, m = run("learn_last4", "learn_last4")
    assert a.start_epoch == 4 and m.readout_mixer.weight[0].item() == 0.25
    run("fixed_last4", "fixed_last4")
    run("off", "off")
    for ckpt, mode in (("off", "learn_last4"), ("learn_last4", "off"),
                       ("fixed_last4", "learn_last4"), ("learn_last4", "fixed_last4")):
        with pytest.raises(AssertionError, match="readout_routing"):
            run(ckpt, mode)


def test_checkpoint_roundtrip_and_half_dump():
    x, _ = _batch()
    m = _model(0, readout_routing="learn_ones")
    with torch.no_grad():
        m.readout_mixer.weight.copy_(torch.linspace(-0.5, 2.0, N))
        ref = m(x)
    m2 = _model(1, readout_routing="learn_ones")
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert torch.equal(m2(x), ref)
    # the --save_for_analysis dump keeps the mixer fp32 (main.py); the predicate is the key prefix
    dump = {k: (v if k.startswith(("router.", "readout_mixer.")) else v.half())
            for k, v in m.state_dict().items()}
    assert dump["readout_mixer.weight"].dtype == torch.float32
    assert dump["head.weight"].dtype == torch.float16
    assert torch.equal(dump["readout_mixer.weight"], m.readout_mixer.weight)
