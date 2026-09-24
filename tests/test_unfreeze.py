"""Progressive layer unfreezing (unfreeze.py + the optim_factory / utils / engine / main hooks),
fast, CPU -- the pre-launch gate of docs/unfreeze_vit_handoff.md ("Checks", step 1).

Ported from the layer-scale repo's tests/test_unfreeze.py. Regression guards: the chunk
partition / unfreeze steps / lr ramp / arm-tag grammar are what the handoff says (and the IN-100
recipe's T = 300 x 247 gives the unfreeze steps the runs record); every parameter of a toy ViT
lands in exactly one group and the decay split equals optim_factory.get_parameter_groups'; a
frozen chunk stays BIT-IDENTICAL to its init with NO Adam state until its unfreeze step and moves
(fresh state, step counting from 1) afterwards; E = 0 reproduces the vanilla optimizer loop bit
for bit (any ramp); a save / load resume mid-run equals the uninterrupted run and a layout
mismatch is refused; utils.auto_load_model / assert_unfreeze_isolated refuse the wrong runs; and
the engine / utils / main sources carry the hooks at the right places.

The step loop below drives the REAL utils.NativeScalerWithGradNormCount (its GradScaler disables
itself without CUDA, so scale / unscale_ are identities and the pre_unscale hook, the grad-norm
logging and optimizer.step run exactly as in engine.py), optim_factory.create_optimizer and
utils.cosine_scheduler.
"""

import copy
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

from timm.models import create_model  # noqa: E402

import models.vision_transformer as mvt  # noqa: E402,F401  (populates the timm registry)
import unfreeze as uf  # noqa: E402

IMG, NCLS = 32, 8
ARGS = dict(depth=4, img_size=IMG, num_classes=NCLS)
EPOCHS, NITER = 4, 5   # T = 20 steps
BASE_ARGV = ("--model vit_small --seed 0 --lr 2e-3 --epochs 300 --warmup_epochs 50 "
             "--batch_size 128 --total_batch_size 512 --update_freq 2 --use_amp true "
             "--data_set IMNET100 --data_path /d --output_dir /o --initialize '' "
             "--init_method default --skip_norm true --stage_wise_metrics true "
             "--detailed_metrics true --procedural_data kdyck --procedural_order standard").split()


def _args(argv):
    import argparse

    import main
    p = argparse.ArgumentParser("t", parents=[main.get_args_parser()])
    return p.parse_args([a.strip("'") for a in argv]), p


def _model(seed=0):
    torch.manual_seed(seed)
    return create_model("vit_b_test", **ARGS)


def _sched():
    import utils
    lr = utils.cosine_scheduler(1e-2, 1e-4, EPOCHS, NITER, warmup_epochs=1)
    wd = utils.cosine_scheduler(0.05, 0.05, EPOCHS, NITER)
    return lr, wd


def _opt(model, plan=None):
    import optim_factory
    args, _ = _args(["--lr", "1e-2", "--weight_decay", "0.05"])
    return optim_factory.create_optimizer(args, model, unfreeze_plan=plan)


def _batches(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(2, 3, IMG, IMG, generator=g), torch.randint(0, NCLS, (2,), generator=g))
            for _ in range(n)]


class _Scaler:
    """utils.NativeScalerWithGradNormCount, built lazily (its GradScaler warns without CUDA)."""
    def __init__(self):
        import utils
        self.s = utils.NativeScalerWithGradNormCount()

    def __call__(self, *a, **kw):
        return self.s(*a, **kw)


def _step(model, optimizer, scaler, batch, it, plan=None, update_freq=2):
    """One engine.train_one_epoch optimizer step: lr / wd on the first micro-step, update_freq
    micro-steps through the loss scaler (mask via pre_unscale on the last), zero_grad."""
    lr_s, wd_s = _sched()
    if plan is not None:
        plan.apply_lr(optimizer, it, lr_s[it])
    for g in optimizer.param_groups:
        if plan is None:
            g["lr"] = lr_s[it] * g.get("lr_scale", 1)
        if g["weight_decay"] > 0:
            g["weight_decay"] = wd_s[it] * g.get("wd_scale", 1)
    x, y = batch
    losses = []
    for m in range(update_freq):
        xs, ys = x[m::update_freq], y[m::update_freq]
        loss = F.cross_entropy(model(xs), ys) / update_freq
        losses.append(loss.item())
        scaler(loss, optimizer, clip_grad=None, parameters=model.named_parameters(),
               update_grad=(m + 1) % update_freq == 0,
               pre_unscale=(lambda: plan.mask_grads(it)) if plan is not None else None)
    optimizer.zero_grad()
    return losses


# --------------------------------------------------------------------------------------
# pure functions
# --------------------------------------------------------------------------------------
def test_partition_orders_chunks_by_unfreeze_order():
    assert uf.partition(12, 4, "front") == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    assert uf.partition(12, 4, "back") == [[9, 10, 11], [6, 7, 8], [3, 4, 5], [0, 1, 2]]
    for bad in ((12, 5, "front"), (12, 1, "front"), (12, 4, "up")):
        with pytest.raises(AssertionError):
            uf.partition(*bad)


def test_unfreeze_iters_and_the_in100_recipe():
    T = 300 * 247    # the r_random recipe: 126,689 train images // 512 = 247 steps per epoch
    assert T == 74100
    assert uf.unfreeze_iters("lin", 4, 0.2, T) == (0, 4940, 9880, 14820)   # epochs 20 / 40 / 60
    assert uf.unfreeze_iters("one", 4, 0.2, T) == (0, 0, 0, 14820)
    assert all(u % 247 == 0 for u in uf.unfreeze_iters("lin", 4, 0.2, T))  # epoch boundaries
    assert uf.unfreeze_iters("lin", 4, 0.0, T) == (0, 0, 0, 0)
    assert uf.unfreeze_iters("lin", 4, 0.008, T) == (0, 198, 395, 593)     # the smoke arm
    assert uf.unfreeze_iters("lin", 4, 0.5, 5000) == (0, 833, 1667, 2500)  # layer-scale's tuples
    assert uf.unfreeze_iters("geo", 4, 0.5, 5000) == (0, 625, 1250, 2500)
    assert uf.unfreeze_iters("one", 2, 0.3, 100) == uf.unfreeze_iters("lin", 2, 0.3, 100)
    assert all(isinstance(u, int) for u in uf.unfreeze_iters("lin", 4, 0.2, T))
    for bad in (("cos", 4, 0.2, T), ("lin", 1, 0.2, T), ("lin", 4, 1.5, T), ("lin", 4, 0.2, 0)):
        with pytest.raises(AssertionError):
            uf.unfreeze_iters(*bad)


def test_lr_mult_ramp_shape():
    assert uf.lr_mult(4, 5, 10) == 0.0
    assert uf.lr_mult(5, 5, 10) == pytest.approx(1 / 11)
    assert uf.lr_mult(14, 5, 10) == pytest.approx(10 / 11)
    assert uf.lr_mult(15, 5, 10) == 1.0 and uf.lr_mult(99, 5, 10) == 1.0
    assert uf.lr_mult(5, 5, 0) == 1.0
    assert uf.lr_mult(0, 0, 10) == 1.0 and uf.lr_mult(3, 0, 10) == 1.0   # u = 0 never ramps


def test_param_chunk_over_every_vit_small_name_and_prefixes():
    names = [n for n, _ in create_model("vit_small", num_classes=100).named_parameters()]
    chunks = {n: uf.param_chunk(n) for n in names}
    assert {n for n, c in chunks.items() if c is None} == {
        "cls_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias",
        "norm.weight", "norm.bias", "head.weight", "head.bias"}
    assert sorted({c for c in chunks.values() if c is not None}) == list(range(12))
    assert all(chunks[n] == int(n.split(".")[1]) for n in names if n.startswith("blocks."))
    for pre in ("", "module.", "_orig_mod.", "module._orig_mod."):
        assert uf.param_chunk(pre + "blocks.7.mlp.fc2.weight") == 7
        assert uf.param_chunk(pre + "head.weight") is None
    assert uf.param_chunk("norm_pre.weight") is None and uf.param_chunk("fc_norm.bias") is None
    assert uf.param_chunk("reg_token") is None
    for bad in ("router.theta.0", "readout_mixer.weight", "distill_head.weight", "blocks.x.a"):
        with pytest.raises(KeyError):
            uf.param_chunk(bad)


def test_arm_tags_round_trip_and_flags():
    for tag in ("u4f-lin0.2", "u4b-lin0.2", "u4f-one0.2", "u4b-one0.2", "u4b-lin0.008",
                "u4f-lin0", "u2b-geo0.5-w100"):
        assert uf.arm_tag(uf.parse_arm(tag)) == tag
        assert uf.train_flags(uf.parse_arm(tag)) == ["--unfreeze_arm", tag]
    assert uf.parse_arm("u4f-lin0.2") == dict(chunks=4, direction="front", sched="lin", end=0.2,
                                               warmup=None)
    assert uf.parse_arm("u4b-one0.20-w5")["warmup"] == 5
    assert uf.arm_tag(uf.parse_arm("u4f-lin0.20")) == "u4f-lin0.2"
    for bad in ("u4x-lin0.2", "u4f-cos0.2", "u1f-lin0.2", "u4f-lin1.5", "4f-lin0.2", ""):
        with pytest.raises(ValueError):
            uf.parse_arm(bad)


# --------------------------------------------------------------------------------------
# the plan bound to a model
# --------------------------------------------------------------------------------------
def test_param_groups_cover_every_param_once_with_get_parameter_groups_rule():
    import optim_factory
    model = _model()
    plan = uf.UnfreezePlan.from_arm(model, "u4f-lin0.5", EPOCHS * NITER, NITER)
    assert plan.warmup == NITER and plan.iters == (0, 3, 7, 10)
    assert plan.layers == [[0], [1], [2], [3]]
    opt = _opt(model, plan)
    groups = opt.param_groups
    assert len(groups) == 2 * 4 + 2
    assert [g["group_name"] for g in groups][-2:] == ["always/decay", "always/no_decay"]
    assert [g["chunk"] for g in groups] == [0, 0, 1, 1, 2, 2, 3, 3, None, None]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    ref = optim_factory.get_parameter_groups(model, 0.05, model.no_weight_decay())
    ref_decay = {id(p) for g in ref if g["weight_decay"] > 0 for p in g["params"]}
    assert {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]} == ref_decay
    assert all(g["lr_scale"] == 1.0 for g in groups)
    assert set(plan.chunk_names[2]) == {n for n, _ in model.named_parameters()
                                        if n.startswith("blocks.2.")}
    # vit_b_test has pre_norm -> norm_pre is an always-trainable tensor
    assert "norm_pre.weight" in plan.always_names and "cls_token" in plan.always_names
    assert plan.total == sum(p.numel() for p in model.parameters())


def test_frozen_chunks_are_bit_identical_with_no_adam_state_until_unfrozen():
    """Handoff (a) / (b) / (c): through the real scaler, optimizer and lr schedule."""
    model = _model()
    plan = uf.UnfreezePlan.from_arm(model, "u4b-lin0.5-w2", EPOCHS * NITER, NITER)
    assert plan.layers == [[3], [2], [1], [0]] and plan.iters == (0, 3, 7, 10)
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt, scaler = _opt(model, plan), _Scaler()
    lr_s, _ = _sched()
    for it, batch in enumerate(_batches(14)):
        _step(model, opt, scaler, batch, it, plan)
        t = it + 1  # weights after t updates
        for k, (ps, names) in enumerate(zip(plan.chunk_params, plan.chunk_names)):
            u = plan.iters[k]
            if t <= u:   # still frozen (step u's own update is the first to move it)
                assert all(torch.equal(p, init[n]) for p, n in zip(ps, names)), (t, k)
                assert all(not opt.state.get(p) for p in ps), (t, k)
            else:   # fresh Adam state; moved once it had a step with lr > 0 (the recipe's
                    # warmup starts at lr 0, so step 0 moves nothing even for a live chunk)
                assert all(int(opt.state[p]["step"]) == t - u for p in ps), (t, k)
                if t > max(u, 1):
                    assert not any(torch.equal(p, init[n]) for p, n in zip(ps, names)), (t, k)
        if t >= 2:
            for p, n in zip(plan.always_params, plan.always_names):
                assert not torch.equal(p, init[n]), (t, n)
        # (c) the group lrs of this step: frozen 0, ramp (it - u + 1) / (W + 1), alive 1
        for g in opt.param_groups:
            m = 1.0 if g["chunk"] is None else uf.lr_mult(it, plan.iters[g["chunk"]], 2)
            assert g["lr"] == lr_s[it] * m, (it, g["group_name"])
        s = plan.summary(it)
        assert s["n_unfrozen"] == sum(it >= u for u in plan.iters)
        assert s["trainable_layers"] == s["n_unfrozen"]
        scal = plan.wandb_scalars(it, 1.0)
        assert [scal[f"unfreeze/lr_chunk{k}"] for k in range(4)] == s["mults"]
        assert 0 < s["trainable_param_frac"] <= 1
    assert plan.is_event(3) and plan.is_event(10) and not plan.is_event(0) and not plan.is_event(4)
    assert plan.summary(20)["next_unfreeze"] is None and plan.summary(0)["next_unfreeze"] == 3
    assert "chunk0=B3-3@0" in plan.describe() and "unfreeze event" in plan.describe(3)


def test_end_zero_is_the_vanilla_run_bit_for_bit():
    """Handoff (d): E = 0 -- the per-chunk group layout and the (never-triggered) masking must not
    change a single bit vs create_optimizer's two groups + engine's plain lr loop, any ramp."""
    for tag in ("u4f-lin0", "u4b-one0-w5"):
        ref, model = _model(), _model()
        ref_opt, s1 = _opt(ref), _Scaler()
        plan = uf.UnfreezePlan.from_arm(model, tag, EPOCHS * NITER, NITER)
        opt, s2 = _opt(model, plan), _Scaler()
        for it, batch in enumerate(_batches(8)):
            a = _step(ref, ref_opt, s1, batch, it)
            b = _step(model, opt, s2, batch, it, plan)
            assert a == b, (tag, it, a, b)
        for (n, p), (m, q) in zip(ref.named_parameters(), model.named_parameters()):
            assert n == m and torch.equal(p, q), (tag, n)


def test_resume_mid_run_equals_the_uninterrupted_run_and_refuses_another_layout():
    """Handoff (e) / (f)."""
    batches = _batches(12)
    full = _model()
    plan = uf.UnfreezePlan.from_arm(full, "u4f-lin0.5-w2", EPOCHS * NITER, NITER)
    opt, sc = _opt(full, plan), _Scaler()
    states = None
    for it, batch in enumerate(batches):
        _step(full, opt, sc, batch, it, plan)
        if it + 1 == 5:   # a checkpoint after 5 updates: the chunk at u = 3 is live, u = 7 not
            states = copy.deepcopy(dict(model=full.state_dict(), optimizer=opt.state_dict()))
    res = _model(seed=99)
    res.load_state_dict(states["model"])
    plan2 = uf.UnfreezePlan.from_arm(res, "u4f-lin0.5-w2", EPOCHS * NITER, NITER)
    opt2, sc2 = _opt(res, plan2), _Scaler()
    opt2.load_state_dict(states["optimizer"])
    for it in range(5, 12):
        _step(res, opt2, sc2, batches[it], it, plan2)
    for (n, p), (m, q) in zip(full.named_parameters(), res.named_parameters()):
        assert n == m and torch.equal(p, q), n
    # a different chunk count = a different group layout: torch refuses the optimizer state
    other = _model()
    opt3 = _opt(other, uf.UnfreezePlan.from_arm(other, "u2f-lin0.5", EPOCHS * NITER, NITER))
    with pytest.raises(ValueError):
        opt3.load_state_dict(states["optimizer"])
    with pytest.raises(ValueError):   # and the vanilla two-group optimizer
        _opt(_model()).load_state_dict(states["optimizer"])


def test_pre_unscale_runs_before_unscale_and_the_step():
    """The mask must sit between the backward and GradScaler.unscale_ (handoff item 5), and only
    on the update micro-step."""
    import utils
    calls = []

    class SpyScaler:
        def scale(self, loss):
            calls.append("scale")
            return loss

        def unscale_(self, opt):
            calls.append("unscale_")

        def step(self, opt):
            calls.append("step")

        def update(self):
            calls.append("update")

    s = utils.NativeScalerWithGradNormCount.__new__(utils.NativeScalerWithGradNormCount)
    s._scaler = SpyScaler()
    w = torch.nn.Parameter(torch.ones(3))
    s((w * 2).sum(), None, parameters=[("w", w)], update_grad=False,
      pre_unscale=lambda: calls.append("mask"))
    assert calls == ["scale"]
    calls.clear()
    s((w * 2).sum(), None, parameters=[("w", w)], update_grad=True,
      pre_unscale=lambda: calls.append("mask"))
    assert calls == ["scale", "mask", "unscale_", "step", "update"]


# --------------------------------------------------------------------------------------
# repo integration: the isolation guard, the resume guard, the source contract
# --------------------------------------------------------------------------------------
def test_unfreeze_isolation_guard():
    import utils
    for tag in ("u4f-lin0.2", "u4b-one0.2"):
        args, p = _args(BASE_ARGV + ["--unfreeze_arm", tag])
        utils.assert_unfreeze_isolated(args, p)     # the real command line passes
        utils.assert_routing_isolated(args, p)      # routing off -> no-op
    args, p = _args(BASE_ARGV + ["--unfreeze_arm", "u4b-lin0.008", "--smoke_snapshot_every", "100",
                                 "--smoke_snapshot_steps", "197,198", "--smoke_stop_step", "650",
                                 "--enable_wandb", "false"])
    utils.assert_unfreeze_isolated(args, p)
    for extra in (["--model_ema", "true"], ["--layer_decay", "0.9"], ["--freeze_blocks", "0,1"],
                  ["--custom_lr_layer", "true"], ["--lr_match_ckpt", "x.pth"], ["--opt", "sgd"],
                  ["--readout_routing", "fixed_last3"], ["--initialize", "x.pth"]):
        args, p = _args(BASE_ARGV + ["--unfreeze_arm", "u4f-lin0.2"] + extra)
        with pytest.raises(AssertionError, match="composes with no other intervention"):
            utils.assert_unfreeze_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--unfreeze_arm", "u4f-lin0.2", "--resid_routing", "true"])
    with pytest.raises(AssertionError):
        utils.assert_routing_isolated(args, p)      # the routing guard refuses it from its side
    args, p = _args(BASE_ARGV + ["--unfreeze_arm", "u4q-lin0.2"])
    with pytest.raises(ValueError):
        utils.assert_unfreeze_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--model_ema", "true"])   # flag off -> no-op
    utils.assert_unfreeze_isolated(args, p)


def test_auto_load_model_refuses_another_arm_T_or_a_vanilla_checkpoint(tmp_path):
    import argparse

    import main
    import utils

    def mkargs(arm, T, out):
        p = argparse.ArgumentParser("t", parents=[main.get_args_parser()])
        a = p.parse_args(["--model", "vit_b_test", "--unfreeze_arm", arm, "--output_dir", str(out)])
        a.auto_resume, a.resume, a.model_ema, a.start_epoch = True, "", False, 0
        if arm:
            a.unfreeze_T = T
        return a

    class Stub:
        def load_state_dict(self, sd):
            pass

    def opt_for(m, arm, T):
        plan = uf.UnfreezePlan.from_arm(m, arm, T, NITER) if arm else None
        return _opt(m, plan)

    dirs = {}
    for tag, arm in (("arm", "u4f-lin0.2"), ("vanilla", "")):
        d = tmp_path / tag
        d.mkdir()
        m = _model()
        torch.save({"model": m.state_dict(), "args": mkargs(arm, 20, d), "epoch": 3,
                    "optimizer": opt_for(m, arm, 20).state_dict(), "scaler": {}},
                   d / "checkpoint-3.pth")
        dirs[tag] = d

    def run(ckpt, arm, T=20):
        a = mkargs(arm, T, dirs[ckpt])
        m = _model(seed=5)
        utils.auto_load_model(args=a, model=m, model_without_ddp=m, optimizer=opt_for(m, arm, T),
                              loss_scaler=Stub(), model_ema=None)
        return a

    assert run("arm", "u4f-lin0.2").start_epoch == 4     # matching: resumes
    assert run("vanilla", "").start_epoch == 4           # the vanilla path is untouched
    with pytest.raises(AssertionError, match="unfreeze_arm"):
        run("arm", "u4b-lin0.2")
    with pytest.raises(AssertionError, match="unfreeze_arm"):
        run("arm", "")                                    # an unfreeze ckpt resumed as vanilla
    with pytest.raises(AssertionError, match="unfreeze_arm"):
        run("vanilla", "u4f-lin0.2")                      # a vanilla ckpt resumed as an arm
    with pytest.raises(AssertionError, match="unfreeze_T"):
        run("arm", "u4f-lin0.2", T=40)


def test_sources_carry_the_hooks_at_the_right_places():
    eng = (REPO / "engine.py").read_text()
    ut = (REPO / "utils.py").read_text()
    mn = (REPO / "main.py").read_text()
    of = (REPO / "optim_factory.py").read_text()
    # engine: apply_lr in the per-step lr block, before the forward and the scaler call, which
    # carries the mask; the fp32 path masks before its step
    i_lr = eng.index("unfreeze_plan.apply_lr(optimizer, it, lr_schedule_values[it])")
    assert eng.index("if data_iter_step % update_freq == 0:") < i_lr
    assert i_lr < eng.index("output = model(samples)") < eng.index(
        "pre_unscale=(lambda: unfreeze_plan.mask_grads(it))")
    assert "if lr_schedule_values is not None and unfreeze_plan is None:" in eng
    fp32 = eng.index("else: # full precision")
    assert fp32 < eng.index("unfreeze_plan.mask_grads(it)  # after DDP") < eng.index(
        "optimizer.step()", fp32)
    # utils: pre_unscale() under update_grad, before unscale_
    call = ut.index("def __call__(")
    assert ut.index("if update_grad:", call) < ut.index("pre_unscale()", call) < ut.index(
        "self._scaler.unscale_(optimizer)", call)
    # main: plan before the optimizer, the optimizer before the resume; the guard before main()
    assert mn.index("UnfreezePlan.from_arm(") < mn.index("optimizer = create_optimizer(") < \
        mn.index("utils.auto_load_model(")
    assert "unfreeze_plan = unfreeze_plan" in mn and "unfreeze_plan=unfreeze_plan" in mn
    tail = mn.index("if __name__ == '__main__':")
    assert tail < mn.index("utils.assert_unfreeze_isolated(args, parser)") < mn.index(
        "\n    main(args)\n", tail)
    assert "unfreeze_plan.wandb_scalars(" in mn
    assert "parameters = unfreeze_plan.param_groups(args.weight_decay, skip)" in of
