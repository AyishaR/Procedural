"""The --resid_routing flag: independently weighted residual inputs (models/residual_routing.py).

Every attention branch, every MLP branch and the readout learns its own softplus-weighted sum
over the embedding stream and every earlier branch output; the mixes replace the residual adds.
Key regression guards: the flag off is bit-identical to vanilla; the router draws no RNG (so a
routed model shares a same-seed baseline's backbone bit for bit); all-ones routing reproduces the
baseline logits EXACTLY (same summation order); a hand-written reference pins the source indexing
and the independent readout; masks give exact zeros while keeping every theta in the graph (DDP);
checkpoints round-trip; the thetas land in optim_factory's no_decay group.

Ported from the layer-scale repo's tests/test_residual_routing.py. The GPT-only tests (KV-cache
inference path, generate, residual_ratios) are replaced by their ViT counterparts, and the repo
integration this port adds -- utils.assert_routing_isolated, utils.RoutedHookCollector,
engine.routed_layer_metrics, the strict=False resume hazard -- is covered here too.

routing_impl="buf" (fused history buffer, deferred per-source gradients) must compute the same
function and gradients as the reviewed "loop" path up to fp32 summation order; its tests compare
the two directly (logits, every parameter gradient, masks, the per-forward scratch state).
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
from models.residual_routing import (  # noqa: E402
    THETA_ONE,
    ResidualHistory,
    ResidualRouter,
    consumer_labels,
    has_hooks,
    routed_forward_buf,
    source_labels,
)

DEPTH, IMG, NCLS = 2, 32, 8
ARGS = dict(pretrained=False, num_classes=NCLS, drop_path_rate=0.0, depth=DEPTH, img_size=IMG)
M = 2 * DEPTH
N = M + 1  # consumers (branches + readout) == matrix size


def _model(seed, **kw):
    torch.manual_seed(seed)
    return create_model("vit_b_test", **ARGS, **kw).eval()


def _batch(seed=7, n=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 3, IMG, IMG, generator=g),
            torch.randint(0, NCLS, (n,), generator=g))


def _random_C(seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.tril(torch.rand(N, N, generator=g) * 1.5 + 0.25)


def _reference(m, x, C):
    """Hand-written routed forward over the model's own submodules: pins the indexing convention
    (row r = consumer r+1, col c = source c) and the independent readout row."""
    hist = [m.norm_pre(m._pos_embed(m.patch_embed(x)))]
    mods = []
    for blk in m.blocks:
        mods += [(blk.norm1, blk.attn, blk.ls1, blk.drop_path1),
                 (blk.norm2, blk.mlp, blk.ls2, blk.drop_path2)]
    for i, (norm, branch, ls, dp) in enumerate(mods, start=1):
        mixed = sum(C[i - 1, j] * hist[j] for j in range(i))
        hist.append(dp(ls(branch(norm(mixed)))))
    mixed = sum(C[M, j] * hist[j] for j in range(M + 1))
    z = m.norm(mixed)
    z = z[:, m.num_prefix_tokens:].mean(dim=1) if m.global_pool == "avg" else z[:, 0]
    return m.head(m.fc_norm(z)), hist


def _history(model, x):
    """The M branch outputs of a forward pass, hooked on drop_path1 / drop_path2 -- the LAST
    module of each branch, so this is what the vanilla Block adds to the stream and what the
    routed model writes into history[k]. Works for both paths."""
    out, handles = [], []
    for blk in model.blocks:
        for mod in (blk.drop_path1, blk.drop_path2):
            handles.append(mod.register_forward_hook(lambda _m, _a, o: out.append(o.detach())))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    assert len(out) == M
    return out


def _grads(model, x, y):
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def _assert_grads_close(ga, gb, rtol=1e-4, theta_rtol=1e-3):
    """Max relative deviation per tensor -- but the thetas share one scale and a looser bound.

    They are a single logical coefficient vector split across a ParameterList, and two entries
    are numerically delicate: consumer 1's is structurally ~0 (see
    test_first_consumer_coefficient_gradient_is_structurally_zero), and the readout row's
    embedding entry is a dot product against h0 that partly cancels, because the readout mix
    feeds a LayerNorm and its gradient is nearly orthogonal to the mix. In fp32 the REVIEWED loop
    path is itself only ~1.7e-4 accurate there against a float64 reference, and buf ~2.6e-4 --
    the same noise band, not a defect in buf. test_buf_is_as_accurate_as_loop_in_float64 pins
    that directly; this bound just has to sit above fp32 noise.
    """
    theta_scale = max((g.abs().max().item() for n, g in ga.items()
                       if n.startswith("router.theta.")), default=0.0)
    for n in ga:
        is_theta = n.startswith("router.theta.")
        scale = theta_scale if is_theta else ga[n].abs().max().item()
        dev = (ga[n] - gb[n]).abs().max().item() / (scale + 1e-12)
        assert dev < (theta_rtol if is_theta else rtol), f"{n}: max rel grad dev {dev:.2e}"


# --------------------------------------------------------------------------------------
# the flag off, and the routed model at init
# --------------------------------------------------------------------------------------
def test_default_off_bit_identical():
    x, _ = _batch()
    v = _model(0)
    for kw in ({}, {"resid_routing": False, "routing_bound": 2.0}):
        m = _model(0, **kw)
        assert not hasattr(m, "router")
        assert m.state_dict().keys() == v.state_dict().keys()
        for k, val in v.state_dict().items():
            assert torch.equal(m.state_dict()[k], val)
        with torch.no_grad():
            assert torch.equal(m(x), v(x))
        # the vanilla forward_features contract is untouched
        feats, flist = m.forward_features(x)
        assert isinstance(feats, torch.Tensor) and len(flist) == m.distill_num_heads


def test_rng_neutral_and_keys():
    v = _model(0)
    rv = torch.rand(3)
    r = _model(0, resid_routing=True)
    rr_ = torch.rand(3)
    assert torch.equal(rv, rr_), "router construction must not consume RNG"
    sv, sr = v.state_dict(), r.state_dict()
    router_keys = {k for k in sr if k.startswith("router.")}
    assert router_keys == {"router.mask"} | {f"router.theta.{k}" for k in range(N)}
    assert set(sr) - router_keys == set(sv)
    for k in sv:
        assert torch.equal(sr[k], sv[k]), k
    assert sum(p.numel() for p in r.router.parameters()) == N * (N + 1) // 2


@pytest.mark.parametrize("impl", ["loop", "buf"])
@pytest.mark.parametrize("bound", [0.0, 2.0, 8.0])
def test_all_ones_matches_baseline(impl, bound):
    x, y = _batch()
    v = _model(1)
    r = _model(1, resid_routing=True, routing_bound=bound, routing_impl=impl)
    with torch.no_grad():
        lv, lr = v(x), r(x)
        lossv = F.cross_entropy(lv, y)
        lossr = F.cross_entropy(lr, y)
    if impl == "loop" and bound in (0.0, 2.0):
        # init coefficient bit-exact 1.0 and the same summation order -> identical arithmetic
        assert torch.equal(lr, lv)
        assert torch.equal(lossr, lossv)
    else:  # bound 8 is 1 ulp off 1.0; buf differs only in fp32 summation order
        torch.testing.assert_close(lr, lv, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lossr, lossv, rtol=1e-6, atol=1e-6)


def test_coefficient_matrix_and_summary_at_init():
    r = _model(0, resid_routing=True)
    C = r.router.coefficient_matrix()
    assert C.shape == (N, N)
    assert torch.equal(C, torch.tril(torch.ones(N, N)))
    for i in range(1, N + 1):
        assert r.router.coefficients(i).shape == (i,)
    s = r.router.summary()
    assert s["n_edges"] == N * (N + 1) // 2
    for k in ("mean", "min", "max", "emb_col_mean", "prev_diag_mean", "readout_row_mean"):
        assert s[k] == 1.0, k
    r.router.set_edge_mask(2, 0, 0.0)
    C2 = r.router.coefficient_matrix()
    assert C2[1, 0] == 0.0 and (C != C2).sum() == 1
    assert r.router.summary()["n_edges"] == N * (N + 1) // 2 - 1
    rb = _model(0, resid_routing=True, routing_bound=8.0)
    torch.testing.assert_close(rb.router.coefficient_matrix(), C, rtol=0, atol=2e-7)
    assert consumer_labels(2) == ["attn_0", "mlp_0", "attn_1", "mlp_1", "readout"]
    assert source_labels(2) == ["emb", "attn_0", "mlp_0", "attn_1", "mlp_1"]
    assert F.softplus(torch.tensor(THETA_ONE)).item() == 1.0


# --------------------------------------------------------------------------------------
# what the model computes
# --------------------------------------------------------------------------------------
def test_reference_random_coefficients():
    x, _ = _batch()
    r = _model(2, resid_routing=True)
    C = _random_C()
    r.router.set_coefficient_matrix(C)
    torch.testing.assert_close(r.router.coefficient_matrix(), C, rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        logits = r(x)
        ref, _ = _reference(r, x, C)
    torch.testing.assert_close(logits, ref, rtol=1e-5, atol=1e-5)
    rb = _model(2, resid_routing=True, routing_bound=4.0)
    rb.router.set_coefficient_matrix(C)
    torch.testing.assert_close(rb.router.coefficient_matrix(), C, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(rb(x), ref, rtol=1e-4, atol=1e-4)


def test_history_entries_are_the_vanilla_branch_outputs():
    """history[k] is exactly what the vanilla Block would have added to the stream, so at init
    the routed model's branch outputs equal the baseline's tensor for tensor."""
    x, _ = _batch()
    v, r = _model(8), _model(8, resid_routing=True)
    for a, b in zip(_history(v, x), _history(r, x), strict=True):
        assert torch.equal(a, b)


def test_readout_row_independent():
    x, _ = _batch()
    r = _model(2, resid_routing=True)
    C = _random_C()
    r.router.set_coefficient_matrix(C)
    before = _history(r, x)
    with torch.no_grad():
        l0 = r(x)
    C2 = C.clone()
    C2[M] *= 2.0
    r.router.set_coefficient_matrix(C2)
    after = _history(r, x)
    with torch.no_grad():
        l1 = r(x)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(l0, l1)


def test_readout_path_and_features_list():
    """forward_features returns (norm(readout mix), [readout mix]); the pre-norm entry is what
    the vanilla path puts in features_list, and forward's distill indexing still works."""
    x, _ = _batch()
    r = _model(5, resid_routing=True)
    r.router.set_coefficient_matrix(_random_C())
    with torch.no_grad():
        feats, flist = r.forward_features(x)
        logits = r(x)
    assert len(flist) == r.distill_num_heads == 1
    assert torch.equal(feats, r.norm(flist[0]))
    z = flist[0]
    with torch.no_grad():
        expect = r.head(r.fc_norm(r.norm(z)[:, 0]))
    torch.testing.assert_close(logits, expect, rtol=1e-6, atol=1e-6)
    assert torch.equal(r(x, output_dict=True)["logits"], logits)
    r.train()   # every dropout rate is 0, so train/eval must agree
    with torch.no_grad():
        assert torch.equal(r(x), logits)


def test_gradients_flow():
    x, y = _batch()
    r = _model(3, resid_routing=True)
    _grads(r, x, y)
    for k, th in enumerate(r.router.theta):
        assert th.grad is not None, k
        assert torch.isfinite(th.grad).all() and (th.grad != 0).all(), k
    for name, p in r.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# --------------------------------------------------------------------------------------
# edge masks
# --------------------------------------------------------------------------------------
def test_first_consumer_coefficient_gradient_is_structurally_zero():
    """Consumer 1's mix has a SINGLE source, so its coefficient only rescales the input of
    norm1 -- and LayerNorm is scale-invariant. dL/dtheta[1,0] is therefore ~0 whatever the seed
    (it survives only through the LayerNorm eps), while every other consumer mixes >= 2 sources
    and its coefficients change direction, not just scale.

    Recorded because it is why _assert_grads_close normalises the thetas by a shared scale, and
    because a future "the coefficients are not learning" report about theta.0 would otherwise
    look like a bug.
    """
    for seed, bseed in ((2, 7), (11, 31), (23, 7)):
        x, y = _batch(seed=bseed)
        m = _model(seed, resid_routing=True)
        m.router.set_coefficient_matrix(_random_C(seed))
        _, g = _grads(m, x, y)
        theta_max = max(v.abs().max().item() for k, v in g.items()
                        if k.startswith("router.theta."))
        assert g["router.theta.0"].abs().max().item() < 1e-4 * theta_max
        assert (g["router.theta.0"] != 0).all()      # nonzero, just tiny


def test_edge_mask_router_level():
    router = ResidualRouter(2)
    g = torch.Generator().manual_seed(0)
    hist = [torch.randn(2, 3, 4, generator=g) for _ in range(3)]
    alt = list(hist)
    alt[1] = torch.randn(2, 3, 4, generator=g)
    assert not torch.equal(router(3, hist), router(3, alt))
    router.set_edge_mask(3, 1, 0.0)
    assert torch.equal(router(3, hist), router(3, alt))
    assert router.coefficient_matrix()[2, 1] == 0.0
    assert router.coefficient_matrix()[2, 0] == 1.0
    for bad in ((2, 2), (0, 0), (6, 0), (1, -1)):
        with pytest.raises(AssertionError):
            router.set_edge_mask(*bad, 0.0)


def test_edge_mask_model_level():
    x, y = _batch()
    r = _model(2, resid_routing=True)
    r.router.set_coefficient_matrix(_random_C())
    r.router.set_edge_mask(3, 1, 0.0)
    with torch.no_grad():
        logits = r(x)
        ref, _ = _reference(r, x, r.router.coefficient_matrix())
    torch.testing.assert_close(logits, ref, rtol=1e-5, atol=1e-5)
    _grads(r, x, y)
    g = r.router.theta[2].grad
    assert g is not None            # masked edge: still in the graph (DDP-safe) ...
    assert g[1].item() == 0.0       # ... with an exactly-zero gradient
    assert (g[[0, 2]] != 0).all()


# --------------------------------------------------------------------------------------
# configuration and checkpoints
# --------------------------------------------------------------------------------------
def test_incompatible_configs_rejected():
    for kw in ({"routing_bound": 1.0}, {"routing_bound": -1.0}, {"routing_bound": 0.5},
               {"routing_impl": "fast"}, {"init_values": 1e-5}, {"distill_num_heads": 2},
               {"distill_head_dim": 64}, {"is_teacher": True}, {"global_pool": ""},
               {"block_fn": mvt.ResPostBlock}):
        with pytest.raises(AssertionError):
            _model(0, resid_routing=True, **kw)
    with pytest.raises(AssertionError):   # stochastic depth is a second residual intervention
        torch.manual_seed(0)
        create_model("vit_b_test", pretrained=False, num_classes=NCLS, depth=DEPTH, img_size=IMG,
                     drop_path_rate=0.1, resid_routing=True)
    # accepted
    assert hasattr(_model(0, resid_routing=True, routing_bound=2.0), "router")
    assert hasattr(_model(0, resid_routing=True, global_pool="avg"), "router")
    assert hasattr(_model(0, resid_routing=True, routing_impl="buf"), "router")


def test_grad_checkpointing_refused():
    """checkpoint_seq would run the routed traversal twice, which the buf path refuses by design."""
    x, _ = _batch()
    r = _model(0, resid_routing=True)
    r.set_grad_checkpointing(True)
    with pytest.raises(AssertionError, match="gradient checkpointing"):
        r(x)


def test_checkpoint_roundtrip():
    x, _ = _batch()
    r = _model(4, resid_routing=True)
    with torch.no_grad():
        for k, th in enumerate(r.router.theta):
            th.add_(0.1 * (k + 1))
    r.router.set_edge_mask(4, 0, 0.0)
    with torch.no_grad():
        l0 = r(x)
    sd = {k: v.clone() for k, v in r.state_dict().items()}
    assert "router.mask" in sd and sd["router.mask"][3, 0] == 0.0

    fresh = _model(0, resid_routing=True)
    fresh.load_state_dict(sd)                       # strict
    with torch.no_grad():
        assert torch.equal(fresh(x), l0)

    # a vanilla checkpoint cannot be strict-loaded into a routed model (no accidental warm start)
    with pytest.raises(RuntimeError):
        _model(0, resid_routing=True).load_state_dict(_model(4).state_dict())

    # ... and utils.auto_load_model's strict=False load drops the router in SILENCE. This is the
    # exact hazard utils.auto_load_model's routing guard exists to catch; pin the behaviour.
    got = _model(0, resid_routing=True).load_state_dict(_model(4).state_dict(), strict=False)
    assert sorted(got.missing_keys) == sorted(k for k in sd if k.startswith("router."))
    assert got.unexpected_keys == []

    # --save_for_analysis dumps state_dict().half() but must keep the router in fp32, or the
    # "every coefficient is exactly 1 at init" property is lost to ~2.4e-4
    init = _model(0, resid_routing=True)
    dump = {k: (v if k.startswith("router.") else v.half()) for k, v in init.state_dict().items()}
    back = _model(0, resid_routing=True)
    back.load_state_dict({k: v.float() for k, v in dump.items()})
    assert torch.equal(back.router.coefficient_matrix(), init.router.coefficient_matrix())
    naive = {k: v.half().float() for k, v in init.state_dict().items()}
    back2 = _model(0, resid_routing=True)
    back2.load_state_dict(naive)
    assert not torch.equal(back2.router.coefficient_matrix(), init.router.coefficient_matrix())


# --------------------------------------------------------------------------------------
# repo integration: optimizer groups, the isolation guard, the hook collectors
# --------------------------------------------------------------------------------------
def _args(argv):
    import argparse

    import main
    p = argparse.ArgumentParser("t", parents=[main.get_args_parser()])
    return p.parse_args(argv), p


BASE_ARGV = ("--model vit_base --seed 0 --lr 2e-3 --epochs 300 --warmup_epochs 50 "
             "--batch_size 128 --total_batch_size 4096 --update_freq 8 --use_amp true "
             "--data_set IMNET --data_path /d --output_dir /o").split()


def test_thetas_land_in_the_no_decay_group():
    import optim_factory
    args, _ = _args(BASE_ARGV + ["--resid_routing", "true"])
    r = _model(0, resid_routing=True)
    opt = optim_factory.create_optimizer(args, r)
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    nodecay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    router_ids = {id(p) for p in r.router.parameters()}
    assert router_ids <= nodecay and not (router_ids & decay)
    assert not (decay & nodecay)
    assert len(decay | nodecay) == len(list(r.parameters()))
    for g in opt.param_groups:      # same LR as the backbone
        assert g.get("lr_scale", 1) == 1.0


def test_routing_isolation_guard():
    import utils
    args, p = _args(BASE_ARGV + ["--resid_routing", "true", "--routing_impl", "buf"])
    utils.assert_routing_isolated(args, p)      # the real routed command line passes
    for extra in (["--initialize", "x.pth"], ["--init_method", "ortho"],
                  ["--freeze_blocks", "0,1"], ["--drop_path", "0.1"],
                  ["--attention_residual_scaling", "11[0.5]"]):
        args, p = _args(BASE_ARGV + ["--resid_routing", "true"] + extra)
        with pytest.raises(AssertionError, match="composes with no other intervention"):
            utils.assert_routing_isolated(args, p)
    args, p = _args(BASE_ARGV + ["--freeze_blocks", "0,1"])   # flag off -> no-op
    utils.assert_routing_isolated(args, p)


def test_hook_collectors():
    """HookCollector reconstructs a per-block stream routing does not have, so it must refuse a
    routed model; RoutedHookCollector captures what routing does define."""
    import utils
    x, _ = _batch()
    v, r = _model(9), _model(9, resid_routing=True)
    with pytest.raises(AssertionError, match="RoutedHookCollector"):
        utils.HookCollector(r)
    with pytest.raises(AssertionError, match="use HookCollector"):
        utils.RoutedHookCollector(v)
    with utils.RoutedHookCollector(r) as acts:
        with torch.no_grad():
            r(x)
        hist = _history(r, x)
        for layer in range(DEPTH):
            a = acts[layer]
            assert set(a) >= {"attn_map", "attn_out", "mlp_out", "attn_mix", "mlp_mix"}
            assert torch.equal(a["attn_out"], hist[2 * layer])
            assert torch.equal(a["mlp_out"], hist[2 * layer + 1])
            assert a["attn_map"].shape[:2] == (x.shape[0], r.blocks[0].attn.num_heads)
        assert "mix" in acts["readout"]
    assert not has_hooks(r), "collector must remove its hooks on exit"


def test_routed_layer_metrics_match_the_baseline_definition_at_init():
    """The routed delta norms are the same tensors the vanilla path measures, and with every
    coefficient at 1 the ratios use the same denominator too -- so both arms agree at init."""
    import engine
    import utils
    x, _ = _batch(n=3)
    v, r = _model(10), _model(10, resid_routing=True)

    with utils.HookCollector(v) as va:
        with torch.no_grad():
            v(x)
        base = {}
        for i in range(DEPTH):
            attn_delta = torch.norm(va[i]["attn_out"], dim=-1)
            mlp_delta = torch.norm(va[i]["blk"] - va[i]["attn"], dim=-1)
            base[f"attn_delta_norm_layer{i}"] = attn_delta.mean().item()
            base[f"mlp_delta_norm_layer{i}"] = mlp_delta.mean().item()
            base[f"attn_delta_norm_ratio_layer{i}"] = (
                attn_delta / (torch.norm(va[i]["inp"], dim=-1) + 1e-8)).mean().item()
            base[f"mlp_delta_norm_ratio_layer{i}"] = (
                mlp_delta / (torch.norm(va[i]["attn"], dim=-1) + 1e-8)).mean().item()

    logger = utils.MetricLogger()
    with utils.RoutedHookCollector(r) as ra:
        with torch.no_grad():
            r(x)
        for i in range(DEPTH):
            engine.routed_layer_metrics(ra, i, x, logger, r.blocks[0].attn.num_heads,
                                        detailed=True)
    for k, want in base.items():
        assert k in logger.meters, k
        assert abs(logger.meters[k].global_avg - want) < 1e-5, k
    for i in range(DEPTH):
        assert f"attn_mix_norm_layer{i}" in logger.meters
        assert f"attn_entropy_layer{i}" in logger.meters       # survives routing untouched
        for absent in ("blk_acc", "attn_acc", "blk_act_norm", "inp_norm"):
            assert f"{absent}_layer{i}" not in logger.meters, absent


# --------------------------------------------------------------------------------------
# routing_impl = "buf"
# --------------------------------------------------------------------------------------
def test_buf_matches_loop_logits_and_all_gradients():
    x, y = _batch()
    C = _random_C()
    loop, buf = _model(2, resid_routing=True), _model(2, resid_routing=True, routing_impl="buf")
    for m in (loop, buf):
        m.router.set_coefficient_matrix(C)
    with torch.no_grad():
        torch.testing.assert_close(buf(x), loop(x), rtol=1e-5, atol=1e-5)
        ref, _ = _reference(loop, x, C)
        torch.testing.assert_close(buf(x), ref, rtol=1e-5, atol=1e-5)
    la, ga = _grads(loop, x, y)
    lb, gb = _grads(buf, x, y)
    torch.testing.assert_close(lb, la, rtol=1e-6, atol=1e-6)
    _assert_grads_close(ga, gb)
    for k in range(N):   # every theta gets its gradient through the deferred path
        assert gb[f"router.theta.{k}"] is not None and (gb[f"router.theta.{k}"] != 0).all()
    assert buf.state_dict().keys() == loop.state_dict().keys()


def test_buf_is_as_accurate_as_loop_in_float64():
    """The real correctness claim for the fused path: against a float64 reference, buf's
    gradients are as accurate as the reviewed loop path's. A plain loop-vs-buf tolerance cannot
    say this -- in fp32 both paths lose the same ~2e-4 on the readout row's embedding entry
    (a partly-cancelling dot product against h0, whose LayerNorm makes the gradient nearly
    orthogonal to the mix), so the two disagree by more than either is wrong.
    """
    x, y = _batch()
    C = _random_C()

    def built(impl, dtype):
        m = _model(2, resid_routing=True, routing_impl=impl).to(dtype)
        m.router.set_coefficient_matrix(C.to(dtype))
        return m

    _, truth = _grads(built("loop", torch.float64), x.double(), y)
    _, gl = _grads(built("loop", torch.float32), x, y)
    _, gb = _grads(built("buf", torch.float32), x, y)
    for n, t in truth.items():
        t = t.float()
        scale = t.abs().max().item() + 1e-12
        dev_loop = (gl[n] - t).abs().max().item() / scale
        dev_buf = (gb[n] - t).abs().max().item() / scale
        assert dev_buf < 1e-3, f"{n}: buf is {dev_buf:.2e} off the float64 reference"
        assert dev_buf <= 4 * dev_loop + 1e-6, (
            f"{n}: buf {dev_buf:.2e} is materially worse than loop {dev_loop:.2e}")


def test_buf_edge_mask_and_bounded():
    x, y = _batch()
    loop = _model(2, resid_routing=True, routing_bound=4.0)
    buf = _model(2, resid_routing=True, routing_bound=4.0, routing_impl="buf")
    for m in (loop, buf):
        m.router.set_coefficient_matrix(_random_C())
        m.router.set_edge_mask(4, 1, 0.0)
        m.router.set_edge_mask(N, 0, 0.0)      # readout ignores the embedding
    with torch.no_grad():
        torch.testing.assert_close(buf(x), loop(x), rtol=1e-5, atol=1e-5)
    _, ga = _grads(loop, x, y)
    _, gb = _grads(buf, x, y)
    _assert_grads_close(ga, gb)
    assert gb["router.theta.3"][1].item() == 0.0
    assert gb[f"router.theta.{N - 1}"][0].item() == 0.0


def test_buf_state_is_per_forward_and_reusable():
    """Two forwards (and their backwards) do not share scratch; no state lives on the module."""
    x, y = _batch()
    buf = _model(3, resid_routing=True, routing_impl="buf")
    l1, g1 = _grads(buf, x, y)
    l2, g2 = _grads(buf, x, y)
    assert torch.equal(l1, l2)
    for n in g1:
        assert torch.equal(g1[n], g2[n]), n
    x2, y2 = _batch(seed=9)
    l3, _ = _grads(buf, x2, y2)
    assert not torch.equal(l3, l1)
    assert not any(isinstance(v, ResidualHistory) for v in vars(buf.router).values())
    assert len(buf.state_dict()) == len(_model(3, resid_routing=True).state_dict())


def test_buf_checkpoint_interoperates_with_loop():
    x, _ = _batch()
    buf = _model(4, resid_routing=True, routing_impl="buf")
    buf.router.set_coefficient_matrix(_random_C())
    with torch.no_grad():
        full = buf(x)
    fresh = _model(0, resid_routing=True, routing_impl="buf")
    fresh.load_state_dict(buf.state_dict())
    with torch.no_grad():
        assert torch.equal(fresh(x), full)
    as_loop = _model(0, resid_routing=True, routing_impl="loop")
    as_loop.load_state_dict(buf.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(as_loop(x), full, rtol=1e-5, atol=1e-5)
    assert routed_forward_buf is not None


def test_buf_rejects_double_and_repeated_backward_and_releases_scratch():
    x, y = _batch()
    buf = _model(5, resid_routing=True, routing_impl="buf")
    buf.train()
    loss = F.cross_entropy(buf(x), y)
    with pytest.raises(RuntimeError, match="double backward"):
        torch.autograd.grad(loss, list(buf.router.theta), create_graph=True)

    buf.zero_grad(set_to_none=True)
    loss = F.cross_entropy(buf(x), y)
    loss.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="one backward per forward"):
        loss.backward()

    # a backward rooted at an intermediate (partial pass) is refused, not silently wrong
    buf.zero_grad(set_to_none=True)
    captured = []
    h = buf.blocks[0].attn.register_forward_hook(lambda _m, _a, o: captured.append(o))
    F.cross_entropy(buf(x), y)
    h.remove()
    with pytest.raises(RuntimeError, match="partial backward"):
        captured[0].sum().backward()

    # the loop implementation supports all three
    loop = _model(5, resid_routing=True)
    loop.train()
    g = torch.autograd.grad(F.cross_entropy(loop(x), y), list(loop.router.theta),
                            create_graph=True)
    assert all(t.requires_grad for t in g)


def test_buf_hooks_detected_for_eager_branches():
    """has_hooks must be RECURSIVE here: utils.HookCollector registers on block.mlp.act and
    block.mlp.fc2, which are descendants of a branch module rather than the branch module."""
    buf = _model(6, resid_routing=True, routing_impl="buf")
    assert not has_hooks(buf)
    for mod in (buf.blocks[1].mlp, buf.blocks[1].mlp.fc2, buf.blocks[0].mlp.act,
                buf.blocks[0].attn, buf.blocks[0].drop_path1):
        h = mod.register_forward_hook(lambda _m, _a, o: None)
        assert has_hooks(buf), mod
        h.remove()
        assert not has_hooks(buf), mod
    h = buf.blocks[0].norm1.register_forward_pre_hook(lambda _m, _a: None)
    assert has_hooks(buf)
    h.remove()
    assert not has_hooks(buf)


def test_buf_theta_only_grad_then_partial_backward_raises_and_scratch_is_released(monkeypatch):
    """autograd.grad w.r.t. the thetas only runs the consumer nodes, never _Write_0: a second
    (partial) pass on the retained graph must raise instead of reading stale rows; and with a
    frozen embedding (no _Write_0 node at all) the scratch is still released by _Mix_1."""
    import models.residual_routing as arr

    states = []

    class Recording(arr.ResidualHistory):
        def __init__(self, n, x0):
            super().__init__(n, x0)
            states.append(self)

    monkeypatch.setattr(arr, "ResidualHistory", Recording)
    x, y = _batch()
    buf = _model(7, resid_routing=True, routing_impl="buf")
    buf.train()
    loss = F.cross_entropy(buf(x), y)
    g = torch.autograd.grad(loss, list(buf.router.theta), retain_graph=True)
    assert all(t is not None for t in g)
    st = states[-1]
    assert st.G is not None            # not released: _Write_0 never ran
    with pytest.raises(RuntimeError, match="partial backward|backward twice"):
        loss.backward()                # second pass over the same graph
    assert st.G is None                # released on the error path

    # frozen embedding: h0 carries no gradient, so _Mix_1 is the last user of the scratch
    frozen = _model(7, resid_routing=True, routing_impl="buf")
    frozen.train()
    for mod in (frozen.patch_embed, frozen.norm_pre):
        for p in mod.parameters():
            p.requires_grad_(False)
    frozen.cls_token.requires_grad_(False)
    frozen.pos_embed.requires_grad_(False)
    loss3 = F.cross_entropy(frozen(x), y)
    st3 = states[-1]
    loss3.backward()
    assert st3.G is None and st3.buf is None and st3.h0 is None
    assert frozen.blocks[0].attn.proj.weight.grad is not None
    assert all(th.grad is not None for th in frozen.router.theta)

    # the normal full backward releases too
    buf.zero_grad(set_to_none=True)
    loss4 = F.cross_entropy(buf(x), y)
    st4 = states[-1]
    loss4.backward()
    assert st4.G is None
