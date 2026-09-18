"""CPU regression tests for the training-dynamics primitives (tiny `vit_b_test` model, seeded).

Guards: branch gains (1 = the real block bit for bit, 0 = bypass), residual-stream states vs the model's own
forward, the last-k bypass = raw lens identity, gain gradients vs finite differences, exact softmax invariance
to the k-bias, block-function change localised to the changed block, chunked = full-batch gradients, gradient
norm grouping, the epoch <-> file convention and the registry's epoch names and gate.
"""
import copy
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore", category=FutureWarning)
import models.vision_transformer  # noqa: E402,F401  (registers vit_b_test with timm)
import timm  # noqa: E402

from dynamics import common  # noqa: E402
from dynamics import registry as dreg  # noqa: E402
from probes import registry as preg  # noqa: E402

DEPTH, IMG, NCLS = 6, 32, 8


def _model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    m = timm.create_model("vit_b_test", pretrained=False, depth=DEPTH, img_size=IMG, num_classes=NCLS)
    with torch.no_grad():                       # non-trivial norms/biases so nothing is accidentally symmetric
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return m.eval().to(dtype)


def _batch(seed=7, n=3, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, IMG, IMG, generator=g).to(dtype), torch.randint(0, NCLS, (n,), generator=g)


def test_gain_one_is_the_block_bit_for_bit():
    m = _model()
    x = torch.randn(3, 5, 8)
    for blk in m.blocks:
        assert torch.equal(common.block_forward(blk, x, 1.0, 1.0), blk(x))
        ones = torch.ones(3, 1, 1)
        assert torch.equal(common.block_forward(blk, x, ones, ones), blk(x))


def test_gain_zero_bypasses_the_block():
    m = _model()
    x = torch.randn(3, 5, 8)
    for blk in m.blocks:
        assert torch.equal(common.block_forward(blk, x, 0.0, 0.0), x)
        y = common.block_forward(blk, x, 0.0, 1.0)              # only the MLP branch
        assert torch.equal(y, x + common.mlp_branch(blk, x))


def test_states_match_the_model_forward():
    m = _model()
    img, _ = _batch()
    hs, mids = common.states(m, img, mid=True)
    assert len(hs) == DEPTH + 1 and len(mids) == DEPTH
    for l, blk in enumerate(m.blocks):
        assert torch.equal(hs[l + 1], blk(hs[l]))
        assert torch.equal(mids[l], hs[l] + common.attn_branch(blk, hs[l]))
    with torch.no_grad():
        assert torch.allclose(common.head_on(m, hs[-1]), m(img), atol=1e-5, rtol=1e-5)
        assert torch.allclose(common.suffix(m, hs[2], 2), hs[-1], atol=1e-6)


def test_last3_bypass_is_the_depth_lens():
    m = _model()
    img, _ = _batch()
    hs = common.states(m, img)
    h = hs[DEPTH - 3]
    for l in range(DEPTH - 3, DEPTH):
        h = common.block_forward(m.blocks[l], h, 0.0, 0.0)
    assert torch.equal(h, hs[DEPTH - 3])
    assert torch.equal(common.head_on(m, h), common.head_on(m, hs[DEPTH - 3]))


def test_gain_gradient_matches_finite_difference():
    m = _model(dtype=torch.float64)
    img, y = _batch(dtype=torch.float64)
    B = img.shape[0]

    def loss_at(g):
        return F.cross_entropy(common.head_on(m, common.forward_with_gains(m, img, g)), y, reduction="sum")

    g = torch.ones(B, DEPTH, 2, dtype=torch.float64, requires_grad=True)
    loss_at(g).backward()
    for (i, l, b) in [(0, 2, 1), (1, 4, 0), (2, 0, 0), (1, DEPTH - 1, 1)]:
        eps = 1e-6
        gp, gm = torch.ones(B, DEPTH, 2, dtype=torch.float64), torch.ones(B, DEPTH, 2, dtype=torch.float64)
        gp[i, l, b] += eps
        gm[i, l, b] -= eps
        with torch.no_grad():
            fd = (loss_at(gp) - loss_at(gm)) / (2 * eps)
        assert abs(float(g.grad[i, l, b]) - float(fd)) <= 1e-6 * max(1.0, abs(float(fd))), (i, l, b, g.grad[i, l, b], fd)


def test_k_bias_is_softmax_invariant_and_zeroed_at_load():
    m = _model(dtype=torch.float64)
    img, _ = _batch(dtype=torch.float64)
    with torch.no_grad():
        for l, blk in enumerate(m.blocks):
            D = blk.attn.qkv.bias.shape[0] // 3
            blk.attn.qkv.bias[D:2 * D] = 50.0 * (l + 1) * torch.randn(D, dtype=torch.float64)
        before = m(img)
        norms0 = common.k_bias_norms(m)
        assert min(norms0.values()) > 10
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        removed = preg.zero_k_bias_(sd)
        m.load_state_dict(sd)
        after = m(img)
    assert torch.allclose(before, after, atol=1e-9, rtol=1e-9)
    assert set(removed) == {str(l) for l in range(DEPTH)}
    assert all(abs(removed[str(l)] - norms0[l]) < 1e-6 for l in range(DEPTH))
    assert max(common.k_bias_norms(m).values()) == 0.0


def test_block_update_change_is_localised():
    mA = _model()
    mB = copy.deepcopy(mA)
    with torch.no_grad():
        mB.blocks[3].mlp.fc2.weight.add_(0.5 * torch.randn_like(mB.blocks[3].mlp.fc2.weight))
    img, _ = _batch()
    hs = common.states(mA, img)
    with torch.no_grad():
        for l in range(DEPTH):
            a, mm, FA = common.block_update(mA.blocks[l], hs[l])
            _, _, FB = common.block_update(mB.blocks[l], hs[l])
            assert torch.allclose(FA + hs[l], mA.blocks[l](hs[l]), atol=1e-5)
            if l == 3:
                assert (FA - FB).norm() > 1e-3
            else:
                assert torch.equal(FA, FB)


def test_chunked_gradient_equals_full_batch():
    m = _model(dtype=torch.float64)
    img, y = _batch(n=6, dtype=torch.float64)
    img[4:] *= 30.0                                                     # uneven chunks with very different gradient sizes
    for objective in ("smooth", "hard"):
        m.zero_grad(set_to_none=True)
        full_loss = common.loss_fn(m(img), y, objective)
        full_loss.backward()
        full = {n: p.grad.clone() for n, p in m.named_parameters()}
        ref_norms = []
        for s in (0, 4):
            m.zero_grad(set_to_none=True)
            common.loss_fn(m(img[s:s + 4]), y[s:s + 4], objective).backward()
            ref_norms.append(sum(float(p.grad.pow(2).sum()) for p in m.parameters()) ** 0.5)
        loss, chunk_norms = common.accumulate_grads(m, img, y, objective, chunk=4)
        assert abs(loss - float(full_loss)) < 1e-12
        assert len(chunk_norms) == 2 and all(abs(a - b) < 1e-9 * max(1, b) for a, b in zip(chunk_norms, ref_norms))
        for n, p in m.named_parameters():
            assert torch.allclose(p.grad, full[n], atol=1e-10, rtol=1e-9), (objective, n)


def test_pred_is_argmax_on_ties():
    logits = torch.zeros(2, 8)
    logits[1, 3] = 1.0
    pm = common.per_image(logits, torch.tensor([0, 3]))
    assert pm["pred"].tolist() == [0, 3] and pm["correct"].tolist() == [True, True]


def test_precision_context_restores_state():
    torch.set_float32_matmul_precision("highest")
    cudnn0 = torch.backends.cudnn.allow_tf32                            # PyTorch's default is True
    with common.precision("tf32"):
        assert torch.get_float32_matmul_precision() == "high" and torch.backends.cudnn.allow_tf32
        with common.precision("fp32"):
            assert torch.get_float32_matmul_precision() == "highest" and not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high" and torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest" and torch.backends.cudnn.allow_tf32 == cudnn0
    m = _model()
    img, _ = _batch()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        with common.precision("fp32"):                                  # inherited autocast must be switched off
            assert m(img).dtype == torch.float32


def test_grad_norm_grouping_recombines():
    m = _model()
    img, y = _batch()
    common.loss_fn(m(img), y).backward()
    gn = common.grad_norms(m)
    tot2 = sum(v ** 2 for v in gn["per_block"].values()) + gn["other"] ** 2
    assert abs(gn["total"] - tot2 ** 0.5) < 1e-6
    for l in range(DEPTH):
        blk = sum(v ** 2 for n, v in gn["per_tensor"].items() if n.startswith(f"blocks.{l}.")) ** 0.5
        assert abs(blk - gn["per_block"][l]) < 1e-6
        for t in ("weight", "bias"):
            parts = sum(gn["qkv_split"][f"blocks.{l}.attn.{p}.{t}"] ** 2 for p in "qkv") ** 0.5
            assert abs(parts - gn["per_tensor"][f"blocks.{l}.attn.qkv.{t}"]) < 1e-6
    pn = common.param_norms(m)
    assert set(pn["per_block"]) == set(range(DEPTH)) and pn["other"] > 0
    # the training loop's grouping double-counts every qkv tensor (whole tensor + its q/k/v slices)
    total_hist, groups, per_block_hist = common.hist_style_grad_norms(m)
    qkv2 = sum(v ** 2 for n, v in gn["per_tensor"].items() if ".attn.qkv." in n)
    assert abs(total_hist - (gn["total"] ** 2 + qkv2) ** 0.5) < 1e-5
    qkv2_b2 = sum(v ** 2 for n, v in gn["per_tensor"].items() if n.startswith("blocks.2.attn.qkv."))
    assert abs(per_block_hist[2] - (gn["per_block"][2] ** 2 + qkv2_b2) ** 0.5) < 1e-5
    assert abs(groups["module.blocks.2.attn.qkv"] - qkv2_b2 ** 0.5) < 1e-5          # rsplit-level bucket: plain
    assert "module.blocks.2" not in groups                                            # no per-block bucket in the training code


def test_epoch_file_convention():
    assert dreg.file_index(1) == 0 and dreg.epoch_of(299) == 300 and dreg.epoch_of(dreg.file_index(225)) == 225
    assert dreg.CORE_FILES == [0, 1, 2, 4, 9, 19, 39, 79, 149, 224, 299]
    assert set(range(4, 300, 5)) <= set(dreg.DENSE_FILES) and 224 in dreg.DENSE_FILES and 0 in dreg.DENSE_FILES
    assert dreg.LENS_FILES == [0, 2, 9, 39, 149, 299]
    lr = dreg.lr_epoch_means()
    assert lr.shape == (300,) and lr[0] < 1e-4 and abs(lr[49] - 2e-3) < 5e-5 and lr[299] < 2e-6
    assert all(r["name"] in preg.BY_NAME for r in dreg.RUNS)
    assert {r["role"] for r in dreg.RUNS} <= set(dreg.ROLE_ORDER)
    for r in dreg.RUNS:                                                 # every dynamics entry zeroes the k-bias ...
        e = dreg.entry(r["name"])
        assert e["zero_k_bias"] is True and dreg.epoch_entry(e, 5)["zero_k_bias"] is True
    assert not preg.BY_NAME["ftbqmlnvo_s0"].get("zero_k_bias")            # ... while the probes entry stays untouched


def test_registry_epoch_names_and_gate(tmp_path, monkeypatch):
    run = tmp_path / "imnet_base" / "results_IMNET_BASE_29523316" / "s0"
    run.mkdir(parents=True)
    (run / "checkpoint-150-model.pth").touch()
    (run / "checkpoint-299-model.pth").touch()
    with open(run / "log.txt", "w") as f:
        for e in range(300):
            rec = {"epoch": e, "train_loss": 3.0 - e / 200}
            if (e + 1) % 5 == 0:
                rec["test_acc1"] = 50.0 + e / 10
            f.write(json.dumps(rec) + "\n")
    monkeypatch.setattr(preg, "RESULTS", tmp_path)
    monkeypatch.setattr(preg, "_ARM_TRUTH", [])
    base = preg.BY_NAME["ftbqmlnvo_s0"]
    assert preg.finished(base) == (True, "ok")
    e = preg.resolve_models("ftbqmlnvo_s0@e150")[0]
    assert e["epoch"] == 150 and e["base"] == "ftbqmlnvo_s0" and e["name"] == "ftbqmlnvo_s0@e150"
    assert preg.ckpt_path(e).name == "checkpoint-150-model.pth" and "ftbqmlnvo_s0@e150" in preg.BY_NAME
    assert preg.final_test_acc(e) is None                      # epoch index 150 is not on the 5-epoch eval grid
    assert preg.final_test_acc(preg.with_epoch(base, 149)) == 50.0 + 149 / 10
    assert preg.final_test_acc(base) == 50.0 + 299 / 10
    with pytest.raises(SystemExit):
        preg.resolve_models("ftbqmlnvo_s0@e151")               # dump missing
    with pytest.raises(SystemExit):
        preg.resolve_models("nosucharm_s0@e150")
    assert dreg.available_files(base) == [150, 299]
    assert dreg.full_ckpt(base, 299) is None
    dyn = tmp_path / "imnet_base" / "results_IMNET_BASE_29518360" / "s0"                # a finished dynamics-only run ...
    dyn.mkdir(parents=True)
    (dyn / "checkpoint-299-model.pth").touch()
    (dyn / "log.txt").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in range(300)))
    assert preg.finished(preg.BY_NAME["ftbqu_s0"]) == (True, "ok")
    assert [m["name"] for m in preg.resolve_models("all", quiet=True)] == ["ftbqmlnvo_s0"]   # ... stays out of 'all'
    preg.resolve_models("ftbqu_s0@e299")
    assert [m["name"] for m in preg.resolve_models("all", quiet=True)] == ["ftbqmlnvo_s0"]   # also after epoch registration
    # load_state on a real (tiny) checkpoint file: strict load, k-bias zeroed for a dynamics entry, recorded, excluded from norms
    tiny = _model()
    with torch.no_grad():
        for blk in tiny.blocks:
            D = blk.attn.qkv.bias.shape[0] // 3
            blk.attn.qkv.bias[D:2 * D] = 7.0
    torch.save({k: v.half() for k, v in tiny.state_dict().items()}, run / "checkpoint-150-model.pth")   # ftbqmlnvo_s0 tree
    probes_load = _model(seed=2)
    assert preg.load_state(probes_load, preg.with_epoch(preg.BY_NAME["ftbqmlnvo_s0"], 150)) == {}     # probes entry: untouched
    pn_with = common.param_norms(probes_load, exclude_k_bias=False)                                  # same fp16-rounded weights
    fresh = _model(seed=1)
    info = preg.load_state(fresh, dreg.epoch_entry(dreg.entry("ftbqmlnvo_s0"), 150))
    assert set(info) == {str(l) for l in range(DEPTH)} and all(abs(v - 7.0 * 8 ** 0.5) < 1e-6 for v in info.values())
    assert max(common.k_bias_norms(fresh).values()) == 0.0 and fresh.k_bias_removed == info
    pn_wo = common.param_norms(fresh, exclude_k_bias=True)
    for l in range(DEPTH):
        assert abs(pn_with["per_block"][l] ** 2 - pn_wo["per_block"][l] ** 2 - info[str(l)] ** 2) < 1e-4   # fp32 norm in zero_k_bias_
    assert torch.allclose(fresh.blocks[2].attn.qkv.weight, tiny.blocks[2].attn.qkv.weight.half().float())
    assert max(common.k_bias_norms(probes_load).values()) > 1
    with open(run / "log.txt", "a") as f:                       # a run that has not reached 299 is rejected
        pass
    (run / "log.txt").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in range(298)))
    assert preg.finished(base)[0] is False


def test_hist_parser_conventions():
    from dynamics import hist
    rows = []
    for ep in (None, 4, 299):
        for L in range(12):
            rows.append(dict(epoch=ep, layer=L, layer_sub=L - 0.5, acc=1.0 + L, delta_norm_ratio=0.3))
            rows.append(dict(epoch=ep, layer=L, layer_sub=L, acc=2.0 + L, delta_norm_ratio=0.5, blk_act_norm=100.0 + L, grad_norm=-1 if ep == 4 else 0.1))
            rows.append(dict(epoch=ep, layer=L, layer_sub=L, head=3, attn_entropy=1.0))          # per-head row, no acc: ignored
    rows.append(dict(epoch=299, layer=5, layer_sub=5, acc=0.1))                                    # duplicate 299 row (shuffled pass): ignored
    st = hist.parse_stats(rows)
    assert st["files"] == [-1, 4, 299]
    i = st["files"].index(299)
    assert st["top1"][i, hist.HALF.index("b5")] == 7.0 and st["top1"][i, hist.HALF.index("a5")] == 6.0
    assert all(st["top1"][:, hist.HALF.index("emb")] != st["top1"][:, hist.HALF.index("emb")])   # emb slot stays NaN
    assert st["mlp_delta_norm_ratio"][i, 5] == 0.5 and st["attn_delta_norm_ratio"][i, 5] == 0.3
    assert st["grad_norm"][st["files"].index(4), 5] != st["grad_norm"][st["files"].index(4), 5]  # -1 sentinel -> NaN
    assert st["grad_norm"][i, 5] == 0.1 and st["blk_act_norm"][i, 11] == 111.0


def test_funcchange_compare_is_per_block():
    from dynamics import funcchange
    mA = _model()
    mB = copy.deepcopy(mA)
    with torch.no_grad():
        mB.blocks[3].mlp.fc2.weight.add_(0.5 * torch.randn_like(mB.blocks[3].mlp.fc2.weight))
        mB.blocks[5].attn.proj.bias.add_(0.5)
    img, _ = _batch(n=4)
    same = funcchange.compare(mA, copy.deepcopy(mA), img, chunk=2)
    assert all(v == 0.0 for k in ("U_cls", "U_patch", "U_frob", "Ua_frob", "Um_frob", "R_frob") for v in same[k])
    assert all(v > 0 for v in same["W_frob"])                                  # write ratio of the unchanged block itself
    diff = funcchange.compare(mA, mB, img, chunk=2)
    for l in range(DEPTH):
        if l in (3, 5):
            assert diff["U_frob"][l] > 1e-4 and diff["U_cls"][l] > 1e-4
        else:
            assert diff["U_frob"][l] == 0.0 and diff["U_cls"][l] == 0.0
    assert diff["Ua_frob"][3] == 0.0 and diff["Um_frob"][3] > 0          # only the MLP changed in block 3
    assert diff["Ua_frob"][5] > 0 and diff["Um_frob"][5] == 0.0          # only the attention changed in block 5
    assert len(diff["U_cls_sd"]) == DEPTH and diff["U_cls_sd"][3] >= 0
    rel, norms = funcchange.weight_change(mA, mB)
    assert all(r == 0.0 for l, r in enumerate(rel) if l not in (3, 5)) and rel[3] > 0 and rel[5] > 0 and all(n > 0 for n in norms)
