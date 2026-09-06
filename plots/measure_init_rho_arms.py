#!/usr/bin/env python3
"""Measure rho = ||Delta_sublayer|| / ||r_in|| per block, AT INIT, for every fig5 arm.

Why standalone rather than main.py's --attention_residual_analysis: that flag returns at
main.py:1086, which is BEFORE utils.shuffle_weights runs (main.py:1225) and before the
upscale_random block scaling (~main.py:1385). It would therefore silently measure an
un-shuffled model for ftb4e3. Here each arm is constructed explicitly so what is measured is
what is trained.

rho matches engine.attention_residual_analysis exactly, so the numbers are comparable to the
ones quoted throughout docs/i100_late_block_scaling.md:

    rho_attn = mean_tokens ||attn(norm1(x))|| / ||x||
    rho_mlp  = mean_tokens ||mlp(norm2(x'))|| / ||x'||        x' = x + attn(norm1(x))

Usage (needs a GPU and the ImageNet val set):
    python plots/measure_init_rho_arms.py --out plots/cache/init_rho.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root: utils, main, datasets
import utils


ARMS = ["r", "ftb4o", "ftbnorm", "ftbqm", "ftbqmln", "ftbqmbias", "ftbqm1d",
        "ftbqm1dv", "ftb4e3", "ftb3i"]
RECIPE_BLOCKS = [9, 10, 11]
RECIPE_TARGET = 1.4
SCALED = list(range(9))          # blocks 0-8, as in every fig5 arm
RANDOM_LATE = [9, 10, 11]


# ----------------------------------------------------------------------------------------
# building each arm
# ----------------------------------------------------------------------------------------
def _rand_model(args):
    m, mw, _ = utils.pr_load_model(path="", args=args, device="cpu")
    return mw


def _proc_model(args, random_blocks):
    a = argparse.Namespace(**vars(args))
    a.random_blocks = list(random_blocks)
    m, mw, _ = utils.pr_load_model(path=args.initialize, args=a, device="cpu")
    return mw


def _target_params(args):
    mw = _proc_model(args, random_blocks=[])
    return {k: v.detach().clone() for k, v in mw.named_parameters()}


def _block_idx(name):
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return None


def build_recipe(args, x, target=RECIPE_TARGET, blocks=RECIPE_BLOCKS):
    """`ftbrho`: a fully random model whose blocks 9-11 are scaled so rho hits `target`.

    Mirrors main.py's upscale_random_match_delta_norms with --target_ratio_absolute:
      * attention is scaled through attn.v and attn.proj -> 2 tensors, so each takes
        scale_sq ** (1/2) and the delta moves by the square (utils.scale_layer_weights:1156).
        "v" is the last third of the fused qkv rows.
      * the MLP is scaled through mlp.fc2 alone -> 1 tensor, so the factor IS the multiplier.
      * blocks are calibrated SEQUENTIALLY with a fresh measurement before each, because
        scaling block 9 inflates the stream that block 10 is then measured against
        (the compounding of docs 3.10.2). --simultaneous_init_scaling is off in the arm.
    """
    mw = _rand_model(args).cuda()
    log = []
    for b in blocks:
        st = measure(mw, x)                                   # attention first
        s_a = (target / st[b]["rho_attn"]) ** 0.5
        with torch.no_grad():
            blk = mw.blocks[b]
            e = blk.attn.qkv.weight.shape[0] // 3
            blk.attn.qkv.weight.data[2 * e:3 * e, :] *= s_a   # "v"
            blk.attn.proj.weight.data *= s_a
        st = measure(mw, x)                                   # then the MLP, re-measured
        s_m = target / st[b]["rho_mlp"]
        with torch.no_grad():
            mw.blocks[b].mlp.fc2.weight.data *= s_m
        log.append((b, s_a, s_m))
    for b, sa, sm in log:
        print(f"  recipe blk {b}: v & proj x{sa:.2f} (delta x{sa**2:.1f}), fc2 x{sm:.2f}", flush=True)
    return mw


def build_ftb4o(args, x, tgt_model):
    """`ftb4o`: random model, blocks 0-7 calibrated to PROC's measured delta-norm ratios.

    This is the arm that supposedly ruled rho out as the early-block mechanism (77.27, -0.81).
    Mirrors main.py's upscale_random_match_delta_norms with no --target_ratio_absolute, so the
    target is the proc model's own per-block rho, measured on the same batch.
    """
    target = measure(tgt_model.cuda(), x)
    mw = _rand_model(args).cuda()
    for b in range(8):
        st = measure(mw, x)
        s_a = (target[b]["rho_attn"] / st[b]["rho_attn"]) ** 0.5
        with torch.no_grad():
            blk = mw.blocks[b]
            e = blk.attn.qkv.weight.shape[0] // 3
            blk.attn.qkv.weight.data[2 * e:3 * e, :] *= s_a
            blk.attn.proj.weight.data *= s_a
        st = measure(mw, x)
        s_m = target[b]["rho_mlp"] / st[b]["rho_mlp"]
        with torch.no_grad():
            mw.blocks[b].mlp.fc2.weight.data *= s_m
    return mw


def build_arm(arm, args, tgt):
    """Return a model_without_ddp with the arm's init applied to blocks 0-8."""
    if arm == "p":
        return _proc_model(args, random_blocks=[])
    if arm == "ftb3i":
        return _proc_model(args, random_blocks=RANDOM_LATE)
    if arm == "ftb4e3":
        mw = _proc_model(args, random_blocks=RANDOM_LATE)
        with torch.no_grad():
            for n, p in mw.named_parameters():
                if not n.startswith("blocks.") or _block_idx(n) not in SCALED:
                    continue
                if n.endswith("attn.qkv.weight"):
                    # utils.shuffle_weights shuffles attn.qk.weight (rows 0:2e) and
                    # attn.v.weight (rows 2e:3e) as SEPARATE pools -- not one flat shuffle.
                    # Pooling here silently widens v and was a bug in an earlier version.
                    e = p.data.shape[0] // 3
                    for lo, hi in [(0, 2 * e), (2 * e, 3 * e)]:
                        sl = p.data[lo:hi].flatten()
                        p.data[lo:hi].copy_(
                            sl[torch.randperm(sl.numel())].view_as(p.data[lo:hi]))
                    continue
                f = p.data.flatten()
                p.data.copy_(f[torch.randperm(f.numel())].view_as(p.data))
        return mw

    mw = _rand_model(args)                          # every remaining arm starts random
    if arm == "r":
        return mw                                   # baseline: nothing from the checkpoint
    with torch.no_grad():
        for n, p in mw.named_parameters():
            if not n.startswith("blocks.") or _block_idx(n) not in SCALED:
                continue
            t = tgt.get(n)
            if t is None or t.numel() != p.numel():
                continue
            is_ln = ".norm1." in n or ".norm2." in n

            if arm == "ftbnorm":                    # rescale to the target's norm
                cur, tn = p.data.norm().item(), t.norm().item()
                if cur > 0 and tn > 0:
                    p.data.mul_(tn / cur)
                continue

            if p.dim() < 2:                         # 1-D handling differs per arm
                if arm in ("ftbqm", "ftbqmvo"):      # no 1-D params at all
                    continue
                if arm == "ftbqmbias" and is_ln:
                    continue
                if arm in ("ftbqmln", "ftbqmlnvo") and not is_ln:   # LayerNorm 1-D only
                    continue
                d = t.flatten().float()
                if arm == "ftbqm1dpar":
                    draw = torch.randn(d.numel()) * d.std() + d.mean()
                    p.data.copy_(draw.view_as(p.data).to(p.data.dtype))
                else:
                    p.data.copy_(d[torch.randperm(d.numel())].view_as(p.data).to(p.data.dtype))
                continue

            # 2-D: rank map the target's sorted values onto the random tensor's order.
            # ftbqm1dv matches the qk and v slices of the fused qkv as SEPARATE pools,
            # mirroring ftb4e3's shuffle (docs 3.10.9.5).
            # ftbqm1dqk / ftbqm1dvo match ONE slice against its own target and leave the
            # other on the POOLED map, mirroring main.py:969-1001 exactly.
            # ftbqmvo / ftbqmlnvo are the v_only cells with no 1-D and LN-only 1-D respectively;
            # added 2026-09-04 so the full 3x2 grid of docs 0c.8 can be measured at init.
            QKV_MODE = {"ftbqm1dv": ["qk", "v"], "ftbqm1dqk": ["qk"], "ftbqm1dvo": ["v"],
                        "ftbqmvo": ["v"], "ftbqmlnvo": ["v"]}
            if arm in QKV_MODE and n.endswith("attn.qkv.weight"):
                e = p.data.shape[0] // 3
                want = QKV_MODE[arm]
                spans = [(0, 2 * e, "qk"), (2 * e, 3 * e, "v")]
                if len(want) == 1:
                    ps, _ = torch.sort(t.flatten().float())
                    oa = torch.argsort(p.data.flatten().float())
                    tmp = torch.empty_like(ps); tmp[oa] = ps
                    tmp = tmp.view_as(p.data)
                    for lo, hi, tag in spans:
                        if tag not in want:
                            p.data[lo:hi].copy_(tmp[lo:hi].to(p.data.dtype))
                for lo, hi, tag in [x for x in spans if x[2] in want]:
                    d, _ = torch.sort(t[lo:hi].flatten().float())
                    o = torch.argsort(p.data[lo:hi].flatten().float())
                    z = torch.empty_like(d); z[o] = d
                    p.data[lo:hi].copy_(z.view_as(p.data[lo:hi]).to(p.data.dtype))
                continue
            ds, _ = torch.sort(t.flatten().float())
            order = torch.argsort(p.data.flatten().float())
            out = torch.empty_like(ds)
            out[order] = ds
            p.data.copy_(out.view_as(p.data).to(p.data.dtype))
    return mw


# ----------------------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def measure(model, x):
    """rho per block, matching engine.attention_residual_analysis."""
    stats = {}

    def wrap(i, blk):
        def fwd(t):
            r_in = t
            d_attn = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(t))))
            r_out = r_in + d_attn
            d_mlp = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(r_out))))
            out = r_out + d_mlp
            n = lambda z: torch.norm(z.float(), dim=-1)
            stats[i] = dict(
                rho_attn=float((n(d_attn) / (n(r_in) + 1e-8)).mean()),
                rho_mlp=float((n(d_mlp) / (n(r_out) + 1e-8)).mean()),
                rin=float(n(r_in).mean()), rout=float(n(out).mean()),
            )
            return out
        return fwd

    orig = []
    for i, blk in enumerate(model.blocks):
        orig.append(blk.forward)
        blk.forward = wrap(i, blk)
    model.eval()
    model(x)
    for blk, f in zip(model.blocks, orig):
        blk.forward = f
    return stats


def main():
    ap = argparse.ArgumentParser(parents=[__import__("main").get_args_parser()],
                                 add_help=False, conflict_handler="resolve")
    ap.add_argument("--out", default="plots/cache/init_rho.json")
    ap.add_argument("--n_images", type=int, default=256)
    args = ap.parse_args()
    args.nb_classes = 1000
    # main.py normalises these from strings into dicts/lists after parse_args (main.py:600-760).
    # This script bypasses main(), so replicate the empty forms it would produce.
    for k in ["skip_attn_segments", "weight_shuffle", "target_model_weight_shuffle",
              "init_method_copied_blocks", "attention_residual_scaling",
              "attention_out_scaling", "learning_rate_scaling_params"]:
        setattr(args, k, {})
    for k in ["random_blocks", "clip_outlier_blocks", "delete_blocks",
              "init_method_scaled_blocks", "freeze_blocks", "hold_back_blocks"]:
        setattr(args, k, [])
    for k in ["skip_load_blocks", "skip_load_block_attributes", "freeze_block_attributes"]:
        v = getattr(args, k, "")
        setattr(args, k, [x for x in v.split(",")] if isinstance(v, str) and v else [])
    # utils.pr_load_model touches distributed state that main() sets up via
    # utils.init_distributed_mode; this script runs single-process.
    args.distributed = False
    args.gpu, args.rank, args.world_size = 0, 0, 1
    torch.manual_seed(0)

    from datasets import build_dataset
    ds, _ = build_dataset(is_train=False, args=args)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.n_images, shuffle=False,
                                         num_workers=8, pin_memory=True)
    x = next(iter(loader))[0].cuda()
    print(f"measuring on {tuple(x.shape)} real val images", flush=True)

    tgt = _target_params(args)
    out = {}
    for arm in ARMS:
        torch.manual_seed(0)
        if arm == "ftbrho":
            m = build_recipe(args, x)
        elif arm == "ftb4o":
            m = build_ftb4o(args, x, _proc_model(args, random_blocks=[]))
        else:
            m = build_arm(arm, args, tgt).cuda()
        out[arm] = measure(m, x)
        del m
        torch.cuda.empty_cache()
        ra = " ".join(f"{out[arm][i]['rho_attn']:.3f}" for i in range(12))
        rm = " ".join(f"{out[arm][i]['rho_mlp']:.3f}" for i in range(12))
        print(f"{arm:11} rho_attn: {ra}", flush=True)
        print(f"{'':11} rho_mlp : {rm}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
