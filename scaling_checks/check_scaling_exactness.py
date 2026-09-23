"""Does utils.scale_layer_weights actually multiply the measured delta-norm ratio by r?

The init methods `*_match_*_delta_norms` scale the V-slice of qkv.weight and
attn.proj.weight by sqrt(r) (=> Delta_attn * r) and mlp.fc2.weight by r (=> Delta_mlp * r).
Biases are NOT scaled, which makes the match inexact whenever they are non-zero:

    Delta_attn(scaled) = s^2 (W_p A W_v x) + s (W_p A b_v) + b_p

This script measures the resulting error for both directions, and compares against an
exact alternative (scale the output projection's weight AND bias by r).

Usage:  python scaling_checks/check_scaling_exactness.py [--ckpt PATH]
"""
import argparse, copy, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model

import models.vision_transformer  # noqa: F401  (registers vit_base / vit_tiny)
import utils

DROP_KEYS = ['head.weight', 'head.bias', 'cls_token', 'pos_embed',
             'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']


def ratios(model, blk, x):
    """(attn delta / block input, mlp delta / attn-residual output), mean over tokens."""
    with utils.HookCollector(model) as acts:
        with torch.no_grad():
            model(x)
    rin, delta = acts[blk]['inp'], acts[blk]['attn_out']
    rout, mlpd = acts[blk]['attn'], acts[blk]['blk'] - acts[blk]['attn']
    return ((delta.norm(dim=-1) / (rin.norm(dim=-1) + 1e-8)).mean().item(),
            (mlpd.norm(dim=-1) / (rout.norm(dim=-1) + 1e-8)).mean().item())


def compare(model, blk, x, targets, tag):
    a0, m0 = ratios(model, blk, x)
    print(f"\n[{tag}] baseline attn ratio={a0:.5f}  mlp ratio={m0:.5f}")
    print(f"{'requested r':>12} | {'attn repo':>10} {'attn exact':>11} | {'mlp repo':>9} {'mlp exact':>10}")
    for r in targets:
        m1 = copy.deepcopy(model)                                     # repo recipe, attn
        utils.scale_layer_weights(m1, [blk], {"norm1": 1.0, "qk": 1.0,
                                              "v": math.sqrt(r), "proj": math.sqrt(r)})
        a_repo = ratios(m1, blk, x)[0] / a0

        m2 = copy.deepcopy(model)                                     # repo recipe, mlp
        utils.scale_layer_weights(m2, [blk], {"norm2": 1.0, "fc1": 1.0, "fc2": r})
        m_repo = ratios(m2, blk, x)[1] / m0

        m3 = copy.deepcopy(model)                                     # exact: proj w + b
        m3.blocks[blk].attn.proj.weight.data *= r
        m3.blocks[blk].attn.proj.bias.data *= r
        a_exact = ratios(m3, blk, x)[0] / a0

        m4 = copy.deepcopy(model)                                     # exact: fc2 w + b
        m4.blocks[blk].mlp.fc2.weight.data *= r
        m4.blocks[blk].mlp.fc2.bias.data *= r
        m_exact = ratios(m4, blk, x)[1] / m0

        print(f"{r:12.5f} | {100*(a_repo/r-1):+9.1f}% {100*(a_exact/r-1):+10.1f}% |"
              f" {100*(m_repo/r-1):+8.1f}% {100*(m_exact/r-1):+9.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/pr_vitb_n/pr_6066174_final.pth")
    p.add_argument("--block", type=int, default=11)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224, device=dev)

    # --- 1. random init: timm zero-inits every Linear bias, so scaling is exact
    torch.manual_seed(0)
    rnd = create_model("vit_base", pretrained=False, num_classes=1000, drop_path_rate=0.0).eval().to(dev)
    b = rnd.blocks[args.block]
    print(f"random init, block {args.block}: |qkv.bias|_max={b.attn.qkv.bias.abs().max():.3g} "
          f"|proj.bias|_max={b.attn.proj.bias.abs().max():.3g}  <- zero => upscale direction is exact")
    compare(rnd, args.block, x, [2.0, 4.0, 132.0], "random init (upscale direction)")

    # --- 2. proc-pretrained: non-zero biases, error grows as r gets small
    if not os.path.exists(args.ckpt):
        print(f"\n[skip] checkpoint not found: {args.ckpt}")
        return
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state", ck.get("model", ck))
    for k in DROP_KEYS:
        sd.pop(k, None)
    torch.manual_seed(0)
    proc = create_model("vit_base", pretrained=False, num_classes=1000, drop_path_rate=0.0).eval().to(dev)
    missing, unexpected = proc.load_state_dict(sd, strict=False)
    assert len(unexpected) == 0, unexpected
    pb = proc.blocks[args.block]
    print(f"\nproc ckpt, block {args.block}: ||proj.bias||={pb.attn.proj.bias.norm():.3f} vs "
          f"||proj.weight||={pb.attn.proj.weight.norm():.1f}")
    # r values a downscale_pr_* run would ask for (inverse of the factors ftb6s applied)
    compare(proc, args.block, x, [0.00757, 0.0102, 0.0289, 0.0771, 0.25, 4.0],
            "proc init (downscale direction)")


if __name__ == "__main__":
    main()
