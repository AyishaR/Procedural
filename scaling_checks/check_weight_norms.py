"""Where do the upscaled-random weights land relative to the proc-pretrained weights?

The v/proj multipliers look huge next to random init (up to 11.5x), but proc weights are
themselves several times larger than random init, so the gap between the two arms is much
smaller. This prints that gap per block.

Usage:  python scaling_checks/check_weight_norms.py [--ckpt PATH] [--factors "2.41,..."]
"""
import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model

import models.vision_transformer  # noqa: F401

DROP_KEYS = ['head.weight', 'head.bias', 'cls_token', 'pos_embed',
             'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']
# factors actually applied by run ft_29377630_ftb6s, blocks 6..11 (see logs/)
DEFAULT_FACTORS = "2.4094460424570183,2.9437560758632864,3.6011419496704486," \
                  "5.855314681883812,9.904411098229593,11.492624795198438"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/pr_vitb_n/pr_6066174_final.pth")
    p.add_argument("--blocks", default="6,7,8,9,10,11")
    p.add_argument("--factors", default=DEFAULT_FACTORS,
                   help="v/proj multiplier per block, as printed by 'Scale for layer N' in the run log")
    args = p.parse_args()

    blocks = [int(b) for b in args.blocks.split(",")]
    factors = dict(zip(blocks, [float(f) for f in args.factors.split(",")]))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state", ck.get("model", ck))
    for k in DROP_KEYS:
        sd.pop(k, None)

    torch.manual_seed(0)
    rnd = create_model("vit_base", pretrained=False, num_classes=1000)
    proc = create_model("vit_base", pretrained=False, num_classes=1000)
    proc.load_state_dict(sd, strict=False)
    D = rnd.blocks[0].attn.proj.weight.shape[0]

    def norms(m, i):
        b = m.blocks[i]
        return b.attn.qkv.weight.data[2 * D:3 * D, :].norm().item(), b.attn.proj.weight.data.norm().item()

    print(f"{'blk':>3} {'s':>6} | {'||Wv|| rand':>11} {'x s':>8} {'proc':>8} {'over':>6} |"
          f" {'||Wp|| rand':>11} {'x s':>8} {'proc':>8} {'over':>6}")
    for i in blocks:
        vr, pr = norms(rnd, i)
        vp, pp = norms(proc, i)
        s = factors[i]
        print(f"{i:>3} {s:6.2f} | {vr:11.2f} {vr*s:8.2f} {vp:8.2f} {vr*s/vp:5.2f}x |"
              f" {pr:11.2f} {pr*s:8.2f} {pp:8.2f} {pr*s/pp:5.2f}x")
    print("\n'over' = upscaled-random weight norm / proc weight norm (1.0 = perfect match)")
    print("For the downscale direction the mirror number is (proc x s) / random-init norm.")


if __name__ == "__main__":
    main()
