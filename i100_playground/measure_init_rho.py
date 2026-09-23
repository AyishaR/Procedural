"""Measure rho = ||Delta_sublayer|| / ||r_in|| per block at INIT for different weight
initialisations, with no training and no data.

The recipe in docs/i100_late_block_scaling.md 3.12.3 assumes the late blocks start out
writing too little: timm's plain ViT init gives rho ~0.107 in blocks 9-11, and pushing it to
~1.4 is worth +2.68. Whether that headroom exists elsewhere depends on whether rho ~0.107 is
a timm quirk or a general property of standard inits. This measures it directly.

Inits compared (all applied to the same ViT-S architecture, so only the init differs):
  timm        - what every arm in the study used
  nanogpt     - GPT-2 style: normal(0, 0.02), output projections scaled by 1/sqrt(2L)
  xavier      - Xavier/Glorot uniform on all 2-D weights
  small       - normal(0, 0.02) everywhere, no output-projection scaling

Run:  .venv/bin/python i100_playground/measure_init_rho.py
"""
import math
import warnings

import torch

warnings.filterwarnings("ignore")

import models.vision_transformer  # noqa: F401  (registers vit_small)
from timm.models import create_model


def apply_init(model, scheme):
    if scheme == "timm":
        return model
    depth = len(model.blocks)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() < 2:
                continue
            if scheme == "nanogpt":
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
                # GPT-2: scale residual output projections by 1/sqrt(2*n_layer)
                if name.endswith("attn.proj.weight") or name.endswith("mlp.fc2.weight"):
                    p.mul_(1.0 / math.sqrt(2 * depth))
            elif scheme == "xavier":
                torch.nn.init.xavier_uniform_(p)
            elif scheme == "small":
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
    return model


@torch.no_grad()
def measure(model, x):
    """rho per block for attn and mlp sublayers, matching engine.attention_residual_analysis:
    attn is measured against the block input, mlp against the post-attention stream."""
    out = {}
    h = model.patch_embed(x)
    if getattr(model, "cls_token", None) is not None:
        h = torch.cat([model.cls_token.expand(h.shape[0], -1, -1), h], dim=1)
    if getattr(model, "pos_embed", None) is not None:
        h = h + model.pos_embed[:, : h.shape[1]]
    for i, blk in enumerate(model.blocks):
        r_in = h
        d_attn = blk.ls1(blk.attn(blk.norm1(r_in))) if not isinstance(blk.ls1, torch.nn.Identity) \
            else blk.attn(blk.norm1(r_in))
        r_mid = r_in + d_attn
        d_mlp = blk.mlp(blk.norm2(r_mid))
        h = r_mid + d_mlp
        n = lambda t: t.norm(dim=-1).mean().item()
        out[i] = (n(d_attn) / max(n(r_in), 1e-8), n(d_mlp) / max(n(r_mid), 1e-8), n(r_in))
    return out


def main():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 224, 224)
    schemes = ["timm", "nanogpt", "xavier", "small"]
    res = {}
    for s in schemes:
        m = create_model("vit_small", pretrained=False, num_classes=100)
        apply_init(m, s)
        m.eval()
        res[s] = measure(m, x)

    print("rho_attn per block (target for the recipe: ~1.4 in the last 3 blocks)")
    print(f'{"blk":>3s} ' + " ".join(f"{s:>10s}" for s in schemes))
    for b in range(12):
        print(f"{b:3d} " + " ".join(f"{res[s][b][0]:10.3f}" for s in schemes))
    print()
    print("rho_mlp per block")
    print(f'{"blk":>3s} ' + " ".join(f"{s:>10s}" for s in schemes))
    for b in range(12):
        print(f"{b:3d} " + " ".join(f"{res[s][b][1]:10.3f}" for s in schemes))
    print()
    print("mean rho_attn over blocks 9-11, and implied amplification to reach 1.4:")
    for s in schemes:
        m = sum(res[s][b][0] for b in (9, 10, 11)) / 3
        print(f"  {s:10s} {m:7.4f}   x{1.4 / max(m, 1e-8):7.1f}")


if __name__ == "__main__":
    main()
