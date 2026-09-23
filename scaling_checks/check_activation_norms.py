"""Do the activation norms of the upscaled-random arm match the proc-init arm?

Reproduces the ftb6 (proc attn in blocks 6-11) vs ftb6s (all random + matched attn
delta-norm ratio) initialisations exactly, runs the repo's sequential matching procedure,
then compares the residual stream block by block on real ImageNet val images.

The matching enforces one scalar per block: mean ||Delta_attn|| / ||r_in||. Everything
else -- the direction of the attention update, its alignment with the residual stream,
the resulting ||r_out||, and the MLP contribution -- is free to differ. This script shows
how much it actually differs.

Usage:  python scaling_checks/check_activation_norms.py [--ckpt PATH] [--n 256]
"""
import argparse, glob, math, os, sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model

import models.vision_transformer  # noqa: F401
import utils

DROP_KEYS = ['head.weight', 'head.bias', 'cls_token', 'pos_embed',
             'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']


def load_images(root, n, dev):
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    files = sorted(glob.glob(os.path.join(root, "*/*.JPEG")))
    files = files[::max(1, len(files) // n)][:n]
    print(f"{len(files)} val images from {root}")
    return torch.stack([tf(Image.open(f).convert("RGB")) for f in files]).to(dev)


def acts(model, x, bs=64):
    """Per-block activations, concatenated over mini-batches."""
    out = {i: {k: [] for k in ('inp', 'attn_out', 'attn', 'blk')} for i in range(len(model.blocks))}
    for i in range(0, x.shape[0], bs):
        with utils.HookCollector(model) as a:
            with torch.no_grad():
                model(x[i:i + bs])
        for b in out:
            for k in out[b]:
                out[b][k].append(a[b][k].float().cpu())
    return {b: {k: torch.cat(v) for k, v in d.items()} for b, d in out.items()}


def attn_ratio(a, i):
    """||Delta_attn|| / ||r_in||  -- what the repo currently matches."""
    return (a[i]['attn_out'].norm(dim=-1) / (a[i]['inp'].norm(dim=-1) + 1e-8)).mean().item()


def out_in_ratio(a, i):
    """||r_in + Delta_attn|| / ||r_in||  -- how much the sublayer scales its input."""
    return (a[i]['attn'].norm(dim=-1) / (a[i]['inp'].norm(dim=-1) + 1e-8)).mean().item()


def solve_out_in(a, i, target, lo=1e-4, hi=1e4, iters=60):
    """Find the factor f on Delta_attn that makes mean_t ||r_in + f*Delta||/||r_in|| = target.

    Delta is exactly linear in f (biases are zero at random init), so this needs no extra
    forward passes: we solve on the cached r_in / Delta tensors. Returns (f, achievable_min).
    """
    rin, d = a[i]['inp'], a[i]['attn_out']
    n_in = rin.norm(dim=-1)

    def stat(f):
        return ((rin + f * d).norm(dim=-1) / (n_in + 1e-8)).mean().item()

    # with cos<0 the curve dips before rising; the reachable floor is mean sin(r_in, Delta)
    cos = torch.nn.functional.cosine_similarity(rin, d, dim=-1)
    floor = (1 - cos ** 2).clamp(min=0).sqrt().mean().item()
    if target <= floor:
        return None, floor
    while hi - lo > 1e-9 and iters > 0:           # bisect on the increasing branch
        mid = (lo * hi) ** 0.5
        if stat(mid) < target:
            lo = mid
        else:
            hi = mid
        iters -= 1
    return (lo * hi) ** 0.5, floor


def stats(a, i):
    rin, d, rout, blk = a[i]['inp'], a[i]['attn_out'], a[i]['attn'], a[i]['blk']
    mlpd = blk - rout
    return dict(
        rin=rin.norm(dim=-1).mean().item(),
        dn=d.norm(dim=-1).mean().item(),
        ratio=(d.norm(dim=-1) / (rin.norm(dim=-1) + 1e-8)).mean().item(),
        cos=torch.nn.functional.cosine_similarity(rin, d, dim=-1).mean().item(),
        rout=rout.norm(dim=-1).mean().item(),
        stream=(rout.norm(dim=-1) / (rin.norm(dim=-1) + 1e-8)).mean().item(),
        mlpr=(mlpd.norm(dim=-1) / (rout.norm(dim=-1) + 1e-8)).mean().item(),
        out=blk.norm(dim=-1).mean().item(),
        cls=blk[:, 0].norm(dim=-1).mean().item(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/pr_vitb_n/pr_6066174_final.pth")
    p.add_argument("--val", default="/data/datasets/ILSVRC2012/val")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--blocks", default="6,7,8,9,10,11")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", choices=["delta_in", "out_in"], default="delta_in",
                   help="delta_in: match ||Delta||/||r_in|| (what the repo does). "
                        "out_in: match ||r_out||/||r_in||, i.e. how much the sublayer scales its input.")
    args = p.parse_args()

    blocks = [int(b) for b in args.blocks.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    x = load_images(args.val, args.n, dev)

    def build():
        m = create_model("vit_base", pretrained=False, num_classes=1000, drop_path_rate=0.0)
        return m.eval().to(dev)

    torch.manual_seed(args.seed)
    base_sd = {k: v.clone() for k, v in build().state_dict().items()}

    # proc checkpoint, filtered exactly like pr_load_model with the ftb6 args:
    #   "pr" in path          -> drop head / cls_token / pos_embed / patch_embed
    #   --skip_norm true      -> drop final norm.*
    #   --random_blocks 0-5   -> drop those blocks entirely (stay random)
    #   --skip_load_blocks 6-11 with attrs norm2,mlp.* -> keep only norm1 + attn there
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state", ck.get("model", ck))
    for k in DROP_KEYS:
        sd.pop(k, None)
    keep = {}
    for k, v in sd.items():
        if not k.startswith("blocks."):
            continue
        bi = int(k.split(".")[1])
        if bi < min(blocks) or k.split(f"blocks.{bi}.")[1].startswith(("norm2", "mlp")):
            continue
        keep[k] = v
    print(f"proc keys loaded: {len(keep)} (blocks {min(blocks)}-{max(blocks)}: norm1 + attn only)")

    A = build(); A.load_state_dict(base_sd); A.load_state_dict(keep, strict=False)  # ftb6
    B = build(); B.load_state_dict(base_sd)                                          # ftb6s

    # --- the repo's sequential matching procedure (main.py:881-914)
    stat_fn = attn_ratio if args.target == "delta_in" else out_in_ratio
    aA0 = acts(A, x)
    target = {i: stat_fn(aA0, i) for i in blocks}
    print(f"\nmatching target = {args.target}")
    for i in blocks:
        aB = acts(B, x)
        cur = stat_fn(aB, i)
        if args.target == "delta_in":
            f = target[i] / cur                       # Delta is linear in f
            note = ""
        else:
            f, floor = solve_out_in(aB, i, target[i])
            if f is None:
                print(f"blk {i}: target={target[i]:.4f} UNREACHABLE (floor={floor:.4f}) -- skipped")
                continue
            note = f" floor={floor:.4f}"
        s = math.sqrt(f)
        utils.scale_layer_weights(B, [i], {"norm1": 1.0, "qk": 1.0, "v": s, "proj": s})
        print(f"blk {i}: target={target[i]:.4f} current={cur:.4f} -> Delta x{f:.2f}, s={s:.3f}{note}",
              flush=True)

    # --- compare
    aA, aB = acts(A, x), acts(B, x)
    keys = [('rin', '||r_in||'), ('dn', '||d_attn||'), ('ratio', 'attn delta ratio ||d||/||r_in||'),
            ('stream', 'attn stream scaling ||r_out||/||r_in||'), ('cos', 'cos(r_in,d)'),
            ('rout', '||r_out||'), ('mlpr', 'mlp ratio'),
            ('out', '||blk out||'), ('cls', '||CLS out||')]
    for k, name in keys:
        print(f"\n{name}")
        print(f"{'blk':>3} {'proc':>10} {'scaled':>10} {'diff':>8}")
        for i in range(len(A.blocks)):
            va, vb = stats(aA, i)[k], stats(aB, i)[k]
            d = f"{100*(vb/va-1):+7.1f}%" if abs(va) > 1e-9 else "     n/a"
            matched_key = 'ratio' if args.target == 'delta_in' else 'stream'
            mark = "  <-- matched" if (i in blocks and k == matched_key) else ""
            print(f"{i:>3} {va:10.4f} {vb:10.4f} {d}{mark}")


if __name__ == "__main__":
    main()
