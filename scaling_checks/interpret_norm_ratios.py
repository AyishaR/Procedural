"""What is procedural pretraining actually doing to the residual stream?

Raw norm ratios are hard to read, so decompose each sublayer's additive update Delta
against the stream r_in it is written into:

    alpha = 1 + rho*cos      surviving fraction of the incoming stream
    beta  = rho*sqrt(1-cos²) newly written content, in units of ||r_in||
    R     = sqrt(alpha²+beta²) = ||r_out|| / ||r_in||     (the sublayer's input->output scaling)

with rho = ||Delta|| / ||r_in|| and cos = cos(r_in, Delta).

alpha ~ 1 -> the sublayer refines the stream;  alpha ~ 0 -> it erases it;
alpha < 0 -> it inverts it.  beta says how much is written in its place.

Three models are compared on the same images:
  random   - vit_base at init
  ftb6     - the hybrid the experiment actually initialises: proc norm1+attn in blocks
             6-11, everything else random (this is what the norm ratios in the run logs
             describe)
  proc     - every proc weight in the checkpoint, including its own patch/pos embedding

Usage:  python scaling_checks/interpret_norm_ratios.py [--ckpt PATH] [--n 256] [--tokens all|cls|patch]
"""
import argparse, glob, os, sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timm.models import create_model

import models.vision_transformer  # noqa: F401
import utils

PR_DROP = ['head.weight', 'head.bias', 'cls_token', 'pos_embed',
           'patch_embed.proj.weight', 'patch_embed.proj.bias', 'norm.weight', 'norm.bias']


def load_images(root, n, dev):
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    files = sorted(glob.glob(os.path.join(root, "*/*.JPEG")))
    files = files[::max(1, len(files) // n)][:n]
    return torch.stack([tf(Image.open(f).convert("RGB")) for f in files]).to(dev)


def acts(model, x, bs=64):
    out = {i: {k: [] for k in ('inp', 'attn_out', 'attn', 'blk')} for i in range(len(model.blocks))}
    for i in range(0, x.shape[0], bs):
        with utils.HookCollector(model) as a:
            with torch.no_grad():
                model(x[i:i + bs])
        for b in out:
            for k in out[b]:
                out[b][k].append(a[b][k].float().cpu())
    return {b: {k: torch.cat(v) for k, v in d.items()} for b, d in out.items()}


def decompose(r_in, delta, sel):
    """rho, cos, alpha, beta, R for one sublayer, averaged over the selected tokens."""
    r_in, delta = r_in[:, sel], delta[:, sel]
    n_in = r_in.norm(dim=-1)
    rho = delta.norm(dim=-1) / (n_in + 1e-8)
    cos = F.cosine_similarity(r_in, delta, dim=-1)
    alpha = 1 + rho * cos
    beta = rho * (1 - cos ** 2).clamp(min=0).sqrt()
    R = (r_in + delta).norm(dim=-1) / (n_in + 1e-8)
    return [v.mean().item() for v in (rho, cos, alpha, beta, R)] + [n_in.mean().item()]


def report(name, model, x, sel):
    a = acts(model, x)
    print(f"\n=== {name} " + "=" * (66 - len(name)))
    print(f"{'':<4}{'ATTENTION SUBLAYER':^46}|{'MLP SUBLAYER':^38}")
    print(f"{'blk':>3} {'rho':>6} {'cos':>7} {'alpha':>7} {'beta':>6} {'R':>6} {'||r_in||':>9} |"
          f" {'rho':>6} {'cos':>7} {'alpha':>7} {'beta':>6} {'R':>6}")
    for i in range(len(model.blocks)):
        ar = decompose(a[i]['inp'], a[i]['attn_out'], sel)
        mr = decompose(a[i]['attn'], a[i]['blk'] - a[i]['attn'], sel)
        print(f"{i:>3} {ar[0]:6.3f} {ar[1]:+7.3f} {ar[2]:+7.3f} {ar[3]:6.3f} {ar[4]:6.3f} {ar[5]:9.1f} |"
              f" {mr[0]:6.3f} {mr[1]:+7.3f} {mr[2]:+7.3f} {mr[3]:6.3f} {mr[4]:6.3f}")
    print("alpha = surviving fraction of the incoming stream, beta = newly written content")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/pr_vitb_n/pr_6066174_final.pth")
    p.add_argument("--val", default="/data/datasets/ILSVRC2012/val")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--tokens", choices=["all", "cls", "patch"], default="all")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, " tokens:", args.tokens, flush=True)
    x = load_images(args.val, args.n, dev)
    sel = slice(0, None) if args.tokens == "all" else (slice(0, 1) if args.tokens == "cls" else slice(1, None))

    def build():
        return create_model("vit_base", pretrained=False, num_classes=1000, drop_path_rate=0.0).eval().to(dev)

    torch.manual_seed(args.seed)
    base_sd = {k: v.clone() for k, v in build().state_dict().items()}

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    full = ck.get("state", ck.get("model", ck))            # every proc weight, incl. embeddings
    stripped = {k: v for k, v in full.items() if k not in PR_DROP}
    hybrid = {}                                            # what ftb6 actually loads
    for k, v in stripped.items():
        if not k.startswith("blocks."):
            continue
        bi = int(k.split(".")[1])
        if bi < 6 or k.split(f"blocks.{bi}.")[1].startswith(("norm2", "mlp")):
            continue
        hybrid[k] = v

    R = build(); R.load_state_dict(base_sd)
    H = build(); H.load_state_dict(base_sd); H.load_state_dict(hybrid, strict=False)

    # the proc model is a sequence model (196 positions, 128-way head, no CLS token), so its
    # pos_embed / head cannot be loaded into the image ViT -- keep every block instead.
    ref = R.state_dict()
    allblocks = {k: v for k, v in full.items()
                 if k in ref and ref[k].shape == v.shape and (k.startswith("blocks.") or k.startswith("norm."))}
    P = build(); P.load_state_dict(base_sd); P.load_state_dict(allblocks, strict=False)
    print(f"proc keys loaded: hybrid={len(hybrid)}, all-blocks={len(allblocks)} "
          f"(skipped by shape: {sorted(set(full) - set(allblocks) - set(PR_DROP))})")

    report("random init (vit_base)", R, x, sel)
    report("ftb6 hybrid: proc norm1+attn in blocks 6-11, rest random", H, x, sel)
    report("all 12 proc blocks (embeddings/head random)", P, x, sel)
    print("\nNote: the proc model was pretrained on k-Dyck sequences; ImageNet images are "
          "off-distribution for it.\nThe hybrid rows are the ones that describe the actual "
          "experiment initialisation.")


if __name__ == "__main__":
    main()
