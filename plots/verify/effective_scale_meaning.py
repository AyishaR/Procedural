"""What the effective scale e = rms(W diag(gamma)) / 0.02 does and does not say (docs/proc_init_recipe.md, section 3).

LayerNorm fixes each token's norm (||xhat||^2 = d), not the distribution of directions, so
    E ||M xhat||^2 = tr(M Sigma M^T),   Sigma = E[xhat xhat^T],   tr(Sigma) = d,
equals ||M||_F^2 only if M is independent of xhat (any random matrix) or Sigma is the identity (it is not). Three tables, for
the checkpoint prefix (timm seed 0 + the checkpoint's blocks 0-7), a recipe initialisation and plain timm:

  A  anisotropy of the standardised stream (largest eigenvalue of Sigma / d; isotropic 1/768) and
     R = E||M xhat||^2 / ||M||_F^2 per slice, M = W diag(gamma); R_gain = the same for a random matrix behind the model's gain.
  B  each input-side output (q, k, v, fc1 pre-activation; W y + b, y = LayerNorm(stream)) split into the part common to all
     tokens and the token-specific part: common share of the energy, rms of the token-specific part, share of ||W||_F^2 on
     the stream's mean direction (random 1/768).
  C  weights only, checkpoint: share of ||M||_F^2 in the top 1 / 8 singular directions (random 768x768: 0.5% / 3.7%), the
     scale left without the top 8, the median singular value relative to a Gaussian matrix of std 0.02, per-head q / k scales.

Images: `--images` of the VALIDATION folder by default (listing the training folder takes minutes; the quantities do not depend
on the draw), evaluation transform, utils.calibration_images. CPU is enough (about 3 minutes).

usage: .venv/bin/python plots/verify/effective_scale_meaning.py [--recipe_init X.pth] [--checkpoint C.pth] [--split val|train]
  --recipe_init: a state dict of a recipe initialisation, e.g. debug/out/ftbanaperab7i/init.pth (bash debug/run_debug.sh main
  ftbanaperab7i --save_init: the launched protocol) or results/init_dumps/ftbanaperab7i_s0.pth (calibrated on validation images)."""
import argparse, contextlib, io, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M_, utils
    from datasets import build_transform
from torchvision import datasets as tvd
from extract_profile import load_state_dict

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="results/pr_vitb_n/pr_6066174_final.pth")
ap.add_argument("--recipe_init", default="debug/out/ftbanaperab7i/init.pth")
ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet")
ap.add_argument("--split", default="val", choices=("val", "train"))
ap.add_argument("--images", type=int, default=32)
ap.add_argument("--last_block", type=int, default=7)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
E, BLOCKS = 768, range(a.last_block + 1)
args = M_.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tvd.ImageFolder(os.path.join(a.data_path, a.split))
images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), a.images, 0)
checkpoint = load_state_dict(a.checkpoint)
rms = lambda t: float(t.double().pow(2).mean().sqrt())


def build(kind):
    torch.manual_seed(0); model = utils.build_model(args)
    if kind == "prefix":
        model.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith("blocks.") and int(k.split(".")[1]) in BLOCKS}, strict=False)
    elif kind == "recipe":
        model.load_state_dict(torch.load(a.recipe_init, map_location="cpu", weights_only=True))
    return model.eval()


def slices(blk):
    W, b = blk.attn.qkv.weight, blk.attn.qkv.bias
    return (("q", "1", W[:E], b[:E]), ("k", "1", W[E:2 * E], b[E:2 * E]), ("v", "1", W[2 * E:], b[2 * E:]), ("fc1", "2", blk.mlp.fc1.weight, blk.mlp.fc1.bias))


@torch.no_grad()
def tables_a_b(kind):
    model = build(kind); stream = utils.block_input_stream(model, images); rows_a, rows_b = [], []
    for b in BLOCKS:
        blk = model.blocks[b]
        raw = {"1": stream, "2": stream + blk.attn(blk.norm1(stream))}            # what norm1 / norm2 standardise
        read = {"1": blk.norm1(stream), "2": utils.fc1_input(blk, stream)}          # what q, k, v / fc1 read
        gain = {"1": blk.norm1.weight.double(), "2": blk.norm2.weight.double()}
        xhat = {i: F.layer_norm(raw[i], (E,), eps=1e-6).reshape(-1, E).double() for i in raw}
        top = {i: float(torch.linalg.eigvalsh(xhat[i].T @ xhat[i] / len(xhat[i]))[-1] / E) for i in raw}
        r_gain = {i: float((xhat[i] * gain[i]).pow(2).sum(1).mean() / gain[i].pow(2).sum()) for i in raw}
        cells_a, cells_b = [], []
        for _, i, W, bias in slices(blk):
            M = W.double() * gain[i][None, :]
            cells_a.append(float((xhat[i] @ M.T).pow(2).sum(1).mean() / M.pow(2).sum()))
            out = F.linear(read[i], W, bias).reshape(-1, W.shape[0]).double()
            common = out.mean(0, keepdim=True)
            direction = F.normalize(read[i].reshape(-1, E).mean(0), dim=0)
            cells_b.append((float(common.pow(2).sum() / out.pow(2).sum(1).mean()), rms(out - common), float((W @ direction).pow(2).sum() / W.pow(2).sum()), float(out.mean())))
        y = read["1"].reshape(-1, E).double()
        rows_a.append((b, top, r_gain, cells_a)); rows_b.append((b, float(y.mean(0).pow(2).sum() / y.pow(2).sum(1).mean()), cells_b))
        stream = blk(stream)
    print(f"\n== A  {kind}: top eigenvalue share of Sigma (isotropic {1 / E:.4f}), R_gain, R = E||M xhat||^2 / ||M||_F^2")
    print("block | norm1: top-eig R_gain |    R_q     R_k     R_v | norm2: top-eig R_gain |  R_fc1")
    for b, top, r_gain, c in rows_a:
        print(f"{b:5d} |      {top['1']:6.3f} {r_gain['1']:6.2f} | {c[0]:6.2f}  {c[1]:6.2f}  {c[2]:6.2f} |      {top['2']:6.3f} {r_gain['2']:6.2f} | {c[3]:6.2f}")
    print(f"== B  {kind}: c = common share of the output energy, s = rms of the token-specific part, W.c = share of ||W||_F^2 on the mean direction")
    print("block | stream c |    q: c      s    W.c |    k: c      s    W.c |    v: c      s    W.c |  fc1: c      s    W.c | fc1 pre-activation mean")
    for b, stream_common, cells in rows_b:
        print(f"{b:5d} |   {stream_common:5.2f}  | " + " | ".join(f"{c:6.3f} {s:6.3f} {w:6.3f}" for c, s, w, _ in cells) + f" | {cells[3][3]:+.2f}")


def table_c():
    reference = {}
    def gaussian_median(shape):
        if shape not in reference:
            reference[shape] = float(torch.linalg.svdvals(torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.02).median())
        return reference[shape]
    print(f"\n== C  {a.checkpoint}, weights only (gain folded)")
    print("block slice |    e   | top1  top8 | e without top 8 (x e) | median sv / Gaussian 0.02 | per-head e: min..max | per-head e_q*e_k: min..max (pooled)")
    for b in BLOCKS:
        W = checkpoint[f"blocks.{b}.attn.qkv.weight"].double(); g1 = checkpoint[f"blocks.{b}.norm1.weight"].double(); g2 = checkpoint[f"blocks.{b}.norm2.weight"].double()
        parts = {"q": W[:E] * g1, "k": W[E:2 * E] * g1, "v": W[2 * E:] * g1, "fc1": checkpoint[f"blocks.{b}.mlp.fc1.weight"].double() * g2}
        for name, M in parts.items():
            sv = torch.linalg.svdvals(M); total = float(sv.pow(2).sum()); e = rms(M) / 0.02
            without = ((total - float(sv[:8].pow(2).sum())) / M.numel()) ** 0.5 / 0.02
            heads = ""
            if name in ("q", "k"):
                per = [rms(M[h * 64:(h + 1) * 64]) / 0.02 for h in range(12)]; heads = f"{min(per):5.2f}..{max(per):5.2f}"
                if name == "k":
                    product = [rms(parts["q"][h * 64:(h + 1) * 64]) / 0.02 * k for h, k in enumerate(per)]
                    heads += f"          | {min(product):5.2f}..{max(product):5.2f} ({rms(parts['q']) / 0.02 * e:.2f})"
            print(f"{b:5d} {name:5s} | {e:6.3f} | {float(sv[0] ** 2) / total:5.3f} {float(sv[:8].pow(2).sum()) / total:5.3f} | {without:6.3f} ({without / e:4.2f})"
                  f"         | {float(sv.median()) / gaussian_median(tuple(M.shape)):6.3f}                    | {heads}")


for kind in ("prefix", "recipe", "timm"):
    if kind == "recipe" and not os.path.exists(a.recipe_init):
        print(f"\n(recipe tables skipped: {a.recipe_init} not found, see --recipe_init)"); continue
    tables_a_b(kind)
table_c()
