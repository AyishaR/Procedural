"""How much do the joint-statistic targets depend on the random context they are measured in?

extract_profile.py measures the functional targets (attention entropy, fraction of active fc1 units, mean fc1 pre-activation,
attention / MLP write ratio, token cosine of block 0's output) on ONE prefix model (fresh timm parts from --seed) and ONE draw
of 256 training images (same seed). This script repeats the measurement and reports the spread:
  A  seed s = 0..4 for both the model's random parts and the image draw (what another --seed would give)
  B  model seed 0, image seeds 1, 2          (image draw alone)
  C  model seeds 1, 2, image seed 0          (random patch / position embeddings, class token alone)
usage: python plots/verify/target_stability.py CHECKPOINT [--blocks 0-7] [--images 256] [--data_path D]"""
import argparse, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
from torchvision import datasets as tv_datasets
import main as M, utils, extract_profile
from datasets import build_transform
ap = argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("--blocks", default="0-7"); ap.add_argument("--images", type=int, default=256)
ap.add_argument("--data_path", default="/data/datasets/ILSVRC2012"); a = ap.parse_args()
first, last = (int(x) for x in a.blocks.split("-")); blocks = list(range(first, last + 1))
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train")); transform = build_transform(False, args)
state = extract_profile.load_state_dict(a.checkpoint)
prefix_tensors = {k: v for k, v in state.items() if k.startswith("blocks.") and int(k.split(".")[1]) in blocks}
images_of, model_of = {}, {}
def measure(model_seed, image_seed):
    if image_seed not in images_of:
        images_of[image_seed] = utils.calibration_images(folder.samples, folder.loader, transform, a.images, image_seed)
    torch.manual_seed(model_seed); model = utils.build_model(args); model.load_state_dict(prefix_tensors, strict=False); model.to(device).eval()
    out, imgs = {}, images_of[image_seed]
    for start in range(0, len(imgs), 64):      # chunks of 64 images; every quantity is a mean over images, so chunks average exactly
        part = utils.joint_statistics_per_block(model, imgs[start:start + 64].to(device), blocks)
        for b, values in part.items():
            for k, v in values.items():
                out.setdefault(b, {}).setdefault(k, []).append(v)
    return {b: {k: sum(v) / len(v) for k, v in values.items()} for b, values in out.items()}
runs = {"A": [(s, s) for s in range(5)], "B": [(0, 1), (0, 2)], "C": [(1, 0), (2, 0)]}
results = {key: measure(*key) for keys in runs.values() for key in keys}
QUANTITIES = [("entropy", "attention entropy (nats)", 3), ("active_units", "active fc1 units (fraction > 0)", 5), ("pre_activation_mean", "mean fc1 pre-activation", 3),
              ("attention_write", "attention write ratio", 4), ("mlp_write", "MLP write ratio", 4), ("token_cosine", "token cosine of the block's output", 3)]
ref = results[(0, 0)]
for key, label, digits in QUANTITIES:
    print(f"\n{label}: block | specification's context (seed 0) | A: mean +- sd over 5 seeds (sd / mean) | B: image draws 1, 2 | C: model seeds 1, 2")
    for b in blocks:
        values = torch.tensor([results[k][b][key] for k in runs["A"]], dtype=torch.float64)
        fmt = lambda x: f"{x:.{digits}f}"
        print(f"   b{b} | {fmt(ref[b][key])} | {fmt(float(values.mean()))} +- {fmt(float(values.std()))} ({float(values.std() / values.mean().abs()):.1%}) | "
              + " ".join(fmt(results[k][b][key]) for k in runs["B"]) + " | " + " ".join(fmt(results[k][b][key]) for k in runs["C"]))
