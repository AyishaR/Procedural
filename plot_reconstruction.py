"""Plot how faithfully a checkpoint-free specification reconstructs the procedural prefix at initialisation.

Companion of plot_profile.py: that figure shows the weight statistics the recipe reads off the checkpoint; this one
shows what the resulting network DOES on images, block by block, next to the checkpoint prefix it is derived from.

    checkpoint prefix       fresh timm ViT-B with the checkpoint's tensors in the specification's blocks (as main.py builds ftb3i for 0..8, ftb4i for 0..7)
    specification           the same fresh model after utils.apply_analytic_profile(SPEC) and, when the specification
                            carries joint statistics, utils.calibrate_joint_statistics -- exactly the initialisation main.py produces
    specification, second moments only   the specification without its joint statistics ("qk_entropy", "fc1_gate", "common_write"), drawn when it has any
    random                  the fresh timm model

Two figures. <figure>_scales.png: the six effective scales (exact folding) read back from each initialised model, the
quantity the specification's numbers control. <figure>.png: twelve panels, all measured on TRAINING images under the evaluation transform (utils.calibration_images, a draw that
the calibration did not use unless --image_seed equals --seed):
    attention entropy | mass on the most-attended key | share of the query common to all tokens | token-specific share
    of the attention write | attention write ratio | mean fc1 pre-activation | fraction of active fc1 units (GELU gate) |
    GELU output rms | MLP write ratio | token cosine of the block input.
Entropy and the mean pre-activation are the two calibrated targets; everything else is a consequence.

usage: .venv/bin/python plot_reconstruction.py CHECKPOINT SPEC.json [--figure plots/out/reconstruction_<spec>.png]
                                               [--blocks 0-7] [--images 64] [--seed 0] [--image_seed 1] [--no_tex]
for example:
       .venv/bin/python plot_reconstruction.py results/pr_vitb_n/pr_6066174_final.pth vitbase_runs/profile_ftbanaper.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets as torchvision_datasets
from tueplots import bundles

import extract_profile
import main as training_main
import utils
from datasets import build_transform

ROOT = os.path.dirname(os.path.abspath(__file__))
PALETTE = sns.color_palette("tab10")
STYLES = {"checkpoint prefix": dict(color=PALETTE[0], marker="o", linestyle="-", linewidth=1.6, zorder=2),
          # dashed with hollow markers, drawn on top, so the checkpoint's line stays visible where the two coincide
          "specification": dict(color=PALETTE[1], marker="s", linestyle=(0, (4, 2.5)), linewidth=1.1, markerfacecolor="none",
                                markeredgewidth=0.8, zorder=3),
          "specification, second moments only": dict(color=PALETTE[1], marker="s", linestyle=":", linewidth=0.9, alpha=0.6,
                                                     markerfacecolor="none", markeredgewidth=0.6, zorder=1),
          "random": dict(color="#8a8a8a", marker="^", linestyle="-.", linewidth=0.9, zorder=1)}
PANELS = [("entropy", "attn entropy (nats)", False), ("sink_share", "top-key mass", False),
          ("common_query", "common query", False), ("attention_specific", "token-spec. attn", True),
          ("attention_write", "attn write ratio", True), ("stream_rms", "stream rms", True),
          ("pre_activation_mean", "fc1 pre-act. mean", False), ("gate", "active fc1 units", False),
          ("gelu_rms", "GELU out rms", True), ("mlp_specific", "token-spec. MLP", True),
          ("mlp_write", "MLP write ratio", True), ("token_cosine", "token cosine", False)]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint"); parser.add_argument("specification")
    parser.add_argument("--figure", default=None)
    parser.add_argument("--blocks", default=None, help="blocks the specification covers (and the checkpoint prefix uses); default: read from the specification")
    parser.add_argument("--images", type=int, default=64, help="images for the calibration and for the figure (main.py uses the specification's count)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the model and of the sink calibration images (as in training)")
    parser.add_argument("--image_seed", type=int, default=1, help="seed of the images the figure is measured on")
    parser.add_argument("--data_path", default="/data/datasets/ILSVRC2012")
    parser.add_argument("--no_tex", action="store_true")
    arguments = parser.parse_args()
    if arguments.figure is None:
        stem = os.path.splitext(os.path.basename(arguments.specification))[0].replace("profile_", "")
        arguments.figure = os.path.join(ROOT, "plots", "out", f"reconstruction_{stem}.png")
    return arguments


@torch.no_grad()
def forward_statistics(model, images):
    """Per block: the quantities of PANELS."""
    model.eval()
    stream, result = utils.block_input_stream(model, images), []
    for block in model.blocks:
        probabilities, logits, query = utils._attention_rows(block, stream, block.attn.qkv.weight)
        query_flat = query.transpose(1, 2).reshape(query.shape[0], query.shape[2], -1)
        query_mean = query_flat.mean(1, keepdim=True)
        attention_out = block.attn(block.norm1(stream))
        after_attention = stream + attention_out
        pre_activation = block.mlp.fc1(block.norm2(after_attention))
        mlp_out = block.mlp(block.norm2(after_attention))
        patches, mlp_patches = attention_out[:, 1:], mlp_out[:, 1:]
        tokens = torch.nn.functional.normalize(stream[:, 1:], dim=-1)
        result.append({
            "entropy": float(-(probabilities * (probabilities + 1e-12).log()).sum(-1).mean()),
            "sink_share": float(probabilities.mean(2).max(-1).values.mean()),
            "common_query": float((query_mean.pow(2).sum(-1) / query_flat.pow(2).sum(-1).mean(1, keepdim=True)).mean()),
            "attention_specific": float((patches - patches.mean(1, keepdim=True)).pow(2).sum() / patches.pow(2).sum()),
            "attention_write": float((attention_out.norm(dim=-1) / stream.norm(dim=-1)).mean()),
            "stream_rms": float(stream.pow(2).mean().sqrt()),
            "mlp_specific": float((mlp_patches - mlp_patches.mean(1, keepdim=True)).pow(2).sum() / mlp_patches.pow(2).sum()),
            "mlp_write": float((mlp_out.norm(dim=-1) / after_attention.norm(dim=-1)).mean()),
            "gate": float((pre_activation > 0).float().mean()),
            "pre_activation_mean": float(pre_activation.mean()),
            "gelu_rms": float(block.mlp.act(pre_activation).pow(2).mean().sqrt()),
            "token_cosine": float((tokens @ tokens.transpose(1, 2)).mean()),
        })
        stream = after_attention + mlp_out
    return result


def effective_scales_of(model, init_standard_deviation=0.02):
    """Per block, the six effective scales of a MODEL (exact folding, rms(W diag(gain)) / init std) -- the quantity the
    specification's scale numbers control, read back from the initialised weights with extract_profile's own code."""
    state_dict = {name: tensor.detach() for name, tensor in model.state_dict().items()}
    return [extract_profile.effective_scales(state_dict, block, init_standard_deviation, gain_fold="exact")
            for block in range(len(model.blocks))]


def plot_scales(arguments, models, blocks, figure_path):
    """Second figure, laid out like plot_profile.py: effective scale per linear weight and block."""
    scales = {name: effective_scales_of(model) for name, model in models.items()}
    depth = len(next(iter(scales.values())))
    plt.rcParams.update(bundles.iclr2024(usetex=not arguments.no_tex, family="serif", nrows=2, ncols=3))
    width = plt.rcParams["figure.figsize"][0]
    figure, axes = plt.subplots(2, 3, sharex=True, figsize=(width, 0.62 * width))
    for axis, weight_name in zip(axes.flat, extract_profile.LINEAR_WEIGHTS):
        for name in ("checkpoint prefix", "specification", "specification, second moments only", "random"):
            if name in scales:
                axis.plot(range(depth), [scales[name][b][weight_name] for b in range(depth)], markersize=2.4, label=name, **STYLES[name])
        axis.axvspan(blocks[-1] + 0.5, depth - 0.5, color="#b0b0b0", alpha=0.08, linewidth=0)
        axis.set_title({"q": "query", "k": "key", "v": "value"}.get(weight_name, weight_name))
        axis.grid(True, color="#e6e6e6", linewidth=0.5); axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xticks(range(depth)); axis.set_xticklabels([str(b) if b % 2 == 0 else "" for b in range(depth)])
    for axis in axes[-1]:
        axis.set_xlabel("block")
    for axis in axes[:, 0]:
        axis.set_ylabel("effective scale")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    figure.savefig(figure_path, dpi=300); figure.savefig(os.path.splitext(figure_path)[0] + ".pdf")
    print(f"wrote {figure_path} and .pdf")
    print(f"{'effective scale (exact folding)':36s} {'model':24s} " + " ".join(f"b{b:<6d}" for b in range(depth)))
    for weight_name in extract_profile.LINEAR_WEIGHTS:
        for name in scales:
            print(f"{weight_name:36s} {name:24s} " + " ".join(f"{scales[name][b][weight_name]:7.3f}" for b in range(depth)))


def main():
    arguments = parse_arguments()
    specification = json.load(open(arguments.specification))
    if arguments.blocks is None:
        blocks = sorted(int(block) for block in specification["q"]["per_block"])
    else:
        first, last = (int(x) for x in arguments.blocks.split("-"))
        blocks = list(range(first, last + 1))

    model_arguments = training_main.get_args_parser().parse_args(
        ["--model", "vit_base", "--data_set", "IMNET", "--data_path", arguments.data_path, "--input_size", "224", "--nb_classes", "1000"])
    model_arguments.nb_classes = 1000
    folder = torchvision_datasets.ImageFolder(os.path.join(arguments.data_path, "train"))
    transform = build_transform(False, model_arguments)
    take = lambda seed: utils.calibration_images(folder.samples, folder.loader, transform, arguments.images, seed)
    calibration_images, figure_images = take(arguments.seed), take(arguments.image_seed)

    def fresh():
        torch.manual_seed(arguments.seed)
        return utils.build_model(model_arguments)

    models = {}
    prefix = fresh()
    state_dict = extract_profile.load_state_dict(arguments.checkpoint)
    prefix.load_state_dict({name: tensor for name, tensor in state_dict.items()
                            if name.startswith("blocks.") and int(name.split(".")[1]) in blocks}, strict=False)
    models["checkpoint prefix"] = prefix
    recipe = fresh()
    utils.apply_analytic_profile(recipe, specification, blocks, seed=arguments.seed)
    joint = {key: specification[key] for key in ("qk_entropy", "fc1_gate", "common_write", "write_ratio") if key in specification}
    if joint:
        without_joint = fresh()
        utils.apply_analytic_profile(without_joint, specification, blocks, seed=arguments.seed)
        models["specification, second moments only"] = without_joint
        utils.calibrate_joint_statistics(recipe, {**joint, **{key: specification[key] for key in ("gain_fold",) if key in specification}}, calibration_images, seed=arguments.seed)
    models["specification"] = recipe
    models["random"] = fresh()

    plot_scales(arguments, models, blocks, os.path.splitext(arguments.figure)[0] + "_scales.png")
    statistics = {name: forward_statistics(model, figure_images) for name, model in models.items()}
    depth = len(prefix.blocks)
    print(f"{'quantity':36s} {'model':24s} " + " ".join(f"b{b:<6d}" for b in range(depth)))
    for key, label, _ in PANELS:
        for name in statistics:
            print(f"{key:36s} {name:24s} " + " ".join(f"{statistics[name][b][key]:7.3f}" for b in range(depth)))

    plt.rcParams.update(bundles.iclr2024(usetex=not arguments.no_tex, family="serif", nrows=2, ncols=6))
    width = plt.rcParams["figure.figsize"][0]
    figure, axes = plt.subplots(2, 6, sharex=True, figsize=(width, 0.36 * width))
    order = ["checkpoint prefix", "specification", "specification, second moments only", "random"]
    for axis, (key, label, log_scale) in zip(axes.flat, PANELS):
        for name in order:
            if name in statistics:
                axis.plot(range(depth), [statistics[name][b][key] for b in range(depth)], markersize=2.4, label=name, **STYLES[name])
        axis.axvspan(blocks[-1] + 0.5, depth - 0.5, color="#b0b0b0", alpha=0.08, linewidth=0)
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(label, fontsize=plt.rcParams["font.size"] - 2)
        axis.grid(True, color="#e6e6e6", linewidth=0.5); axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xticks(range(depth)); axis.set_xticklabels([str(b) if b % 2 == 0 else "" for b in range(depth)])
    for axis in axes[-1]:
        axis.set_xlabel("block")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    os.makedirs(os.path.dirname(arguments.figure), exist_ok=True)
    figure.savefig(arguments.figure, dpi=300)
    figure.savefig(os.path.splitext(arguments.figure)[0] + ".pdf")
    print(f"wrote {arguments.figure} and .pdf")


if __name__ == "__main__":
    main()
