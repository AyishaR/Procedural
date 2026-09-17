from __future__ import annotations

import csv
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tueplots import bundles

plt.rcParams.update(bundles.iclr2024())
plt.rcParams.update(bundles.iclr2024(usetex=True, rel_width=1.0, nrows=1, ncols=1, family='serif'))
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


def load_state_dict_only(path: str | Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at {path}")

    cleaned = OrderedDict()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v.detach().cpu().float()

    return cleaned


def _extract_layer_idx(name: str):
    m = re.match(r"layer_(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def _layer_sort_key(name: str):
    idx = _extract_layer_idx(name)
    if idx is None:
        return (10**9, name)
    return (idx, name)


def _display_label(name: str) -> str:
    if name == "misc":
        return "norm"
    idx = _extract_layer_idx(name)
    if idx is not None:
        return f"Layer {idx}"
    return name.replace("_", " ")


def infer_epoch_from_name(path: Path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def list_checkpoints_sorted(src_dir: str | Path):
    src_dir = Path(src_dir)
    files = list(src_dir.glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No .pth files found in {src_dir}")

    with_epoch = []
    without_epoch = []

    for p in files:
        ep = infer_epoch_from_name(p)
        if ep is None:
            without_epoch.append(p)
        else:
            with_epoch.append((ep, p))

    if with_epoch:
        with_epoch.sort(key=lambda x: x[0])
        ordered = [p for _, p in with_epoch]
        if without_epoch:
            ordered.extend(sorted(without_epoch))
    else:
        ordered = sorted(files)

    return ordered


def squared_distance(a: torch.Tensor, b: torch.Tensor, norm_mode: str = "l2"):
    diff = (a - b).float()

    if norm_mode == "l2":
        return float(torch.sum(diff * diff).item())
    elif norm_mode == "fro":
        if diff.ndim >= 2:
            return float(torch.norm(diff, p="fro").item() ** 2)
        return float(torch.sum(diff * diff).item())
    else:
        raise ValueError(f"Unsupported norm_mode={norm_mode}")


def parse_component_name(param_key: str):
    k = param_key

    if "patch_embed" in k:
        return "patch_embed"
    if "pos_embed" in k:
        return "pos_embed"
    if "cls_token" in k:
        return "cls_token"

    m = re.search(r"(blocks|layers|encoder\.layer|h|stage)\.(\d+)", k)
    if m:
        layer_idx = int(m.group(2))
        if any(x in k for x in ["attn", "attention", "self_attn"]):
            return f"layer_{layer_idx:02d}.attn"
        if any(x in k for x in ["mlp", "ffn", "fc1", "fc2"]):
            return f"layer_{layer_idx:02d}.mlp"
        if "norm1" in k or "ln_1" in k:
            return f"layer_{layer_idx:02d}.norm1"
        if "norm2" in k or "ln_2" in k:
            return f"layer_{layer_idx:02d}.norm2"
        if "norm" in k or "ln" in k:
            return f"layer_{layer_idx:02d}.norm"
        return f"layer_{layer_idx:02d}.other"

    m2 = re.search(r"layer\.?(\d+)", k)
    if m2:
        layer_idx = int(m2.group(1))
        if any(x in k for x in ["attn", "attention", "self_attn"]):
            return f"layer_{layer_idx:02d}.attn"
        if any(x in k for x in ["mlp", "ffn", "fc1", "fc2"]):
            return f"layer_{layer_idx:02d}.mlp"
        if "norm1" in k:
            return f"layer_{layer_idx:02d}.norm1"
        if "norm2" in k:
            return f"layer_{layer_idx:02d}.norm2"
        if "norm" in k:
            return f"layer_{layer_idx:02d}.norm"
        return f"layer_{layer_idx:02d}.other"

    if any(x in k for x in ["head", "classifier"]):
        return "head"

    return "misc"


def coarse_from_fine(fine_name: str):
    if fine_name.startswith("layer_"):
        return fine_name.split(".", 1)[0]
    return fine_name


def _ordered_layer_palette():
    return [
        "#FFD000",  # 01 yellow
        "#FF8C00",  # 02 strong orange
        "#E31A1C",  # 03 red
        "#B22222",  # 04 dark red
        "#FF5A8A",  # 05 hot pink
        "#C51B8A",  # 06 pink-magenta
        "#8E44AD",  # 07 purple
        "#5E2B97",  # 08 deep violet
        "#1F77FF",  # 09 blue
        "#00A676",  # 10 green-teal
        "#8A9A00",  # 11 olive
        "#7F4F24",  # 12 brown

        "#FFE680",  # 13 pale yellow
        "#FFB366",  # 14 light orange
        "#F4A3B4",  # 15 soft pink
        "#B39DDB",  # 16 lavender
        "#7FDBFF",  # 17 light blue
        "#6B8E23",  # 18 olive-drab
        "#808080",  # 19 gray
        "#000000",  # 20 black
    ]


def compute_coarse_normalized_curves_for_seed(checkpoint_dir: str, norm_mode: str = "l2"):
    """Coarse-grained normalized distance-to-final curve for every coarse
    component (layers and non-layer components alike) for a single seed."""
    checkpoint_paths = list_checkpoints_sorted(checkpoint_dir)
    n = len(checkpoint_paths)
    if n < 2:
        raise ValueError(f"Need at least 2 checkpoints in {checkpoint_dir}")

    states = [load_state_dict_only(p) for p in checkpoint_paths]
    final_state = states[-1]

    common_keys = set(final_state.keys())
    for sd in states:
        common_keys &= set(sd.keys())
    common_keys = sorted(common_keys)

    fine_groups = defaultdict(list)
    for k in common_keys:
        fine_groups[parse_component_name(k)].append(k)

    coarse_groups = defaultdict(list)
    for fine_name, keys in fine_groups.items():
        coarse_groups[coarse_from_fine(fine_name)].extend(keys)

    def group_dist2(state_dict, key_list):
        total = 0.0
        for k in key_list:
            total += squared_distance(state_dict[k], final_state[k], norm_mode=norm_mode)
        return total

    normalized = {}
    for comp, keys in coarse_groups.items():
        d = [group_dist2(sd, keys) for sd in states]
        d0 = d[0]
        if d0 <= 1e-30:
            normalized[comp] = [0.0 for _ in range(n)]
        else:
            normalized[comp] = [x / d0 for x in d]

    return normalized, n


def compute_and_save_mean_curve_csv(
    seed_dirs: list[str],
    output_dir: str | Path,
    norm_mode: str = "l2",
) -> Path:
    """Compute the coarse-grained normalized distance curve (all components)
    for each seed, average across seeds, and write the result to a CSV so
    the plot can later be reproduced without re-reading any checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "coarse_normalized_distance_all_mean_across_seeds.csv"
    if csv_path.exists():
        return csv_path

    per_seed_curves = []
    ns = []
    for seed_dir in seed_dirs:
        normalized, n = compute_coarse_normalized_curves_for_seed(seed_dir, norm_mode=norm_mode)
        per_seed_curves.append(normalized)
        ns.append(n)

    min_n = min(ns)

    common_components = set(per_seed_curves[0].keys())
    for curves in per_seed_curves[1:]:
        common_components &= set(curves.keys())
    common_components = sorted(common_components, key=_layer_sort_key)

    mean_curves = {}
    for comp in common_components:
        stacked = np.array([curves[comp][:min_n] for curves in per_seed_curves])
        mean_curves[comp] = stacked.mean(axis=0).tolist()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + common_components)
        for t in range(min_n):
            writer.writerow([t] + [mean_curves[comp][t] for comp in common_components])

    return csv_path


def plot_focus_from_csv(csv_path: str | Path, output_path: str | Path, title_suffix: str = ""):
    """Read the averaged curve CSV back in and produce the two side-by-side
    ICLR-styled panels: the full coarse-grained normalized distance curve
    (fixed 0-1.6 scale, as in the original figure) and a zoomed-in view of
    epochs 100-200 (fixed 0-1 scale) for more visibility into the curves."""
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].to_numpy()
    components = [c for c in df.columns if c != "epoch"]
    ordered = sorted(components, key=_layer_sort_key)

    palette = _ordered_layer_palette()
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(ordered)}

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(8, 5))

    for ax in (ax_full, ax_zoom):
        for name in ordered:
            vals = df[name].to_numpy()
            linestyle = "-" if name.startswith("layer_") else "--"
            ax.plot(
                epochs,
                vals,
                linewidth=2.0,
                linestyle=linestyle,
                color=color_map[name],
                label=_display_label(name),
            )
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 299)
        ax.set_ylabel(r"$\|\theta_t - \theta^*\|^2 / \|\theta_0 - \theta^*\|^2$")
        ax.grid(True, alpha=0.3)

    title_suffix = f" ({title_suffix})" if title_suffix else ""
    # ax_full.set_title(f"Coarse-grained normalized distance{title_suffix}\nAll components, mean across 3 seeds")
    ax_full.set_ylim(0, 1.6)

    ax_zoom.set_xlim(100, 199)
    ax_zoom.set_ylim(0, 1)
    # ax_zoom.set_title("Zoom: epochs 100-199")

    handles, labels = ax_full.get_legend_handles_labels()
    ncol = min(len(ordered), 6) if len(ordered) <= 20 else min(len(ordered), 8)

    fig.tight_layout()
    legend = fig.legend(handles, labels, fontsize=10, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 0.0))

    # Measure the legend's actual rendered height and push the axes up by
    # that much so it can never overlap the x-axis labels, regardless of
    # how many rows it ends up wrapping to.
    fig.canvas.draw()
    legend_height_fig = legend.get_window_extent().height / fig.bbox.height
    fig.subplots_adjust(bottom=legend_height_fig + 0.12)

    fig.savefig(output_path, dpi=1500, bbox_inches="tight")
    plt.close(fig)


def make_layer_convergence_focus_plot(
    seed_dirs: list[str],
    output_dir: str | Path,
    title_suffix: str = "",
    norm_mode: str = "l2",
):
    csv_path = compute_and_save_mean_curve_csv(seed_dirs, output_dir, norm_mode=norm_mode)
    plot_path = Path(output_dir) / "coarse_normalized_distance_focus.png"
    plot_focus_from_csv(csv_path, plot_path, title_suffix=title_suffix)
    return {"csv_path": str(csv_path), "plot_path": str(plot_path)}


if __name__ == "__main__":
    # Example usage:
    # python layer_convergence_focus.py
    #
    # Then edit these paths as needed.
    dir_list = [
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912449",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_6848384",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_6848392",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_6089446",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_6221557",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_6098398",
    ]
    output_main_dir = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/metrics_v2/layer_convergence_focus"
    for dir_item in dir_list:
        seed_dirs = []
        for i in range(3):
            if not os.path.exists(dir_item + f"/s{i}/checkpoint-299.pth"):
                print(f"Warning: {dir_item}/s{i} does not exist. Skipping.")
                continue
            slurm_id = dir_item.split("_")[-1]
            seed_dirs.append(dir_item + f"/s{i}")

        if not seed_dirs:
            continue

        output_dir = output_main_dir + f"/{slurm_id}"
        os.makedirs(output_dir, exist_ok=True)

        results = make_layer_convergence_focus_plot(
            seed_dirs=seed_dirs,
            output_dir=output_dir,
            title_suffix=f"slurm {slurm_id}",
            norm_mode="l2",  # change to "fro" to use the alternate option
        )
        print(results)
