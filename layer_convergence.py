from __future__ import annotations

import argparse
import csv
import os
import json
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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
        # non-layers go after layers; second element keeps deterministic order
        return (10**9, name)
    return (idx, name)

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


def _extract_layer_idx(name: str):
    m = re.match(r"layer_(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def _bucket_name_for_component(name: str):
    idx = _extract_layer_idx(name)
    if idx is None:
        return "non_layers"
    if 0 <= idx <= 3:
        return "layers_0_3"
    if 4 <= idx <= 7:
        return "layers_4_7"
    if 8 <= idx <= 11:
        return "layers_8_11"
    return "layers_other"


def _split_curve_dict_by_bucket(curve_dict: dict[str, list[float]]):
    buckets = defaultdict(dict)
    for name, vals in curve_dict.items():
        buckets[_bucket_name_for_component(name)][name] = vals
    return buckets


def _split_rank_dict_by_bucket(rank_dict: dict[str, float]):
    buckets = defaultdict(dict)
    for name, score in rank_dict.items():
        buckets[_bucket_name_for_component(name)][name] = score
    return buckets


def _parent_group(name: str):
    if name.startswith("layer_"):
        return name.split(".", 1)[0]
    return name


def _line_style(name: str):
    if name.endswith(".attn"):
        return "-"
    elif name.endswith(".mlp"):
        return "--"
    elif name.endswith(".norm1"):
        return "-."
    elif name.endswith(".norm2"):
        return (0, (3, 1, 1, 1))
    elif name.endswith(".norm"):
        return (0, (5, 1))
    elif name.endswith(".other"):
        return (0, (1, 1))
    else:
        return "--"


def plot_curves(curve_dict, title, ylabel, out_path, n, top_k=None, rank_by=None, group_by_parent_color=False):
    if not curve_dict:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    items = sorted(curve_dict.items(), key=lambda kv: _layer_sort_key(kv[0]))
    if top_k is not None and rank_by is not None:
        ordered_keys = [k for k, _ in sorted(rank_by.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        items = [(k, curve_dict[k]) for k in ordered_keys if k in curve_dict]

    xs = np.arange(n)

    palette = _ordered_layer_palette()

    if group_by_parent_color:
        parent_groups = []
        for name, _ in items:
            pg = _parent_group(name)
            if pg not in parent_groups:
                parent_groups.append(pg)
        color_map = {pg: palette[i % len(palette)] for i, pg in enumerate(parent_groups)}
    else:
        color_map = {name: palette[i % len(palette)] for i, (name, _) in enumerate(items)}

    for name, vals in items:
        color = color_map[_parent_group(name)] if group_by_parent_color else color_map[name]
        linestyle = _line_style(name) if group_by_parent_color else ("-" if name.startswith("layer_") else "--")

        ax.plot(
            xs,
            vals,
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            label=name,
        )

    ax.set_xlabel("Epoch / checkpoint index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "cumulative" not in title.lower():
        ax.set_ylim(0, 1.6)
    ax.grid(True, alpha=0.3)

    if len(items) <= 20:
        ax.legend(fontsize=8, ncol=2)
    elif len(items) <= 40:
        ax.legend(fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_partitioned_plots(curve_dict, rank_dict, prefix, title_prefix, ylabel, output_dir, n, group_by_parent_color):
    curve_buckets = _split_curve_dict_by_bucket(curve_dict)
    rank_buckets = _split_rank_dict_by_bucket(rank_dict)

    bucket_order = ["layers_0_3", "layers_4_7", "layers_8_11", "non_layers"]
    bucket_titles = {
        "layers_0_3": "Layers 0-3",
        "layers_4_7": "Layers 4-7",
        "layers_8_11": "Layers 8-11",
        "non_layers": "Non-layer components",
        "layers_other": "Other layers",
    }

    for bucket in bucket_order:
        if bucket not in curve_buckets or not curve_buckets[bucket]:
            continue

        plot_curves(
            curve_dict=curve_buckets[bucket],
            title=f"{title_prefix} - {bucket_titles[bucket]}",
            ylabel=ylabel,
            out_path=Path(output_dir) / f"{prefix}_{bucket}.png",
            n=n,
            top_k=None,
            rank_by=rank_buckets.get(bucket, None),
            group_by_parent_color=group_by_parent_color,
        )

    for bucket in curve_buckets:
        if bucket in bucket_order:
            continue
        plot_curves(
            curve_dict=curve_buckets[bucket],
            title=f"{title_prefix} - {bucket_titles.get(bucket, bucket)}",
            ylabel=ylabel,
            out_path=Path(output_dir) / f"{prefix}_{bucket}.png",
            n=n,
            top_k=None,
            rank_by=rank_buckets.get(bucket, None),
            group_by_parent_color=group_by_parent_color,
        )


def write_summary_csv(path: Path, ranked_dict: dict[str, float]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "convergence_rate_value"])
        for k, v in ranked_dict.items():
            writer.writerow([k, v])


def write_curve_csv(path: Path, curve_dict: dict[str, list[float]], n: int):
    keys = list(curve_dict.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for t in range(n):
            writer.writerow([t] + [curve_dict[k][t] for k in keys])


def compute_layer_convergence(
    checkpoint_dir: str,
    output_dir: str,
    norm_mode: str = "l2",
):
    checkpoint_paths = list_checkpoints_sorted(checkpoint_dir)
    n = len(checkpoint_paths)
    if n < 2:
        raise ValueError("Need at least 2 checkpoints to compute convergence.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    states = [load_state_dict_only(p) for p in checkpoint_paths]
    final_state = states[-1]

    common_keys = set(final_state.keys())
    for sd in states:
        common_keys &= set(sd.keys())
    common_keys = sorted(common_keys)
    if not common_keys:
        raise ValueError("No common tensor keys across checkpoints.")

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

    def build_metrics(groups: dict[str, list[str]]):
        distances = {}
        normalized = {}
        per_epoch_rate = {}
        cumulative_rate = {}
        final_rate_value = {}

        for comp, keys in groups.items():
            d = [group_dist2(sd, keys) for sd in states]
            d0 = d[0]

            if d0 <= 1e-30:
                norm_curve = [0.0 for _ in range(n)]
                epoch_curve = [0.0 for _ in range(n)]
                cum_curve = [0.0 for _ in range(n)]
                score = 0.0
            else:
                norm_curve = [x / d0 for x in d]

                epoch_curve = [0.0]
                for t in range(1, n):
                    c = (d[t - 1] - d[t]) / d0
                    epoch_curve.append(c)

                cum_curve = [0.0]
                for t in range(1, n):
                    c = (d[0] - d[t]) / (t * d0)
                    cum_curve.append(c)

                score = cum_curve[-1]

            distances[comp] = d
            normalized[comp] = norm_curve
            per_epoch_rate[comp] = epoch_curve
            cumulative_rate[comp] = cum_curve
            final_rate_value[comp] = score

        return {
            "raw_distance_sq_to_final": distances,
            "normalized_distance_curve": normalized,
            "per_epoch_convergence_rate": per_epoch_rate,
            "cumulative_convergence_rate": cumulative_rate,
            "final_convergence_rate_value": final_rate_value,
        }

    fine_metrics = build_metrics(fine_groups)
    coarse_metrics = build_metrics(coarse_groups)

    ranked_fine = dict(sorted(
        fine_metrics["final_convergence_rate_value"].items(),
        key=lambda x: x[1],
        reverse=True
    ))
    ranked_coarse = dict(sorted(
        coarse_metrics["final_convergence_rate_value"].items(),
        key=lambda x: x[1],
        reverse=True
    ))

    fine_metrics["final_convergence_rate_value"] = ranked_fine
    coarse_metrics["final_convergence_rate_value"] = ranked_coarse

    meta = {
        "formula": "C_l^(t1,t2) = (||theta_l^(t1)-theta_l*||^2 - ||theta_l^(t2)-theta_l*||^2) / ((t2-t1) * ||theta_l^(t0)-theta_l*||^2)",
        "num_checkpoints": n,
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
        "reference_checkpoint": str(checkpoint_paths[-1]),
        "norm_mode": norm_mode,
        "fine_group_names": sorted(fine_groups.keys()),
        "coarse_group_names": sorted(coarse_groups.keys()),
    }

    with open(output_dir / "fine_grained_convergence.json", "w") as f:
        json.dump({"meta": meta, "metrics": fine_metrics}, f, indent=2)

    with open(output_dir / "coarse_grained_convergence.json", "w") as f:
        json.dump({"meta": meta, "metrics": coarse_metrics}, f, indent=2)

    write_summary_csv(output_dir / "fine_grained_convergence_summary.csv", ranked_fine)
    write_summary_csv(output_dir / "coarse_grained_convergence_summary.csv", ranked_coarse)

    write_curve_csv(output_dir / "fine_normalized_distance_curves.csv", fine_metrics["normalized_distance_curve"], n)
    write_curve_csv(output_dir / "coarse_normalized_distance_curves.csv", coarse_metrics["normalized_distance_curve"], n)
    write_curve_csv(output_dir / "fine_per_epoch_rate_curves.csv", fine_metrics["per_epoch_convergence_rate"], n)
    write_curve_csv(output_dir / "coarse_per_epoch_rate_curves.csv", coarse_metrics["per_epoch_convergence_rate"], n)
    write_curve_csv(output_dir / "fine_cumulative_rate_curves.csv", fine_metrics["cumulative_convergence_rate"], n)
    write_curve_csv(output_dir / "coarse_cumulative_rate_curves.csv", coarse_metrics["cumulative_convergence_rate"], n)

    # Overall plots: all components in one figure
    plot_curves(
        curve_dict=fine_metrics["normalized_distance_curve"],
        title="Fine-grained normalized distance to final checkpoint - All components",
        ylabel="||theta_t - theta*||^2 / ||theta_0 - theta*||^2",
        out_path=output_dir / "fine_normalized_distance_all.png",
        n=n,
        top_k=None,
        rank_by=ranked_fine,
        group_by_parent_color=True,
    )

    plot_curves(
        curve_dict=coarse_metrics["normalized_distance_curve"],
        title="Coarse-grained normalized distance to final checkpoint - All components",
        ylabel="||theta_t - theta*||^2 / ||theta_0 - theta*||^2",
        out_path=output_dir / "coarse_normalized_distance_all.png",
        n=n,
        top_k=None,
        rank_by=ranked_coarse,
        group_by_parent_color=False,
    )

    plot_curves(
        curve_dict=fine_metrics["cumulative_convergence_rate"],
        title="Fine-grained cumulative convergence rate - All components",
        ylabel="C_l^(0,t)",
        out_path=output_dir / "fine_cumulative_rate_all.png",
        n=n,
        top_k=None,
        rank_by=ranked_fine,
        group_by_parent_color=True,
    )

    plot_curves(
        curve_dict=coarse_metrics["cumulative_convergence_rate"],
        title="Coarse-grained cumulative convergence rate - All components",
        ylabel="C_l^(0,t)",
        out_path=output_dir / "coarse_cumulative_rate_all.png",
        n=n,
        top_k=None,
        rank_by=ranked_coarse,
        group_by_parent_color=False,
    )
    
    save_partitioned_plots(
        curve_dict=fine_metrics["normalized_distance_curve"],
        rank_dict=ranked_fine,
        prefix="fine_normalized_distance",
        title_prefix="Fine-grained normalized distance to final checkpoint",
        ylabel="||theta_t - theta*||^2 / ||theta_0 - theta*||^2",
        output_dir=output_dir,
        n=n,
        group_by_parent_color=True,
    )

    save_partitioned_plots(
        curve_dict=coarse_metrics["normalized_distance_curve"],
        rank_dict=ranked_coarse,
        prefix="coarse_normalized_distance",
        title_prefix="Coarse-grained normalized distance to final checkpoint",
        ylabel="||theta_t - theta*||^2 / ||theta_0 - theta*||^2",
        output_dir=output_dir,
        n=n,
        group_by_parent_color=False,
    )

    save_partitioned_plots(
        curve_dict=fine_metrics["cumulative_convergence_rate"],
        rank_dict=ranked_fine,
        prefix="fine_cumulative_rate",
        title_prefix="Fine-grained cumulative convergence rate",
        ylabel="C_l^(0,t)",
        output_dir=output_dir,
        n=n,
        group_by_parent_color=True,
    )

    save_partitioned_plots(
        curve_dict=coarse_metrics["cumulative_convergence_rate"],
        rank_dict=ranked_coarse,
        prefix="coarse_cumulative_rate",
        title_prefix="Coarse-grained cumulative convergence rate",
        ylabel="C_l^(0,t)",
        output_dir=output_dir,
        n=n,
        group_by_parent_color=False,
    )

    return {
        "fine_grained_convergence_rate": ranked_fine,
        "coarse_grained_convergence_rate": ranked_coarse,
        "output_dir": str(output_dir),
    }

import pandas as pd

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

def _rank_series_low_is_good(value_dict: dict[str, float]):
    items = sorted(value_dict.items(), key=lambda x: x[1])
    rank_map = {}
    for i, (name, _) in enumerate(items, start=1):
        rank_map[name] = i
    return rank_map


def _get_available_epoch_indices(selected_epochs, n):
    return [e for e in selected_epochs if 0 <= e < n]


def _extract_coarse_layer_normalized_from_checkpoint_dir(checkpoint_dir: str, norm_mode: str = "l2"):
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

    coarse_normalized = {}
    for comp, keys in coarse_groups.items():
        if not comp.startswith("layer_"):
            continue
        d = [group_dist2(sd, keys) for sd in states]
        d0 = d[0]
        if d0 <= 1e-30:
            coarse_normalized[comp] = [0.0 for _ in range(n)]
        else:
            coarse_normalized[comp] = [x / d0 for x in d]

    return coarse_normalized, n

def plot_value_heatmap_fixed_scale(
    value_df,
    output_path,
    title,
    global_layers,
    selected_epochs,
    by_seed=True,
    annotate=True,
    cmap="turbo",
    vmin=0.0,
    vmax=1.6,
):
    """
    Plot a heatmap colored by actual normalized-distance values
    using a fixed color scale across all heatmaps.

    Parameters
    ----------
    value_df : pd.DataFrame
        Must contain columns:
        - seed
        - epoch
        - layer
        - normalized_distance
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    global_layers : list[str]
        Desired numeric layer order.
    selected_epochs : list[int]
        Epochs to include, e.g. [100, 150, 200, 250].
    by_seed : bool
        If True, columns are MultiIndex (seed, epoch).
        If False, values are averaged across seeds and columns are epochs only.
    annotate : bool
        If True, write the actual normalized distance in each heatmap cell.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float
        Fixed color scale bounds.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = value_df[value_df["epoch"].isin(selected_epochs)].copy()

    if by_seed:
        pivot_df = df.pivot(index="layer", columns=["seed", "epoch"], values="normalized_distance")
    else:
        mean_df = (
            df.groupby(["layer", "epoch"], as_index=False)["normalized_distance"]
            .mean()
        )
        pivot_df = mean_df.pivot(index="layer", columns="epoch", values="normalized_distance")

    pivot_df = pivot_df.reindex(global_layers)

    fig, ax = plt.subplots(figsize=(16, max(6, len(global_layers) * 0.45)))
    im = ax.imshow(
        pivot_df.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_yticklabels(pivot_df.index)

    ax.set_xticks(np.arange(pivot_df.shape[1]))
    if by_seed:
        ax.set_xticklabels(
            [f"{seed}\n{epoch}" for seed, epoch in pivot_df.columns],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels([str(epoch) for epoch in pivot_df.columns], rotation=0)

    if annotate:
        threshold = (vmin + vmax) / 2.0
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                val = pivot_df.iat[i, j]
                if pd.notna(val):
                    text_color = "white" if val > threshold else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Normalized distance to final checkpoint ({vmin:.1f} to {vmax:.1f})")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

def plot_epoch_relative_gap_heatmap(
    value_df,
    output_path,
    title,
    global_layers,
    selected_epochs,
    by_seed=True,
    annotate=True,
    cmap="turbo",
    vmin=0.0,
    vmax=0.4,
):
    """
    Plot a heatmap where each cell color is:
        normalized_distance - minimum normalized_distance in that column

    So for each epoch-column, the best layer has color 0 and others are colored
    by how much larger their value is than the best layer at that epoch.

    Parameters
    ----------
    value_df : pd.DataFrame
        Must contain columns:
        - seed
        - epoch
        - layer
        - normalized_distance
    output_path : str or Path
        Path to save the figure.
    title : str
        Plot title.
    global_layers : list[str]
        Desired numeric layer order.
    selected_epochs : list[int]
        Epochs to include, e.g. [100, 150, 200, 250].
    by_seed : bool
        If True, columns are (seed, epoch).
        If False, values are averaged across seeds and columns are epochs only.
    annotate : bool
        If True, annotate each cell with the raw normalized distance value.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float
        Fixed color scale for the gap values.
        Example: if typical within-epoch gaps are up to 0.2, use vmax=0.2.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = value_df[value_df["epoch"].isin(selected_epochs)].copy()

    if by_seed:
        raw_pivot = df.pivot(index="layer", columns=["seed", "epoch"], values="normalized_distance")
    else:
        mean_df = (
            df.groupby(["layer", "epoch"], as_index=False)["normalized_distance"]
            .mean()
        )
        raw_pivot = mean_df.pivot(index="layer", columns="epoch", values="normalized_distance")

    raw_pivot = raw_pivot.reindex(global_layers)

    # Subtract the minimum value in each column
    column_min = raw_pivot.min(axis=0)
    gap_pivot = raw_pivot.subtract(column_min, axis=1)

    fig, ax = plt.subplots(figsize=(16, max(6, len(global_layers) * 0.45)))
    im = ax.imshow(
        gap_pivot.values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(gap_pivot.shape[0]))
    ax.set_yticklabels(gap_pivot.index)

    ax.set_xticks(np.arange(gap_pivot.shape[1]))
    if by_seed:
        ax.set_xticklabels(
            [f"{seed}\n{epoch}" for seed, epoch in gap_pivot.columns],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticklabels([str(epoch) for epoch in gap_pivot.columns], rotation=0)

    if annotate:
        threshold = vmin + 0.55 * (vmax - vmin)
        for i in range(raw_pivot.shape[0]):
            for j in range(raw_pivot.shape[1]):
                raw_val = raw_pivot.iat[i, j]
                gap_val = gap_pivot.iat[i, j]
                if pd.notna(raw_val):
                    # text_color = "white" if gap_val > threshold else "black"
                    text_color = "white" if gap_val < 0.05 or gap_val > 0.35 else "black"
                    ax.text(
                        j,
                        i,
                        f"{raw_val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value minus column minimum")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def visualize_layer_rankings_across_seeds(
    seed_dirs: list[str],
    output_dir: str,
    selected_epochs: list[int] = [100, 150, 200, 250],
    norm_mode: str = "l2",
):
    """
    For 3 seed directories:
      - compute coarse layer normalized distance curves
      - rank layers at selected epochs (lower normalized distance = better rank)
      - save CSVs
      - save annotated heatmap of ranks
      - save bump chart per seed
      - save bump chart of mean rank across seeds
    """
    # if len(seed_dirs) != 3:
    #     raise ValueError("Please provide exactly 3 seed directories.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_data = {}
    global_layers = set()
    min_n = None

    for i, seed_dir in enumerate(seed_dirs):
        coarse_norm, n = _extract_coarse_layer_normalized_from_checkpoint_dir(
            checkpoint_dir=seed_dir,
            norm_mode=norm_mode,
        )
        seed_name = f"seed_{i}"
        seed_data[seed_name] = {
            "checkpoint_dir": seed_dir,
            "coarse_normalized": coarse_norm,
            "n": n,
        }
        global_layers.update(coarse_norm.keys())
        min_n = n if min_n is None else min(min_n, n)

    global_layers = sorted(global_layers, key=_layer_sort_key)
    valid_epochs = _get_available_epoch_indices(selected_epochs, min_n)

    if not valid_epochs:
        raise ValueError(
            f"None of the selected epochs {selected_epochs} exist across all seeds. "
            f"Minimum checkpoint count across seeds is {min_n}."
        )

    rank_records = []
    value_records = []

    for seed_name, info in seed_data.items():
        coarse_norm = info["coarse_normalized"]

        for epoch in valid_epochs:
            epoch_values = {}
            for layer in global_layers:
                if layer in coarse_norm and epoch < len(coarse_norm[layer]):
                    epoch_values[layer] = coarse_norm[layer][epoch]

            rank_map = _rank_series_low_is_good(epoch_values)

            for layer in epoch_values:
                rank_records.append({
                    "seed": seed_name,
                    "epoch": epoch,
                    "layer": layer,
                    "normalized_distance": epoch_values[layer],
                    "rank": rank_map[layer],
                })
                value_records.append({
                    "seed": seed_name,
                    "epoch": epoch,
                    "layer": layer,
                    "normalized_distance": epoch_values[layer],
                })

    rank_df = pd.DataFrame(rank_records)
    value_df = pd.DataFrame(value_records)

    rank_df.to_csv(output_dir / "layer_rankings_selected_epochs.csv", index=False)
    value_df.to_csv(output_dir / "layer_normalized_distance_selected_epochs.csv", index=False)

    palette = _ordered_layer_palette()
    layer_color = {layer: palette[i % len(palette)] for i, layer in enumerate(global_layers)}

    # 1) Annotated heatmap of ranks across all seeds and selected epochs
    heatmap_df = rank_df.pivot(index="layer", columns=["seed", "epoch"], values="rank")
    heatmap_df = heatmap_df.reindex(global_layers)

    # rank heatmap, but annotate with normalized distance values
    rank_pivot = rank_df.pivot(index="layer", columns=["seed", "epoch"], values="rank").reindex(global_layers)
    value_pivot = value_df.pivot(index="layer", columns=["seed", "epoch"], values="normalized_distance").reindex(global_layers)

    fig, ax = plt.subplots(figsize=(30, max(6, len(global_layers) * 0.45)))
    im = ax.imshow(rank_pivot.values, aspect="auto", cmap="turbo")

    ax.set_xticks(np.arange(rank_pivot.shape[1]))
    ax.set_yticks(np.arange(rank_pivot.shape[0]))
    ax.set_xticklabels([f"{s}_e{e}" for s, e in rank_pivot.columns], rotation=90, ha="right")
    ax.set_yticklabels(rank_pivot.index)

    for i in range(rank_pivot.shape[0]):
        for j in range(rank_pivot.shape[1]):
            rank_val = rank_pivot.values[i, j]
            dist_val = value_pivot.values[i, j]
            if pd.notna(rank_val) and pd.notna(dist_val):
                txt = f"{dist_val:.2f}\n(r{int(rank_val)})"
                text_color = "white" if rank_val >= (len(global_layers) / 2) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=7)

    ax.set_title("Layer rank heatmap across seeds and selected epochs")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1 = lowest normalized distance)")

    fig.tight_layout()
    fig.savefig(output_dir / "layer_rank_heatmap_selected_epochs_annotated_detailed_focussed.png", dpi=220)
    plt.close(fig)


    # 2) Bump chart per seed
    for seed_name in sorted(rank_df["seed"].unique()):
        sdf = rank_df[rank_df["seed"] == seed_name]
        pivot = sdf.pivot(index="layer", columns="epoch", values="rank").reindex(global_layers)

        fig, ax = plt.subplots(figsize=(12, 7))
        xs = valid_epochs

        for layer in global_layers:
            ys = pivot.loc[layer, xs].values.astype(float)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=layer_color[layer],
                label=layer,
            )

        ax.set_title(f"Layer rank bump chart - {seed_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rank (1 = lowest normalized distance)")
        ax.invert_yaxis()
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

        if len(global_layers) <= 16:
            ax.legend(fontsize=8, ncol=2)

        fig.tight_layout()
        fig.savefig(output_dir / f"{seed_name}_layer_rank_bump.png", dpi=220)
        plt.close(fig)

    plot_value_heatmap_fixed_scale(
        value_df=value_df,
        output_path=output_dir / "layer_value_heatmap_fixed_scale_by_seed.png",
        title="Layer normalized distance heatmap (fixed scale 0 to 1.6)",
        global_layers=global_layers,
        selected_epochs=valid_epochs,
        by_seed=True,
        annotate=True,
        cmap="turbo",
        vmin=0.0,
        vmax=1.6,
    )

    plot_value_heatmap_fixed_scale(
        value_df=value_df,
        output_path=output_dir / "layer_value_heatmap_fixed_scale_mean_across_seeds.png",
        title="Layer normalized distance heatmap, mean across seeds (fixed scale 0 to 1.6)",
        global_layers=global_layers,
        selected_epochs=valid_epochs,
        by_seed=False,
        annotate=True,
        cmap="turbo",
        vmin=0.0,
        vmax=1.6,
    )

    plot_epoch_relative_gap_heatmap(
        value_df=value_df,
        output_path=output_dir / "layer_gap_heatmap_by_seed.png",
        title="Layer heatmap colored by value minus best layer in each epoch",
        global_layers=global_layers,
        selected_epochs=valid_epochs,
        by_seed=True,
        annotate=True,
        cmap="turbo",
        vmin=0.0,
        vmax=0.4,   # adjust if your gaps are usually smaller or larger
    )

    plot_epoch_relative_gap_heatmap(
        value_df=value_df,
        output_path=output_dir / "layer_gap_heatmap_mean_across_seeds.png",
        title="Layer heatmap colored by value minus best layer in each epoch (mean across seeds)",
        global_layers=global_layers,
        selected_epochs=valid_epochs,
        by_seed=False,
        annotate=True,
        cmap="turbo",
        vmin=0.0,
        vmax=0.4,
    )


    # 3) Mean-across-seeds rank bump chart
    mean_df = value_df.groupby(["epoch", "layer"], as_index=False)["normalized_distance"].mean()

    mean_rank_records = []
    for epoch in valid_epochs:
        edf = mean_df[mean_df["epoch"] == epoch]
        epoch_values = {row["layer"]: row["normalized_distance"] for _, row in edf.iterrows()}
        rank_map = _rank_series_low_is_good(epoch_values)

        for layer, rank in rank_map.items():
            mean_rank_records.append({
                "epoch": epoch,
                "layer": layer,
                "rank": rank,
                "mean_normalized_distance": epoch_values[layer],
            })

    mean_rank_df = pd.DataFrame(mean_rank_records)
    mean_rank_df.to_csv(output_dir / "layer_mean_rankings_selected_epochs.csv", index=False)

    pivot = mean_rank_df.pivot(index="layer", columns="epoch", values="rank").reindex(global_layers)

    fig, ax = plt.subplots(figsize=(12, 7))
    xs = valid_epochs

    for layer in global_layers:
        ys = pivot.loc[layer, xs].values.astype(float)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=layer_color[layer],
            label=layer,
        )

    ax.set_title("Layer rank bump chart - mean across 3 seeds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rank (1 = lowest normalized distance)")
    ax.invert_yaxis()
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)

    if len(global_layers) <= 16:
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_dir / "layer_rank_bump_mean_across_seeds.png", dpi=220)
    plt.close(fig)

    return {
        "selected_epochs_used": valid_epochs,
        "seed_dirs": seed_dirs,
        "output_dir": str(output_dir),
        "num_layers": len(global_layers),
        "files": [
            "layer_rankings_selected_epochs.csv",
            "layer_normalized_distance_selected_epochs.csv",
            "layer_rank_heatmap_selected_epochs.png",
            "seed_1_layer_rank_bump.png",
            "seed_2_layer_rank_bump.png",
            "seed_3_layer_rank_bump.png",
            "layer_mean_rankings_selected_epochs.csv",
            "layer_rank_bump_mean_across_seeds.png",
        ],
    }

if __name__ == "__main__":
    # Example usage:
    # python layer_convergence.py
    #
    # Then edit these paths as needed.
    dir_list = [
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912447",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912448",
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912449",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912446",
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4912439",
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922834",
        # # # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922831",
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922829",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922828",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922830",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4922832",
        # # # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4951438",
        # # # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4951437",
        # # # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4951413",
        # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4950779",
        # # # # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4960278",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992673",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992674",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992676",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992677",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992678",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_4992679",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5009777",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5013237",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5010149",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5023168",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5022596",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5022825",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5023531",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5032891",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5032892",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5032893",
        # "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5032750",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5045603",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5045604",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5045605",
        "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/results_v2/imnet100_small/results_IMNET100_SMALL_5045606",
    ]
    output_main_dir = "/pfs/work9/workspace/scratch/fr_ad457-pr_pretrain/metrics_v2/layer_convergence"
    for dir_item in dir_list:
        seed_dirs = []
        for i in range(3):
            if not os.path.exists(dir_item + f"/s{i}/checkpoint-299.pth"):
                print(f"Warning: {dir_item}/s{i} does not exist. Skipping.")
                continue
            slurm_id = dir_item.split("_")[-1]
            checkpoint_dir = dir_item + f"/s{i}"
            output_dir = output_main_dir + f"/{slurm_id}/s{i}"
            os.makedirs(output_dir, exist_ok=True)
            results = compute_layer_convergence(
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                norm_mode="l2",   # change to "fro" to use the alternate option
                # plot_top_k=20,
            )
            seed_dirs.append(checkpoint_dir)
        results = visualize_layer_rankings_across_seeds(
            seed_dirs=seed_dirs,
            output_dir=output_main_dir + f"/{slurm_id}",
            # selected_epochs=[100, 150, 200, 250],
            selected_epochs=list(range(50, 200, 10)),  # use this line instead to include more epochs in the analysis
            norm_mode="l2",
        )

        # print(json.dumps(results, indent=2))
