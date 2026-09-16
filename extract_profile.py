"""Extract the checkpoint-free early-lever recipe from a procedural checkpoint.

The recipe (see docs/i100_synthesis.md, section 6) describes blocks 0..8 of a
procedurally pretrained ViT-B by second moments only, so a random timm initialisation
can be rescaled to it without touching the checkpoint at training time. Two
ingredients are read off:

1. Effective weight scales, one per block and linear weight (q, k, v, proj, fc1, fc2):

       effective scale = root mean square(gain of the preceding LayerNorm)
                         * root mean square(weight) / initialisation standard deviation

   "Effective" means: how large the linear layer looks to the forward pass, relative to
   a fresh timm initialisation. In a pre-norm block the input to q, k, v is
   gain * LN(x) + bias, so q = W_q (gain * LN(x)) = (W_q diag(gain)) LN(x). LN(x) has unit
   variance per channel, so the matrix that acts on a unit-variance input is
   W_q diag(gain), whose root mean square is root mean square(W_q) * root mean square(gain) when gain and weight
   columns are uncorrelated. Dividing by timm's truncated-normal standard deviation
   (0.02) turns this into a multiplier: 1.0 means "as large as a random initialisation",
   0.36 means "about a third of it". q, k, v take norm1's gain and fc1 takes norm2's.
   proj and fc2 have no LayerNorm in front, so their effective scale is the raw
   root mean square(W) / 0.02.

   Why the recipe is written in these units: the procedural checkpoints have gains
   around 0.4, not 1. Copying root mean square(W) alone would give a forward pass 2.5 times louder
   than the checkpoint's. utils.apply_analytic_profile therefore samples gains with the
   checkpoint's statistics (ingredient 2) and sets the raw weight to
   effective scale / root mean square(sampled gain). The forward pass then matches the checkpoint
   while the raw input-side matrices come out about 2.5 times larger than timm's, which
   is what makes Adam's relative step on them small (the early lever's carrier, see the
   synthesis document).

   Caveat: root mean square(W) * root mean square(gain) is exact only if gain and weight columns are uncorrelated.
   See "Manual corrections" below for where that fails.

   Default form ("linear", the ftbanap recipe): block 0 keeps its measured value and
   blocks 1..8 are replaced by a least-squares straight line from block 1 to block 8.
   Six linear weights times (block 0, start, end) = 18 numbers.
   With --exact ("exact", the ftbanakx form): every block keeps its measured value,
   6 linear weights times 9 blocks = 54 numbers.

2. LayerNorm statistics: mean and standard deviation of the gain and of the bias of
   norm1 and norm2, per block. 4 vectors times 2 moments times 9 blocks = 72 numbers,
   written inline under "ln.stats" so utils.apply_analytic_profile can sample them
   without the checkpoint. --no_layernorm omits them (gains stay 1, biases 0: the
   ftbana form).

Output: a profile specification for
    --init_method analytic_profile --profile_spec OUT --init_method_scaled_blocks 0,...,8

Manual corrections. The kdyck specification in use (vitbase_runs/profile_ftbanap.json)
deviates from the raw measurement in two places that were found on forward-pass dumps,
not from the weights: q and k flat at 1.32 (the checkpoint's q and k columns are
anti-correlated with the LayerNorm gain, large-gain channels have small q and k weights,
so the folded root mean square overstates the attention logit scale; the measured
1.4 to 1.8 gave logits 1.3 to 1.5 times too sharp on a forward pass) and the fc2 line
ending at 0.95 instead of the fitted value
(block 8's fc2 already grows toward the loud top blocks while its GELU is still off).
--query_key_flat 1.32 --fc2_end 0.95 reproduce them; the printed "max line misfit" then
shows how far the override sits from the measurement.

usage: .venv/bin/python extract_profile.py CHECKPOINT OUT.json [--blocks 0-8] [--exact] [--no_layernorm]
                                            [--query_key_flat X] [--fc2_end Y] [--init_standard_deviation 0.02]
for example:
       .venv/bin/python extract_profile.py results/pr_vitb_n/pr_6066174_final.pth /tmp/kdyck.json --query_key_flat 1.32 --fc2_end 0.95
"""
import argparse
import json

import numpy as np
import torch

LINEAR_WEIGHTS = ("q", "k", "v", "proj", "fc1", "fc2")


# ----------------------------------------------------------------------------- measurement

def root_mean_square(tensor):
    return float(tensor.float().pow(2).mean().sqrt())


def load_state_dict(path):
    """Model weights of a procedural checkpoint (accepts the 'state', 'model' or bare layouts)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("state", checkpoint.get("model", checkpoint))


def effective_scales(state_dict, block, init_standard_deviation):
    """Effective scale of each linear weight of one block, relative to the initialisation standard deviation.

    How large the layer looks to the forward pass: root mean square(W diag(gain)) / 0.02, approximated as
    root mean square(W) * root mean square(gain) / 0.02 (see the module docstring). q, k, v are the three row groups of
    the fused attn.qkv.weight; they take norm1's gain and fc1 takes norm2's. proj and fc2 have
    no LayerNorm in front, so their scale is the raw root mean square(W) / 0.02."""
    prefix = f"blocks.{block}."
    fused_qkv = state_dict[prefix + "attn.qkv.weight"]
    width = fused_qkv.shape[1]
    query, key, value = fused_qkv[:width], fused_qkv[width:2 * width], fused_qkv[2 * width:]
    gain1 = root_mean_square(state_dict[prefix + "norm1.weight"])
    gain2 = root_mean_square(state_dict[prefix + "norm2.weight"])
    return {
        "q": gain1 * root_mean_square(query) / init_standard_deviation,
        "k": gain1 * root_mean_square(key) / init_standard_deviation,
        "v": gain1 * root_mean_square(value) / init_standard_deviation,
        "proj": root_mean_square(state_dict[prefix + "attn.proj.weight"]) / init_standard_deviation,
        "fc1": gain2 * root_mean_square(state_dict[prefix + "mlp.fc1.weight"]) / init_standard_deviation,
        "fc2": root_mean_square(state_dict[prefix + "mlp.fc2.weight"]) / init_standard_deviation,
    }


def layernorm_statistics(state_dict, block):
    """Mean and standard deviation of gain and bias for norm1 and norm2 of one block.

    The key names (gain_mean, gain_std, ...) are what utils.apply_analytic_profile reads."""
    statistics = {}
    for norm_name in ("norm1", "norm2"):
        gain = state_dict[f"blocks.{block}.{norm_name}.weight"].float()
        bias = state_dict[f"blocks.{block}.{norm_name}.bias"].float()
        statistics[norm_name] = {
            "gain_mean": float(gain.mean()), "gain_std": float(gain.std()),
            "bias_mean": float(bias.mean()), "bias_std": float(bias.std()),
        }
    return statistics


# ----------------------------------------------------------------------------- linear fit

def fit_line(values):
    """Least-squares straight line through `values` (one per fitted block). Returns (start, end)."""
    positions = np.arange(len(values))
    slope, intercept = np.polyfit(positions, np.asarray(values), 1)
    return float(intercept), float(intercept + slope * (len(values) - 1))


def line_misfit(values, start, end):
    """Largest relative deviation of the line (start -> end) from the measured values."""
    positions = np.arange(len(values))
    line = start + (end - start) * positions / max(1, len(values) - 1)
    return float(np.max(np.abs(line / np.asarray(values) - 1)))


def apply_corrections(weight_name, start, end, query_key_flat, fc2_end):
    """The two manual overrides of the kdyck recipe (see the module docstring)."""
    if weight_name in ("q", "k") and query_key_flat is not None:
        start = end = query_key_flat
    if weight_name == "fc2" and fc2_end is not None:
        end = fc2_end
    return start, end


# ----------------------------------------------------------------------------- specification + report

def build_profile_specification(arguments):
    """Measure the checkpoint. Returns (specification for apply_analytic_profile, blocks, report rows)."""
    state_dict = load_state_dict(arguments.checkpoint)
    first, last = (int(x) for x in arguments.blocks.split("-"))
    blocks = list(range(first, last + 1))
    first_block, fitted_blocks = blocks[0], blocks[1:]

    scales = {block: effective_scales(state_dict, block, arguments.init_standard_deviation) for block in blocks}

    specification, rows = {}, []
    for weight_name in LINEAR_WEIGHTS:
        measured = [scales[block][weight_name] for block in fitted_blocks]
        start, end = fit_line(measured)
        start, end = apply_corrections(weight_name, start, end, arguments.query_key_flat, arguments.fc2_end)
        misfit = line_misfit(measured, start, end)

        if arguments.exact:
            specification[weight_name] = {"per_block": {str(block): round(scales[block][weight_name], 4) for block in blocks}}
        else:
            specification[weight_name] = {"b0": round(scales[first_block][weight_name], 3),
                                         "start": round(start, 3), "end": round(end, 3)}
        rows.append((weight_name, [scales[block][weight_name] for block in blocks],
                     scales[first_block][weight_name], start, end, misfit))

    if not arguments.no_layernorm:
        specification["ln"] = {
            "gain": True, "bias": True, "source": "parametric",
            "stats": {str(block): layernorm_statistics(state_dict, block) for block in blocks},
            "ckpt": arguments.checkpoint,
        }
    return specification, blocks, rows


def print_scale_table(arguments, blocks, rows):
    print(f"{arguments.checkpoint}: effective scales "
          f"(root mean square(gain) * root mean square(weight) / {arguments.init_standard_deviation}) "
          f"and the linear fit over blocks {blocks[1]}-{blocks[-1]}")
    print("weight| " + " ".join(f"b{block:<5d}" for block in blocks) + " | b0   start -> end   | max line misfit")
    for weight_name, per_block, first_block_value, start, end, misfit in rows:
        cells = " ".join(f"{value:5.2f} " for value in per_block)
        print(f"{weight_name:5s} | {cells} | {first_block_value:4.2f}  {start:5.2f} -> {end:5.2f} | {misfit:5.1%}")


def print_layernorm_table(statistics):
    print("LayerNorm statistics (mean +- standard deviation of gain and bias), norm1 / norm2:")
    for block, per_norm in statistics.items():
        columns = []
        for norm_name in ("norm1", "norm2"):
            moments = per_norm[norm_name]
            columns.append(f"gain {moments['gain_mean']:.3f}+-{moments['gain_std']:.3f} "
                           f"bias {moments['bias_mean']:+.3f}+-{moments['bias_std']:.3f}")
        print(f"  b{block}: " + " / ".join(columns))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="procedural checkpoint (.pth)")
    parser.add_argument("out", help="profile specification to write (.json)")
    parser.add_argument("--blocks", default="0-8",
                        help="inclusive block range; the first block keeps its value, the rest are fitted by a line")
    parser.add_argument("--exact", action="store_true",
                        help="per-block measured scales instead of block 0 + linear fit")
    parser.add_argument("--no_layernorm", action="store_true",
                        help="omit the LayerNorm statistics (gains 1, biases 0: the ftbana form)")
    parser.add_argument("--query_key_flat", type=float, default=None,
                        help="override the q and k lines with this constant (kdyck: 1.32)")
    parser.add_argument("--fc2_end", type=float, default=None,
                        help="override the end of the fc2 line (kdyck: 0.95)")
    parser.add_argument("--init_standard_deviation", type=float, default=0.02,
                        help="standard deviation of timm's truncated-normal initialisation that the scales are relative to")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    specification, blocks, rows = build_profile_specification(arguments)

    print_scale_table(arguments, blocks, rows)
    if "ln" in specification:
        print_layernorm_table(specification["ln"]["stats"])

    with open(arguments.out, "w") as file:
        json.dump(specification, file, indent=1)

    form = "exact per-block" if arguments.exact else "block 0 + linear fit"
    layernorm_note = ("" if arguments.no_layernorm
                      else f" + {4 * len(blocks)} LayerNorm statistics inline (no checkpoint needed at initialisation)")
    print(f"wrote {arguments.out}: {form} scales{layernorm_note}")


if __name__ == "__main__":
    main()
