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
   norm1(x) = gain * standardise(x) + bias, so

       q = W_q (gain * standardise(x) + bias)
         = (W_q diag(gain)) standardise(x)  +  W_q bias

   standardise(x) has unit variance per channel, so the matrix that acts on a
   unit-variance input is W_q diag(gain), whose root mean square is
   root mean square(W_q) * root mean square(gain) when gain and weight columns are
   uncorrelated. That product is the effective scale. Dividing by timm's
   truncated-normal standard deviation (0.02) turns it into a multiplier: 1.0 means "as
   large as a random initialisation", 0.36 means "about a third of it". q, k, v take
   norm1's gain and fc1 takes norm2's. proj and fc2 have no LayerNorm in front, so their
   effective scale is the raw root mean square(W) / 0.02.

   The second term, W_q bias, is a constant offset added to every token, independent of
   the input, so it is not part of the scale. Ingredient 2 records the mean and standard
   deviation of each LayerNorm bias, which reproduces the size of that offset, but not
   its direction relative to the rows of W, since the sampled bias and the random W are
   independent. On kdyck that direction is inert (ftbanab; removing the norm2 bias
   moves the fc1 pre-activation mean by at most 0.4 of its -2 to -2.7). On ksd it is part
   of the mechanism that switches the middle MLPs off: the fc1 pre-activations have a
   mean of -2.3 to -3.2, of which 1.0 to 1.5 comes from the rows' anti-alignment with the
   norm2 bias and the rest from their alignment with the standardised stream's common
   direction (measured 2026-09-14/16); no second-moment recipe reproduces either part.

   Why the recipe is written in these units: the procedural checkpoints have gains
   around 0.4, not 1. Copying root mean square(W) alone would give a forward pass 2.5 times louder
   than the checkpoint's. utils.apply_analytic_profile therefore samples gains with the
   checkpoint's statistics (ingredient 2) and sets the raw weight to
   effective scale / root mean square(sampled gain). The forward pass then matches the checkpoint
   while the raw input-side matrices come out about 2.5 times larger than timm's, which
   is what makes Adam's relative step on them small (the early lever's carrier, see the
   synthesis document).

   Caveat: root mean square(W) * root mean square(gain) equals the exact
   root mean square(W diag(gain)) only if gain_j squared is uncorrelated with the mean
   squared magnitude of column j of W. Measured on both checkpoints (2026-09-16), the
   exact value is larger in every block and weight: by 2 to 11 percent in blocks 1..8
   and by up to 15 percent for q of block 0, so the two are mildly positively correlated. The product is kept as the default because every
   existing specification, arm and verification target is expressed in it;
   --gain_fold exact writes the exact value instead.

   Default form ("linear", the ftbanap recipe): block 0 keeps its measured value and
   blocks 1..8 are replaced by a least-squares straight line from block 1 to block 8.
   Six linear weights times (block 0, start, end) = 18 numbers.
   With --exact ("exact", the ftbanakx form): every block keeps its measured value,
   6 linear weights times 9 blocks = 54 numbers.

2. LayerNorm statistics: mean and standard deviation of the gain and of the bias of
   norm1 and norm2, per block. 4 vectors times 2 moments times 9 blocks = 72 numbers,
   written inline under "ln.stats" so utils.apply_analytic_profile can sample them
   without the checkpoint (it only opens the file named under "ln.ckpt" when "ln.stats"
   is absent, so that key is provenance here). --no_layernorm omits them (gains stay 1, biases 0: the
   ftbana form).

Output: a profile specification for
    --init_method analytic_profile --profile_spec OUT --init_method_scaled_blocks 0,...,8

Corrections. The kdyck specification in use (vitbase_runs/profile_ftbanap.json)
deviates from the raw per-slice measurement in two places.

q and k flat at 1.32. This is not a free constant: it is the effective scale of the
*fused* qkv matrix treated as one tensor (product folding, mean over blocks 1..8 =
1.321, range 1.27..1.42), i.e. the convention of the quantile twin ftbqmlnvo, whose
q and k slices are drawn from the pooled q+k+v value distribution
(--quantile_qkv_mode v_only pools q and k). The recipe was calibrated to reproduce that
twin on a forward pass, and the measured 1.4 (q) / 1.7 (k) gave logits 1.3 to 1.5 times
sharper than the twin's because the twin itself is at the pooled 1.32. Whether the
checkpoint's unpooled q/k scale would work as well is untested as a Gaussian recipe
(the qk_v shuffle ftb4e3fix, which keeps it, ends at 79.50 against the pooled twin's
79.93, within seed noise). --query_key pooled derives the value from the checkpoint;
--query_key_flat X sets it by hand.

fc2 line ending at 0.95 instead of the fitted 1.05. Block 8's fc2 (1.19) already grows
toward the loud top blocks while its GELU is still off, and a least-squares line over
blocks 1..8 is pulled up by that one point (it misfits blocks 1..7 by up to 8%). Fitting
the early-regime trend over blocks 1..7 and extrapolating it to block 8 gives
0.738 -> 0.956 (misfit over 1..7: 2.7%), i.e. the hand value; --fit_upto 7 derives it
for every slice with one rule (v, proj, fc1 move by a few percent, q/k are pooled-flat
anyway). --fc2_end X sets the end by hand; the printed "max line misfit" shows how far
each line sits from the measurement.

usage: .venv/bin/python extract_profile.py CHECKPOINT OUT.json [--blocks 0-8] [--exact] [--no_layernorm]
                                            [--query_key_flat X] [--fc2_end Y] [--init_standard_deviation 0.02]
                                            [--query_key separate|pooled] [--gain_fold product|exact]
                                            [--fit_upto B]
for example (the ftbanap specification, derived, no hand constant):
       .venv/bin/python extract_profile.py results/pr_vitb_n/pr_6066174_final.pth /tmp/kdyck.json --query_key pooled --fit_upto 7
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state", checkpoint.get("model", checkpoint))


def gain_folded_scale(weight, gain, init_standard_deviation, gain_fold):
    """Root mean square of the matrix the forward pass applies to the standardised stream, W diag(gain),
    relative to the initialisation standard deviation. "product" approximates it as
    root mean square(W) * root mean square(gain) (the units of every existing specification);
    "exact" computes it."""
    if gain_fold == "exact":
        return root_mean_square(weight.float() * gain.float()[None, :]) / init_standard_deviation
    return root_mean_square(weight) * root_mean_square(gain) / init_standard_deviation


def effective_scales(state_dict, block, init_standard_deviation, gain_fold="product"):
    """Effective scale of each linear weight of one block, relative to the initialisation standard deviation.

    How large the layer looks to the forward pass (see the module docstring). q, k, v are the
    three row groups of the fused attn.qkv.weight; they take norm1's gain and fc1 takes norm2's.
    proj and fc2 have no LayerNorm in front, so their scale is the raw root mean square(W) / 0.02."""
    prefix = f"blocks.{block}."
    fused_qkv = state_dict[prefix + "attn.qkv.weight"]
    width = fused_qkv.shape[1]
    query, key, value = fused_qkv[:width], fused_qkv[width:2 * width], fused_qkv[2 * width:]
    gain1 = state_dict[prefix + "norm1.weight"]
    gain2 = state_dict[prefix + "norm2.weight"]
    return {
        "q": gain_folded_scale(query, gain1, init_standard_deviation, gain_fold),
        "k": gain_folded_scale(key, gain1, init_standard_deviation, gain_fold),
        # the fused matrix as one tensor: the twin's convention for q and k (--query_key pooled)
        "qkv_pooled": gain_folded_scale(fused_qkv, gain1, init_standard_deviation, gain_fold),
        "v": gain_folded_scale(value, gain1, init_standard_deviation, gain_fold),
        "proj": root_mean_square(state_dict[prefix + "attn.proj.weight"]) / init_standard_deviation,
        "fc1": gain_folded_scale(state_dict[prefix + "mlp.fc1.weight"], gain2, init_standard_deviation, gain_fold),
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
    first, last = (int(x) for x in arguments.blocks.split("-"))   # validated in parse_arguments
    blocks = list(range(first, last + 1))
    first_block, fitted_blocks = blocks[0], blocks[1:]

    scales = {block: effective_scales(state_dict, block, arguments.init_standard_deviation, arguments.gain_fold)
              for block in blocks}

    specification, rows = {}, []
    for weight_name in LINEAR_WEIGHTS:
        source = "qkv_pooled" if (weight_name in ("q", "k") and arguments.query_key == "pooled") else weight_name
        per_block = [scales[block][source] for block in blocks]
        if arguments.exact:
            specification[weight_name] = {"per_block": {str(block): round(scales[block][source], 4) for block in blocks}}
            rows.append((weight_name, per_block, None))
            continue
        measured = [scales[block][source] for block in fitted_blocks]
        if arguments.fit_upto is not None:      # fit the early-regime trend and extrapolate it to the last block
            n_fit = arguments.fit_upto - fitted_blocks[0] + 1
            start, end_fit = fit_line(measured[:n_fit])
            end = start + (end_fit - start) * (len(measured) - 1) / max(1, n_fit - 1)
        else:
            start, end = fit_line(measured)
        start, end = apply_corrections(weight_name, start, end, arguments.query_key_flat, arguments.fc2_end)
        misfit = line_misfit(measured, start, end)
        specification[weight_name] = {"b0": round(scales[first_block][source], 3),
                                      "start": round(start, 3), "end": round(end, 3)}
        rows.append((weight_name, per_block, (scales[first_block][source], start, end, misfit)))

    if not arguments.no_layernorm:
        specification["ln"] = {
            "gain": True, "bias": True, "source": "parametric",
            "stats": {str(block): layernorm_statistics(state_dict, block) for block in blocks},
            "ckpt": arguments.checkpoint,
        }
    return specification, blocks, rows


def print_scale_table(arguments, blocks, rows):
    formula = ("root mean square(weight diag(gain))" if arguments.gain_fold == "exact"
               else "root mean square(gain) * root mean square(weight)")
    fit_note = "" if arguments.exact else f" and the linear fit over blocks {blocks[1]}-{blocks[-1]}"
    print(f"{arguments.checkpoint}: effective scales ({formula} / {arguments.init_standard_deviation}){fit_note}")
    header = "weight| " + " ".join(f"b{block:<5d}" for block in blocks)
    print(header if arguments.exact else header + " | b0   start -> end   | max line misfit")
    for weight_name, per_block, line in rows:
        cells = " ".join(f"{value:5.2f} " for value in per_block)
        if line is None:
            print(f"{weight_name:5s} | {cells}")
        else:
            first_block_value, start, end, misfit = line
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


def build_parser(description=__doc__, output_required=True):
    """The extractor's command line. plot_profile.py reuses it so both scripts take the same options."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="procedural checkpoint (.pth)")
    if output_required:
        parser.add_argument("out", help="profile specification to write (.json)")
    parser.add_argument("--blocks", default="0-8",
                        help="inclusive block range; the first block keeps its value, the rest are fitted by a line")
    parser.add_argument("--exact", action="store_true",
                        help="per-block measured scales instead of block 0 + linear fit")
    parser.add_argument("--no_layernorm", action="store_true",
                        help="omit the LayerNorm statistics (gains 1, biases 0: the ftbana form)")
    parser.add_argument("--query_key", choices=("separate", "pooled"), default="separate",
                        help="'separate' = q and k measured as their own row groups; 'pooled' = both take the effective scale of the "
                             "fused qkv matrix (the quantile twin's convention; kdyck: 1.32 over blocks 1-8)")
    parser.add_argument("--query_key_flat", type=float, default=None,
                        help="override the q and k lines with this constant (kdyck: 1.32, which --query_key pooled derives)")
    parser.add_argument("--fc2_end", type=float, default=None,
                        help="override the end of the fc2 line (kdyck: 0.95, which --fit_upto 7 derives)")
    parser.add_argument("--fit_upto", type=int, default=None,
                        help="fit the line over blocks 1..B only and extrapolate it to the last block (all slices); "
                             "kdyck: B=7 keeps block 8's turn toward the loud top out of the fit and reproduces the ftbanap line")
    parser.add_argument("--gain_fold", choices=("product", "exact"), default="product",
                        help="how the LayerNorm gain enters q, k, v, fc1: 'product' = root mean square(W) * root mean square(gain), "
                             "the units of every existing specification; 'exact' = root mean square(W diag(gain))")
    parser.add_argument("--init_standard_deviation", type=float, default=0.02,
                        help="standard deviation of timm's truncated-normal initialisation that the scales are relative to")
    return parser


def parse_arguments():
    parser = build_parser()
    arguments = parser.parse_args()
    validate_arguments(parser, arguments)
    return arguments


def validate_arguments(parser, arguments):
    """Rejects option combinations that would silently do the wrong thing."""
    first, last = parse_block_range(parser, arguments.blocks)
    if not arguments.exact and last - first < 1:
        parser.error("the linear form needs at least two blocks (block 0 plus one to fit)")
    if not arguments.exact and first != 0:
        parser.error("the linear form anchors on block 0 (utils.apply_analytic_profile treats block 0 literally as 'b0' "
                     "and ramps over the other listed blocks); use --blocks 0-N or --exact")
    if arguments.query_key == "pooled" and arguments.query_key_flat is not None:
        parser.error("--query_key pooled derives the q/k value; do not combine it with --query_key_flat")
    if arguments.exact and (arguments.query_key_flat is not None or arguments.fc2_end is not None or arguments.fit_upto is not None):
        parser.error("--query_key_flat, --fc2_end and --fit_upto shape the fitted line and have no meaning with --exact")
    if arguments.fit_upto is not None and not (first + 2 <= arguments.fit_upto < last):
        parser.error(f"--fit_upto must lie in [{first + 2}, {last - 1}] (at least two fitted blocks, and at least one extrapolated)")


def parse_block_range(parser, text):
    """'0-8' -> (0, 8); rejects malformed or descending ranges."""
    try:
        first, last = (int(x) for x in text.split("-"))
    except ValueError:
        parser.error(f"--blocks must look like 0-8, got {text!r}")
    if first < 0 or first > last:
        parser.error(f"--blocks must be an ascending range of non-negative blocks, got {text!r}")
    return first, last


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
                      else f" + {8 * len(blocks)} LayerNorm statistics inline (no checkpoint needed at initialisation)")
    print(f"wrote {arguments.out}: {form} scales{layernorm_note}")


if __name__ == "__main__":
    main()
