"""Extract the checkpoint-free early-lever recipe from a procedural checkpoint.

The recipe (see docs/i100_synthesis.md, section 6) describes blocks 0..8 of a
procedurally pretrained ViT-B by second moments only, so a random timm initialisation
can be rescaled to it without touching the checkpoint at training time. Two
ingredients are read off:

1. Effective weight scales, one per block and linear weight (q, k, v, proj, fc1, fc2):

       effective scale = root mean square(gain of the preceding LayerNorm)
                         * root mean square(weight) / initialisation standard deviation

   "Effective" means: the size of the matrix the forward pass applies, relative to a fresh
   timm initialisation. In a pre-norm block the input to q, k, v is
   norm1(x) = gain * standardise(x) + bias, so

       q = W_q (gain * standardise(x) + bias)
         = (W_q diag(gain)) standardise(x)  +  W_q bias

   so the matrix that acts on the standardised token is W_q diag(gain), and only that
   product matters to the function: (c W, gain / c) is the same layer for any c. Its root
   mean square is root mean square(W_q) * root mean square(gain) when gain and weight columns
   are uncorrelated. That product is the effective scale. Dividing by timm's
   truncated-normal standard deviation (0.02) turns it into a multiplier: 1.0 means "as
   large as a random initialisation", 0.36 means "about a third of it". q, k, v take
   norm1's gain and fc1 takes norm2's. proj and fc2 have no LayerNorm in front, so their
   effective scale is the raw root mean square(W) / 0.02.

   What the number does and does not say (plots/verify/effective_scale_meaning.py,
   docs/proc_init_recipe.md section 3). standardise(x) normalises each TOKEN over its
   channels: every token has squared norm d, but the tokens are far from isotropic (one
   direction carries 35-50% of the energy in a random network, 83-85% in the kdyck prefix).
   The size of a matrix therefore fixes the size of its output only for a matrix that is
   independent of the stream -- which the random matrices this recipe rescales are: their
   output root mean square is scale * 0.02 * sqrt(d), measured within 5%. It is NOT the
   size of the checkpoint's own output: the checkpoint's matrices are aligned with the
   stream (half of their size sits in eight singular directions) and their output energy is
   5 to 120 times larger than the scale predicts. That aligned part is what no per-tensor
   moment carries and what the joint statistics (ingredient 3) put back for q and fc1. The
   recipe thus replaces a strongly anisotropic matrix by an isotropic one of equal size.

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
   effective scale / root mean square(sampled gain). The matrix the forward pass applies then
   has the checkpoint's size (not its output: see above) while the raw input-side matrices
   come out about 2.5 times larger than timm's, which is what makes Adam's relative step on
   them small (the early lever's carrier, see the synthesis document).

   Caveat: root mean square(W) * root mean square(gain) equals the exact
   root mean square(W diag(gain)) only if gain_j squared is uncorrelated with the mean
   squared magnitude of column j of W. Measured on both checkpoints (2026-09-16), the
   exact value is larger in every block and weight: by 2 to 11 percent in blocks 1..8
   and by up to 15 percent for q of block 0, so the two are mildly positively correlated. The product is kept as the default because every
   existing specification, arm and verification target is expressed in it;
   --gain_fold exact writes the exact value instead.

   Which weights get an effective scale is a choice (--scale_weights, default all six). The scale carries from one
   network to another only for the weights behind a LayerNorm (q, k, v, fc1), whose input is normalised. proj and fc2
   write into the un-normalised residual stream, so what carries for them is the write ratio
   ||sublayer output|| / ||stream||: --write_ratio names the weights (among v, proj, fc2) that are set that way instead,
   and the checkpoint prefix's per-block ratios are measured and stored as targets (docs/proc_init_recipe.md, section 8).

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

3. Optional joint statistics (--qk_entropy, --fc1_gate, --common_write): how a weight matrix is aligned with the residual stream, which no
   per-tensor moment expresses. In both procedural prefixes two alignments dominate blocks 1..8:
     * W_q maps the direction all tokens share onto one query, so attention is a sink (entropy 0.4 to 1.0 nats on kdyck
       against 5.2 for any random q/k pair; the top singular component holds 22% of W_q's energy, 0.5% if random);
     * the average row of fc1 (22 to 30% of its energy on kdyck, 0.03% if random; the per-tensor scalar mean is only 1% of
       the std, which is why "zero mean" looked safe) points against the stream's common direction and shifts every
       pre-activation by -2 to -2.7, so the GELU is off.
     * (--common_write) block 0's MLP writes one vector shared by all tokens, 80 times the size of the patch embeddings
       (the mean column of its fc2 holds 2.6% of the energy, 0.03% if random). It makes the tokens nearly parallel
       (cosine 0.91) and is the shared direction the two alignments above read. Target: the token cosine of block 0's output.
   Each is written as one functional target per block (attention entropy; mean fc1 pre-activation), measured with a
   forward pass of the checkpoint prefix on training images -- the only data-dependent ingredients. main.py realises them
   with one rank-one component per tensor along the initialised model's own stream direction
   (utils.calibrate_joint_statistics), rescaling the random part so every effective scale above stays exact.
   8 + 8 + 1 numbers, each behind its own flag and specification key, so any subset can be ablated. On kdyck neither is needed for the accuracy gain (second moments suffice, ftbanap 80.24); they make the
   initialisation reproduce the prefix's function, not only its moments.

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
                                            [--fit_upto B] [--qk_entropy] [--fc1_gate] [--common_write] [--data_path D] [--joint_images 256] [--seed 0]
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

    The size of the matrix the forward pass applies, not of its output in the checkpoint (see the module docstring). q, k, v are the
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


# ----------------------------------------------------------------------------- attention-sink targets (forward pass)

def mean_row_energy_share(state_dict, block):
    """Share of fc1's energy (gain folded) carried by its average row: n * ||mean row||^2 / ||W diag(gain)||_F^2.
    A random matrix has 1/n = 0.0003; the kdyck checkpoint 0.22 to 0.30 in blocks 1..8. Weights only, no data."""
    folded = state_dict[f"blocks.{block}.mlp.fc1.weight"].float() * state_dict[f"blocks.{block}.norm2.weight"].float()[None, :]
    return float(folded.shape[0] * folded.mean(0).pow(2).sum() / folded.pow(2).sum())


def mean_column_energy_share(state_dict, block):
    """Share of fc2's energy carried by its average column (over hidden units): n * ||mean column||^2 / ||W||_F^2.
    Random: 1/n = 0.0003; block 0 of the kdyck checkpoint 0.026, its blocks 1..8 0.24 to 0.40. Weights only."""
    weight = state_dict[f"blocks.{block}.mlp.fc2.weight"].float()
    return float(weight.shape[1] * weight.mean(1).pow(2).sum() / weight.pow(2).sum())


def measure_joint_targets(arguments, state_dict, blocks):
    """Functional targets of the two joint statistics, read off the checkpoint prefix at initialisation.

    The prefix is built as main.py builds it for the prefix arms (ftb3i): a fresh timm ViT-B with the checkpoint's tensors
    in `blocks`, everything else (later blocks, ImageNet patch and position embeddings, head) random. Measured on TRAINING
    images under the evaluation transform, chosen by utils.calibration_images -- the protocol main.py uses when it
    calibrates -- for blocks[1:]: block 0 reads the raw embeddings, has no sink (entropy ~4.8) and its MLP is on.
    Imports are local: this is the only part of the script that needs the model code and data."""
    import os
    from torchvision import datasets as torchvision_datasets
    import main as training_main
    import utils
    from datasets import build_transform

    model_arguments = training_main.get_args_parser().parse_args(
        ["--model", "vit_base", "--data_set", "IMNET", "--data_path", arguments.data_path, "--input_size", "224", "--nb_classes", "1000"])
    model_arguments.nb_classes = 1000
    prefix = {name: tensor for name, tensor in state_dict.items()
              if name.startswith("blocks.") and int(name.split(".")[1]) in blocks}
    train_folder = torchvision_datasets.ImageFolder(os.path.join(arguments.data_path, "train"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The targets depend on the random context they are measured in, mostly on the prefix model's random patch / position
    # embeddings and hardly on the image draw (plots/verify/target_stability.py: entropy and write ratios move by 3-17% across
    # seeds, the kdyck active-unit fractions by up to 56%). They are therefore averaged over --target_seeds contexts
    # (seed, seed + 1, ...: a fresh timm model and a fresh draw of training images each), so that no single random
    # embedding is baked into the recipe. The standard deviation over the contexts is stored next to each target.
    seeds = list(range(arguments.seed, arguments.seed + arguments.target_seeds))
    per_seed = []
    for context_seed in seeds:
        torch.manual_seed(context_seed)
        model = utils.build_model(model_arguments)
        model.load_state_dict(prefix, strict=False)
        model.to(device)
        images = utils.calibration_images(train_folder.samples, train_folder.loader, build_transform(False, model_arguments),
                                          arguments.joint_images, context_seed)
        chunks = [utils.joint_statistics_per_block(model, images[start:start + 64].to(device), blocks)
                  for start in range(0, len(images), 64)]      # every quantity is a mean over images: size-weighted chunks average exactly
        sizes = [min(64, len(images) - start) for start in range(0, len(images), 64)]
        per_seed.append({block: {key: float(np.average([chunk[block][key] for chunk in chunks], weights=sizes)) for key in chunks[0][block]}
                         for block in chunks[0]})
    everything = {block: {key: float(np.mean([one[block][key] for one in per_seed])) for key in per_seed[0][block]}
                  for block in per_seed[0]}
    spread = {block: {key: float(np.std([one[block][key] for one in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0
                      for key in per_seed[0][block]} for block in per_seed[0]}
    measured = {block: values for block, values in everything.items() if block != 0}     # block 0 reads the raw embeddings: no sink, MLP on
    print(f"joint-statistic targets (checkpoint prefix, {arguments.joint_images} training images, evaluation transform, "
          f"mean over {len(seeds)} contexts, seeds {seeds}; +- = standard deviation over the contexts):")
    print(f"  b{blocks[0]}: token cosine of the block's output {everything[blocks[0]]['token_cosine']:.3f} "
          f"(mean-column energy share of fc2 {mean_column_energy_share(state_dict, blocks[0]):.4f})")
    for block, values in measured.items():
        print(f"  b{block}: attention entropy {values['entropy']:.3f} +- {spread[block]['entropy']:.3f} nats (top key {values['sink_share']:.2f} of the mass) | "
              f"active units {values['active_units']:.5f} +- {spread[block]['active_units']:.5f}, "
              f"mean fc1 pre-activation {values['pre_activation_mean']:+.3f} +- {spread[block]['pre_activation_mean']:.3f} ("
              f"mean-row energy share {mean_row_energy_share(state_dict, block):.3f})")
    protocol = "training images, evaluation transform, utils.calibration_images"
    context = {"seeds": seeds}           # which random contexts the targets are averaged over
    spread_of = lambda key, which: {str(block): float(f"{spread[block][key]:.3g}") for block in which}
    targets = {}
    if arguments.qk_entropy:
        targets["qk_entropy"] = {"entropy": {str(block): round(values["entropy"], 3) for block, values in measured.items()},
                              **context, "entropy_sd_over_seeds": spread_of("entropy", measured),
                              "images": arguments.joint_images, "renormalize": True, "protocol": protocol}
    if arguments.fc1_gate:
        if arguments.fc1_gate_target == "active_units":
            gate_target = {"active_units": {str(block): float(f"{values['active_units']:.5g}") for block, values in measured.items()},
                           "checkpoint_pre_activation_mean": {str(block): round(values["pre_activation_mean"], 3) for block, values in measured.items()}}
        else:
            gate_target = {"pre_activation_mean": {str(block): round(values["pre_activation_mean"], 3) for block, values in measured.items()},
                           "checkpoint_active_units": {str(block): float(f"{values['active_units']:.5g}") for block, values in measured.items()}}
        targets["fc1_gate"] = {**gate_target, **context, "target_sd_over_seeds": spread_of(arguments.fc1_gate_target, measured),
                               "images": arguments.joint_images, "renormalize": True, "protocol": protocol,
                               # reference only (weights-only reading; the calibration reports what it needed):
                               "checkpoint_mean_row_energy_share": {str(block): round(mean_row_energy_share(state_dict, block), 3) for block in measured}}
    if arguments.write_ratio:
        print("  write ratios (attention / MLP): " + "  ".join(f"b{block} {values['attention_write']:.4f} / {values['mlp_write']:.4f}"
                                                                for block, values in everything.items()))
        targets["write_ratio"] = {"tensors": list(arguments.write_ratio), **context, "images": arguments.joint_images, "protocol": protocol,
                                  "attention_sd_over_seeds": spread_of("attention_write", everything), "mlp_sd_over_seeds": spread_of("mlp_write", everything)}
        if {"v", "proj"} & set(arguments.write_ratio):
            targets["write_ratio"]["attention"] = {str(block): float(f"{values['attention_write']:.5g}") for block, values in everything.items()}
        if "fc2" in arguments.write_ratio:
            targets["write_ratio"]["mlp"] = {str(block): float(f"{values['mlp_write']:.5g}") for block, values in everything.items()}
    if arguments.common_write:
        targets["common_write"] = {"token_cosine": {str(blocks[0]): round(everything[blocks[0]]["token_cosine"], 3)},
                                   **context, "token_cosine_sd_over_seeds": spread_of("token_cosine", [blocks[0]]),
                                   "images": arguments.joint_images, "renormalize": True, "protocol": protocol,
                                   "checkpoint_mean_column_energy_share": {str(blocks[0]): round(mean_column_energy_share(state_dict, blocks[0]), 4)}}
    return targets


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
        if weight_name not in arguments.scale_weights:      # left at timm's initialisation, or set by its write ratio
            continue
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
        if weight_name in ("q", "k") and arguments.query_key == "pooled":
            # the twin's convention is one number: the fused-qkv scale averaged over ALL fitted blocks (kdyck 1.321), flat,
            # independent of --fit_upto. (Until 2026-09-17 this fitted a sloped line through the pooled values, 1.34 -> 1.26,
            # contrary to the docstring; no recorded specification was generated with it.)
            start = end = float(np.mean(measured))
        start, end = apply_corrections(weight_name, start, end, arguments.query_key_flat, arguments.fc2_end)
        misfit = line_misfit(measured, start, end)
        specification[weight_name] = {"b0": round(scales[first_block][source], 3),
                                      "start": round(start, 3), "end": round(end, 3)}
        rows.append((weight_name, per_block, (scales[first_block][source], start, end, misfit)))

    specification["gain_fold"] = arguments.gain_fold          # the folding convention is recorded whatever the realisation
    if arguments.realise == "exact":
        specification["realise"] = "exact"
    if not arguments.no_layernorm:
        specification["ln"] = {
            "gain": True, "bias": True, "source": "parametric",
            "stats": {str(block): layernorm_statistics(state_dict, block) for block in blocks},
            "ckpt": arguments.checkpoint,
        }
    if arguments.qk_entropy or arguments.fc1_gate or arguments.common_write or arguments.write_ratio:
        specification.update(measure_joint_targets(arguments, state_dict, blocks))
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
    parser.add_argument("--realise", choices=("exact", "multiplier"), default="exact",
                        help="how utils.apply_analytic_profile turns a scale into weights: 'exact' (default since 2026-09-17) rescales each "
                             "slice, after the LayerNorm gains have been sampled, until the declared quantity (--gain_fold) holds to float "
                             "precision; 'multiplier' = W <- (scale / rms(gain)) * W_timm, which meets it within 0.2%% and is what every "
                             "specification written before that date does (they carry no 'realise' key)")
    parser.add_argument("--qk_entropy", action="store_true",
                        help="joint statistic 1 (target = mean attention ENTROPY per block, i.e. how sharp attention is; that all queries pick the "
                             "same key is a property of the rank-one construction, reported as 'sink share' and compared in "
                             "plot_reconstruction.py, not part of the target): also measure the checkpoint prefix's per-block attention entropy at initialisation "
                             "and write it as 'qk_entropy' targets; main.py then installs a rank-one coupled q/k sink "
                             "(utils.calibrate_joint_statistics; effective scales preserved)")
    parser.add_argument("--fc1_gate", action="store_true",
                        help="joint statistic 2: also measure the prefix's per-block mean fc1 pre-activation and write it as 'fc1_gate' "
                             "targets; main.py then installs a rank-one mean-row component in fc1 (the GELU gate)")
    parser.add_argument("--fc1_gate_target", choices=("active_units", "pre_activation_mean"), default="active_units",
                        help="what the fc1 gate is calibrated to: 'active_units' = the fraction of fc1 pre-activations > 0 over images, tokens and "
                             "units (a proxy for how much of the MLP is switched on: GELU is small but not zero below 0, and the same fraction "
                             "can come from different distributions, so mean, std and GELU rms are reported next to it), or 'pre_activation_mean', the "
                             "target of the first reconstruction specifications (ftbanaperg*, ftbanakperg*)")
    parser.add_argument("--common_write", action="store_true",
                        help="joint statistic 3: also measure the token cosine of the first block's output in the prefix and write it as "
                             "'common_write' target; main.py then installs a rank-one mean-column component in that block's fc2 "
                             "(block 0 floods the stream with one shared vector, which the sink and the gate of later blocks read)")
    parser.add_argument("--scale_weights", default=",".join(LINEAR_WEIGHTS),
                        help="comma-separated linear weights that receive an effective scale; the others stay at timm's initialisation "
                             "(q,k,fc1 = the input side without v: the ftbanapeb7i form)")
    parser.add_argument("--write_ratio", default="",
                        help="comma-separated write-side weights among v,proj,fc2 that are NOT given an effective scale but multiplied by one "
                             "scalar per sublayer until the block's write ratio ||sublayer output|| / ||stream|| equals the checkpoint "
                             "prefix's, measured here and written as 'write_ratio' targets (utils.calibrate_joint_statistics). "
                             "'v,proj,fc2': attention factor split evenly over v and proj, as the late lever does; 'proj,fc2': v keeps "
                             "its effective scale (list it in --scale_weights) and proj takes the whole attention factor")
    parser.add_argument("--data_path", default="/data/datasets/ILSVRC2012",
                        help="ImageNet root with a train/ folder; the joint statistics (--qk_entropy, --fc1_gate, --common_write, --write_ratio) "
                             "are the only parts of this script that run a forward pass")
    parser.add_argument("--joint_images", type=int, default=256, help="training images used to measure the targets and, in main.py, to calibrate")
    parser.add_argument("--seed", type=int, default=0, help="first seed of the random parts of the prefix model and of the image choice")
    parser.add_argument("--target_seeds", type=int, default=5,
                        help="number of random contexts (seed, seed + 1, ...) the functional targets are averaged over; 1 = the single-context "
                             "targets of every specification written before 2026-09-17")
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
    if not arguments.exact and last - first < 2:
        parser.error("the linear form needs the first block plus at least two fitted blocks (a line through one point is undetermined)")
    if not arguments.exact and first != 0:
        parser.error("the linear form anchors on block 0 (utils.apply_analytic_profile treats block 0 literally as 'b0' "
                     "and ramps over the other listed blocks); use --blocks 0-N or --exact")
    arguments.scale_weights = [name for name in arguments.scale_weights.split(",") if name]
    arguments.write_ratio = [name for name in arguments.write_ratio.split(",") if name]
    if set(arguments.scale_weights) - set(LINEAR_WEIGHTS):
        parser.error(f"--scale_weights must be among {','.join(LINEAR_WEIGHTS)}, got {arguments.scale_weights}")
    if set(arguments.write_ratio) - {"v", "proj", "fc2"}:
        parser.error(f"--write_ratio must be among v,proj,fc2 (the weights on the write path), got {arguments.write_ratio}")
    if set(arguments.write_ratio) & set(arguments.scale_weights):
        parser.error(f"{sorted(set(arguments.write_ratio) & set(arguments.scale_weights))} would be set twice: a weight is given either an "
                     "effective scale (--scale_weights) or a write ratio (--write_ratio)")
    if "v" in arguments.write_ratio and "proj" not in arguments.write_ratio:
        parser.error("--write_ratio v without proj leaves proj's scale undefined for the attention write; use v,proj or proj")
    if arguments.common_write and first != 0:
        parser.error("--common_write describes block 0 (it floods the stream with one shared vector); use a block range starting at 0")
    if arguments.target_seeds < 1:
        parser.error("--target_seeds must be at least 1")
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

    form = ("exact per-block" if arguments.exact else "block 0 + linear fit") + f" ({','.join(arguments.scale_weights)})"
    layernorm_note = ("" if arguments.no_layernorm
                      else f" + {8 * len(blocks)} LayerNorm statistics inline (no checkpoint needed at initialisation)")
    sink_note = ("" if not arguments.qk_entropy else f" + {len(specification['qk_entropy']['entropy'])} attention-sink targets") + \
                ("" if not arguments.fc1_gate else f" + {len(specification['fc1_gate'][arguments.fc1_gate_target])} fc1-gate targets ({arguments.fc1_gate_target})") + \
                ("" if not arguments.common_write else " + 1 common-write target") + \
                ("" if not arguments.write_ratio else
                 f" + {sum(len(specification['write_ratio'].get(key, {})) for key in ('attention', 'mlp'))} write-ratio targets "
                 f"({','.join(arguments.write_ratio)})")
    print(f"wrote {arguments.out}: {form} scales{layernorm_note}{sink_note}")


if __name__ == "__main__":
    main()
