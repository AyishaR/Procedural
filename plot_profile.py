"""Plot the effective scales of a procedural checkpoint before and after the linear fit.

One panel per linear weight (q, k, v, proj, fc1, fc2), blocks on the x axis, effective
scale (relative to timm's initialisation, see extract_profile.py) on the y axis:

    measured                 the checkpoint's per-block effective scale, what extract_profile.py reads
    linear fit to measured   the least-squares straight line through blocks 1..8
    effective scales used    block 0 plus the line as written to the specification, i.e. the
                             linear fit after the corrections (--query_key pooled /
                             --query_key_flat / --fc2_end); where no correction applies it
                             covers the linear fit

Under --query_key pooled the measured q and k stay the separate per-weight values, so
the panels show what the pooling changed. Blocks outside --blocks (9-11 by default, left
at the random initialisation by the recipe) are drawn in grey for context.

Takes the same options as extract_profile.py, so the figure shows exactly the
specification that the same command line would write.

Styled with tueplots' ICLR 2024 bundle (full text width, LaTeX text; --no_tex without LaTeX).

usage: .venv/bin/python plot_profile.py CHECKPOINT [--figure plots/out/profile_<checkpoint>.png] [extract_profile.py options]
for example (the ftbanap specification):
       .venv/bin/python plot_profile.py results/pr_vitb_n/pr_6066174_final.pth --query_key pooled --fc2_end 0.95
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from tueplots import bundles

import extract_profile

ROOT = os.path.dirname(os.path.abspath(__file__))
PALETTE = sns.color_palette("tab10")
COLOR_MEASURED, COLOR_SPECIFICATION, COLOR_LINEAR_FIT, COLOR_REFERENCE = PALETTE[0], PALETTE[1], "#8a8a8a", "#b0b0b0"
LABELS = {"q": "query", "k": "key", "v": "value", "proj": "proj",
          "fc1": "fc1", "fc2": "fc2"}


def parse_arguments():
    parser = extract_profile.build_parser(description=__doc__, output_required=False)
    parser.add_argument("--figure", default=None,
                        help="output image; default plots/out/profile_<checkpoint stem>.png (a .pdf is written alongside)")
    parser.add_argument("--no_tex", action="store_true", help="render text with matplotlib instead of LaTeX")
    arguments = parser.parse_args()
    extract_profile.validate_arguments(parser, arguments)
    if arguments.exact:
        parser.error("--exact writes the measured values unchanged; there is no fit to plot")
    if arguments.figure is None:
        stem = os.path.splitext(os.path.basename(arguments.checkpoint))[0]
        arguments.figure = os.path.join(ROOT, "plots", "out", f"profile_{stem}.png")
    return arguments


def line_over(blocks, start, end):
    """The straight line from `start` at the first block to `end` at the last one."""
    positions = np.arange(len(blocks))
    return start + (end - start) * positions / max(1, len(blocks) - 1)


def main():
    arguments = parse_arguments()
    specification, blocks, rows = extract_profile.build_profile_specification(arguments)
    fitted_blocks = blocks[1:]
    # "measured" is always the checkpoint's separate per-weight value, also for q and k under --query_key pooled,
    # so the panel shows what the pooling changed
    state_dict = extract_profile.load_state_dict(arguments.checkpoint)
    depth = 1 + max(int(key.split(".")[1]) for key in state_dict if key.startswith("blocks."))
    reference_blocks = [block for block in range(depth) if block not in blocks]     # outside the recipe, shown for context
    measured = {block: extract_profile.effective_scales(state_dict, block, arguments.init_standard_deviation,
                                                        arguments.gain_fold) for block in range(depth)}

    plt.rcParams.update(bundles.iclr2024(usetex=not arguments.no_tex, family="serif", nrows=2, ncols=3))
    width = plt.rcParams["figure.figsize"][0]
    figure, axes = plt.subplots(2, 3, sharex=True, figsize=(width, 0.62 * width))

    for axis, (weight_name, _, (first_block_value, start, end, misfit)) in zip(axes.flat, rows):
        per_block = [measured[block][weight_name] for block in blocks]
        linear_fit = extract_profile.fit_line(per_block[1:])

        axis.plot(blocks, per_block, "o-", color=COLOR_MEASURED, linewidth=1, markersize=2.5, label="measured")
        if reference_blocks:
            axis.plot(reference_blocks, [measured[block][weight_name] for block in reference_blocks], "o-",
                      color=COLOR_MEASURED, alpha=0.35, linewidth=1, markersize=2.5,
                      label="measured, blocks left random")
            axis.axvspan(blocks[-1] + 0.5, reference_blocks[-1] + 0.5, color=COLOR_REFERENCE, alpha=0.08, linewidth=0)
        axis.plot(fitted_blocks, line_over(fitted_blocks, *linear_fit), "--", color=COLOR_LINEAR_FIT,
                  linewidth=1, label="linear fit to measured")
        used_line, = axis.plot(fitted_blocks, line_over(fitted_blocks, start, end), "-", color=COLOR_SPECIFICATION,
                               linewidth=1.5, label="effective scales used")
        used_block0, = axis.plot([blocks[0]], [first_block_value], "s", color=COLOR_SPECIFICATION, markersize=4,
                                 markerfacecolor="white", markeredgewidth=1.2, label="_nolegend_")

        axis.set_title(LABELS[weight_name])
        axis.grid(True, color="#e6e6e6", linewidth=0.5)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        all_blocks = blocks + reference_blocks
        axis.set_xticks(all_blocks)
        axis.set_xticklabels([str(block) if block % 2 == 0 else "" for block in all_blocks])

    for axis in axes[-1]:
        axis.set_xlabel("block")
    for axis in axes[:, 0]:
        axis.set_ylabel("effective scale")

    handles, labels = {}, []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in handles:
                handles[label] = handle
                labels.append(label)
    handles["effective scales used"] = (used_block0, used_line)      # block-0 square and the line share one entry
    figure.legend([handles[l] for l in labels], labels, loc="outside lower center", ncol=2, frameon=False,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

    os.makedirs(os.path.dirname(arguments.figure), exist_ok=True)
    figure.savefig(arguments.figure, dpi=300)
    figure.savefig(os.path.splitext(arguments.figure)[0] + ".pdf")
    print(f"wrote {arguments.figure} and .pdf")


if __name__ == "__main__":
    main()
