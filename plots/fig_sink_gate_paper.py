"""Paper figure: attention sink, MLP gate and effective weight scales of the procedurally pretrained blocks, against a random
initialisation. Plot style follows visualise/report_plots_clean.ipynb and visualise/delta_norm_epoch_analysis_clean.ipynb
(tueplots iclr2024 bundle with usetex, no top/right spines, seaborn tab10 colours, legend above the axes, grid alpha 0.3,
mean +- standard deviation bands).

Per-block quantities of a ViT-B/16 BEFORE any ImageNet training, measured on random ImageNet training images:

  entropy   attention entropy: mean of -sum_j p_j ln p_j over the attention rows (images, heads, query tokens), in nats;
            ln(197) = 5.28 is uniform attention, a sink drives it towards 0
  preact    mean of the MLP's fc1 output before the GELU, over images, tokens and hidden units (--preact mean, the statistic the
            recipe's gate is calibrated to), or the mean L2 norm of a token's pre-activation vector (--preact norm)
  scale     effective weight scale of q, k and fc1, the three matrices the committed recipe rescales (extract_profile.py,
            --gain_fold exact): rms(W diag(gamma)) / 0.02 with gamma the gain of the LayerNorm the matrix reads, i.e. the size of
            the matrix the forward pass applies relative to the initialisation standard deviation. Weights only, no images;
            the random init reads 1.00 by construction.
  active    fraction of positive fc1 pre-activations (the GELU passes those; the rest is switched off); off by default

Series: random init (timm), k-Dyck-D4 (kdyck) and k-Dyck-Shuffled-D98 (ksd). A procedural series is a fresh timm ViT-B whose
12 blocks hold the checkpoint's tensors; the ImageNet patch / position embeddings and the class token stay random, as in every
run that loads a procedural checkpoint (utils.pr_load_model drops them). The statistics of block b depend on blocks 0..b only,
so blocks 0-7 are the same construction as the blocks-0-7 prefix arm (ftb4i), up to the seed of the random parts. Every series
is measured in --seeds random contexts (seed s:
torch.manual_seed(s), then the model is built; the three series of one seed share the random embeddings) on the same
--n_images training images (seeded uniform draw, evaluation transform). Lines are the mean over the contexts, bands +- one
standard deviation (the procedural scales do not depend on the context: no band).

Everything is measured here with hooks on the model's own forward pass; nothing is read from a cache or a profile JSON. The
numbers are written next to the figures (statistics.json), and --replot redraws from that file without a GPU.

usage (on a GPU node, ~3 min):   .venv/bin/python plots/fig_sink_gate_paper.py
       restyle only:             .venv/bin/python plots/fig_sink_gate_paper.py --replot [--preact norm] [--blocks 0-7] [--panels entropy,preact,scale,active]
output: plots/out/fig_sink_gate/early_{row,column}.pdf, early_<panel>.pdf + early_legend.pdf (and .png previews)
  early_row      one figure at full line width, one panel per quantity side by side
  early_column   one figure of --panel_width line widths (default 1/2), the panels stacked and sharing the layer axis
  early_<panel>  the panels as separate files of --panel_width line widths each (for subfigures), early_legend.pdf the legend strip"""
import argparse, contextlib, hashlib, io, json, math, os, shutil, sys, textwrap, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKPOINTS = {"kdyck": "results/pr_vitb_n/pr_6066174_final.pth", "ksd": "results/pr_vitb_ksd/pr_6463456_final.pth"}
SERIES = [("random", "Random init", 0), ("kdyck", "k-Dyck-D4", 1), ("ksd", "k-Dyck-Shuffled-D98", 2)]   # key, label, tab10 slot
SCALED = [("q", "-", r"$W_Q$"), ("k", "--", r"$W_K$"), ("fc1", ":", r"$W_{\mathrm{fc1}}$")]   # the recipe's matrices: key, line style, label
INIT_STD, GAIN_FOLD = 0.02, "exact"                                                           # the committed arms' convention
PANELS = {"entropy": ("entropy", "Attention entropy"), "preact": None, "scale": ("scale", "Effective weight scale"),
          "active": ("active_units", "Fraction of active fc1 units")}


# ---------------------------------------------------------------------------------------------------------------- measurement
class BlockStatistics:
    """Sums of the forward-pass quantities per block, collected by hooks while the model runs its own forward pass.

    attention: a pre-hook on block.attn receives norm1(stream), i.e. exactly what the attention reads; the probabilities are
    recomputed from it with the block's own qkv, q/k norm and scale (fused attention does not expose them).
    MLP: a forward hook on block.mlp.fc1 receives the pre-activations themselves."""

    def __init__(self, model):
        self.sums = [dict(entropy=0.0, rows=0, preact=0.0, preact_norm=0.0, active=0.0, tokens=0, entries=0) for _ in model.blocks]
        self.sequence_length = None                                                                # tokens per image, set by the first hook
        self.handles = []
        for index, block in enumerate(model.blocks):
            self.handles.append(block.attn.register_forward_pre_hook(lambda module, inputs, index=index: self._attention(index, module, inputs[0])))
            self.handles.append(block.mlp.fc1.register_forward_hook(lambda module, inputs, output, index=index: self._fc1(index, output)))

    @torch.no_grad()
    def _attention(self, index, attn, x):
        B, N, C = x.shape
        self.sequence_length = N
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if hasattr(attn, "q_norm"):                                                              # timm >= 0.9 attention; the repo's own has none
            q, k = attn.q_norm(q), attn.k_norm(k)
        probabilities = ((q @ k.transpose(-2, -1)).double() * attn.scale).softmax(dim=-1)        # (B, heads, N, N), rows sum to 1
        self.sums[index]["entropy"] += float(torch.special.entr(probabilities).sum())
        self.sums[index]["rows"] += B * attn.num_heads * N

    @torch.no_grad()
    def _fc1(self, index, pre_activation):
        s, z = self.sums[index], pre_activation.double()                                        # (B, N, hidden)
        s["preact"] += float(z.sum()); s["preact_norm"] += float(z.norm(dim=-1).sum()); s["active"] += float((z > 0).sum())
        s["tokens"] += z.shape[0] * z.shape[1]; s["entries"] += z.numel()

    def result(self):
        for handle in self.handles:
            handle.remove()
        return {"entropy": [s["entropy"] / s["rows"] for s in self.sums],
                "preact_mean": [s["preact"] / s["entries"] for s in self.sums],
                "preact_norm": [s["preact_norm"] / s["tokens"] for s in self.sums],
                "active_units": [s["active"] / s["entries"] for s in self.sums]}


def effective_scales(model):
    """{"q" | "k" | "fc1": [per block]} of the model's current weights, in the committed recipe's convention."""
    from extract_profile import effective_scales as scales_of_block
    state_dict = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    per_block = [scales_of_block(state_dict, block, INIT_STD, GAIN_FOLD) for block in range(len(model.blocks))]
    return {name: [one[name] for one in per_block] for name, *_ in SCALED}


def training_images(data_path, n, seed, model_arguments):
    """`n` training images drawn uniformly without replacement (seeded), under the evaluation transform. Returns (tensor, paths)."""
    from torchvision import datasets as torchvision_datasets
    from datasets import build_transform
    t0 = time.time()
    folder = torchvision_datasets.ImageFolder(os.path.join(data_path, "train"), transform=build_transform(False, model_arguments))
    index = torch.randperm(len(folder), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(folder, index), batch_size=50, shuffle=False,
                                         num_workers=min(16, len(os.sched_getaffinity(0))))
    images = torch.cat([batch for batch, _ in loader])
    print(f"{len(images)} of {len(folder)} training images from {data_path} in {time.time() - t0:.0f} s", flush=True)
    return images, [folder.samples[i][0] for i in index]


@torch.no_grad()
def measure(arguments):
    with contextlib.redirect_stdout(io.StringIO()):          # main.py prints on import
        import main as training_main
        import utils
    from extract_profile import load_state_dict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible, measuring on the CPU (slow; use a GPU node)", flush=True)
    model_arguments = training_main.get_args_parser().parse_args(
        ["--model", "vit_base", "--data_set", "IMNET", "--data_path", arguments.data_path, "--input_size", "224", "--nb_classes", "1000"])
    model_arguments.nb_classes = 1000
    images, paths = training_images(arguments.data_path, arguments.n_images, arguments.image_seed, model_arguments)
    blocks_of = {name: {key: tensor for key, tensor in load_state_dict(os.path.join(ROOT, path)).items() if key.startswith("blocks.")}
                 for name, path in CHECKPOINTS.items()}

    values = {name: [] for name, *_ in SERIES}               # series -> one {quantity: [per block]} per seed
    for seed in range(arguments.seeds):
        torch.manual_seed(seed)
        model = utils.build_model(model_arguments).to(device).eval()
        for name, *_ in SERIES:                              # "random" first: the procedural tensors overwrite all 12 blocks afterwards
            if name != "random":
                missing, unexpected = model.load_state_dict(blocks_of[name], strict=False)
                assert not unexpected and not any(key.startswith("blocks.") for key in missing), (name, missing, unexpected)
            statistics = BlockStatistics(model)
            for start in range(0, len(images), arguments.batch_size):
                model(images[start:start + arguments.batch_size].to(device))
            values[name].append({**statistics.result(), "scale": effective_scales(model)})
            print(f"seed {seed} {name:7s} entropy " + " ".join(f"{v:5.2f}" for v in values[name][-1]["entropy"]), flush=True)
    return {"values": values, "blocks": len(model.blocks), "tokens": statistics.sequence_length,
            "seeds": arguments.seeds, "n_images": len(images), "image_seed": arguments.image_seed, "data_path": arguments.data_path,
            "images_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(), "first_images": paths[:5],
            "checkpoints": CHECKPOINTS, "scale_convention": f"rms(W diag(gain)) / {INIT_STD}, gain_fold {GAIN_FOLD}",
            "device": device, "date": time.strftime("%Y-%m-%d %H:%M")}


# ------------------------------------------------------------------------------------------------------------------- figure
def per_seed(record, name, key, shown):
    """(seeds, shown blocks) array of one quantity; `key` may be "scale/q"."""
    rows = []
    for one in record["values"][name]:
        value = one
        for part in key.split("/"):
            value = value[part]
        rows.append(value)
    return np.array(rows, float)[:, shown]


def draw(arguments, record, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    import seaborn as sns
    from tueplots import bundles, figsizes

    usetex = arguments.usetex and shutil.which("latex") is not None
    if arguments.usetex and not usetex:
        print("no latex on this node: rendering the text with matplotlib's own fonts")
    bundle, figsize = getattr(bundles, arguments.venue), getattr(figsizes, arguments.venue)

    narrow = {"panel": False}                                # set by style(): a panel below ~0.45 line widths gets wrapped labels, fewer ticks

    def style(rel_width, nrows, ncols, height_to_width_ratio=None):
        """The notebooks' pattern: the bundle sets fonts, sizes and the golden-ratio figure height, spines off. A height ratio is
        given only where the bundle's height is too small for the labels (several narrow panels side by side)."""
        plt.rcParams.update(bundle(usetex=usetex, rel_width=rel_width, nrows=nrows, ncols=ncols, family="serif"))
        if height_to_width_ratio is not None:
            plt.rcParams.update(figsize(rel_width=rel_width, nrows=nrows, ncols=ncols, height_to_width_ratio=height_to_width_ratio))
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
        narrow["panel"] = rel_width / ncols < 0.45

    tab10 = sns.color_palette("tab10")
    colour = {name: tab10[slot] for name, _, slot in SERIES}
    first, last = (int(v) for v in arguments.blocks.split("-"))
    shown = np.arange(first, last + 1)
    PANELS["preact"] = {"mean": ("preact_mean", "Mean fc1 pre-activation"), "norm": ("preact_norm", "fc1 pre-activation norm")}[arguments.preact]
    panels = [(tag, *PANELS[tag]) for tag in arguments.panels.split(",")]
    uniform = dict(color="black", linestyle="--", linewidth=1)                                  # uniform attention, ln(tokens)

    def line(ax, name, key, linestyle="-", band=True):
        data = per_seed(record, name, key, shown)
        mean, std = data.mean(0), (data.std(0, ddof=1) if len(data) > 1 else np.zeros(len(shown)))
        if band and std.max() > 0:
            ax.fill_between(shown, mean - std, mean + std, color=colour[name], alpha=0.2, linewidth=0)
        ax.plot(shown, mean, marker="o", markersize=1.5, linewidth=1, linestyle=linestyle, color=colour[name], zorder=2)

    def panel(ax, tag, key, label, x_label=True, label_as_title=False):
        if tag == "scale":
            for name, *_ in SERIES:
                for weight, linestyle, _ in SCALED:
                    line(ax, name, f"scale/{weight}", linestyle)
        else:
            for name, *_ in SERIES:
                line(ax, name, key)
        if tag == "entropy":                                                                    # the reference belongs to this panel only
            ax.axhline(math.log(record["tokens"]), zorder=3, **uniform)
            ax.set_ylim(bottom=0)
            ax.legend(handles=[Line2D([], [], label="uniform attention", **uniform)], loc="upper right", bbox_to_anchor=(1.0, 0.9),
                      frameon=False, handlelength=1.8)
        if tag == "scale":                                                                      # the line styles belong to this panel only
            low, high = ax.get_ylim()
            ax.set_ylim(top=high + 0.3 * (high - low))
            ax.legend(handles=[Line2D([], [], color="black", linewidth=1, linestyle=linestyle, label=label) for _, linestyle, label in SCALED],
                      loc="upper left", ncol=len(SCALED), frameon=False, handlelength=1.3, handletextpad=0.5, columnspacing=0.8)
        if tag == "preact" and arguments.preact == "norm":
            ax.set_ylim(bottom=0)
        if tag == "active":                                                                     # spans 4 decades on kdyck
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(numticks=6)); ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(top=1.0)
        if label_as_title:                                                                      # stacked panels: no room for a vertical label
            ax.set_title(label, loc="left", fontsize=plt.rcParams["axes.labelsize"])
        else:                                                                                   # a narrow panel is short
            ax.set_ylabel(textwrap.fill(label, 18, break_on_hyphens=False) if narrow["panel"] and len(label) > 20 else label)
        ax.set_xlabel("Layer" if x_label else "")
        ax.set_xlim(first - 0.5, last + 0.5)
        ax.set_xticks(shown[::2] if narrow["panel"] and len(shown) > 8 else shown)
        ax.grid(True, alpha=0.3)

    handles = [Line2D([], [], color=colour[name], marker="o", markersize=1.5, linewidth=1, label=label) for name, label, _ in SERIES]

    def save(fig, name):
        for extension, dpi in (("pdf", None), ("png", 1500)):
            fig.savefig(os.path.join(out_dir, f"{name}.{extension}"), bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"wrote {os.path.relpath(out_dir, ROOT)}/{name}.pdf ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in before the legend)")

    # row: full line width, one panel per quantity
    style(1.0, 1, len(panels), 0.85)
    fig, axes = plt.subplots(1, len(panels))
    for ax, (tag, key, label) in zip(np.atleast_1d(axes), panels):
        panel(ax, tag, key, label)
    fig.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 4), frameon=False)
    save(fig, "early_row")

    # column: the panels stacked and sharing the layer axis
    style(arguments.panel_width, len(panels), 1)
    fig, axes = plt.subplots(len(panels), 1, sharex=True)
    for position, (ax, (tag, key, label)) in enumerate(zip(np.atleast_1d(axes), panels)):
        panel(ax, tag, key, label, x_label=position == len(panels) - 1, label_as_title=True)
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save(fig, "early_column")

    # separate panels, and the legend alone as a strip
    style(arguments.panel_width, 1, 1)
    for tag, key, label in panels:
        fig, ax = plt.subplots()
        panel(ax, tag, key, label)
        save(fig, f"early_{tag}")
    style(1.0, 1, 1)
    fig = plt.figure(figsize=(plt.rcParams["figure.figsize"][0], 0.2))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    save(fig, "early_legend")


def table(record, arguments):
    print(f"\nmean over {record['seeds']} contexts, {record['n_images']} training images")
    keys = [("entropy", "{:6.3f}"), ({"mean": "preact_mean", "norm": "preact_norm"}[arguments.preact], "{:+7.3f}"),
            ("active_units", "{:8.5f}")] + [(f"scale/{weight}", "{:6.3f}") for weight, *_ in SCALED]
    for key, form in keys:
        print(f"  {key}")
        for name, *_ in SERIES:
            print(f"    {name:7s}" + " ".join(form.format(v) for v in per_seed(record, name, key, np.arange(record["blocks"])).mean(0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet", help="ImageNet root with a train/ folder")
    parser.add_argument("--n_images", type=int, default=1000, help="random training images")
    parser.add_argument("--image_seed", type=int, default=0, help="seed of the image draw")
    parser.add_argument("--seeds", type=int, default=5, help="random contexts (model seeds 0..n-1) each series is measured in")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out", default=os.path.join(ROOT, "plots/out/fig_sink_gate"), help="output directory")
    parser.add_argument("--replot", action="store_true", help="redraw from <out>/statistics.json without measuring")
    parser.add_argument("--panels", default="entropy,preact,scale", help="panels in order, from entropy, preact, scale, active")
    parser.add_argument("--preact", choices=("mean", "norm"), default="mean", help="preact panel: mean pre-activation or mean token L2 norm")
    parser.add_argument("--blocks", default="0-11", help="blocks shown, e.g. 0-7")
    parser.add_argument("--panel_width", type=float, default=0.5, help="width of the separate panels and the column, in line widths")
    parser.add_argument("--venue", default="iclr2024", help="tueplots bundle")
    parser.add_argument("--no_usetex", dest="usetex", action="store_false", help="do not render the text with LaTeX (the notebooks do)")
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)
    record_path = os.path.join(arguments.out, "statistics.json")
    if arguments.replot:
        record = json.load(open(record_path))
    else:
        record = measure(arguments)
        json.dump(record, open(record_path, "w"), indent=1)
        print(f"wrote {os.path.relpath(record_path, ROOT)}")
    table(record, arguments)
    draw(arguments, record, arguments.out)
