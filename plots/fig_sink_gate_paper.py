"""Paper figure: attention sink and MLP gate of the procedurally pretrained blocks, against a random initialisation.

Three per-block quantities of a ViT-B/16 BEFORE any ImageNet training, measured on random ImageNet training images:

  attention entropy       mean entropy (nats) of the attention rows, over images, heads and query tokens; ln(197) = 5.28
                          is uniform attention, a sink drives it towards 0
  fc1 pre-activation      mean of the MLP's fc1 output before the GELU, over images, tokens and hidden units (--preact mean,
                          the statistic the recipe's gate is calibrated to), or the mean L2 norm of a token's pre-activation
                          vector (--preact norm)
  active units            fraction of positive fc1 pre-activations (the GELU passes those; the rest is switched off)

Series: random init (timm), kdyck, ksd. A procedural series is a fresh timm ViT-B whose 12 blocks hold the checkpoint's
tensors; the ImageNet patch / position embeddings and the class token stay random, as in every run that loads a procedural
checkpoint (utils.pr_load_model drops them). The statistics of block b depend on blocks 0..b only, so blocks 0-7 are also
exactly the blocks-0-7 prefix arm (ftb4i). Every series is measured in --seeds random contexts (seed s: torch.manual_seed(s),
then the model is built; the three series of one seed share the random embeddings) on the same --n_images training images
(seeded uniform draw, evaluation transform). Lines are the mean over the contexts, bands the range (min to max).

Everything is measured here with hooks on the model's own forward pass; nothing is read from a cache or a profile JSON. The
numbers are written next to the figures (statistics.json), and --replot redraws from that file without a GPU.

usage (on a GPU node, ~3 min):   .venv/bin/python plots/fig_sink_gate_paper.py
       restyle only:             .venv/bin/python plots/fig_sink_gate_paper.py --replot [--preact norm] [--blocks 0-7] [--usetex]
output: plots/out/fig_sink_gate/{row,column}.pdf, panel_{entropy,preact,active}.pdf + legend.pdf (and .png previews)
  row     one figure at full line width, three panels of 1/3 line width each
  column  one figure of 1/3 line width, three stacked panels sharing the block axis
  panel_* the three panels as separate files of 1/3 line width each (for subfigures), legend.pdf the matching legend strip"""
import argparse, contextlib, hashlib, io, json, math, os, sys, textwrap, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKPOINTS = {"kdyck": "results/pr_vitb_n/pr_6066174_final.pth", "ksd": "results/pr_vitb_ksd/pr_6463456_final.pth"}
# series key, legend label, colour, marker, line style. Colour follows the entity (the neutral gray is the baseline on purpose);
# marker and line style repeat the identity for print and colour-blind readers.
SERIES = [("random", "random init", "#898781", "o", (0, (4, 1.5))),
          ("kdyck", "k-Dyck-D4", "#2a78d6", "s", "-"),
          ("ksd", "k-Dyck-Shuffled-D98", "#eb6834", "^", "-")]
UNIFORM = dict(color="black", lw=0.7, ls=(0, (1, 1.2)))                                              # uniform attention, ln(tokens)


# ---------------------------------------------------------------------------------------------------------------- measurement
class BlockStatistics:
    """Sums of the three quantities per block, collected by hooks while the model runs its own forward pass.

    attention: a pre-hook on block.attn receives norm1(stream), i.e. exactly what the attention reads; the probabilities are
    recomputed from it with the block's own qkv, q/k norm and scale (timm's fused attention does not expose them).
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
            values[name].append(statistics.result())
            print(f"seed {seed} {name:7s} entropy " + " ".join(f"{v:5.2f}" for v in values[name][-1]["entropy"]), flush=True)
    return {"values": values, "blocks": len(model.blocks), "tokens": statistics.sequence_length,
            "seeds": arguments.seeds, "n_images": len(images), "image_seed": arguments.image_seed, "data_path": arguments.data_path,
            "images_sha256": hashlib.sha256("\n".join(paths).encode()).hexdigest(), "first_images": paths[:5],
            "checkpoints": CHECKPOINTS, "device": device, "date": time.strftime("%Y-%m-%d %H:%M")}


# ------------------------------------------------------------------------------------------------------------------- figure
def draw(arguments, record, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
    from tueplots import bundles, figsizes

    bundle, figsize = getattr(bundles, arguments.venue), getattr(figsizes, arguments.venue)
    plt.rcParams.update(bundle(usetex=arguments.usetex))

    first, last = (int(v) for v in arguments.blocks.split("-"))
    shown = np.arange(first, last + 1)
    preact = {"mean": ("preact_mean", "Mean fc1 pre-activation"), "norm": ("preact_norm", "fc1 pre-activation norm")}[arguments.preact]
    panels = [("entropy", "entropy", "Attention entropy"), ("preact", *preact), ("active", "active_units", "Fraction of active fc1 units")]

    def panel(ax, tag, key, label, x_label=True, label_as_title=False):
        for name, _, colour, marker, style in SERIES:
            per_seed = np.array([one[key] for one in record["values"][name]])[:, shown]        # (seeds, blocks)
            ax.fill_between(shown, per_seed.min(0), per_seed.max(0), color=colour, alpha=0.2, lw=0)
            ax.plot(shown, per_seed.mean(0), color=colour, ls=style, lw=1.1, marker=marker, ms=2.6, mew=0, zorder=3)
        if tag == "entropy":                                                                    # uniform attention over all tokens
            uniform = math.log(record["tokens"])
            ax.axhline(uniform, zorder=4, **UNIFORM)                                            # on top: the random series sits on it
            ax.set_ylim(0, uniform * 1.15)
        if tag == "preact" and arguments.preact == "mean":
            ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)
        if tag == "preact" and arguments.preact == "norm":
            ax.set_ylim(bottom=0)
        if tag == "active":                                                                     # spans 4 decades on kdyck
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(numticks=6)); ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(top=1.0)
        if label_as_title:                                                                      # stacked panels: no room for a vertical label
            ax.set_title(label, loc="left", fontsize=plt.rcParams["axes.labelsize"], pad=2.5)
        else:                                                                                   # a 1/3-width panel is ~1.1 in tall
            ax.set_ylabel(textwrap.fill(label, 18, break_on_hyphens=False) if len(label) > 20 else label)
        ax.set_xlabel("Layer" if x_label else "")
        ax.xaxis.set_major_locator(MultipleLocator(2 if len(shown) > 8 else 1)); ax.set_xlim(first - 0.4, last + 0.4)
        ax.grid(which="major")
        ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], color=colour, ls=style, lw=1.1, marker=marker, ms=2.6, mew=0) for _, _, colour, marker, style in SERIES]
    labels = [label for _, label, *_ in SERIES]
    handles.append(Line2D([], [], **UNIFORM)); labels.append("uniform attention")
    legend_inches = 0.17                                                                        # height of one legend line

    def save(fig, name):
        for extension in ("pdf", "png"):
            fig.savefig(os.path.join(out_dir, f"{name}.{extension}"), dpi=400)
        plt.close(fig)
        print(f"wrote {os.path.relpath(out_dir, ROOT)}/{name}.pdf ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in)")

    # row: full line width, three panels
    width, height = figsize(nrows=1, ncols=3, height_to_width_ratio=0.8)["figure.figsize"]
    fig, axes = plt.subplots(1, 3, figsize=(width, height + legend_inches))
    for ax, (tag, key, label) in zip(axes, panels):
        panel(ax, tag, key, label)
    fig.legend(handles, labels, loc="outside upper center", ncols=len(handles), frameon=False, handlelength=2.2, columnspacing=1.2, borderaxespad=0)
    save(fig, "row")

    # column: 1/3 line width, three stacked panels sharing the block axis
    width, height = figsize(rel_width=1 / 3, nrows=3, ncols=1, height_to_width_ratio=0.62)["figure.figsize"]
    fig, axes = plt.subplots(3, 1, figsize=(width, height + 2 * legend_inches), sharex=True)     # legend in two lines
    for position, (ax, (tag, key, label)) in enumerate(zip(axes, panels)):
        panel(ax, tag, key, label, x_label=position == 2, label_as_title=True)
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, handlelength=1.6, columnspacing=0.8,
               handletextpad=0.4, borderaxespad=0)
    save(fig, "column")

    # separate panels of 1/3 line width each, and the legend as its own strip
    width, height = figsize(rel_width=1 / 3, nrows=1, ncols=1, height_to_width_ratio=0.8)["figure.figsize"]
    for tag, key, label in panels:
        fig, ax = plt.subplots(figsize=(width, height))
        panel(ax, tag, key, label)
        save(fig, f"panel_{tag}")
    fig = plt.figure(figsize=(figsize(nrows=1, ncols=1)["figure.figsize"][0], legend_inches))
    fig.legend(handles, labels, loc="center", ncols=len(handles), frameon=False, handlelength=2.2, columnspacing=1.2, borderaxespad=0)
    save(fig, "legend")


def table(record, preact_key):
    print(f"\nmean over {record['seeds']} contexts [min, max], {record['n_images']} training images")
    for key, form in (("entropy", "{:6.3f}"), (preact_key, "{:+7.3f}"), ("active_units", "{:8.5f}")):
        print(f"  {key}")
        for name, *_ in SERIES:
            per_seed = np.array([one[key] for one in record["values"][name]])
            print(f"    {name:7s}" + " ".join(form.format(v) for v in per_seed.mean(0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet", help="ImageNet root with a train/ folder")
    parser.add_argument("--n_images", type=int, default=1000, help="random training images")
    parser.add_argument("--image_seed", type=int, default=0, help="seed of the image draw")
    parser.add_argument("--seeds", type=int, default=5, help="random contexts (model seeds 0..n-1) each series is measured in")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out", default=os.path.join(ROOT, "plots/out/fig_sink_gate"), help="output directory")
    parser.add_argument("--replot", action="store_true", help="redraw from <out>/statistics.json without measuring")
    parser.add_argument("--preact", choices=("mean", "norm"), default="mean", help="middle panel: mean pre-activation or mean token L2 norm")
    parser.add_argument("--blocks", default="0-11", help="blocks shown, e.g. 0-7")
    parser.add_argument("--venue", default="iclr2024", help="tueplots bundle (iclr2024 and neurips2024: 5.5 in line width)")
    parser.add_argument("--usetex", action="store_true", help="render the text with LaTeX (needs a LaTeX installation on the node)")
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)
    record_path = os.path.join(arguments.out, "statistics.json")
    if arguments.replot:
        record = json.load(open(record_path))
    else:
        record = measure(arguments)
        json.dump(record, open(record_path, "w"), indent=1)
        print(f"wrote {os.path.relpath(record_path, ROOT)}")
    table(record, {"mean": "preact_mean", "norm": "preact_norm"}[arguments.preact])
    draw(arguments, record, arguments.out)
