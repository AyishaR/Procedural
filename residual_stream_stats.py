"""Residual-stream activation statistics: procedural-trained vs random-init ViT-B.

Runs a procedurally-pretrained ViT-B (loaded the *usual ImageNet-training* way via
``utils.pr_load_model`` -> blocks + final norm from the checkpoint, image stem/head freshly
random) and a fully random-init version of the same architecture over a random subset of
ImageNet-1k val, and computes streaming statistics of the residual stream per layer:

  1. residual-stream magnitude after each addition (attn add, mlp add)
  2. magnitude each block contributes (attention output, mlp output)
  3. the per-token channel mean/std that each LayerNorm sees + post-LN magnitude
  4. how much each block (and each attn/mlp sub-block) scales its input to its output

Everything is accumulated with running (Welford) moments so no per-token tensors are kept
across batches. Outputs a JSON of {mean, std} per (model, token-group, metric, layer), a set of
comparison plots, and a console summary table.

Example
-------
    python residual_stream_stats.py                       # full 10k run (GPU node)
    python residual_stream_stats.py --num-images 64 --batch-size 16 --device cpu   # smoke test
"""
import argparse
import copy
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import models.vision_transformer  # noqa: F401  (registers the custom 'vit_base')
import utils
from datasets import build_dataset
from main import get_args_parser


# --------------------------------------------------------------------------------------------
# Running (Welford) moments -- accumulate mean/std over a stream of 1-D value tensors.
# --------------------------------------------------------------------------------------------
class RunningMoments:
    """Numerically-stable streaming mean/variance over a flat stream of scalars."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, values: torch.Tensor):
        """values: 1-D tensor of samples to fold into the running estimate."""
        v = values.detach().reshape(-1).double()
        n = v.numel()
        if n == 0:
            return
        batch_mean = v.mean().item()
        batch_var = v.var(unbiased=False).item() if n > 1 else 0.0
        # Chan et al. parallel variance combination.
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.mean += delta * n / new_count
        self.M2 += batch_var * n + delta * delta * self.count * n / new_count
        self.count = new_count

    @property
    def std(self):
        if self.count < 2:
            return 0.0
        return math.sqrt(self.M2 / self.count)

    def as_dict(self):
        return {"mean": self.mean, "std": self.std, "count": self.count}


# --------------------------------------------------------------------------------------------
# Hook-based collector. Mirrors utils.HookCollectorTrain but hooks sub-modules (norm1/attn/
# norm2/mlp/block) to avoid recomputing attention, and folds per-token quantities straight into
# RunningMoments accumulators. One collector instance == one model.
# --------------------------------------------------------------------------------------------
TOKEN_GROUPS = ("all", "cls", "patch")

# metric -> which residual point it lives on (documented for the reader)
METRICS = [
    # residual-stream magnitude after addition
    "resid_after_attn_norm", "resid_after_mlp_norm",
    "resid_after_attn_rms", "resid_after_mlp_rms",
    # block contribution magnitude
    "attn_contrib_norm", "mlp_contrib_norm",
    # layernorm input channel-mean / channel-std + post-LN magnitude
    "ln1_in_mean", "ln1_in_std", "ln2_in_mean", "ln2_in_std",
    "ln1_out_norm", "ln2_out_norm",
    # block scaling (per-token ratios)
    "block_scale", "attn_subblock_scale", "mlp_subblock_scale",
    "attn_rel_contrib", "mlp_rel_contrib",
    # repo-native attention input->output ratio + its multiplicative decomposition
    # (mirrors engine.py:1273-1291 / custom_utils.py:64-77). norm_ratio == attn_subblock_scale.
    "norm_ratio", "norm_ratio_ln1_inp", "norm_ratio_qkvp1_ln1",
    "norm_ratio_ls1_qkvp1", "norm_ratio_out_ls1", "norm_ratio_resout_out",
    "cosine_rin_rout",
]
# metrics that are not per-block but per residual "position" (embedding, final norm)
EXTRA_METRICS = ["embed_norm", "final_norm_norm"]


def _token_slices(n_tokens, num_prefix):
    """Return {group: slice} over the token dim. cls = prefix tokens, patch = the rest."""
    return {
        "all": slice(0, n_tokens),
        "cls": slice(0, num_prefix),
        "patch": slice(num_prefix, n_tokens),
    }


class ResidualStatsCollector:
    def __init__(self, model, num_prefix_tokens=1):
        try:
            self.mwd = model.module
        except AttributeError:
            self.mwd = model
        self.num_prefix = num_prefix_tokens
        self.handles = []
        # acc[layer][metric][group] -> RunningMoments ; layer=-1 used for extra positions
        self.acc = {}
        self._scratch = {}  # per-forward temporaries keyed by layer

    def _get(self, layer, metric, group):
        return (
            self.acc.setdefault(layer, {})
            .setdefault(metric, {g: RunningMoments() for g in TOKEN_GROUPS})[group]
        )

    def _fold(self, layer, metric, per_token):
        """per_token: [B, N] tensor -> fold into the three token groups."""
        n = per_token.shape[1]
        for g, sl in _token_slices(n, self.num_prefix).items():
            self._get(layer, metric, g).update(per_token[:, sl])

    @staticmethod
    def _tok_norm(x):
        return x.norm(dim=-1)  # [B, N]

    def __enter__(self):
        d = self.mwd.embed_dim if hasattr(self.mwd, "embed_dim") else None

        def make_hooks(idx):
            def norm1_hook(mod, inp, out):
                r_in = inp[0]
                self._scratch[idx] = {"r_in": r_in, "ln1_out": out}
                # LN sees channel mean/std of its input (the stats it normalizes away).
                self._fold(idx, "ln1_in_mean", r_in.mean(dim=-1))
                self._fold(idx, "ln1_in_std", r_in.std(dim=-1, unbiased=False))
                self._fold(idx, "ln1_out_norm", self._tok_norm(out))

            def attn_hook(mod, inp, out):
                self._scratch[idx]["a_out"] = out  # qkv+proj output (pre-LayerScale)

            def ls1_hook(mod, inp, out):
                self._scratch[idx]["ls1_out"] = out

            def dp1_hook(mod, inp, out):
                self._scratch[idx]["attn_added"] = out  # actual term added to the residual

            def norm2_hook(mod, inp, out):
                r_mid = inp[0]
                self._scratch[idx]["r_mid"] = r_mid
                self._fold(idx, "ln2_in_mean", r_mid.mean(dim=-1))
                self._fold(idx, "ln2_in_std", r_mid.std(dim=-1, unbiased=False))
                self._fold(idx, "ln2_out_norm", self._tok_norm(out))

            def mlp_hook(mod, inp, out):
                self._scratch[idx]["m_out"] = out

            def block_hook(mod, inp, out):
                s = self._scratch.pop(idx)
                r_in, a_out, r_mid, m_out, r_out = (
                    s["r_in"], s["a_out"], s["r_mid"], s["m_out"], out,
                )
                dim = r_in.shape[-1]
                sqrt_d = math.sqrt(dim)

                n_rin = self._tok_norm(r_in)
                n_rmid = self._tok_norm(r_mid)      # residual after attention add
                n_rout = self._tok_norm(r_out)      # residual after mlp add (= block out)
                n_attn = self._tok_norm(a_out)      # attention contribution
                n_mlp = self._tok_norm(m_out)       # mlp contribution
                eps = 1e-12

                # 1. residual magnitude after each addition (L2 + RMS)
                self._fold(idx, "resid_after_attn_norm", n_rmid)
                self._fold(idx, "resid_after_mlp_norm", n_rout)
                self._fold(idx, "resid_after_attn_rms", n_rmid / sqrt_d)
                self._fold(idx, "resid_after_mlp_rms", n_rout / sqrt_d)
                # 2. block contribution magnitude
                self._fold(idx, "attn_contrib_norm", n_attn)
                self._fold(idx, "mlp_contrib_norm", n_mlp)
                # 4. scaling ratios (per token)
                self._fold(idx, "block_scale", n_rout / (n_rin + eps))
                self._fold(idx, "attn_subblock_scale", n_rmid / (n_rin + eps))
                self._fold(idx, "mlp_subblock_scale", n_rout / (n_rmid + eps))
                self._fold(idx, "attn_rel_contrib", n_attn / (n_rin + eps))
                self._fold(idx, "mlp_rel_contrib", n_mlp / (n_rmid + eps))

                # --- repo-native attention input->output ratio + decomposition ---
                # (engine.py:1273-1291 / custom_utils.py:64-77). rout == r_mid (attn residual).
                n_ln1 = self._tok_norm(s["ln1_out"])          # ||norm1(r_in)||
                n_qkvp1 = n_attn                               # attn qkv+proj output
                n_ls1 = self._tok_norm(s["ls1_out"])          # after LayerScale (Identity here)
                n_add = self._tok_norm(s["attn_added"])       # actual term added to residual
                self._fold(idx, "norm_ratio", n_rmid / (n_rin + eps))
                self._fold(idx, "norm_ratio_ln1_inp", n_ln1 / (n_rin + eps))
                self._fold(idx, "norm_ratio_qkvp1_ln1", n_qkvp1 / (n_ln1 + eps))
                self._fold(idx, "norm_ratio_ls1_qkvp1", n_ls1 / (n_qkvp1 + eps))
                self._fold(idx, "norm_ratio_out_ls1", n_add / (n_ls1 + eps))
                self._fold(idx, "norm_ratio_resout_out", n_rmid / (n_add + eps))
                self._fold(idx, "cosine_rin_rout", F.cosine_similarity(r_in, r_mid, dim=-1))

                # embedding output == input to block 0 (residual stream start)
                if idx == 0:
                    self._fold(-1, "embed_norm", n_rin)

            return norm1_hook, attn_hook, ls1_hook, dp1_hook, norm2_hook, mlp_hook, block_hook

        for i, block in enumerate(self.mwd.blocks):
            n1, ah, l1, d1, n2, mh, bh = make_hooks(i)
            self.handles += [
                block.norm1.register_forward_hook(n1),
                block.attn.register_forward_hook(ah),
                block.ls1.register_forward_hook(l1),
                block.drop_path1.register_forward_hook(d1),
                block.norm2.register_forward_hook(n2),
                block.mlp.register_forward_hook(mh),
                block.register_forward_hook(bh),
            ]

        # final LayerNorm output magnitude (post the model's trunk norm)
        def final_norm_hook(mod, inp, out):
            self._fold(-1, "final_norm_norm", self._tok_norm(out))

        self.handles.append(self.mwd.norm.register_forward_hook(final_norm_hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def to_dict(self):
        out = {}
        for layer, metrics in self.acc.items():
            for metric, groups in metrics.items():
                for g, rm in groups.items():
                    out.setdefault(g, {}).setdefault(metric, {})[str(layer)] = rm.as_dict()
        return out


# --------------------------------------------------------------------------------------------
# Model construction (mirrors usual ImageNet training).
# --------------------------------------------------------------------------------------------
def make_args(cli):
    args = get_args_parser().parse_args([])
    args.model = cli.model
    args.nb_classes = 1000
    args.drop_path = 0.0
    args.data_set = "IMNET"
    args.data_path = cli.data_path
    args.input_size = cli.input_size
    args.initialize = cli.ckpt
    args.distributed = False
    args.seed = cli.seed
    # str -> list/dict args, post-processed exactly as main.py does before pr_load_model
    args.skip_load_blocks = []
    args.random_blocks = []
    args.delete_blocks = []
    args.skip_load_block_attributes = []
    args.skip_attn_segments = {}
    args.hold_back_blocks = []
    args.custom_pr_load = ""
    return args


def build_models(args, device, independent_random):
    """Return (trained_mwd, random_mwd), both eval, fused_attn off."""
    # 1) fully-random baseline (seed fixed inside pr_load_model via random.seed(args.seed))
    _, random_mwd, _ = utils.pr_load_model(path="", args=args, device=device)

    # 2) trained model: same random stem, blocks + final norm overwritten from the checkpoint.
    #    deepcopy the random model first so both share the identical random image stem.
    trained_seed_model = None if independent_random else copy.deepcopy(random_mwd)
    _, trained_mwd, _ = utils.pr_load_model(
        path=args.initialize, args=args, device=device, model=trained_seed_model,
    )

    for m in (trained_mwd, random_mwd):
        for block in m.blocks:
            block.attn.fused_attn = False
        m.to(device).eval()
    return trained_mwd, random_mwd


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def build_loader(args, cli):
    val_ds, _ = build_dataset(is_train=False, args=args)
    g = torch.Generator().manual_seed(cli.seed)
    n = min(cli.num_images, len(val_ds))
    idx = torch.randperm(len(val_ds), generator=g)[:n].tolist()
    subset = Subset(val_ds, idx)
    loader = DataLoader(
        subset, batch_size=cli.batch_size, shuffle=False,
        num_workers=cli.num_workers, pin_memory=True, drop_last=False,
    )
    return loader, n


@torch.no_grad()
def run_model(model, loader, device, num_prefix, tag):
    collector = ResidualStatsCollector(model, num_prefix_tokens=num_prefix)
    seen = 0
    with collector:
        for bi, batch in enumerate(loader):
            images = batch[0].to(device, non_blocking=True)
            model(images)
            seen += images.shape[0]
            if bi % 10 == 0:
                print(f"  [{tag}] {seen} images", flush=True)
    print(f"  [{tag}] done: {seen} images", flush=True)
    return collector.to_dict()


# --------------------------------------------------------------------------------------------
# Plotting + reporting
# --------------------------------------------------------------------------------------------
def _series(stats, group, metric, n_layers):
    """Return (means, stds) arrays over layers 0..n_layers-1 for a per-block metric."""
    d = stats.get(group, {}).get(metric, {})
    means = [d.get(str(l), {}).get("mean", float("nan")) for l in range(n_layers)]
    stds = [d.get(str(l), {}).get("std", float("nan")) for l in range(n_layers)]
    return means, stds


def _plot_family(trained, random, group, metrics, labels, title, ylabel, path, n_layers,
                 logy=False):
    xs = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, (metric, lab) in enumerate(zip(metrics, labels)):
        for stats, name, ls in ((trained, "trained", "-"), (random, "random", "--")):
            m, s = _series(stats, group, metric, n_layers)
            m = torch.tensor(m); s = torch.tensor(s)
            lo, hi = m - s, m + s
            if logy:  # clamp lower band so log-scale fill stays valid
                lo = torch.clamp(lo, min=1e-6)
            color = cmap(i % 10)
            ax.plot(xs, m, ls, color=color, marker="o", ms=3, label=f"{lab} ({name})")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("block index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def make_plots(trained, random, out_dir, n_layers, group="all", suffix=""):
    fam = [
        (["resid_after_attn_norm", "resid_after_mlp_norm"],
         ["after attn-add", "after mlp-add"],
         "Residual-stream L2 norm (after addition)", "||x||_2", "resid_norm.png", True),
        (["attn_contrib_norm", "mlp_contrib_norm"],
         ["attn contribution", "mlp contribution"],
         "Block contribution magnitude", "||contrib||_2", "block_contrib.png", True),
        (["ln1_in_std", "ln2_in_std"],
         ["norm1 input std", "norm2 input std"],
         "LayerNorm input channel-std", "std over channels", "ln_std.png", False),
        (["ln1_in_mean", "ln2_in_mean"],
         ["norm1 input mean", "norm2 input mean"],
         "LayerNorm input channel-mean", "mean over channels", "ln_mean.png", False),
        (["ln1_out_norm", "ln2_out_norm"],
         ["post-norm1", "post-norm2"],
         "Post-LayerNorm magnitude", "||LN(x)||_2", "postln_norm.png", False),
        (["block_scale", "attn_subblock_scale", "mlp_subblock_scale"],
         ["full block", "attn sub-block", "mlp sub-block"],
         "Block input->output scaling (per-token ratio)", "||out|| / ||in||",
         "block_scaling.png", True),
        # repo-native attention input->output ratio (norm_ratio) + its decomposition
        (["norm_ratio", "norm_ratio_ln1_inp", "norm_ratio_qkvp1_ln1",
          "norm_ratio_ls1_qkvp1", "norm_ratio_out_ls1", "norm_ratio_resout_out"],
         ["norm_ratio (rout/rin)", "ln1/rin", "qkvp1/ln1",
          "ls1/qkvp1", "add/ls1", "rout/add"],
         "Repo attention norm_ratio & decomposition", "per-token ratio",
         "norm_ratio.png", True),
        (["cosine_rin_rout"], ["cos(rin, rout)"],
         "Repo cosine(rin, rout)  [attention residual]", "cosine similarity",
         "cosine_rin_rout.png", False),
    ]
    for metrics, labels, title, ylabel, fname, logy in fam:
        stem, ext = os.path.splitext(fname)
        out = os.path.join(out_dir, f"{stem}{suffix}{ext}")
        gtitle = title if group == "all" else f"{title}  [{group} token]"
        _plot_family(trained, random, group, metrics, labels, gtitle, ylabel,
                     out, n_layers, logy=logy)


def plot_ln_weights(trained_mwd, random_mwd, out_dir, model_name):
    """Plot per-layer LayerNorm learned weight (gamma) and bias (beta): proc-init vs random.

    Parameter-only (no data). Random init is torch default: gamma=1, beta=0 everywhere.
    """
    n = len(trained_mwd.blocks)
    xs = list(range(n))

    def stat(mwd, sub, attr):
        # per-layer (mean, std) across channels of block.<sub>.<attr>
        means, stds = [], []
        for b in mwd.blocks:
            v = getattr(getattr(b, sub), attr).detach().float()
            means.append(v.mean().item())
            stds.append(v.std(unbiased=False).item())
        return torch.tensor(means), torch.tensor(stds)

    for attr, gname, fname in (("weight", "gamma (LN weight)", "ln_learned_weight"),
                               ("bias", "beta (LN bias)", "ln_learned_bias")):
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("tab10")
        for i, sub in enumerate(("norm1", "norm2")):
            for mwd, name, ls in ((trained_mwd, "proc", "-"), (random_mwd, "random", "--")):
                m, s = stat(mwd, sub, attr)
                c = cmap(i)
                ax.plot(xs, m, ls, color=c, marker="o", ms=3, label=f"{sub} ({name})")
                ax.fill_between(xs, m - s, m + s, color=c, alpha=0.12)
        # final trunk norm as an extra point at x = n
        for mwd, name, mk in ((trained_mwd, "proc", "*"), (random_mwd, "random", "x")):
            v = getattr(mwd.norm, attr).detach().float()
            ax.errorbar([n], [v.mean().item()], yerr=[v.std(unbiased=False).item()],
                        fmt=mk, color="black", ms=9, capsize=4, label=f"final norm ({name})")
        ax.set_xlabel("block index  (x=%d is final trunk norm)" % n)
        ax.set_ylabel(gname + "  (mean +/- std over channels)")
        ax.set_title(f"{model_name}: learned LayerNorm {gname} — proc-init vs random")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{fname}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  wrote {path}")

    # quick console summary: how far proc gammas drift from the random default (1.0)
    print(f"\n{model_name} LN weight (gamma) drift from random default (1.0), mean|std|maxabs:")
    print(f"{'blk':>3} | {'norm1 mean':>11} {'norm1 std':>10} {'n1 max':>8} | "
          f"{'norm2 mean':>11} {'norm2 std':>10} {'n2 max':>8}")
    for l in range(n):
        w1 = trained_mwd.blocks[l].norm1.weight.detach().float()
        w2 = trained_mwd.blocks[l].norm2.weight.detach().float()
        print(f"{l:>3} | {w1.mean():>11.4f} {w1.std(unbiased=False):>10.4f} {w1.abs().max():>8.3f} | "
              f"{w2.mean():>11.4f} {w2.std(unbiased=False):>10.4f} {w2.abs().max():>8.3f}")
    wf = trained_mwd.norm.weight.detach().float()
    print(f"final norm: mean={wf.mean():.4f} std={wf.std(unbiased=False):.4f} maxabs={wf.abs().max():.3f}")


def print_table(trained, random, n_layers, group="all"):
    cols = [
        ("resid(mlp)", "resid_after_mlp_norm"),
        ("attn", "attn_contrib_norm"),
        ("mlp", "mlp_contrib_norm"),
        ("blk_scale", "block_scale"),
    ]
    header = f"{'blk':>3} | " + " | ".join(
        f"{name+' T':>11} {name+' R':>11}" for name, _ in cols
    )
    print("\nPer-layer summary (token group = %s), T=trained R=random, mean values:" % group)
    print(header)
    print("-" * len(header))
    for l in range(n_layers):
        row = [f"{l:>3}"]
        for _, metric in cols:
            t = trained.get(group, {}).get(metric, {}).get(str(l), {}).get("mean", float("nan"))
            r = random.get(group, {}).get(metric, {}).get(str(l), {}).get("mean", float("nan"))
            row.append(f"{t:>11.3f} {r:>11.3f}")
        print(" | ".join(row))
    # extra residual positions
    for name, metric in (("embed", "embed_norm"), ("final_norm", "final_norm_norm")):
        t = trained.get(group, {}).get(metric, {}).get("-1", {}).get("mean", float("nan"))
        r = random.get(group, {}).get(metric, {}).get("-1", {}).get("mean", float("nan"))
        print(f"{name:>12}: trained={t:.3f}  random={r:.3f}")

    # repo-native attention input->output ratio (rout/rin) + cosine
    print("\nRepo norm_ratio = ||r_in + attn||/||r_in|| (== attn_subblock_scale), token=%s:" % group)
    print(f"{'blk':>3} | {'norm_ratio T':>13} {'norm_ratio R':>13} | {'cos(rin,rout) T':>16} {'R':>8}")
    for l in range(n_layers):
        nt = trained.get(group, {}).get("norm_ratio", {}).get(str(l), {}).get("mean", float("nan"))
        nr = random.get(group, {}).get("norm_ratio", {}).get(str(l), {}).get("mean", float("nan"))
        ct = trained.get(group, {}).get("cosine_rin_rout", {}).get(str(l), {}).get("mean", float("nan"))
        cr = random.get(group, {}).get("cosine_rin_rout", {}).get(str(l), {}).get("mean", float("nan"))
        print(f"{l:>3} | {nt:>13.4f} {nr:>13.4f} | {ct:>16.4f} {cr:>8.4f}")


# --------------------------------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="results/pr_vitb/pr_27267764_final.pth")
    p.add_argument("--model", default="vit_base",
                   help="repo model name for build_model (e.g. vit_base, vit_small)")
    p.add_argument("--data-path", default="/data/datasets/ILSVRC2012")
    p.add_argument("--num-images", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--output-dir", default="results/residual_stats")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--independent-random", action="store_true",
                   help="draw an independent random model instead of sharing the random stem")
    p.add_argument("--group", default="all", choices=list(TOKEN_GROUPS),
                   help="token group used for plots + console table")
    p.add_argument("--from-json", default=None,
                   help="skip the GPU run: load an existing residual_stats.json and just "
                        "(re)generate plots + table for --group")
    p.add_argument("--ln-weights-only", action="store_true",
                   help="parameter-only mode (no data/GPU): plot learned LayerNorm "
                        "weight/bias for proc-init vs random and exit")
    return p.parse_args()


def _emit(trained_stats, random_stats, cli, n_layers, group):
    """Write plots for `group` (filenames suffixed unless group=='all') and print its table."""
    suffix = "" if group == "all" else f"_{group}"
    make_plots(trained_stats, random_stats, cli.output_dir, n_layers, group=group, suffix=suffix)
    print_table(trained_stats, random_stats, n_layers, group=group)


def main():
    cli = parse_cli()

    # --- re-plot only mode: no model / no data / no GPU ---
    if cli.from_json:
        with open(cli.from_json) as f:
            blob = json.load(f)
        n_layers = blob.get("meta", {}).get("n_layers", 12)
        os.makedirs(cli.output_dir, exist_ok=True)
        print(f"Re-plotting from {cli.from_json} for token group '{cli.group}'")
        _emit(blob["trained"], blob["random"], cli, n_layers, cli.group)
        return

    # --- LayerNorm learned-weight mode: parameters only, no data / no GPU ---
    if cli.ln_weights_only:
        os.makedirs(cli.output_dir, exist_ok=True)
        args = make_args(cli)
        print(f"Building models ({cli.model}) on CPU for LN-weight comparison...")
        trained_mwd, random_mwd = build_models(args, torch.device("cpu"),
                                               cli.independent_random)
        plot_ln_weights(trained_mwd, random_mwd, cli.output_dir, cli.model)
        return

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    os.makedirs(cli.output_dir, exist_ok=True)
    print(f"Device: {device}")

    args = make_args(cli)

    print("Building models (usual ImageNet-training loading path)...")
    trained_mwd, random_mwd = build_models(args, device, cli.independent_random)
    num_prefix = getattr(trained_mwd, "num_prefix_tokens", 1)
    n_layers = len(trained_mwd.blocks)
    print(f"num_prefix_tokens={num_prefix}, blocks={n_layers}")

    print("Building 10k ImageNet-val subset...")
    loader, n_used = build_loader(args, cli)
    print(f"Using {n_used} images, batch_size={cli.batch_size}")

    print("Running trained model...")
    trained_stats = run_model(trained_mwd, loader, device, num_prefix, "trained")
    print("Running random-init model...")
    random_stats = run_model(random_mwd, loader, device, num_prefix, "random")

    out = {
        "meta": {
            "ckpt": cli.ckpt, "num_images": n_used, "seed": cli.seed,
            "device": str(device), "input_size": cli.input_size,
            "independent_random": cli.independent_random,
            "num_prefix_tokens": num_prefix, "n_layers": n_layers,
            "token_groups": list(TOKEN_GROUPS),
        },
        "trained": trained_stats,
        "random": random_stats,
    }
    json_path = os.path.join(cli.output_dir, "residual_stats.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {json_path}")

    _emit(trained_stats, random_stats, cli, n_layers, cli.group)


if __name__ == "__main__":
    main()
