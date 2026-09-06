#!/usr/bin/env python3
"""Structural statistics of each arm's blocks 0-8 AT INIT -- no training, no GPU.

Answers: what does each arm actually carry from the procedural checkpoint, measured on the
weights themselves rather than by ablation?

  2-D weights : stable rank (||s||_2^2 / s_max^2), row-norm variance, top singular value
                -> these capture ARRANGEMENT / low-rank structure
  1-D params  : mean, std, kurtosis of LayerNorm gains and biases
                -> these capture the VALUE DISTRIBUTION of the 1-D tensors

Reuses build_arm() from measure_init_rho_arms.py so the arms are constructed exactly as they
are measured there (and, per that file's docstring, as they are trained).
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_init_rho_arms as M

ARMS = ["r", "ftbqm", "ftbqm1d", "ftbqm1dv", "ftbqmln", "ftbqmbias", "ftb4e3", "ftb3i"]
W2D = ["attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
SLICES = True   # also report qkv rows [0:2e] and [2e:3e] separately
P1D = ["norm1.weight", "norm2.weight", "mlp.fc1.bias", "attn.qkv.bias"]


def stable_rank(M_):
    s = torch.linalg.svdvals(M_.float())
    return float(s.pow(2).sum() / s.pow(2).max())


def main():
    ap = argparse.ArgumentParser(parents=[__import__("main").get_args_parser()],
                                 add_help=False, conflict_handler="resolve")
    ap.add_argument("--out", default="plots/cache/init_structure.json")
    args = ap.parse_args()
    args.nb_classes = 1000
    for k in ["skip_attn_segments", "weight_shuffle", "target_model_weight_shuffle",
              "init_method_copied_blocks", "attention_residual_scaling",
              "attention_out_scaling", "learning_rate_scaling_params"]:
        setattr(args, k, {})
    for k in ["random_blocks", "clip_outlier_blocks", "delete_blocks",
              "init_method_scaled_blocks", "freeze_blocks", "hold_back_blocks"]:
        setattr(args, k, [])
    for k in ["skip_load_blocks", "skip_load_block_attributes", "freeze_block_attributes"]:
        v = getattr(args, k, ""); setattr(args, k, [x for x in v.split(",")] if isinstance(v, str) and v else [])
    args.distributed = False; args.gpu = args.rank = 0; args.world_size = 1

    tgt = M._target_params(args)
    out = {}
    for arm in ARMS:
        torch.manual_seed(0)
        mw = M.build_arm(arm, args, tgt)
        rec = {"w2d": {}, "p1d": {}}
        for nm in W2D:
            sr, rv = [], []
            for b in range(9):
                W = dict(mw.named_parameters())[f"blocks.{b}.{nm}"].data
                sr.append(stable_rank(W)); rv.append(float(W.norm(dim=1).var()))
            rec["w2d"][nm] = {"stable_rank": sum(sr)/9, "row_norm_var": sum(rv)/9}
        # qkv slices: v is what the pooling bug distorted, so report it on its own
        for tag, lo_f, hi_f in [("qkv[qk]", 0.0, 2/3), ("qkv[v]", 2/3, 1.0)]:
            sr, sd = [], []
            for b in range(9):
                W = dict(mw.named_parameters())[f"blocks.{b}.attn.qkv.weight"].data
                n0, n1 = int(lo_f*W.shape[0]), int(hi_f*W.shape[0])
                sr.append(stable_rank(W[n0:n1])); sd.append(float(W[n0:n1].float().std()))
            rec["w2d"][tag] = {"stable_rank": sum(sr)/9, "std": sum(sd)/9}
        for nm in P1D:
            mu, sd, ku = [], [], []
            for b in range(9):
                v = dict(mw.named_parameters())[f"blocks.{b}.{nm}"].data.float()
                mu.append(float(v.mean())); sd.append(float(v.std()))
                ku.append(float((((v - v.mean()) / (v.std() + 1e-12)) ** 4).mean()))
            rec["p1d"][nm] = {"mean": sum(mu)/9, "std": sum(sd)/9, "kurtosis": sum(ku)/9}
        out[arm] = rec
        print(f"{arm:10} qkv stable-rank {rec['w2d']['attn.qkv.weight']['stable_rank']:7.1f}   "
              f"norm1.weight mean {rec['p1d']['norm1.weight']['mean']:.3f} "
              f"std {rec['p1d']['norm1.weight']['std']:.3f} "
              f"kurt {rec['p1d']['norm1.weight']['kurtosis']:.2f}", flush=True)
        del mw
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
