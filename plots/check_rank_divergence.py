#!/usr/bin/env python3
"""Do the DDP ranks hold DIFFERENT weights after main.py's init surgery?

utils.pr_load_model wraps the model in DDP (utils.py:968) and returns model.module. Every init
operation in main.py then runs on each rank's own replica, and DDP all-reduces gradients only --
it never re-syncs parameters. main.py:527 sets `seed = args.seed + utils.get_rank()`, so any
operation using torch.randperm draws a DIFFERENT permutation per rank, while one using
torch.argsort over the (DDP-broadcast, hence identical) tensor is rank-consistent.

Prints, per rank, for each of blocks 0-8:
  norm      -- permutation-INVARIANT: must agree across ranks either way
  checksum  -- sum(W * arange), permutation-SENSITIVE: differs iff the ranks diverge

Usage:  torchrun --nproc_per_node=2 plots/check_rank_divergence.py <main.py args>
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M, utils


def _probe(model, *a, **k):
    mw = model.module if hasattr(model, "module") else model
    r = utils.get_rank()
    out = []
    for b in [0, 4, 8]:
        for nm in ["attn.qkv.weight", "mlp.fc1.weight", "norm1.weight"]:
            W = dict(mw.named_parameters())[f"blocks.{b}.{nm}"].data.float().flatten()
            idx = torch.arange(W.numel(), device=W.device, dtype=W.dtype)
            out.append((f"blocks.{b}.{nm}", float(W.norm()), float((W * idx).sum())))
    for nm, n, c in out:
        print(f"[rank{r}] {nm:28} norm={n:12.5f}  checksum={c:20.4f}", flush=True)
    torch.distributed.barrier()
    raise SystemExit(0)


if __name__ == "__main__":
    args = M.get_args_parser().parse_args()
    M.train_one_epoch = _probe
    M.main(args)
