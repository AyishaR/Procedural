#!/usr/bin/env python3
"""Dump the model EXACTLY as main.py initialises it, without training a single step.

Why this and not a reconstruction: plots/measure_init_rho_arms.py rebuilds each arm by hand,
so it can only ever prove that two of MY reconstructions agree. This runs main.py's own code
path -- pr_load_model, weight_shuffle, the quantile matcher, everything -- and grabs the model
at the moment before the first optimiser step, by replacing main.train_one_epoch (whose first
positional argument is the model) with a function that saves and exits.

Usage (1 GPU):
    torchrun --nproc_per_node=1 plots/dump_init.py --dump_to X.pth  <all the arm's main.py args>
"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M


def _capture(dump_to):
    def fake_train_one_epoch(model, *a, **k):
        sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        sd = {kk: v.detach().cpu() for kk, v in sd.items()}
        torch.save(sd, dump_to)
        print(f"[dump_init] wrote {dump_to} with {len(sd)} tensors, exiting before any training",
              flush=True)
        raise SystemExit(0)
    return fake_train_one_epoch


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dump_to", required=True)
    pre.add_argument("--dummy_wandb", action="store_true",
                     help="Replace utils.WandbLogger with a no-op so init paths that call "
                          "wandb_logger.update_config unconditionally (the delta-norm matching) "
                          "run without a wandb server. Pass --enable_wandb true alongside.")
    known, rest = pre.parse_known_args()
    if known.dummy_wandb:
        class _Dummy:
            """Absorbs any attribute access / call chain (wandb_logger._wandb.log(...), etc.)."""
            def __init__(self, *a, **k): pass
            def __getattr__(self, name): return _Dummy()
            def __call__(self, *a, **k): return _Dummy()
            def __bool__(self): return True
        M.utils.WandbLogger = _Dummy
        # main.py runs engine.model_analyse (a ~5 min pass over the val set) before training
        # when wandb is on; it is irrelevant to the init and is skipped here.
        if hasattr(M, "model_analyse"):
            M.model_analyse = lambda *a, **k: None
    parser = argparse.ArgumentParser(parents=[M.get_args_parser()])
    args = parser.parse_args(rest)
    M.train_one_epoch = _capture(known.dump_to)     # patched in main's namespace
    M.main(args)
