#!/usr/bin/env python3
"""Minimal reproduction: does a post-DDP weight edit diverge across ranks?

Mirrors exactly what main.py does, with nothing else attached:
  1. seed = args.seed + get_rank()          (main.py:527)
  2. build the model, wrap in DDP           (utils.pr_load_model:968)
  3. edit model.module's weights AFTER that (main.py:1294 / :983 / :1035)

DDP broadcasts parameters once at construction and thereafter all-reduces GRADIENTS only,
so any post-construction edit using per-rank randomness leaves the ranks holding different
models for the whole run.

Two edits are compared, the two primitives the real arms use:
  randperm  -- utils.shuffle_weights           (ftb4e3, ftb11is, every --weight_shuffle arm)
  argsort   -- the 2-D quantile rank map       (ftbqm1d*, all quantile arms)

norm is permutation-invariant (must agree either way); checksum = sum(W*arange) is
permutation-sensitive (differs iff the ranks hold different arrangements).

Usage:  torchrun --nproc_per_node=2 plots/rank_rng_probe.py
"""
import sys, torch, torch.distributed as dist
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def stamp(t):
    W = t.detach().float().flatten()
    idx = torch.arange(W.numel(), device=W.device, dtype=W.dtype)
    return float(W.norm()), float((W * idx).sum())


def main():
    dist.init_process_group("nccl")
    r, n = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r)
    seed = 0 + r                                   # exactly main.py:527
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    donor = torch.randn(4096, generator=torch.Generator().manual_seed(1234))  # same on all ranks
    m = torch.nn.Linear(64, 64, bias=False).cuda()
    torch.nn.init.normal_(m.weight, std=0.02)
    ddp = torch.nn.parallel.DistributedDataParallel(m, device_ids=[r])
    mw = ddp.module
    a_n, a_c = stamp(mw.weight)                    # after DDP broadcast: must be identical

    # --- edit A: randperm, as utils.shuffle_weights does -------------------------------
    with torch.no_grad():
        flat = mw.weight.data.view(-1)
        mw.weight.data.copy_(flat[torch.randperm(flat.numel(), device=flat.device)].view_as(mw.weight.data))
    b_n, b_c = stamp(mw.weight)

    # --- edit B: argsort rank-map, as the 2-D quantile matcher does ---------------------
    torch.nn.init.normal_(mw.weight, std=0.02)     # per-rank draw
    dist.broadcast(mw.weight.data, src=0)          # emulate DDP's construction-time sync
    with torch.no_grad():
        ds, _ = torch.sort(donor.cuda())
        order = torch.argsort(mw.weight.data.flatten().float())
        out = torch.empty_like(ds); out[order] = ds
        mw.weight.data.copy_(out.view_as(mw.weight.data))
    c_n, c_c = stamp(mw.weight)

    # write per-rank results to separate files, then a barrier, then a NORMAL return.
    # An earlier version called os._exit(0) after destroy_process_group(), which let rank 0
    # tear the group down while rank 1 was still inside the broadcast -- rank 1 then hung.
    import json
    out = {"rank": r, "world": n,
           "after_ddp":      [a_n, a_c],
           "after_randperm": [b_n, b_c],
           "after_argsort":  [c_n, c_c]}
    Path("plots/cache").mkdir(parents=True, exist_ok=True)
    Path(f"plots/cache/rank_probe_{r}.json").write_text(json.dumps(out))
    for tag, (nn_, cc) in [("after DDP construction", (a_n, a_c)),
                           ("after randperm  edit  ", (b_n, b_c)),
                           ("after argsort   edit  ", (c_n, c_c))]:
        print(f"[rank{r}/{n}] {tag}  norm={nn_:12.6f}  checksum={cc:18.4f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
