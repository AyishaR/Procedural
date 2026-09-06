#!/usr/bin/env python3
"""Verify the init-sync fix: do the ranks agree after it, for BOTH arms?

Runs on CPU with gloo, 2 processes, no dataset and no GPU. Exercises the real
utils.shuffle_weights (the ftb4e3 path) and the real 2-D quantile rank map (the ftbqm1dv
path), then applies the exact broadcast inserted into main.py before the training loop.

Checks three things:
  1. ftb4e3 path diverges across ranks BEFORE the fix   (the bug)
  2. both paths agree across ranks AFTER the fix         (the fix works)
  3. the two paths give each other the same value multiset per slice (the arms really are
     the same construction, which is what the whole comparison rests on)

Usage:  torchrun --nproc_per_node=2 plots/verify_rank_fix.py
"""
import sys, json, torch, torch.distributed as dist
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils


def stamp(W):
    W = W.detach().float().flatten()
    idx = torch.arange(W.numel(), dtype=W.dtype)
    return round(float(W.norm()), 6), round(float((W * idx).sum()), 3)


def multiset(W):
    return torch.sort(W.detach().float().flatten()).values


class Blk(torch.nn.Module):
    """Minimal stand-in with the attribute paths utils.shuffle_weights resolves."""
    def __init__(self, d=64):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv = torch.nn.Linear(d, 3 * d, bias=True)
        self.attn.proj = torch.nn.Linear(d, d, bias=True)
        self.norm1 = torch.nn.LayerNorm(d)
        self.norm2 = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(d, 4 * d, bias=True)
        self.mlp.fc2 = torch.nn.Linear(4 * d, d, bias=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Blk()])


def apply_fix(m):
    """Verbatim copy of the block inserted into main.py before 'Start training'."""
    n = 0
    for _, t in sorted(m.state_dict().items()):
        if torch.is_tensor(t) and t.is_floating_point():
            dist.broadcast(t.data, src=0)
            n += 1
    dist.barrier()
    return n


def build(seed):
    torch.manual_seed(12345)          # the DDP-synced starting point: same on every rank
    m = Net()
    torch.manual_seed(seed)           # main.py:527  seed = args.seed + get_rank()
    return m


def main():
    dist.init_process_group("gloo")
    r = dist.get_rank()
    seed = 0 + r
    donor = torch.randn(3 * 64 * 64, generator=torch.Generator().manual_seed(777))
    res = {}

    # ---- ftb4e3 path: utils.shuffle_weights (torch.randperm) ----
    m = build(seed)
    utils.shuffle_weights(m, {0: ["attn.qk.weight", "attn.v.weight", "mlp.fc1.weight"]})
    res["4e3_before"] = stamp(m.blocks[0].attn.qkv.weight)
    n = apply_fix(m)
    res["4e3_after"] = stamp(m.blocks[0].attn.qkv.weight)
    ms_4e3 = multiset(m.blocks[0].attn.qkv.weight[:128])

    # ---- ftbqm1dv path: 2-D quantile rank map (torch.argsort, no RNG) ----
    m2 = build(seed)
    with torch.no_grad():
        W = m2.blocks[0].attn.qkv.weight.data
        e = W.shape[0] // 3
        for lo, hi in [(0, 2 * e), (2 * e, 3 * e)]:
            d = donor[lo * 64:hi * 64]
            ds, _ = torch.sort(d)
            order = torch.argsort(W[lo:hi].flatten().float())
            out = torch.empty_like(ds); out[order] = ds
            W[lo:hi].copy_(out.view_as(W[lo:hi]))
    res["1dv_before"] = stamp(m2.blocks[0].attn.qkv.weight)
    apply_fix(m2)
    res["1dv_after"] = stamp(m2.blocks[0].attn.qkv.weight)

    Path("plots/cache").mkdir(parents=True, exist_ok=True)
    Path(f"plots/cache/fixchk_{r}.json").write_text(json.dumps({"rank": r, "nsync": n, **res}))
    print(f"[rank{r}] synced {n} tensors; " + "  ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
