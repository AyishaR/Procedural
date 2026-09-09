"""Zero the attention k-bias in a training checkpoint (model and EMA weights).

Why: softmax(q.(k+b_k)) == softmax(q.k + q.b_k) and q.b_k is the same for every key of a
given query, so the k-bias is a null direction for the attention output.  Training never
pushes it back, and in ftbrhos s0 it random-walked to |b_k| = 39 / 54 in blocks 9 / 10 by
epoch 271, at which point the fp16 q.k product overflowed on one batch and every resume
died with a non-finite loss at the same iteration (2026-09-09, job 29546613).  Zeroing it
is an exact no-op for the forward pass (up to fp16 rounding) and removes the overflow.

Usage:
    .venv/bin/python plots/verify/zero_kbias.py results/.../checkpoint-271.pth            # writes checkpoint-271.pth.kbias0
    .venv/bin/python plots/verify/zero_kbias.py results/.../checkpoint-271.pth --inplace  # keeps checkpoint-271.pth.orig_kbias
"""
import argparse, shutil, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--inplace", action="store_true", help="overwrite ckpt, keeping <ckpt>.orig_kbias")
ap.add_argument("--dim", type=int, default=768)
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = ck["model"]
names = [k for k in m if k.endswith("attn.qkv.bias")]
sl = slice(a.dim, 2 * a.dim)

# Adam state is left alone: the optimizer's param ids follow the param-group order, not the
# model's key order, so they cannot be mapped without rebuilding the optimizer.  The residual
# momentum can move the bias by at most lr per step (6e-5 at epoch 272), i.e. < 2.5 over the
# remaining 28 epochs -- far below the 39-54 that overflowed.
for n in names:
    before = m[n][sl].abs().max().item()
    m[n][sl].zero_()
    if ck.get("model_ema"):
        ck["model_ema"][n][sl].zero_()
    print(f"{n:28s} max|k-bias| {before:6.1f} -> 0")

out = a.ckpt if a.inplace else a.ckpt + ".kbias0"
if a.inplace:
    shutil.copy2(a.ckpt, a.ckpt + ".orig_kbias")
    print("backup:", a.ckpt + ".orig_kbias")
torch.save(ck, out)
print("wrote", out)
