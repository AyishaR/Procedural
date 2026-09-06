#!/usr/bin/env python3
"""Compare two inits dumped straight out of main.py (see plots/dump_init.py).

The claim under test: ftb4e3 (proc's tensors, permuted within each slice) and ftbqm1dv
(a random model given proc's per-slice value multisets by rank map) are the SAME construction
in blocks 0-8, so any accuracy difference between them is run-level noise.

The algebra says yes -- both are a uniform permutation of the same multiset -- but their final
train losses do not overlap across three seeds each (2.573-2.590 vs 2.316-2.335), which says no.
This checks the actual initialised tensors rather than the argument.

Run:  .venv/bin/python plots/compare_inits.py
"""
import numpy as np, torch
from pathlib import Path

C = Path(__file__).resolve().parent / "cache"


def slices(name, W):
    if name.endswith("attn.qkv.weight"):
        e = W.shape[0] // 3
        return [("[q] ", W[:e]), ("[k] ", W[e:2*e]), ("[v] ", W[2*e:3*e]),
                ("[qk]", W[:2*e]), ("    ", W)]
    return [("    ", W)]


def main():
    A = torch.load(C / "init_ftb4e3.pth", map_location="cpu")
    B = torch.load(C / "init_ftbqm1dv.pth", map_location="cpu")
    assert set(A) == set(B), "different key sets"
    print(f"{len(A)} tensors in each dump\n")

    rows, mism = [], 0
    for k in sorted(A):
        if not k.startswith("blocks."):
            continue
        b = int(k.split(".")[1])
        if b > 8:
            continue
        for tag, _ in slices(k, A[k].float()):
            wa = dict(slices(k, A[k].float()))[tag]
            wb = dict(slices(k, B[k].float()))[tag]
            sa, sb = torch.sort(wa.flatten()).values, torch.sort(wb.flatten()).values
            same = torch.allclose(sa, sb, rtol=1e-5, atol=1e-6)
            rows.append((k, tag, float(wa.norm()), float(wb.norm()),
                         float(wa.std()), float(wb.std()), same))
            if not same:
                mism += 1

    n = len(rows)
    print(f"MULTISET IDENTICAL in {n-mism}/{n} tensor-slices of blocks 0-8")
    if mism:
        print("\nslices whose SORTED VALUES differ:")
        print(f"  {'tensor':34}{'slice':6}{'|W| 4e3':>10}{'|W| 1dv':>10}{'std 4e3':>10}{'std 1dv':>10}")
        for k, tag, na, nb, da, db, same in rows:
            if not same:
                print(f"  {k:34}{tag:6}{na:10.4f}{nb:10.4f}{da:10.5f}{db:10.5f}")
    print("\nblock 0, every slice:")
    print(f"  {'tensor':34}{'slice':6}{'|W| 4e3':>10}{'|W| 1dv':>10}{'same set':>10}")
    for k, tag, na, nb, da, db, same in rows:
        if k.startswith("blocks.0."):
            print(f"  {k:34}{tag:6}{na:10.4f}{nb:10.4f}{str(same):>10}")

    # everything OUTSIDE blocks 0-8 should be an independent random draw, not identical
    out = [k for k in A if not (k.startswith("blocks.") and int(k.split(".")[1]) <= 8)]
    r = [float(A[k].float().norm()) / max(float(B[k].float().norm()), 1e-12) for k in out
         if A[k].numel() and float(B[k].float().norm()) > 0]
    print(f"\noutside blocks 0-8 ({len(out)} tensors): |W| ratio median {np.median(r):.4f} "
          f"[{min(r):.4f}, {max(r):.4f}]  (~1 = same distribution, different draw)")


if __name__ == "__main__":
    main()
