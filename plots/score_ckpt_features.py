#!/usr/bin/env python3
"""Rank every init-time feature by how well it explains the ImageNet-1k accuracy ordering.

Reads plots/cache/ckpt_diff.json (written by analyse_ckpt_differences.py) and scores each
feature against last-epoch test accuracy across the arms that have finished training.

Three gates, in increasing strictness:

  1. |Spearman| across all scored arms.  Descriptive only -- with 9-10 arms almost anything
     can clear r = 0.6 by chance, so this ranks rather than decides.
  2. `4o_ok`: does the feature place `ftb4o` on the FAR side of random init from the winners?
     ftb4o is the only arm that scored BELOW random (-0.81); a mechanism claiming "more of X
     is better" must therefore have ftb4o overshooting X, not undershooting it.  Every
     statistic published so far fails here, which is why it is a separate column.
  3. `sep`: the gap between the two +2 arms (ftb4e3, ftb3i) and the six middle arms, in units
     of the middle arms' own spread.  A feature that correlates but does not separate the
     winners from the pack is not the mechanism.
  4. `3i~4e3`: `ftb4e3` is `ftb3i` with every tensor randomly permuted WITHIN its slice, and
     the two score 80.16 and 79.99 -- statistically the same run.  So any feature on which
     they differ a lot is, by construction, not what produces the +2.  This gate needs no
     accuracy fit at all and is the strictest of the four; it is what kills attention
     entropy, logit spread, token collapse and the gradient-to-weight ratio, all of which
     move by 2-70x between two arms that train identically.

Two scoring modes.  `--mode monotone` (default) asks "more of X is better".  `--mode ushape`
asks instead whether accuracy tracks CLOSENESS to the procedural checkpoint's own value of X,
scoring against -|x - x_proc|.  Several quantities are calibrated rather than maximised -- the
recipe itself sets rho to a target, not to the largest value it can reach -- so a feature that
is flat under `monotone` can still be the mechanism under `ushape`.

Nothing here is causal.  It is a filter that says which measurements are worth turning into
an ablation, and -- just as usefully -- which candidates are already dead.

Usage:  python plots/score_ckpt_features.py [--json plots/cache/ckpt_diff.json] [--top 30]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
WIN = ["ftb4e3", "ftb3i"]                 # the +2 arms
MID = ["ftbqmln", "ftbqm1d", "ftbqm1dpar", "ftbnorm", "ftbqm", "ftbqmbias"]
BASE = "r"
BELOW = "ftb4o"
PROC = "p"            # the procedural checkpoint itself; the reference point for `ushape`
SHUF = ("ftb3i", "ftb4e3")   # identical up to a within-slice permutation, and identical in accuracy


def spearman(a, b):
    # scipy, not argsort-of-argsort: several features tie six arms at the same value and a
    # tie-naive rank breaks them arbitrarily, which understates the correlation.
    return float(stats.spearmanr(a, b).statistic)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def collect(d):
    """{feature: {arm: scalar}} -- per-block features are reduced to their blocks 0-8 mean,
    and additionally kept as first-block / last-block variants since some effects are
    depth-localised (the recipe's whole point is that block index matters)."""
    feats = {}
    arms = [a for a in d if not a.startswith("_")]
    for arm in arms:
        for k, v in d[arm]["per_block"].items():
            v = [x for x in v if x is not None]
            feats.setdefault(k, {})[arm] = float(np.mean(v))
            feats.setdefault(k + "@b0", {})[arm] = float(v[0])
            feats.setdefault(k + "@b8", {})[arm] = float(v[-1])
        for k, v in d[arm]["scalar"].items():
            feats.setdefault(k, {})[arm] = float(v)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="plots/cache/ckpt_diff.json")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--mode", choices=["monotone", "ushape"], default="monotone")
    a = ap.parse_args()
    d = json.load(open(ROOT / a.json if not Path(a.json).is_absolute() else a.json))
    acc = d["_acc"]
    feats = collect(d)
    scored = [x for x in acc if x != "p"]          # `p` is a different model family, not on the matrix

    rows = []
    for f, vals in feats.items():
        arms = [x for x in scored if x in vals]
        if len(arms) < 7:
            continue
        y = np.array([acc[x] for x in arms])
        v = np.array([vals[x] for x in arms])
        if np.allclose(v.std(), 0):
            continue
        if a.mode == "ushape":
            ref = vals.get(PROC, np.mean([vals[w] for w in WIN if w in vals]))
            scale = np.std(v) + 1e-12
            v = -np.abs(v - ref) / scale
            if np.allclose(v.std(), 0):
                continue
        sp, pe = spearman(v, y), pearson(v, y)

        # gate 2: does ftb4o overshoot in the winners' direction?
        ok4o = None
        if BELOW in vals and BASE in vals and all(w in vals for w in WIN):
            win_dir = np.mean([vals[w] for w in WIN]) - vals[BASE]
            o_dir = vals[BELOW] - vals[BASE]
            ok4o = bool(win_dir * o_dir < 0 and abs(o_dir) > 0.05 * abs(win_dir))

        # gate 3: winners vs the middle pack, in units of the pack's spread
        mid = [vals[m] for m in MID if m in vals]
        sep = None
        if len(mid) >= 4 and all(w in vals for w in WIN):
            # the middle arms are often numerically IDENTICAL on a weight-space feature
            # (pooled quantile matching gives them the same tensors), which sends a
            # spread-normalised gap to infinity.  Floor the spread at 2% of the winners'
            # own offset so `sep` stays a readable number rather than a divide-by-zero.
            off = abs(np.mean([vals[w] for w in WIN]) - np.mean(mid))
            sd = max(np.std(mid), 0.02 * off)
            sep = float((np.mean([vals[w] for w in WIN]) - np.mean(mid)) / (sd + 1e-12))
        shuf = None
        if all(z in vals for z in SHUF):
            lo, hi = vals[SHUF[0]], vals[SHUF[1]]
            den = max(abs(lo), abs(hi), 1e-12)
            shuf = float(abs(hi - lo) / den)      # relative disagreement between the two +2 arms
        rows.append((f, sp, pe, ok4o, sep, shuf, len(arms), vals))

    rows.sort(key=lambda r: -abs(r[1]))
    print(f"\n{len(rows)} features scored on {len(scored)} arms "
          f"({', '.join(scored)})\n")
    hdr = f"{'feature':38} {'spear':>6} {'pears':>6} {'4o_ok':>6} {'sep':>7} {'3i~4e3':>7}"
    print(hdr); print("-" * len(hdr))
    for f, sp, pe, ok, sep, shuf, n, _ in rows[:a.top]:
        print(f"{f:38} {sp:6.2f} {pe:6.2f} {str(ok):>6} "
              f"{'   --  ' if sep is None else f'{sep:7.1f}'} "
              f"{'   --  ' if shuf is None else f'{shuf:6.0%} '}")

    print("\n\n=== features that pass ALL FOUR gates (|spearman| > 0.7, 4o on the far side, "
          "|sep| > 2,\n    and the two +2 arms agreeing to within 25%) ===\n")
    win = [r for r in rows if abs(r[1]) > 0.7 and r[3] and r[4] is not None and abs(r[4]) > 2
           and r[5] is not None and r[5] < 0.25]
    if not win:
        print("  none.  No single measured statistic orders the arms, puts ftb4o on the far\n"
              "  side of random, AND stays put under the within-slice shuffle.  Either the\n"
              "  mechanism is a conjunction, or it is something not in this battery.")
    for f, sp, pe, ok, sep, shuf, n, vals in win:
        print(f"  {f:36} spearman {sp:5.2f}  sep {sep:6.1f}  3i~4e3 {shuf:.0%}")
        order = sorted([x for x in scored if x in vals], key=lambda x: -acc[x])
        for x in order:
            print(f"      {x:12} acc {acc[x]:6.2f}   {f.split('.')[-1]:24} {vals[x]:12.5g}")
        print()

    # a compact by-arm dump of the headline candidates, whether or not they pass
    print("\n=== by-arm values for the pre-registered candidates ===\n")
    cand = ["W.qk_over_v", "W.logit_scale", "W.value_write", "W.logit_over_write",
            "W.eff_q_norm", "W.ln1_gain_mean", "F.attn_entropy", "F.logit_std",
            "F.tok_cos", "F.eff_rank", "F.rho_attn", "G.g_over_w",
            "handoff.tok_cos", "handoff.eff_rank", "G.gnorm_blk9"]
    cand = [c for c in cand if c in feats]
    allarms = sorted({x for c in cand for x in feats[c]},
                     key=lambda x: -acc.get(x, -99))
    print(f"{'arm':12} {'acc':>6} " + " ".join(f"{c.split('.')[-1][:11]:>12}" for c in cand))
    for x in allarms:
        s = f"{x:12} " + (f"{acc[x]:6.2f} " if x in acc else f"{'--':>6} ")
        s += " ".join(f"{feats[c].get(x, float('nan')):12.4g}" for c in cand)
        print(s)


if __name__ == "__main__":
    main()
