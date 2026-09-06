#!/usr/bin/env python3
"""Mine the per-layer wandb traces recorded DURING training for anything that separates the
+2 arms from the rest -- and does so consistently with ftb4o.

plots/cache/wandb_epochwise.json holds five per-layer families that the training loop logs
every 10 epochs but that no figure or section has used except `acc_layer`:

    acc_layer{0..11}              linear read-out probe on each block's output
    attn_entropy_layer{0..11}     Shannon entropy of the attention distribution
    blk_act_rms_layer{0..11}      RMS of the residual stream leaving each block
    delta_norm_ratio_layer{0..11} rho, but measured during training rather than at init
    grad_norm_layer{0..11}        per-block gradient norm

This script turns each family into a set of scalar summaries (per-depth values, early/late
means, the early->late growth ratio, the depth at which the read-out probe first becomes
non-trivial) and scores every summary at every recorded epoch against final accuracy.

Scoring uses the same three gates as plots/score_ckpt_features.py, and for the same reason:
with only eight arms, |spearman| alone is far too easy to clear.  The gate that has killed
every candidate so far is `4o_ok` -- ftb4o is the single arm that ended BELOW random init, so
a real mechanism has to place it on the far side of random, not between random and the winners.

`stab` reports on what fraction of recorded epochs a summary passes all three gates.  A
mechanism should not switch on at epoch 200; the accuracy gap itself is latent for 200 epochs
(docs 3.10.9.8), so a summary that only works late is describing the outcome, not the cause.

Usage:  python plots/analyse_training_traces.py [--top 25]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "plots" / "cache" / "wandb_epochwise.json"

FAMILIES = ["acc_layer", "attn_entropy_layer", "blk_act_rms_layer",
            "delta_norm_ratio_layer", "grad_norm_layer"]
WIN = ["ftb4e3", "ftb3i"]
MID = ["ftbqmln", "ftbqm1d", "ftbqm", "ftbqmbias"]
BASE, BELOW = "r", "ftb4o"
SENTINEL = -1.0          # the training loop writes -1 when a metric is not computed


def accs():
    """Last-epoch test top-1 per arm, read from log.txt (never max-over-epochs)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "acd", ROOT / "plots" / "analyse_ckpt_differences.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: v[0] for k, v in m.arm_acc().items()}


def load():
    """{arm: {family: {epoch: np.array(12,) with nan for missing}}}"""
    raw = json.load(open(CACHE))
    out = {}
    for arm, rows in raw.items():
        fam = defaultdict(dict)
        for ep, row in rows.items():
            e = int(ep)
            if e < 0:
                continue
            for f in FAMILIES:
                v = np.array([row.get(f"Epoch-wise/{f}{l}", np.nan) for l in range(12)],
                             dtype=float)
                # a whole row of the sentinel means "not logged", not "measured as -1"
                if np.all(np.isnan(v)) or np.allclose(np.nan_to_num(v, nan=SENTINEL), SENTINEL):
                    continue
                fam[f][e] = v
        if fam:
            out[arm] = dict(fam)
    return out


def summarise(f, v):
    """Scalar summaries of one family's 12-vector."""
    early, late = v[:9], v[9:]
    s = {}
    for l in [0, 4, 8, 9, 11]:
        s[f"{f}@b{l}"] = v[l]
    s[f"{f}.early_mean"] = np.nanmean(early)
    s[f"{f}.late_mean"] = np.nanmean(late)
    s[f"{f}.sum"] = np.nansum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        s[f"{f}.late_over_early"] = np.nanmean(late) / np.nanmean(early)
        s[f"{f}.b8_over_b0"] = v[8] / v[0]
    # slope of the depth profile over blocks 0-8, in units of the profile's own mean
    ok = ~np.isnan(early)
    if ok.sum() >= 4:
        sl = np.polyfit(np.arange(9)[ok], early[ok], 1)[0]
        s[f"{f}.early_slope"] = sl / (abs(np.nanmean(early)) + 1e-12)
    if f == "acc_layer":
        # the depth at which the read-out probe first clears 10% -- "how deep the model has
        # to go before anything linearly decodable exists"
        idx = np.where(v > 10.0)[0]
        s["acc_layer.depth10"] = float(idx[0]) if len(idx) else 12.0
        idx = np.where(v > 1.0)[0]
        s["acc_layer.depth1"] = float(idx[0]) if len(idx) else 12.0
    return s


def spearman(a, b):
    # scipy, not argsort-of-argsort: several features tie six arms at the same value and a
    # tie-naive rank breaks them arbitrarily, which understates the correlation.
    return float(stats.spearmanr(a, b).statistic)


def gates(vals, acc):
    arms = [a for a in vals if a in acc and np.isfinite(vals[a])]
    if len(arms) < 6:
        return None
    y = np.array([acc[a] for a in arms])
    x = np.array([vals[a] for a in arms])
    if np.allclose(x.std(), 0):
        return None
    sp = spearman(x, y)
    ok4o = None
    if BELOW in vals and BASE in vals and all(w in vals for w in WIN):
        wd = np.mean([vals[w] for w in WIN]) - vals[BASE]
        od = vals[BELOW] - vals[BASE]
        ok4o = bool(wd * od < 0 and abs(od) > 0.05 * abs(wd))
    mid = [vals[m] for m in MID if m in vals and np.isfinite(vals[m])]
    sep = None
    if len(mid) >= 3 and all(w in vals for w in WIN):
        sep = float((np.mean([vals[w] for w in WIN]) - np.mean(mid)) / (np.std(mid) + 1e-12))
    return sp, ok4o, sep, len(arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--epoch", type=int, default=149,
                    help="epoch for the headline table (default 149, mid-training)")
    a = ap.parse_args()
    acc = accs()
    data = load()
    print(f"arms with traces: {', '.join(data)}")
    print(f"arms with accuracy: {', '.join(sorted(set(data) & set(acc)))}\n")

    # summary[epoch][name][arm]
    summary = defaultdict(lambda: defaultdict(dict))
    for arm, fams in data.items():
        for f, byep in fams.items():
            for ep, v in byep.items():
                for k, s in summarise(f, v).items():
                    if np.isfinite(s):
                        summary[ep][k][arm] = float(s)

    eps = sorted(summary)
    # stability: on how many epochs does a summary pass all three gates?
    passes, best = defaultdict(list), {}
    for ep in eps:
        for k, vals in summary[ep].items():
            g = gates(vals, acc)
            if g is None:
                continue
            sp, ok, sep, n = g
            good = abs(sp) > 0.7 and bool(ok) and sep is not None and abs(sep) > 1.5
            passes[k].append(good)
            if ep == a.epoch:
                best[k] = (sp, ok, sep, n)

    print(f"=== all summaries at epoch {a.epoch}, ranked by |spearman| ===\n")
    hdr = f"{'summary':34} {'spear':>6} {'4o_ok':>6} {'sep':>7} {'n':>3} {'stab':>6}"
    print(hdr); print("-" * len(hdr))
    for k, (sp, ok, sep, n) in sorted(best.items(), key=lambda kv: -abs(kv[1][0]))[:a.top]:
        st = np.mean(passes[k]) if passes[k] else 0.0
        print(f"{k:34} {sp:6.2f} {str(ok):>6} "
              f"{'   --  ' if sep is None else f'{sep:7.1f}'} {n:3d} {st:6.0%}")

    print("\n\n=== summaries passing all three gates on >= 60% of recorded epochs ===\n")
    hits = [k for k in passes if passes[k] and np.mean(passes[k]) >= 0.6]
    if not hits:
        print("  none.\n")
    for k in sorted(hits, key=lambda z: -np.mean(passes[z])):
        print(f"  {k}   (passes on {np.mean(passes[k]):.0%} of {len(passes[k])} epochs)")
        vals = summary[a.epoch].get(k, {})
        for arm in sorted(vals, key=lambda z: -acc.get(z, -99)):
            tag = "  <-- WIN" if arm in WIN else ("  <-- BELOW random" if arm == BELOW else "")
            print(f"      {arm:11} acc {acc.get(arm, float('nan')):6.2f}   "
                  f"{vals[arm]:12.4g}{tag}")
        print()

    # raw depth profiles at the headline epoch, for eyeballing
    print(f"\n=== raw depth profiles at epoch {a.epoch} ===")
    for f in FAMILIES:
        rows = [(arm, data[arm][f][a.epoch]) for arm in data
                if f in data.get(arm, {}) and a.epoch in data[arm][f]]
        if not rows:
            continue
        print(f"\n-- {f}")
        for arm, v in sorted(rows, key=lambda r: -acc.get(r[0], -99)):
            print(f"{arm:11} acc {acc.get(arm, float('nan')):6.2f}  " +
                  " ".join("   na " if not np.isfinite(x) else f"{x:6.2f}" for x in v))


if __name__ == "__main__":
    main()
