"""Shared loading/plotting helpers for the probes figures (results live under $PROBE_ROOT/results)."""
import json
from collections import defaultdict

import numpy as np

from probes import registry

DEPTH_X = np.arange(13)


def load_json_dir(sub, suffix=".json", exclude=("__knn", "__lowlevel", "__limit")):
    out = []
    for f in sorted((registry.RES / sub).glob(f"*{suffix}")):
        if any(x in f.name for x in exclude):
            continue
        try:
            out.append(json.load(open(f)))
        except json.JSONDecodeError:
            print(f"[figutils] unreadable {f}")
    return out


def by_arm(records, key_fn):
    """records -> {arm: {seed: value}} via key_fn(rec) -> value (or None to skip)."""
    out = defaultdict(dict)
    for r in records:
        v = key_fn(r)
        if v is not None:
            out[r["arm"]][r["seed"]] = np.asarray(v, dtype=float)
    return out


def arm_stats(d):
    """{seed: [13]} -> (mean [13], sd [13], n)."""
    a = np.stack(list(d.values()))
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1])), len(a)


def plot_arms(ax, per_arm, arms=None, band=True, lw=1.6, ls="-", label_n=True, alpha=1.0):
    """per_arm {arm: {seed: [13]}} -> one line per arm in ARM_ORDER with +/-1 sd band; returns arms drawn."""
    drawn = []
    for arm in (arms or registry.ARM_ORDER):
        if arm not in per_arm or not per_arm[arm]:
            continue
        m, s, n = arm_stats(per_arm[arm])
        lab = f"{arm} (n={n})" if label_n else arm
        ax.plot(DEPTH_X, m, color=registry.ARM_COLOR[arm], lw=lw, ls=ls, label=lab, alpha=alpha)
        if band and n > 1:
            ax.fill_between(DEPTH_X, m - s, m + s, color=registry.ARM_COLOR[arm], alpha=0.15, lw=0)
        drawn.append(arm)
    return drawn


def style_depth_axis(ax, fontsize=7):
    ax.set_xticks(DEPTH_X)
    ax.set_xticklabels(registry.DEPTH_LABELS, fontsize=fontsize - 1, rotation=90)
    ax.tick_params(labelsize=fontsize)
    ax.grid(alpha=0.3)


def missing_report(records, datasets, tag):
    have = {(r["arm"], r["seed"], r["dataset"]) for r in records}
    arms = {m["arm"]: [] for m in registry.MODELS}
    for m in registry.MODELS:
        arms[m["arm"]].append(m["seed"])
    miss = [(a, s, d) for a, seeds in arms.items() for s in seeds for d in datasets if (a, s, d) not in have]
    if miss:
        print(f"[{tag}] missing {len(miss)} (arm, seed, dataset) combinations, e.g. {miss[:6]}")
    return miss
