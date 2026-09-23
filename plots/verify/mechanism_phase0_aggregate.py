"""Aggregate the sharded Phase-0 GPU results (plots/out/phase0/*.json) for docs/early_lever_mechanism_plan.md. Login-node safe.
usage: .venv/bin/python plots/verify/mechanism_phase0_aggregate.py [zoo] [depth] [fitgap]"""
import glob, json, sys
import numpy as np
from scipy.stats import spearmanr, rankdata
ROOT = "/home/schrodi/Procedural"
def merged(mode):
    out = {}
    for f in sorted(glob.glob(f"{ROOT}/plots/out/phase0/{mode}_shard*.json")): out.update(json.load(open(f)))
    return out
sections = sys.argv[1:] or ["zoo", "depth", "fitgap"]
m17 = lambda f, n: float(np.mean([f[str(b)][n] for b in range(1, 8)]))
m811 = lambda f, n: float(np.mean([f[str(b)][n] for b in range(8, 12)]))

if "zoo" in sections:
    Z = merged("zoo"); print(f"== ZOO: {len(Z)} runs")
    rows = []
    for key, r in Z.items():
        f, d, p = r["functional"], r["drift"], r["probe"]
        if not all(str(e) in f for e in (0, 4, 9, 19, 29)) or "29" not in p: continue
        task = "ksd" if r["arm"].startswith("ftbanak") else r["task"]          # the log-based label missed some ksd recipe arms
        v = {"final": r["final"], "train_loss": r["train_loss"], "task": task, "arm": r["arm"], "key": key}
        if "0" in d:   # early-lever family: blocks 9-11 untouched (q, k, fc1 at timm scale after the first epoch); covers the 0-7 and the 0-8 recipes
            v["late_timm"] = all(abs(np.mean([d["0"][str(b)][n] for b in range(9, 12)]) - 1.0) < 0.04 for n in ("rms_q", "rms_k", "rms_fc1"))
        for e in (0, 4, 9, 19, 29):
            for n in ("D", "att_specific", "mlp_specific", "att_common", "mlp_common", "entropy", "gelu_rms", "common_query"):
                v[f"{n}@{e}"] = m17(f[str(e)], n)
        for e in (0, 4, 9, 19):
            if str(e) in d:
                for n in ("q", "k", "v", "proj", "fc1", "fc2"):
                    v[f"drift_{n}@{e}"] = m17(d[str(e)], n); v[f"drift_{n}_late@{e}"] = m811(d[str(e)], n)
                v[f"drift_in@{e}"] = np.mean([v[f"drift_{n}@{e}"] for n in ("q", "k", "fc1")]); v[f"drift_in_rel@{e}"] = v[f"drift_in@{e}"] / np.mean([v[f"drift_{n}_late@{e}"] for n in ("q", "k", "fc1")])
        v["rms_q"] = m17(d["0"], "rms_q") if "0" in d else np.nan; v["rms_fc1"] = m17(d["0"], "rms_fc1") if "0" in d else np.nan
        for e in ("29", "299"):
            if e in p:
                for n in ("probe7", "probe9", "lens7", "lens9", "model"): v[f"{n}@{e}"] = p[e][n]
        v["share7@29"] = v["lens7@29"] / max(v["model@29"], 1e-9); v["probe_share7@29"] = v["probe7@29"] / max(v["model@29"], 1e-9)
        rows.append(v)
    print(f"   complete records: {len(rows)} ({sum(r['task'] == 'ksd' for r in rows)} ksd)")
    late_like = [r for r in rows if r["lens7@29"] < 3.0 and r["probe7@29"] > 8]      # loud late blocks: the head cannot read block 7 although the class is there
    print(f"   late-lever signature (lens7@29 < 3 while probe7@29 > 8): {sorted(set(r['arm'] for r in late_like))}")
    def table(name, sel, cands):
        y = np.array([r["final"] for r in sel]); base = np.array([r["lens7@29"] for r in sel])
        print(f"-- {name}: n = {len(sel)}  (final {y.min():.1f}..{y.max():.1f})   Spearman with the final | partial, given lens7@29")
        res = []
        for c in cands:
            x = np.array([r.get(c, np.nan) for r in sel], dtype=float); ok = np.isfinite(x)
            if ok.sum() < 8: continue
            rho = spearmanr(x[ok], y[ok])[0]
            rx, ry, rb = rankdata(x[ok]), rankdata(y[ok]), rankdata(base[ok])
            res_x = rx - np.polyval(np.polyfit(rb, rx, 1), rb); res_y = ry - np.polyval(np.polyfit(rb, ry, 1), rb)
            part = float(np.corrcoef(res_x, res_y)[0, 1]) if c != "lens7@29" else float("nan")
            res.append((abs(rho), c, rho, part))
        for _, c, rho, part in sorted(res, reverse=True)[:22]: print(f"      {c:22s} {rho:+.2f} | {part:+.2f}")
    cands = [k for k in rows[0] if k not in ("final", "train_loss", "task", "arm", "key", "late_timm") and not k.endswith("@299")] + ["train_loss"]
    core = [r for r in rows if r not in late_like]
    fam = [r for r in core if r.get("late_timm")]
    print(f"   early-lever family (blocks 9-11 at timm scale, no late-lever signature): {len(fam)} runs, {len(set(r['arm'] for r in fam))} arms")
    table("all runs", rows, cands); table("without the late-lever signature", core, cands)
    table("EARLY-LEVER FAMILY", fam, cands); table("  family, kdyck", [r for r in fam if r["task"] == "kdyck"], cands); table("  family, ksd", [r for r in fam if r["task"] == "ksd"], cands)
    table("  family, winners only (final >= 79)", [r for r in fam if r["final"] >= 79.0], cands)
    table("  kdyck only", [r for r in core if r["task"] == "kdyck"], cands); table("  ksd only", [r for r in core if r["task"] == "ksd"], cands)
    table("  winners only (final >= 79)", [r for r in core if r["final"] >= 79.0], cands)
    print("-- end-state footprints against the final (epoch 299), without the late-lever signature")
    y = np.array([r["final"] for r in core])
    for c in ("probe7@299", "probe9@299", "lens7@299", "lens9@299", "train_loss"):
        x = np.array([r.get(c, np.nan) for r in core], dtype=float); ok = np.isfinite(x); print(f"      {c:12s} Spearman {spearmanr(x[ok], y[ok])[0]:+.2f} (n = {ok.sum()})")
    print("-- per arm (seed mean), sorted by final: final | lens7@29 probe7@29 | D@0 D@9 | att spec @9 | mlp spec @9 | drift q,k,fc1 @4 (%) | raw rms q, fc1 | probe7@299")
    arms = {}
    for r in rows: arms.setdefault((r["arm"], r["task"]), []).append(r)
    for (arm, task), rs in sorted(arms.items(), key=lambda kv: -np.mean([r["final"] for r in kv[1]])):
        g = lambda n: np.nanmean([r.get(n, np.nan) for r in rs])
        print(f"      {arm:16s} {task:5s} n={len(rs)} {g('final'):6.2f} | {g('lens7@29'):5.1f} {g('probe7@29'):5.1f} | {g('D@0'):.3f} {g('D@9'):.3f} | {g('att_specific@9'):.3f} | {g('mlp_specific@9'):.3f} | {100 * g('drift_q@4'):5.1f} {100 * g('drift_k@4'):5.1f} {100 * g('drift_fc1@4'):5.1f} | {g('rms_q'):.2f} {g('rms_fc1'):.2f} | {g('probe7@299'):5.1f}")

if "depth" in sections:
    Dp = merged("depth"); print(f"\n== DEPTH: {len(Dp)} models: trained probe (top) and head lens (bottom) per block 0..11")
    arms = sorted(set(k.split("@")[0] for k in Dp))
    for arm in arms:
        eps = sorted(int(k.split("@")[1]) for k in Dp if k.split("@")[0] == arm); print(f"-- {arm}: {Dp[f'{arm}@{eps[0]}']['what']}")
        for e in eps:
            r = Dp[f"{arm}@{e}"]
            print(f"   ep {e:3d} probe " + " ".join(f"{r['probe'][str(b)]:5.1f}" for b in range(12)) + f"\n          lens  " + " ".join(f"{r['lens'][str(b)]:5.1f}" for b in range(12)))

if "fitgap" in sections:
    F = merged("fitgap"); print(f"\n== FIT GAP at the end of training: {len(F)} models (25 images per class; train WITHOUT augmentation)")
    print("   arm              clean-train acc / ce | val acc / ce | gap acc, ce")
    for arm, r in sorted(F.items(), key=lambda kv: -kv[1]["val_acc"]):
        print(f"   {arm:16s} {r['train_clean_acc']:6.2f} / {r['train_clean_ce']:.3f} | {r['val_acc']:6.2f} / {r['val_ce']:.3f} | {r['train_clean_acc'] - r['val_acc']:5.2f}, {r['val_ce'] - r['train_clean_ce']:.3f}   {r['what']}")
