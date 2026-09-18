"""Collect the stage results into per-run tables, compute the loss-matched epochs, and print the summary tables
for docs/training_dynamics.md.

  python -m dynamics.summarize matched      # results/hist/matched_epochs.json from the rawlens dense grid
  python -m dynamics.summarize tables       # markdown tables to stdout (+ results/summary/*.csv)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from dynamics import registry as dreg
from dynamics.rawlens import HALF

FINAL = len(HALF) - 1
INCLUDE_LIMITED = os.environ.get("DYN_INCLUDE_LIMITED") == "1"      # smoke-root figure checks only


def _limited(fname):
    return "__limit" in os.path.basename(fname)


# ------------------------------------------------------------------------------------------------- loading
def load_rawlens(run, split="test", tag=""):
    """-> dict(files, E, top1 [n,25], ce, ce_smooth, margin_mean, margin_median, maxp, agree_final, lr_mean) sorted by file."""
    recs = {}
    for f in glob.glob(str(dreg.RES / "rawlens" / f"{run}@*__{split}*.json")):
        if _limited(f) and not INCLUDE_LIMITED:
            continue
        r = json.load(open(f))
        if r.get("tag", "") != tag:
            continue
        key = r["file"]
        if key in recs and recs[key]["n"] >= r["n"]:               # duplicate (run, file, split, tag): keep the larger cohort
            continue
        recs[key] = r
    recs = sorted(recs.values(), key=lambda r: r["file"])
    if not recs:
        return None
    out = dict(run=run, files=[r["file"] for r in recs], E=[r["E"] for r in recs], lr_mean=[r.get("lr_mean") for r in recs],
               test_acc1_log=[r.get("test_acc1_log") for r in recs])
    for k in ("top1", "top5", "ce", "ce_smooth", "margin_mean", "margin_median", "maxp", "agree_final"):
        out[k] = np.array([r[k] for r in recs], float)
    return out


def load_hist(run):
    p = dreg.RES / "hist" / f"{run}.json"
    if not p.exists():
        return None
    h = json.load(open(p))
    if not h["files"]:
        return None
    h["top1"] = np.array([[np.nan if v is None else v for v in row] for row in h["top1"]], float)
    return h


def load_stage(stage, run, tag=""):
    """All per-epoch JSONs of a stage for a run (no smoke files, only the given tag), sorted by file index; duplicates
    of one (file, split/kind) keep the larger cohort."""
    recs = {}
    for f in glob.glob(str(dreg.RES / stage / f"{run}@*.json")):
        if _limited(f) and not INCLUDE_LIMITED:
            continue
        r = json.load(open(f))
        if r.get("tag", "") != tag:
            continue
        key = (r.get("file_from", r.get("file")), r.get("split", r.get("kind")))
        n = r.get("n") or sum(v.get("n", 0) for v in r.get("sets", {}).values() if isinstance(v, dict))
        if key in recs and recs[key][0] >= n:
            continue
        recs[key] = (n, r)
    key = "file_from" if stage == "funcchange" else "file"
    return sorted((r for _, r in recs.values()), key=lambda r: r.get(key, -1))


def censored_U(run, set_name="diag_train2k", key="U_frob", censor=3.0):
    """Function-change series with the documented censoring: (e_from [n], E_to [n], U [n,12], mask [n,12], floor [12], src [n]).
    mask marks fp16-pair values below `censor` x the run's measured per-block fp16 floor (fp32 pairs are never censored)."""
    fc = [r for r in load_stage("funcchange", run) if r["kind"] == "pair" and set_name in r["sets"]]
    if not fc:
        return None
    e = np.array([r["file_from"] for r in fc], float)
    order = np.argsort(e)
    e = e[order]
    U = np.array([r["sets"][set_name][key] for r in fc], float)[order]
    src = np.array([r["source"] for r in fc])[order]
    floors = [r for r in load_stage("funcchange", run) if r["kind"] == "floor" and set_name in r["sets"]]
    floor = np.max([r["sets"][set_name][key] for r in floors], 0) if floors else np.zeros(U.shape[1])
    mask = (U < censor * floor[None, :]) & (src[:, None] == "fp16")
    return e, e + 2, U, mask, floor, src


# ------------------------------------------------------------------------------------------------- loss matching
def monotone_min(x):
    return np.minimum.accumulate(np.asarray(x, float))


def matched_epochs(role_ref="random-like", rel_tol=0.05, write=True):
    """Loss-matched epochs per run, selected among the checkpoints that rawlens has evaluated.

    L(E) = the full model's hard-label CE on the official val set (rawlens slot b11), made monotone by the running
    minimum.  Level grid L* = the reference role's seed-mean L at its core epochs.  For every run and level the
    evaluated checkpoint whose monotone L is closest to L* is selected; the achieved L, its relative deviation,
    E and lr are recorded and a selection is `ok` only if |L_sel - L*| <= rel_tol * L*.  `common_levels` are the
    levels every run matches within tolerance (the primary cross-run comparison); per-run extra levels are kept
    but flagged.  Runs never reaching a level get no epoch for it."""
    curves = {}
    for r in dreg.RUNS:
        rl = load_rawlens(r["name"])
        if rl is None or len(rl["files"]) < 5:
            continue
        E = np.array(rl["E"], float)
        keep = E > 0
        curves[r["name"]] = (E[keep], monotone_min(rl["ce"][keep, FINAL]), np.array(rl["files"])[keep], np.array(rl["lr_mean"], float)[keep])
    ref = [n for n in curves if dreg.ROLE[n] == role_ref]
    assert ref, f"no rawlens results for role {role_ref}"
    E_core = np.array(dreg.EPOCHS_CORE, float)
    ref_L = np.mean([np.interp(E_core, curves[n][0], curves[n][1]) for n in ref], 0)
    detail, matched = {}, {}
    for n, (E, L, files, lr) in curves.items():
        rows = []
        for Ec, Lstar in zip(dreg.EPOCHS_CORE, ref_L):
            i = int(np.argmin(np.abs(L - Lstar)))
            dev = float((L[i] - Lstar) / Lstar)
            rows.append(dict(level_E=int(Ec), L_star=float(Lstar), file=int(files[i]), E=int(E[i]), L=float(L[i]), rel_dev=dev,
                             lr=float(lr[i]) if np.isfinite(lr[i]) else None, ok=abs(dev) <= rel_tol))
        detail[n] = rows
    common = [Ec for k, Ec in enumerate(dreg.EPOCHS_CORE) if all(detail[n][k]["ok"] for n in detail)]
    for n, rows in detail.items():
        matched[n] = sorted({r["file"] for r in rows if r["ok"] and r["level_E"] in common})
    meta = dict(reference_role=role_ref, reference_runs=ref, E_core=dreg.EPOCHS_CORE, L_star=ref_L.tolist(), rel_tol=rel_tol,
                common_levels=common, detail=detail,
                note="file indices of evaluated checkpoints whose monotone full-model test CE is within rel_tol of the reference "
                     "role's CE at each core epoch; primary comparison uses common_levels only", **dreg.stamp())
    if write:
        dreg.write_json(dreg.RES / "hist" / "matched_epochs.json", {**matched, "_meta": meta})
    return dict(matched=matched, meta=meta)


def transition_epochs(depths=("b8", "b9"), taus=(0.5, 0.9), width=2, write=True):
    """Epoch files bracketing each run's raw-lens confidence transitions (sustained crossing of tau * terminal
    top-1 at depth d, test split): the crossing file and `width` files on either side, restricted to dumps that
    exist.  Used to refine the sampling around candidate transitions (handoff: expand around transitions)."""
    out = {}
    for r in dreg.RUNS:
        rl = load_rawlens(r["name"])
        if rl is None:
            continue
        E = np.array(rl["E"], float)
        keep = E > 0
        E = E[keep]
        avail = set(dreg.available_files(dreg.entry(r["name"])))
        files = set()
        for d in depths:
            v = rl["top1"][keep, HALF.index(d)]
            for tau in taus:
                t, _ = sustained_first_E(E, v, tau * v[int(np.argmax(E))])
                if t is not None:
                    files |= {f for f in range(int(t) - 1 - width, int(t) - 1 + width + 1) if f in avail}
        out[r["name"]] = sorted(files)
    if write:
        dreg.write_json(dreg.RES / "hist" / "transition_epochs.json", {**out, "_meta": dict(depths=depths, taus=taus, width=width, **dreg.stamp())})
    return out


# ------------------------------------------------------------------------------------------------- sequence statistics
def sustained_first_E(E, v, thr):
    """(E_cross, E_prev): the smallest sampled E from which v stays >= thr at every later sampled epoch, and the
    previous sampled epoch (the transition lies in (E_prev, E_cross]); (None, None) if never sustained."""
    E, v = np.asarray(E, float), np.asarray(v, float)
    ok = np.isfinite(v) & (v >= thr)
    if not ok.any() or not ok[-1]:
        return None, None
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    return float(E[i]), (float(E[i - 1]) if i > 0 else 0.0)


def last_crossing(E, v, final, tol=0.1):
    """Plateau time: the smallest sampled E such that |v(e) - final| <= tol * |final| for every sampled e >= E
    (None if never); with the previous sampled epoch as bracket."""
    E = np.asarray(E, float)
    ok = np.abs(np.asarray(v, float) - final) <= tol * abs(final) + 1e-12
    if not ok.any() or not ok[-1]:
        return None, None
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    return float(E[i]), (float(E[i - 1]) if i > 0 else 0.0)


def interp_logE(E, v, e0):
    """Linear interpolation in log E."""
    E, v = np.asarray(E, float), np.asarray(v, float)
    return float(np.interp(np.log(e0), np.log(E), v))


C_MIN_ABS = 0.02       # CE units: usefulness fractions are undefined when |C_final| is below this floor


def sequence_stats(run, depths=("b8", "b9", "b11"), taus=(0.5, 0.9), late_blocks=(9, 10, 11), tol=0.1, split="test",
                   fc_set="diag_train2k", fc_key="U_frob", gr_set="diag_train4k", gr_obj="smooth", censor=3.0):
    """The handoff's sequence test for one run (definitions frozen before production, docs section 0.4).

    t_conf(d, tau): sustained crossing of tau * terminal top-1 at depth d (terminal = the run's own E = 300 value at
    the same depth); confidence variants: median correct-class margin > 0, mean max-prob >= 0.5 * terminal.  Every
    time comes with the previous sampled epoch (bracket).  Late-block usefulness C_l: plateau = last crossing within
    +-tol of the terminal value; fraction acquired after t_conf uses log-E interpolation, reports the signed
    numerator and is undefined when |C_final| < C_MIN_ABS.  Functional change: U values below `censor` x the run's
    measured fp16 floor (per block) are censored; the cumulative path A_l and the late share are reported with
    censored terms set to 0 (lower) and to the floor (upper).  Shares before/after t_conf are ratios of sums."""
    out = dict(run=run, role=dreg.ROLE.get(run), complete=True)
    rl = load_rawlens(run, split)
    if rl is None:
        return out
    E = np.array(rl["E"], float)
    keep = E > 0
    E = E[keep]
    final_i = int(np.argmax(E))
    if E[final_i] != 300 or len(E) < len(dreg.DENSE_FILES):            # terminal values need E = 300 and the full dense grid
        out["complete"] = False
    for d in depths:
        j = HALF.index(d)
        top = rl["top1"][keep, j]
        for tau in taus:
            t, tp = sustained_first_E(E, top, tau * top[final_i])
            out[f"t_conf_{d}_top1_{tau}"], out[f"t_conf_{d}_top1_{tau}_prev"] = t, tp
        out[f"t_conf_{d}_margin_pos"], out[f"t_conf_{d}_margin_pos_prev"] = sustained_first_E(E, rl["margin_median"][keep, j], 0.0)
        mp = rl["maxp"][keep, j]
        out[f"t_conf_{d}_maxp_0.5"], out[f"t_conf_{d}_maxp_0.5_prev"] = sustained_first_E(E, mp, 0.5 * mp[final_i])
        out[f"final_top1_{d}"] = float(top[final_i])
        out[f"final_maxp_{d}"] = float(mp[final_i])
    t_ref = out.get("t_conf_b9_top1_0.5")                                # the reference confidence time
    # experiment 2: usefulness of the late blocks
    bp = [r for r in load_stage("bypass", run) if r["split"] == split]
    if bp:
        Eb = np.array([r["E"] for r in bp], float)
        order = np.argsort(Eb)
        Eb = Eb[order]
        if Eb.max() != 300 or not set(dreg.EPOCHS_CORE) <= set(Eb.tolist()):
            out["complete"] = False
        for l in list(late_blocks) + ["last3"]:
            C = np.array([r["multi"]["last3"]["C"] if l == "last3" else r["by_gain"]["0.0"]["C"][l] for r in bp], float)[order]
            fin = C[-1]
            out[f"C_{l}_final"] = float(fin)
            out[f"t90_C_{l}"], out[f"t90_C_{l}_prev"] = last_crossing(Eb, C, fin, tol)
            if t_ref is not None and Eb.min() <= t_ref <= Eb.max():
                num = fin - interp_logE(Eb, C, t_ref)
                out[f"gain_after_tconf_C_{l}"] = float(num)
                out[f"frac_after_tconf_C_{l}"] = float(num / fin) if abs(fin) >= C_MIN_ABS else None
    # experiment 3: functional change with censoring bounds; a pair i -> i+1 is the interval (E_from, E_to] = (i+1, i+2]
    cu = censored_U(run, fc_set, fc_key, censor)
    if cu is not None:
        e, E_to, U, cens, floor, src = cu
        if len(e) < 299:
            out["complete"] = False
        out["U_censored_pairs"] = int(cens.any(1).sum())
        out["U_censored_first_file"] = int(e[cens.any(1)].min()) if cens.any() else None
        lo, hi = np.where(cens, 0.0, U), np.where(cens, np.broadcast_to(floor, U.shape), U)
        for tag, UU in (("lo", lo), ("hi", hi)):
            share = UU[:, list(late_blocks)].sum(1) / np.maximum(UU.sum(1), 1e-30)
            for l in late_blocks:
                A = np.cumsum(UU[:, l])
                out[f"t90_A_{l}_{tag}"], out[f"t90_A_{l}_{tag}_prev"] = last_crossing(E_to, A, A[-1], tol)
            if t_ref is not None:
                b, a = E_to <= t_ref, (E_to - 1) >= t_ref          # before: interval ends by t_conf; after: starts at or after it
                out[f"late_share_U_before_tconf_{tag}"] = float(UU[b][:, list(late_blocks)].sum() / max(UU[b].sum(), 1e-30)) if b.any() else None
                out[f"late_share_U_after_tconf_{tag}"] = float(UU[a][:, list(late_blocks)].sum() / max(UU[a].sum(), 1e-30)) if a.any() else None
            out[f"late_share_U_final10_{tag}"] = float(share[-10:].mean())
        out["U_late_final10_abs"] = float(U[-10:, list(late_blocks)].mean())
        out["U_late_peak_abs"] = float(U[:, list(late_blocks)].max())
    # experiment 4: gradient late share before / after t_conf (ratio of sums of squared norms)
    gr = [r for r in load_stage("grads", run) if gr_set in r["sets"]]
    if gr:
        Eg = np.array([r["E"] for r in gr], float)
        order = np.argsort(Eg)
        Eg = Eg[order]
        pb = np.array([r["sets"][gr_set]["objectives"][gr_obj]["per_block"] for r in gr], float)[order] ** 2
        late = pb[:, list(late_blocks)].sum(1) / pb.sum(1)
        if Eg.max() != 300:
            out["complete"] = False
        out["t90_grad_late_share"], out["t90_grad_late_share_prev"] = last_crossing(Eg, late, late[-1], tol)
        if t_ref is not None:
            b, a = Eg <= t_ref, Eg > t_ref
            out["grad_late_share_before_tconf"] = float(pb[b][:, list(late_blocks)].sum() / pb[b].sum()) if b.any() else None
            out["grad_late_share_after_tconf"] = float(pb[a][:, list(late_blocks)].sum() / pb[a].sum()) if a.any() else None
        out["grad_late_abs_final"] = float(np.sqrt(pb[-1, list(late_blocks)].sum()))
    return out


def table_sequence(split="test"):
    rows = [sequence_stats(r["name"], split=split) for r in dreg.RUNS]
    keys = [k for k in rows[0] if k not in ("run", "role")] if rows else []
    keys = sorted({k for r in rows for k in r if k not in ("run", "role")}, key=lambda k: (keys.index(k) if k in keys else 999))
    (dreg.RES / "summary").mkdir(parents=True, exist_ok=True)
    import csv
    with open(dreg.RES / "summary" / f"sequence_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "role"] + keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ["run", "role"] + keys})
    show = [k for k in ("complete", "t_conf_b9_top1_0.5", "t_conf_b9_top1_0.9", "t_conf_b9_margin_pos", "t_conf_b9_maxp_0.5", "t90_C_9", "t90_C_10", "t90_C_11",
                        "t90_C_last3", "frac_after_tconf_C_last3", "t90_A_11_lo", "late_share_U_before_tconf_lo", "late_share_U_after_tconf_lo",
                        "U_censored_first_file", "grad_late_share_before_tconf", "grad_late_share_after_tconf") if k in keys]
    print(f"\n### sequence statistics ({split}; rows with complete=False are provisional)\n")
    print("| run | role | " + " | ".join(show) + " |")
    print("|---|---|" + "---|" * len(show))
    for r in rows:
        print(f"| {r['run']} | {r['role']} | " + " | ".join(str(r.get(k)) if isinstance(r.get(k), bool) else fmt(r.get(k), 2 if isinstance(r.get(k), float) and abs(r.get(k)) < 10 else 0) for k in show) + " |")
    return rows


# ------------------------------------------------------------------------------------------------- tables
def fmt(v, d=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def table_rawlens(split="test", depths=("b5", "b8", "b9", "b10", "b11"), E_sel=(1, 3, 10, 40, 150, 300)):
    print(f"\n### raw lens top-1 ({split}) by run, depth x epoch E\n")
    print("| run | role | depth | " + " | ".join(f"E{E}" for E in E_sel) + " |")
    print("|---|---|---|" + "---|" * len(E_sel))
    for r in dreg.RUNS:
        rl = load_rawlens(r["name"], split)
        if rl is None:
            continue
        for d in depths:
            row = []
            for E in E_sel:
                if E in rl["E"]:
                    row.append(fmt(rl["top1"][rl["E"].index(E), HALF.index(d)]))
                else:
                    row.append("-")
            print(f"| {r['name']} | {r['role']} | {d} | " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["matched", "transitions", "tables"])
    ap.add_argument("--ref_role", default="random-like")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    if args.what == "matched":
        rec = matched_epochs(args.ref_role)
        for n, files in rec["matched"].items():
            bad = [r["level_E"] for r in rec["meta"]["detail"][n] if not r["ok"]]
            print(f"{n:14s} matched files (common levels): {files}  out-of-tolerance levels E: {bad}")
        print("L*:", [f"{l:.3f}" for l in rec["meta"]["L_star"]], "common levels E:", rec["meta"]["common_levels"])
    elif args.what == "transitions":
        for n, files in transition_epochs().items():
            print(f"{n:14s} transition files: {files}")
    else:
        table_rawlens()
        table_sequence()


if __name__ == "__main__":
    main()
