"""Verification of the progressive-unfreezing mechanism on the REAL trainer (DDP, fp16 GradScaler,
update_freq 2) -- port of layer-scale's analysis/unfreeze_check.py (docs/unfreeze_vit_handoff.md,
"Checks" steps 2-3, plus a post-run check).

Smoke mode (--smoke <dir>, the output of unfreeze/sbatch/smoke.sh: uf / baseA / baseB / ctrl run
dirs with snapshots/step_<t>.pt, t = optimizer steps done, t = 0 the init):

  snapshots  the uf snapshots carry the arm, T and the recorded unfreeze steps; u_k is re-derived
             with unfreeze.unfreeze_iters (never re-implemented here) and must equal the record
  frozen     t <= u_k => every tensor of chunk k is BIT-IDENTICAL to step_0; t > u_k => every
             tensor moved; the always-trainable tensors moved at every t >= 2 (the warmup's step
             0 has lr 0); every model tensor finite
  adam       optimizer state exists exactly for the unfrozen chunks (and the always groups) with
             step == the non-skipped optimizer steps in [u_k, t) (GradScaler skips a step on an
             overflow; the snapshots record which), absent for the frozen chunks; every moment
             finite; the saved groups carry chunk / group_name
  lr         every saved group lr == the recipe's cosine schedule at step t-1 x the chunk's ramp
  coverage   the snapshots reach the stop step and bracket every delayed unfreeze (t = u_k and
             t = u_k + 1 both present)
  timing     s / it of epochs 0-1 (the engine's "Total time" lines) of uf vs baseA vs baseB
  control    per-tensor relative deviation ||x - x_baseA|| / ||x_baseA|| (max over tensors) of
             ctrl (E = 0) and of baseB at every common snapshot: ctrl must not exceed the
             base-vs-base noise (bit-identical when the runs are deterministic)
  recipe     baseA's epoch-0 / 1 train_loss and test_acc1 vs schrodi's r_random / r_random_k4
             logs (the base this experiment pairs against): inside their 6-run envelope

Epoch mode (--run <dir>, a finished or running arm): from the per-epoch analysis dumps
checkpoint-<e>-model.pth, a chunk that is frozen for the whole of epochs 0..e is identical in
dump e and dump 0, and differs from dump 0 once it has trained; log.txt's unfreeze/* scalars match
the plan at each epoch's last step.

  python unfreeze/check.py --smoke /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/smoke/<job>
  python unfreeze/check.py --run /work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/runs/u4f-lin0.2/s0
Exits non-zero on any FAIL; --json writes the table.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import unfreeze as uf  # noqa: E402

BASE_ROOT = "/work/dlc2workfs3/schrodi-procedural/results/i100_playground"
BASE_RUNS = [f"{r}/s{s}" for r in ("r_random", "r_random_k4") for s in range(3)]
TIME_RE = re.compile(r"^Epoch: \[(\d+)\] Total time: \S+ \(([\d.]+) s / it\)")

ROWS = []


def row(check, ok, detail):
    ROWS.append(dict(check=check, ok=bool(ok), detail=detail))
    print(f"{'OK  ' if ok else 'FAIL'} {check:10s} {detail}", flush=True)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def snapshots(run_dir):
    out = {}
    for p in glob.glob(os.path.join(run_dir, "snapshots", "step_*.pt")):
        out[int(re.search(r"step_(\d+)\.pt$", p).group(1))] = p
    assert out, f"no snapshots/step_<t>.pt in {run_dir}"
    return dict(sorted(out.items()))


def lr_schedule(a, niter):
    import utils
    return utils.cosine_scheduler(a.lr, a.min_lr, a.epochs, niter,
                                  warmup_epochs=a.warmup_epochs, warmup_steps=a.warmup_steps)


def plan_for(a, model_sd):
    """The run's plan, rebuilt on a meta-free stand-in (names + shapes from the state dict)."""
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Module() for _ in range(1 + max(uf.param_chunk(n) or 0 for n in model_sd))])
            self._named = [(n, torch.nn.Parameter(v.float(), requires_grad=False))
                           for n, v in model_sd.items()]

        def named_parameters(self, *_, **__):
            return iter(self._named)
    niter = a.unfreeze_T // a.epochs
    return uf.UnfreezePlan.from_arm(Stub(), a.unfreeze_arm, a.unfreeze_T, niter), niter


def param_names(sd, a):
    """The parameters among the state-dict keys: the model's buffers (none in vanilla vit_small)
    would be refused by param_chunk, so keep the keys param_chunk accepts."""
    out = []
    for n in sd:
        try:
            uf.param_chunk(n)
            out.append(n)
        except KeyError:
            pass
    return out


# --------------------------------------------------------------------------------------
# smoke mode
# --------------------------------------------------------------------------------------
def check_uf(run_dir, stop):
    snaps = snapshots(run_dir)
    s0 = load(snaps[0])
    a = s0["args"]
    init = s0["model"]
    names = param_names(init, a)
    plan, niter = plan_for(a, {n: init[n] for n in names})
    us = plan.iters
    ok = (tuple(a.unfreeze_iters) == us and a.unfreeze_T == a.epochs * niter
          and all(load(p)["args"].unfreeze_arm == a.unfreeze_arm for p in list(snaps.values())[1:3]))
    row("snapshots", ok, f"arm {a.unfreeze_arm}, T {a.unfreeze_T}, u {us} (recorded "
        f"{tuple(a.unfreeze_iters)}), ramp {plan.warmup}, {len(snaps)} snapshots t={list(snaps)}")
    chunk_of = {n: k for k, ns in enumerate(plan.chunk_names) for n in ns}
    lr_s = lr_schedule(a, niter)
    frozen_ok = adam_ok = lr_ok = finite_ok = True
    notes = []
    for t, p in snaps.items():
        if t == 0:
            continue
        s = load(p)
        sd, osd, skipped = s["model"], s["optimizer"], set(s["skipped"])
        assert s["t"] == t
        finite_ok &= all(torch.isfinite(sd[n]).all().item() for n in names)
        for n in names:
            k = chunk_of.get(n)
            same = torch.equal(sd[n], init[n])
            if k is not None and t <= us[k]:
                if not same:
                    frozen_ok = False
                    notes.append(f"t={t} {n} moved while frozen (u={us[k]})")
            elif (k is None and t >= 2) or (k is not None and t > max(us[k], 1)):
                if same:
                    frozen_ok = False
                    notes.append(f"t={t} {n} did not move")
        # adam: the state dict's params are indices in group order
        groups = osd["param_groups"]
        if not all("chunk" in g and "group_name" in g for g in groups) or len(groups) != 2 * plan.chunks + 2:
            adam_ok = False
            notes.append(f"t={t}: group layout {[g.get('group_name') for g in groups]}")
        for g in groups:
            k = g["chunk"]
            u = 0 if k is None else us[k]
            want = sum(1 for it in range(u, t) if it not in skipped)
            for idx in g["params"]:
                st = osd["state"].get(idx)
                if t <= u:
                    if st:
                        adam_ok = False
                        notes.append(f"t={t} {g['group_name']}: state while frozen")
                        break
                else:
                    if not st or int(st["step"]) != want:
                        adam_ok = False
                        notes.append(f"t={t} {g['group_name']}: step "
                                     f"{None if not st else int(st['step'])} != {want}")
                        break
                    if not (torch.isfinite(st["exp_avg"]).all() and torch.isfinite(st["exp_avg_sq"]).all()):
                        finite_ok = False
            # lr of the step just taken (it = t-1)
            m = 1.0 if k is None else uf.lr_mult(t - 1, u, plan.warmup)
            want_lr = lr_s[t - 1] * g.get("lr_scale", 1) * m
            if not np.isclose(g["lr"], want_lr, rtol=1e-12, atol=0):
                lr_ok = False
                notes.append(f"t={t} {g['group_name']}: lr {g['lr']} != {want_lr}")
    last = load(snaps[max(snaps)])
    row("frozen", frozen_ok, "chunks bit-identical to step_0 while t <= u_k, moved after; "
        "always tensors moved from t=2")
    row("adam", adam_ok, f"state only for unfrozen chunks, step = non-skipped steps since u_k; "
        f"skipped steps {last['skipped']}")
    row("finite", finite_ok, "every model tensor and Adam moment finite")
    row("lr", lr_ok, "group lr == cosine schedule[t-1] x lr_mult (rtol 1e-12)")
    # the last snapshot: the stop step itself (engine snapshots it), or for smokes that predate
    # that the last scheduled one before the stop
    every = int(getattr(a, "smoke_snapshot_every", 0) or 0)
    sched = {int(x) for x in str(a.smoke_snapshot_steps or "").split(",") if x.strip()}
    sched |= set(range(every, stop + 1, every)) if every else set()
    cov = max(snaps) in (stop, max(t for t in sched if t <= stop)) and \
        all(u in snaps and u + 1 in snaps for u in us if u > 0)
    row("coverage", cov, f"last snapshot t={max(snaps)} (stop {stop}); brackets u_k>0: "
        f"{[(u, u in snaps, u + 1 in snaps) for u in us if u > 0]}")
    for n in notes[:12]:
        print("     ", n)
    return snaps


def epoch_times(log):
    out = {}
    if os.path.exists(log):
        for line in open(log):
            m = TIME_RE.match(line.strip())
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def rel_dev(a_sd, b_sd, names):
    return max(((a_sd[n].float() - b_sd[n].float()).norm() / b_sd[n].float().norm().clamp_min(1e-30)).item()
               for n in names)


def check_control(smoke):
    sa, sb, sc = (snapshots(os.path.join(smoke, x)) for x in ("baseA", "baseB", "ctrl"))
    common = sorted(set(sa) & set(sb) & set(sc))
    names = param_names(load(sa[0])["model"], None)
    ok, lines = True, []
    for t in common:
        A = load(sa[t])["model"]
        dB, dC = rel_dev(load(sb[t])["model"], A, names), rel_dev(load(sc[t])["model"], A, names)
        lines.append((t, dB, dC))
        # ctrl may not deviate from baseA beyond the base-vs-base floor (x2 + fp32 eps slack)
        ok &= dC <= 2 * dB + 1e-6
    row("control", ok, "t: baseB-vs-baseA / ctrl-vs-baseA max rel dev: "
        + ", ".join(f"{t}: {b:.2e} / {c:.2e}" for t, b, c in lines))
    return lines


def check_timing(smoke):
    times = {x: epoch_times(f"{smoke}.{x}.log") for x in ("uf", "baseA", "baseB", "ctrl")}
    base = [v[1] for x, v in times.items() if x.startswith("base") and 1 in v]
    ok = 1 in times["uf"] and base and times["uf"][1] <= 1.05 * max(base)
    row("timing", ok, "s/it epoch 0 / 1: " + ", ".join(
        f"{x} {v.get(0, float('nan')):.4f} / {v.get(1, float('nan')):.4f}" for x, v in times.items()))
    return times


def read_log(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d["epoch"]] = d   # last wins (a requeue repeats an epoch)
    return out


def check_recipe(smoke):
    mine = read_log(os.path.join(smoke, "baseA", "log.txt"))
    ok, parts = True, []
    for e in (0, 1):
        for key in ("train_loss", "test_acc1"):
            ref = [read_log(os.path.join(BASE_ROOT, r, "log.txt"))[e][key] for r in BASE_RUNS]
            lo, hi = min(ref), max(ref)
            pad = 0.5 * (hi - lo)
            v = mine[e][key]
            inside = lo - pad <= v <= hi + pad
            ok &= inside
            parts.append(f"ep{e} {key} {v:.4f} vs [{lo:.4f}, {hi:.4f}] "
                         f"(s0 {ref[0]:.4f}, k4 s0 {ref[3]:.4f}){'' if inside else ' OUTSIDE'}")
    row("recipe", ok, "baseA vs r_random / r_random_k4 s0-2: " + "; ".join(parts))


# --------------------------------------------------------------------------------------
# epoch mode
# --------------------------------------------------------------------------------------
def check_run(run_dir):
    dumps = {}
    for p in glob.glob(os.path.join(run_dir, "checkpoint-*-model.pth")):
        dumps[int(re.search(r"checkpoint-(\d+)-model\.pth$", p).group(1))] = p
    full = sorted(glob.glob(os.path.join(run_dir, "checkpoint-[0-9]*.pth")))
    full = [p for p in full if re.search(r"checkpoint-\d+\.pth$", p)]
    assert dumps and full, f"no dumps / checkpoints in {run_dir}"
    a = load(full[-1])["args"]
    d0 = load(dumps[0])
    names = param_names(d0, a)
    plan, niter = plan_for(a, {n: d0[n] for n in names})
    us = plan.iters
    row("snapshots", tuple(a.unfreeze_iters) == us,
        f"arm {a.unfreeze_arm}, T {a.unfreeze_T}, u {us} = epochs {[u / niter for u in us]}, "
        f"{len(dumps)} dumps (epochs {min(dumps)}-{max(dumps)})")
    ok, notes = True, []
    delayed = [k for k in range(plan.chunks) if us[k] >= niter]   # frozen through epoch 0
    for e in sorted(dumps):
        # only the dumps around each unfreeze (and the last) are needed
        if not any(abs(e - (us[k] // niter)) <= 1 for k in delayed) and e != max(dumps):
            continue
        d = load(dumps[e])
        for k in delayed:
            frozen_through = (e + 1) * niter <= us[k]
            for n in plan.chunk_names[k]:
                same = torch.equal(d[n], d0[n])
                if frozen_through != same:
                    ok = False
                    notes.append(f"epoch {e} chunk {k} (u {us[k]}) {n}: same={same}")
                    break
    row("frozen", ok, f"delayed chunks {delayed}: identical to dump 0 while frozen, moved after "
        f"(dumps are fp16)")
    for n in notes[:8]:
        print("     ", n)
    log = read_log(os.path.join(run_dir, "log.txt"))
    lr_s = lr_schedule(a, niter)
    ok = True
    for e, d in log.items():
        last = (e + 1) * niter - 1
        want = plan.wandb_scalars(last, lr_s[last])
        ok &= all(np.isclose(d.get(k, np.nan), v) for k, v in want.items())
    row("log", ok, f"unfreeze/* scalars in log.txt match the plan at every epoch's last step "
        f"({len(log)} epochs, last {max(log)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", help="smoke job dir (unfreeze/sbatch/smoke.sh)")
    ap.add_argument("--run", help="an arm's run dir (epoch mode)")
    ap.add_argument("--stop", type=int, default=650)
    ap.add_argument("--json")
    args = ap.parse_args()
    extra = {}
    if args.smoke:
        check_uf(os.path.join(args.smoke, "uf"), args.stop)
        extra["control"] = check_control(args.smoke)
        extra["timing"] = check_timing(args.smoke)
        check_recipe(args.smoke)
    if args.run:
        check_run(args.run)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(rows=ROWS, **extra), f, indent=1)
    bad = [r["check"] for r in ROWS if not r["ok"]]
    print("ALL OK" if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
