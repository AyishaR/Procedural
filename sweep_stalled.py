#!/usr/bin/env python3
"""Find training runs that are INCOMPLETE and have NO job in the queue, and optionally resume them.

Jobs die silently here in two ways: preemption without --requeue (the job simply vanishes), and
crashes that exhaust the in-job retry loop. Either way nothing is left to finish the run, and the
only symptom is a log.txt that stops growing. On 2026-08-31 this had stranded nine seeds, three of
them 5-6 epochs from completion and idle for over a day.

Default is a DRY RUN. Pass --submit to actually resubmit.

Safeguards:
  * per-(arm,seed) attempt counter in .sweep_state.json -- a seed that keeps dying is left alone
    after --max-tries resubmissions, so a deterministically diverging run (ftb11s s0 reproduces a
    NaN at epoch 35 from scratch) cannot be resubmitted forever.
  * BLACKLIST for seeds known to be unrecoverable.
  * resubmits with SLURM_ID pinned to the results directory, so the run resumes in place rather
    than starting a new one, and with --requeue so the next preemption does not kill it again.
  * sizes --time from the epochs actually remaining (see time_limit_for), so a run that is a
    few epochs from done asks for a couple of hours and gets backfilled instead of waiting
    days for a full 23:30 window.

Usage:
    python sweep_stalled.py                 # report
    python sweep_stalled.py --submit        # report + resume
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "imnet_base"
LOGS = ROOT / "logs"
STATE = ROOT / ".sweep_state.json"
TARGET_EPOCHS = 300
PARTITION = "alldlc2_gpu-h200"

# Sizing the --time request from the work actually left, instead of always asking for the
# partition maximum. A 23:30 request for a run that is one epoch from done both delays that
# job (it can only start when a full 23.5 h window opens) and blocks backfill for everything
# behind it. Measured 2026-09-03: right-sizing thirteen such reservations pulled ftbqmlnvo
# from a Sep 6 estimate to Sep 3, and two 1-2 epoch jobs went from "Sep 3" to finished within
# two hours. EP_PER_HOUR is deliberately pessimistic -- observed 8.9-13.7 on 4xH200, and a
# job killed by an under-sized limit costs a whole queue wait to recover.
EP_PER_HOUR = 8.0
STARTUP_H = 1.5          # checkpoint load + the init/analysis path before epoch 1
MAX_TIME_H = 23.5        # partition maximum (MaxTime is 1-00:00:00, scripts ask 23:29:59)


def time_limit_for(epochs_left):
    """--time value for a run with `epochs_left` epochs to go.

    Returns None when the work needs (near) the full window, so the caller leaves the
    script's own #SBATCH --time in place rather than asking for something tighter.
    """
    need = epochs_left / EP_PER_HOUR + STARTUP_H
    if need >= MAX_TIME_H - 1.0:
        return None
    hours = min(MAX_TIME_H, need * 1.15 + 0.5)      # 15% headroom on the rate estimate
    h = int(hours)
    return f"{h:02d}:{int(round((hours - h) * 60)):02d}:00"

# (arm, seed) pairs that must never be resubmitted, with the reason.
BLACKLIST = {
    ("ftb11s", "s0"): "diverges deterministically at ~ep35 (reproduced twice from scratch)",
    ("ftb11s", "s1"): "diverged at ep235, all 12 in-job retries exhausted",
    # Parked 2026-09-05 (docs 0d.8): superseded by ftbqks / ftbqmlnvog / ftbrhos. Checkpoints kept;
    # resume by hand with --export=SLURM_ID=<id>,SEED=<s> if ever wanted again.
    ("ftbqm1dv", "s0"): "parked 2026-09-05, low value (clean re-run of the qk_v variant)",
    ("ftbqm1dv", "s1"): "parked 2026-09-05, low value",
    ("ftbqm1dv", "s2"): "parked 2026-09-05, low value",
    ("ftbqm1dqk", "s0"): "parked 2026-09-05, low value (clean re-run of the qk_only variant)",
    ("ftbqm1dqk", "s1"): "parked 2026-09-05, low value",
    ("ftbqm1dqk", "s2"): "parked 2026-09-05, low value",
}


def queued_arms():
    """Arms with a job in the queue.

    Fails LOUDLY rather than returning an empty set: if squeue errors transiently and we treat
    the queue as empty, every incomplete run looks stalled and the sweep resubmits all of them.
    """
    r = subprocess.run(["squeue", "-u", "schrodi", "-h", "-o", "%j"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"squeue failed (rc={r.returncode}): {r.stderr.strip()[:200]}\n"
                 f"Refusing to run: an unreadable queue would look like an empty one.")
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def slurmid_to_arm():
    """Map a results-dir SLURM_ID to its arm name via 'Running with ID' in the job logs."""
    m = {}
    for f in LOGS.glob("ft_*_*.out"):
        mt = re.match(r"ft_\d+_(.+)\.out$", f.name)
        if not mt:
            continue
        try:
            with open(f, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if "Running with ID" in line:
                        m.setdefault(line.split()[-1].strip(), mt.group(1))
                        break
                    if i > 400:
                        break
        except OSError:
            pass
    return m


def max_epoch_reached(logfile):
    """Epochs completed = highest "epoch" in log.txt, plus one. See the call site."""
    hi = -1
    try:
        with open(logfile, errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line).get("epoch")
                except json.JSONDecodeError:
                    continue
                if isinstance(e, int) and e > hi:
                    hi = e
    except OSError:
        return 0
    return hi + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--max-tries", type=int, default=3)
    ap.add_argument("--max-submit", type=int, default=6,
                    help="refuse to act if more than this many runs look stalled at once")
    ap.add_argument("--max-age-days", type=float, default=7.0,
                    help="ignore runs whose log.txt has not been touched in this long; they are\n                          abandoned duplicates, not stalled runs (several arms were re-run under\n                          a fresh SLURM_ID and the original left behind).")
    a = ap.parse_args()

    inq, id2arm = queued_arms(), slurmid_to_arm()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    stalled = []
    for d in sorted(RES.glob("results_IMNET_BASE_*")):
        if d.name.endswith("contaminated") or "diverged" in d.name:
            continue
        sid = d.name.rsplit("_", 1)[1]
        arm = id2arm.get(sid)
        if arm is None or arm in inq:          # unknown, or something is already running for it
            continue
        for sd in sorted(d.glob("s[0-9]")):
            lf = sd / "log.txt"
            if not lf.exists():
                continue
            # Progress is the highest epoch REACHED, not the line count: log.txt loses lines
            # when a job is preempted mid-write, so several completed runs (ftbvu, ftbqu, ftb7e
            # on 2026-09-03) had 294-299 lines but had reached epoch 299. Counting lines made
            # this sweep resubmit finished runs -- three times for ftbvu -- and each resubmit
            # ran ~1 min, exited 0, and was then recorded FAILED by the script's
            # "Runtime too short. Stop chain." guard, which looks exactly like a crash loop.
            n = max_epoch_reached(lf)
            if n >= TARGET_EPOCHS:
                continue
            age = (time.time() - lf.stat().st_mtime) / 86400
            stalled.append((arm, sid, sd.name, n, age))

    if not stalled:
        print("no stalled runs: every incomplete run has a job in the queue")
        return
    live = [r for r in stalled if r[4] <= a.max_age_days and (r[0], r[2]) not in BLACKLIST]
    if a.submit and len(live) > a.max_submit:
        sys.exit(f"{len(live)} runs look stalled at once (> --max-submit {a.max_submit}). "
                 f"That usually means the queue was misread, not that everything died. "
                 f"Re-run without --submit and check.")
    stalled.sort(key=lambda r: r[3] * -1)          # nearest to done first
    print(f"{'arm':12}{'slurm_id':>10}{'seed':>5}{'epochs':>8}{'short':>7}{'age_d':>7}   action")
    for arm, sid, seed, n, age in stalled:
        key = f"{arm}|{seed}"
        tries = state.get(key, 0)
        if age > a.max_age_days:
            act = f"skip (idle {age:.0f}d > {a.max_age_days:.0f}d -- abandoned, not stalled)"
        elif (arm, seed) in BLACKLIST:
            act = f"SKIP (blacklist: {BLACKLIST[(arm, seed)]})"
        elif tries >= a.max_tries:
            act = f"SKIP ({tries} resubmits already; investigate before another)"
        elif not a.submit:
            act = (f"would resume (try {tries+1}/{a.max_tries}, "
                   f"--time {time_limit_for(TARGET_EPOCHS - n) or 'script default'}) -- pass --submit")
        else:
            script = ROOT / "vitbase_runs" / f"run_train_{arm}.sh"
            if not script.exists():
                act = f"SKIP (no {script.name})"
            else:
                tl = time_limit_for(TARGET_EPOCHS - n)
                cmd = ["sbatch", "--parsable", "--requeue",
                       f"--partition={PARTITION}",
                       f"--export=SLURM_ID={sid},SEED={seed[1:]}"]
                if tl:
                    cmd.append(f"--time={tl}")
                cmd.append(str(script))
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    act = f"SUBMIT FAILED: {r.stderr.strip()[:60]}"
                else:
                    state[key] = tries + 1
                    act = (f"resumed -> job {r.stdout.strip()} (try {tries+1}, "
                           f"--time {tl or 'script default'})")
        print(f"{arm:12}{sid:>10}{seed:>5}{n:>8}{TARGET_EPOCHS-n:>7}{age:>7.1f}   {act}")
    if a.submit:
        STATE.write_text(json.dumps(state, indent=1))


if __name__ == "__main__":
    main()
