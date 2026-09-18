"""Run table, epoch conventions and output paths for the post-hoc training-dynamics analysis
(docs/posthoc_training_dynamics_handoff.md, docs/training_dynamics.md).

Runs are `probes.registry` entries with complete per-epoch dump ladders (checkpoint-<e>-model.pth for every
e in 0..299).  The canonical r / p / ftb3b / ftbrho runs lost their ladders (pruned 2026-08-23, only 298-299
left) and enter only through their training-time logs (`dynamics.hist`).  Every dynamics entry zeroes the
softmax-invariant k-bias at load (`probes.registry.zero_k_bias_`).  All outputs go under DYN_ROOT; the
collaborator's dumps are read in place and never copied.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from probes import registry as preg

DYN_ROOT = Path(os.environ.get("DYN_ROOT", "/work/dlc2workfs3/schrodi-procedural/kempfe/dynamics"))
RES = DYN_ROOT / "results"
FIGS = DYN_ROOT / "figures"
LOGS = DYN_ROOT / "logs"
SUBSETS = DYN_ROOT / "subsets"
PROBES_SUBSETS = Path("/work/dlc2workfs3/schrodi-procedural/kempfe/probes/subsets")   # the frozen section-0e splits
STAGES = ("rawlens", "bypass", "funcchange", "grads", "hist", "lens", "probes", "smoke")


def _r(name, role, note=""):
    return dict(name=name, role=role, note=note)


# ------------------------------------------------------------------------------------------------- run table
RUNS = [
    _r("ftblrm_s0", "random-like", "plain timm init; proc-matched per-tensor lr; k-bias hand-patched after file 227"),
    _r("ftbqu_s0", "random-like", "random weights, q/k x2.177 in blocks 0-8; historical lens JSON only from epoch 189; init dump"),
    _r("ftbqu_s1", "random-like", "as ftbqu_s0; no historical lens JSON"),
    _r("ftbqu_s2", "random-like", "as ftbqu_s0; no historical lens JSON; log.txt lacks the epoch-169 line (requeue)"),
    _r("ftbrhos_s0", "random-like", "random, blocks 0-8 rho-matched to shuffled proc (75.1 % final); k-bias patched after file 271; update_freq 4"),
    _r("ftb4_s0", "proc-like", "plain proc init, all 12 blocks, from the kdyck-shuffled ckpt pr_6463456 (p used pr_6066174)"),
    _r("ftb4i_s1", "proc-like", "proc 0-7 (pr_6066174) + random 8-11; seed 0 of this job was pruned"),
    _r("ftb4i_s2", "proc-like", "proc 0-7 (pr_6066174) + random 8-11"),
    _r("ftb4is_s0", "proc-like", "proc 0-7 from the kdyck-shuffled ckpt + random 8-11; own arm, NOT an ftb4i seed"),
    _r("ftbqmlnvo_s0", "intermediate", "proc marginals rank-mapped into 0-8 (section-0e arm); init dump"),
    _r("ftbqmlnvo_s1", "intermediate", "as ftbqmlnvo_s0"),
    _r("ftbqmlnvo_s2", "intermediate", "as ftbqmlnvo_s0"),
    _r("ftbrhopl_s0", "rescaled", "checkpoint-free ftbrho lever + per-tensor lr multipliers (train_lr peaks 0.163), trained in bf16; "
                                  "block-11 weight norm ~9.6k; k-bias patched after files 270 and 299"),
    _r("ftbrhop_s0", "rescaled", "checkpoint-free ftbrho lever: random, blocks 9-11 proj/fc2 amplified to write ratio 1.4; init dump"),
    _r("ftbcomp1_s0", "rescaled", "random + proc block 0 copied, blocks 9-11 to rho 0.25; no historical lens JSON"),
    _r("ftbcomp1_s1", "rescaled", "as ftbcomp1_s0"),
    _r("ftbcomp1_s2", "rescaled", "as ftbcomp1_s0"),
]
HISTORY_RUNS = [f"{a}_s{s}" for a in ("r", "p", "ftb3b", "ftbrho") for s in range(3)]   # training-time logs only
ROLE = {r["name"]: r["role"] for r in RUNS}
ROLE.update({n: "history" for n in HISTORY_RUNS})
ROLE_ORDER = ["random-like", "intermediate", "proc-like", "rescaled", "history"]
ROLE_OF_ARM = {n.rsplit("_s", 1)[0]: role for n, role in ROLE.items()}
# init dumps whose job identity was verified (results/init_dumps/dump_cmds*.txt); ftb4i_s0.pth belongs to the
# pruned 29448854 s0 and must not be attached to ftb4is (29547835)
INIT_DUMPS = {"ftbqu_s0": "ftbqu_s0.pth", "ftbqmlnvo_s0": "ftbqmlnvo_s0.pth", "ftbrhop_s0": "ftbrhop_s0.pth"}
K_BIAS_PATCHES = {"ftblrm_s0": [227], "ftbrhos_s0": [271], "ftbrhopl_s0": [270, 299]}   # checkpoint-<e>.pth.orig_kbias

# ------------------------------------------------------------------------------------------------- epochs
# E = "after E epochs of training" (handoff numbering) <-> file checkpoint-(E-1)-model.pth (0-based epoch index)
EPOCHS_CORE = [1, 2, 3, 5, 10, 20, 40, 80, 150, 225, 300]
LENS_EPOCHS = [1, 3, 10, 40, 150, 300]              # tuned lens + probes (+ loss-matched epochs)


def file_index(E):
    return int(E) - 1


def epoch_of(idx):
    return int(idx) + 1


CORE_FILES = [file_index(E) for E in EPOCHS_CORE]
# every epoch over E = 1..20 (the expected transition window), the historical 5-epoch grid, and the core
DENSE_FILES = sorted(set(range(0, 20)) | set(range(4, 300, 5)) | set(CORE_FILES))
LENS_FILES = [file_index(E) for E in LENS_EPOCHS]
FP32_FILES = (297, 298, 299)                                        # full fp32 checkpoints survive for these
N_EPOCHS, WARMUP_EPOCHS, BASE_LR, MIN_LR, NITER_PER_EPOCH = 300, 50, 2e-3, 1e-6, 1281167 // 4096


def lr_epoch_means():
    """Per-epoch mean learning rate of the runs' schedule (utils.cosine_scheduler: 50 warmup epochs from 0 to
    2e-3, cosine to 1e-6 at epoch 300, per-iteration); ftbrhopl/ftblrm apply per-tensor multipliers on top."""
    w = WARMUP_EPOCHS * NITER_PER_EPOCH
    warm = np.linspace(0, BASE_LR, w)
    n = N_EPOCHS * NITER_PER_EPOCH - w
    cos = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * np.arange(n) / n))
    sched = np.concatenate([warm, cos])
    return sched.reshape(N_EPOCHS, NITER_PER_EPOCH).mean(1)


# ------------------------------------------------------------------------------------------------- entries
def entry(name, allow_history=False):
    """The run's registry entry as the dynamics stages use it: a copy flagged zero_k_bias=True (the section-0e
    entries of ftbqmlnvo / ftbrhos / ftblrm carry no flag, and the probes pipeline must keep loading them as is).
    The canonical r / p / ftb3b / ftbrho seeds (dumps 298-299 only) are reachable with allow_history=True."""
    if name not in ROLE or (name in HISTORY_RUNS and not allow_history):
        raise KeyError(f"{name} is not a dynamics run")
    return dict(preg.BY_NAME[name], zero_k_bias=True)


def run_dir(e):
    return preg.ckpt_path(e).parent


def available_files(e):
    """Sorted 0-based epoch indices whose fp16 dump exists in the run directory."""
    out = []
    for f in run_dir(e).glob("checkpoint-*-model.pth"):
        s = f.name[len("checkpoint-"):-len("-model.pth")]
        if s.isdigit():
            out.append(int(s))
    return sorted(out)


def full_ckpt(e, idx):
    """Path of the fp32 full checkpoint (with optimizer) for epoch index idx, or None if it was rotated away."""
    p = run_dir(e) / f"checkpoint-{int(idx)}.pth"
    return p if p.exists() else None


def init_dump(e):
    f = INIT_DUMPS.get(e["name"])
    if f is None:
        return None
    p = preg.RESULTS / "init_dumps" / f
    return p if p.exists() else None


def epoch_entry(e, idx):
    """Entry for the dump of epoch index idx (name `<run>@e<idx>`); passed to the stages directly, never
    registered in probes.registry.BY_NAME (whose flag state would otherwise depend on call order)."""
    return preg.with_epoch(e, idx)


def resolve_runs(spec="all", quiet=False):
    """'all' -> every dynamics run whose log ends at 299 and whose dump ladder exists; 'canonical' -> the 12
    canonical seeds (endpoint dumps 298-299 only); else a comma list of names (canonical names allowed)."""
    if spec == "canonical":
        names = list(HISTORY_RUNS)
    else:
        names = [r["name"] for r in RUNS] if spec in ("all", "", None) else [s for s in spec.split(",") if s]
    out = []
    for n in names:
        e = entry(n, allow_history=True)
        ok, why = preg.finished(e)
        if ok:
            out.append(e)
        elif spec in ("all", "", None):
            if not quiet:
                print(f"[dynamics] excluded {n}: {why}", flush=True)
        else:
            raise SystemExit(f"run {n}: {why}")
    return out


def _files_from(fname, name):
    p = RES / "hist" / fname
    if not p.exists():
        return []
    return [int(i) for i in json.load(open(p)).get(name, [])]


def matched_files(name):
    """Loss-matched epoch indices for a run (written by `dynamics.summarize matched` after rawlens); [] if absent."""
    return _files_from("matched_epochs.json", name)


def transition_files(name):
    """Epoch indices bracketing the run's confidence transitions (written by `dynamics.summarize transitions`)."""
    return _files_from("transition_epochs.json", name)


# ------------------------------------------------------------------------------------------------- io
def ensure_dirs():
    os.umask(0o002)
    for d in [FIGS, LOGS, SUBSETS, DYN_ROOT / "features"] + [RES / s for s in STAGES]:
        d.mkdir(parents=True, exist_ok=True)


def check_subsets():
    """The frozen ImageNet splits used here must be the section-0e ones (same sha256)."""
    for split in ("train", "val", "test"):
        a, b = preg.SUBSETS / f"imnet_{split}.json", PROBES_SUBSETS / f"imnet_{split}.json"
        if a.resolve() == b.resolve():
            continue
        assert a.exists() and b.exists(), (a, b)
        sa, sb = (json.load(open(x))["sha256"] for x in (a, b))
        assert sa == sb, f"imnet_{split} subset differs from the probes copy ({sa[:8]} vs {sb[:8]})"


def stamp():
    return dict(git=preg.git_commit(), date=datetime.now().isoformat(timespec="seconds"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def result_path(stage, name, idx=None, ext=".json", suffix=""):
    fn = name if idx is None else f"{name}@e{int(idx)}"
    return RES / stage / f"{fn}{suffix}{ext}"


if __name__ == "__main__":
    for r in RUNS:
        e = entry(r["name"])
        ok, why = preg.finished(e)
        files = available_files(e) if ok else []
        complete = files == list(range(300))
        print(f"{r['name']:14s} {r['role']:12s} {'OK ' if ok else 'NO '} {why:20s} dumps={len(files):3d}{' complete' if complete else ' INCOMPLETE'}"
              f" fp32={[i for i in FP32_FILES if full_ckpt(e, i)]} init={'yes' if init_dump(e) else 'no'} acc299={preg.final_test_acc(e)}")
