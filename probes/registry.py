"""Model registry, output paths and the final-checkpoint gate for the layerwise-probe / lens experiment.

Everything the pipeline writes goes under PROBE_ROOT (default: the kempfe/probes folder on the shared workspace),
never into the repo or the collaborator's results tree, which is only read.  A model is evaluated only if its
epoch-299 weight file exists AND the run's log.txt ends at epoch 299 (`finished`).  No init dumps, no
checkpoint-best, no intermediate epochs.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("PROC_RESULTS", ROOT / "results"))          # read-only
PROBE_ROOT = Path(os.environ.get("PROBE_ROOT", "/work/dlc2workfs3/schrodi-procedural/kempfe/probes"))
FEATURES = PROBE_ROOT / "features"
DATA = PROBE_ROOT / "data"
RES = PROBE_ROOT / "results"
FIGS = PROBE_ROOT / "figures"
SUBSETS = PROBE_ROOT / "subsets"
IMNET = Path(os.environ.get("IMNET_PATH", "/work/dlcsmall2/schrodi-imagenet"))   # SSD copy, read-only
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "/data/datasets"))

DEPTHS = list(range(13))          # 0 = patch+pos embedding, d = output of block d-1
DEPTH_LABELS = ["emb"] + [f"b{i}" for i in range(12)]

ARM_ORDER = ["r", "r0ls", "r0mim", "ftblrm", "ftbrhos", "ftb4o", "ftbrho", "ftb3b", "ftb3i", "ftbqmlnvo", "ftbanapermb7i", "ftbc7lm", "p", "pd98"]
ARM_LABEL = {
    "r": "random",
    "r0ls": "random + LayerScale",
    "r0mim": "random + mimetic init",
    "ftblrm": "random, Adam steps matched to proc 0-8",
    "ftbrhos": "random, 0-8 rho-matched to shuffled proc",
    "ftb4o": "random, 0-7 upscaled to proc rho",
    "ftbrho": "random, 9-11 upscaled (rho 1.4)",
    "ftb3b": "random, 9-11 rho-matched to proc",
    "ftb3i": "proc 0-8, random 9-11",
    "ftbqmlnvo": "proc marginals rank-mapped 0-8",
    "ftbanapermb7i": "statistics-only profile 0-7, checkpoint-free",
    "ftbc7lm": "analytic profile 0-7, checkpoint-free",
    "p": "proc (all 12)",
    "pd98": "proc k-Dyck shuffled D98 (all 12)",
}
ARM_COLOR = {
    "r": "0.15", "r0ls": "#8c564b", "r0mim": "#9467bd", "ftblrm": "#1b9e77", "ftbrhos": "#a6761d", "ftb4o": "#66a61e",
    "ftbrho": "#e6ab02", "ftb3b": "#d95f02", "ftb3i": "#7570b3", "ftbqmlnvo": "#e7298a", "ftbanapermb7i": "#795548", "ftbc7lm": "#e31a1c", "p": "#1f78b4", "pd98": "#a6cee3",
}
# accuracies from log.txt (last epoch) are read at runtime; these are only for labelling
_IB = "imnet_base/results_IMNET_BASE_{sid}/s{seed}/checkpoint-299-model.pth"


def _e(arm, seed, sid, **build):
    """build: extra timm kwargs the architecture needs, e.g. init_values for a LayerScale run."""
    return dict(name=f"{arm}_s{seed}", arm=arm, seed=seed, slurm_id=str(sid), ckpt=_IB.format(sid=sid, seed=seed),
                build=build)


MODELS = [
    *[_e("r", s, 29384839) for s in range(3)],
    # LayerScale: the checkpoint carries blocks.*.ls{1,2}.gamma, so the model must be built with them
    *[_e("r0ls", s, 30054092, init_values=1e-4) for s in range(3)],
    *[_e("r0mim", s, 30060920) for s in range(3)],     # random + mimetic init baseline
    *[_e("p", s, 29377576) for s in range(3)],
    _e("ftbrho", 0, 29453944), _e("ftbrho", 1, 29461252), _e("ftbrho", 2, 29461253),
    _e("ftb3b", 0, 29388202), _e("ftb3b", 1, 29406778), _e("ftb3b", 2, 29406779),
    *[_e("ftb3i", s, 29469072) for s in range(3)],
    *[_e("ftbqmlnvo", s, 29523316) for s in range(3)],
    _e("ftb4o", 0, 29451652),
    _e("ftbrhos", 0, 29538140),
    _e("ftblrm", 0, 29545846),
    *[_e("ftbanapermb7i", s, 29737095) for s in range(3)],   # statistics-only profile
    *[_e("ftbc7lm", s, 29768863) for s in range(3)],        # checkpoint-free analytic profile on blocks 0-7 (bf16 AMP), 80.48
    _e("pd98", 0, 29547831),           # full proc init from k-Dyck shuffled D98 (pr_vitb_ksd/pr_6463456), 80.21
]
BY_NAME = {m["name"]: m for m in MODELS}


def ckpt_path(entry):
    return RESULTS / entry["ckpt"]


def log_path(entry):
    return ckpt_path(entry).parent / "log.txt"


def _log_lines(entry):
    p = log_path(entry)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def final_test_acc(entry):
    """Last-epoch (299) test top-1 from log.txt, the repo's authoritative accuracy source (never wandb)."""
    for rec in reversed(_log_lines(entry)):
        if rec.get("epoch") == 299 and rec.get("test_acc1") is not None:
            return float(rec["test_acc1"])
    return None


_ARM_TRUTH = None


def arm_truth():
    global _ARM_TRUTH
    if _ARM_TRUTH is None:
        p = RESULTS / "init_dumps" / "arm_truth.json"
        _ARM_TRUTH = json.load(open(p)) if p.exists() else []
    return _ARM_TRUTH


def finished(entry):
    """(ok, reason).  ok only if the epoch-299 weights exist and log.txt ends at epoch 299 (and arm_truth agrees)."""
    if not ckpt_path(entry).exists():
        return False, "no checkpoint-299-model.pth"
    lines = _log_lines(entry)
    if not lines:
        return False, "no log.txt"
    last = lines[-1].get("epoch")
    if last != 299:
        return False, f"log.txt ends at epoch {last}"
    for rec in arm_truth():
        if str(rec.get("slurm_id")) == entry["slurm_id"] and int(rec.get("seed", -1)) == entry["seed"]:
            if rec.get("max_epoch") != 299:
                return False, f"arm_truth max_epoch={rec.get('max_epoch')}"
            break
    return True, "ok"


def resolve_models(spec, quiet=False):
    """'all' -> every finished model (excluded ones are printed); explicit names must all pass the gate."""
    if spec in ("all", "", None):
        out = []
        for m in MODELS:
            ok, why = finished(m)
            if ok:
                out.append(m)
            elif not quiet:
                print(f"[registry] excluded {m['name']}: {why}", file=sys.stderr, flush=True)
        return out
    out = []
    for name in [s for s in spec.split(",") if s]:
        if name not in BY_NAME:
            raise SystemExit(f"unknown model {name!r}; known: {sorted(BY_NAME)}")
        ok, why = finished(BY_NAME[name])
        if not ok:
            raise SystemExit(f"model {name} has not finished training ({why}); only final checkpoints are evaluated")
        out.append(BY_NAME[name])
    return out


def build_args(extra=()):
    """The repo's own argparse namespace with the flags every analysis script uses (defaults elsewhere)."""
    sys.path.insert(0, str(ROOT))
    import main as M  # noqa: E402  (repo entry point; only get_args_parser is used)
    args = M.get_args_parser().parse_args(
        ["--model", "vit_base", "--data_set", "IMNET", "--data_path", str(IMNET), "--input_size", "224", *extra])
    args.nb_classes = 1000
    return args


def load_model(entry, device="cuda"):
    """ViT-B/16 with the run's epoch-299 weights; asserts the state dict matches the model exactly."""
    import torch
    sys.path.insert(0, str(ROOT))
    import models.vision_transformer  # noqa: F401  (registers vit_base with timm)
    import utils
    args = build_args()
    build = entry.get("build") or {}
    if build:                      # timm kwargs the architecture needs (e.g. LayerScale's init_values)
        import timm
        model = timm.create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                                  drop_path_rate=args.drop_path, **build)
    else:
        model = utils.build_model(args)
    sd = torch.load(ckpt_path(entry), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k: v.float() for k, v in sd.items() if hasattr(v, "float")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (entry["name"], missing[:5], unexpected[:5])
    return model.eval().to(device)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def ensure_dirs():
    os.umask(0o002)
    for d in (FEATURES, DATA, RES / "probes", RES / "lens", RES / "dense", FIGS, SUBSETS, PROBE_ROOT / "logs"):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for m in MODELS:
        ok, why = finished(m)
        acc = final_test_acc(m) if ok else None
        print(f"{m['name']:14s} {'OK ' if ok else 'NO '} {why:28s} acc299={acc}")
