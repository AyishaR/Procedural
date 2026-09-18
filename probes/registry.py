"""Model registry, output paths and the final-checkpoint gate for the layerwise-probe / lens experiment.

Everything the pipeline writes goes under PROBE_ROOT (default: the kempfe/probes folder on the shared workspace),
never into the repo or the collaborator's results tree, which is only read.  A model is evaluated only if its
epoch-299 weight file exists AND the run's log.txt ends at epoch 299 (`finished`).  No init dumps, no
checkpoint-best.  Intermediate epochs of FINISHED runs are reachable for the training-dynamics analysis through
`with_epoch` / names of the form `<arm>_s<seed>@e<file index>` (file index = 0-based epoch after which the dump
was written); the `dynamics` package is the only consumer.
"""
import json
import os
import re
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

ARM_ORDER = ["r", "ftblrm", "ftbrhos", "ftb4o", "ftbrho", "ftb3b", "ftb3i", "ftbqmlnvo", "p",
             # dynamics-only arms (complete per-epoch ladders; see dynamics/registry.py)
             "ftbqu", "ftbrhop", "ftbrhopl", "ftbcomp1", "ftb4i", "ftb4is", "ftb4"]
ARM_LABEL = {
    "r": "random",
    "ftblrm": "random, Adam steps matched to proc 0-8",
    "ftbrhos": "random, 0-8 rho-matched to shuffled proc",
    "ftb4o": "random, 0-7 upscaled to proc rho",
    "ftbrho": "random, 9-11 upscaled (rho 1.4)",
    "ftb3b": "random, 9-11 rho-matched to proc",
    "ftb3i": "proc 0-8, random 9-11",
    "ftbqmlnvo": "proc marginals rank-mapped 0-8",
    "p": "proc (all 12)",
    "ftbqu": "random, q/k x2.18 in 0-8",
    "ftbrhop": "random, 9-11 to rho 1.4 (analytic profile)",
    "ftbrhopl": "ftbrhop + per-tensor lr, bf16",
    "ftbcomp1": "random + proc block 0, 9-11 to rho 0.25",
    "ftb4i": "proc 0-7, random 8-11",
    "ftb4is": "proc 0-7 (kdyck-shuffled ckpt), random 8-11",
    "ftb4": "proc (all 12, kdyck-shuffled ckpt)",
}
ARM_COLOR = {
    "r": "0.15", "ftblrm": "#1b9e77", "ftbrhos": "#a6761d", "ftb4o": "#66a61e",
    "ftbrho": "#e6ab02", "ftb3b": "#d95f02", "ftb3i": "#7570b3", "ftbqmlnvo": "#e7298a", "p": "#1f78b4",
    "ftbqu": "#969696", "ftbrhop": "#ff7f00", "ftbrhopl": "#b2182b", "ftbcomp1": "#fb9a99",
    "ftb4i": "#6baed6", "ftb4is": "#9ecae1", "ftb4": "#08519c",
}
# accuracies from log.txt (last epoch) are read at runtime; these are only for labelling
_IB = "imnet_base/results_IMNET_BASE_{sid}/s{seed}/checkpoint-{epoch}-model.pth"


def _e(arm, seed, sid, epoch=299, **flags):
    d = dict(name=f"{arm}_s{seed}", arm=arm, seed=seed, slurm_id=str(sid), epoch=int(epoch),
             ckpt=_IB.format(sid=sid, seed=seed, epoch=int(epoch)))
    d.update(flags)
    return d


def with_epoch(entry, epoch):
    """Copy of an entry pointing at the dump written after 0-based epoch `epoch` (checkpoint-<epoch>-model.pth).

    The name becomes `<base>@e<epoch>`; `base` keeps the run name so results can be grouped per run."""
    base = entry.get("base", entry["name"])
    return dict(entry, name=f"{base}@e{int(epoch)}", base=base, epoch=int(epoch),
                ckpt=_IB.format(sid=entry["slurm_id"], seed=entry["seed"], epoch=int(epoch)))


MODELS = [
    *[_e("r", s, 29384839) for s in range(3)],
    *[_e("p", s, 29377576) for s in range(3)],
    _e("ftbrho", 0, 29453944), _e("ftbrho", 1, 29461252), _e("ftbrho", 2, 29461253),
    _e("ftb3b", 0, 29388202), _e("ftb3b", 1, 29406778), _e("ftb3b", 2, 29406779),
    *[_e("ftb3i", s, 29469072) for s in range(3)],
    *[_e("ftbqmlnvo", s, 29523316) for s in range(3)],
    _e("ftb4o", 0, 29451652),
    _e("ftbrhos", 0, 29538140),
    _e("ftblrm", 0, 29545846),
]
# Runs used only by the training-dynamics analysis (complete 0..299 dump ladders; not part of the section-0e
# "all" set).  Every dynamics entry zeroes the softmax-invariant k-bias at load (see zero_k_bias_).
DYN_MODELS = [
    *[_e("ftbqu", s, 29518360, dyn_only=True, zero_k_bias=True) for s in range(3)],
    _e("ftb4", 0, 29547831, dyn_only=True, zero_k_bias=True),
    _e("ftb4i", 1, 29448854, dyn_only=True, zero_k_bias=True),
    _e("ftb4i", 2, 29448854, dyn_only=True, zero_k_bias=True),
    _e("ftb4is", 0, 29547835, dyn_only=True, zero_k_bias=True),
    _e("ftbrhopl", 0, 29632673, dyn_only=True, zero_k_bias=True),
    _e("ftbrhop", 0, 29626634, dyn_only=True, zero_k_bias=True),
    *[_e("ftbcomp1", s, 29498148, dyn_only=True, zero_k_bias=True) for s in range(3)],
]
BY_NAME = {m["name"]: m for m in MODELS + DYN_MODELS}
_EPOCH_NAME = re.compile(r"^(.+)@e(\d+)$")


def ckpt_path(entry):
    return RESULTS / entry["ckpt"]


def log_path(entry):
    """The run's log.txt (next to the checkpoint, or the entry's explicit `log` for init dumps stored elsewhere)."""
    if "log" in entry:
        return RESULTS / entry["log"]
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
    """Test top-1 at the entry's epoch (299 for final entries) from log.txt, the repo's authoritative accuracy
    source (never wandb).  None when that epoch was not evaluated (vit_base evaluates every 5th epoch only)."""
    want = int(entry.get("epoch", 299))
    for rec in reversed(_log_lines(entry)):
        if rec.get("epoch") == want and rec.get("test_acc1") is not None:
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
        return False, f"no {Path(entry['ckpt']).name}"
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
            m = _EPOCH_NAME.match(name)
            if m and m.group(1) in BY_NAME:
                BY_NAME[name] = with_epoch(BY_NAME[m.group(1)], int(m.group(2)))   # extract_features looks it up by name
            else:
                raise SystemExit(f"unknown model {name!r}; known: {sorted(BY_NAME)}")
        ok, why = finished(BY_NAME[name])
        if not ok:
            raise SystemExit(f"model {name}: run not finished or dump missing ({why}); only finished runs are evaluated")
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


def new_model():
    """An untrained ViT-B/16 exactly as the training script builds it (timm registry, 1000 classes)."""
    sys.path.insert(0, str(ROOT))
    import models.vision_transformer  # noqa: F401  (registers vit_base with timm)
    import utils
    return utils.build_model(build_args())


def load_state_dict_file(path):
    """fp32 flat state dict from a per-epoch dump (bare fp16 dict) or a full checkpoint ({'model': ...})."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return {k: v.float() for k, v in sd.items() if hasattr(v, "float")}


def zero_k_bias_(sd):
    """Zero the k rows of every `attn.qkv.bias` in place; returns {block index: removed L2 norm}.

    The key bias adds q.b_k to every attention logit of a query, a constant over keys, so the softmax and the
    model output are exactly invariant to it.  AdamW performs a random walk on it (no gradient, weight decay,
    Adam normalisation), reaching norms of several hundred in block 10 of random-init runs; three runs were
    hand-patched mid-run.  Removing it keeps parameter norms and weight deltas meaningful and avoids the
    bf16 precision loss of the large cancelling q.b_k term."""
    out = {}
    for k, v in sd.items():
        if k.endswith("attn.qkv.bias"):
            D = v.shape[0] // 3
            out[k.split(".")[1] if k.startswith("blocks.") else k] = float(v[D:2 * D].float().norm())
            v[D:2 * D] = 0
    return out


def load_state(model, entry, zero_k_bias=None):
    """Load the entry's weights into an existing model (strict); returns {block: removed k-bias norm} or {}.

    zero_k_bias=None takes the entry's own flag (True for every dynamics entry, unset for the section-0e ones)."""
    if zero_k_bias is None:
        zero_k_bias = bool(entry.get("zero_k_bias", False))
    sd = load_state_dict_file(ckpt_path(entry))
    info = zero_k_bias_(sd) if zero_k_bias else {}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (entry["name"], missing[:5], unexpected[:5])
    model.k_bias_removed = info
    return info


def load_model(entry, device="cuda", zero_k_bias=None):
    """ViT-B/16 with the entry's weights (epoch 299 unless the entry carries an epoch); strict load."""
    model = new_model()
    load_state(model, entry, zero_k_bias)
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
