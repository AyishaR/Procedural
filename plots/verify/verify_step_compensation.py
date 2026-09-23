"""Verify a learning-rate file of the step-size arms (docs/early_lever_mechanism_plan.md, group A) against the multipliers measured
on the launched init (results/init_dumps/wave1_step_multipliers.json, written by make_step_compensation.py).
usage: python plots/verify/verify_step_compensation.py ARM      (ARM in ftbc7a1, ftbc7a1g, ftbc7a2, ftbck7a1, ftbck7a1g, ftbck7a3qk, ftbck7a3f)"""
import contextlib, io, json, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from row_lr_mask import split_lr_scale_spec, install_row_lr_masks
from optim_factory import build_lr_scaled_param_groups
arm = sys.argv[1]; E = 768; ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
base = "ftbanakpermb7i" if arm.startswith("ftbck7") else "ftbanapermb7i"
m = {int(b): v for b, v in json.load(open("results/init_dumps/wave1_step_multipliers.json"))[base].items()}
spec = json.load(open(f"vitbase_runs/lrscale_{arm}.json")); kind = arm.replace("ftbck7", "").replace("ftbc7", "")
want = {}
for b in range(8):
    if kind in ("a1", "a1g", "a3qk", "a2"): want[f"blocks.{b}.attn.qkv.weight"] = {"q": 0.5 if kind == "a2" else m[b]["q"], "k": 0.5 if kind == "a2" else m[b]["k"]}
    if kind in ("a1", "a1g", "a3f", "a2"): want[f"blocks.{b}.mlp.fc1.weight"] = 0.5 if kind == "a2" else m[b]["fc1"]
    if kind == "a1g": want[f"blocks.{b}.norm1.weight"] = m[b]["gain1"]; want[f"blocks.{b}.norm2.weight"] = m[b]["gain2"]
check("the file names exactly the intended tensors", set(spec) == set(want), f"{len(spec)} entries; difference {sorted(set(spec) ^ set(want))[:4]}")
dev = 0.0; rows_ok = True
for name, w in want.items():
    v = spec.get(name)
    if isinstance(w, dict):
        rows_ok &= isinstance(v, dict) and [r[:2] for r in v["rows"]] == [[0, E], [E, 2 * E]]
        if rows_ok: dev = max(dev, abs(v["rows"][0][2] / w["q"] - 1), abs(v["rows"][1][2] / w["k"] - 1))
    else:
        dev = max(dev, abs(v / w - 1) if isinstance(v, (int, float)) else 1.0)
check("q rows [0, 768) and k rows [768, 1536) are masked, the v rows are not named (lambda 1)", rows_ok)
check("every lambda equals its target (m of the launched init, or 0.5)", dev < 1e-5, f"max relative deviation {dev:.1e}")
if kind == "a1g": check("the LayerNorm gains get lambda < 1 (they are smaller than timm's gains of 1)", all(spec[f"blocks.{b}.norm{i}.weight"] < 1 for b in range(8) for i in (1, 2)), f"range {min(spec[f'blocks.{b}.norm{i}.weight'] for b in range(8) for i in (1, 2)):.3f} .. {max(spec[f'blocks.{b}.norm{i}.weight'] for b in range(8) for i in (1, 2)):.3f}")
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
model = utils.build_model(args); scalar, rows = split_lr_scale_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    groups = build_lr_scaled_param_groups(model, 0.05, scalar); opt = torch.optim.AdamW(groups, lr=1e-3); install_row_lr_masks(opt, model, rows)
decay_ok = all(abs(g["lr_scale"] * g["wd_scale"] - 1.0) < 1e-9 for g in groups if g["weight_decay"] > 0)
check("optimizer builds: scalar groups + row masks; lr scale x wd scale = 1 in every decayed group (decay per step unchanged)", decay_ok, f"{len(groups)} groups, {len(rows)} masked tensors")
rel = [spec[n]["rows"][0][2] / m[int(n.split('.')[1])]["q"] for n in spec if isinstance(spec[n], dict)]
if rel: print(f"  nominal relative step of q against timm's: x{min(rel):.3f} .. x{max(rel):.3f}   (1 = random-init relative steps; the natural value of the committed arm is 1/m = x{1 / max(m[b]['q'] for b in m):.2f} .. x{1 / min(m[b]['q'] for b in m):.2f})")
print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
