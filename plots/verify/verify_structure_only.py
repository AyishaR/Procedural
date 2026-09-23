"""Verify a STRUCTURE-ONLY cell of the mechanism study (docs/early_lever_mechanism_plan.md, group D): sink + gate on a plain timm init,
spec without scales and without LayerNorm statistics, components added to the matrices ("renormalize": false).
  1  utils.apply_analytic_profile changes nothing (no slice, no "ln" in the spec): the model is timm seed 0 bit for bit
  2  utils.calibrate_joint_statistics (256 training images, seed 0, as main.py) changes exactly attn.qkv.weight and mlp.fc1.weight of
     the target blocks; v rows, every bias, LayerNorm, proj, fc2 and all other blocks stay timm's
  3  each change is rank one: W' - W_timm has sigma2 / sigma1 < 1e-3 for the q rows, the k rows and fc1 (the random part is untouched);
     q and k share the left vector (the sink direction), fc1's left vector is constant over hidden units
  4  targets met on the calibration images (entropy within 0.02 nats, mean pre-activation within 0.02) and every component reports matched;
     the same statistics on an unseen draw are printed
  5  what it costs: raw rms of q, k, fc1 against timm (reported, not a check); forward pass finite
usage (GPU): python plots/verify/verify_structure_only.py --spec vitbase_runs/profile_ftbc7sg.json"""
import argparse, contextlib, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet"); a = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
spec = json.load(open(a.spec)); E = 768; ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
COMPONENTS = [k for k in ("qk_entropy", "fc1_gate") if k in spec]          # both (S + G), or one of them (sink alone, gate alone)
check("specification holds no scale and no LayerNorm statistics; its component(s) with renormalize false",
      COMPONENTS and set(spec) == {"realise", "gain_fold", *COMPONENTS} and all(spec[k]["renormalize"] is False for k in COMPONENTS), str(sorted(spec)))
sink_t = {int(b): float(v) for b, v in spec.get("qk_entropy", {}).get("entropy", {}).items()}; gate_t = {int(b): float(v) for b, v in spec.get("fc1_gate", {}).get("pre_activation_mean", {}).items()}
torch.manual_seed(0); timm = {k: v.detach().clone() for k, v in utils.build_model(args).state_dict().items()}
torch.manual_seed(0); model = utils.build_model(args).to(DEV)
with contextlib.redirect_stdout(io.StringIO()):
    utils.apply_analytic_profile(model, a.spec, list(range(8)), seed=0)
check("1. the profile step is a no-op: model == timm seed 0, bit for bit", all(torch.equal(v.cpu(), timm[k]) for k, v in model.state_dict().items()))
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train")); n = max(int(spec[k].get("images", 64)) for k in COMPONENTS)
images = lambda seed: utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), n, seed).to(DEV)
report = utils.calibrate_joint_statistics(model, {**{k: spec[k] for k in COMPONENTS}, "gain_fold": spec["gain_fold"]}, images(0), seed=0)
after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
expected = sorted([f"blocks.{b}.attn.qkv.weight" for b in sink_t] + [f"blocks.{b}.mlp.fc1.weight" for b in gate_t]); changed = sorted(k for k in after if not torch.equal(after[k], timm[k]))
check(f"2. changed tensors are exactly qkv of blocks {sorted(sink_t)} and fc1 of blocks {sorted(gate_t)}", changed == expected, f"{len(changed)} tensors")
check("   v rows of every qkv stay timm's", all(torch.equal(after[f"blocks.{b}.attn.qkv.weight"][2 * E:], timm[f"blocks.{b}.attn.qkv.weight"][2 * E:]) for b in range(12)))
def rank_one(delta):
    U, S, Vh = torch.linalg.svd(delta.double(), full_matrices=False); return float(S[1] / S[0]), U[:, 0]
worst, shared, const = 0.0, 1.0, 0.0
print("   blk | raw rms q, k, fc1 against timm | entropy -> target | mean pre-activation -> target | active units, GELU rms | matched")
for b in sorted(set(sink_t) | set(gate_t)):
    W, W0 = after[f"blocks.{b}.attn.qkv.weight"], timm[f"blocks.{b}.attn.qkv.weight"]; F, F0 = after[f"blocks.{b}.mlp.fc1.weight"], timm[f"blocks.{b}.mlp.fc1.weight"]
    rms = lambda t: float(t.double().pow(2).mean().sqrt()); line = f"   {b:3d} | {rms(W[:E]) / rms(W0[:E]):.3f}, {rms(W[E:2 * E]) / rms(W0[E:2 * E]):.3f}, {rms(F) / rms(F0):.3f} | "
    if b in sink_t:
        rq, uq = rank_one(W[:E] - W0[:E]); rk, uk = rank_one(W[E:2 * E] - W0[E:2 * E]); worst = max(worst, rq, rk); shared = min(shared, abs(float(uq @ uk))); s = report[b]["qk_entropy"]
        line += f"{s['entropy']:.3f} -> {sink_t[b]:.3f} ({s['matched']}) | "
    else: line += "no sink | "
    if b in gate_t:
        rf, uf = rank_one(F - F0); worst = max(worst, rf); const = max(const, float(uf.std() / uf.mean().abs())); g = report[b]["fc1_gate"]
        line += f"{g['pre_activation_mean']:+.3f} -> {gate_t[b]:+.3f} ({g['matched']}) | {g['active_units']:.4f}, {g['gelu_rms']:.3f}"
    else: line += "no gate"
    print(line)
check("3. every change is rank one (sigma2 / sigma1)", worst < 1e-3, f"{worst:.1e}")
if sink_t: check("   q and k share the left vector (the sink direction)", shared > 0.999, f"min |cos| {shared:.5f}")
if gate_t: check("   fc1's left vector is constant over the hidden units", const < 1e-3, f"std / |mean| {const:.1e}")
check("4. every target met on the calibration images and reported matched", all(abs(report[b]["qk_entropy"]["entropy"] - sink_t[b]) <= 0.02 and report[b]["qk_entropy"]["matched"] for b in sink_t)
      and all(abs(report[b]["fc1_gate"]["pre_activation_mean"] - gate_t[b]) <= 0.02 and report[b]["fc1_gate"]["matched"] for b in gate_t))
B1 = utils.joint_statistics_per_block(model, images(1), sorted(set(sink_t) | set(gate_t)))
print("   unseen draw: entropy " + " ".join(f"{B1[b]['entropy']:.2f}" for b in sorted(B1)) + " | mean pre-activation " + " ".join(f"{B1[b]['pre_activation_mean']:+.2f}" for b in sorted(B1)))
with torch.no_grad(): check("5. forward pass finite", bool(torch.isfinite(model(images(0)[:8])).all()))
print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
