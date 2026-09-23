"""Verify the compose arm ftbc7l (docs/early_lever_mechanism_plan.md): committed early lever C on blocks 0-7 + ftbrhop's late lever on 9-11.
  1  the compose spec equals C's spec plus ftbrhop's "extra" (nothing else differs)
  2  init built as main.py builds it (apply_analytic_profile + calibrate_joint_statistics on 256 training images, seed 0): every
     tensor of blocks 0-7 and every tensor outside the blocks equals C's init built the same way, bit for bit (the late lever
     does not touch the early blocks' stream, so the joint calibration is identical)
  3  blocks 9-11: attn.proj.weight and mlp.fc2.weight = timm x multiplier (relative deviation < 1e-6), every other tensor of
     blocks 8-11 = timm bit for bit
  4  the write ratios of blocks 8-11 on 256 training images (attention and MLP: ||branch|| / ||stream||) against timm and
     against the ftbrho init (results/init_dumps/ftbrho_s0.pth if present): the late lever's signature is reproduced
  5  forward pass finite in fp16
writes plots/out/compose_late_ratios.json (numbers for the late-lever figure). usage (GPU): python plots/verify/verify_compose.py"""
import contextlib, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"; DATA = "/work/dlcsmall2/schrodi-imagenet"; DEV = torch.device("cuda"); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", DATA, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
comp = json.load(open(f"{ROOT}/vitbase_runs/profile_ftbc7l.json")); c = json.load(open(f"{ROOT}/vitbase_runs/profile_ftbanapermb7i.json")); late = json.load(open(f"{ROOT}/vitbase_runs/profile_ftbrhoplv.json"))
check("1. compose spec == C spec + ftbrhoplv extra (v, proj, fc2)", {k: v for k, v in comp.items() if k != "extra"} == c and comp["extra"] == late["extra"])
folder = tv_datasets.ImageFolder(os.path.join(DATA, "train")); images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), 256, 0).to(DEV)
def build(spec_path, spec):
    torch.manual_seed(0); m = utils.build_model(args).to(DEV)
    with contextlib.redirect_stdout(io.StringIO()): utils.apply_analytic_profile(m, spec_path, list(range(8)), seed=0)
    joint = {k: spec[k] for k in ("qk_entropy", "fc1_gate") if k in spec}
    rep = utils.calibrate_joint_statistics(m, {**joint, "gain_fold": spec["gain_fold"]}, images, seed=0)
    assert all(p.get("matched", p["reachable"]) for parts in rep.values() for p in parts.values()), "calibration target not met"
    return m, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
torch.manual_seed(0); timm = {k: v.detach().cpu().clone() for k, v in utils.build_model(args).state_dict().items()}
model, comp_sd = build(f"{ROOT}/vitbase_runs/profile_ftbc7l.json", comp); _, c_sd = build(f"{ROOT}/vitbase_runs/profile_ftbanapermb7i.json", c)
early = [k for k in comp_sd if not k.startswith("blocks.") or int(k.split(".")[1]) <= 7]
check(f"2. blocks 0-7 and everything outside the blocks == C's init bit for bit ({len(early)} tensors)", all(torch.equal(comp_sd[k], c_sd[k]) for k in early))
dev = 0.0; untouched = True; E = 768
for b in (9, 10, 11):
    m = late["extra"][str(b)]
    for name, key in (("proj", f"blocks.{b}.attn.proj.weight"), ("fc2", f"blocks.{b}.mlp.fc2.weight")):
        dev = max(dev, float((comp_sd[key] / timm[key].to(comp_sd[key].dtype) / float(m[name]) - 1).abs().max()))
    qkv, qkv0 = comp_sd[f"blocks.{b}.attn.qkv.weight"], timm[f"blocks.{b}.attn.qkv.weight"]
    dev = max(dev, float((qkv[2 * E:] / qkv0[2 * E:] / float(m["v"]) - 1).abs().max())); untouched &= torch.equal(qkv[:2 * E], qkv0[:2 * E])
for b in (8, 9, 10, 11):
    for k in comp_sd:
        if k.startswith(f"blocks.{b}.") and not (b >= 9 and k.endswith(("attn.proj.weight", "mlp.fc2.weight", "attn.qkv.weight"))): untouched &= torch.equal(comp_sd[k], timm[k])
check("3. blocks 9-11: v rows, proj, fc2 == timm x multiplier", dev < 1e-6, f"max relative deviation {dev:.1e}"); check("   q, k rows and every other tensor of blocks 8-11 == timm", untouched)
def write_ratios(m):
    out = {}
    with torch.no_grad():
        x = utils.block_input_stream(m, images)
        for b, blk in enumerate(m.blocks):
            att = blk.attn(blk.norm1(x)); mid = x + att; mlp = blk.mlp(blk.norm2(mid))
            out[b] = {"attn": float((att.norm(dim=-1) / x.norm(dim=-1)).mean()), "mlp": float((mlp.norm(dim=-1) / mid.norm(dim=-1)).mean())}; x = mid + mlp
    return out
ratios = {"compose": write_ratios(model)}; torch.manual_seed(0); rnd = utils.build_model(args).to(DEV); ratios["timm"] = write_ratios(rnd)
p = f"{ROOT}/results/init_dumps/ftbrho_s0.pth"
if os.path.exists(p):
    sd = torch.load(p, map_location="cpu", weights_only=True); rnd.load_state_dict({k: v.float() for k, v in sd.items()}); ratios["ftbrho"] = write_ratios(rnd)
    torch.manual_seed(0); rnd = utils.build_model(args).to(DEV)
_, cm = build(f"{ROOT}/vitbase_runs/profile_ftbanapermb7i.json", c); rnd.load_state_dict(cm); ratios["C"] = write_ratios(rnd)
print("4. write ratios (attention | MLP) per block, 256 training images:"); print("   blk | " + " | ".join(f"{n:>18s}" for n in ratios))
for b in range(12): print(f"   {b:3d} | " + " | ".join(f"{ratios[n][b]['attn']:8.3f} {ratios[n][b]['mlp']:8.3f}" for n in ratios))
if "ftbrho" in ratios:
    worst = max(abs(ratios["compose"][b][k] / ratios["ftbrho"][b][k] - 1) for b in (9, 10, 11) for k in ("attn", "mlp"))
    check("   late lever present: blocks 9-11 write ratios >= 1.2 (timm 0.14-0.25) and within 15% of ftbrho's", all(ratios["compose"][b][k] >= 1.2 for b in (9, 10, 11) for k in ("attn", "mlp")) and worst < 0.15,
          f"max deviation from ftbrho {100 * worst:.0f}% (the multipliers were fit on timm's stream; C's early stack feeds block 9 differently, block 9 comes out louder)")
json.dump({n: {str(b): v for b, v in r.items()} for n, r in ratios.items()}, open(f"{ROOT}/plots/out/compose_late_ratios.json", "w"), indent=1)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): check("5. fp16 forward pass finite", bool(torch.isfinite(model(images[:16])).all()))
print("VERDICT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
