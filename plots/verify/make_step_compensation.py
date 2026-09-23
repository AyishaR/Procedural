"""Learning-rate files for the step-size arms of the mechanism study (docs/early_lever_mechanism_plan.md, group A).

m = rms(W of the committed arm's LAUNCHED init) / rms(W of timm, same seed), per block 0-7 and per tensor: the q rows and the k rows
of the fused attn.qkv.weight, mlp.fc1.weight, and the two LayerNorm gains. The init is rebuilt exactly as main.py builds it:
utils.apply_analytic_profile, then utils.calibrate_joint_statistics on the 256 TRAINING images of the launched protocol (seed 0,
evaluation transform), on the GPU -- not read off results/init_dumps/<arm>_s0.pth, whose sink and gate were calibrated on validation
images (memory note init-dumps-calibrate-on-val). Sink and gate keep the gain-folded size, so the raw rms of q, k, fc1 moves with them;
that is why m comes from the final tensors and not from the specification.

Written (weight decay is divided by the same number in optim_factory, so the decay per step is unchanged):
  lrscale_<pre>a1.json    lambda = m on q rows, k rows (row lr mask) and fc1            -> random-init nominal relative steps
  lrscale_<pre>a1g.json   the same + lambda = m (< 1) on norm1.weight and norm2.weight
  lrscale_ftbc7a2.json    lambda = 0.5 on q rows, k rows, fc1 (kdyck)                  -> slower than natural
  lrscale_ftbck7a3qk.json lambda = m on q and k rows only (ksd);  lrscale_ftbck7a3f.json  lambda = m on fc1 only (ksd)
usage (GPU): python plots/verify/make_step_compensation.py"""
import contextlib, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"; DATA = "/work/dlcsmall2/schrodi-imagenet"; E = 768; BLOCKS = list(range(8))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", DATA, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tv_datasets.ImageFolder(os.path.join(DATA, "train"))
rms = lambda t: float(t.double().pow(2).mean().sqrt())
torch.manual_seed(0); timm = {k: v.detach().clone() for k, v in utils.build_model(args).state_dict().items()}
TABLE = {}
for pre, base in (("ftbc7", "ftbanapermb7i"), ("ftbck7", "ftbanakpermb7i")):
    spec_path = f"{ROOT}/vitbase_runs/profile_{base}.json"; spec = json.load(open(spec_path))
    torch.manual_seed(0); model = utils.build_model(args).to(DEV)
    with contextlib.redirect_stdout(io.StringIO()):
        utils.apply_analytic_profile(model, spec_path, BLOCKS, seed=0)
    joint = {k: spec[k] for k in ("qk_entropy", "fc1_gate") if k in spec}
    images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), max(int(p.get("images", 64)) for p in joint.values()), 0).to(DEV)
    report = utils.calibrate_joint_statistics(model, {**joint, "gain_fold": spec["gain_fold"]}, images, seed=0)
    unmet = [(b, k) for b, parts in report.items() for k, p in parts.items() if not p.get("matched", p["reachable"])]
    assert not unmet, f"{base}: calibration targets not met: {unmet}"
    init = {k: v.detach().cpu() for k, v in model.state_dict().items()}; torch.save(init, f"{ROOT}/results/init_dumps/launched_init_{base}.pth")
    m = {}
    for b in BLOCKS:
        W, W0 = init[f"blocks.{b}.attn.qkv.weight"], timm[f"blocks.{b}.attn.qkv.weight"]
        m[b] = {"q": rms(W[:E]) / rms(W0[:E]), "k": rms(W[E:2 * E]) / rms(W0[E:2 * E]), "v": rms(W[2 * E:]) / rms(W0[2 * E:]),
                "fc1": rms(init[f"blocks.{b}.mlp.fc1.weight"]) / rms(timm[f"blocks.{b}.mlp.fc1.weight"]),
                "proj": rms(init[f"blocks.{b}.attn.proj.weight"]) / rms(timm[f"blocks.{b}.attn.proj.weight"]), "fc2": rms(init[f"blocks.{b}.mlp.fc2.weight"]) / rms(timm[f"blocks.{b}.mlp.fc2.weight"]),
                "gain1": rms(init[f"blocks.{b}.norm1.weight"]), "gain2": rms(init[f"blocks.{b}.norm2.weight"])}
        assert abs(m[b]["v"] - 1) < 1e-6 and abs(m[b]["proj"] - 1) < 1e-6 and abs(m[b]["fc2"] - 1) < 1e-6, f"{base} block {b}: the write side is not at timm"
    others = [k for k in timm if not any(k.startswith(f"blocks.{b}.") for b in BLOCKS) and not torch.equal(init[k], timm[k])]
    assert not others, others
    TABLE[base] = m; r6 = lambda x: float(f"{x:.6g}")
    rows = lambda b, names: {"rows": [[i * E, (i + 1) * E, r6(m[b][n])] for i, n in enumerate(("q", "k")) if n in names]}
    a1 = {}; a1g = {}; a3qk = {}; a3f = {}; a2 = {}
    for b in BLOCKS:
        a1[f"blocks.{b}.attn.qkv.weight"] = rows(b, ("q", "k")); a1[f"blocks.{b}.mlp.fc1.weight"] = r6(m[b]["fc1"])
        a3qk[f"blocks.{b}.attn.qkv.weight"] = rows(b, ("q", "k")); a3f[f"blocks.{b}.mlp.fc1.weight"] = r6(m[b]["fc1"])
        a2[f"blocks.{b}.attn.qkv.weight"] = {"rows": [[0, E, 0.5], [E, 2 * E, 0.5]]}; a2[f"blocks.{b}.mlp.fc1.weight"] = 0.5
    a1g = {**a1, **{f"blocks.{b}.norm{i}.weight": r6(m[b][f"gain{i}"]) for b in BLOCKS for i in (1, 2)}}
    out = {f"{pre}a1": a1, f"{pre}a1g": a1g}
    if pre == "ftbc7": out["ftbc7a2"] = a2
    else: out["ftbck7a3qk"] = a3qk; out["ftbck7a3f"] = a3f
    for name, d in out.items(): json.dump(d, open(f"{ROOT}/vitbase_runs/lrscale_{name}.json", "w"), indent=1)
    print(f"== {base}: m = rms(launched init) / rms(timm seed 0); all calibration targets matched; v, proj, fc2 and everything outside blocks 0-7 == timm")
    print("   blk |     q       k     fc1   gain1   gain2 | sink s_q, s_k; gate s (share of the random part kept)")
    for b in BLOCKS:
        p = report.get(b, {}); s = f"{p['qk_entropy']['s_q']:.3f}, {p['qk_entropy']['s_k']:.3f}; {p['fc1_gate']['s']:.3f}" if "qk_entropy" in p else "-"
        print(f"   {b:3d} | {m[b]['q']:6.3f}  {m[b]['k']:6.3f}  {m[b]['fc1']:6.3f}  {m[b]['gain1']:6.3f}  {m[b]['gain2']:6.3f} | {s}")
    dump = f"{ROOT}/results/init_dumps/{base}_s0.pth"
    if os.path.exists(dump):
        d = torch.load(dump, map_location="cpu", weights_only=True); worst = 0.0
        for b in BLOCKS:
            Wd = d[f"blocks.{b}.attn.qkv.weight"]; worst = max(worst, abs(rms(Wd[:E]) / rms(timm[f"blocks.{b}.attn.qkv.weight"][:E]) / m[b]["q"] - 1), abs(rms(d[f"blocks.{b}.mlp.fc1.weight"]) / rms(timm[f"blocks.{b}.mlp.fc1.weight"]) / m[b]["fc1"] - 1))
        print(f"   against the validation-image dump {os.path.basename(dump)}: m differs by at most {100 * worst:.2f}% (q, fc1)")
    print("   wrote " + ", ".join(f"lrscale_{n}.json" for n in out))
json.dump({k: {str(b): v for b, v in t.items()} for k, t in TABLE.items()}, open(f"{ROOT}/results/init_dumps/wave1_step_multipliers.json", "w"), indent=1)
print("DONE")
