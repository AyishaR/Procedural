"""Verify the v-inclusive late-lever step-size arms (blocks 9-11; docs/proc_init_recipe.md section 13):
  ftb3bpl  ftb3b's init (v, proj x sqrt(a_b); fc2 x f_b, a_b / f_b = proc's measured write ratios) via profile_ftb3bpl.json "extra" + lr x the same multipliers
             (proj, fc2 as scalar groups; the v rows of the fused qkv via the row lr mask)   -- loud writes, normal relative steps
  ftb3bsl  timm init + lr x the reciprocals on the same tensors                             -- random writes, slow relative steps
  ftb3b      (exists, n = 3, 80.00 +- 0.14) is the loud + slow cell; random (n = 3) the random + normal cell.
Checks
  A  reconstruction through utils.apply_analytic_profile: in blocks 9-11 the v rows, proj and fc2 are timm's times ONE scalar each
     (the profile's multipliers), q/k rows, fc1, biases and LayerNorms bit-identical to timm; nothing outside blocks 9-11 touched.
  B  the reconstruction equals ftb3b's own init dump (results/init_dumps/ftb3b_s0.pth) to the rounding of the multipliers.
  C  lrscale_ftb3bpl.json: scalar entries == the profile multipliers, row entries == [[1536, 2304, v_b]] exactly;
     lrscale_ftb3bsl.json: the reciprocals (relative deviation < 1e-5); no other tensor named.
  D  --dump_pl / --dump_sl (main.py's own init, plots/dump_init.py): ftb3bpl dump == reconstruction bit for bit; ftb3bsl dump ==
     timm (seed 0) bit for bit.
  E  write ratios at init on 64 training images: the reconstruction and ftb3b's dump agree per block (attention, MLP).
usage: .venv/bin/python plots/verify/verify_late_trio_v.py [--dump_pl X.pth] [--dump_sl Y.pth] [--data_path DIR]"""
import argparse, contextlib, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
ap = argparse.ArgumentParser(); ap.add_argument("--dump_pl"); ap.add_argument("--dump_sl"); ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet")
ap.add_argument("--profile", default="vitbase_runs/profile_ftb3bpl.json"); ap.add_argument("--lr_pl", default="vitbase_runs/lrscale_ftb3bpl.json")
ap.add_argument("--lr_sl", default="vitbase_runs/lrscale_ftb3bsl.json"); ap.add_argument("--ftb3b", default="results/init_dumps/ftb3b_s0.pth")
a = ap.parse_args()
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
E = 768; BLOCKS = (9, 10, 11)
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
spec = json.load(open(a.profile)); extra = {int(b): v for b, v in spec["extra"].items()}
check("profile has only the 'extra' key for blocks 9-11 with v, proj, fc2", set(spec) == {"extra"} and sorted(extra) == list(BLOCKS) and all(set(extra[b]) == {"v", "proj", "fc2"} and extra[b]["v"] == extra[b]["proj"] for b in BLOCKS), str(extra))

print("A. reconstruction: utils.apply_analytic_profile with the 'extra' multipliers (main.py's call, no scaled blocks)")
torch.manual_seed(0); timm = {k: v.detach().clone() for k, v in utils.build_model(args).state_dict().items()}
torch.manual_seed(0); model = utils.build_model(args)
with contextlib.redirect_stdout(io.StringIO()):
    utils.apply_analytic_profile(model, a.profile, [], seed=0)
rec = {k: v.detach().clone() for k, v in model.state_dict().items()}
def one_scalar(W, W0):
    r = (W.double() / W0.double()); return float(r.mean()), float((r - r.mean()).abs().max() / r.mean().abs())
spread, dev = 0.0, 0.0
for b in BLOCKS:
    W, W0 = rec[f"blocks.{b}.attn.qkv.weight"], timm[f"blocks.{b}.attn.qkv.weight"]
    for name, sl, m in (("v rows", slice(2 * E, 3 * E), extra[b]["v"]),):
        f, s = one_scalar(W[sl], W0[sl]); spread = max(spread, s); dev = max(dev, abs(f / m - 1))
    for name, m in (("attn.proj.weight", extra[b]["proj"]), ("mlp.fc2.weight", extra[b]["fc2"])):
        f, s = one_scalar(rec[f"blocks.{b}.{name}"], timm[f"blocks.{b}.{name}"]); spread = max(spread, s); dev = max(dev, abs(f / m - 1))
    check(f"block {b}: q/k rows, fc1, biases, LayerNorms bit-identical to timm", torch.equal(W[:2 * E], W0[:2 * E]) and all(torch.equal(rec[f"blocks.{b}.{n}"], timm[f"blocks.{b}.{n}"]) for n in ("mlp.fc1.weight", "attn.qkv.bias", "attn.proj.bias", "mlp.fc1.bias", "mlp.fc2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")))
check("v rows, proj, fc2 of blocks 9-11 = timm x one scalar each", spread < 1e-6, f"max elementwise spread {spread:.1e}")
check("that scalar is the profile's multiplier", dev < 1e-6, f"max relative deviation {dev:.1e}")
untouched = all(torch.equal(rec[k], timm[k]) for k in timm if not any(k.startswith(f"blocks.{b}.") for b in BLOCKS))
check("everything outside blocks 9-11 bit-identical to timm", untouched)

print("B. reconstruction vs ftb3b's own init dump")
if os.path.exists(a.ftb3b):
    rho = torch.load(a.ftb3b, map_location="cpu", weights_only=False); rho = rho.get("model", rho)
    worst = max(float((rec[k].double() - rho[k].double()).abs().max() / rho[k].double().abs().max()) for k in rec if k in rho and rho[k].dtype.is_floating_point and rho[k].numel() > 0 and float(rho[k].abs().max()) > 0)
    check("max relative deviation over all tensors (multipliers rounded to 4 decimals)", worst < 5e-5, f"{worst:.1e}")
    same_out = all(torch.equal(rho[k], timm[k]) for k in timm if not any(k.startswith(f"blocks.{b}.") for b in BLOCKS))
    check("ftb3b's dump is timm outside blocks 9-11 (same seed 0 random init)", same_out)
else:
    check("ftb3b dump present", False, a.ftb3b)

print("C. learning-rate files")
pl, sl = json.load(open(a.lr_pl)), json.load(open(a.lr_sl))
exp_keys = {f"blocks.{b}.{n}" for b in BLOCKS for n in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc2.weight")}
check("both files name exactly the nine tensors (v rows of qkv, proj, fc2 of blocks 9-11)", set(pl) == exp_keys and set(sl) == exp_keys, f"{sorted(set(pl) ^ exp_keys)} {sorted(set(sl) ^ exp_keys)}")
dev_pl = max(max(abs(pl[f"blocks.{b}.attn.proj.weight"] / extra[b]["proj"] - 1), abs(pl[f"blocks.{b}.mlp.fc2.weight"] / extra[b]["fc2"] - 1)) for b in BLOCKS)
rows_pl = all(pl[f"blocks.{b}.attn.qkv.weight"] == {"rows": [[2 * E, 3 * E, extra[b]["v"]]]} for b in BLOCKS)
check("ftb3bpl: lambda = the init multipliers (proj, fc2 scalars; v rows [1536, 2304) exactly)", dev_pl == 0 and rows_pl, f"scalar deviation {dev_pl:.1e}")
dev_sl = max(max(abs(sl[f"blocks.{b}.attn.proj.weight"] * extra[b]["proj"] - 1), abs(sl[f"blocks.{b}.mlp.fc2.weight"] * extra[b]["fc2"] - 1),
                 abs(sl[f"blocks.{b}.attn.qkv.weight"]["rows"][0][2] * extra[b]["v"] - 1)) for b in BLOCKS)
rows_sl = all(sl[f"blocks.{b}.attn.qkv.weight"]["rows"][0][:2] == [2 * E, 3 * E] and len(sl[f"blocks.{b}.attn.qkv.weight"]["rows"]) == 1 for b in BLOCKS)
check("ftb3bsl: lambda = the reciprocals (rows [1536, 2304))", dev_sl < 1e-5 and rows_sl, f"max relative deviation {dev_sl:.1e}")
from row_lr_mask import split_lr_scale_spec
for f, d in ((a.lr_pl, pl), (a.lr_sl, sl)):
    try: split_lr_scale_spec(d); check(f"{os.path.basename(f)} parses (finite positive lambdas, integer rows)", True)
    except Exception as e: check(f"{os.path.basename(f)} parses", False, str(e))

print("D. main.py's own init (plots/dump_init.py)")
if a.dump_pl and os.path.exists(a.dump_pl):
    d = torch.load(a.dump_pl, map_location="cpu", weights_only=False); d = d.get("model", d)
    check("ftb3bpl dump == reconstruction, bit for bit, every tensor", all(torch.equal(d[k], rec[k]) for k in rec), f"{len(rec)} tensors")
else:
    print("  (no --dump_pl: skipped)")
if a.dump_sl and os.path.exists(a.dump_sl):
    d = torch.load(a.dump_sl, map_location="cpu", weights_only=False); d = d.get("model", d)
    check("ftb3bsl dump == timm seed 0, bit for bit, every tensor", all(torch.equal(d[k], timm[k]) for k in timm), f"{len(timm)} tensors")
else:
    print("  (no --dump_sl: skipped)")

print("E. write ratios at init on 64 training images (reconstruction vs ftb3b's dump)")
train_dir = os.path.join(a.data_path, "train")
if os.path.isdir(train_dir) and os.path.exists(a.ftb3b):
    from torchvision import datasets as tv_datasets
    from datasets import build_transform
    folder = tv_datasets.ImageFolder(train_dir)
    images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), 64, 0)
    def ratios(sd):
        m = utils.build_model(args); m.load_state_dict(sd); m.eval()
        with torch.no_grad():
            x = utils.block_input_stream(m, images); out = {}
            for b, blk in enumerate(m.blocks):
                att, mlp, x = utils.sublayer_write_ratios(blk, x)
                if b in BLOCKS: out[b] = (att, mlp)
        return out
    r_rec, r_ref = ratios(rec), ratios(rho)
    worst = max(abs(r_rec[b][i] / r_ref[b][i] - 1) for b in BLOCKS for i in (0, 1))
    for b in BLOCKS:
        print(f"    block {b}: attention {r_rec[b][0]:.4f} (ftb3b {r_ref[b][0]:.4f})   MLP {r_rec[b][1]:.4f} (ftb3b {r_ref[b][1]:.4f})")
    check("reconstruction and ftb3b agree per block", worst < 1e-3, f"max relative deviation {worst:.1e}")
    T = {9: (1.380, 4.757), 10: (2.098, 4.459), 11: (0.777, 0.526)}
    worst_t = max(abs(r_rec[b][i] / T[b][i] - 1) for b in BLOCKS for i in (0, 1))
    check("write ratios close to the launched targets (64 images here against 5000 at calibration)", worst_t < 0.15, f"max relative deviation from the targets {100 * worst_t:.1f}%")
else:
    print("  (no training images or no ftb3b dump: skipped)")
print("VERDICT:", "PASS" if ok else "FAIL")
