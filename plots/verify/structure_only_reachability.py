"""Structure-only cell of the scale x structure table (docs/proc_init_recipe.md section 12): sink + gate on a plain timm init.
Are the committed arms' entropy / gate targets reachable when the components must keep the matrix size (renormalize = true, the
convention of every launched arm), and what happens to the matrix size when they need not (renormalize = false)?
usage: .venv/bin/python plots/verify/structure_only_reachability.py [--images 64] [--specs vitbase_runs/profile_ftbanaperab7i.json ...]"""
import argparse, contextlib, copy, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ap = argparse.ArgumentParser(); ap.add_argument("--images", type=int, default=64); ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet")
ap.add_argument("--specs", nargs="+", default=["vitbase_runs/profile_ftbanaperab7i.json", "vitbase_runs/profile_ftbanakperab7i.json"]); a = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), a.images, 0).to(DEV)
for path in a.specs:
    spec = json.load(open(path)); gate_key = "active_units" if "active_units" in spec["fc1_gate"] else "pre_activation_mean"
    for renorm in (True, False):
        joint = {"qk_entropy": {**copy.deepcopy(spec["qk_entropy"]), "renormalize": renorm}, "fc1_gate": {**copy.deepcopy(spec["fc1_gate"]), "renormalize": renorm}, "gain_fold": "exact"}
        torch.manual_seed(0); model = utils.build_model(args).to(DEV)
        report = utils.calibrate_joint_statistics(model, joint, images, seed=0)
        print(f"== {os.path.basename(path)} on a timm init, renormalize = {renorm} ({a.images} training images, {DEV})")
        print("   blk | entropy target -> reached (matched) | raw rms q, k vs timm | gate target -> reached (matched) | raw rms fc1 vs timm | GELU rms")
        for b, parts in sorted(report.items()):
            s, g = parts.get("qk_entropy"), parts.get("fc1_gate")
            line = f"   {b:3d} |"
            line += f" {s['target']:.3f} -> {s['entropy']:.3f} ({s['matched']}) | {s['rms_q_ratio']:.2f}, {s['rms_k_ratio']:.2f} |" if s else " - | - |"
            line += f" {g['target']:.5f} -> {g[gate_key]:.5f} ({g['matched']}) | {g['rms_ratio']:.2f} | {g['gelu_rms']:.3f}" if g else " - | - | -"
            print(line)
        n_s = sum(1 for p in report.values() if "qk_entropy" in p); n_g = sum(1 for p in report.values() if "fc1_gate" in p)
        print(f"   matched: entropy {sum(p['qk_entropy']['matched'] for p in report.values() if 'qk_entropy' in p)}/{n_s}, gate {sum(p['fc1_gate']['matched'] for p in report.values() if 'fc1_gate' in p)}/{n_g}")
