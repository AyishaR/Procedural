"""Init signatures: what each arm's initialised model DOES on training images, next to its final accuracy.

For every arm with an init dump (results/init_dumps/<arm>_s0.pth, made through main.py) and for the checkpoint prefixes, the
quantities of plot_reconstruction.forward_statistics are measured on 256 training images (evaluation transform, image seed 1)
and averaged over blocks 1-7 (common to the 0-7 and 0-8 arms): attention entropy, top-key mass, active fc1 units, mean fc1
pre-activation, GELU rms, attention / MLP write ratio, token cosine; plus block 0's MLP write ratio, the stream rms entering
block 8, the relative write of the random tail (blocks 9-11) and the raw q / k scale of blocks 1-7 (step-size proxy).
Results are cached per arm in plots/out/init_signatures.json; rerun to add arms.
usage: python plots/verify/init_signature_table.py [--data_path D] [--arms a,b,...] [--recompute]"""
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
from torchvision import datasets as tv_datasets
import main as M, utils, extract_profile, plot_reconstruction as PR
from datasets import build_transform

KDYCK, KSD = "results/pr_vitb_n/pr_6066174_final.pth", "results/pr_vitb_ksd/pr_6463456_final.pth"
RANDOM = 78.08
# (task, last-epoch top-1 or None if not run, description). Accuracies: docs/synthesis_data.md, docs/i100_late_block_scaling.md.
ARMS = {
    "random":          ("-",     78.08, "timm random init (n=3, sd 0.19)"),
    "prefix_kdyck_0-7": ("kdyck", 79.89, "checkpoint blocks 0-7, random 8-11 (ftb4i, n=3)"),
    "ftb3i":           ("kdyck", 79.99, "checkpoint blocks 0-8 (n=3)"),
    "ftb4m":           ("kdyck", 80.00, "checkpoint 0-7, v/proj/fc2 downscaled to random write ratios"),
    "ftb4o":           ("kdyck", 77.27, "random 0-7, v/proj/fc2 write-matched to proc, nothing else"),
    "ftbqmlnvo":       ("kdyck", 79.93, "quantile twin 0-8 (n=3)"),
    "ftbana":          ("kdyck", 76.61, "scale profile, LN gains 1 (no large raw matrices)"),
    "ftbanag":         ("kdyck", 80.70, "ftbana + permuted proc LN gains"),
    "ftbanau":         ("kdyck", 77.35, "ftbana + isotropic gains N(1, 0.25)"),
    "ftbanap":         ("kdyck", 80.24, "corrected second-moment recipe 0-8 (pooled q/k 1.32)"),
    "ftbanape":        ("kdyck", 79.44, "exact second-moment recipe 0-8"),
    "ftbanai":         ("kdyck", 78.11, "ftbanap with input-side effective scales reset to random"),
    "ftbanal":         ("kdyck", 79.78, "ftbanap's relative Adam steps via lr scales, gains 1 (init rebuilt from its spec)"),
    "ftbanapx":        ("kdyck", 79.18, "exact folding, q/k separate, ramp (init rebuilt from its spec)"),
    "ftbanac":         ("kdyck", 80.55, "ftbanap + blocks 9-11 amplified to write ratio 1.4"),
    "ftbanab":         ("kdyck", 76.70, "ftbana + permuted proc LN biases"),
    "ftbanaf":         ("kdyck", 76.41, "ftbana + flat MLP top"),
    "ftbanaperab7":    ("kdyck", None,  "NEW scales(all six) + entropy + active-unit gate"),
    "ftbanaperab7i":   ("kdyck", None,  "NEW (1) q,k,fc1 scales + entropy + gate; v/proj/fc2 timm"),
    "ftbanaperab7w":   ("kdyck", None,  "NEW (2) (1) + v,proj,fc2 write-matched"),
    "ftbanaperab7vw":  ("kdyck", None,  "NEW (3) (1) + v scale + proj,fc2 write-matched"),
    "ftbanapermb7":    ("kdyck", None,  "NEW control: ftbanaperab7 with the gate calibrated to the MEAN pre-activation"),
    "ftbanapermb7vw":  ("kdyck", None,  "NEW control: ftbanaperab7vw with the mean-calibrated gate"),
    "ftbanapermb7w":   ("kdyck", None,  "NEW control: ftbanaperab7w with the mean-calibrated gate"),
    "ftbanapermb7i":   ("kdyck", None,  "mean gate, v/proj/fc2 at timm"),
    "ftbanakpermb7":   ("ksd",   None,  "mean gate, all six scales"),
    "ftbanakpermb7i":  ("ksd",   None,  "mean gate, v/proj/fc2 at timm"),
    "ftbanaperg0b7":   ("kdyck", None,  "earlier set-up: scales + entropy + mean gate + block-0 common write"),
    "ftbanaperg0b7vw": ("kdyck", None,  "earlier set-up: (3) with mean gate + common write"),
    "prefix_ksd_0-7":  ("ksd",   80.05, "ksd checkpoint blocks 0-7 (coworker's ftb4i, n=1)"),
    "pksd3i":          ("ksd",   None,  "ksd checkpoint blocks 0-8 (init only)"),
    "ftbanak":         ("ksd",   77.86, "second-moment recipe 0-8"),
    "ftbanakx":        ("ksd",   77.70, "exact scales 0-8 (epoch 298, final pending)"),
    "ftbanakg":        ("ksd",   77.64, "exact scales + permuted LN vectors"),
    "ftbqmlnvok":      ("ksd",   77.83, "quantile twin on ksd"),
    "ftbanakw":        ("ksd",   76.65, "recipe + fitted write profile"),
    "ftbanakb":        ("ksd",   78.25, "recipe + transient fc1 bias gate"),
    "ftbanakd":        ("ksd",   78.04, "recipe + gate, variant d"),
    "ftbanaksw":       ("ksd",   78.73, "recipe + weak q-bias sink"),
    "ftbanaks":        ("ksd",   79.58, "recipe + q-bias sink at the prefix's entropy"),
    "ftbanaksg":       ("ksd",   79.92, "recipe + sink + persistent fc1 bias gate (active-unit matched)"),
    "ftbanakbs":       ("ksd",   None,  "recipe + PERSISTENT fc1 bias gate (lr x0.02), NO sink: control of ftbanaksg (running)"),
    "ftbanakperab7":   ("ksd",   None,  "NEW scales(all six) + entropy + active-unit gate"),
    "ftbanakperab7i":  ("ksd",   None,  "NEW (1)"),
    "ftbanakperab7w":  ("ksd",   None,  "NEW (2)"),
    "ftbanakperab7vw": ("ksd",   None,  "NEW (3)"),
    "ftbanakperg0b7":  ("ksd",   None,  "earlier set-up: scales + entropy + mean gate + common write"),
}
ap = argparse.ArgumentParser(); ap.add_argument("--data_path", default="/data/datasets/ILSVRC2012"); ap.add_argument("--arms", default="")
ap.add_argument("--recompute", action="store_true"); ap.add_argument("--images", type=int, default=256); a = ap.parse_args()
OUT = "/home/schrodi/Procedural/plots/out/init_signatures.json"
cache = {} if a.recompute or not os.path.exists(OUT) else json.load(open(OUT))
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train")); images = None
rms = lambda t: float(t.detach().float().pow(2).mean().sqrt())

def build(arm):
    torch.manual_seed(0); model = utils.build_model(args)
    if arm == "random":
        return model
    if arm.startswith("prefix_"):
        state = extract_profile.load_state_dict(KDYCK if "kdyck" in arm else KSD)
        model.load_state_dict({k: v for k, v in state.items() if k.startswith("blocks.") and int(k.split(".")[1]) <= 7}, strict=False); return model
    path = f"/home/schrodi/Procedural/results/init_dumps/{arm}_s0.pth"
    if not os.path.exists(path):
        # no dump through main.py: rebuild the init from the arm's specification if it is a pure scale / LayerNorm recipe
        spec_path, script = f"/home/schrodi/Procedural/vitbase_runs/profile_{arm}.json", f"/home/schrodi/Procedural/vitbase_runs/run_train_{arm}.sh"
        if not (os.path.exists(spec_path) and os.path.exists(script)):
            return None
        spec = json.load(open(spec_path))
        if any(k in spec for k in ("qk_entropy", "fc1_gate", "common_write", "write_ratio", "q_sink", "fc1_bias")):
            return None
        import re
        blocks = [int(b) for b in re.search(r'--init_method_scaled_blocks "([0-9,]*)"', open(script).read()).group(1).split(",")]
        utils.apply_analytic_profile(model, spec, blocks, seed=0); return model
    dump = torch.load(path, map_location="cpu"); model.load_state_dict(dump.get("model", dump), strict=False); return model

for arm in ([x for x in a.arms.split(",") if x] or list(ARMS)):
    if arm in cache:
        continue
    model = build(arm)
    if model is None:
        print(f"[skip] {arm}: no init dump"); continue
    if images is None:
        images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), a.images, 1)
    model.to(device).eval(); parts = [PR.forward_statistics(model, images[s:s + 64].to(device)) for s in range(0, len(images), 64)]
    per_block = [{k: sum(p[b][k] for p in parts) / len(parts) for k in parts[0][b]} for b in range(12)]
    mid = lambda k: sum(per_block[b][k] for b in range(1, 8)) / 7
    E = 768; sd = model.state_dict()
    cache[arm] = {"per_block": per_block,
                  "entropy": mid("entropy"), "top_key": mid("sink_share"), "active_units": mid("gate"), "pre_activation_mean": mid("pre_activation_mean"),
                  "gelu_rms": mid("gelu_rms"), "attention_write": mid("attention_write"), "mlp_write": mid("mlp_write"), "token_cosine": mid("token_cosine"),
                  "mlp_write_b0": per_block[0]["mlp_write"], "stream_rms_b8": per_block[8]["stream_rms"],
                  "tail_attention_write": sum(per_block[b]["attention_write"] for b in (9, 10, 11)) / 3, "tail_mlp_write": sum(per_block[b]["mlp_write"] for b in (9, 10, 11)) / 3,
                  "raw_q": sum(rms(sd[f"blocks.{b}.attn.qkv.weight"][:E]) for b in range(1, 8)) / 7 / 0.02, "raw_k": sum(rms(sd[f"blocks.{b}.attn.qkv.weight"][E:2 * E]) for b in range(1, 8)) / 7 / 0.02}
    json.dump(cache, open(OUT, "w"), indent=1); print(f"[done] {arm}", flush=True)

print(f"\n{'arm':17s} {'task':5s} {'top-1':>6s} {'d rnd':>6s} | {'entropy':>7s} {'topkey':>6s} | {'active':>7s} {'preact':>6s} {'GELU':>5s} | {'attn wr':>7s} {'MLP wr':>7s} {'b0 MLP':>6s} | {'strm b8':>7s} {'tail a/m %':>10s} | {'cos':>5s} | {'raw q/k':>9s} | description")
for arm, (task, acc, text) in ARMS.items():
    if arm not in cache: continue
    c = cache[arm]; acc_s = f"{acc:6.2f}" if acc is not None else "   -  "; d = f"{acc - RANDOM:+6.2f}" if acc is not None else "   -  "
    print(f"{arm:17s} {task:5s} {acc_s} {d} | {c['entropy']:7.2f} {c['top_key']:6.2f} | {c['active_units']:7.4f} {c['pre_activation_mean']:6.2f} {c['gelu_rms']:5.2f} | {c['attention_write']:7.3f} {c['mlp_write']:7.3f} {c['mlp_write_b0']:6.1f} | "
          f"{c['stream_rms_b8']:7.1f} {100 * c['tail_attention_write']:4.1f}/{100 * c['tail_mlp_write']:4.1f} | {c['token_cosine']:5.2f} | {c['raw_q']:4.2f}/{c['raw_k']:4.2f} | {text}")
