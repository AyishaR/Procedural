"""Verify the rank-one joint statistics (spec keys "qk_sink" and "fc1_gate", utils.calibrate_joint_statistics).

Spec mode (CPU, no main.py):   verify_joint_statistics.py --spec vitbase_runs/profile_X.json [--images 64]
    builds a fresh timm ViT-B (seed 0), applies the profile, keeps a copy, calibrates on training images (evaluation
    transform, utils.calibration_images -- the protocol main.py uses), then checks
      1. only attn.qkv.weight (q and k rows; v rows identical) of the sink blocks and mlp.fc1.weight of the gate blocks changed
      2. the effective scales rms(W diag(gamma)) of q, k and fc1 are unchanged (what the specification declares);
         the raw rms ratio is printed for information
      3. W - s * W0 is rank one for q, k and fc1; for fc1 its left singular vector is constant (every hidden unit shifted alike)
      4. attention entropy and mean fc1 pre-activation hit their targets; forward finite; max |logit| far below fp16's 65504
      5. behaviour on images the calibration never saw (another seeded draw, and the training augmentation)
Dump mode (after a dump through main.py):   verify_joint_statistics.py --spec ... --dump ARM_s0.pth --base BASE_s0.pth
    checks 1-5 on the dumped weights against the base arm's dump (same seed, no joint statistics). Pass the --data_path the
    dump was calibrated on (results/init_dumps/imnet_small for the dump jobs), otherwise check 4 compares different images."""
import argparse, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
from torchvision import datasets as tv_datasets
import main as M, utils
from datasets import build_transform
torch.set_num_threads(32)
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--dump"); ap.add_argument("--base")
ap.add_argument("--images", type=int, default=None, help="override the number of calibration images (CPU speed)")
ap.add_argument("--data_path", default="/data/datasets/ILSVRC2012"); a = ap.parse_args()
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
spec = json.load(open(a.spec)); E = 768
joint = {k: spec[k] for k in ("qk_sink", "fc1_gate") if k in spec}
sink_t = {int(b): float(v) for b, v in joint.get("qk_sink", {}).get("entropy", {}).items()}
gate_t = {int(b): float(v) for b, v in joint.get("fc1_gate", {}).get("pre_activation_mean", {}).items()}
n_images = a.images or max(int(p.get("images", 64)) for p in joint.values())
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
def images(seed, train_transform=False):
    return utils.calibration_images(folder.samples, folder.loader, build_transform(train_transform, args), n_images, seed)
if a.dump:
    load = lambda p: (lambda d: d.get("model", d))(torch.load(p, map_location="cpu"))
    after, before = load(a.dump), load(a.base); model = utils.build_model(args); model.load_state_dict(after, strict=False); report = None
else:
    torch.manual_seed(0); model = utils.build_model(args); utils.apply_analytic_profile(model, spec, list(range(9)), seed=0)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report = utils.calibrate_joint_statistics(model, joint, images(0), seed=0)
    after = model.state_dict()
ok = True
expected = sorted([f"blocks.{b}.attn.qkv.weight" for b in sink_t] + [f"blocks.{b}.mlp.fc1.weight" for b in gate_t])
changed = sorted(k for k in after if not torch.equal(after[k], before[k]))
ok &= changed == expected
print(f"1. changed tensors: {len(changed)}; exactly the {len(sink_t)} qkv weights and {len(gate_t)} fc1 weights expected: {changed == expected}")
rms = lambda t: float(t.float().pow(2).mean().sqrt())
def rank_one(W, W0):
    """(sigma2/sigma1 of W - s*W0 with s fitted, s, left singular vector of the rank-one part)"""
    s = float((W * W0).sum() / (W0 * W0).sum()); R = W - s * W0
    for _ in range(3):
        U, S, Vh = torch.linalg.svd(R, full_matrices=False); R1 = S[0] * torch.outer(U[:, 0], Vh[0])
        s = float(((W - R1) * W0).sum() / (W0 * W0).sum()); R = W - s * W0
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    return float(S[1] / S[0]), s, U[:, 0]
if sink_t:
    print("2./3. sink | blk | v rows identical | effective q, k ratio (must be 1) | raw rms q, k ratio | rank-one residual (q, k) | s_q    s_k")
    for b in sorted(sink_t):
        W, W0 = after[f"blocks.{b}.attn.qkv.weight"].float(), before[f"blocks.{b}.attn.qkv.weight"].float(); g = after[f"blocks.{b}.norm1.weight"].float()[None, :]
        v_same = torch.equal(W[2 * E:], W0[2 * E:]); q, k = slice(0, E), slice(E, 2 * E)
        eq, ek = rms(W[q] * g) / rms(W0[q] * g), rms(W[k] * g) / rms(W0[k] * g); rq, rk = rank_one(W[q], W0[q]), rank_one(W[k], W0[k])
        ok &= v_same and abs(eq - 1) < 2e-3 and abs(ek - 1) < 2e-3 and rq[0] < 1e-3 and rk[0] < 1e-3
        print(f"           |  {b}  | {v_same!s:16s} | {eq:.5f}, {ek:.5f}                 | {rms(W[q]) / rms(W0[q]):.4f}, {rms(W[k]) / rms(W0[k]):.4f}     | {rq[0]:.1e}, {rk[0]:.1e}        | {rq[1]:.3f}  {rk[1]:.3f}")
if gate_t:
    print("2./3. gate | blk | effective fc1 ratio (must be 1) | raw rms ratio | rank-one residual | left vector constant (std/|mean|) | s     | mean-row energy share (folded)")
    for b in sorted(gate_t):
        W, W0 = after[f"blocks.{b}.mlp.fc1.weight"].float(), before[f"blocks.{b}.mlp.fc1.weight"].float(); g = after[f"blocks.{b}.norm2.weight"].float()[None, :]
        e = rms(W * g) / rms(W0 * g); r = rank_one(W, W0); const = float(r[2].std() / r[2].mean().abs()); F = W * g
        share = float(F.shape[0] * F.mean(0).pow(2).sum() / F.pow(2).sum())
        ok &= abs(e - 1) < 2e-3 and r[0] < 1e-3 and const < 1e-3
        print(f"           |  {b}  | {e:.5f}                         | {rms(W) / rms(W0):.4f}        | {r[0]:.1e}           | {const:.1e}                           | {r[1]:.3f} | {share:.3f}")
B0, B1 = utils.joint_statistics_per_block(model, images(0), sorted(set(sink_t) | set(gate_t))), utils.joint_statistics_per_block(model, images(1), sorted(set(sink_t) | set(gate_t)))
B2 = utils.joint_statistics_per_block(model, images(2, train_transform=True), sorted(set(sink_t) | set(gate_t)))
with torch.no_grad():
    finite = bool(torch.isfinite(model(images(0)[:8])).all())
ok &= finite
print(f"4./5. behaviour on calibration images / unseen draw / training augmentation | target   (forward finite: {finite})")
for b in sorted(set(sink_t) | set(gate_t)):
    line = f"  b{b}:"
    if b in sink_t:
        ok &= abs(B0[b]["entropy"] - sink_t[b]) < 0.02
        line += f" entropy {B0[b]['entropy']:.3f} / {B1[b]['entropy']:.3f} / {B2[b]['entropy']:.3f} | {sink_t[b]:.3f}  (top key {B0[b]['sink_share']:.2f})"
    if b in gate_t:
        ok &= abs(B0[b]["pre_activation_mean"] - gate_t[b]) < 0.02
        line += f"   mean pre-activation {B0[b]['pre_activation_mean']:+.3f} / {B1[b]['pre_activation_mean']:+.3f} / {B2[b]['pre_activation_mean']:+.3f} | {gate_t[b]:+.3f}  (active units {B0[b]['active_units']:.4f})"
    print(line)
if report:
    for b, parts in report.items():
        ok &= all(p["reachable"] for p in parts.values())
        print(f"  calibration b{b}: " + "  ".join((f"sink alpha {p['alpha']:.2f} max |logit| {p['max_abs_logit']:.0f}" if k == "qk_sink" else f"gate beta {p['beta']:.2f} GELU rms {p['gelu_rms']:.3f}") + ("" if p["reachable"] else " UNREACHABLE") for k, p in parts.items()))
print("VERDICT:", "PASS" if ok else "FAIL")
