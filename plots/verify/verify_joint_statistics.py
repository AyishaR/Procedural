"""Verify the rank-one joint statistics (spec keys "qk_entropy", "fc1_gate", "common_write"; utils.calibrate_joint_statistics).

Spec mode (CPU, no main.py):   verify_joint_statistics.py --spec vitbase_runs/profile_X.json [--images 64]
    builds a fresh timm ViT-B (seed 0), applies the profile, keeps a copy, calibrates on training images (evaluation
    transform, utils.calibration_images -- the protocol main.py uses), then checks
      1. only attn.qkv.weight (q and k rows; v rows identical) of the sink blocks and mlp.fc1.weight of the gate blocks changed
      2. the effective scales rms(W diag(gamma)) of q, k and fc1 are unchanged (what the specification declares);
         the raw rms ratio is printed for information
      3. W - s * W0 is rank one for q, k, fc1 and fc2; fc1's left and fc2's right singular vector are constant over the hidden
         units (every unit shifted / writing alike); fc2's raw rms unchanged (it has no LayerNorm in front)
      4. attention entropy, mean fc1 pre-activation and the token cosine of the common-write block's output hit their targets; forward finite; max |logit| far below fp16's 65504
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
import time
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")          # the calibration runs on the GPU in main.py (rank 0); same here when one is present
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True   # as main.py
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", len(os.sched_getaffinity(0)))))   # never more threads than the job owns
T0 = time.time(); lap = lambda: f"{time.time() - T0:.0f} s"
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--dump"); ap.add_argument("--base")
ap.add_argument("--images", type=int, default=None, help="override the number of calibration images (CPU speed)")
ap.add_argument("--data_path", default="/data/datasets/ILSVRC2012"); a = ap.parse_args()
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
spec = json.load(open(a.spec)); E = 768
joint = {k: spec[k] for k in ("qk_entropy", "fc1_gate", "common_write", "write_ratio") if k in spec}
kept = (lambda g: g) if spec.get("gain_fold", "exact") == "exact" else (lambda g: torch.ones_like(g))   # the norm the components keep (declared scale)
write = joint.get("write_ratio", {}); write_tensors = write.get("tensors", [])
attention_write_t = {int(b): float(v) for b, v in write.get("attention", {}).items()}
mlp_write_t = {int(b): float(v) for b, v in write.get("mlp", {}).items()}
sink_t = {int(b): float(v) for b, v in joint.get("qk_entropy", {}).get("entropy", {}).items()}
gate_key = "active_units" if "active_units" in joint.get("fc1_gate", {}) else "pre_activation_mean"   # what the gate is calibrated to
gate_t = {int(b): float(v) for b, v in joint.get("fc1_gate", {}).get(gate_key, {}).items()}
common_t = {int(b): float(v) for b, v in joint.get("common_write", {}).get("token_cosine", {}).items()}
n_images = a.images or max(int(p.get("images", 64)) for p in joint.values())
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
def images(seed, train_transform=False):
    return utils.calibration_images(folder.samples, folder.loader, build_transform(train_transform, args), n_images, seed).to(DEV)
if a.dump:
    load = lambda p: (lambda d: d.get("model", d))(torch.load(p, map_location="cpu"))
    after, before = load(a.dump), load(a.base); model = utils.build_model(args).to(DEV); model.load_state_dict(after, strict=False); report = None
else:
    torch.manual_seed(0); model = utils.build_model(args).to(DEV); utils.apply_analytic_profile(model, spec, sorted(int(b) for b in spec["q"]["per_block"]), seed=0)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report = utils.calibrate_joint_statistics(model, {**joint, **{k: spec[k] for k in ("gain_fold",) if k in spec}}, images(0), seed=0)
    after = model.state_dict()
    print(f"[time] model + profile + calibration on {DEV}: {lap()}")
ok = True
expected = sorted(set([f"blocks.{b}.attn.qkv.weight" for b in sink_t] + [f"blocks.{b}.mlp.fc1.weight" for b in gate_t] +
                      [f"blocks.{b}.mlp.fc2.weight" for b in common_t] +
                      [f"blocks.{b}.attn.qkv.weight" for b in attention_write_t if "v" in write_tensors] +
                      [f"blocks.{b}.attn.proj.weight" for b in attention_write_t if "proj" in write_tensors] +
                      [f"blocks.{b}.mlp.fc2.weight" for b in mlp_write_t]))
changed = sorted(k for k in after if not torch.equal(after[k], before[k]))
ok &= changed == expected
print(f"1. changed tensors: {len(changed)}; exactly the expected ones (sink {len(sink_t)} qkv, gate {len(gate_t)} fc1, common write {len(common_t)} fc2, "
      f"write ratio {write_tensors} in {len(attention_write_t)} / {len(mlp_write_t)} blocks): {changed == expected}")
rms = lambda t: float(t.float().pow(2).mean().sqrt())
def rank_one(W, W0):
    """(sigma2/sigma1 of W - s*W0 with s fitted, s, left singular vector of the rank-one part)"""
    s = float((W * W0).sum() / (W0 * W0).sum()); R = W - s * W0
    for _ in range(3):
        U, S, Vh = torch.linalg.svd(R, full_matrices=False); R1 = S[0] * torch.outer(U[:, 0], Vh[0])
        s = float(((W - R1) * W0).sum() / (W0 * W0).sum()); R = W - s * W0
    U, S, Vh = torch.linalg.svd(R, full_matrices=False)
    return float(S[1] / S[0]), s, U[:, 0], float(R.norm() / W.norm())
if sink_t:
    print("2./3. sink | blk | v rows identical | effective q, k ratio (must be 1) | raw rms q, k ratio | rank-one residual (q, k) | s_q    s_k")
    for b in sorted(sink_t):
        W, W0 = after[f"blocks.{b}.attn.qkv.weight"].float(), before[f"blocks.{b}.attn.qkv.weight"].float(); g = kept(after[f"blocks.{b}.norm1.weight"].float()[None, :])
        v_same = torch.equal(W[2 * E:], W0[2 * E:]) or "v" in write_tensors; q, k = slice(0, E), slice(E, 2 * E)   # write-matched v: checked below
        eq, ek = rms(W[q] * g) / rms(W0[q] * g), rms(W[k] * g) / rms(W0[k] * g); rq, rk = rank_one(W[q], W0[q]), rank_one(W[k], W0[k])
        ok &= v_same and abs(eq - 1) < 2e-3 and abs(ek - 1) < 2e-3 and rq[0] < 1e-3 and rk[0] < 1e-3
        print(f"           |  {b}  | {v_same!s:16s} | {eq:.5f}, {ek:.5f}                 | {rms(W[q]) / rms(W0[q]):.4f}, {rms(W[k]) / rms(W0[k]):.4f}     | {rq[0]:.1e}, {rk[0]:.1e}        | {rq[1]:.3f}  {rk[1]:.3f}")
if gate_t:
    print("2./3. gate | blk | effective fc1 ratio (must be 1) | raw rms ratio | rank-one residual | left vector constant (std/|mean|) | s     | mean-row energy share (folded)")
    for b in sorted(gate_t):
        W, W0 = after[f"blocks.{b}.mlp.fc1.weight"].float(), before[f"blocks.{b}.mlp.fc1.weight"].float(); g = after[f"blocks.{b}.norm2.weight"].float()[None, :]
        e = rms(W * kept(g)) / rms(W0 * kept(g)); r = rank_one(W, W0); const = float(r[2].std() / r[2].mean().abs()); F = W * g
        share = float(F.shape[0] * F.mean(0).pow(2).sum() / F.pow(2).sum())
        ok &= abs(e - 1) < 2e-3 and r[0] < 1e-3 and const < 1e-3
        print(f"           |  {b}  | {e:.5f}                         | {rms(W) / rms(W0):.4f}        | {r[0]:.1e}           | {const:.1e}                           | {r[1]:.3f} | {share:.3f}")
if common_t:
    print("2./3. common write | blk | raw rms ratio of fc2 (1, unless fc2 is also write-matched) | rank-one residual | right vector constant over hidden units (std/|mean|) | s      | mean-column energy share")
    for b in sorted(common_t):
        W, W0 = after[f"blocks.{b}.mlp.fc2.weight"].float(), before[f"blocks.{b}.mlp.fc2.weight"].float()
        r = rank_one(W.T, W0.T); const = float(r[2].std() / r[2].mean().abs()); share = float(W.shape[1] * W.mean(1).pow(2).sum() / W.pow(2).sum())
        joint_fc2 = b in mlp_write_t                # fc2 solved for write ratio AND token cosine: its scale is an outcome, not preserved
        no_component = joint_fc2 and r[3] < 1e-6    # joint solve kept m = 0 (target already exceeded): fc2 is a pure scalar multiple
        ok &= (joint_fc2 or abs(rms(W) / rms(W0) - 1) < 2e-3) and (no_component or (r[0] < 1e-3 and const < 1e-3))
        print(f"                   |  {b}  | {rms(W) / rms(W0):.5f}                          | {r[0]:.1e}           | {const:.1e}                                                | {r[1]:.4f} | {share:.4f}")
if write:
    print("2./3. write ratio | blk | factor per tensor (tensor = factor * second-moment-stage tensor; relative residual) | q, k rows identical")
    for b in sorted(set(attention_write_t) | set(mlp_write_t)):
        W, W0 = after[f"blocks.{b}.attn.qkv.weight"].double(), before[f"blocks.{b}.attn.qkv.weight"].double()
        named = {"v": (W[2 * E:], W0[2 * E:]), "proj": (after[f"blocks.{b}.attn.proj.weight"].double(), before[f"blocks.{b}.attn.proj.weight"].double()),
                 "fc2": (after[f"blocks.{b}.mlp.fc2.weight"].double(), before[f"blocks.{b}.mlp.fc2.weight"].double())}
        cells = []
        for name in write_tensors:
            f_ = float((named[name][0] * named[name][1]).sum() / named[name][1].pow(2).sum()); res = float((named[name][0] - f_ * named[name][1]).norm() / named[name][0].norm())
            if name == "fc2" and b in common_t:     # scale + rank-one component, checked in the common-write table
                cells.append(f"fc2 x{f_:.3f} + common write"); continue
            ok &= res < 1e-6; cells.append(f"{name} x{f_:.3f} ({res:.0e})")
        qk_same = bool(torch.equal(W[:2 * E], W0[:2 * E])) or b in sink_t; ok &= qk_same
        print(f"                  |  {b}  | " + "  ".join(cells) + f" | {qk_same}")
print(f"[time] tensor checks: {lap()}")
ALL = sorted(set(sink_t) | set(gate_t) | set(common_t) | set(attention_write_t) | set(mlp_write_t))
B0, B1 = utils.joint_statistics_per_block(model, images(0), ALL), utils.joint_statistics_per_block(model, images(1), ALL)
B2 = utils.joint_statistics_per_block(model, images(2, train_transform=True), ALL)
with torch.no_grad():
    finite = bool(torch.isfinite(model(images(0)[:8])).all())
ok &= finite
print(f"4./5. behaviour on calibration images / unseen draw / training augmentation | target   (forward finite: {finite})")
for b in ALL:
    line = f"  b{b}:"
    if b in attention_write_t:
        ok &= abs(B0[b]["attention_write"] / attention_write_t[b] - 1) < 2e-3
        line += f" attention write {B0[b]['attention_write']:.4f} / {B1[b]['attention_write']:.4f} / {B2[b]['attention_write']:.4f} | {attention_write_t[b]:.4f}"
    if b in mlp_write_t:
        ok &= abs(B0[b]["mlp_write"] / mlp_write_t[b] - 1) < 2e-3
        line += f"   MLP write {B0[b]['mlp_write']:.4f} / {B1[b]['mlp_write']:.4f} / {B2[b]['mlp_write']:.4f} | {mlp_write_t[b]:.4f}"
    if b in sink_t:
        ok &= abs(B0[b]["entropy"] - sink_t[b]) < 0.02
        line += f" entropy {B0[b]['entropy']:.3f} / {B1[b]['entropy']:.3f} / {B2[b]['entropy']:.3f} | {sink_t[b]:.3f}  (top key {B0[b]['sink_share']:.2f})"
    if b in common_t:
        exceeded = bool(report and report.get(b, {}).get("common_write_at_ratio", {}).get("exceeded_without_component")) or \
                   (report is None and b in mlp_write_t and B0[b]["token_cosine"] > common_t[b] + 0.005)
        ok &= exceeded or abs(B0[b]["token_cosine"] - common_t[b]) < 0.005
        line += " [token cosine target EXCEEDED by the write-matched random fc2; no component added]" if exceeded else ""
        line += f" token cosine of the output {B0[b]['token_cosine']:.3f} / {B1[b]['token_cosine']:.3f} / {B2[b]['token_cosine']:.3f} | {common_t[b]:.3f}"
    if b in gate_t:
        ok &= (abs(B0[b]["active_units"] - gate_t[b]) < max(2e-5, 0.03 * gate_t[b])) if gate_key == "active_units" else (abs(B0[b]["pre_activation_mean"] - gate_t[b]) < 0.02)
        line += f"   active units {B0[b]['active_units']:.5f} / {B1[b]['active_units']:.5f} / {B2[b]['active_units']:.5f}" + (f" | {gate_t[b]:.5f}" if gate_key == "active_units" else "")
        line += f"   mean pre-activation {B0[b]['pre_activation_mean']:+.3f} / {B1[b]['pre_activation_mean']:+.3f} / {B2[b]['pre_activation_mean']:+.3f} " + (f" | {gate_t[b]:+.3f}" if gate_key == "pre_activation_mean" else " (diagnostic)")
    print(line)
if report:
    for b, parts in report.items():
        ok &= all(p["reachable"] or p.get("exceeded_without_component") for p in parts.values())
        describe = {"qk_entropy": lambda p: f"sink alpha {p['alpha']:.2f} max |logit| {p['max_abs_logit']:.0f}",
                    "fc1_gate": lambda p: f"gate beta {p['beta']:.2f} GELU rms {p['gelu_rms']:.3f}",
                    "common_write": lambda p: f"common write beta {p['beta']:.2f} MLP write ratio {p['mlp_write_ratio']:.2f} (common share {p['common_share_of_write']:.2f})",
                    "common_write_at_ratio": lambda p: f"fc2 jointly: m {p['m']:.3f} c {p['c']:.2f} -> token cosine {p['token_cosine']:.3f} (without component {p['token_cosine_without_component']:.3f}), MLP write {p['mlp_write_ratio']:.3f}, common share {p['common_share_of_write']:.2f}",
                    "write_ratio_attention": lambda p: f"attention write {p['before']:.4f} -> {p['write_ratio']:.4f} ({'/'.join(p['tensors'])} x{p['factor_per_tensor']:.3f}{' each' if len(p['tensors']) > 1 else ''})",
                    "write_ratio_mlp": lambda p: f"MLP write {p['before']:.4f} -> {p['write_ratio']:.4f} (fc2 x{p['factor_per_tensor']:.3f})"}
        print(f"  calibration b{b}: " + "  ".join(describe[k](p) + ("" if p["reachable"] else (" EXCEEDED" if p.get("exceeded_without_component") else " UNREACHABLE")) for k, p in parts.items()))
print(f"[time] total: {lap()} on {DEV}")
print("VERDICT:", "PASS" if ok else "FAIL")
