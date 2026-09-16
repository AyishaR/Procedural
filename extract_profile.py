"""Extract the checkpoint-free early-lever recipe from a procedural checkpoint.

For blocks 0..8 (default) and each slice q, k, v, proj, fc1, fc2 the *effective* scale
    rms(gamma of the preceding LayerNorm) * rms(W) / timm_std       (gamma folded in for q, k, v with norm1 and fc1 with norm2;
                                                                     proj and fc2 have no LayerNorm in front: raw rms)
is measured; block 0 is kept as its own number and blocks 1..8 are fitted by least squares with a straight line
(start at block 1, end at block 8) -> 18 numbers. The four LayerNorm vectors of each block contribute their mean and std
-> 36 numbers. Together: the `ftbanap` form of the recipe, written as a spec for `utils.apply_analytic_profile`
(`--init_method analytic_profile --profile_spec <out> --init_method_scaled_blocks 0,...,8`). With `--exact` the six
slices get their measured per-block values instead of the ramp (54 numbers, the `ftbanakx` form).

The kdyck spec in use (profile_ftbanap.json) additionally carries two manual corrections found on forward-pass dumps:
q and k flat at 1.32 (the checkpoint's q/k columns are anti-correlated with gamma, so the folded rms overstates the logit
scale) and fc2 ending at 0.95 instead of the fitted value (block 8 already turns toward the loud top). Reproduce them
with --qk_flat 1.32 --fc2_end 0.95; this script otherwise reports the raw measurement and its ramp misfit.

usage: .venv/bin/python extract_profile.py CKPT OUT.json [--blocks 0-8] [--exact] [--qk_flat X] [--fc2_end Y]
   e.g. .venv/bin/python extract_profile.py results/pr_vitb_n/pr_6066174_final.pth /tmp/kdyck.json --qk_flat 1.32 --fc2_end 0.95"""
import sys, json, argparse, torch, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--blocks", default="0-8")
ap.add_argument("--exact", action="store_true"); ap.add_argument("--qk_flat", type=float, default=None); ap.add_argument("--fc2_end", type=float, default=None)
ap.add_argument("--timm_std", type=float, default=0.02); ap.add_argument("--no_ln", action="store_true", help="18 numbers only (gains 1, biases 0): the ftbana form")
a = ap.parse_args()
lo, hi = (int(x) for x in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = ck.get("state", ck.get("model", ck))
E = sd["blocks.0.attn.qkv.weight"].shape[1]
rms = lambda t: float(t.float().pow(2).mean().sqrt())
eff, stats = {s: {} for s in ("q", "k", "v", "proj", "fc1", "fc2")}, {}
for b in blocks:
    W = sd[f"blocks.{b}.attn.qkv.weight"]; g1 = rms(sd[f"blocks.{b}.norm1.weight"]); g2 = rms(sd[f"blocks.{b}.norm2.weight"])
    eff["q"][b] = g1 * rms(W[:E]) / a.timm_std; eff["k"][b] = g1 * rms(W[E:2 * E]) / a.timm_std; eff["v"][b] = g1 * rms(W[2 * E:]) / a.timm_std
    eff["proj"][b] = rms(sd[f"blocks.{b}.attn.proj.weight"]) / a.timm_std
    eff["fc1"][b] = g2 * rms(sd[f"blocks.{b}.mlp.fc1.weight"]) / a.timm_std; eff["fc2"][b] = rms(sd[f"blocks.{b}.mlp.fc2.weight"]) / a.timm_std
    stats[str(b)] = {f"norm{i}": {"gain_mean": float(sd[f"blocks.{b}.norm{i}.weight"].float().mean()), "gain_std": float(sd[f"blocks.{b}.norm{i}.weight"].float().std()),
                                  "bias_mean": float(sd[f"blocks.{b}.norm{i}.bias"].float().mean()), "bias_std": float(sd[f"blocks.{b}.norm{i}.bias"].float().std())} for i in (1, 2)}
spec = {}
ramp_blocks = [b for b in blocks if b != blocks[0]]
print(f"{a.ckpt}: effective scales (rms(gamma)*rms(W)/{a.timm_std}) and the ramp fit over blocks {ramp_blocks[0]}-{ramp_blocks[-1]}")
print("slice | " + " ".join(f"b{b:<5d}" for b in blocks) + " | b0   start -> end   | max ramp misfit")
for s, vals in eff.items():
    y = np.array([vals[b] for b in ramp_blocks]); x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1); start, end = icpt, icpt + slope * (len(y) - 1)
    if s in ("q", "k") and a.qk_flat is not None: start = end = a.qk_flat
    if s == "fc2" and a.fc2_end is not None: end = a.fc2_end
    fit = start + (end - start) * x / max(1, len(y) - 1); misfit = float(np.max(np.abs(fit / y - 1)))
    spec[s] = {"per_block": {str(b): round(vals[b], 4) for b in blocks}} if a.exact else {"b0": round(vals[blocks[0]], 3), "start": round(float(start), 3), "end": round(float(end), 3)}
    print(f"{s:5s} | " + " ".join(f"{vals[b]:5.2f} " for b in blocks) + f" | {vals[blocks[0]]:4.2f}  {start:5.2f} -> {end:5.2f} | {misfit:5.1%}")
if not a.no_ln:
    spec["ln"] = {"gain": True, "bias": True, "source": "parametric", "stats": stats, "ckpt": a.ckpt}
    print("LayerNorm statistics (mean/std of gain, bias), norm1 / norm2:")
    for b in blocks:
        n1, n2 = stats[str(b)]["norm1"], stats[str(b)]["norm2"]
        print(f"  b{b}: gain {n1['gain_mean']:.3f}+-{n1['gain_std']:.3f} bias {n1['bias_mean']:+.3f}+-{n1['bias_std']:.3f} / gain {n2['gain_mean']:.3f}+-{n2['gain_std']:.3f} bias {n2['bias_mean']:+.3f}+-{n2['bias_std']:.3f}")
json.dump(spec, open(a.out, "w"), indent=1)
n = (len(blocks) * 6 if a.exact else 18) + (0 if a.no_ln else 4 * 2 * len(blocks) // 2 * 1)
print(f"wrote {a.out}: {'exact per-block' if a.exact else 'block 0 + ramp'} scales" + ("" if a.no_ln else f" + {4*len(blocks)} LayerNorm statistics inline (no checkpoint needed at init)"))
