"""Check, on the real code path, every equation that docs/proc_init_recipe.md states about the statistics-only initialisation.

Nothing here re-uses the installers' internals: directions, scales and targets are recomputed from their definitions and
compared with what utils.apply_analytic_profile + utils.calibrate_joint_statistics actually produced.

  A  timm initialisation: rms of every block weight, LayerNorm gains 1 / biases 0, linear biases 0
  B  second-moment part: W_s = m_s * W_timm elementwise; rms(W_s) rms(gamma) / sigma0 = e_{b,s} (q, k, v with norm1, fc1 with
     norm2), rms(W_s) / sigma0 = e_{b,s} (proj, fc2); the exactly folded rms(W_s diag gamma) / sigma0 next to it; sampled
     LayerNorm vectors have the specification's mean / std
  C  specification numbers == independent recomputation from the checkpoint (effective scales, LayerNorm statistics)
  D  joint statistics: W' = s W + a * (left vector)(right vector)^T with the DEFINED vectors, residual ~ 0;
     rms(W' diag gamma) = rms(W diag gamma); targets met on the calibration images
usage: .venv/bin/python plots/verify/verify_recipe_statement.py CHECKPOINT SPEC.json [--images 64]"""
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
ap = argparse.ArgumentParser(); ap.add_argument("checkpoint"); ap.add_argument("spec"); ap.add_argument("--images", type=int, default=64)
ap.add_argument("--data_path", default="/data/datasets/ILSVRC2012"); a = ap.parse_args()
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
spec = json.load(open(a.spec)); S0 = 0.02; E = 768; ok = True
BLOCKS = sorted(int(b) for b in spec["q"]["per_block"])  # the blocks the specification covers (0..8 or 0..7)
assert BLOCKS == list(range(len(BLOCKS))), BLOCKS
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True); P = ck.get("state", ck.get("model", ck))
rms = lambda t: float(t.detach().float().pow(2).mean().sqrt())
def slices(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"]
    return {"q": W[:E], "k": W[E:2 * E], "v": W[2 * E:], "proj": sd[f"blocks.{b}.attn.proj.weight"], "fc1": sd[f"blocks.{b}.mlp.fc1.weight"], "fc2": sd[f"blocks.{b}.mlp.fc2.weight"]}
NORM = {"q": "norm1", "k": "norm1", "v": "norm1", "fc1": "norm2"}
def check(name, cond, detail):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}: {detail}")

torch.manual_seed(0); model = utils.build_model(args).to(DEV); model.eval(); timm = {k: v.detach().clone() for k, v in model.state_dict().items()}
print("A. timm initialisation")
r = [rms(w) / S0 for b in range(12) for w in slices(timm, b).values()]
check("rms of all 72 block weights / 0.02", max(abs(x - 1) for x in r) < 0.01, f"min {min(r):.4f} max {max(r):.4f}")
check("LayerNorm gains 1, biases 0; linear biases 0", all(float((timm[f'blocks.{b}.norm{i}.weight'] - 1).abs().max()) == 0 and float(timm[f'blocks.{b}.norm{i}.bias'].abs().max()) == 0 for b in range(12) for i in (1, 2))
      and all(float(timm[f'blocks.{b}.{n}'].abs().max()) == 0 for b in range(12) for n in ("attn.qkv.bias", "attn.proj.bias", "mlp.fc1.bias", "mlp.fc2.bias")), "exact")

utils.apply_analytic_profile(model, spec, BLOCKS, seed=0); second = {k: v.detach().clone() for k, v in model.state_dict().items()}
print("B. second-moment part (utils.apply_analytic_profile)")
dev_prod, dev_fold, dev_elem, ln_dev = 0, 0, 0, 0
SCALED = [s for s in ("q", "k", "v", "proj", "fc1", "fc2") if s in spec]; UNSCALED = [s for s in ("q", "k", "v", "proj", "fc1", "fc2") if s not in spec]
unscaled_same = all(torch.equal(slices(second, b)[s], slices(timm, b)[s]) for b in BLOCKS for s in UNSCALED)
for b in BLOCKS:
    for s, W in slices(second, b).items():
        if s in UNSCALED: continue
        e = spec[s]["per_block"][str(b)] if "per_block" in spec[s] else None
        g = second[f"blocks.{b}.{NORM[s]}.weight"].float() if s in NORM else None
        prod = rms(W) * (rms(g) if g is not None else 1.0) / S0; fold = rms(W.float() * g[None, :]) / S0 if g is not None else prod
        dev_prod = max(dev_prod, abs(prod / e - 1)); dev_fold = max(dev_fold, abs(fold / e - 1))
        ratio = W.float() / slices(timm, b)[s].float(); dev_elem = max(dev_elem, float((ratio - ratio.mean()).abs().max() / ratio.mean().abs()))
    for i in (1, 2):
        st = spec["ln"]["stats"][str(b)][f"norm{i}"]; g, be = second[f"blocks.{b}.norm{i}.weight"].float(), second[f"blocks.{b}.norm{i}.bias"].float()
        ln_dev = max(ln_dev, abs(float(g.mean()) - st["gain_mean"]) / st["gain_mean"], abs(float(g.std()) - st["gain_std"]) / st["gain_std"], abs(float(be.std()) - st["bias_std"]) / st["bias_std"])
check(f"weights without an effective scale in the specification {UNSCALED} are timm's, bit for bit", unscaled_same, f"scaled: {SCALED}")
check("W_s = m * W_timm elementwise (one scalar per slice)", dev_elem < 1e-4, f"max relative spread of W_s / W_timm within a slice {dev_elem:.1e}")
check("rms(W_s) rms(gamma) / 0.02 = e_{b,s}", dev_prod < 0.01, f"max deviation {dev_prod:.4f} (residual = timm's own rms not being exactly 0.02)")
check("exactly folded rms(W_s diag gamma) / 0.02 vs e_{b,s}", dev_fold < 0.01, f"max deviation {dev_fold:.4f} (W and gamma independent, so product and fold agree)")
if spec.get("realise") == "exact":
    declared_dev = dev_fold if spec.get("gain_fold", "exact") == "exact" else dev_prod
    check(f"'realise': 'exact': the declared scale ({spec.get('gain_fold', 'exact')} folding; rms(W) for proj, fc2) holds to float precision", declared_dev < 1e-5, f"max relative deviation {declared_dev:.1e}")
check("sampled LayerNorm vectors vs specified mean / std", ln_dev < 0.08, f"max relative deviation {ln_dev:.3f} (768 samples: ~0.04 expected)")
untouched = all(torch.equal(second[k], timm[k]) for k in timm if not (k.startswith("blocks.") and int(k.split(".")[1]) in BLOCKS))
changed_kinds = sorted(set(k.split(".", 2)[2] for k in timm if k.startswith("blocks.") and int(k.split(".")[1]) in BLOCKS and not torch.equal(second[k], timm[k])))
check(f"only blocks {BLOCKS[0]}-{BLOCKS[-1]} touched; tensors changed there", untouched, str(changed_kinds))

print("C. specification == independent recomputation from the checkpoint")
dev_e, dev_ln = 0, 0
for b in BLOCKS:
    for s, W in slices(P, b).items():
        if s in UNSCALED: continue
        g = P[f"blocks.{b}.{NORM[s]}.weight"].float() if s in NORM else None
        e_ck = (rms(W.float() * g[None, :]) if g is not None else rms(W)) / S0; dev_e = max(dev_e, abs(spec[s]["per_block"][str(b)] / e_ck - 1))
    for i in (1, 2):
        st = spec["ln"]["stats"][str(b)][f"norm{i}"]; g, be = P[f"blocks.{b}.norm{i}.weight"].float(), P[f"blocks.{b}.norm{i}.bias"].float()
        dev_ln = max(dev_ln, abs(st["gain_mean"] - float(g.mean())), abs(st["gain_std"] - float(g.std())), abs(st["bias_mean"] - float(be.mean())), abs(st["bias_std"] - float(be.std())))
check("e_{b,s} = rms(W^ckpt_s diag gamma^ckpt) / 0.02  (proj, fc2: rms(W^ckpt_s) / 0.02)", dev_e < 1e-3, f"max relative deviation {dev_e:.1e}")
check("LayerNorm statistics = mean / std of the checkpoint's vectors", dev_ln < 1e-6, f"max absolute deviation {dev_ln:.1e}")

joint = {k: spec[k] for k in ("qk_entropy", "fc1_gate", "common_write", "write_ratio") if k in spec}
# the norm a rank-one component must keep: the declared scale, i.e. folded W diag(gamma) for "gain_fold": "exact", raw W for "product"
kept = (lambda g: g) if spec.get("gain_fold", "exact") == "exact" else (lambda g: torch.ones_like(g))
if "write_ratio" in spec:
    # targets recomputed without utils.sublayer_write_ratios: prefix model = fresh timm (seed 0) + checkpoint tensors in BLOCKS,
    # the specification's own image count and protocol
    wr = spec["write_ratio"]; folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
    seeds_t = wr.get("seeds", [0]); got = {"attention": {}, "mlp": {}}      # the targets are means over these random contexts (model seed = image seed)
    for seed_t in seeds_t:
        imgs_t = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), int(wr["images"]), seed_t).to(DEV)
        torch.manual_seed(seed_t); prefix = utils.build_model(args).to(DEV); prefix.eval()
        prefix.load_state_dict({k: v for k, v in P.items() if k.startswith("blocks.") and int(k.split(".")[1]) in BLOCKS}, strict=False)
        with torch.no_grad():
            x = utils.block_input_stream(prefix, imgs_t)
            for b, blk in enumerate(prefix.blocks):
                if b > BLOCKS[-1]: break
                att = blk.attn(blk.norm1(x)); mid = x + att; mlp = blk.mlp(blk.norm2(mid))
                got["attention"].setdefault(b, []).append(float((att.norm(dim=-1) / x.norm(dim=-1)).mean())); got["mlp"].setdefault(b, []).append(float((mlp.norm(dim=-1) / mid.norm(dim=-1)).mean()))
                x = mid + mlp
    dev_w = max(abs(sum(got[sub][int(b)]) / len(seeds_t) / wr[sub][b] - 1) for sub in ("attention", "mlp") for b in wr.get(sub, {}))
    check("write-ratio targets = mean_tokens ||sublayer output|| / ||its input stream|| of the checkpoint prefix, averaged over the spec's seeds", dev_w < 1e-3, f"max relative deviation {dev_w:.1e} over {len(wr.get('attention', {})) + len(wr.get('mlp', {}))} targets ({wr['images']} images x seeds {seeds_t})")
if joint:
    folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
    imgs = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), a.images, 0).to(DEV)
    report = utils.calibrate_joint_statistics(model, {**joint, **{k: spec[k] for k in ("gain_fold",) if k in spec}}, imgs, seed=0); final = model.state_dict()
    print(f"D. joint statistics (utils.calibrate_joint_statistics, {a.images} training images)   [time] {lap()} on {DEV}")
    def fit(Wn, W0, left, right):
        """least squares (s, a) in Wn = s W0 + a left right^T via the 2x2 normal equations in float64; relative residual"""
        Wn, W0, D_ = Wn.double(), W0.double(), torch.outer(left.double().to(Wn.device), right.double().to(Wn.device))
        G = torch.tensor([[float((W0 * W0).sum()), float((W0 * D_).sum())], [float((W0 * D_).sum()), float((D_ * D_).sum())]], dtype=torch.float64)
        rhs = torch.tensor([float((Wn * W0).sum()), float((Wn * D_).sum())], dtype=torch.float64)
        s_, a_ = torch.linalg.solve(G, rhs).tolist()
        return s_, a_, float((Wn - s_ * W0 - a_ * D_).norm() / Wn.norm())
    with torch.no_grad():
        stream = utils.block_input_stream(model, imgs); res_max, eff_max, tgt = 0, 0, []
        wr_res, wr_split, wr_tgt, wr_factors = 0, 0, [], {}
        gate_dev = 0
        notes = []
        for b, blk in enumerate(model.blocks):
            if b > BLOCKS[-1]: break
            if str(b) in joint.get("qk_entropy", {}).get("entropy", {}):
                c1 = torch.nn.functional.normalize(blk.norm1(stream).mean((0, 1)), dim=0)
                Pv = utils.sink_directions(12, 64, 0, b).reshape(E); r_ = torch.nn.functional.normalize(torch.randn(E, generator=torch.Generator().manual_seed(3000 + b)), dim=0)
                g = kept(final[f"blocks.{b}.norm1.weight"].float()[None, :])
                for sl, right in ((slice(0, E), c1), (slice(E, 2 * E), r_)):
                    Wn, W0 = final[f"blocks.{b}.attn.qkv.weight"][sl].float(), second[f"blocks.{b}.attn.qkv.weight"][sl].float()
                    s_, a_, res = fit(Wn, W0, Pv, right); res_max = max(res_max, res); eff_max = max(eff_max, abs(rms(Wn * g) / rms(W0 * g) - 1))
                pr, _, _ = utils._attention_rows(blk, stream, blk.attn.qkv.weight); tgt.append(abs(float(-(pr * (pr + 1e-12).log()).sum(-1).mean()) - joint["qk_entropy"]["entropy"][str(b)]))
            gate_key = "active_units" if "active_units" in joint.get("fc1_gate", {}) else "pre_activation_mean"
            if str(b) in joint.get("fc1_gate", {}).get(gate_key, {}):
                y = utils.fc1_input(blk, stream); c2 = torch.nn.functional.normalize(y.mean((0, 1)), dim=0); n = 3072
                Wn, W0 = final[f"blocks.{b}.mlp.fc1.weight"].float(), second[f"blocks.{b}.mlp.fc1.weight"].float(); g = kept(final[f"blocks.{b}.norm2.weight"].float()[None, :])
                s_, a_, res = fit(Wn, W0, -torch.ones(n) / n ** 0.5, c2); res_max = max(res_max, res); eff_max = max(eff_max, abs(rms(Wn * g) / rms(W0 * g) - 1))
                if gate_key == "active_units":      # fraction of positive pre-activations; relative tolerance, the targets span 1e-4 .. 0.15
                    reached, wanted = float((blk.mlp.fc1(y) > 0).float().mean()), joint["fc1_gate"]["active_units"][str(b)]
                    gate_dev = max(gate_dev, abs(reached - wanted) / max(wanted, 1e-3))
                else:
                    tgt.append(abs(float(blk.mlp.fc1(y).mean()) - joint["fc1_gate"]["pre_activation_mean"][str(b)]))
            if str(b) in joint.get("common_write", {}).get("token_cosine", {}):
                n = 3072; u_ = torch.nn.functional.normalize(torch.randn(E, generator=torch.Generator().manual_seed(5000 + b)), dim=0)
                Wn, W0 = final[f"blocks.{b}.mlp.fc2.weight"].float(), second[f"blocks.{b}.mlp.fc2.weight"].float()
                s_, a_, res = fit(Wn, W0, u_, torch.ones(n) / n ** 0.5); res_max = max(res_max, res)
                joint_fc2 = str(b) in joint.get("write_ratio", {}).get("mlp", {})     # fc2 solved for write ratio AND token cosine: its scale is an outcome
                if not joint_fc2: eff_max = max(eff_max, abs(rms(Wn) / rms(W0) - 1))
                exceeded = bool(report.get(b, {}).get("common_write_at_ratio", {}).get("exceeded_without_component"))
                if exceeded: notes.append(f"block {b}: token cosine {utils._token_cosine(blk(stream)):.3f} exceeds the target {joint['common_write']['token_cosine'][str(b)]} with the write-matched random fc2 alone; no component added (a = {a_:.1e})")
                else: tgt.append(abs(utils._token_cosine(blk(stream)) - joint["common_write"]["token_cosine"][str(b)]))
                if joint_fc2: notes.append(f"block {b}: fc2 = {s_:.3f} * W + {a_:.2f} * u 1^T/sqrt(n) (two targets, two numbers), relative residual {res:.1e}")
            if "write_ratio" in joint:
                wr = joint["write_ratio"]; att = blk.attn(blk.norm1(stream)); mid = stream + att; mlp = blk.mlp(blk.norm2(mid))
                named = {"v": (final[f"blocks.{b}.attn.qkv.weight"][2 * E:], second[f"blocks.{b}.attn.qkv.weight"][2 * E:]),
                         "proj": (final[f"blocks.{b}.attn.proj.weight"], second[f"blocks.{b}.attn.proj.weight"]),
                         "fc2": (final[f"blocks.{b}.mlp.fc2.weight"], second[f"blocks.{b}.mlp.fc2.weight"])}
                factors = {}
                for name in wr["tensors"]:
                    Wn, W0 = named[name][0].double(), named[name][1].double(); f_ = float((Wn * W0).sum() / (W0 * W0).sum())
                    factors[name] = f_
                    if not (name == "fc2" and str(b) in joint.get("common_write", {}).get("token_cosine", {})):   # that fc2 also carries the rank-one part, fitted above
                        wr_res = max(wr_res, float((Wn - f_ * W0).norm() / Wn.norm()))
                if "v" in factors: wr_split = max(wr_split, abs(factors["v"] / factors["proj"] - 1))
                if str(b) in wr.get("attention", {}): wr_tgt.append(abs(float((att.norm(dim=-1) / stream.norm(dim=-1)).mean()) / wr["attention"][str(b)] - 1))
                if str(b) in wr.get("mlp", {}): wr_tgt.append(abs(float((mlp.norm(dim=-1) / mid.norm(dim=-1)).mean()) / wr["mlp"][str(b)] - 1))
                wr_factors[b] = factors
            stream = blk(stream)
    if tgt:
        check("W' = s W + a * left right^T with the DEFINED vectors (P, c1 | P, r | -1/sqrt n, c2 | u, 1/sqrt n)", res_max < 1e-4, f"max relative residual {res_max:.1e}")
        check(f"declared scale unchanged by the rank-one components ({spec.get('gain_fold', 'exact')} convention: rms(W' diag gamma) = rms(W diag gamma) for exact, rms(W') = rms(W) for product)", eff_max < 1e-4, f"max relative deviation {eff_max:.1e}")
        check("functional targets met on the calibration images", max(tgt) < 5e-3, f"max absolute deviation {max(tgt):.1e} over {len(tgt)} targets")
    if "active_units" in joint.get("fc1_gate", {}):
        check("fc1 gate: fraction of active units met on the calibration images", gate_dev < 0.03, f"max deviation {gate_dev:.1e} (relative to max(target, 1e-3))")
    if "write_ratio" in joint:
        check(f"write ratio: each of {joint['write_ratio']['tensors']} is its second-moment-stage tensor times ONE scalar", wr_res < 1e-6, f"max relative residual {wr_res:.1e}")
        if "v" in joint["write_ratio"]["tensors"]:
            check("write ratio: v and proj carry the same factor (sqrt of the attention factor)", wr_split < 1e-5, f"max relative difference {wr_split:.1e}")
        check("write ratios met on the calibration images", max(wr_tgt) < 1e-3, f"max relative deviation {max(wr_tgt):.1e} over {len(wr_tgt)} targets")
        for b in sorted(wr_factors): print(f"      b{b}: " + "  ".join(f"{n} x{f:.3f}" for n, f in wr_factors[b].items()))
    check("every target reachable within the tensor's norm budget (or documented as exceeded)", all(p["reachable"] or p.get("exceeded_without_component") for r_ in report.values() for p in r_.values()), "")
    for note in notes: print("      " + note)
    wr_t = joint.get("write_ratio", {}).get("tensors", [])
    may_change = ("attn.qkv.weight", "mlp.fc1.weight", "mlp.fc2.weight") + (("attn.proj.weight",) if "proj" in wr_t else ())
    other = all(torch.equal(final[k], second[k]) for k in final if not any(k.endswith(x) for x in may_change))
    rows_same = all(torch.equal(final[f"blocks.{b}.attn.qkv.weight"][sl], second[f"blocks.{b}.attn.qkv.weight"][sl]) for b in BLOCKS
                    for sl in ([] if "v" in wr_t else [slice(2 * E, 3 * E)]) + ([] if "qk_entropy" in joint else [slice(0, 2 * E)]))
    check(f"nothing else changed by the calibration (LayerNorms, biases, embeddings, blocks {BLOCKS[-1] + 1}-11; v rows unless write-matched, q/k rows unless a sink is installed, proj unless write-matched)", other and rows_same, "")
print(f"[time] total: {lap()} on {DEV}")
print("VERDICT:", "PASS" if ok else "FAIL")
