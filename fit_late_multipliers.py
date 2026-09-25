"""Read the late-lever multipliers off an init dump (docs/late_lever_measured_profile_brief.md, step 2) and write the spec files.
For blocks 9-11: fit dump = m * timm per tensor for the v rows [1536, 2304) of attn.qkv.weight, attn.proj.weight and mlp.fc2.weight
against the timm seed-0 model (results/init_dumps/r0_s0.pth); assert the residual is ~1e-7, that v and proj carry the same factor
(main.py splits the attention factor as its square root over both), and that EVERY other tensor of the dump is bit-identical to timm.
Writes vitbase_runs/profile_<arm>pl.json ("extra"), lrscale_<arm>pl.json (nine entries, v rows through the row mask) and
lrscale_<arm>sl.json (the reciprocals).  usage: python plots/verify/fit_late_multipliers.py results/init_dumps/ftb3b_s0.pth ftb3b"""
import json, sys, torch
dump_path, arm = sys.argv[1], sys.argv[2]; E = 768; BLOCKS = (9, 10, 11); R = "/home/schrodi/Procedural/vitbase_runs"
d = torch.load(dump_path, map_location="cpu", weights_only=True); t = torch.load("/home/schrodi/Procedural/results/init_dumps/r0_s0.pth", map_location="cpu", weights_only=True)
def fit(W, W0):
    W, W0 = W.double().flatten(), W0.double().flatten(); m = float((W @ W0) / (W0 @ W0)); res = float(((W - m * W0).norm() / W.norm()))
    return m, res
extra, lr_pl, lr_sl = {}, {}, {}; ok = True
for b in BLOCKS:
    qkv, qkv0 = d[f"blocks.{b}.attn.qkv.weight"], t[f"blocks.{b}.attn.qkv.weight"]
    mv, rv = fit(qkv[2 * E:], qkv0[2 * E:]); mp, rp = fit(d[f"blocks.{b}.attn.proj.weight"], t[f"blocks.{b}.attn.proj.weight"]); mf, rf = fit(d[f"blocks.{b}.mlp.fc2.weight"], t[f"blocks.{b}.mlp.fc2.weight"])
    print(f"block {b}: v x{mv:.6f} (residual {rv:.1e})  proj x{mp:.6f} ({rp:.1e})  fc2 x{mf:.6f} ({rf:.1e})  | attention factor v*proj = {mv * mp:.4f}")
    ok &= max(rv, rp, rf) < 1e-6 and abs(mv / mp - 1) < 1e-5
    a = round((mv + mp) / 2, 6); f = round(mf, 6)
    extra[str(b)] = {"v": a, "proj": a, "fc2": f}
    lr_pl[f"blocks.{b}.attn.qkv.weight"] = {"rows": [[2 * E, 3 * E, a]]}; lr_pl[f"blocks.{b}.attn.proj.weight"] = a; lr_pl[f"blocks.{b}.mlp.fc2.weight"] = f
    lr_sl[f"blocks.{b}.attn.qkv.weight"] = {"rows": [[2 * E, 3 * E, round(1 / a, 8)]]}; lr_sl[f"blocks.{b}.attn.proj.weight"] = round(1 / a, 8); lr_sl[f"blocks.{b}.mlp.fc2.weight"] = round(1 / f, 8)
scaled = {f"blocks.{b}.{n}" for b in BLOCKS for n in ("attn.proj.weight", "mlp.fc2.weight")}
other_same = all(torch.equal(d[k], t[k]) for k in t if k not in scaled and not (k.endswith("attn.qkv.weight") and int(k.split(".")[1]) in BLOCKS))
qk_same = all(torch.equal(d[f"blocks.{b}.attn.qkv.weight"][:2 * E], t[f"blocks.{b}.attn.qkv.weight"][:2 * E]) for b in BLOCKS)
print(f"every other tensor (q/k rows of blocks 9-11, fc1, biases, LayerNorms, blocks 0-8, embeddings, head) bit-identical to timm seed 0: {other_same and qk_same}")
ok &= other_same and qk_same
if ok:
    json.dump({"extra": extra}, open(f"{R}/profile_{arm}pl.json", "w"), indent=1); json.dump(lr_pl, open(f"{R}/lrscale_{arm}pl.json", "w"), indent=1); json.dump(lr_sl, open(f"{R}/lrscale_{arm}sl.json", "w"), indent=1)
    print(f"wrote profile_{arm}pl.json, lrscale_{arm}pl.json, lrscale_{arm}sl.json:", json.dumps(extra))
print("FIT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
