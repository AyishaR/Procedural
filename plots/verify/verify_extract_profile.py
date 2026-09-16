"""Verify extract_profile.py on both procedural checkpoints.
For each checkpoint: extract the ramp spec (kdyck with the two documented corrections) and the exact spec, apply them to a
fresh timm ViT-B (seed 0) with utils.apply_analytic_profile, and check
  (1) weights: effective scale rms(gamma)*rms(W)/0.02 of the initialised model equals the spec (ramp) or the checkpoint (exact)
  (2) LayerNorm: per-block mean/std of the sampled gains and biases match the checkpoint's within sampling error
  (3) function: forward write ratios / logit std / GELU rms on 16 val images vs the arm dumps made through main.py
      (kdyck ramp+corrections vs ftbanap_s0, ksd ramp vs ftbanak_s0) and, for the exact form, vs the ksd exact dump ftbanakx_s0
usage: .venv/bin/python plots/verify/verify_extract_profile.py"""
import sys, json, subprocess, torch, numpy as np; sys.path.insert(0, "/home/schrodi/Procedural")
import main as M, utils
from datasets import build_dataset
torch.set_num_threads(32)
ROOT = "/home/schrodi/Procedural"; D = f"{ROOT}/results/init_dumps"; E = 768; TMP = "/tmp/claude-16853/-home-schrodi-Procedural/57ac4c02-f3c3-4598-85f2-c462137a5cec/scratchpad"
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args); g = torch.Generator().manual_seed(0)
x = torch.stack([ds[i][0] for i in torch.randperm(len(ds), generator=g)[:16].tolist()])
def measure(m):
    for blk in m.blocks: blk.attn.fused_attn = False
    st = {}
    def mk(i):
        def f(blk, inp, o):
            t = inp[0]; y = blk.norm1(t); a = blk.attn; B_, N, C = y.shape
            qkv = a.qkv(y).reshape(B_, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
            logits = (q @ k.transpose(-2, -1)) * a.scale; r_out = t + blk.attn(y); pre = blk.mlp.fc1(blk.norm2(r_out))
            st[i] = dict(rho_a=(blk.attn(y).norm(dim=-1) / t.norm(dim=-1)).mean().item(), rho_m=(blk.mlp(blk.norm2(r_out)).norm(dim=-1) / r_out.norm(dim=-1)).mean().item(),
                         lstd=logits.std().item(), gelu=blk.mlp.act(pre).pow(2).mean().sqrt().item())
        return f
    hs = [blk.register_forward_hook(mk(i)) for i, blk in enumerate(m.blocks)]
    with torch.no_grad(): m(x)
    for h in hs: h.remove()
    return st
def fresh(spec):
    torch.manual_seed(0); m = utils.build_model(args); m.eval(); utils.apply_analytic_profile(m, spec, list(range(9)), seed=0); return m
def eff(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float(); g1 = sd[f"blocks.{b}.norm1.weight"].float().pow(2).mean().sqrt(); g2 = sd[f"blocks.{b}.norm2.weight"].float().pow(2).mean().sqrt()
    r = lambda t: t.float().pow(2).mean().sqrt() / 0.02
    return np.array([g1 * r(W[:E]), g1 * r(W[E:2 * E]), g1 * r(W[2 * E:]), r(sd[f"blocks.{b}.attn.proj.weight"]), g2 * r(sd[f"blocks.{b}.mlp.fc1.weight"]), r(sd[f"blocks.{b}.mlp.fc2.weight"])])
def ramp(sl, b): return sl["b0"] if b == 0 else sl["start"] + (sl["end"] - sl["start"]) * (b - 1) / 7
def load_dump(n): sd = torch.load(f"{D}/{n}.pth", map_location="cpu"); sd = sd.get("model", sd); m = utils.build_model(args); m.load_state_dict(sd, strict=False); m.eval(); return m
CASES = [("kdyck", f"{ROOT}/results/pr_vitb_n/pr_6066174_final.pth", ["--qk_flat", "1.32", "--fc2_end", "0.95"], "ftbanap_s0", "ftbanap"),
         ("ksd", f"{ROOT}/results/pr_vitb_ksd/pr_6463456_final.pth", [], "ftbanak_s0", "ftbanak")]
ok_all = True
for name, ck, flags, dump, arm in CASES:
    P = torch.load(ck, map_location="cpu", weights_only=False); P = P.get("state", P.get("model", P))
    for form in ("ramp", "exact"):
        out = f"{TMP}/{name}_{form}.json"
        subprocess.run([f"{ROOT}/.venv/bin/python", f"{ROOT}/extract_profile.py", ck, out] + (flags if form == "ramp" else ["--exact"]), check=True, capture_output=True)
        spec = json.load(open(out)); m = fresh(spec); sd = m.state_dict(); ok = True
        # (1) effective scales
        dev = []
        for b in range(9):
            got = eff(sd, b)
            want = np.array([ramp(spec[s], b) for s in ("q", "k", "v", "proj", "fc1", "fc2")]) if form == "ramp" else eff(P, b)
            dev.append(np.abs(got / want - 1).max())
        ok &= max(dev) < 0.02
        # (2) LN statistics
        lnd = []
        for b in range(9):
            for i in (1, 2):
                gm, gs = sd[f"blocks.{b}.norm{i}.weight"].float().mean().item(), sd[f"blocks.{b}.norm{i}.weight"].float().std().item()
                pm, ps = P[f"blocks.{b}.norm{i}.weight"].float().mean().item(), P[f"blocks.{b}.norm{i}.weight"].float().std().item()
                bs, pbs = sd[f"blocks.{b}.norm{i}.bias"].float().std().item(), P[f"blocks.{b}.norm{i}.bias"].float().std().item()
                lnd.append(max(abs(gm - pm) / pm, abs(gs - ps) / ps, abs(bs - pbs) / max(pbs, 1e-3)))
        ok &= max(lnd) < 0.08          # 768 samples: relative error of mean/std ~ 1/sqrt(768) ~ 4%
        untouched = all(torch.equal(sd[k], load_dump("r_s0").state_dict()[k]) for k in sd if k.startswith(("blocks.9.", "blocks.10.", "blocks.11.")))
        ok &= untouched
        line = f"{name} {form:5s}: eff-scale dev vs {'spec' if form == 'ramp' else 'checkpoint'} max {max(dev):.4f} | LN mean/std rel. dev max {max(lnd):.3f} | blocks 9-11 untouched {untouched}"
        # (3) function vs the arm dumps
        ref = dump if form == "ramp" else ("ftbanakx_s0" if name == "ksd" else None)
        if ref:
            S, R = measure(m), measure(load_dump(ref)); rat = {k: np.array([S[b][k] / R[b][k] for b in range(9)]) for k in ("rho_a", "rho_m", "lstd", "gelu")}
            worst = max(np.abs(v - 1).max() for v in rat.values()); ok &= worst < 0.15
            line += f" | forward vs {ref}: attn write {rat['rho_a'].min():.2f}-{rat['rho_a'].max():.2f}, MLP write {rat['rho_m'].min():.2f}-{rat['rho_m'].max():.2f}, logit std {rat['lstd'].min():.2f}-{rat['lstd'].max():.2f}, GELU {rat['gelu'].min():.2f}-{rat['gelu'].max():.2f} (ratios over blocks 0-8)"
        print(line, "->", "PASS" if ok else "FAIL"); ok_all &= ok
print("VERDICT:", "PASS" if ok_all else "FAIL")
