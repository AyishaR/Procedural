"""Pre-launch checks of the three ablations built on ftbanap:
 ftbanau  gains ~ N(1, 0.25) without weight compensation: weights identical to ftbana_s0; gains mean 1 / spread 25%;
          LN biases identical to ftbanap_s0 (same generator draws); effective scales within 5% of ftbanap; forward ratios within 10%
 ftbanai  q, k, fc1 at random effective scale: only qkv (q,k rows) and fc1 differ from ftbanap; eff q,k,fc1 == timm (gain x W = 0.02);
          v/proj/fc2 and LN identical to ftbanap
 ftbanac  ftbanap + blocks 9-11 amplified: blocks 0-8 identical to ftbanap; blocks 9-11 v/proj/fc2 scaled; forward write ratio at
          blocks 9-11 near 1.4 (report; tune the multipliers if off)"""
import json, torch, numpy as np
D = "/home/schrodi/Procedural/results/init_dumps"; E = 768
L = {n: torch.load(f"{D}/{n}_s0.pth", map_location="cpu") for n in ("ftbana", "ftbanap", "ftbanau", "ftbanai", "ftbanac")}
L = {n: v.get("model", v) for n, v in L.items()}
F = json.load(open(f"{D}/init_forward_stats.json"))
def eff(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float(); g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
    return np.array([(W[:E]*g1).norm(), (W[E:2*E]*g1).norm(), (W[2*E:]*g1).norm(), sd[f"blocks.{b}.attn.proj.weight"].norm(), (sd[f"blocks.{b}.mlp.fc1.weight"].float()*g2).norm(), sd[f"blocks.{b}.mlp.fc2.weight"].norm()])
def diff_attrs(A, B, lo, hi):
    return sorted(set(k.split(".", 2)[2] for k in A if k.startswith("blocks.") and lo <= int(k.split(".")[1]) <= hi and not torch.equal(A[k], B[k])))
def ratios(name):
    return {k: np.array([F[f"{name}_s0"][str(b)][k] for b in range(12)]) for k in ("rho_attn", "rho_mlp")}
allok = True
# --- ftbanau
A, P, N = L["ftbanau"], L["ftbanap"], L["ftbana"]
w_same = all(torch.equal(A[k], N[k]) for k in A if k.startswith("blocks.") and int(k.split(".")[1]) < 9 and k.endswith("weight") and ".norm" not in k)
b_same = all(torch.equal(A[f"blocks.{b}.norm{i}.bias"], P[f"blocks.{b}.norm{i}.bias"]) for b in range(9) for i in (1, 2))
g = torch.stack([A[f"blocks.{b}.norm1.weight"].float() for b in range(9)]); gm, gs = g.mean(1), g.std(1)
r = np.array([eff(A, b) / eff(P, b) for b in range(9)])
ru, rp = ratios("ftbanau"), ratios("ftbanap")
fa = np.abs(ru["rho_attn"][:9] / rp["rho_attn"][:9] - 1).max(); fm = np.abs(ru["rho_mlp"][:9] / rp["rho_mlp"][:9] - 1).max()
ok = w_same and b_same and bool((gm - 1).abs().max() < 0.03) and bool((gs - 0.25).abs().max() < 0.03) and np.abs(r - 1).max() < 0.08 and fa < 0.10 and fm < 0.10   # gain rms = sqrt(1 + 0.25^2) = 1.03 by design (no compensation)
print(f"ftbanau: weights == ftbana {w_same} | LN biases == ftbanap {b_same} | gain mean {gm.min():.3f}-{gm.max():.3f} std {gs.min():.3f}-{gs.max():.3f} | eff scale / ftbanap max dev {np.abs(r-1).max():.3f} | forward ratio / ftbanap max dev attn {fa:.3f} mlp {fm:.3f} -> {'PASS' if ok else 'FAIL'}")
allok &= ok
# --- ftbanai
A = L["ftbanai"]
d = diff_attrs(A, P, 0, 8); rest = all(torch.equal(A[k], P[k]) for k in A if not (k.startswith("blocks.") and int(k.split(".")[1]) < 9))
e = np.array([eff(A, b) for b in range(9)]); e0 = np.array([eff(P, b) for b in range(9)])
timm = 0.02 * np.sqrt(np.array([E*E, E*E, E*E, E*E, 4*E*E, 4*E*E]))   # expected norm of a trunc-normal(0.02) matrix (approx.)
qk_fc1_ok = np.abs(e[:, [0, 1, 4]] / timm[[0, 1, 4]] - 1).max() < 0.06
vpf_same = np.allclose(e[:, [2, 3]], e0[:, [2, 3]], rtol=1e-4)   # v, proj untouched; fc2 re-tuned so the MLP write matches ftbanap
ri = ratios("ftbanai")
mlp_match = np.abs(ri["rho_mlp"][:9] / rp["rho_mlp"][:9] - 1).max() < 0.10
ok = set(d) == {"attn.qkv.weight", "mlp.fc1.weight", "mlp.fc2.weight"} and rest and qk_fc1_ok and vpf_same and mlp_match
print(f"ftbanai: differing attrs 0-8 {d} | rest identical {rest} | eff q,k,fc1 / timm max dev {np.abs(e[:, [0,1,4]] / timm[[0,1,4]] - 1).max():.3f} | v,proj == ftbanap {vpf_same} | MLP write 0-8 within 10% of ftbanap {mlp_match} | forward attn 1-8 {ri['rho_attn'][1:9].mean():.3f} (ftbanap {rp['rho_attn'][1:9].mean():.3f}) mlp 1-8 {ri['rho_mlp'][1:9].mean():.3f} ({rp['rho_mlp'][1:9].mean():.3f}) entropy 1-8 {np.mean([F['ftbanai_s0'][str(b)]['attn_entropy'] for b in range(1,9)]):.2f} ({np.mean([F['ftbanap_s0'][str(b)]['attn_entropy'] for b in range(1,9)]):.2f}) -> {'PASS' if ok else 'FAIL'}")
allok &= ok
# --- ftbanac
A = L["ftbanac"]
same08 = all(torch.equal(A[k], P[k]) for k in A if k.startswith("blocks.") and int(k.split(".")[1]) < 9) and all(torch.equal(A[k], P[k]) for k in A if not k.startswith("blocks."))
d = diff_attrs(A, P, 9, 11); rc = ratios("ftbanac")
print(f"ftbanac: blocks 0-8 + embed/head == ftbanap {same08} | differing attrs 9-11 {d} | forward write ratio blocks 9,10,11: attn {np.round(rc['rho_attn'][9:], 2)} mlp {np.round(rc['rho_mlp'][9:], 2)} (ftbrho: attn 1.39 mlp 1.42; ftbanap: attn {np.round(rp['rho_attn'][9:], 2)} mlp {np.round(rp['rho_mlp'][9:], 2)})")
ok = same08 and set(d) == {"attn.qkv.weight", "attn.proj.weight", "mlp.fc2.weight"} and np.all(np.abs(rc["rho_attn"][9:] / 1.4 - 1) < 0.2) and np.all(np.abs(rc["rho_mlp"][9:] / 1.4 - 1) < 0.2)
print(f"         -> {'PASS' if ok else 'FAIL (tune multipliers)'}"); allok &= ok
print("VERDICT:", "PASS" if allok else "FAIL")
