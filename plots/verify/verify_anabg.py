"""Pre-launch check of ftbanab (analytic profile + proc LN biases, permuted) and ftbanag (analytic profile
+ proc LN gain pattern, permuted, input-side multipliers divided by rms gamma) against ftbana_s0 and the
proc checkpoint. Weights: only the intended LN vectors differ (ftbanab) / LN gains plus the q,k,v,fc1
rescaling (ftbanag); LN vectors are permutations of proc's; effective scales rms(gamma)*||W|| equal
ftbana's within 1%. Forward: write ratios within ~15% of ftbana_s0 (the LN change is allowed to move them a
little), blocks 9-11 identical."""
import json, torch, numpy as np
D = "/home/schrodi/Procedural/results/init_dumps"; E = 768
B = torch.load(f"{D}/ftbana_s0.pth", map_location="cpu"); B = B.get("model", B)
P = torch.load("/home/schrodi/Procedural/results/pr_vitb_n/pr_6066174_final.pth", map_location="cpu", weights_only=False)["state"]
F = json.load(open(f"{D}/init_forward_stats.json"))
def eff(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float(); g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
    return np.array([(W[:E]*g1).norm(), (W[E:2*E]*g1).norm(), (W[2*E:]*g1).norm(), sd[f"blocks.{b}.attn.proj.weight"].norm(), (sd[f"blocks.{b}.mlp.fc1.weight"].float()*g2).norm(), sd[f"blocks.{b}.mlp.fc2.weight"].norm()])
allok = True
for arm, expect in (("ftbanab", {"norm1.bias", "norm2.bias"}), ("ftbanag", {"norm1.weight", "norm2.weight", "attn.qkv.weight", "mlp.fc1.weight"})):
    A = torch.load(f"{D}/{arm}_s0.pth", map_location="cpu"); A = A.get("model", A); ok = True
    diff = sorted(set(k.split(".", 2)[2] for k in A if not torch.equal(A[k], B[k]) and k.startswith("blocks.") and int(k.split(".")[1]) < 9))
    untouched = all(torch.equal(A[k], B[k]) for k in A if not (k.startswith("blocks.") and int(k.split(".")[1]) < 9))
    print(f"=== {arm}: attrs differing from ftbana in blocks 0-8: {diff}  (expected {sorted(expect)}) | everything else identical: {untouched}")
    ok = ok and set(diff) == expect and untouched
    for b in range(9):
        for i in (1, 2):
            for n in (["bias"] if arm == "ftbanab" else ["weight"]):
                a = A[f"blocks.{b}.norm{i}.{n}"].float(); p = P[f"blocks.{b}.norm{i}.{n}"].float()
                perm_ok = torch.allclose(a.sort().values, p.sort().values, atol=1e-6); ok = ok and perm_ok
                if not perm_ok: print(f"  block {b} norm{i}.{n} is NOT a permutation of proc's")
        r = eff(A, b) / eff(B, b); ok = ok and np.all(np.abs(r - 1) < 0.01)
        if np.any(np.abs(r - 1) >= 0.01): print(f"  block {b} effective scales vs ftbana (q k v proj fc1 fc2): {np.round(r, 3)}")
    print(f"  LN vectors are permutations of proc's, effective scales within 1% of ftbana: {ok}")
    if f"{arm}_s0" in F:
        for key in ("rho_attn", "rho_mlp", "attn_entropy"):
            a = np.array([F[f"{arm}_s0"][str(b)][key] for b in range(12)]); r0 = np.array([F["ftbana_s0"][str(b)][key] for b in range(12)])
            print(f"  {key:12s} {arm}/ftbana: " + " ".join(f"{x:5.2f}" for x in a / r0))
            if key != "attn_entropy": ok = ok and np.all(np.abs(a / r0 - 1) < 0.15)
    else:
        print(f"  {arm}_s0 not in forward stats"); ok = False
    print(f"  {arm}: {'PASS' if ok else 'FAIL'}"); allok = allok and ok
print("VERDICT:", "PASS" if allok else "FAIL")
