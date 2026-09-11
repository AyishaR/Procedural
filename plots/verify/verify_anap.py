"""Pre-launch check of ftbanap (analytic profile + Gaussian-sampled LN gains and biases with proc's per-block
mean/std) against ftbana_s0 / ftbanag_s0 and the proc checkpoint: only LN vectors + qkv/fc1 differ from ftbana;
LN vectors have proc's mean/std (3%) but are not permutations; effective scales within 1% of ftbana; forward
write ratios within 15% of ftbana_s0."""
import json, torch, numpy as np
D = "/home/schrodi/Procedural/results/init_dumps"; E = 768
A = torch.load(f"{D}/ftbanap_s0.pth", map_location="cpu"); A = A.get("model", A)
B = torch.load(f"{D}/ftbana_s0.pth", map_location="cpu"); B = B.get("model", B)
P = torch.load("/home/schrodi/Procedural/results/pr_vitb_n/pr_6066174_final.pth", map_location="cpu", weights_only=False)["state"]
F = json.load(open(f"{D}/init_forward_stats.json"))
def eff(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float(); g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
    return np.array([(W[:E]*g1).norm(), (W[E:2*E]*g1).norm(), (W[2*E:]*g1).norm(), sd[f"blocks.{b}.attn.proj.weight"].norm(), (sd[f"blocks.{b}.mlp.fc1.weight"].float()*g2).norm(), sd[f"blocks.{b}.mlp.fc2.weight"].norm()])
ok = True
diff = sorted(set(k.split(".", 2)[2] for k in A if not torch.equal(A[k], B[k]) and k.startswith("blocks.") and int(k.split(".")[1]) < 9))
untouched = all(torch.equal(A[k], B[k]) for k in A if not (k.startswith("blocks.") and int(k.split(".")[1]) < 9))
print("attrs differing from ftbana in 0-8:", diff, "| rest identical:", untouched)
ok = ok and set(diff) == {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.weight", "mlp.fc1.weight"} and untouched
print("blk | gain mean/std model vs proc (norm1) | bias rms model vs proc (norm1) | permutation? | eff scale ratio to ftbana (max |dev|)")
for b in range(9):
    g, pg = A[f"blocks.{b}.norm1.weight"].float(), P[f"blocks.{b}.norm1.weight"].float(); bb, pb = A[f"blocks.{b}.norm1.bias"].float(), P[f"blocks.{b}.norm1.bias"].float()
    perm = torch.allclose(g.sort().values, pg.sort().values, atol=1e-6)
    r = eff(A, b) / eff(B, b)
    stat_ok = abs(g.mean()-pg.mean()) < 0.03*abs(pg.mean()) + 0.01 and abs(g.std()-pg.std()) < 0.1*pg.std() + 0.005 and abs(bb.pow(2).mean().sqrt()-pb.pow(2).mean().sqrt()) < 0.1*pb.pow(2).mean().sqrt() + 0.005
    ok = ok and stat_ok and not perm and np.all(np.abs(r-1) < 0.01)
    print(f" {b} | {g.mean():.3f}/{g.std():.3f} vs {pg.mean():.3f}/{pg.std():.3f} | {bb.pow(2).mean().sqrt():.3f} vs {pb.pow(2).mean().sqrt():.3f} | {perm} | {np.abs(r-1).max():.4f}")
if "ftbanap_s0" in F:
    for key in ("rho_attn", "rho_mlp", "attn_entropy"):
        a = np.array([F["ftbanap_s0"][str(b)][key] for b in range(12)]); r0 = np.array([F["ftbana_s0"][str(b)][key] for b in range(12)])
        print(f" {key:12s} ftbanap/ftbana: " + " ".join(f"{x:5.2f}" for x in a/r0))
        if key != "attn_entropy": ok = ok and np.all(np.abs(a/r0-1) < 0.15)
else:
    print(" ftbanap_s0 not in forward stats"); ok = False
print("VERDICT:", "PASS" if ok else "FAIL")
