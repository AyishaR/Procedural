"""Pre-launch check of ftbanaf (analytic profile + blocks 9-11 fc2 x0.30) against ftbana_s0:
weights identical everywhere except blocks 9-11 fc2 (ratio 0.30); forward MLP write ratio of blocks 9-11
equal to the blocks-1-8 mean (contrast ~1 instead of 3.4); attention untouched."""
import json, torch, numpy as np
D = "/home/schrodi/Procedural/results/init_dumps"
A = torch.load(f"{D}/ftbanaf_s0.pth", map_location="cpu"); A = A.get("model", A)
B = torch.load(f"{D}/ftbana_s0.pth", map_location="cpu"); B = B.get("model", B)
diff = [k for k in A if not torch.equal(A[k], B[k])]
print("tensors that differ from ftbana_s0:", diff)
ok = sorted(diff) == sorted(f"blocks.{b}.mlp.fc2.weight" for b in (9, 10, 11))
for b in (9, 10, 11):
    r = (A[f"blocks.{b}.mlp.fc2.weight"].norm() / B[f"blocks.{b}.mlp.fc2.weight"].norm()).item(); print(f" block {b} fc2 norm ratio {r:.3f}"); ok = ok and abs(r - 0.30) < 0.005
F = json.load(open(f"{D}/init_forward_stats.json"))
if "ftbanaf_s0" in F:
    for key in ("rho_attn", "rho_mlp", "attn_entropy"):
        for lab in ("r_s0", "ftbana_s0", "ftbanaf_s0"):
            print(f" {key:12s} {lab:11s} " + " ".join(f"{F[lab][str(b)][key]:5.3f}" for b in range(12)))
    rm = np.array([F["ftbanaf_s0"][str(b)]["rho_mlp"] for b in range(12)]); ra = np.array([F["ftbanaf_s0"][str(b)]["rho_attn"] for b in range(12)])
    rm0 = np.array([F["ftbana_s0"][str(b)]["rho_mlp"] for b in range(12)]); ra0 = np.array([F["ftbana_s0"][str(b)]["rho_attn"] for b in range(12)])
    c = rm[9:].mean() / rm[1:9].mean(); print(f" MLP top/mid contrast: ftbana {rm0[9:].mean()/rm0[1:9].mean():.2f} -> ftbanaf {c:.2f}")
    same = np.allclose(rm[:9], rm0[:9], rtol=0.02) and np.allclose(ra, ra0, rtol=0.02); print(" blocks 0-8 MLP and all attention ratios unchanged (2%):", same)
    ok = ok and 0.7 < c < 1.4 and same
else:
    print(" ftbanaf_s0 not in forward stats yet"); ok = False
print("VERDICT:", "PASS" if ok else "FAIL")
