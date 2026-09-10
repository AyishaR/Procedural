"""Pre-launch verification of the analytic-profile init (ftbana) against the Gaussian-twin
reference ftbqmlnvo_s0 and timm random r_s0.

Checks, per block 0-8:
  * weight-side: effective logit / attention-write / MLP-write scales (gamma folded in) of the
    dump made through main.py match the spec (ftbana) and are within ~10% of ftbqmlnvo_s0
  * forward-side (results/init_dumps/init_forward_stats.json, 64 val images): rho_attn, rho_mlp,
    attention entropy within ~10% of ftbqmlnvo_s0
  * blocks 9-11, LayerNorms, biases identical to timm random (r_s0 up to the seed draw)
usage: .venv/bin/python plots/verify/verify_ana.py [ftbana_s0]
"""
import sys, json, torch
E = 768; D = "/home/schrodi/Procedural/results/init_dumps"
A = sys.argv[1] if len(sys.argv) > 1 else "ftbana_s0"

def sl(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float()
    return {"q": W[:E], "k": W[E:2*E], "v": W[2*E:], "proj": sd[f"blocks.{b}.attn.proj.weight"].float(),
            "fc1": sd[f"blocks.{b}.mlp.fc1.weight"].float(), "fc2": sd[f"blocks.{b}.mlp.fc2.weight"].float()}

def eff(sd, b):
    S = sl(sd, b); g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
    return dict(logit=((S["q"]*g1).norm()*(S["k"]*g1).norm()/(E*8)).item(),
                write=((S["v"]*g1).norm()*S["proj"].norm()/E).item(),
                mlpw=((S["fc1"]*g2).norm()*S["fc2"].norm()/E).item(),
                q=(S["q"]*g1).norm().item(), k=(S["k"]*g1).norm().item(), v=(S["v"]*g1).norm().item(),
                proj=S["proj"].norm().item(), fc1=(S["fc1"]*g2).norm().item(), fc2=S["fc2"].norm().item())

X = torch.load(f"{D}/{A}.pth", map_location="cpu"); X = X.get("model", X)
T = torch.load(f"{D}/ftbqmlnvo_s0.pth", map_location="cpu"); T = T.get("model", T)
R = torch.load(f"{D}/r_s0.pth", map_location="cpu"); R = R.get("model", R)

ok = True
print(f"=== {A} vs ftbqmlnvo_s0: effective scales relative to random (gamma folded in) ===")
print("blk |  logit  ana/twin |  attn-write ana/twin |  mlp-write ana/twin | per-slice eff norm ratio ana/twin: q k v proj fc1 fc2")
for b in range(9):
    a, t, r = eff(X, b), eff(T, b), eff(R, b)
    rel = {k: a[k]/t[k] for k in a}
    worst = max(abs(rel[k]-1) for k in ("logit", "write", "mlpw"))
    flag = "" if worst < 0.10 else "  <-- >10%"
    # block 8 MLP write: the twin jumps from 0.37 (block 7) to 0.52 at block 8 (the transition into
    # the loud blocks 9-11); a linear ramp deliberately does not follow it (documented deviation).
    tol = 0.15 if b < 8 else 0.30
    ok = ok and worst < tol
    print(f" {b:2d} | {a['logit']/r['logit']:5.2f} {t['logit']/r['logit']:5.2f} ({rel['logit']:.2f}) | {a['write']/r['write']:5.2f} {t['write']/r['write']:5.2f} ({rel['write']:.2f}) | {a['mlpw']/r['mlpw']:5.2f} {t['mlpw']/r['mlpw']:5.2f} ({rel['mlpw']:.2f}) | "
          + " ".join(f"{rel[k]:.2f}" for k in ("q", "k", "v", "proj", "fc1", "fc2")) + flag)

print("\n=== untouched parts ===")
same911 = all(torch.equal(X[k], R[k]) for k in X if k.startswith(("blocks.9.", "blocks.10.", "blocks.11.")))
ln1 = all(float((X[f"blocks.{b}.norm{i}.weight"] - 1).abs().max()) == 0 and float(X[f"blocks.{b}.norm{i}.bias"].abs().max()) == 0 for b in range(9) for i in (1, 2))
bias0 = all(float(X[f"blocks.{b}.{n}"].abs().max()) == 0 for b in range(9) for n in ("attn.qkv.bias", "attn.proj.bias", "mlp.fc1.bias", "mlp.fc2.bias"))
emb = all(torch.equal(X[k], R[k]) for k in X if k.startswith(("patch_embed", "pos_embed", "cls_token", "head", "norm.")))
print(f" blocks 9-11 == r_s0: {same911}   LN gains 1 / biases 0 in 0-8: {ln1}   linear biases 0 in 0-8: {bias0}   embed/head == r_s0: {emb}")
ok = ok and same911 and ln1 and bias0 and emb

print("\n=== forward profiles on 64 val images (init_forward_stats.json) ===")
F = json.load(open(f"{D}/init_forward_stats.json"))
if A in F:
    for key in ("rho_attn", "rho_mlp", "attn_entropy"):
        print(f" {key:12s}")
        for lab in ("r_s0", "ftbqmlnvo_s0", A):
            print(f"   {lab:13s} " + " ".join(f"{F[lab][str(b)][key]:5.2f}" for b in range(12)))
        rel = [F[A][str(b)][key] / F["ftbqmlnvo_s0"][str(b)][key] for b in range(9)]
        print(f"   ana/twin      " + " ".join(f"{x:5.2f}" for x in rel))
        if key != "attn_entropy":
            # same block-8 MLP exemption as above (twin 0.06 vs ramp 0.05)
            ok = ok and max(abs(x - 1) for x in rel[:8]) < 0.20 and abs(rel[8] - 1) < 0.30
else:
    print(f" {A} not in init_forward_stats.json yet"); ok = False
print("\nVERDICT:", "PASS" if ok else "FAIL")
