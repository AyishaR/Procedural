"""Per-slice weight-distribution statistics of every dumped init (results/init_dumps/*.pth).
Everything is measured on the tensors main.py actually trains (dump_init.py runs main.py's own
init path).  For each block and slice (q, k, v, proj, fc1, fc2): scale, shape, stable rank,
and the LayerNorm-composed EFFECTIVE scales the forward pass sees (W diag(gamma)).  Also the
W1 distance of each slice's marginal to proc's same slice, raw (scale+shape) and standardised
(shape only).  Output: one JSON, arm -> block -> stats."""
import sys, os, json, glob, torch, numpy as np
torch.set_num_threads(16)
D = "/home/schrodi/Procedural/results/init_dumps"; OUT = sys.argv[1] if len(sys.argv) > 1 else D
E = 768; DH = 64
Q = torch.linspace(0.0005, 0.9995, 2001)
def qs(x): return torch.quantile(x.flatten().float(), Q)
def slices(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float()
    return {"q": W[:E], "k": W[E:2*E], "v": W[2*E:], "proj": sd[f"blocks.{b}.attn.proj.weight"].float(),
            "fc1": sd[f"blocks.{b}.mlp.fc1.weight"].float(), "fc2": sd[f"blocks.{b}.mlp.fc2.weight"].float()}
def shape_stats(x):
    x = x.flatten(); m = x.mean(); s = x.std(); z = (x - m) / s
    return dict(mean=m.item(), std=s.item(), norm=x.norm().item(), skew=(z**3).mean().item(), kurt=(z**4).mean().item(),
                tail3=(z.abs() > 3).float().mean().item(), maxabs_std=z.abs().max().item())
def stable_rank(W):
    s = torch.linalg.svdvals(W); return float((s**2).sum() / (s**2).max()), float(s.max())
proc = torch.load("results/pr_vitb_n/pr_6066174_final.pth", map_location="cpu", weights_only=False)["state"]
PQ = {}   # proc quantiles per block/slice, raw and standardised
for b in range(12):
    PQ[b] = {}
    for n, W in slices(proc, b).items():
        q = qs(W); PQ[b][n] = (q, (q - W.mean()) / W.std())
res = {}
files = sorted(glob.glob(f"{D}/*.pth"))
for f in files:
    arm = os.path.basename(f)[:-4]
    sd = torch.load(f, map_location="cpu")
    A = {}
    for b in range(12):
        S = slices(sd, b)
        g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
        b1 = sd[f"blocks.{b}.norm1.bias"].float(); b2 = sd[f"blocks.{b}.norm2.bias"].float()
        qb = sd[f"blocks.{b}.attn.qkv.bias"].float()
        st = {"gamma1_mean": g1.mean().item(), "gamma1_sd": g1.std().item(), "gamma1_rms": g1.pow(2).mean().sqrt().item(),
              "gamma2_mean": g2.mean().item(), "gamma2_sd": g2.std().item(), "gamma2_rms": g2.pow(2).mean().sqrt().item(),
              "ln1_bias_std": b1.std().item(), "ln2_bias_std": b2.std().item(),
              "qbias_std": qb[:E].std().item(), "kbias_std": qb[E:2*E].std().item(), "vbias_std": qb[2*E:].std().item(),
              "projbias_std": sd[f"blocks.{b}.attn.proj.bias"].float().std().item(),
              "fc1bias_std": sd[f"blocks.{b}.mlp.fc1.bias"].float().std().item(),
              "fc2bias_std": sd[f"blocks.{b}.mlp.fc2.bias"].float().std().item()}
        for n, W in S.items():
            d = shape_stats(W)
            if b <= 8:
                d["stable_rank"], d["smax"] = stable_rank(W)
            q = qs(W); qz = (q - W.mean()) / W.std()
            d["w1_proc_raw"] = (q - PQ[b][n][0]).abs().mean().item()
            d["w1_proc_shape"] = (qz - PQ[b][n][1]).abs().mean().item()
            for kk, vv in d.items(): st[f"{n}.{kk}"] = vv
        # LayerNorm-composed effective scales (what attention / the MLP actually see)
        qe = (S["q"] * g1).norm().item(); ke = (S["k"] * g1).norm().item(); ve = (S["v"] * g1).norm().item()
        f1e = (S["fc1"] * g2).norm().item()
        st.update({"q_eff": qe, "k_eff": ke, "v_eff": ve, "fc1_eff": f1e, "proj_norm": S["proj"].norm().item(), "fc2_norm": S["fc2"].norm().item(),
                   "logit_eff": qe * ke / (E * DH**0.5), "attn_write_eff": ve * S["proj"].norm().item() / E,
                   "mlp_write_eff": f1e * S["fc2"].norm().item() / E})
        A[b] = st
    res[arm] = A
    print(f"{arm:20s} b0 logit {A[0]['logit_eff']:.4f} write {A[0]['attn_write_eff']:.3f} mlp {A[0]['mlp_write_eff']:.3f} | "
          f"b4 logit {A[4]['logit_eff']:.4f} write {A[4]['attn_write_eff']:.3f} mlp {A[4]['mlp_write_eff']:.3f} v_std {A[4]['v.std']:.4f} kurt fc1 {A[4]['fc1.kurt']:.2f}", flush=True)
json.dump(res, open(f"{OUT}/init_dist_stats.json", "w"))
print("wrote", f"{OUT}/init_dist_stats.json", len(res), "arms")
