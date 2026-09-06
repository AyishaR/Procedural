"""Shape of the procedural checkpoint's early-block weight marginals, per slice and block.
No training, no dumps: reads the checkpoint and a timm random init directly.
Questions: how non-Gaussian is each slice (kurtosis, skew, tail mass)?  How much of the
Wasserstein-1 distance between proc's marginal and a Gaussian of the same norm is removed by a
Student-t fitted to (norm, kurtosis) -- i.e. would a 4-moment parametric donor reproduce proc's
value distribution, or is there structure beyond four moments (skew, bimodality, spikes)?"""
import sys, json, math, torch, numpy as np
sys.path.insert(0, "/home/schrodi/Procedural")
torch.manual_seed(0)
ck = torch.load("results/pr_vitb_n/pr_6066174_final.pth", map_location="cpu", weights_only=False)
sd = ck["state"]
import timm
rnd = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000).state_dict()
E = 768
def slices(b, sd):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float()
    return {"q": W[:E], "k": W[E:2*E], "v": W[2*E:], "qk": W[:2*E],
            "proj": sd[f"blocks.{b}.attn.proj.weight"].float(),
            "fc1": sd[f"blocks.{b}.mlp.fc1.weight"].float(),
            "fc2": sd[f"blocks.{b}.mlp.fc2.weight"].float()}
Q = torch.linspace(0.0005, 0.9995, 2001)
def qs(x): return torch.quantile(x.flatten(), Q)
def w1(a, b): return (qs(a) - qs(b)).abs().mean().item()          # W1 via quantile functions
def moments(x):
    x = x.flatten(); m = x.mean(); s = x.std(); z = (x - m) / s
    return dict(mean=m.item(), std=s.item(), norm=x.norm().item(), skew=(z**3).mean().item(),
                kurt=(z**4).mean().item(), tail3=(z.abs() > 3).float().mean().item(),
                tail5=(z.abs() > 5).float().mean().item(), maxabs_over_std=(z.abs().max()).item(),
                frac_near0=(z.abs() < 0.1).float().mean().item())   # Gaussian: 0.0797
def student_t_like(x):
    """Student-t sample fitted the way main.py's --quantile_source parametric does it."""
    z = (x - x.mean()) / x.std(); kurt = (z**4).mean().item()
    df = 4.0 + 6.0 / max(kurt - 3.0, 1e-3); df = float(min(max(df, 4.5), 100.0))
    samp = torch.distributions.StudentT(df).sample((x.numel(),))
    return samp / samp.norm() * x.norm(), df
out = {}
print(f"{'blk':>3} {'slice':>5} {'std_p':>7} {'std_r':>7} {'ratio':>6} {'kurt':>5} {'skew':>6} {'tail3':>7} {'near0':>6} | W1(proc,gauss) W1(proc,t)  t/gauss  df")
for b in range(12):
    P, R = slices(b, sd), slices(b, rnd)
    out[b] = {}
    for name in ["q", "k", "v", "proj", "fc1", "fc2"]:
        x = P[name].flatten(); r = R[name].flatten()
        m = moments(x); mr = moments(r)
        g = torch.randn(x.numel()) * x.std() + x.mean()          # Gaussian with proc's mean/std
        t, df = student_t_like(x)
        w_g, w_t = w1(x, g), w1(x, t)
        out[b][name] = dict(m, std_random=mr["std"], std_ratio=m["std"]/mr["std"], w1_gauss=w_g, w1_t=w_t, t_df=df,
                            w1_t_over_gauss=w_t / w_g)
        if b <= 8 or b == 11:
            print(f"{b:>3} {name:>5} {m['std']:7.4f} {mr['std']:7.4f} {m['std']/mr['std']:6.3f} {m['kurt']:5.2f} {m['skew']:6.2f} {m['tail3']:7.4f} {m['frac_near0']:6.3f} | {w_g:12.6f} {w_t:10.6f} {w_t/w_g:7.3f} {df:5.1f}")
    # 1-D
    for name in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.bias", "attn.proj.bias", "mlp.fc1.bias", "mlp.fc2.bias"]:
        x = sd[f"blocks.{b}.{name}"].float()
        mm = moments(x); out[b][name] = mm
    if b <= 8:
        g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
        qb = sd[f"blocks.{b}.attn.qkv.bias"].float()
        print(f"     1-D: gamma1 mean {g1.mean():.3f} sd {g1.std():.3f} min {g1.min():.3f} max {g1.max():.3f} | gamma2 mean {g2.mean():.3f} sd {g2.std():.3f} | "
              f"qkv.bias std q {qb[:E].std():.4f} k {qb[E:2*E].std():.4f} v {qb[2*E:].std():.4f} | proj.b std {sd[f'blocks.{b}.attn.proj.bias'].float().std():.4f} fc1.b {sd[f'blocks.{b}.mlp.fc1.bias'].float().std():.4f} fc2.b {sd[f'blocks.{b}.mlp.fc2.bias'].float().std():.4f}")
json.dump(out, open(f"{sys.argv[1]}/proc_shape.json", "w"), indent=1)
print("random-init reference: std q/k/v/proj/fc1/fc2 =", [round(moments(slices(0, rnd)[n])["std"], 4) for n in ["q","k","v","proj","fc1","fc2"]], "kurt", round(moments(slices(0, rnd)["fc1"])["kurt"], 3))
