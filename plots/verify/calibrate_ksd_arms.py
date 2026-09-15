"""Per-block calibration on data (16 val images, CPU) of the ksd structural arms, starting from the ftbanak_s0 dump:
  ftbanaks  : + attention sink per block 1-8 (q-bias norm B_b) hitting the ksd prefix's per-block attention entropy at init
  ftbanaksg : + sink and + fc1 bias gate (fraction of positive pre-activations = ksd prefix's), sink re-tuned with gates present
  ftbanakd  : q,k rows rescaled so the logit std is the kdyck recipe's (0.55, diffuse) + fc1 bias gate
Sequential: block b is calibrated with blocks < b already set (the stream feeding block b is the final one).
Writes vitbase_runs/profile_ftbanaks.json, profile_ftbanaksg.json, profile_ftbanakd.json."""
import sys, json, math, torch, numpy as np; sys.path.insert(0, "/home/schrodi/Procedural")
import main as M, utils
from datasets import build_dataset
from statistics import NormalDist
torch.set_num_threads(32)
ROOT = "/home/schrodi/Procedural"; D = f"{ROOT}/results/init_dumps"; E = 768; H = 12; DH = 64
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args); g = torch.Generator().manual_seed(0)
x = torch.stack([ds[i][0] for i in torch.randperm(len(ds), generator=g)[:16].tolist()])
ENT_T = {1: 0.37, 2: 0.17, 3: 1.33, 4: 0.80, 5: 1.75, 6: 2.10, 7: 2.71, 8: 2.97}          # pksd3i attention entropy at init
FRAC_T = {1: 0.056, 2: 0.160, 3: 0.115, 4: 0.098, 5: 0.111, 6: 0.131, 7: 0.126, 8: 0.155}  # pksd3i fraction of positive fc1 pre-activations
LOGIT_T = 0.55                                                                             # ftbanap's logit std (diffuse route)
base = json.load(open(f"{ROOT}/vitbase_runs/profile_ftbanak.json"))

class Stop(Exception): pass
def fresh():
    sd = torch.load(f"{D}/ftbanak_s0.pth", map_location="cpu"); sd = sd.get("model", sd)
    m = utils.build_model(args); m.load_state_dict(sd, strict=False); m.eval()
    for blk in m.blocks: blk.attn.fused_attn = False
    return m
def stats_at(model, b):
    """forward up to block b; return that block's attention entropy, logit std, max|logit|, sink share, row cosine, fc1 pre-act std/mean/frac_pos"""
    out = {}
    def hook(blk, inp, o):
        t = inp[0]; y = blk.norm1(t); a = blk.attn; B_, N, C = y.shape
        qkv = a.qkv(y).reshape(B_, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
        logits = (q @ k.transpose(-2, -1)) * a.scale; attn = logits.softmax(-1)
        r_out = t + blk.attn(y); pre = blk.mlp.fc1(blk.norm2(r_out))
        rows = attn[:, :, 1:, :]
        out.update(ent=-(attn * (attn + 1e-12).log()).sum(-1).mean().item(), logit_std=logits.std().item(), logit_max=logits.abs().max().item(),
                   sink=attn.mean(2).max(-1).values.mean().item(), rowsim=torch.nn.functional.cosine_similarity(rows[:, :, :, None, :], rows[:, :, None, :, :], dim=-1).mean().item(),
                   pre_std=(pre - pre.mean()).pow(2).mean().sqrt().item(), pre_mean=pre.mean().item(), frac_pos=(pre > 0).float().mean().item(),
                   attn_rho=(blk.attn(y).norm(dim=-1) / t.norm(dim=-1)).mean().item())
        raise Stop
    h = model.blocks[b].register_forward_hook(hook)
    try:
        with torch.no_grad(): model(x)
    except Stop: pass
    h.remove(); return out
def set_sink(model, b, B):
    dirs = utils.sink_directions(H, DH, 0, b); model.blocks[b].attn.qkv.bias.data[:E] = (dirs * B).reshape(-1)
def tune_sink(model, b, target):
    lo, hi = 0.0, 1.0
    set_sink(model, b, hi)
    while stats_at(model, b)["ent"] > target and hi < 4096: hi *= 2; set_sink(model, b, hi)
    for _ in range(12):
        mid = 0.5 * (lo + hi); set_sink(model, b, mid)
        if stats_at(model, b)["ent"] > target: lo = mid
        else: hi = mid
    set_sink(model, b, hi); return hi
def set_gate(model, b, frac):
    s = stats_at(model, b); val = -s["pre_mean"] + s["pre_std"] * NormalDist().inv_cdf(frac)
    model.blocks[b].mlp.fc1.bias.data.fill_(val); return val
report = {}
# ---- ftbanaks: sink only
m = fresh(); spec = json.loads(json.dumps(base)); spec["q_sink"] = {}
for b in range(1, 9):
    B = tune_sink(m, b, ENT_T[b]); spec["q_sink"][str(b)] = round(B, 3); s = stats_at(m, b)
    print(f"ftbanaks  b{b}: B={B:8.3f} ent {s['ent']:.2f} (target {ENT_T[b]}) sink {s['sink']:.2f} rowsim {s['rowsim']:.2f} logit std {s['logit_std']:.2f} max {s['logit_max']:.1f} attn rho {s['attn_rho']:.3f} frac_pos {s['frac_pos']:.2f}", flush=True)
json.dump(spec, open(f"{ROOT}/vitbase_runs/profile_ftbanaks.json", "w"), indent=2)
# ---- ftbanaksg: sink + gate, calibrated together block by block
m = fresh(); spec = json.loads(json.dumps(base)); spec["q_sink"] = {}; spec["fc1_bias"] = {}
for b in range(1, 9):
    B = tune_sink(m, b, ENT_T[b]); spec["q_sink"][str(b)] = round(B, 3)
    val = set_gate(m, b, FRAC_T[b]); spec["fc1_bias"][str(b)] = round(val, 3); s = stats_at(m, b)
    print(f"ftbanaksg b{b}: B={B:8.3f} ent {s['ent']:.2f} sink {s['sink']:.2f} | gate {val:+.3f} frac_pos {s['frac_pos']:.3f} (target {FRAC_T[b]}) | logit max {s['logit_max']:.1f}", flush=True)
json.dump(spec, open(f"{ROOT}/vitbase_runs/profile_ftbanaksg.json", "w"), indent=2)
# ---- ftbanakd: diffuse q,k (logit std -> 0.55) + gate
m = fresh(); spec = json.loads(json.dumps(base)); spec["fc1_bias"] = {}
def ramp(sl, b): return sl["b0"] if b == 0 else sl["start"] + (sl["end"] - sl["start"]) * (b - 1) / 7
qpb, kpb = {"0": round(base["q"]["b0"], 4)}, {"0": round(base["k"]["b0"], 4)}
for b in range(1, 9):
    s0 = stats_at(m, b); f = math.sqrt(LOGIT_T / s0["logit_std"])
    W = m.blocks[b].attn.qkv.weight.data; W[:E] *= f; W[E:2 * E] *= f
    qpb[str(b)] = round(ramp(base["q"], b) * f, 4); kpb[str(b)] = round(ramp(base["k"], b) * f, 4)
    val = set_gate(m, b, FRAC_T[b]); spec["fc1_bias"][str(b)] = round(val, 3); s = stats_at(m, b)
    print(f"ftbanakd  b{b}: q,k x{f:.3f} -> logit std {s['logit_std']:.3f} ent {s['ent']:.2f} | gate {val:+.3f} frac_pos {s['frac_pos']:.3f}", flush=True)
spec["q"] = {"per_block": qpb}; spec["k"] = {"per_block": kpb}
json.dump(spec, open(f"{ROOT}/vitbase_runs/profile_ftbanakd.json", "w"), indent=2)
print("specs written")
