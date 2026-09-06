"""End-of-training (epoch 299) statistics for the main arms, all available seeds:
per-block LayerNorm gains, per-slice weight norms, LayerNorm-composed effective scales, and a
forward pass on 64 val images (rho, attention entropy).  Same measurement code as the init
scripts, so init and end state are directly comparable."""
import sys, os, json, math, torch, glob
sys.path.insert(0, "/home/schrodi/Procedural"); os.chdir("/home/schrodi/Procedural")
import utils, main as M
from datasets import build_dataset
OUT = os.environ.get("ES_OUT", "/home/schrodi/Procedural/results/init_dumps")
import json as _json
_R = _json.load(open("/home/schrodi/Procedural/results/init_dumps/arm_truth.json"))
def _rng_edit(sig):
    return bool(sig.get("weight_shuffle")) or (sig.get("quantile_1d_mode") not in (None, "skip")) or sig.get("quantile_source") == "parametric" or sig.get("quantile_1d_source") == "parametric"
RUNS = []   # (label, checkpoint path)
for _r in _R:
    if _r["max_epoch"] != 299 or not _r["output_dir"]: continue
    _lab = sorted(_r["jobnames"])[0]
    if _r["pre_fix"] and _rng_edit(_r["sig"]): _lab += "_PRE"
    RUNS.append((f"{_lab}_s{_r['seed']}", f"{_r['output_dir']}/checkpoint-299-model.pth"))
for _sid, _n in [("29384839", "r"), ("29377576", "p")]:
    for _s in range(3): RUNS.append((f"{_n}_s{_s}", f"results/imnet_base/results_IMNET_BASE_{_sid}/s{_s}/checkpoint-299-model.pth"))
RUNS += [tuple(x) for x in _json.load(open("/home/schrodi/Procedural/results/init_dumps/extra_runs.json"))]
if os.environ.get("MAX_RUNS"): RUNS = RUNS[:int(os.environ["MAX_RUNS"])]
E = 768; DH = 64
dev = "cuda" if torch.cuda.is_available() else "cpu"
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224"])
args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args)
g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[:64].tolist()
x = torch.stack([ds[i][0] for i in idx]).to(dev)
def measure(model, x):
    stats = {}
    def wrap(i, blk):
        def fwd(t):
            r_in = t; y = blk.norm1(t); a = blk.attn; B, Nt, C = y.shape
            qkv = a.qkv(y).reshape(B, Nt, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
            attn = ((q @ k.transpose(-2, -1)) * a.scale).softmax(dim=-1)
            ent = -(attn * (attn + 1e-12).log()).sum(-1)
            d_attn = blk.drop_path1(blk.ls1(a.proj((attn @ v).transpose(1, 2).reshape(B, Nt, C)))); r_out = r_in + d_attn
            d_mlp = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(r_out)))); out = r_out + d_mlp
            n = lambda z: torch.norm(z.float(), dim=-1)
            o = torch.nn.functional.normalize(out[:, 1:].float(), dim=-1)
            stats[i] = dict(rho_attn=float((n(d_attn) / (n(r_in) + 1e-8)).mean()), rho_mlp=float((n(d_mlp) / (n(r_out) + 1e-8)).mean()),
                            rin=float(n(r_in).mean()), rout=float(n(out).mean()), attn_entropy=float(ent.mean()), attn_maxp=float(attn.max(-1).values.mean()),
                            token_cos=float((o @ o.transpose(1, 2)).mean()))
            return out
        return fwd
    orig = [blk.forward for blk in model.blocks]
    for i, blk in enumerate(model.blocks): blk.forward = wrap(i, blk)
    model.eval()
    with torch.no_grad(): model(x)
    for blk, f in zip(model.blocks, orig): blk.forward = f
    return stats
def weight_stats(sd):
    A = {}
    for b in range(12):
        W = sd[f"blocks.{b}.attn.qkv.weight"].float(); q, k, v = W[:E], W[E:2*E], W[2*E:]
        proj = sd[f"blocks.{b}.attn.proj.weight"].float(); fc1 = sd[f"blocks.{b}.mlp.fc1.weight"].float(); fc2 = sd[f"blocks.{b}.mlp.fc2.weight"].float()
        g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
        qe, ke, ve, f1e = (q * g1).norm().item(), (k * g1).norm().item(), (v * g1).norm().item(), (fc1 * g2).norm().item()
        kurt = lambda t: ((((t.flatten() - t.mean()) / t.std()) ** 4).mean()).item()
        A[b] = dict(gamma1_mean=g1.mean().item(), gamma2_mean=g2.mean().item(), gamma1_rms=g1.pow(2).mean().sqrt().item(), gamma2_rms=g2.pow(2).mean().sqrt().item(),
                    q_norm=q.norm().item(), k_norm=k.norm().item(), v_norm=v.norm().item(), proj_norm=proj.norm().item(), fc1_norm=fc1.norm().item(), fc2_norm=fc2.norm().item(),
                    q_eff=qe, k_eff=ke, v_eff=ve, fc1_eff=f1e, logit_eff=qe * ke / (E * DH ** 0.5), attn_write_eff=ve * proj.norm().item() / E, mlp_write_eff=f1e * fc2.norm().item() / E,
                    q_kurt=kurt(q), v_kurt=kurt(v), fc1_kurt=kurt(fc1), fc2_kurt=kurt(fc2),
                    ln1_bias_std=sd[f"blocks.{b}.norm1.bias"].float().std().item(), fc1_bias_std=sd[f"blocks.{b}.mlp.fc1.bias"].float().std().item(),
                    qbias_std=sd[f"blocks.{b}.attn.qkv.bias"][:E].float().std().item())
    return A
res = {}
for label, f in RUNS:
    if not os.path.exists(f): print("missing", label, f, flush=True); continue
    sd = torch.load(f, map_location="cpu", weights_only=False)
    if "model" in sd: sd = sd["model"]
    sd = {k: v.float() for k, v in sd.items() if torch.is_tensor(v)}
    model = utils.build_model(args)
    for blk in model.blocks: blk.attn.fused_attn = False
    mis, unexp = model.load_state_dict(sd, strict=False)
    if mis or unexp: print(label, "missing", mis[:3], "unexpected", unexp[:3], flush=True)
    model.to(dev)
    res[label] = {"weights": weight_stats(sd), "forward": measure(model, x)}
    W = res[label]["weights"]; F = res[label]["forward"]
    print(f"{label:16s} g2 " + " ".join(f"{W[b]['gamma2_mean']:.2f}" for b in range(12)) + " | rho_mlp " + " ".join(f"{F[b]['rho_mlp']:.2f}" for b in range(12)) + " | ent " + " ".join(f"{F[b]['attn_entropy']:.1f}" for b in range(12)), flush=True)
    del model
json.dump(res, open(f"{OUT}/end_state_stats.json", "w"))
print("wrote end_state_stats.json", len(res))
