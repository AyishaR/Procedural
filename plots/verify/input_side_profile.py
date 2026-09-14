"""Input-side functional profile AT INIT, blocks 0-8, on 16 ImageNet val images (CPU is fine):
attention logit std (sharpness), fc1 pre-activation rms, fraction of positive pre-activations (GELU gate),
GELU output rms, value rms; plus the decomposition of the pre-activation mean into fc1 bias / W.x / LN-bias terms.
Motivation (docs/i100_late_block_scaling.md 0d.11 "Generality test", 2026-09-14): both procedural prefixes (kdyck
ftb3i, ksd pksd3i) silence the middle MLPs through fc1 rows anti-aligned with the normalised stream (pre-activation
mean -2..-3 rms, GELU off), a structure no per-tensor statistic carries; ftbanap reaches the same state through small
fc1 scale, the ksd recipes (ftbanak, ftbanakw, ftbqmlnvok, ftbanakg) do not.
usage: .venv/bin/python plots/verify/input_side_profile.py [dump names...]   (default: the set in ARMS)"""
import sys, torch; sys.path.insert(0, "/home/schrodi/Procedural")
import main as M, utils
from datasets import build_dataset
torch.set_num_threads(32)
ARMS = sys.argv[1:] or ["r_s0", "ftb3i_s0", "ftbanap_s0", "ftbanai_s0", "pksd3i_s0", "ftbanak_s0", "ftbanakw_s0", "ftbqmlnvok_s0", "ftbanakg_s0", "ftbanakb_s0"]
D = "/home/schrodi/Procedural/results/init_dumps"
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args); g = torch.Generator().manual_seed(0)
x = torch.stack([ds[i][0] for i in torch.randperm(len(ds), generator=g)[:16].tolist()])
def measure(name):
    sd = torch.load(f"{D}/{name}.pth", map_location="cpu"); sd = sd.get("model", sd)
    m = utils.build_model(args); m.load_state_dict(sd, strict=False); m.eval()
    for b in m.blocks: b.attn.fused_attn = False
    st = {}
    def mk(i):
        def f(blk, inp, out):
            t = inp[0]; y = blk.norm1(t); a = blk.attn; B, N, C = y.shape
            qkv = a.qkv(y).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
            logits = (q @ k.transpose(-2, -1)) * a.scale; attn = logits.softmax(-1)
            r_out = t + blk.attn(y); y2 = blk.norm2(r_out); fc = blk.mlp.fc1; wx = y2 @ fc.weight.T; pre = wx + fc.bias; act = blk.mlp.act(pre)
            st[i] = dict(logit_std=logits.float().std().item(), v_rms=v.float().pow(2).mean().sqrt().item(), fc1_pre_rms=pre.float().pow(2).mean().sqrt().item(),
                         pre_mean=pre.mean().item(), wx_mean=wx.mean().item(), bias_mean=fc.bias.mean().item(), frac_pos=(pre > 0).float().mean().item(),
                         gelu_rms=act.float().pow(2).mean().sqrt().item(), ent=-(attn * (attn + 1e-12).log()).sum(-1).mean().item())
        return f
    for i, blk in enumerate(m.blocks): blk.register_forward_hook(mk(i))
    with torch.no_grad(): m(x)
    return st
R = {a: measure(a) for a in ARMS if __import__("os").path.exists(f"{D}/{a}.pth")}
for key, lab in (("logit_std", "attention logit std (sharpness)"), ("fc1_pre_rms", "fc1 pre-activation rms"), ("pre_mean", "fc1 pre-activation mean"),
                 ("frac_pos", "fraction of positive fc1 pre-activations (GELU gate)"), ("gelu_rms", "GELU output rms"), ("v_rms", "value rms"), ("ent", "attention entropy (nats)")):
    print(f"\n{lab}, blocks 0-8:")
    for a in R: print(f"  {a:14s} " + " ".join(f"{R[a][b][key]:7.3f}" for b in range(9)))
