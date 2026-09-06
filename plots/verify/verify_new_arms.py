"""Pre-launch verification of the two new inits against their references.
 arm 1  ftbqmlnvog  vs ftbqmlnvo  : every slice norm / effective scale equal (tol 1%), kurtosis ~3, 1-D identical
 arm 2  ftbrhos     vs ftb4e3fix  : forward rho profile (val images) matches the permuted-proc profile; weights Gaussian, gamma=1"""
import sys, json, torch, numpy as np
sys.path.insert(0, "/home/schrodi/Procedural"); import utils, main as M
from datasets import build_dataset
torch.set_num_threads(16); E = 768; D = "/home/schrodi/Procedural/results/init_dumps"
G = sys.argv[1] if len(sys.argv) > 1 else "ftbqmlnvog_s0"; H = sys.argv[2] if len(sys.argv) > 2 else "ftbrhos_s0"
def sl(sd, b):
    W = sd[f"blocks.{b}.attn.qkv.weight"].float()
    return {"q": W[:E], "k": W[E:2*E], "v": W[2*E:], "proj": sd[f"blocks.{b}.attn.proj.weight"].float(), "fc1": sd[f"blocks.{b}.mlp.fc1.weight"].float(), "fc2": sd[f"blocks.{b}.mlp.fc2.weight"].float()}
def kurt(x): x = x.flatten(); z = (x - x.mean()) / x.std(); return float((z**4).mean())
def eff(sd, b):
    S = sl(sd, b); g1 = sd[f"blocks.{b}.norm1.weight"].float(); g2 = sd[f"blocks.{b}.norm2.weight"].float()
    return dict(q=S["q"].norm().item(), k=S["k"].norm().item(), v=S["v"].norm().item(), proj=S["proj"].norm().item(), fc1=S["fc1"].norm().item(), fc2=S["fc2"].norm().item(),
                logit=((S["q"]*g1).norm()*(S["k"]*g1).norm()/(E*8)).item(), write=((S["v"]*g1).norm()*S["proj"].norm()/E).item(), mlpw=((S["fc1"]*g2).norm()*S["fc2"].norm()/E).item(),
                g1=g1.mean().item(), g2=g2.mean().item(), kv=kurt(S["v"]), kq=kurt(S["q"]), kfc1=kurt(S["fc1"]), kfc2=kurt(S["fc2"]))
A = torch.load(f"{D}/{G}.pth", map_location="cpu"); R = torch.load(f"{D}/ftbqmlnvo_s0.pth", map_location="cpu")
print(f"=== ARM 1: {G} vs ftbqmlnvo (per block: max |rel diff| over q,k,v,proj,fc1,fc2 norms and logit/write/mlpw; kurtosis) ===")
ok1 = True
for b in range(12):
    a, r = eff(A, b), eff(R, b)
    rel = max(abs(a[k]-r[k])/max(abs(r[k]),1e-9) for k in ["q","k","v","proj","fc1","fc2","logit","write","mlpw"])
    same1d = all(torch.equal(A[f"blocks.{b}.{n}"], R[f"blocks.{b}.{n}"]) for n in ["norm1.weight","norm1.bias","norm2.weight","norm2.bias","attn.qkv.bias","attn.proj.bias","mlp.fc1.bias","mlp.fc2.bias"])
    flag = "OK" if (rel < 0.01 and same1d) else "CHECK"
    if b <= 8 and flag != "OK": ok1 = False
    print(f" block {b:2d}: max rel diff {rel:.4f}  1-D identical {same1d}  kurt v {r['kv']:.2f}->{a['kv']:.2f}  q {r['kq']:.2f}->{a['kq']:.2f}  fc1 {r['kfc1']:.2f}->{a['kfc1']:.2f}  fc2 {r['kfc2']:.2f}->{a['kfc2']:.2f}  {flag}")
# biases zero? (lnvo has zero linear biases)
zb = all(float(A[f"blocks.{b}.{n}"].abs().max()) == 0 for b in range(9) for n in ["attn.qkv.bias","attn.proj.bias","mlp.fc1.bias","mlp.fc2.bias"])
print(" linear biases zero in blocks 0-8:", zb); ok1 = ok1 and zb
print(" ARM 1 PASS" if ok1 else " ARM 1 FAIL")
# forward profiles
args = M.get_args_parser().parse_args(["--model","vit_base","--data_set","IMNET","--data_path","/data/datasets/ILSVRC2012","--input_size","224"]); args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args); g = torch.Generator().manual_seed(0); idx = torch.randperm(len(ds), generator=g)[:64].tolist(); x = torch.stack([ds[i][0] for i in idx])
def measure(sd):
    model = utils.build_model(args); model.load_state_dict(sd, strict=False); st = {}
    def wrap(i, blk):
        def fwd(t):
            r_in = t; y = blk.norm1(t); a = blk.attn; B, N, C = y.shape
            qkv = a.qkv(y).reshape(B, N, 3, a.num_heads, C//a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
            attn = ((q @ k.transpose(-2, -1)) * a.scale).softmax(-1); ent = -(attn*(attn+1e-12).log()).sum(-1)
            d_attn = a.proj((attn @ v).transpose(1, 2).reshape(B, N, C)); r_out = r_in + d_attn; d_mlp = blk.mlp(blk.norm2(r_out)); out = r_out + d_mlp
            n = lambda z: torch.norm(z.float(), dim=-1)
            st[i] = (float((n(d_attn)/(n(r_in)+1e-8)).mean()), float((n(d_mlp)/(n(r_out)+1e-8)).mean()), float(ent.mean()), float(n(out).mean()))
            return out
        return fwd
    for i, blk in enumerate(model.blocks): blk.forward = wrap(i, blk)
    model.eval()
    with torch.no_grad(): model(x)
    return st
F = json.load(open(f"{D}/init_forward_stats.json"))
print("\n=== forward profiles on 64 val images (rho_attn | rho_mlp | entropy), blocks 0..11 ===")
for lab, sd in [("ftbqmlnvog", A), ("ftbrhos", torch.load(f"{D}/{H}.pth", map_location="cpu"))]:
    st = measure(sd)
    print(f"{lab:12s} rho_a " + " ".join(f"{st[b][0]:.2f}" for b in range(12)) + f"\n{'':12s} rho_m " + " ".join(f"{st[b][1]:.2f}" for b in range(12)) + f"\n{'':12s} ent   " + " ".join(f"{st[b][2]:.2f}" for b in range(12)) + f"   |r| b8 {st[8][3]:.0f}")
    if lab == "ftbrhos": rhos = st
for ref in ["ftbqmlnvo_s0", "ftb4e3fix_s0", "r_s0"]:
    f = F[ref]; print(f"{ref:12s} rho_a " + " ".join(f"{f[str(b)]['rho_attn']:.2f}" for b in range(12)) + f"\n{'':12s} rho_m " + " ".join(f"{f[str(b)]['rho_mlp']:.2f}" for b in range(12)) + f"\n{'':12s} ent   " + " ".join(f"{f[str(b)]['attn_entropy']:.2f}" for b in range(12)) + f"   |r| b8 {f['8']['rout']:.0f}")
print("\n=== ARM 2: ftbrhos weights (blocks 0-8): gamma, kurtosis, per-slice std multiples of 0.02 ===")
B2 = torch.load(f"{D}/{H}.pth", map_location="cpu")
for b in range(9):
    e = eff(B2, b); S = sl(B2, b)
    print(f" block {b}: g1 {e['g1']:.2f} g2 {e['g2']:.2f} | std q {S['q'].std()/0.02:.2f} k {S['k'].std()/0.02:.2f} v {S['v'].std()/0.02:.2f} proj {S['proj'].std()/0.02:.2f} fc1 {S['fc1'].std()/0.02:.2f} fc2 {S['fc2'].std()/0.02:.2f} | kurt v {e['kv']:.2f} fc2 {e['kfc2']:.2f} | write x{e['write']/eff(torch.load(f'{D}/r_s0.pth',map_location='cpu'),b)['write']:.2f} mlpw x{e['mlpw']/eff(torch.load(f'{D}/r_s0.pth',map_location='cpu'),b)['mlpw']:.2f}")
tgt = F["ftb4e3fix_s0"]
dev = [(rhos[b][0]-tgt[str(b)]["rho_attn"], rhos[b][1]-tgt[str(b)]["rho_mlp"]) for b in range(9)]
print(" max |rho_attn - target| over blocks 0-8: %.3f ; max |rho_mlp - target|: %.3f" % (max(abs(d[0]) for d in dev), max(abs(d[1]) for d in dev)))
