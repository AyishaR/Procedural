"""Forward-pass statistics AT INIT for every dumped init, on real ImageNet val images.
Per block: rho_attn, rho_mlp (as engine.attention_residual_analysis), residual-stream norm in/out,
attention entropy (nats, mean over heads/queries), attention max-prob, and the token cosine
similarity of the block output.  Device: cuda if available else cpu."""
import sys, os, glob, json, math, torch
sys.path.insert(0, "/home/schrodi/Procedural")
import utils, main as M
from datasets import build_dataset
D = "/home/schrodi/Procedural/results/init_dumps"; OUT = sys.argv[1] if len(sys.argv) > 1 else D
N_IMG = int(os.environ.get("N_IMG", "64"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012",
                                       "--input_size", "224", "--nb_classes", "1000"])
args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args)
g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[:N_IMG].tolist()
x = torch.stack([ds[i][0] for i in idx]).to(dev)
print(f"{N_IMG} val images, device {dev}", flush=True)
def measure(model, x):
    stats = {}
    def wrap(i, blk):
        def fwd(t):
            r_in = t
            y = blk.norm1(t)
            # attention internals (fused_attn False -> explicit softmax path reproduced here)
            a = blk.attn
            B, Nt, C = y.shape
            qkv = a.qkv(y).reshape(B, Nt, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * a.scale
            attn = attn.softmax(dim=-1)
            ent = -(attn * (attn + 1e-12).log()).sum(-1)           # B,H,N
            d_attn = blk.drop_path1(blk.ls1(a.proj((attn @ v).transpose(1, 2).reshape(B, Nt, C))))
            r_out = r_in + d_attn
            d_mlp = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(r_out))))
            out = r_out + d_mlp
            n = lambda z: torch.norm(z.float(), dim=-1)
            o = torch.nn.functional.normalize(out[:, 1:].float(), dim=-1)
            cos = (o @ o.transpose(1, 2)).mean().item()
            stats[i] = dict(rho_attn=float((n(d_attn) / (n(r_in) + 1e-8)).mean()), rho_mlp=float((n(d_mlp) / (n(r_out) + 1e-8)).mean()),
                            rin=float(n(r_in).mean()), rout=float(n(out).mean()), d_attn=float(n(d_attn).mean()), d_mlp=float(n(d_mlp).mean()),
                            attn_entropy=float(ent.mean()), attn_entropy_max=math.log(Nt), attn_maxp=float(attn.max(-1).values.mean()),
                            token_cos=cos)
            return out
        return fwd
    orig = [blk.forward for blk in model.blocks]
    for i, blk in enumerate(model.blocks): blk.forward = wrap(i, blk)
    model.eval()
    with torch.no_grad(): model(x)
    for blk, f in zip(model.blocks, orig): blk.forward = f
    return stats
res = {}
for f in sorted(glob.glob(f"{D}/*.pth")):
    arm = os.path.basename(f)[:-4]
    model = utils.build_model(args)
    for blk in model.blocks: blk.attn.fused_attn = False
    sd = torch.load(f, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected: print(arm, "missing", missing[:3], "unexpected", unexpected[:3])
    model.to(dev)
    res[arm] = measure(model, x)
    ra = " ".join(f"{res[arm][i]['rho_attn']:.2f}" for i in range(12)); rm = " ".join(f"{res[arm][i]['rho_mlp']:.2f}" for i in range(12))
    en = " ".join(f"{res[arm][i]['attn_entropy']:.2f}" for i in range(12))
    print(f"{arm:18s} rho_attn {ra}\n{'':18s} rho_mlp  {rm}\n{'':18s} entropy  {en}  rout11 {res[arm][11]['rout']:.1f}", flush=True)
    del model
json.dump(res, open(f"{OUT}/init_forward_stats.json", "w"))
print("wrote", f"{OUT}/init_forward_stats.json")
