"""Verify the ksd structural arms' dumps (made through main.py) against ftbanak_s0 and the targets:
  * exactly the intended tensors differ from ftbanak_s0 (qkv.bias q-part / fc1.bias / q,k rows of qkv.weight), rest identical,
    blocks 9-11 + embeddings + head identical to r_s0
  * per block 1-8: attention entropy, most-attended-key share, row cosine, max |logit| (fp16 headroom), fc1 gate fraction,
    attention write ratio -- against the ksd prefix (pksd3i_s0) and ftbanak_s0
usage: .venv/bin/python plots/verify/verify_ksd_struct.py ftbanaks_s0 ftbanaksg_s0 ftbanakbs_s0 ftbanakd_s0"""
import sys, json, torch, numpy as np; sys.path.insert(0, "/home/schrodi/Procedural")
import main as M, utils
from datasets import build_dataset
torch.set_num_threads(32)
D = "/home/schrodi/Procedural/results/init_dumps"; E = 768
L = lambda n: (lambda d: d.get("model", d))(torch.load(f"{D}/{n}.pth", map_location="cpu"))
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
ds, _ = build_dataset(is_train=False, args=args); g = torch.Generator().manual_seed(0)
x = torch.stack([ds[i][0] for i in torch.randperm(len(ds), generator=g)[:16].tolist()])
def measure(sd):
    m = utils.build_model(args); m.load_state_dict(sd, strict=False); m.eval()
    for blk in m.blocks: blk.attn.fused_attn = False
    st = {}
    def mk(i):
        def f(blk, inp, o):
            t = inp[0]; y = blk.norm1(t); a = blk.attn; B_, N, C = y.shape
            qkv = a.qkv(y).reshape(B_, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4); q, k, v = qkv.unbind(0)
            logits = (q @ k.transpose(-2, -1)) * a.scale; attn = logits.softmax(-1); rows = attn[:, :, 1:, :]
            r_out = t + blk.attn(y); pre = blk.mlp.fc1(blk.norm2(r_out))
            st[i] = dict(ent=-(attn * (attn + 1e-12).log()).sum(-1).mean().item(), sink=attn.mean(2).max(-1).values.mean().item(),
                         rowsim=torch.nn.functional.cosine_similarity(rows[:, :, :, None, :], rows[:, :, None, :, :], dim=-1).mean().item(),
                         lmax=logits.abs().max().item(), lstd=logits.std().item(), frac=(pre > 0).float().mean().item(),
                         rho=(blk.attn(y).norm(dim=-1) / t.norm(dim=-1)).mean().item(), gelu=blk.mlp.act(pre).pow(2).mean().sqrt().item(),
                         finite=bool(torch.isfinite(o).all()))
        return f
    for i, blk in enumerate(m.blocks): blk.register_forward_hook(mk(i))
    with torch.no_grad(): out = m(x)
    return st, bool(torch.isfinite(out).all())
base, r, P = L("ftbanak_s0"), L("r_s0"), L("pksd3i_s0")
SP, _ = measure(P); SB, _ = measure(base)
ALLOWED = {"ftbanaks_s0": {"attn.qkv.bias"}, "ftbanaksw_s0": {"attn.qkv.bias"}, "ftbanaksg_s0": {"attn.qkv.bias", "mlp.fc1.bias"}, "ftbanakbs_s0": {"mlp.fc1.bias"}, "ftbanakd_s0": {"attn.qkv.weight", "mlp.fc1.bias"}}
ok_all = True
for name in sys.argv[1:]:
    A = L(name); ok = True
    diff = sorted(set(k for k in A if not torch.equal(A[k], base[k])))
    attrs = set(k.split(".", 2)[2] for k in diff if k.startswith("blocks."))
    blocks = sorted(set(int(k.split(".")[1]) for k in diff if k.startswith("blocks.")))
    untouched = all(torch.equal(A[k], r[k]) for k in A if k.startswith(("blocks.9.", "blocks.10.", "blocks.11.", "patch_embed", "pos_embed", "cls_token", "head", "norm.")))
    ok &= attrs == ALLOWED[name] and blocks == list(range(1, 9)) and untouched and not any(not k.startswith("blocks.") for k in diff)
    if "attn.qkv.bias" in attrs:   # k and v parts of the bias must stay zero
        ok &= all(torch.equal(A[f"blocks.{b}.attn.qkv.bias"][E:], base[f"blocks.{b}.attn.qkv.bias"][E:]) for b in range(1, 9))
    if "attn.qkv.weight" in attrs: # only q,k rows scaled, v rows identical
        ok &= all(torch.equal(A[f"blocks.{b}.attn.qkv.weight"][2 * E:], base[f"blocks.{b}.attn.qkv.weight"][2 * E:]) for b in range(1, 9))
    S, finite = measure(A); ok &= finite and all(S[b]["finite"] for b in range(12))
    print(f"\n== {name}: differing attrs {sorted(attrs)} in blocks {blocks}; rest == ftbanak and blocks 9-11/embed/head == random: {untouched}; forward finite: {finite}")
    print(" blk | entropy  arm / ksd / ftbanak | sink share arm / ksd | row cos arm / ksd | max|logit| | logit std | gate frac arm / ksd | attn rho arm / ksd / ftbanak | GELU rms arm / ftbanak")
    for b in range(1, 9):
        s, p, q = S[b], SP[b], SB[b]
        print(f"  {b}  | {s['ent']:5.2f} / {p['ent']:5.2f} / {q['ent']:5.2f} | {s['sink']:5.2f} / {p['sink']:5.2f} | {s['rowsim']:5.2f} / {p['rowsim']:5.2f} | {s['lmax']:8.1f} | {s['lstd']:6.2f} | {s['frac']:5.3f} / {p['frac']:5.3f} | {s['rho']:5.3f} / {p['rho']:5.3f} / {q['rho']:5.3f} | {s['gelu']:5.3f} / {q['gelu']:5.3f}")
    ok_all &= ok; print(" structure check:", "PASS" if ok else "FAIL")
print("\nVERDICT:", "PASS" if ok_all else "FAIL")
