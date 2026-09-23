"""fp16 head-room probe: where does the forward pass of a per-epoch checkpoint approach the fp16 limit (65504)?

For every checkpoint: 256 training images (training augmentation, no mixup; seed 0) go through the model under autocast bfloat16,
where nothing overflows, and per block the largest |value| is recorded for: the residual stream entering the block, the attention
logits (q * scale) @ k^T as training computes them (non-fused attention, main.py sets fused_attn = False), the attention branch
output, the fc1 pre-activation, the GELU output and the fc2 output. The same images then go through autocast float16 and the first
module with a non-finite output is reported. Weights come from checkpoint-<e>-model.pth (stored in fp16; weights are small, only
activations are at risk).
usage (GPU): python plots/verify/fp16_headroom_probe.py <arm>:<run dir>:<e0,e1,...> [...]   or  <arm>:<init dump .pth>:0   ->  plots/out/fp16_headroom_<arm>.json"""
import contextlib, io, json, os, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"; DATA = "/work/dlcsmall2/schrodi-imagenet"; LIMIT = 65504.0
DEV = torch.device("cuda"); torch.backends.cudnn.benchmark = False
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", DATA, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tv_datasets.ImageFolder(os.path.join(DATA, "train"))
images = utils.calibration_images(folder.samples, folder.loader, build_transform(True, args), 256, 0)
model = utils.build_model(args).to(DEV).eval()
for blk in model.blocks: blk.attn.fused_attn = False
NAMES = ("stream_in", "logits", "attn_out", "fc1_pre", "gelu", "fc2_out")

def probe(dtype):
    """-> ({block: {name: max |value|}}, {block: {"stream_in_cls": .., "stream_in_patch": ..}}, first non-finite module or None)"""
    peak = {b: {n: 0.0 for n in NAMES} for b in range(12)}; where = {b: {"stream_in_cls": 0.0, "stream_in_patch": 0.0} for b in range(12)}; first_bad = []
    def note(b, n, x, order):
        if not torch.isfinite(x).all(): first_bad.append((order, f"block {b} {n}"))
        peak[b][n] = max(peak[b][n], float(torch.nan_to_num(x.float().abs(), nan=float("inf")).max()))
    handles = []
    for b, blk in enumerate(model.blocks):
        H, scale = blk.attn.num_heads, blk.attn.scale
        def pre(mod, inp, b=b):
            x = inp[0]; note(b, "stream_in", x, 10 * b)
            where[b]["stream_in_cls"] = max(where[b]["stream_in_cls"], float(x[:, 0].float().abs().max())); where[b]["stream_in_patch"] = max(where[b]["stream_in_patch"], float(x[:, 1:].float().abs().max()))
        def qkv(mod, inp, out, b=b, H=H, scale=scale):
            B, N, _ = out.shape; q, k, _ = out.reshape(B, N, 3, H, -1).permute(2, 0, 3, 1, 4).unbind(0)
            note(b, "logits", (q * scale) @ k.transpose(-2, -1), 10 * b + 1)          # in the autocast dtype, exactly as the non-fused attention forms them
        handles += [blk.register_forward_pre_hook(pre), blk.attn.qkv.register_forward_hook(qkv),
                    blk.attn.register_forward_hook(lambda m, i, o, b=b: note(b, "attn_out", o, 10 * b + 2)), blk.mlp.fc1.register_forward_hook(lambda m, i, o, b=b: note(b, "fc1_pre", o, 10 * b + 3)),
                    blk.mlp.act.register_forward_hook(lambda m, i, o, b=b: note(b, "gelu", o, 10 * b + 4)), blk.mlp.fc2.register_forward_hook(lambda m, i, o, b=b: note(b, "fc2_out", o, 10 * b + 5))]
    out_bad = False
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        for i in range(0, len(images), 64): out_bad |= not torch.isfinite(model(images[i:i + 64].to(DEV))).all().item()
    for h in handles: h.remove()
    return peak, where, (min(first_bad)[1] if first_bad else ("model output" if out_bad else None))

for spec in sys.argv[1:]:
    arm, run_dir, epochs = spec.split(":"); result = {}
    print(f"\n======== {arm}  ({run_dir})   largest |value| per block under bf16 autocast; fp16 limit {LIMIT:.0f}")
    for e in [int(x) for x in epochs.split(",")]:
        p = f"{ROOT}/{run_dir}" if run_dir.endswith(".pth") else f"{ROOT}/{run_dir}/checkpoint-{e}-model.pth"   # a plain state-dict path (an init dump) is accepted as well
        if not os.path.exists(p): print(f"-- epoch {e}: no checkpoint"); continue
        model.float().load_state_dict({k: v.float() for k, v in torch.load(p, map_location="cpu", weights_only=True).items()})
        peak, where, _ = probe(torch.bfloat16); _, _, bad16 = probe(torch.float16)
        worst = max((peak[b][n], b, n) for b in range(12) for n in NAMES)
        result[e] = {"peak": peak, "where": where, "first_nonfinite_fp16": bad16, "worst": worst}
        print(f"-- epoch {e}: worst {worst[0]:9.1f} = {100 * worst[0] / LIMIT:5.1f}% of the fp16 limit (block {worst[1]}, {worst[2]});  fp16 forward: {'FIRST NON-FINITE at ' + bad16 if bad16 else 'finite'}")
        print("     blk | " + " ".join(f"{n:>10s}" for n in NAMES) + " | stream_in: cls, patches")
        for b in range(12): print(f"     {b:3d} | " + " ".join(f"{peak[b][n]:10.1f}" for n in NAMES) + f" | {where[b]['stream_in_cls']:9.1f} {where[b]['stream_in_patch']:9.1f}")
    json.dump(result, open(f"{ROOT}/plots/out/fp16_headroom_{arm}.json", "w"), indent=1)
    eps = sorted(result)
    if len(eps) >= 2:
        print(f"   growth of the worst value over epochs: " + ", ".join(f"ep {e}: {result[e]['worst'][0]:.0f} ({result[e]['worst'][2]} b{result[e]['worst'][1]})" for e in eps))
print("DONE")
