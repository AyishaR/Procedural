"""Hunt the rare input that makes the fp16 forward pass non-finite, and localise the overflow.

The structure-only ksd arm (ftbck7sg) lost a finite loss once in epoch 9 and once in epoch 10 (about one event per 1.3 M inputs),
with no warning in loss or gradient norm, while on 51 k augmented inputs no quantity comes within 10x of the fp16 limit
(fp16_headroom_probe.py, fp16_logit_tail.py). So: forward a whole shard of training inputs (the run's augmentation + mixup / cutmix
in micro-batches of 128, as one rank sees them) through the checkpoint under autocast float16, exactly as training does (non-fused
attention). For every batch with a non-finite output: the first non-finite module under fp16, and, under bfloat16 on the same
batch, the largest |value| per block of stream / logits / attention output / fc1 pre-activation / GELU / fc2 output for the
offending images, whether bf16 is finite there, and the inputs themselves (saved for inspection).
usage (GPU): python plots/verify/fp16_nan_hunt.py <arm>:<run dir>:<epoch> <shard> <n shards> <images per shard>  ->  plots/out/fp16_nan_hunt/<arm>_e<epoch>_shard<k>.json"""
import contextlib, io, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from timm.data import Mixup
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"; DATA = "/work/dlcsmall2/schrodi-imagenet"; LIMIT = 65504.0; DEV = torch.device("cuda")
(arm, run_dir, epoch), shard, n_shards, n_img = sys.argv[1].split(":"), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", DATA, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tv_datasets.ImageFolder(os.path.join(DATA, "train"), transform=build_transform(True, args))
perm = torch.randperm(len(folder), generator=torch.Generator().manual_seed(1234)).tolist(); mine = perm[shard::n_shards][:n_img]
torch.manual_seed(1000 + shard); np.random.seed(1000 + shard)
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(folder, mine), batch_size=128, shuffle=False, num_workers=int(os.environ.get("OMP_NUM_THREADS", 8)), drop_last=True,
                                     worker_init_fn=lambda w: np.random.seed(1000 * (shard + 1) + w))
mix = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=1000)
model = utils.build_model(args).to(DEV).train()
for blk in model.blocks: blk.attn.fused_attn = False
model.load_state_dict({k: v.float() for k, v in torch.load(f"{ROOT}/{run_dir}/checkpoint-{epoch}-model.pth", map_location="cpu", weights_only=True).items()})
NAMES = ("stream_in", "logits", "attn_out", "fc1_pre", "gelu", "fc2_out")

def hooked(x, dtype):
    """-> ({block: {name: per-image max |value| (tensor B)}}, [non-finite modules in forward order], output)"""
    peak = {b: {} for b in range(12)}; bad = []; handles = []
    def note(b, n, t, order):
        if not torch.isfinite(t).all(): bad.append((order, f"block {b} {n}"))
        peak[b][n] = torch.nan_to_num(t.float().abs(), nan=float("inf"), posinf=float("inf")).flatten(1).amax(1).cpu()
    for b, blk in enumerate(model.blocks):
        H, scale = blk.attn.num_heads, blk.attn.scale
        def qkv(mod, inp, out, b=b, H=H, scale=scale):
            B, N, _ = out.shape; q, k, _ = out.reshape(B, N, 3, H, -1).permute(2, 0, 3, 1, 4).unbind(0); note(b, "logits", (q * scale) @ k.transpose(-2, -1), 10 * b + 1)
        handles += [blk.register_forward_pre_hook(lambda m, i, b=b: note(b, "stream_in", i[0], 10 * b)), blk.attn.qkv.register_forward_hook(qkv),
                    blk.attn.register_forward_hook(lambda m, i, o, b=b: note(b, "attn_out", o, 10 * b + 2)), blk.mlp.fc1.register_forward_hook(lambda m, i, o, b=b: note(b, "fc1_pre", o, 10 * b + 3)),
                    blk.mlp.act.register_forward_hook(lambda m, i, o, b=b: note(b, "gelu", o, 10 * b + 4)), blk.mlp.fc2.register_forward_hook(lambda m, i, o, b=b: note(b, "fc2_out", o, 10 * b + 5))]
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype): out = model(x)
    for h in handles: h.remove()
    return peak, [n for _, n in sorted(bad)], out

os.makedirs(f"{ROOT}/plots/out/fp16_nan_hunt", exist_ok=True); tag = f"{arm}_e{epoch}_shard{shard}"; events = []; seen = 0; t0 = time.time()
print(f"{tag}: {len(mine)} training inputs in micro-batches of 128 (augmentation + mixup {args.mixup} / cutmix {args.cutmix}), fp16 autocast, non-fused attention", flush=True)
for i, (x, y) in enumerate(loader):
    x, _ = mix(x.to(DEV, non_blocking=True), y.to(DEV)); seen += len(x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): out = model(x)
    if torch.isfinite(out).all(): continue
    rows = (~torch.isfinite(out).all(1)).nonzero().flatten().tolist()
    _, bad16, _ = hooked(x, torch.float16); peak, badbf, outbf = hooked(x, torch.bfloat16)
    worst = sorted(((float(peak[b][n][rows].max()), b, n) for b in range(12) for n in NAMES), reverse=True)[:6]
    typical = {f"{b}.{n}": float(peak[b][n].median()) for _, b, n in worst}
    ev = {"batch": i, "offending_rows": rows, "first_nonfinite_fp16": bad16[:4], "bf16_finite": bool(torch.isfinite(outbf).all()), "bf16_nonfinite_modules": badbf[:4],
          "worst_bf16_on_offenders": [{"value": v, "block": b, "what": n, "batch_median": typical[f"{b}.{n}"]} for v, b, n in worst],
          "input_abs_max": float(x[rows].abs().max()), "input_std": [float(x[r].std()) for r in rows]}
    events.append(ev); torch.save(x[rows].cpu(), f"{ROOT}/plots/out/fp16_nan_hunt/{tag}_batch{i}_inputs.pt")
    print(f"  NON-FINITE at batch {i} (input {seen}): rows {rows}; fp16 first non-finite: {bad16[:3]}; bf16 finite: {ev['bf16_finite']}", flush=True)
    for w in ev["worst_bf16_on_offenders"]: print(f"      bf16 on the offenders: block {w['block']} {w['what']:9s} max {w['value']:12.1f} = {100 * w['value'] / LIMIT:7.1f}% of the fp16 limit (batch median {w['batch_median']:.1f})", flush=True)
json.dump({"arm": arm, "epoch": epoch, "inputs": seen, "events": events, "seconds": time.time() - t0}, open(f"{ROOT}/plots/out/fp16_nan_hunt/{tag}.json", "w"), indent=1)
print(f"{tag}: {seen} inputs, {len(events)} non-finite batches, {time.time() - t0:.0f} s\nDONE", flush=True)
