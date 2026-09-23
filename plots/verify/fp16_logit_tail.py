"""Tail of the attention logits over many TRAINING inputs: does (q * scale) @ k^T cross the fp16 limit (65504) on rare images?

fp16_headroom_probe.py shows that in the structure-only arms the logits of blocks 1-2 are the only quantity near the fp16 range
(5-10% of it on 256 images). A non-finite loss needs ONE logit above 65504 in one of ~1.3 M inputs per epoch, so the tail decides.
Here: N random training images with the run's augmentation (RandAugment, random erasing) AND its mixup / cutmix, forward through
blocks 0..LAST only, logits formed in float32 from the bf16-autocast q and k. Per block: quantiles of the per-image maximum |logit|,
the overall maximum, and how many images exceed the fp16 limit (they would give inf -> NaN in the softmax under fp16 autocast).
usage (GPU): python plots/verify/fp16_logit_tail.py <n images> <arm>:<run dir>:<epoch> [...]   ->  plots/out/fp16_logit_tail.json"""
import contextlib, io, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from timm.data import Mixup
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"; DATA = "/work/dlcsmall2/schrodi-imagenet"; LIMIT = 65504.0; LAST = 3
DEV = torch.device("cuda"); N = int(sys.argv[1])
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", DATA, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
folder = tv_datasets.ImageFolder(os.path.join(DATA, "train"), transform=build_transform(True, args))
g = torch.Generator().manual_seed(0); subset = torch.utils.data.Subset(folder, torch.randperm(len(folder), generator=g)[:N].tolist())
loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False, num_workers=int(os.environ.get("OMP_NUM_THREADS", 8)), drop_last=True, persistent_workers=True)
mix = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=1000)
print(f"{N} training images, augmentation as in training (aa {args.aa}, reprob {args.reprob}), mixup {args.mixup} / cutmix {args.cutmix} (prob {args.mixup_prob}, switch {args.mixup_switch_prob}, mode {args.mixup_mode})", flush=True)
model = utils.build_model(args).to(DEV).eval(); OUT = {}
for spec in sys.argv[2:]:
    arm, run_dir, e = spec.split(":"); p = f"{ROOT}/{run_dir}/checkpoint-{e}-model.pth"
    if not os.path.exists(p): print(f"== {arm} epoch {e}: no checkpoint", flush=True); continue
    model.float().load_state_dict({k: v.float() for k, v in torch.load(p, map_location="cpu", weights_only=True).items()})
    peaks = {b: [] for b in range(LAST)}; torch.manual_seed(0); np.random.seed(0)
    with torch.no_grad():
        for x, y in loader:
            x, _ = mix(x.to(DEV), y.to(DEV))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = model._pos_embed(model.patch_embed(x))
                for name in ("patch_drop", "norm_pre"): x = getattr(model, name)(x) if hasattr(model, name) else x
                for b in range(LAST):
                    blk = model.blocks[b]; qkv = blk.attn.qkv(blk.norm1(x)); B, T, _ = qkv.shape
                    q, k, _ = qkv.reshape(B, T, 3, blk.attn.num_heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
                    with torch.autocast("cuda", enabled=False): peaks[b].append(((q.float() * blk.attn.scale) @ k.float().transpose(-2, -1)).abs().amax(dim=(1, 2, 3)).cpu())
                    x = blk(x)
    OUT[f"{arm}@{e}"] = {}
    print(f"== {arm}, end of epoch {e}: per-image max |logit|   (fp16 limit {LIMIT:.0f})", flush=True)
    for b in range(LAST):
        v = torch.cat(peaks[b]).double().numpy(); qs = np.quantile(v, [0.5, 0.9, 0.99, 0.999])
        OUT[f"{arm}@{e}"][b] = {"median": qs[0], "q90": qs[1], "q99": qs[2], "q999": qs[3], "max": float(v.max()), "over_limit": int((v > LIMIT).sum()), "over_half": int((v > LIMIT / 2).sum()), "n": len(v)}
        print(f"   block {b}: median {qs[0]:9.1f}  90% {qs[1]:9.1f}  99% {qs[2]:9.1f}  99.9% {qs[3]:9.1f}  max {v.max():10.1f} = {100 * v.max() / LIMIT:6.1f}% of the limit | images above the limit: {(v > LIMIT).sum()} of {len(v)}, above half of it: {(v > LIMIT / 2).sum()}", flush=True)
json.dump(OUT, open(f"{ROOT}/plots/out/fp16_logit_tail.json", "w"), indent=1); print("DONE")
