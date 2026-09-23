"""Checkpoint autopsy for the early-lever mechanism plan (docs/early_lever_mechanism_plan.md, Phase 0). No training.
Reads the per-epoch model checkpoints (checkpoint-E-model.pth, fp16) that every run since 2026-09 keeps, and measures on FIXED
inputs what the training logs do not contain:
  A  functional state of every block over training (256 training images, evaluation transform, the calibration draw seed 0):
     attention entropy, common-query share, attention write split into the part common to all patch tokens of an image and the
     token-specific part (both relative to the rms token norm of the stream), fc1 pre-activation mean / std / P(z>0), GELU rms,
     rms of GELU', MLP write split the same way, and D_b = ||Pi (x_out - x_in)|| / ||Pi x_in|| (Pi centres the patch tokens
     within an image: how much the block changes the DIFFERENCES between tokens).
  B  realised relative weight change per epoch, ||W_{e+1} - W_e|| / ||W_e||, for q, k, v, proj, fc1, fc2, the LayerNorm gains
     and the gain-folded input matrices W diag(gamma): the quantity the "slow steps" idea is about, measured instead of nominal.
  C  (--probe) is the class information at block 7 / 9 absent, or only not aligned with the head? Trained linear probe (cls +
     mean patch token, 20 train images per class) against the head lens (model.norm -> fc_norm -> head on the cls token).
usage (GPU): python plots/verify/mechanism_autopsy.py [--probe] [--arms a,b] ; writes plots/out/mechanism_autopsy.json"""
import argparse, contextlib, io, json, math, os, sys, time
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"
ap = argparse.ArgumentParser(); ap.add_argument("--probe", action="store_true"); ap.add_argument("--arms", default=""); ap.add_argument("--out", default="/home/schrodi/Procedural/plots/out/mechanism_autopsy.json")
ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet"); ap.add_argument("--images", type=int, default=256); a = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
# arm -> (results id, seed, task, what it is); random-level references: no plain timm run kept its per-epoch checkpoints
ARMS = {
    "r0":              (29744580, 0, "kdyck", "timm random with per-epoch checkpoints (77.42)"),
    "ftbqu":           (29518360, 0, "kdyck", "random-level reference: timm with q,k x2.18 (78.05, n=3)"),
    "ftbrhosl":        (29626866, 0, "kdyck", "random-level reference: timm, slow late proj/fc2 (78.31)"),
    "ftb4i_kdyck":     (29448854, 1, "kdyck", "kdyck prefix 0-7 (79.89, n=3); seed 1"),
    "ftbanapermb7i":   (29737095, 0, "kdyck", "COMMITTED: scales + LN + entropy + mean gate, write side timm (80.37)"),
    "ftbanaperab7i":   (29733656, 0, "kdyck", "active-unit-gate twin (79.63)"),
    "ftbanap":         (29592459, 0, "kdyck", "scale-only recipe, blocks 0-8 (80.24)"),
    "ftbana":          (29572321, 0, "kdyck", "profile in raw weights, gains 1, normal steps (76.61)"),
    "ftbanal":         (29609180, 0, "kdyck", "ftbana + lr scales = slow steps (79.78)"),
    "ftb4i":           (29547835, 0, "ksd", "ksd prefix 0-7 (80.05)"),
    "ftbanakpermb7i":  (29736861, 0, "ksd", "COMMITTED on ksd (79.90)"),
    "ftbanakperab7i":  (29729558, 0, "ksd", "active-unit-gate twin (79.65)"),
    "ftbanak":         (29626632, 0, "ksd", "ksd scale-only recipe = random level (77.86)"),
}
if a.arms: ARMS = {k: v for k, v in ARMS.items() if k in a.arms.split(",")}
EPOCHS = [0, 1, 2, 4, 9, 19, 29, 49, 99, 199, 299]
PAIRS = [0, 4, 9, 19, 49, 99, 199]                      # drift between checkpoint e and e+1
E = 768
def ckpt(sid, seed, e):
    p = f"{ROOT}/results/imnet_base/results_IMNET_BASE_{sid}/s{seed}/checkpoint-{e}-model.pth"
    return {k: v.float() for k, v in torch.load(p, map_location="cpu", weights_only=True).items()} if os.path.exists(p) else None
folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
images = utils.calibration_images(folder.samples, folder.loader, build_transform(False, args), a.images, 0).to(DEV)
model = utils.build_model(args).to(DEV).eval()
rms_tok = lambda t: float(t.pow(2).sum(-1).mean().sqrt())          # rms token norm
@torch.no_grad()
def functional(model):
    out = {}; x = utils.block_input_stream(model, images)
    for b, blk in enumerate(model.blocks):
        p, _, q = utils._attention_rows(blk, x, blk.attn.qkv.weight)
        ent = float(-(p * (p + 1e-12).log()).sum(-1).mean())
        qf = q.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1); qm = qf.mean(1, keepdim=True)
        common_query = float((qm.pow(2).sum(-1) / qf.pow(2).sum(-1).mean(1, keepdim=True)).mean())
        att = blk.attn(blk.norm1(x)); mid = x + att
        y2 = blk.norm2(mid); z = blk.mlp.fc1(y2); mlp = blk.mlp(y2); xo = mid + mlp
        split = lambda w, ref: (rms_tok(w[:, 1:].mean(1, keepdim=True)) / rms_tok(ref[:, 1:]), rms_tok(w[:, 1:] - w[:, 1:].mean(1, keepdim=True)) / rms_tok(ref[:, 1:]))
        ac, as_ = split(att, x); mc, ms = split(mlp, mid)
        centre = lambda t: t[:, 1:] - t[:, 1:].mean(1, keepdim=True)
        gprime = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        out[b] = {"entropy": ent, "common_query": common_query, "att_common": ac, "att_specific": as_, "mlp_common": mc, "mlp_specific": ms,
                  "z_mean": float(z.mean()), "z_std": float(z.std()), "active": float((z > 0).float().mean()),
                  "gelu_rms": float(torch.nn.functional.gelu(z).pow(2).mean().sqrt()), "gprime_rms": float(gprime.pow(2).mean().sqrt()),
                  "D": float(centre(xo - x).norm() / centre(x).norm()), "cls_write": float(((xo - x)[:, 0].norm(dim=-1) / x[:, 0].norm(dim=-1)).mean())}
        x = xo
    return out
def drift(s0, s1):
    out = {}
    for b in range(12):
        W0, W1 = s0[f"blocks.{b}.attn.qkv.weight"], s1[f"blocks.{b}.attn.qkv.weight"]; g0, g1 = s0[f"blocks.{b}.norm1.weight"], s1[f"blocks.{b}.norm1.weight"]
        rel = lambda x0, x1: float((x1 - x0).norm() / x0.norm())
        d = {n: rel(W0[i * E:(i + 1) * E], W1[i * E:(i + 1) * E]) for i, n in enumerate(("q", "k", "v"))}
        d.update({n + "_folded": rel(W0[i * E:(i + 1) * E] * g0[None], W1[i * E:(i + 1) * E] * g1[None]) for i, n in enumerate(("q", "k", "v"))})
        for n, key in (("proj", "attn.proj.weight"), ("fc1", "mlp.fc1.weight"), ("fc2", "mlp.fc2.weight"), ("gain1", "norm1.weight"), ("gain2", "norm2.weight")):
            d[n] = rel(s0[f"blocks.{b}.{key}"], s1[f"blocks.{b}.{key}"])
        d["fc1_folded"] = rel(s0[f"blocks.{b}.mlp.fc1.weight"] * s0[f"blocks.{b}.norm2.weight"][None], s1[f"blocks.{b}.mlp.fc1.weight"] * s1[f"blocks.{b}.norm2.weight"][None])
        d["rms_q"] = float(W0[:E].pow(2).mean().sqrt() / 0.02); d["rms_fc1"] = float(s0[f"blocks.{b}.mlp.fc1.weight"].pow(2).mean().sqrt() / 0.02)
        out[b] = d
    out["patch_embed"] = float((s1["patch_embed.proj.weight"] - s0["patch_embed.proj.weight"]).norm() / s0["patch_embed.proj.weight"].norm())
    return out
RES = {"functional": {}, "drift": {}, "arms": {k: v[3] for k, v in ARMS.items()}}
t0 = time.time()
for arm, (sid, seed, task, what) in ARMS.items():
    RES["functional"][arm], RES["drift"][arm] = {}, {}
    for e in EPOCHS:
        sd = ckpt(sid, seed, e)
        if sd is None: continue
        model.load_state_dict(sd); RES["functional"][arm][e] = functional(model)
        if e in PAIRS:
            s1 = ckpt(sid, seed, e + 1)
            if s1 is not None: RES["drift"][arm][e] = drift(sd, s1)
    print(f"[{time.time() - t0:5.0f}s] {arm}: functional at {sorted(RES['functional'][arm])}, drift at {sorted(RES['drift'][arm])}", flush=True)
os.makedirs(f"{ROOT}/plots/out", exist_ok=True)
json.dump(RES, open(a.out, "w"))
mean = lambda xs: sum(xs) / len(xs)
print("\n== B. realised relative weight change per epoch, mean over blocks 1-7 | blocks 8-11 (x 1e-2)")
for arm in ARMS:
    print(f"-- {arm}: {ARMS[arm][3]}   [raw rms/0.02 at the first checkpoint, blocks 1-7: q {mean([RES['drift'][arm][min(RES['drift'][arm])][b]['rms_q'] for b in range(1, 8)]):.2f}, fc1 {mean([RES['drift'][arm][min(RES['drift'][arm])][b]['rms_fc1'] for b in range(1, 8)]):.2f}]" if RES["drift"][arm] else f"-- {arm}: no pairs")
    for e, d in sorted(RES["drift"][arm].items()):
        cell = lambda n: f"{100 * mean([d[b][n] for b in range(1, 8)]):5.2f}|{100 * mean([d[b][n] for b in range(8, 12)]):5.2f}"
        print(f"   ep {e:3d}->{e + 1:3d}: q {cell('q')}  k {cell('k')}  v {cell('v')}  proj {cell('proj')}  fc1 {cell('fc1')}  fc2 {cell('fc2')}  gain1 {cell('gain1')}  | folded q {cell('q_folded')} fc1 {cell('fc1_folded')} | patch {100 * d['patch_embed']:.2f}")
print("\n== A. functional state, mean over blocks 1-7 (fixed 256 training images)")
for arm in ARMS:
    print(f"-- {arm}: {ARMS[arm][3]}")
    for e, f in sorted(RES["functional"][arm].items()):
        m = lambda n: mean([f[b][n] for b in range(1, 8)])
        print(f"   ep {e:3d}: entropy {m('entropy'):5.2f} common-q {m('common_query'):.2f} | att write common {m('att_common'):.3f} specific {m('att_specific'):.3f} | z mean {m('z_mean'):+.2f} std {m('z_std'):.2f} active {m('active'):.3f} gelu {m('gelu_rms'):.3f} g' {m('gprime_rms'):.3f} | mlp write common {m('mlp_common'):.3f} specific {m('mlp_specific'):.3f} | D {m('D'):.3f}")

if a.probe:
    print("\n== C. block-7 / block-9 class information: trained linear probe against the head lens")
    import random
    per_class = {}
    for i, (_, c) in enumerate(folder.samples): per_class.setdefault(c, []).append(i)
    rng = random.Random(0); train_idx = [i for c in sorted(per_class) for i in rng.sample(per_class[c], 20)]
    vfolder = tv_datasets.ImageFolder(os.path.join(a.data_path, "val")); vper = {}
    for i, (_, c) in enumerate(vfolder.samples): vper.setdefault(c, []).append(i)
    val_idx = [i for c in sorted(vper) for i in rng.sample(vper[c], 10)]
    tf = build_transform(False, args); folder.transform = tf; vfolder.transform = tf
    loaders = {"train": torch.utils.data.DataLoader(torch.utils.data.Subset(folder, train_idx), batch_size=250, num_workers=14, shuffle=False),
               "val": torch.utils.data.DataLoader(torch.utils.data.Subset(vfolder, val_idx), batch_size=250, num_workers=14, shuffle=False)}
    PROBE = [(arm, e) for arm in ("ftbqu", "ftbanapermb7i", "ftb4i_kdyck", "ftbanap") if arm in ARMS for e in (29, 299)]
    models = {}
    for arm, e in PROBE:
        sd = ckpt(ARMS[arm][0], ARMS[arm][1], e)
        if sd is None: continue
        m = utils.build_model(args).to(DEV).eval(); m.load_state_dict(sd); models[(arm, e)] = m.half()
    feats = {k: {s: {7: [], 9: []} for s in loaders} for k in models}; lens = {k: {7: 0, 9: 0, 11: 0} for k in models}; labels = {s: [] for s in loaders}; n_val = 0
    with torch.no_grad():
        for split, loader in loaders.items():
            for x_img, y in loader:
                x_img = x_img.to(DEV).half(); labels[split].append(y)
                for key, m in models.items():
                    x = utils.block_input_stream(m, x_img).half()
                    for b, blk in enumerate(m.blocks):
                        x = blk(x)
                        if b in (7, 9): feats[key][split][b].append(torch.cat([x[:, 0], x[:, 1:].mean(1)], -1).float().cpu())
                        if split == "val" and b in (7, 9, 11):
                            lens[key][b] += int((m.head(m.fc_norm(m.norm(x))[:, 0]).argmax(-1).cpu() == y).sum())
                if split == "val": n_val += len(y)
    RES["probe"] = {}
    for key in models:
        ytr, yva = torch.cat(labels["train"]).to(DEV), torch.cat(labels["val"]).to(DEV); row = {}
        for b in (7, 9):
            Xtr, Xva = torch.cat(feats[key]["train"][b]).to(DEV), torch.cat(feats[key]["val"][b]).to(DEV)
            mu, sd_ = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6; Xtr, Xva = (Xtr - mu) / sd_, (Xva - mu) / sd_
            torch.manual_seed(0); lin = torch.nn.Linear(Xtr.shape[1], 1000).to(DEV); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
            steps = 600; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            for _ in range(steps):
                opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xtr), ytr, label_smoothing=0.1).backward(); opt.step(); sched.step()
            row[b] = 100 * float((lin(Xva).argmax(-1) == yva).float().mean())
        RES["probe"][f"{key[0]}@{key[1]}"] = {"probe7": row[7], "probe9": row[9], "lens7": 100 * lens[key][7] / n_val, "lens9": 100 * lens[key][9] / n_val, "model": 100 * lens[key][11] / n_val}
        print(f"   {key[0]:15s} epoch {key[1]:3d}: block 7 probe {row[7]:5.1f} lens {100 * lens[key][7] / n_val:5.1f} | block 9 probe {row[9]:5.1f} lens {100 * lens[key][9] / n_val:5.1f} | model {100 * lens[key][11] / n_val:5.1f}   (10k val images, probe on 20k train images)")
    json.dump(RES, open(a.out, "w"))
print("DONE")
