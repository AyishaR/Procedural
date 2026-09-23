"""Phase 0 of docs/early_lever_mechanism_plan.md on the GPU, sharded (one shard per GPU / array task). No training.
Modes
  zoo     every finished run that kept per-epoch checkpoints (plots/out/mechanism_run_index.json): functional state of every block
          on 256 fixed training images at epochs 0, 4, 9, 19, 29 (incl. which key carries the sink), realised relative weight
          change for the pairs 0->1, 4->5, 9->10, 19->20, and at epochs 29 and 299 the head lens and a light trained probe
          (class token + mean patch token; 10 train / 5 val images per class) at blocks 7 and 9.      -> the mediator screen (P0.1)
  depth   reference arms x epochs 9, 19, 29, 49, 99, 299: head lens and trained probe (20 train / 10 val images per class) at
          EVERY block: where is the class computed, and is it absent or only not head-aligned?                        (P0.2)
  fitgap  final checkpoints of reference arms on 25 training images per class WITHOUT augmentation and 25 validation images per
          class: accuracy and cross-entropy, i.e. the generalisation gap behind the fit deficit.                          (P0.3)
usage: python plots/verify/mechanism_phase0_gpu.py --mode zoo --shard 0 --nshards 8   (writes plots/out/phase0/<mode>_shard<i>.json)"""
import argparse, contextlib, io, json, math, os, random, sys, time
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
from torchvision import datasets as tv_datasets, transforms as T
from datasets import build_transform
ROOT = "/home/schrodi/Procedural"
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=("zoo", "depth", "fitgap", "clspath")); ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--only", default="", help="comma separated arms: restrict any mode to them"); ap.add_argument("--tag", default="", help="output name plots/out/phase0/<mode>_shard<tag>.json instead of the shard number")
ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--data_path", default="/work/dlcsmall2/schrodi-imagenet"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
DEV = torch.device("cuda"); torch.backends.cudnn.benchmark = True
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", a.data_path, "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
E = 768
REFERENCE = {  # arm -> (results dir, what)
    "r":               ("results/imnet_base/results_IMNET_BASE_29384839/s0", "timm random (78.28), final checkpoint only"),
    "r0":              ("results/imnet_base/results_IMNET_BASE_29744580/s0", "timm random WITH per-epoch checkpoints (77.42), 2026-09-21"),
    "ftbqu":           ("results/imnet_base/results_IMNET_BASE_29518360/s0", "random-level: timm, q,k x2.18 (78.05 n=3)"),
    "ftbrhosl":        ("results/imnet_base/results_IMNET_BASE_29626866/s0", "random-level: timm, slow late proj/fc2 (78.31)"),
    "ftb4i_kdyck":     ("results/imnet_base/results_IMNET_BASE_29448854/s1", "kdyck prefix 0-7, seed 1"),
    "ftbanapermb7i":   ("results/imnet_base/results_IMNET_BASE_29737095/s0", "C kdyck (80.37)"),
    "ftbanaperab7i":   ("results/imnet_base/results_IMNET_BASE_29733656/s0", "active-unit twin kdyck (79.63)"),
    "ftbanapermb7w":   ("results/imnet_base/results_IMNET_BASE_29729556/s0", "write-matched, mean gate, kdyck (78.91)"),
    "ftbanap":         ("results/imnet_base/results_IMNET_BASE_29592459/s0", "kdyck scale-only 0-8 (80.24)"),
    "ftbana":          ("results/imnet_base/results_IMNET_BASE_29572321/s0", "profile in raw weights, normal steps (76.61)"),
    "ftbanal":         ("results/imnet_base/results_IMNET_BASE_29609180/s0", "ftbana + lr scales (79.78)"),
    "ftb4i":           ("results/imnet_base/results_IMNET_BASE_29547835/s0", "ksd prefix 0-7 (80.05)"),
    "ftbanakpermb7i":  ("results/imnet_base/results_IMNET_BASE_29736861/s0", "C ksd (79.90)"),
    "ftbanakperab7i":  ("results/imnet_base/results_IMNET_BASE_29729558/s0", "active-unit twin ksd (79.65)"),
    "ftbanak":         ("results/imnet_base/results_IMNET_BASE_29626632/s0", "ksd scale-only = random level (77.86)"),
}
def state(d, e):
    p = f"{ROOT}/{d}/checkpoint-{e}-model.pth"
    return {k: v.float() for k, v in torch.load(p, map_location="cpu", weights_only=True).items()} if os.path.exists(p) else None
rms_tok = lambda t: float(t.pow(2).sum(-1).mean().sqrt())

@torch.no_grad()
def functional(model, images):
    out = {}; x = utils.block_input_stream(model, images)
    for b, blk in enumerate(model.blocks):
        p, _, q = utils._attention_rows(blk, x, blk.attn.qkv.weight)
        ent = float(-(p * (p + 1e-12).log()).sum(-1).mean())
        qf = q.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1); qm = qf.mean(1, keepdim=True)
        key_mass = p.mean((1, 2)); top = key_mass.argmax(-1)                       # (B, N) mass per key, averaged over heads and queries
        att = blk.attn(blk.norm1(x)); mid = x + att; y2 = blk.norm2(mid); z = blk.mlp.fc1(y2); mlp = blk.mlp(y2); xo = mid + mlp
        split = lambda w, ref: (rms_tok(w[:, 1:].mean(1, keepdim=True)) / rms_tok(ref[:, 1:]), rms_tok(w[:, 1:] - w[:, 1:].mean(1, keepdim=True)) / rms_tok(ref[:, 1:]))
        ac, as_ = split(att, x); mc, ms = split(mlp, mid); centre = lambda t: t[:, 1:] - t[:, 1:].mean(1, keepdim=True)
        gp = 0.5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        out[b] = {"entropy": ent, "common_query": float((qm.pow(2).sum(-1) / qf.pow(2).sum(-1).mean(1, keepdim=True)).mean()),
                  "top_key_mass": float(key_mass.max(-1).values.mean()), "top_key_is_cls": float((top == 0).float().mean()),
                  "top_key_modal_share": float(torch.bincount(top, minlength=p.shape[-1]).max()) / len(top),
                  "att_common": ac, "att_specific": as_, "mlp_common": mc, "mlp_specific": ms, "z_mean": float(z.mean()), "z_std": float(z.std()),
                  "active": float((z > 0).float().mean()), "gelu_rms": float(torch.nn.functional.gelu(z).pow(2).mean().sqrt()),
                  "gprime_rms": float(gp.pow(2).mean().sqrt()), "D": float(centre(xo - x).norm() / centre(x).norm())}
        x = xo
    return out
def drift(s0, s1):
    out = {}; rel = lambda x0, x1: float((x1 - x0).norm() / x0.norm())
    for b in range(12):
        W0, W1 = s0[f"blocks.{b}.attn.qkv.weight"], s1[f"blocks.{b}.attn.qkv.weight"]
        d = {n: rel(W0[i * E:(i + 1) * E], W1[i * E:(i + 1) * E]) for i, n in enumerate(("q", "k", "v"))}
        for n, key in (("proj", "attn.proj.weight"), ("fc1", "mlp.fc1.weight"), ("fc2", "mlp.fc2.weight"), ("gain1", "norm1.weight"), ("gain2", "norm2.weight")):
            d[n] = rel(s0[f"blocks.{b}.{key}"], s1[f"blocks.{b}.{key}"])
        d["rms_q"] = float(W0[:E].pow(2).mean().sqrt() / 0.02); d["rms_k"] = float(W0[E:2 * E].pow(2).mean().sqrt() / 0.02); d["rms_fc1"] = float(s0[f"blocks.{b}.mlp.fc1.weight"].pow(2).mean().sqrt() / 0.02)
        out[b] = d
    out["patch_embed"] = rel(s0["patch_embed.proj.weight"], s1["patch_embed.proj.weight"])
    return out

# ---- image subsets, cached once per process as uint8 (normalised on the GPU)
full_tf = build_transform(False, args); pre = [t for t in full_tf.transforms if not isinstance(t, (T.ToTensor, T.Normalize))]
norm = [t for t in full_tf.transforms if isinstance(t, T.Normalize)][0]
MEAN, STD = torch.tensor(norm.mean, device=DEV).view(1, 3, 1, 1), torch.tensor(norm.std, device=DEV).view(1, 3, 1, 1)
def subset(split, per_class, seed=0):
    folder = tv_datasets.ImageFolder(os.path.join(a.data_path, split), transform=T.Compose(pre + [T.PILToTensor()]))
    per = {}
    for i, (_, c) in enumerate(folder.samples): per.setdefault(c, []).append(i)
    rng = random.Random(seed); idx = [i for c in sorted(per) for i in rng.sample(per[c], per_class)]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(folder, idx), batch_size=256, num_workers=int(os.environ.get("WORKERS", 12)), shuffle=False)
    xs, ys = [], []
    for x, y in loader: xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys)
def batches(X, Y, bs=500):
    for i in range(0, len(X), bs):
        yield ((X[i:i + bs].to(DEV, non_blocking=True).float() / 255.0 - MEAN) / STD).half(), Y[i:i + bs]
@torch.no_grad()
def features(model, X, Y, blocks):
    """cls + mean-patch features of the listed blocks, and head-lens / model hits, for a half-precision model"""
    feats = {b: [] for b in blocks}; hits = {b: 0 for b in blocks}; hits[11] = 0; ce = 0.0
    for x_img, y in batches(X, Y):
        x = utils.block_input_stream(model, x_img).half()
        for b, blk in enumerate(model.blocks):
            x = blk(x)
            if b in blocks or b == 11:
                logits = model.head(model.fc_norm(model.norm(x))[:, 0]).float()
                hits[b] += int((logits.argmax(-1).cpu() == y).sum())
                if b == 11: ce += float(torch.nn.functional.cross_entropy(logits, y.to(DEV), reduction="sum"))
            if b in blocks: feats[b].append(torch.cat([x[:, 0], x[:, 1:].mean(1)], -1).float().cpu())
    return {b: torch.cat(v) for b, v in feats.items()}, {b: 100.0 * h / len(Y) for b, h in hits.items()}, ce / len(Y)
def probe(Xtr, ytr, Xva, yva, steps=600):
    Xtr, Xva, ytr, yva = Xtr.to(DEV), Xva.to(DEV), ytr.to(DEV), yva.to(DEV)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    torch.manual_seed(0); lin = torch.nn.Linear(Xtr.shape[1], 1000).to(DEV); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xtr), ytr, label_smoothing=0.1).backward(); opt.step(); sched.step()
    with torch.no_grad(): return 100.0 * float((lin(Xva).argmax(-1) == yva).float().mean())

os.makedirs(f"{ROOT}/plots/out/phase0", exist_ok=True); OUT = f"{ROOT}/plots/out/phase0/{a.mode}_shard{a.tag or a.shard}.json"; ONLY = set(x for x in a.only.split(",") if x); t0 = time.time(); RES = {}
model = utils.build_model(args).to(DEV).eval()
if a.mode == "zoo":
    runs = [r for r in json.load(open(f"{ROOT}/plots/out/mechanism_run_index.json")) if not ONLY or r["arm"] in ONLY]; runs = runs[a.shard::a.nshards]
    if a.limit: runs = runs[:a.limit]
    folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train"))
    calib = utils.calibration_images(folder.samples, folder.loader, full_tf, 256, 0).to(DEV)
    Xtr, ytr = subset("train", 10); Xva, yva = subset("val", 5); print(f"[{time.time() - t0:4.0f}s] cached {len(Xtr)} train / {len(Xva)} val images; {len(runs)} runs in this shard", flush=True)
    for r in runs:
        key = f"{r['arm']}|{r['sid']}|s{r['seed']}"; rec = {"arm": r["arm"], "task": r["task"], "final": r["final"], "train_loss": r["train_loss"], "functional": {}, "drift": {}, "probe": {}}
        for e in (0, 4, 9, 19, 29):
            s0 = state(r["dir"], e)
            if s0 is None: continue
            model.float().load_state_dict(s0); rec["functional"][e] = functional(model, calib)
            if e != 29:
                s1 = state(r["dir"], e + 1)
                if s1 is not None: rec["drift"][e] = drift(s0, s1)
        for e in (29, 299):
            s0 = state(r["dir"], e)
            if s0 is None: continue
            model.float().load_state_dict(s0); model.half()
            ftr, _, _ = features(model, Xtr, ytr, (7, 9)); fva, lens, ce = features(model, Xva, yva, (7, 9))
            rec["probe"][e] = {"probe7": probe(ftr[7], ytr, fva[7], yva), "probe9": probe(ftr[9], ytr, fva[9], yva), "lens7": lens[7], "lens9": lens[9], "model": lens[11], "val_ce": ce}
        RES[key] = rec; json.dump(RES, open(OUT, "w"))
        print(f"[{time.time() - t0:4.0f}s] {key} final {r['final']:.2f} | D@0 {sum(rec['functional'][0][b]['D'] for b in range(1, 8)) / 7:.3f} | @29 probe7 {rec['probe'].get(29, {}).get('probe7', float('nan')):.1f} lens7 {rec['probe'].get(29, {}).get('lens7', float('nan')):.1f}", flush=True)
elif a.mode == "depth":
    items = [(arm, e) for arm in REFERENCE if arm != "r" for e in (9, 19, 29, 49, 99, 299)] + [("r", 299)]; items = [x for x in items if not ONLY or x[0] in ONLY][a.shard::a.nshards]
    if a.limit: items = items[:a.limit]
    Xtr, ytr = subset("train", 20); Xva, yva = subset("val", 10); print(f"[{time.time() - t0:4.0f}s] cached {len(Xtr)} / {len(Xva)} images; {len(items)} models in this shard", flush=True)
    for arm, e in items:
        s0 = state(REFERENCE[arm][0], e)
        if s0 is None: print("missing", arm, e); continue
        model.float().load_state_dict(s0); model.half(); blocks = tuple(range(12))
        ftr, _, _ = features(model, Xtr, ytr, blocks); fva, lens, ce = features(model, Xva, yva, blocks)
        RES[f"{arm}@{e}"] = {"what": REFERENCE[arm][1], "lens": lens, "probe": {b: probe(ftr[b], ytr, fva[b], yva) for b in blocks}, "val_ce": ce}
        json.dump(RES, open(OUT, "w"))
        print(f"[{time.time() - t0:4.0f}s] {arm}@{e}: probe " + " ".join(f"{RES[f'{arm}@{e}']['probe'][b]:4.1f}" for b in blocks) + " | lens " + " ".join(f"{lens[b]:4.1f}" for b in blocks), flush=True)
elif a.mode == "clspath":
    # Is the block-7 lens transient a shortcut THROUGH THE CLASS TOKEN (class token pools the patches, early blocks make it class-aligned)?
    # (i) trained probes on the class token alone against the mean patch token alone, blocks 3, 5, 7, 9, 11;
    # (ii) how the class token is fed in blocks 0-7: entropy of its attention row, mass it puts on itself, size of the attention and
    #      MLP updates it receives relative to its norm, and the same two update sizes for the average patch token.
    idx = {r["arm"] + ("_kdyck" if r["arm"] == "ftb4i" and r["task"] == "kdyck" else ""): r["dir"] for r in json.load(open(f"{ROOT}/plots/out/mechanism_run_index.json")) if r["seed"] in (0, 1)}
    ARMS_C = ["r0", "ftbqu", "ftbrhosl", "ftbvd", "ftbanapermb7i", "ftbanaperab7i", "ftb4i_kdyck", "ftbanap", "ftbana", "ftbqmln", "ftbqmlnvo", "ftbanaperab7w", "ftbanakpermb7i", "ftbanak", "ftb4i", "ftbrhop"]
    items = [(arm, e) for arm in ARMS_C if arm in idx and (not ONLY or arm in ONLY) for e in (4, 9, 29)][a.shard::a.nshards]
    folder = tv_datasets.ImageFolder(os.path.join(a.data_path, "train")); calib = utils.calibration_images(folder.samples, folder.loader, full_tf, 256, 0).to(DEV)
    Xtr, ytr = subset("train", 10); Xva, yva = subset("val", 5); print(f"[{time.time() - t0:4.0f}s] cached; {len(items)} models", flush=True)
    BL = (3, 5, 7, 9, 11)
    for arm, e in items:
        s0 = state(idx[arm], e)
        if s0 is None: print("missing", arm, e); continue
        model.float().load_state_dict(s0); stats = {}
        with torch.no_grad():
            x = utils.block_input_stream(model, calib)
            for b, blk in enumerate(model.blocks):
                pr, _, _ = utils._attention_rows(blk, x, blk.attn.qkv.weight); row = pr[:, :, 0, :]              # class-token query, (B, H, N)
                att = blk.attn(blk.norm1(x)); mid = x + att; mlp = blk.mlp(blk.norm2(mid))
                stats[b] = {"cls_entropy": float(-(row * (row + 1e-12).log()).sum(-1).mean()), "cls_self_mass": float(row[..., 0].mean()),
                            "cls_att_update": float((att[:, 0].norm(dim=-1) / x[:, 0].norm(dim=-1)).mean()), "cls_mlp_update": float((mlp[:, 0].norm(dim=-1) / mid[:, 0].norm(dim=-1)).mean()),
                            "patch_att_update": float((att[:, 1:].norm(dim=-1) / x[:, 1:].norm(dim=-1)).mean()), "patch_mlp_update": float((mlp[:, 1:].norm(dim=-1) / mid[:, 1:].norm(dim=-1)).mean()),
                            "cls_att_is_common": float(torch.nn.functional.cosine_similarity(att[:, 0], att[:, 1:].mean(1), dim=-1).mean())}
                x = mid + mlp
        model.half(); ftr, _, _ = features(model, Xtr, ytr, BL); fva, lens, _ = features(model, Xva, yva, BL)
        RES[f"{arm}@{e}"] = {"stats": stats, "lens": lens, "probe_cls": {b: probe(ftr[b][:, :E], ytr, fva[b][:, :E], yva) for b in BL},
                             "probe_patch": {b: probe(ftr[b][:, E:], ytr, fva[b][:, E:], yva) for b in BL}}
        json.dump(RES, open(OUT, "w")); r = RES[f"{arm}@{e}"]
        print(f"[{time.time() - t0:4.0f}s] {arm}@{e}: probe cls " + " ".join(f"{r['probe_cls'][b]:4.1f}" for b in BL) + " | patch-mean " + " ".join(f"{r['probe_patch'][b]:4.1f}" for b in BL) + " | lens " + " ".join(f"{lens[b]:4.1f}" for b in BL), flush=True)
else:
    items = [x for x in REFERENCE if not ONLY or x in ONLY][a.shard::a.nshards]
    Xtr, ytr = subset("train", 25, seed=1); Xva, yva = subset("val", 25, seed=1); print(f"[{time.time() - t0:4.0f}s] cached {len(Xtr)} / {len(Xva)} images; {len(items)} models", flush=True)
    for arm in items:
        s0 = state(REFERENCE[arm][0], 299)
        if s0 is None: print("missing", arm); continue
        model.float().load_state_dict(s0); model.half()
        _, ltr, cetr = features(model, Xtr, ytr, ()); _, lva, ceva = features(model, Xva, yva, ())
        RES[arm] = {"what": REFERENCE[arm][1], "train_clean_acc": ltr[11], "train_clean_ce": cetr, "val_acc": lva[11], "val_ce": ceva}
        json.dump(RES, open(OUT, "w"))
        print(f"[{time.time() - t0:4.0f}s] {arm}: clean train acc {ltr[11]:.2f} ce {cetr:.3f} | val acc {lva[11]:.2f} ce {ceva:.3f} | gap acc {ltr[11] - lva[11]:.2f} ce {ceva - cetr:.3f}", flush=True)
print("DONE", OUT, flush=True)
