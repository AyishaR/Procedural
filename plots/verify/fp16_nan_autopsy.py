"""Autopsy of a non-finite training loss from the dump that engine.py writes (utils.dump_nonfinite_state): the weights of that moment
and the offending micro-batch. Forward under autocast float16 exactly as in training (non-fused attention): which images give a
non-finite output, and which module is the first with a non-finite value; forward under bfloat16 on the same batch: is everything
finite, and how large is every quantity on the offending images against the rest of the batch.
usage (GPU): python plots/verify/fp16_nan_autopsy.py <..._weights.pt> <..._rankR.pt> [<out json>]"""
import contextlib, io, json, sys
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
DEV = torch.device("cuda"); LIMIT = 65504.0
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
model = utils.build_model(args).to(DEV).train()
for blk in model.blocks: blk.attn.fused_attn = False
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True)); batch = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
x = batch["samples"].float().to(DEV); print(f"dump: epoch {batch['epoch']}, micro-step {batch['step']}, {batch['amp_dtype']}; batch {tuple(x.shape)}; stored output finite rows: {int(torch.isfinite(batch['output']).all(1).sum())} of {len(x)}")
NAMES = ("stream_in", "norm1_out", "q", "k", "logits", "attn_out", "fc1_pre", "gelu", "fc2_out")

def hooked(dtype):
    peak = {b: {} for b in range(12)}; bad = []; handles = []
    def note(b, n, t, order):
        if not torch.isfinite(t).all(): bad.append((order, f"block {b} {n}"))
        peak[b][n] = torch.nan_to_num(t.float().abs(), nan=float("inf"), posinf=float("inf")).flatten(1).amax(1).cpu()
    for b, blk in enumerate(model.blocks):
        H, scale = blk.attn.num_heads, blk.attn.scale
        def qkv(mod, inp, out, b=b, H=H, scale=scale):
            B, N, _ = out.shape; q, k, _ = out.reshape(B, N, 3, H, -1).permute(2, 0, 3, 1, 4).unbind(0)
            note(b, "q", q, 10 * b + 2); note(b, "k", k, 10 * b + 3); note(b, "logits", (q * scale) @ k.transpose(-2, -1), 10 * b + 4)
        handles += [blk.register_forward_pre_hook(lambda m, i, b=b: note(b, "stream_in", i[0], 10 * b)), blk.norm1.register_forward_hook(lambda m, i, o, b=b: note(b, "norm1_out", o, 10 * b + 1)),
                    blk.attn.qkv.register_forward_hook(qkv), blk.attn.register_forward_hook(lambda m, i, o, b=b: note(b, "attn_out", o, 10 * b + 5)),
                    blk.mlp.fc1.register_forward_hook(lambda m, i, o, b=b: note(b, "fc1_pre", o, 10 * b + 6)), blk.mlp.act.register_forward_hook(lambda m, i, o, b=b: note(b, "gelu", o, 10 * b + 7)),
                    blk.mlp.fc2.register_forward_hook(lambda m, i, o, b=b: note(b, "fc2_out", o, 10 * b + 8))]
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype): out = model(x)
    for h in handles: h.remove()
    return peak, [n for _, n in sorted(bad)], out

p16, bad16, out16 = hooked(torch.float16); pbf, badbf, outbf = hooked(torch.bfloat16)
rows = (~torch.isfinite(out16).all(1)).nonzero().flatten().tolist(); rest = [i for i in range(len(x)) if i not in rows]
print(f"fp16 forward: non-finite output rows {rows}; non-finite modules in forward order: {bad16[:6]}")
print(f"bf16 forward: output finite: {bool(torch.isfinite(outbf).all())}; non-finite modules: {badbf[:6]}")
result = {"rows": rows, "fp16_nonfinite_modules": bad16, "bf16_finite": bool(torch.isfinite(outbf).all()), "table": {}}
if rows:
    print("largest |value| under bf16: offending images | median of the other images | ratio to the fp16 limit (offenders)")
    print("   blk | " + " ".join(f"{n:>22s}" for n in NAMES))
    for b in range(12):
        cells = []
        for n in NAMES:
            o = float(pbf[b][n][rows].max()); m = float(pbf[b][n][rest].median()) if rest else float("nan"); result["table"][f"{b}.{n}"] = {"offenders": o, "others_median": m}
            cells.append(f"{o:9.3g}|{m:7.3g}{'  !!' if o > LIMIT else '    '}")
        print(f"   {b:3d} | " + " ".join(f"{c:>22s}" for c in cells))
    xs = x[rows]; print(f"offending inputs: abs max {float(xs.abs().max()):.2f}, std {[round(float(v.std()), 3) for v in xs]}, mean {[round(float(v.mean()), 3) for v in xs]} (batch std median {float(x.flatten(1).std(1).median()):.3f})")
    t = batch["targets"][rows]; print(f"targets of the offenders: top label weights {[ [round(float(v), 2) for v in r.topk(2).values] for r in t]}")
if len(sys.argv) > 3: json.dump(result, open(sys.argv[3], "w"), indent=1)
print("DONE")
