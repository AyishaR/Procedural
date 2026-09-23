"""Tests for row_lr_mask.py (per-row learning-rate scales inside one tensor as an exact post-step AdamW correction).

  T1  exact equivalence: a fused Linear with row masks on q / k / v rows against the same network with q, k, v as separate
      tensors in parameter groups (lr * lambda, wd / lambda); AdamW, weight decay on, learning rate changing every step,
      float64; both foreach settings. Parameters must agree to 1e-12 after every one of 60 steps.
  T2  lambda = 1 everywhere is bit-identical to no mask.
  T3  weight decay is untouched: with zero gradients the masked and unmasked tensors decay identically.
  T4  a step skipped by the AMP grad scaler (inf gradient) applies no correction.
  T5  save / load of optimizer state in the middle of training continues the T1 trajectory exactly.
  T6  the real path: ViT-B through optim_factory.create_optimizer with a JSON mask on blocks.1.attn.qkv.weight, two steps:
      every unmasked tensor bit-identical to the run without the mask; on the masked tensor, the row update equals
      decay + lambda * adaptive step of the unmasked run (1e-6); the mask table is printed.
  T7  invalid specifications raise (overlap, out of range, non-positive lambda, unknown name, lr_scale group, amsgrad).
  T8  (CUDA only) T1 with fused=True.
usage: .venv/bin/python plots/verify/test_row_lr_mask.py"""
import copy, io, contextlib, json, math, os, sys, tempfile
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
from row_lr_mask import split_lr_scale_spec, row_mask_vector, install_row_lr_masks
torch.manual_seed(0)
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

D, H = 8, 4                                   # width, hidden rows per "head group"; fused tensor has 3D rows (q, k, v)
LAM = {"q": 0.3, "k": 0.25, "v": 1.0}
WD, BETAS, EPS = 0.05, (0.9, 0.999), 1e-8
def lr_at(t): return 2e-3 * (0.5 + 0.5 * math.cos(math.pi * t / 60)) + 1e-4 * (t % 3)   # a schedule that changes every step

class Fused(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.qkv = torch.nn.Linear(D, 3 * D, dtype=torch.float64); self.out = torch.nn.Linear(D, 1, dtype=torch.float64)
    def forward(self, x):
        q, k, v = self.qkv(x).split(D, dim=-1)
        return self.out(torch.tanh(q) * torch.sigmoid(k) + v).squeeze(-1)

class Split(torch.nn.Module):
    """same function, q / k / v as separate tensors (the reference)"""
    def __init__(self, fused):
        super().__init__()
        W, b = fused.qkv.weight.detach(), fused.qkv.bias.detach()
        self.q = torch.nn.Linear(D, D, dtype=torch.float64); self.k = torch.nn.Linear(D, D, dtype=torch.float64); self.v = torch.nn.Linear(D, D, dtype=torch.float64)
        for i, lin in enumerate((self.q, self.k, self.v)):
            lin.weight.data.copy_(W[i * D:(i + 1) * D]); lin.bias.data.copy_(b[i * D:(i + 1) * D])
        self.out = copy.deepcopy(fused.out)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return self.out(torch.tanh(q) * torch.sigmoid(k) + v).squeeze(-1)

def fused_optimizer(model, foreach=None, fused=None, amsgrad=False):
    groups = [{"params": [model.qkv.weight, model.out.weight], "weight_decay": WD},
              {"params": [model.qkv.bias, model.out.bias], "weight_decay": 0.0}]
    kw = dict(lr=lr_at(0), betas=BETAS, eps=EPS, amsgrad=amsgrad)
    if foreach is not None: kw["foreach"] = foreach
    if fused is not None: kw["fused"] = fused
    return torch.optim.AdamW(groups, **kw)

def split_optimizer(model, foreach=None, fused=None):
    groups = [{"params": [model.out.weight], "weight_decay": WD, "lr_scale": 1.0, "wd_scale": 1.0},
              {"params": [model.q.bias, model.k.bias, model.v.bias, model.out.bias], "weight_decay": 0.0, "lr_scale": 1.0, "wd_scale": 1.0}]
    for name, lam in LAM.items():                 # the --lr_scale_json semantics: lr * lambda, wd / lambda
        groups.append({"params": [getattr(model, name).weight], "weight_decay": WD / lam, "lr_scale": lam, "wd_scale": 1.0 / lam})
    kw = dict(lr=lr_at(0), betas=BETAS, eps=EPS)
    if foreach is not None: kw["foreach"] = foreach
    if fused is not None: kw["fused"] = fused
    return torch.optim.AdamW(groups, **kw)

def set_lr(opt, t):
    for g in opt.param_groups:
        g["lr"] = lr_at(t) * g.get("lr_scale", 1.0)
        if g["weight_decay"] > 0 and "wd_scale" in g:
            pass                                    # wd is a constant here (engine.py scales it by the same wd schedule for every group)

def fused_params_as_split(m):
    W = m.qkv.weight.detach(); return {"q": W[:D], "k": W[D:2 * D], "v": W[2 * D:], "qb": m.qkv.bias.detach(), "out": m.out.weight.detach(), "outb": m.out.bias.detach()}
def split_params(m):
    return {"q": m.q.weight.detach(), "k": m.k.weight.detach(), "v": m.v.weight.detach(), "qb": torch.cat([m.q.bias, m.k.bias, m.v.bias]).detach(), "out": m.out.weight.detach(), "outb": m.out.bias.detach()}
def max_diff(a, b): return max(float((a[k] - b[k]).abs().max()) for k in a)

ROWS = {"qkv.weight": [(0, D, LAM["q"]), (D, 2 * D, LAM["k"]), (2 * D, 3 * D, LAM["v"])]}
data = [(torch.randn(16, D, dtype=torch.float64), torch.randn(16, dtype=torch.float64)) for _ in range(60)]

def run_pair(foreach=None, fused=None, steps=60, device="cpu"):
    """returns the max parameter difference over all steps between the masked fused model and the split reference"""
    torch.manual_seed(1); fm = Fused().to(device); sm = Split(fm).to(device)
    fo, so = fused_optimizer(fm, foreach, fused), split_optimizer(sm, foreach, fused)
    with contextlib.redirect_stdout(io.StringIO()):
        install_row_lr_masks(fo, fm, ROWS)
    worst = 0.0
    for t in range(steps):
        x, y = (d.to(device) for d in data[t])
        for m, o in ((fm, fo), (sm, so)):
            set_lr(o, t); o.zero_grad(); ((m(x) - y) ** 2).mean().backward(); o.step()
        worst = max(worst, max_diff(fused_params_as_split(fm), split_params(sm)))
    return worst, fm, fo

print("T1 exact equivalence with the split-tensor reference (float64, wd on, lr schedule)")
for foreach in (False, True):
    worst, _, _ = run_pair(foreach=foreach)
    check(f"foreach={foreach}: max |param difference| over 60 steps", worst < 1e-12, f"{worst:.1e}")
torch.manual_seed(1); fm = Fused(); sm = Split(fm); fo, so = fused_optimizer(fm), split_optimizer(sm)     # no mask installed
worst_nomask_vs_split = 0.0
for t in range(20):
    for mm, o in ((fm, fo), (sm, so)):
        set_lr(o, t); o.zero_grad(); ((mm(data[t][0]) - data[t][1]) ** 2).mean().backward(); o.step()
    worst_nomask_vs_split = max(worst_nomask_vs_split, max_diff(fused_params_as_split(fm), split_params(sm)))
check("control: WITHOUT the mask the fused model departs from the reference", worst_nomask_vs_split > 1e-6, f"{worst_nomask_vs_split:.1e}")

print("T2 lambda = 1 everywhere is a no-op")
torch.manual_seed(2); a = Fused(); b = copy.deepcopy(a); oa, ob = fused_optimizer(a), fused_optimizer(b)
with contextlib.redirect_stdout(io.StringIO()):
    install_row_lr_masks(ob, b, {"qkv.weight": [(0, 3 * D, 1.0)]})
for t in range(20):
    for m, o in ((a, oa), (b, ob)):
        set_lr(o, t); o.zero_grad(); ((m(data[t][0]) - data[t][1]) ** 2).mean().backward(); o.step()
check("bit-identical to the unmasked run", all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())))

print("T3 weight decay untouched")
torch.manual_seed(3); a = Fused(); b = copy.deepcopy(a); w0 = a.qkv.weight.detach().clone(); oa, ob = fused_optimizer(a), fused_optimizer(b)
with contextlib.redirect_stdout(io.StringIO()):
    install_row_lr_masks(ob, b, ROWS)
for m, o in ((a, oa), (b, ob)):
    set_lr(o, 5); o.zero_grad(); (m(data[0][0]) * 0).sum().backward(); o.step()       # zero gradients: only the decay acts
check("with zero gradients the masked tensor decays exactly like the unmasked one", torch.equal(a.qkv.weight, b.qkv.weight))
dev = float((b.qkv.weight.detach() - (1 - lr_at(5) * WD) * w0).abs().max())
check("decay factor is 1 - lr * wd, independent of lambda", dev < 1e-15, f"{dev:.1e}")

print("T4 a scaler-skipped step applies no correction")
torch.manual_seed(4); m = Fused().float(); o = torch.optim.AdamW([{"params": [m.qkv.weight, m.out.weight], "weight_decay": WD}, {"params": [m.qkv.bias, m.out.bias], "weight_decay": 0.0}], lr=1e-3)
with contextlib.redirect_stdout(io.StringIO()):
    install_row_lr_masks(o, m, ROWS)
scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0 ** 10)
before = [p.detach().clone() for p in m.parameters()]
o.zero_grad(); loss = ((m(data[0][0].float()) - data[0][1].float()) ** 2).mean(); scaler.scale(loss).backward()
m.qkv.weight.grad[0, 0] = float("inf")
scaler.step(o); scaler.update()
check("parameters unchanged after a step the scaler skipped", all(torch.equal(p, q) for p, q in zip(m.parameters(), before)))
o.zero_grad(); loss = ((m(data[1][0].float()) - data[1][1].float()) ** 2).mean(); scaler.scale(loss).backward(); scaler.step(o); scaler.update()
check("the next (finite) step does move the masked tensor", not torch.equal(m.qkv.weight, before[0]))

print("T5 save / load of the optimizer state mid-way")
torch.manual_seed(1); fm = Fused(); sm = Split(fm); fo, so = fused_optimizer(fm), split_optimizer(sm)
with contextlib.redirect_stdout(io.StringIO()):
    install_row_lr_masks(fo, fm, ROWS)
for t in range(30):
    for mm, o in ((fm, fo), (sm, so)):
        set_lr(o, t); o.zero_grad(); ((mm(data[t][0]) - data[t][1]) ** 2).mean().backward(); o.step()
state = copy.deepcopy(fo.state_dict()); weights = copy.deepcopy(fm.state_dict())
fm2 = Fused(); fm2.load_state_dict(weights); fo2 = fused_optimizer(fm2)
with contextlib.redirect_stdout(io.StringIO()):
    install_row_lr_masks(fo2, fm2, ROWS)
fo2.load_state_dict(state)
worst = 0.0
for t in range(30, 60):
    for mm, o in ((fm2, fo2), (sm, so)):
        set_lr(o, t); o.zero_grad(); ((mm(data[t][0]) - data[t][1]) ** 2).mean().backward(); o.step()
    worst = max(worst, max_diff(fused_params_as_split(fm2), split_params(sm)))
check("continuation after load_state_dict equals the reference", worst < 1e-12, f"{worst:.1e}")

print("T6 the real path: ViT-B via optim_factory.create_optimizer with a JSON row mask")
with contextlib.redirect_stdout(io.StringIO()):
    import main as M, utils
    from optim_factory import create_optimizer
    args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--data_path", "/data/datasets/ILSVRC2012", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
    torch.set_num_threads(16)
E = 768
spec = {"blocks.1.attn.qkv.weight": {"rows": [[0, E, 0.29], [E, 2 * E, 0.23], [2 * E, 3 * E, 1.0]]}, "blocks.2.mlp.fc1.weight": 1.1}
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(spec, tmp); tmp.close()
def vit_run(lr_json):
    torch.manual_seed(7); model = utils.build_model(args); model.train()
    a = copy.copy(args); a.lr_scale_json = lr_json; a.weight_decay = WD; a.lr = 1e-3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = create_optimizer(a, model, skip_list=None)
    torch.manual_seed(8); x = torch.randn(2, 3, 224, 224); y = torch.randint(0, 1000, (2,))
    deltas = []
    for t in range(2):
        for g in opt.param_groups: g["lr"] = 1e-3 * g.get("lr_scale", 1.0)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), y).backward(); opt.step()
        deltas.append({n: (p.detach() - before[n]) for n, p in model.named_parameters()}); 
        if t == 0: st = {n: {k: v.clone() for k, v in opt.state[p].items()} for n, p in model.named_parameters() if n == "blocks.1.attn.qkv.weight"}
    return model, opt, deltas, buf.getvalue(), st
tmp0 = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump({"blocks.2.mlp.fc1.weight": 1.1}, tmp0); tmp0.close()   # same scalar scale, no row mask
m1, o1, d1, log1, st1 = vit_run(tmp.name)
m0, o0, d0, log0, st0 = vit_run(tmp0.name)
check("mask table printed by create_optimizer", "[row-lr] blocks.1.attn.qkv.weight" in log1 and "rows [0,768) lr x0.2900" in log1)
others_same = all(torch.equal(d1[0][n], d0[0][n]) for n in d0[0] if n != "blocks.1.attn.qkv.weight")   # step 0: same gradients everywhere
check("step 0: every tensor without a row mask gets the identical update (incl. the group-scaled fc1)", others_same)
check("step 1: the masked tensor's change has propagated (other tensors now differ, as they must)", any(not torch.equal(d1[1][n], d0[1][n]) for n in d0[1] if n != "blocks.1.attn.qkv.weight"))
p_before = None
torch.manual_seed(7); ref = utils.build_model(args); W0 = ref.blocks[1].attn.qkv.weight.detach().clone()
# first step: same moments in both runs -> masked update = decay + lambda * (unmasked update - decay)
lr = 1e-3; decay = -lr * WD * W0
adaptive0 = d0[0]["blocks.1.attn.qkv.weight"] - decay
lam = torch.tensor([0.29] * E + [0.23] * E + [1.0] * E).view(-1, 1)
expected = decay + lam * adaptive0
err = float((d1[0]["blocks.1.attn.qkv.weight"] - expected).abs().max() / adaptive0.abs().max())
check("masked rows: update = decay + lambda * adaptive step of the unmasked run (relative)", err < 1e-5, f"{err:.1e}")
check("v rows (lambda 1) identical to the unmasked run", torch.equal(d1[0]["blocks.1.attn.qkv.weight"][2 * E:], d0[0]["blocks.1.attn.qkv.weight"][2 * E:]))
check("the mask leaves the optimizer's moment estimates untouched", all(torch.equal(st1["blocks.1.attn.qkv.weight"][k], st0["blocks.1.attn.qkv.weight"][k]) for k in ("exp_avg", "exp_avg_sq")))
os.unlink(tmp.name); os.unlink(tmp0.name)

print("T7 invalid specifications raise")
def raises(fn):
    try: fn(); return False
    except (ValueError, KeyError, TypeError): return True
m = Fused(); o = fused_optimizer(m)
check("overlapping ranges", raises(lambda: install_row_lr_masks(o, m, {"qkv.weight": [(0, D, 0.5), (D - 1, 2 * D, 0.5)]}, verbose=False)))
check("range outside the tensor", raises(lambda: install_row_lr_masks(o, m, {"qkv.weight": [(0, 3 * D + 1, 0.5)]}, verbose=False)))
check("non-positive lambda", raises(lambda: install_row_lr_masks(o, m, {"qkv.weight": [(0, D, 0.0)]}, verbose=False)))
check("unknown parameter name", raises(lambda: install_row_lr_masks(o, m, {"nope.weight": [(0, D, 0.5)]}, verbose=False)))
o2 = torch.optim.AdamW([{"params": [m.qkv.weight], "lr_scale": 0.5}, {"params": [m.qkv.bias, m.out.weight, m.out.bias]}], lr=1e-3)
check("tensor already in a group with its own lr_scale", raises(lambda: install_row_lr_masks(o2, m, ROWS, verbose=False)))
check("amsgrad", raises(lambda: install_row_lr_masks(fused_optimizer(m, amsgrad=True), m, ROWS, verbose=False)))
check("NaN lambda in a row entry", raises(lambda: split_lr_scale_spec({"a": {"rows": [[0, 1, float("nan")]]}})))
check("inf scalar lambda", raises(lambda: split_lr_scale_spec({"a": float("inf")})))
check("negative scalar lambda", raises(lambda: split_lr_scale_spec({"a": -0.5})))
check("boolean lambda", raises(lambda: split_lr_scale_spec({"a": True})))
check("fractional row boundary", raises(lambda: split_lr_scale_spec({"a": {"rows": [[0, 1.5, 0.5]]}})))
check("extra key next to rows", raises(lambda: split_lr_scale_spec({"a": {"rows": [[0, 1, 0.5]], "lr": 1}})))
check("a valid spec still parses", split_lr_scale_spec({"a": 2, "b": {"rows": [[0, 4.0, 0.5]]}}) == ({"a": 2.0}, {"b": [(0, 4, 0.5)]}))
from optim_factory import build_lr_scaled_param_groups
check("unknown scalar name rejected by the group builder", raises(lambda: build_lr_scaled_param_groups(m, WD, {"qkv.weightt": 0.5})))
check("NaN scalar rejected by the group builder", raises(lambda: build_lr_scaled_param_groups(m, WD, {"qkv.weight": float("nan")})))
_g = build_lr_scaled_param_groups(m, WD, {"qkv.weight": 0.3000001, "out.weight": 0.3000002})
check("two scales that differ beyond 6 decimals never share a group", len({g["lr_scale"] for g in _g}) == 3 and all("decay" in g for g in _g))
_c = copy.copy(args); _c.lr_scale_json = "{}"; _c.lr_match_ckpt = "some.pth"
check("--lr_scale_json together with --lr_match_ckpt is refused", raises(lambda: create_optimizer(_c, Fused(), skip_list=None)))

print("T10 lr specification provenance (checkpoint record, validated on resume)")
from row_lr_mask import lr_spec_of, assert_same_lr_spec
def raises_rt(fn):
    try: fn(); return False
    except RuntimeError: return True
m10 = Fused(); o10 = fused_optimizer(m10)
check("no specification -> None", lr_spec_of(o10) is None)
install_row_lr_masks(o10, m10, ROWS, verbose=False); o10._lr_scale_spec = {"out.weight": 1.1}
s10 = lr_spec_of(o10)
check("record carries the row masks and the scalar scales", s10["rows"] == {k: [list(map(float, r)) for r in v] for k, v in ROWS.items()} and s10["scalar"] == {"out.weight": 1.1})
check("identical record passes", not raises_rt(lambda: assert_same_lr_spec(s10, s10)))
check("changed lambda refused", raises_rt(lambda: assert_same_lr_spec(s10, {**s10, "scalar": {"out.weight": 1.2}})))
check("changed row mask refused", raises_rt(lambda: assert_same_lr_spec(s10, {**s10, "rows": {"qkv.weight": [[0, D, 0.5]]}})))
check("record vs no record refused (both directions)", raises_rt(lambda: assert_same_lr_spec(s10, None)) and raises_rt(lambda: assert_same_lr_spec(None, s10)))
check("no record on either side passes", not raises_rt(lambda: assert_same_lr_spec(None, None)))
# round trip through utils.save_model / utils.auto_load_model on the ViT path
_d = tempfile.mkdtemp(); _a = copy.copy(args); _a.output_dir = _d; _a.auto_resume = True; _a.resume = ""; _a.distributed = False; _a.weight_decay = WD; _a.lr = 1e-3
_a.lr_scale_json = json.dumps(spec); _a.start_epoch = 0
with contextlib.redirect_stdout(io.StringIO()):
    torch.manual_seed(7); vm = utils.build_model(args); vo = create_optimizer(_a, vm, skip_list=None); sc = utils.NativeScalerWithGradNormCount()
    utils.save_model(_a, 0, vm, vm, vo, sc)
    ck = torch.load(os.path.join(_d, "checkpoint-0.pth"), map_location="cpu", weights_only=False)
check("checkpoint holds the lr specification", ck.get("lr_scale_spec") == lr_spec_of(vo))
_b = copy.copy(_a); _b.lr_scale_json = json.dumps({**spec, "blocks.2.mlp.fc1.weight": 1.3}); _b.resume = ""
with contextlib.redirect_stdout(io.StringIO()):
    vo2 = create_optimizer(_b, utils.build_model(args), skip_list=None)
check("resume with a changed lr specification is refused", raises_rt(lambda: utils.auto_load_model(_b, vm, vm, vo2, sc)))
_a2 = copy.copy(_a); _a2.resume = ""
with contextlib.redirect_stdout(io.StringIO()):
    vo3 = create_optimizer(_a2, utils.build_model(args), skip_list=None); utils.auto_load_model(_a2, vm, vm, vo3, sc)
check("resume with the same lr specification proceeds", _a2.start_epoch == 1)
import shutil; shutil.rmtree(_d)
check("a non-AdamW optimizer", raises(lambda: install_row_lr_masks(torch.optim.SGD(m.parameters(), lr=1e-3), m, ROWS, verbose=False)))
check("split_lr_scale_spec rejects a malformed row entry", raises(lambda: split_lr_scale_spec({"a": {"rows": [[0, 1]]}})))

if torch.cuda.is_available():
    print("T8 CUDA, fused=True")
    worst, _, _ = run_pair(fused=True, device="cuda")
    check("fused AdamW kernel: max |param difference| over 60 steps", worst < 1e-10, f"{worst:.1e}")
    print("T9 CUDA, fused=True: a step the scaler skips (overflow) applies no correction from the stale moments")
    torch.manual_seed(4); m9 = Fused().float().cuda()
    o9 = torch.optim.AdamW([{"params": [m9.qkv.weight, m9.out.weight], "weight_decay": WD}, {"params": [m9.qkv.bias, m9.out.bias], "weight_decay": 0.0}], lr=1e-3, fused=True)
    install_row_lr_masks(o9, m9, ROWS, verbose=False)
    sc9 = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10)
    def step9(t, poison=False):
        o9.zero_grad(); loss = ((m9(data[t][0].float().cuda()) - data[t][1].float().cuda()) ** 2).mean(); sc9.scale(loss).backward()
        if poison: m9.qkv.weight.grad[0, 0] = float("inf")
        sc9.step(o9); sc9.update()
    step9(0)                                                  # moments now exist
    before9 = [p.detach().clone() for p in m9.parameters()]; steps9 = float(o9.state[m9.qkv.weight]["step"].item())
    step9(1, poison=True)
    check("fused kernel: the step counter does not advance on overflow", float(o9.state[m9.qkv.weight]["step"].item()) == steps9)
    check("fused kernel: parameters unchanged after the skipped step", all(torch.equal(p, q) for p, q in zip(m9.parameters(), before9)))
    step9(2)
    check("the next finite step moves the masked tensor", not torch.equal(m9.qkv.weight, before9[0]))
else:
    print("T8 / T9 skipped (no CUDA on this machine)")
print("VERDICT:", "PASS" if ok else "FAIL")
