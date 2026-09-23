"""Guards added after the 2026-09-19 review of the reconstruction code.
  G1  utils._bisect: threshold crossing is not target matching -- a target already passed at strength 0 returns (0, False).
  G2  utils.calibrate_joint_statistics reports "matched" per target from the achieved statistic (ViT-B block 0, random images):
      unattainable entropy / active-unit targets -> False, attainable ones -> True.
  G3  utils.ft_load_model(keep_all=True) loads the head, class token, position embedding and patch projection of a trained
      checkpoint; the fine-tuning default drops them when args.initialize_as_pr is set (the post-training analysis bug).
usage: .venv/bin/python plots/verify/test_calibration_guards.py"""
import contextlib, io, os, sys, tempfile
import torch
sys.path.insert(0, "/home/schrodi/Procedural")
with contextlib.redirect_stdout(io.StringIO()):
    import utils, main as M
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

print("G1 _bisect: threshold crossing is not target matching")
s, r = utils._bisect(lambda a: 0.2 - 0.01 * a, 0.3, 1.0, decreasing=True)
check("target already passed at strength 0 -> (0, False)", s == 0.0 and r is False, f"({s}, {r})")
s, r = utils._bisect(lambda a: 1.0 - a, 0.3, 1.0, decreasing=True, tolerance=1e-7)
check("a bracketed target is found", r and abs(s - 0.7) < 1e-5, f"({s:.6f}, {r})")
s, r = utils._bisect(lambda a: 1.0 - 0.1 * a, 0.3, 1.0, decreasing=True)
check("target beyond the bracket -> (hi, False)", s == 1.0 and r is False, f"({s}, {r})")
s, r = utils._bisect(lambda a: 0.1 * a, 0.5, 10.0, decreasing=False, tolerance=1e-7)
check("increasing statistic", r and abs(s - 5.0) < 1e-4, f"({s:.5f}, {r})")

print("G2 calibrate_joint_statistics: 'matched' follows the achieved statistic (ViT-B block 0, 8 random images)")
args = M.get_args_parser().parse_args(["--model", "vit_base", "--data_set", "IMNET", "--input_size", "224", "--nb_classes", "1000"]); args.nb_classes = 1000
torch.manual_seed(0); images = torch.randn(8, 3, 224, 224)
def calibrate(spec):
    torch.manual_seed(0); model = utils.build_model(args)
    return utils.calibrate_joint_statistics(model, spec, images, seed=0)
bad = calibrate({"qk_entropy": {"entropy": {"0": 6.0}, "renormalize": True}, "fc1_gate": {"active_units": {"0": 0.9}, "renormalize": True}, "gain_fold": "exact"})[0]
check("entropy above the random init's is unattainable: matched False, reachable False", bad["qk_entropy"]["matched"] is False and bad["qk_entropy"]["reachable"] is False, f"entropy {bad['qk_entropy']['entropy']:.3f} target 6.0")
check("active-unit fraction above the random init's is unattainable: matched False", bad["fc1_gate"]["matched"] is False, f"active {bad['fc1_gate']['active_units']:.3f} target 0.9")
good = calibrate({"qk_entropy": {"entropy": {"0": 3.0}, "renormalize": True}, "fc1_gate": {"active_units": {"0": 0.3}, "renormalize": True}, "gain_fold": "exact"})[0]
check("attainable entropy target: matched True and met within 0.02", good["qk_entropy"]["matched"] is True and abs(good["qk_entropy"]["entropy"] - 3.0) <= 0.02, f"entropy {good['qk_entropy']['entropy']:.4f}")
check("attainable active-unit target: matched True and met within 3%", good["fc1_gate"]["matched"] is True and abs(good["fc1_gate"]["active_units"] - 0.3) <= 0.009, f"active {good['fc1_gate']['active_units']:.5f}")
budget = calibrate({"fc1_gate": {"active_units": {"0": 0.05}, "renormalize": True}, "gain_fold": "exact"})[0]["fc1_gate"]
check("target below what the norm budget allows (random init, noise images): reachable False, matched False", budget["reachable"] is False and budget["matched"] is False, f"active {budget['active_units']:.3f} at the budget, target 0.05")

print("G3 ft_load_model: keep_all=True keeps the trained head and embeddings")
torch.manual_seed(0); trained = utils.build_model(args)
with torch.no_grad():
    trained.head.weight.fill_(0.5); trained.cls_token.fill_(0.25); trained.blocks[0].attn.qkv.weight.fill_(0.125)
d = tempfile.mkdtemp(); path = os.path.join(d, "pr_trained_test.pth"); torch.save({"model": trained.state_dict()}, path)
a = args; a.distributed = False
for keep in (False, True):
    torch.manual_seed(1); fresh = utils.build_model(args)
    with contextlib.redirect_stdout(io.StringIO()):
        loaded, _ = utils.ft_load_model(path, a, torch.device("cpu"), model=fresh, keep_all=keep)
    head_ok = bool((loaded.head.weight == 0.5).all()); cls_ok = bool((loaded.cls_token == 0.25).all()); blk_ok = bool((loaded.blocks[0].attn.qkv.weight == 0.125).all())
    if keep:
        check("keep_all=True: head, class token and block weights are the checkpoint's", head_ok and cls_ok and blk_ok, f"head {head_ok} cls {cls_ok} block {blk_ok}")
    else:
        check("fine-tuning default (initialize_as_pr): block loaded, head and class token NOT loaded (the documented behaviour)", blk_ok and not head_ok and not cls_ok, f"head {head_ok} cls {cls_ok} block {blk_ok}")
os.remove(path); os.rmdir(d)
print("VERDICT:", "PASS" if ok else "FAIL")
