"""Make a run directory safe to resume after its job was cancelled or preempted at an arbitrary moment.
(1) The newest full checkpoint (checkpoint-<E>.pth) must load and carry model, optimizer and its own epoch; one that was cut off
    by the kill is moved aside (.corrupt) and the next older one is checked (main.py keeps the last three).
(2) The per-epoch model file of that epoch (checkpoint-<E>-model.pth, half precision, written right after the full checkpoint) is
    re-created from the full checkpoint when the retention rule says it should exist and it is missing or does not load.
Prints the epoch the run will resume after. usage: python plots/verify/verify_resume_checkpoint.py <seed dir> [--dense_until 60] [--every 10] [--epochs 300]"""
import argparse, os, re, sys, torch
p = argparse.ArgumentParser(); p.add_argument("dir"); p.add_argument("--dense_until", type=int, default=60); p.add_argument("--every", type=int, default=10); p.add_argument("--epochs", type=int, default=300)
a = p.parse_args()
for _ in range(3):
    eps = sorted(int(m.group(1)) for f in os.listdir(a.dir) if (m := re.match(r"^checkpoint-(\d+)\.pth$", f)))
    if not eps: sys.exit("no full checkpoint in " + a.dir)
    e = eps[-1]; path = f"{a.dir}/checkpoint-{e}.pth"
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False); assert {"model", "optimizer", "epoch"} <= set(ck) and ck["epoch"] == e, "keys / epoch"
        break
    except Exception as err:
        print(f"[resume-check] {path} does not load ({type(err).__name__}: {err}); moved aside", flush=True); os.replace(path, path + ".corrupt")
else: sys.exit("three unreadable checkpoints in a row")
should_exist = e < a.dense_until or (e + 1) % a.every == 0 or e + 1 == a.epochs; mp = f"{a.dir}/checkpoint-{e}-model.pth"; ok = False
if os.path.exists(mp):
    try: ok = set(torch.load(mp, map_location="cpu", weights_only=True)) == set(ck["model"])
    except Exception: ok = False
if (should_exist and not ok) or (os.path.exists(mp) and not ok):
    torch.save({k: v.half() for k, v in ck["model"].items()}, mp + ".tmp"); os.replace(mp + ".tmp", mp); print(f"[resume-check] re-created {mp} from the full checkpoint", flush=True)
print(f"[resume-check] resume after epoch {e}: checkpoint loads ({len(ck['model'])} tensors), per-epoch model file {'present' if (ok or should_exist) else 'not due at this epoch'}", flush=True)
