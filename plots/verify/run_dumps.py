"""Run every dump_init command from dump_cmds.txt, 8 at a time (one per GPU)."""
import subprocess, os, sys, time, shlex
S = '/home/schrodi/Procedural/results/init_dumps'
ROOT = "/home/schrodi/Procedural"; OUT = f"{ROOT}/results/init_dumps"
NGPU = int(os.environ.get("NGPU", "8"))
jobs = [l.rstrip("\n").split("\t") for l in open(f"{S}/dump_cmds.txt") if l.strip()]
todo = [(n, a) for n, _, a in jobs if not os.path.exists(f"{OUT}/{n}.pth")]
print(f"{len(todo)} dumps to run ({len(jobs)-len(todo)} already present)", flush=True)
running = {}   # slot -> (name, Popen, t0)
i = 0; t_start = time.time()
while todo or running:
    for slot in range(NGPU):
        if slot in running or not todo: continue
        name, args = todo.pop(0); i += 1
        port = 21000 + i
        cmd = (f"CUDA_VISIBLE_DEVICES={slot} torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:{port} "
               f"--nproc_per_node=1 plots/dump_init.py --dump_to {OUT}/{name}.pth --dummy_wandb "
               f"--output_dir {OUT}/tmp_{name} {args}")
        log = open(f"{OUT}/{name}.log", "w"); log.write(cmd + "\n\n"); log.flush()
        p = subprocess.Popen(cmd, shell=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        running[slot] = (name, p, time.time(), log)
        print(f"[{time.time()-t_start:6.0f}s] start {name} on gpu{slot}", flush=True)
    time.sleep(5)
    for slot, (name, p, t0, log) in list(running.items()):
        rc = p.poll()
        if rc is None: continue
        log.close(); del running[slot]
        ok = os.path.exists(f"{OUT}/{name}.pth")
        print(f"[{time.time()-t_start:6.0f}s] done  {name} rc={rc} {'OK' if ok else 'NO DUMP'} ({time.time()-t0:.0f}s)", flush=True)
print("ALL DONE", flush=True)
