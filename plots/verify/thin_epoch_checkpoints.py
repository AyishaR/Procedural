"""Thin the per-epoch MODEL checkpoints (checkpoint-<E>-model.pth) of FINISHED ViT-B runs; nothing else is ever touched
(full checkpoints, checkpoint-best, logs, json stay). Default is a dry run that writes a manifest; --execute deletes exactly the
manifest of a previous dry run after re-validating every entry.

Kept per seed directory:
  general   0, 1, 2, 4, 5, 9, 10, 14, 19, 20, 29, 39, 49, 50, 59, 99, 100, 199, 200, 299, every 25th in both conventions
            (E % 25 == 0 and (E + 1) % 25 == 0), and the last epoch present (aborted runs keep their end state)
  reference arms (REFERENCE below) additionally epochs 0-59 and every (E + 1) % 10 == 0, i.e. the set the wave-1 runs keep
Never touched:
  - every run with a job in the queue (job id or exported SLURM_ID), the wave-1 / R0 ids listed in ACTIVE, and every seed directory
    with a file written in the last 3 hours;
  - the 17 seed directories whose per-epoch checkpoints a collaborator's analysis in <workspace>/kempfe/dynamics refers to (PROTECTED);
  - a seed directory with per-epoch files owned by another user (the collaborator's own runs write into this tree as well);
  - a seed directory without log.txt, or whose kept files look truncated (size off by > 2% from the directory's median).
usage: python plots/verify/thin_epoch_checkpoints.py            (dry run -> results/init_dumps/thinning_manifest.json)
       python plots/verify/thin_epoch_checkpoints.py --execute  (deletes the manifest's files; log -> results/init_dumps/thinning_deleted.log)"""
import json, os, re, subprocess, sys, time
WS = "/work/dlc2workfs3/schrodi-procedural/results/imnet_base"; OUT = "/home/schrodi/Procedural/results/init_dumps"
MANIFEST = f"{OUT}/thinning_manifest.json"; LOG = f"{OUT}/thinning_deleted.log"
PAT = re.compile(r"^/work/dlc2workfs3/schrodi-procedural/results/imnet_base/results_IMNET_BASE_(\d+)(_prefix_contaminated)?/s(\d+)/checkpoint-(\d+)-model\.pth$")
BASE = {0, 1, 2, 4, 5, 9, 10, 14, 19, 20, 29, 39, 49, 50, 59, 99, 100, 199, 200, 299} | {e for e in range(300) if e % 25 == 0 or (e + 1) % 25 == 0}
DENSE = BASE | set(range(60)) | {e for e in range(300) if (e + 1) % 10 == 0}
REFERENCE = {"29737095": "ftbanapermb7i (C kdyck)", "29736861": "ftbanakpermb7i (C ksd)", "29448854": "ftb4i kdyck prefix", "29547835": "ftb4i ksd prefix", "29592459": "ftbanap", "29626632": "ftbanak",
             "29572321": "ftbana", "29733656": "ftbanaperab7i", "29729558": "ftbanakperab7i", "29729553": "ftbanapermb7", "29737097": "ftbanakpermb7"}
ACTIVE = {"29744580", "29744581", "29745240", "29745241", "29745242", "29745243", "29745244", "29745245", "29745246", "29745247", "29745248", "29745250", "29745251", "29745252", "29745253", "29745254",
          "29745255", "29746035", "29746036", "29745319", "29745320", "29745321", "29745322", "29745324", "29745325", "29745327"}
PROTECTED = {("29448854", "1"), ("29448854", "2"), ("29498148", "0"), ("29498148", "1"), ("29498148", "2"), ("29518360", "0"), ("29518360", "1"), ("29518360", "2"), ("29523316", "0"), ("29523316", "1"),
             ("29523316", "2"), ("29538140", "0"), ("29545846", "0"), ("29547831", "0"), ("29547835", "0"), ("29626634", "0"), ("29632673", "0")}

def queued_ids():
    ids = set()
    for jid in subprocess.run(["squeue", "-u", os.environ.get("USER", "schrodi"), "-h", "-o", "%i"], capture_output=True, text=True).stdout.split():
        jid = jid.split("_")[0]; ids.add(jid)
        line = subprocess.run(["scontrol", "show", "job", jid], capture_output=True, text=True).stdout
        ids |= set(re.findall(r"SLURM_ID=(\d+)", line))
    return ids

def plan():
    now = time.time(); busy = queued_ids() | ACTIVE; manifest = []; report = []; skipped = []
    for run in sorted(os.listdir(WS)):
        g = re.match(r"^results_IMNET_BASE_(\d+)(_prefix_contaminated)?$", run)
        if not g or not os.path.isdir(f"{WS}/{run}") or os.path.islink(f"{WS}/{run}"): continue
        rid = g.group(1)
        for seed_dir in sorted(os.listdir(f"{WS}/{run}")):
            d = f"{WS}/{run}/{seed_dir}"; s = re.match(r"^s(\d+)$", seed_dir)
            if not s or not os.path.isdir(d) or os.path.islink(d): continue
            files = {}
            for f in os.scandir(d):
                m = re.match(r"^checkpoint-(\d+)-model\.pth$", f.name)
                if m and f.is_file(follow_symlinks=False): files[int(m.group(1))] = f.stat(follow_symlinks=False)
            if len(files) <= 2: continue
            newest = max(st.st_mtime for st in (e.stat(follow_symlinks=False) for e in os.scandir(d)))
            foreign = sorted({st.st_uid for st in files.values()} - {os.getuid()})
            why = (f"files owned by another user (uid {foreign})" if foreign else "job in the queue / wave-1 / R0" if rid in busy else "referenced by the collaborator's dynamics analysis" if (rid, s.group(1)) in PROTECTED else
                   "written in the last 3 hours" if now - newest < 3 * 3600 else "no log.txt" if not os.path.exists(f"{d}/log.txt") else None)
            if why: skipped.append((f"{run}/{seed_dir}", len(files), why)); continue
            keep = (DENSE if rid in REFERENCE else BASE) | {max(files)}; sizes = sorted(st.st_size for st in files.values()); med = sizes[len(sizes) // 2]
            odd = [e for e in files if e in keep and abs(files[e].st_size - med) > 0.02 * med]
            if odd: skipped.append((f"{run}/{seed_dir}", len(files), f"kept files with an unusual size at epochs {odd}")); continue
            drop = sorted(e for e in files if e not in keep)
            for e in drop: manifest.append({"path": f"{d}/checkpoint-{e}-model.pth", "size": files[e].st_size, "mtime": files[e].st_mtime})
            report.append({"dir": f"{run}/{seed_dir}", "reference": rid in REFERENCE, "present": len(files), "kept": len(files) - len(drop), "deleted": len(drop), "bytes": sum(files[e].st_size for e in drop),
                           "kept_epochs": sorted(e for e in files if e in keep), "log_has_299": '"epoch": 299' in open(f"{d}/log.txt").read()})
    return manifest, report, skipped

def validate(entry):
    p = entry["path"]; m = PAT.match(p)
    if not m or os.path.realpath(p) != p or os.path.islink(p) or not os.path.isfile(p): return "path"
    st = os.stat(p)
    if st.st_uid != os.getuid(): return "owner"
    if st.st_size != entry["size"] or abs(st.st_mtime - entry["mtime"]) > 1: return "changed since the dry run"
    rid, seed, e = m.group(1), m.group(3), int(m.group(4))
    if rid in ACTIVE or (rid, seed) in PROTECTED: return "protected"
    if e in (DENSE if rid in REFERENCE else BASE): return "epoch is in the keep set"
    return None

if "--execute" not in sys.argv:
    manifest, report, skipped = plan(); bad = [(e["path"], validate(e)) for e in manifest if validate(e)]
    assert not bad, bad[:5]
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": manifest, "report": report, "skipped": skipped}, open(MANIFEST, "w"))
    tot = sum(e["size"] for e in manifest)
    print(f"DRY RUN: {len(manifest)} files, {tot / 1e12:.3f} TB in {len(report)} seed directories ({sum(r['reference'] for r in report)} of them reference arms with the dense keep set)")
    print(f"   kept per directory: {sorted(set(r['kept'] for r in report))} files; directories without epoch 299 in log.txt (aborted runs, last epoch kept): {[r['dir'] for r in report if not r['log_has_299']]}")
    print(f"   skipped ({len(skipped)}):"); [print(f"      {d:52s} {n:4d} files  {why}") for d, n, why in skipped]
    print(f"   manifest -> {MANIFEST}")
else:
    M = json.load(open(MANIFEST)); busy = queued_ids() | ACTIVE; done = 0; freed = 0; refused = []
    with open(LOG, "a") as log:
        log.write(f"# thinning started {time.strftime('%Y-%m-%d %H:%M:%S')}, manifest of {M['created']}, {len(M['files'])} files\n")
        for entry in M["files"]:
            why = validate(entry) or ("job in the queue" if PAT.match(entry["path"]).group(1) in busy else None)
            if why: refused.append((entry["path"], why)); continue
            os.remove(entry["path"]); done += 1; freed += entry["size"]; log.write(f"{entry['path']}\t{entry['size']}\n")
            if done % 2000 == 0: print(f"   {done} files, {freed / 1e12:.3f} TB", flush=True)
        log.write(f"# finished {time.strftime('%Y-%m-%d %H:%M:%S')}: {done} files, {freed} bytes; refused {len(refused)}\n")
    print(f"DELETED {done} files, {freed / 1e12:.3f} TB; refused {len(refused)}: {refused[:5]}")
