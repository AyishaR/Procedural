"""Earlier analytic arms vs the blocks-0-7 recipe family during training: block-7 logit lens (the model's own norm + head on
block 7's output), attention entropy and attention / MLP write ratio of blocks 1-7, from the wandb cache.
Refresh running arms first:  FAST=1 ONLY=<arms> REFRESH=<arms> python plots/verify/wandb_layerwise.py
usage: python plots/verify/compare_training_traces.py"""
import json
c = json.load(open("/home/schrodi/Procedural/plots/cache/verify/wandb_layerwise.json"))
def seeds_mean(arm, e, key):
    vals = [s[str(e)][key] for s in c.get(arm, {}).values() if str(e) in s and key in s[str(e)]]
    return sum(vals) / len(vals) if vals else float("nan")
def mid(arm, e, fam):
    vals = [v for v in (seeds_mean(arm, e, f"{fam}_layer{l}") for l in range(1, 8)) if v == v]
    return sum(vals) / len(vals) if vals else float("nan")
GROUPS = [("reference", [("r", "random", 78.08), ("ftb4i_kdyck", "kdyck prefix 0-7 (n=3)", 79.89), ("ftb3i", "kdyck prefix 0-8 (n=3)", 79.99)]),
          ("earlier analytic arms, winners", [("ftbanag", "profile + permuted gains", 80.70), ("ftbanac", "ftbanap + late lever (lens ~0 is an artefact)", 80.55), ("ftbanap", "ramp, pooled q/k 1.32", 80.24),
                                              ("ftbanal", "slow steps via lr, gains 1", 79.78), ("ftbanape", "exact per block", 79.44), ("ftbanapx", "exact fold, ramp", 79.18)]),
          ("earlier analytic arms, losers", [("ftbanai", "input side reset", 78.11), ("ftbanau", "isotropic gains", 77.35), ("ftbana", "profile, gains 1", 76.61), ("ftb4o", "random 0-7 write-matched", 77.27)]),
          ("blocks-0-7 family", [("ftbanaperab7", "kdyck: scales + entropy + active-unit gate", None), ("ftbanaperab7w", "kdyck: + write-matched", None), ("ftbanaperab7i", "kdyck: v/proj/fc2 timm", None),
                                 ("ftbanapermb7", "kdyck: mean-gate control", None), ("ftbanapermb7w", "kdyck: mean-gate control, write-matched", None)]),
          ("ksd", [("ftb4i", "ksd prefix 0-7", 80.05), ("ftbanak", "ksd scale-only", 77.86), ("ftbanakw", "ksd write profile", 76.65), ("ftbanaks", "ksd sink", 79.58), ("ftbanaksg", "ksd sink + gate", 79.92),
                   ("ftbanakperab7", "NEW ksd base", None), ("ftbanakperab7w", "NEW ksd write-matched", None), ("ftbanakperab7i", "NEW ksd v/proj/fc2 timm", None)])]
EPOCHS = (4, 9, 19, 29, 39, 49, 99, 149, 289)
fmt = lambda v, d=1: f"{v:6.{d}f}" if v == v else "     -"
for fam, label, d in (("acc", "BLOCK-7 LOGIT LENS (top-1 %)", 1), ("attn_entropy", "ATTENTION ENTROPY, blocks 1-7 (nats)", 2),
                      ("attn_delta_norm_ratio", "ATTENTION WRITE RATIO, blocks 1-7", 3), ("delta_norm_ratio", "MLP WRITE RATIO, blocks 1-7", 3)):
    print(f"\n{label}\n{'arm':15s} {'final':>6s} | " + " ".join(f"ep{e:<4d}" for e in EPOCHS) + (" | peak" if fam == "acc" else ""))
    for title, arms in GROUPS:
        print(f"-- {title}")
        for arm, text, final in arms:
            if arm not in c: continue
            row = [seeds_mean(arm, e, "acc_layer7") if fam == "acc" else mid(arm, e, fam) for e in EPOCHS]
            tail = ""
            if fam == "acc":
                every = [(seeds_mean(arm, int(e), "acc_layer7"), int(e)) for e in next(iter(c[arm].values()))]
                every = [x for x in every if x[0] == x[0]]; tail = f" | {max(every)[0]:.1f} @ {max(every)[1]}  {text}" if every else f" | -  {text}"
            print(f"{arm:15s} {final if final else '   -  ':>6} | " + " ".join(fmt(v, d) for v in row) + tail)
