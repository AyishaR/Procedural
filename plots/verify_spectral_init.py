#!/usr/bin/env python3
"""Verify the "spectral" custom init BEFORE spending GPU-days on it -- and place the two new
arms against every arm that already claims to be "proc's values, arrangement destroyed".

Six models, blocks 0-8, at init, no training. Existing arms are built by
`measure_init_rho_arms.build_arm`, i.e. exactly as they are trained; the two new ones go
through `main.spectral_reinit`, i.e. the code the run scripts will call.

  r          random init                                   arm `r`          78.08 measured
  ftbqm1dv   proc's sorted values rank-mapped onto random  arm `ftbqm1dv`   +1.40 CONTAMINATED (1-D)
  ftb4e3     proc, entrywise permuted per slice            arm `ftb4e3fix`  78.58 PREDICTED
  ftb3i      proc intact                                   arm `ftb3i`      79.99 measured
  sb         proc, values=mp    basis=keep     NEW         arm `ftbsb`      proc's directions, random spectrum
  sv         proc, values=keep  basis=random   NEW         arm `ftbsv`      proc's spectrum, random directions

`ftbqm1dv` matters here because docs 0.5 calls it and `ftb4e3` "the same construction ...
no init feature can separate them", and treats their 0.68 gap as unexplained noise. Both are
random arrangements of proc's values, so under the rank hypothesis both belong in the SAME
low cell of the 2x2, and the gap is two different DDP bugs rather than noise.

Run:  python plots/verify_spectral_init.py --fig      (CPU, ~4 min)
"""
import argparse, contextlib, io, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_init_rho_arms as M          # noqa: E402
from main import spectral_reinit           # noqa: E402

BLOCKS = list(range(9))
ARMS = ["r", "ftbqm1dv", "ftb4e3", "ftb3i", "sb", "sv"]

LABEL = {"r":        ("random init",                      78.08, "measured"),
         "ftbqm1dv": ("proc's values, rank-mapped",       None,  "contaminated"),
         "ftb4e3":   ("proc's values, shuffled",          78.58, "predicted"),
         "ftb3i":    ("proc intact",                      79.99, "measured"),
         "sb":       ("proc dirs, random spectrum",       None,  ""),
         "sv":       ("proc spectrum, random dirs",       None,  "")}
COL = {"r": "#8a8a8a", "ftbqm1dv": "#b05fc0", "ftb4e3": "#c94f4f",
       "ftb3i": "#1b6ca8", "sb": "#e08a1e", "sv": "#3f8f5b"}


def get_args():
    ap = argparse.ArgumentParser(parents=[__import__("main").get_args_parser()],
                                 add_help=False, conflict_handler="resolve")
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--out", default="plots/cache/spectral_verify.json")
    args = ap.parse_args()
    args.nb_classes = 1000
    args.model = "vit_base"
    args.initialize = args.initialize or "results/pr_vitb_n/pr_6066174_final.pth"
    args.skip_norm = True
    for k in ["skip_attn_segments", "weight_shuffle", "target_model_weight_shuffle",
              "init_method_copied_blocks", "attention_residual_scaling",
              "attention_out_scaling", "learning_rate_scaling_params"]:
        setattr(args, k, {})
    for k in ["random_blocks", "clip_outlier_blocks", "delete_blocks",
              "init_method_scaled_blocks", "freeze_blocks", "hold_back_blocks"]:
        setattr(args, k, [])
    for k in ["skip_load_blocks", "skip_load_block_attributes", "freeze_block_attributes"]:
        v = getattr(args, k, "")
        setattr(args, k, [x for x in v.split(",")] if isinstance(v, str) and v else [])
    args.distributed = False
    args.gpu = args.rank = 0
    args.world_size = 1
    return args


def build_all(args):
    tgt = M._target_params(args)
    out = {}
    for arm in ARMS:
        torch.manual_seed(0)
        if arm in ("sb", "sv"):
            mw = M.build_arm("ftb3i", args, tgt)       # same base as ftb3i: proc 0-8, random 9-11
            spectral_reinit(mw, BLOCKS,
                            values="mp" if arm == "sb" else "keep",
                            basis="keep" if arm == "sb" else "random",
                            seed=0, tag=f"[{arm}]")
        else:
            mw = M.build_arm(arm, args, tgt)
        out[arm] = mw
    return out


def tensors(model, b):
    p = dict(model.named_parameters())
    qkv = p[f"blocks.{b}.attn.qkv.weight"].data
    e = qkv.shape[0] // 3
    return {"qkv[q]": qkv[:e], "qkv[k]": qkv[e:2 * e], "qkv[v]": qkv[2 * e:],
            "proj": p[f"blocks.{b}.attn.proj.weight"].data,
            "fc1": p[f"blocks.{b}.mlp.fc1.weight"].data,
            "fc2": p[f"blocks.{b}.mlp.fc2.weight"].data}


def spectrum(W):
    s = torch.linalg.svdvals(W.double())
    f2 = (s ** 2).sum()
    q = (s ** 2) / f2
    return {"s": s.numpy(), "fro": float(f2.sqrt()),
            "stable_rank": float(f2 / s[0] ** 2),
            "eff_rank": float(torch.exp(-(q * (q + 1e-300).log()).sum())),
            "sigma_max": float(s[0]),
            "top5_energy": float((s[:5] ** 2).sum() / f2)}


def value_write(model, blocks):
    """docs 0's scalar: gamma_norm1 * ||W_v|| * ||W_proj|| / d, mean over blocks."""
    p = dict(model.named_parameters())
    out = []
    for b in blocks:
        qkv = p[f"blocks.{b}.attn.qkv.weight"].data
        e, d = qkv.shape[0] // 3, qkv.shape[1]
        out.append(float(p[f"blocks.{b}.norm1.weight"].data.abs().mean()
                         * qkv[2 * e:].norm()
                         * p[f"blocks.{b}.attn.proj.weight"].data.norm() / d))
    return float(np.mean(out))


def main():
    args = get_args()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        V = build_all(args)
    print(f"(suppressed {len(buf.getvalue().splitlines())} lines of init logging)")

    stats = {n: {b: {t: spectrum(W) for t, W in tensors(m, b).items()} for b in BLOCKS}
             for n, m in V.items()}
    tn_all = list(stats["ftb3i"][0].keys())

    def agg(n, t, k):
        return float(np.mean([stats[n][b][t][k] for b in BLOCKS]))

    for key, title, w in [("stable_rank", "STABLE RANK  ||W||_F^2/sigma_1^2  (194 / 343 = unstructured)", 10),
                          ("sigma_max", "SIGMA_MAX", 10),
                          ("fro", "||W||_F", 10)]:
        print(f"\n=== {title} ===")
        print(f"{'tensor':9s}" + "".join(f"{n:>{w}s}" for n in ARMS))
        for t in tn_all:
            print(f"{t:9s}" + "".join(f"{agg(n, t, key):>{w}.1f}" for n in ARMS))

    print("\n=== value_write (docs 0's scalar -- blind to everything above) ===")
    vw = {n: value_write(V[n], BLOCKS) for n in ARMS}
    print("".join(f"{n:>10s}" for n in ARMS))
    print("".join(f"{vw[n]:>10.4f}" for n in ARMS))

    print("\n=== CHECKS ===")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    # 1. the new arms are exactly norm-preserving, tensor by tensor
    for n in ["sb", "sv"]:
        rel = max(abs(stats[n][b][t]["fro"] - stats["ftb3i"][b][t]["fro"]) / stats["ftb3i"][b][t]["fro"]
                  for b in BLOCKS for t in tn_all)
        check(f"{n}: ||W||_F identical to ftb3i on all 54 matrices", rel < 1e-5, f"max rel dev {rel:.2e}")

    # 2. no norm-based quantity in docs 0 can separate any of them
    for n in ["sb", "sv", "ftb4e3", "ftbqm1dv"]:
        check(f"{n}: same value_write as ftb3i", abs(vw[n] - vw["ftb3i"]) / vw["ftb3i"] < 1e-4,
              f"{vw[n]:.4f} vs {vw['ftb3i']:.4f}")

    # 3. the spectral surgery does what it says
    d = max(abs(np.sort(stats["sv"][b][t]["s"])[::-1] - np.sort(stats["ftb3i"][b][t]["s"])[::-1]).max()
            / stats["ftb3i"][b][t]["sigma_max"] for b in BLOCKS for t in tn_all)
    check("sv: reproduces proc's singular VALUES exactly", d < 1e-8, f"max rel dev {d:.2e}")
    check("sv: keeps proc's stable rank",
          all(abs(agg("sv", t, "stable_rank") - agg("ftb3i", t, "stable_rank")) < 1e-6 for t in tn_all))
    for n in ["sb", "ftb4e3", "ftbqm1dv"]:
        worst = max(abs(agg(n, t, "stable_rank") - agg("r", t, "stable_rank")) / agg("r", t, "stable_rank")
                    for t in tn_all)
        check(f"{n}: stable rank sits on random init's", worst < 0.15, f"max rel dev {worst:.2f}")
    check("sb and sv are different models",
          abs(agg("sb", "fc1", "stable_rank") - agg("sv", "fc1", "stable_rank")) > 10)

    # 4. the intervention is confined to blocks 0-8
    check("blocks 9-11 untouched by the spectral init",
          all(abs(spectrum(tensors(V["sv"], b)[t])["fro"] - spectrum(tensors(V["ftb3i"], b)[t])["fro"]) < 1e-6
              and abs(spectrum(tensors(V["sb"], b)[t])["fro"] - spectrum(tensors(V["ftb3i"], b)[t])["fro"]) < 1e-6
              for b in [9, 10, 11] for t in tn_all))

    # 5. ftb4e3's shuffle pools q and k -- a real property of that arm, worth recording
    dq = abs(agg("ftb4e3", "qkv[q]", "fro") - agg("ftb3i", "qkv[q]", "fro")) / agg("ftb3i", "qkv[q]", "fro")
    print(f"  [note] ftb4e3 shuffles rows 0:2e as ONE pool, so it does NOT preserve ||W_q||/||W_k|| "
          f"separately ({agg('ftb4e3','qkv[q]','fro'):.2f}/{agg('ftb4e3','qkv[k]','fro'):.2f} vs proc's "
          f"{agg('ftb3i','qkv[q]','fro'):.2f}/{agg('ftb3i','qkv[k]','fro'):.2f}, {dq:.1%}). "
          f"||W_v|| and ||W_proj|| ARE preserved, so value_write is untouched.")

    print(f"\n{'ALL CHECKS PASSED' if ok else '*** SOME CHECKS FAILED ***'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({n: {str(b): {t: {k: v for k, v in s.items() if k != "s"} for t, s in bb.items()}
                   for b, bb in stats[n].items()} for n in ARMS}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    if args.fig:
        figure(stats, tn_all, vw)
    return 0 if ok else 1


def figure(stats, tn_all, vw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def agg(n, t, k):
        return float(np.mean([stats[n][b][t][k] for b in BLOCKS]))

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.40, wspace=0.26,
                          left=0.055, right=0.985, top=0.855, bottom=0.075)

    # Four arms have literally the same spectrum and two more share proc's -- that coincidence
    # IS the result, so draw it as two bands with decreasing linewidth rather than 6 legend rows.
    STYLE = {"ftb3i": ("-", 3.4, 0.9), "sv": ((0, (4, 3)), 1.7, 1.0),
             "r": ("-", 3.4, 0.55), "ftbqm1dv": ((0, (5, 3)), 2.3, 0.85),
             "ftb4e3": ((0, (1, 2)), 1.9, 0.95), "sb": ((0, (6, 2, 1, 2)), 1.1, 1.0)}
    for j, t in enumerate(["qkv[v]", "proj", "fc1"]):
        ax = fig.add_subplot(gs[0, j])
        for n in ["r", "ftbqm1dv", "ftb4e3", "sb", "ftb3i", "sv"]:
            ls, lw, al = STYLE[n]
            s_ = stats[n][4][t]["s"]
            ax.semilogy(np.arange(1, len(s_) + 1), s_ / s_[0], color=COL[n], lw=lw, ls=ls,
                        alpha=al, label=LABEL[n][0] if j == 0 else None)
        ax.set_title(f"block 4 · {t}", fontsize=11)
        ax.set_xlabel("singular value index")
        if j == 0:
            ax.set_ylabel(r"$\sigma_i/\sigma_1$")
            ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
        k = len(stats["ftb3i"][4][t]["s"])
        ax.set_xlim(0, k); ax.set_ylim(1e-3, 1.3); ax.grid(alpha=0.25)
        if j == 2:
            ax.annotate("4 arms, one curve\n(random spectrum)", xy=(0.62 * k, 0.42), fontsize=8.5,
                        color="#7a4a1a", ha="center",
                        bbox=dict(fc="#fdf3e4", ec="#e08a1e", lw=0.9, boxstyle="round,pad=0.3"))
            ax.annotate("2 arms, one curve\n(proc's spectrum)", xy=(0.62 * k, 0.008), fontsize=8.5,
                        color="#14405f", ha="center",
                        bbox=dict(fc="#e8f1f7", ec="#1b6ca8", lw=0.9, boxstyle="round,pad=0.3"))

    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(tn_all)); w = 0.14
    for i, n in enumerate(ARMS):
        ax.bar(x + (i - 2.5) * w, [agg(n, t, "stable_rank") for t in tn_all], w,
               color=COL[n], label=LABEL[n][0])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tn_all)
    ax.set_ylabel(r"stable rank  $\|W\|_F^2/\sigma_1^2$")
    ax.set_title(f"Blocks 0-8: every arm shares proc's value_write ({vw['ftb3i']:.3f}); "
                 "only the rank moves", fontsize=11)
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=8.5, ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title("The 2×2 this design closes", fontsize=11.5, pad=16)
    cells = [[("proc spectrum\nproc directions", ["ftb3i"]), ("proc spectrum\nrandom directions", ["sv"])],
             [("random spectrum\nproc directions", ["sb"]), ("random spectrum\nrandom directions",
                                                             ["ftb4e3", "ftbqm1dv"])]]
    for r in range(2):
        for c in range(2):
            txt, keys = cells[r][c]
            k0 = keys[0]
            ax.add_patch(plt.Rectangle((c * .5, .55 - r * .5), .46, .46, transform=ax.transAxes,
                                       facecolor=COL[k0], alpha=.15, edgecolor=COL[k0], lw=2))
            ax.text(c * .5 + .23, .55 - r * .5 + .355, txt, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.5)
            acc = LABEL[k0][1]
            ax.text(c * .5 + .23, .55 - r * .5 + .195, f"{acc:.2f}" if acc else "?",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14 if acc else 19, fontweight="bold", color=COL[k0])
            sub = "\n".join(f"{k}" + (f"  ({LABEL[k][2]})" if LABEL[k][2] else "") for k in keys)
            ax.text(c * .5 + .23, .55 - r * .5 + .065, sub, transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.2, color="#333")

    fig.suptitle("Is the early-block benefit the SPECTRUM or the DIRECTIONS?   "
                 "ViT-B blocks 0-8 at init, no training", fontsize=14, y=0.955)
    fig.text(0.5, 0.905, "All five non-random arms carry proc's exact value_write (0.492) and its exact "
             "$\\|W_v\\|$, $\\|W_{proj}\\|$, $\\|W_{fc}\\|$.  "
             "The only arm that has ever won is the only one that is low-rank.",
             ha="center", fontsize=10, color="#444")
    os.makedirs("plots/out", exist_ok=True)
    fig.savefig("plots/out/fig14_spectra.png", dpi=170)
    print("wrote plots/out/fig14_spectra.png")


if __name__ == "__main__":
    sys.exit(main())
