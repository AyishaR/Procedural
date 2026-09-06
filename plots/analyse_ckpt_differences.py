#!/usr/bin/env python3
"""What separates the +2 arms from the rest?  A wide battery of init-time measurements.

Motivation
----------
Two arms reach ~+2 over random init (`ftb4e3` +2.08, `ftb3i` +1.91); six land in
[+0.05, +0.78]; one lands BELOW random (`ftb4o`, -0.81).  Every published statistic so far
either fails to order them or fails specifically on `ftb4o`:

  * rho at init            -- r = -0.27 (p = 0.52) once ftb4o and ftbnorm are included
  * summed acc_layer probe -- r = -0.878, but ftb4o sits BETWEEN the winners and random,
                              i.e. on the wrong side for a monotone reading

So the bar a candidate mechanism has to clear here is deliberately set higher than
"correlates": it must also put `ftb4o` on the FAR side of random init from the winners,
because ftb4o is the one arm that made things worse.  That is the `4o_ok` column below.

What is measured
----------------
Everything is measured at INIT on the reconstructed arm (via measure_init_rho_arms.build_arm,
which reproduces what is actually trained -- see that file's docstring for why main.py's own
analysis flag cannot be used).  Three families:

  W.*  weight space.  Norms/std/kurtosis of each tensor, the q/k/v slices of the fused qkv
       SEPARATELY, per-head dispersion, stable rank, and -- new here -- every quantity
       COMPOSED with the preceding LayerNorm gain, since attention never sees W_q but
       W_q diag(gamma_1), and proc's gains (0.31-0.44) partly cancel its 4x larger Q/K.
  F.*  forward pass on real val images.  rho (as before) plus the attention distribution
       itself: logit spread, entropy, max prob, CLS mass, spatial attention distance; and
       the representation: token cosine similarity, participation-ratio effective rank,
       channel kurtosis, feature norm.
  G.*  gradients from one backward pass at init.  Per-block gradient norm and, more to the
       point, ||g||/||W|| -- the quantity that sets how far one AdamW step actually moves a
       tensor, and the standing explanation for the slow starts.

  handoff.*  the same representation statistics measured specifically at the INPUT TO BLOCK 9.
       Blocks 9-11 are byte-identical random weights in every arm on this matrix, so the only
       thing an arm can do is hand them a different representation.  These features isolate
       that channel.

Usage (GPU + ImageNet val):
    python plots/analyse_ckpt_differences.py --model vit_base --data_set IMNET \
        --data_path /data/datasets/ILSVRC2012 \
        --initialize results/pr_vitb_n/pr_6066174_final.pth \
        --procedural_data kdyck --procedural_order standard --skip_norm true \
        --out plots/cache/ckpt_diff.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_init_rho_arms as M          # noqa: E402  (build_arm / build_ftb4o / measure)

BLOCKS = list(range(9))        # the manipulated blocks in every fig5 arm
HANDOFF = 9                    # first block that is random-and-identical in every arm

# Arms to measure.  Those with a training result are scored; the rest are predictions.
ARMS = ["r", "ftb4o", "ftbnorm", "ftbqm", "ftbqmbias", "ftbqm1dpar", "ftbqm1d",
        "ftbqmln", "ftb4e3", "ftb3i", "ftbqm1dv", "ftbqm1dqk", "ftbqm1dvo", "p"]

# (slurm_id, seed) -> last-epoch test_acc1.  Mirrors plots/make_figures.py:ARMS.
RUNS = {
    "r":          [("29384839", "s0"), ("29384839", "s1"), ("29384839", "s2")],
    "p":          [("29377576", "s0"), ("29377576", "s1"), ("29377576", "s2")],
    "ftbqm":      [("29498141", "s0"), ("29498141", "s1"), ("29498141", "s2")],
    "ftbqm1dpar": [("29502416", "s0"), ("29502416", "s1"), ("29502416", "s2")],
    "ftbqmln":    [("29504032", "s0"), ("29504032", "s1"), ("29504032", "s2")],
    "ftbqmbias":  [("29501780", "s0"), ("29501780", "s1"), ("29501780", "s2")],
    "ftb4o":      [("29451652", "s0")],
    "ftbnorm":    [("29482212", "s0"), ("29482212", "s1"), ("29482212", "s2")],
    "ftb4e3":     [("29451642", "s0"), ("29451642", "s1"), ("29451642", "s2")],
    "ftb3i":      [("29469072", "s0"), ("29469072", "s1"), ("29469072", "s2")],
    "ftbqm1d":    [("29501773", "s0"), ("29501773", "s1"), ("29501773", "s2")],
    "ftbqm1dv":   [("29507368", "s0"), ("29507368", "s1"), ("29507368", "s2")],
    "ftbqm1dqk":  [("29511670", "s0"), ("29511670", "s1"), ("29511670", "s2")],
    "ftbqm1dvo":  [("29511673", "s0"), ("29511673", "s1"), ("29511673", "s2")],
}
RESULTS = ROOT / "results" / "imnet_base"


def last_epoch_acc(sid, seed, min_epochs=300):
    """Last-epoch test top-1 -- never max-over-epochs (docs, convention banner).

    Returns None for a seed that has not reached `min_epochs`: a still-training run's last
    logged epoch is not a result, and the winners here are BEHIND until ~epoch 250
    (docs 3.10.9.8), so a partial read is biased against exactly the arms under test.
    """
    p = RESULTS / f"results_IMNET_BASE_{sid}" / seed / "log.txt"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if len(rows) < min_epochs:
        return None
    acc = [r.get("test_acc1") for r in rows]
    acc = [a for a in acc if a is not None and a > 0]
    return acc[-1] if acc else None


def arm_acc():
    out = {}
    for arm, runs in RUNS.items():
        vals = [a for a in (last_epoch_acc(s, sd) for s, sd in runs) if a is not None]
        if vals:
            out[arm] = (sum(vals) / len(vals), len(vals))
    return out


# =========================================================================================
# W.*  weight-space statistics
# =========================================================================================
def _kurt(v):
    v = v.float().flatten()
    return float((((v - v.mean()) / (v.std() + 1e-12)) ** 4).mean())


def _stable_rank(W):
    s = torch.linalg.svdvals(W.float())
    return float(s.pow(2).sum() / s.pow(2).max())


@torch.no_grad()
def weight_stats(mw, heads=12):
    """Per-block weight statistics for blocks 0-8, returned as {feature: [b0..b8]}."""
    P = dict(mw.named_parameters())
    out = {}

    def put(k, b, v):
        out.setdefault(k, [None] * len(BLOCKS))[b] = float(v)

    for b in BLOCKS:
        qkv = P[f"blocks.{b}.attn.qkv.weight"].data.float()
        e = qkv.shape[0] // 3
        q, k, v = qkv[0:e], qkv[e:2 * e], qkv[2 * e:3 * e]
        qk = qkv[0:2 * e]
        proj = P[f"blocks.{b}.attn.proj.weight"].data.float()
        fc1 = P[f"blocks.{b}.mlp.fc1.weight"].data.float()
        fc2 = P[f"blocks.{b}.mlp.fc2.weight"].data.float()
        g1 = P[f"blocks.{b}.norm1.weight"].data.float()
        g2 = P[f"blocks.{b}.norm2.weight"].data.float()
        b1 = P[f"blocks.{b}.norm1.bias"].data.float()
        b2 = P[f"blocks.{b}.norm2.bias"].data.float()
        qkvb = P[f"blocks.{b}.attn.qkv.bias"].data.float()

        # --- raw tensor scale -------------------------------------------------------
        for nm, T in [("q", q), ("k", k), ("v", v), ("proj", proj), ("fc1", fc1), ("fc2", fc2)]:
            put(f"W.{nm}_norm", b, T.norm())
            put(f"W.{nm}_std", b, T.std())
        put("W.qk_norm", b, qk.norm())

        # --- the qk/v asymmetry, the one thing ftb4e3 keeps and ftbqm1d pools away ---
        put("W.qk_over_v", b, qk.std() / (v.std() + 1e-12))
        put("W.q_over_v", b, q.std() / (v.std() + 1e-12))

        # --- LN-COMPOSED: attention sees W diag(gamma), not W ------------------------
        # gamma_1 multiplies the input channels, i.e. the COLUMNS of W_q/W_k/W_v.
        qg, kg, vg = q * g1, k * g1, v * g1
        f1g = fc1 * g2
        put("W.eff_q_norm", b, qg.norm())
        put("W.eff_k_norm", b, kg.norm())
        put("W.eff_v_norm", b, vg.norm())
        put("W.eff_fc1_norm", b, f1g.norm())

        d_h = e // heads
        # per-head logit scale: E[q.k] ~ (1/sqrt(d_h)) * ||W_q^h|| ||W_k^h|| / d, per head
        lg, vw = [], []
        for h in range(heads):
            sl = slice(h * d_h, (h + 1) * d_h)
            lg.append(float(qg[sl].norm() * kg[sl].norm() / (d_h ** 0.5) / e))
            vw.append(float(vg[sl].norm()))
        put("W.logit_scale", b, sum(lg) / heads)
        put("W.logit_scale_head_disp", b,
            (torch.tensor(lg).std() / (torch.tensor(lg).mean() + 1e-12)))
        put("W.value_write", b, (vg.norm() * proj.norm() / e))
        put("W.logit_over_write", b,
            math.log((sum(lg) / heads) / (float(vg.norm() * proj.norm() / e) + 1e-12) + 1e-12))
        put("W.mlp_gain", b, f1g.norm() * fc2.norm() / e)
        put("W.attn_over_mlp", b,
            math.log(float(vg.norm() * proj.norm()) / (float(f1g.norm() * fc2.norm()) + 1e-12) + 1e-12))

        # --- per-head dispersion of the raw row norms (arrangement-sensitive) --------
        for nm, T in [("q", q), ("k", k), ("v", v)]:
            hn = torch.tensor([float(T[h * d_h:(h + 1) * d_h].norm()) for h in range(heads)])
            put(f"W.{nm}_head_disp", b, hn.std() / (hn.mean() + 1e-12))
            rn = T.norm(dim=1)
            put(f"W.{nm}_rownorm_cv", b, rn.std() / (rn.mean() + 1e-12))

        # --- shape of the value distribution ----------------------------------------
        for nm, T in [("qkv", qkv), ("proj", proj), ("fc1", fc1), ("fc2", fc2)]:
            put(f"W.{nm}_kurt", b, _kurt(T))
        put("W.qkv_srank", b, _stable_rank(qkv))
        put("W.proj_srank", b, _stable_rank(proj))
        put("W.fc1_srank", b, _stable_rank(fc1))

        # --- 1-D parameters ----------------------------------------------------------
        put("W.ln1_gain_mean", b, g1.mean())
        put("W.ln1_gain_std", b, g1.std())
        put("W.ln2_gain_mean", b, g2.mean())
        put("W.ln2_gain_std", b, g2.std())
        put("W.ln_bias_norm", b, (b1.norm() + b2.norm()) / 2)
        put("W.qkv_bias_norm", b, qkvb.norm())
        # the qkv bias also splits; a large q/k bias shifts every logit by a constant
        put("W.qk_bias_norm", b, qkvb[:2 * e].norm())
        put("W.v_bias_norm", b, qkvb[2 * e:].norm())
    return out


# =========================================================================================
# F.*  forward-pass statistics
# =========================================================================================
class AttnProbe:
    """Replaces Attention.forward with a copy that also records distributional stats.

    Mirrors models/vision_transformer.py:236-248 exactly (explicit softmax, no SDPA), so the
    probabilities recorded are the ones the model actually uses.
    """

    def __init__(self, model, grid=14):
        self.model, self.grid, self.stats, self._orig = model, grid, {}, []
        # pairwise patch distance in grid units, for the attention-distance measure
        ij = torch.stack(torch.meshgrid(torch.arange(grid), torch.arange(grid),
                                        indexing="ij"), -1).reshape(-1, 2).float()
        self.D = torch.cdist(ij, ij)

    def __enter__(self):
        for i, blk in enumerate(self.model.blocks):
            self._orig.append(blk.attn.forward)
            blk.attn.forward = self._wrap(i, blk.attn)
        return self

    def __exit__(self, *a):
        for blk, f in zip(self.model.blocks, self._orig):
            blk.attn.forward = f

    def _wrap(self, i, at):
        def fwd(x):
            B, N, C = x.shape
            qkv = at.qkv(x).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            logits = (q @ k.transpose(-2, -1)) * at.scale
            attn = logits.softmax(dim=-1)
            with torch.no_grad():
                p = attn.float()
                ent = -(p * (p + 1e-12).log()).sum(-1)              # B,H,N
                D = self.D.to(p.device)
                dist = (p[..., 1:, 1:] * D).sum(-1)                  # B,H,Npatch
                rec = dict(
                    logit_std=float(logits.float().std()),
                    logit_absmax=float(logits.float().abs().amax()),
                    attn_entropy=float(ent.mean() / math.log(N)),
                    attn_maxp=float(p.amax(-1).mean()),
                    attn_cls_mass=float(p[..., 1:, 0].mean()),
                    # distance is only meaningful when the row is renormalised over patches
                    attn_dist=float((dist / (p[..., 1:, 1:].sum(-1) + 1e-12)).mean()
                                    / float(D.mean())),
                    # how much of the mass a head puts on the diagonal (self-attention)
                    attn_self=float(p.diagonal(dim1=-2, dim2=-1).mean()),
                )
                # per-head spread of entropy: are heads differentiated at init?
                he = ent.mean(dim=(0, 2))
                rec["attn_entropy_head_disp"] = float(he.std() / (he.mean() + 1e-12))
                acc = self.stats.setdefault(i, {})
                for kk, vv in rec.items():
                    acc.setdefault(kk, []).append(vv)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return at.proj(out)
        return fwd


@torch.no_grad()
def repr_stats(x_tokens):
    """Statistics of one block's output representation (B, N, C)."""
    z = x_tokens.float()
    patches = z[:, 1:, :]
    zc = patches - patches.mean(dim=1, keepdim=True)
    zn = F.normalize(patches, dim=-1)
    cos = zn @ zn.transpose(1, 2)
    n = cos.shape[-1]
    off = (cos.sum(dim=(1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(1)) / (n * (n - 1))
    # participation-ratio effective rank of the token covariance
    C_ = (zc.transpose(1, 2) @ zc) / zc.shape[1]
    ev = torch.linalg.eigvalsh(C_.mean(0).double()).clamp_min(0)
    eff = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
    ch = z.reshape(-1, z.shape[-1])
    return dict(
        tok_cos=float(off.mean()),
        eff_rank=eff,
        feat_norm=float(z.norm(dim=-1).mean()),
        # channel outliers: how concentrated the representation's energy is on few channels
        chan_kurt=float((((ch - ch.mean(0)) / (ch.std(0) + 1e-12)) ** 4).mean()),
        chan_energy_top1=float((ch.var(0) / ch.var(0).sum()).amax()),
    )


@torch.no_grad()
def forward_stats(mw, x):
    """F.* per block plus handoff.* at the input to block 9."""
    caps = {}
    rho = {}

    def wrap(i, blk):
        def fwd(t):
            r_in = t
            d_attn = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(t))))
            r_out = r_in + d_attn
            d_mlp = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(r_out))))
            out = r_out + d_mlp
            nn_ = lambda z: torch.norm(z.float(), dim=-1)
            rho[i] = dict(rho_attn=float((nn_(d_attn) / (nn_(r_in) + 1e-8)).mean()),
                          rho_mlp=float((nn_(d_mlp) / (nn_(r_out) + 1e-8)).mean()))
            caps[i] = out.detach()
            if i == 0:
                caps[-1] = r_in.detach()
            return out
        return fwd

    orig = [blk.forward for blk in mw.blocks]
    for i, blk in enumerate(mw.blocks):
        blk.forward = wrap(i, blk)
    mw.eval()
    with AttnProbe(mw) as probe:
        mw(x)
    for blk, f in zip(mw.blocks, orig):
        blk.forward = f

    out = {}
    for kk in probe.stats[0]:
        out[f"F.{kk}"] = [probe.stats[b][kk][0] for b in BLOCKS]
    out["F.rho_attn"] = [rho[b]["rho_attn"] for b in BLOCKS]
    out["F.rho_mlp"] = [rho[b]["rho_mlp"] for b in BLOCKS]
    rs = {b: repr_stats(caps[b]) for b in BLOCKS}
    for kk in rs[0]:
        out[f"F.{kk}"] = [rs[b][kk] for b in BLOCKS]

    # what blocks 9-11 (identical random weights in every arm) actually receive
    hand = repr_stats(caps[HANDOFF - 1])
    hand["rho_attn_blk9"] = rho[HANDOFF]["rho_attn"]
    hand["rho_mlp_blk9"] = rho[HANDOFF]["rho_mlp"]
    hand["attn_entropy_blk9"] = probe.stats[HANDOFF]["attn_entropy"][0]
    hand["logit_std_blk9"] = probe.stats[HANDOFF]["logit_std"][0]
    scalars = {f"handoff.{k}": v for k, v in hand.items()}
    return out, scalars


# =========================================================================================
# G.*  gradient statistics from one backward pass at init
# =========================================================================================
def grad_stats(mw, x, y):
    mw.train()
    mw.zero_grad(set_to_none=True)
    loss = F.cross_entropy(mw(x), y)
    loss.backward()
    out, scal = {}, {"G.loss": float(loss)}
    P = dict(mw.named_parameters())
    for b in BLOCKS:
        g2 = w2 = 0.0
        for n, p in P.items():
            if n.startswith(f"blocks.{b}.") and p.grad is not None:
                g2 += float(p.grad.float().pow(2).sum())
                w2 += float(p.data.float().pow(2).sum())
        out.setdefault("G.gnorm", [None] * 9)[b] = g2 ** 0.5
        # ||g||/||W||: AdamW's step is ~lr per coordinate regardless of |g|, but the RELATIVE
        # movement of a tensor per step scales with 1/||W||; this is the slow-start proxy.
        out.setdefault("G.g_over_w", [None] * 9)[b] = g2 ** 0.5 / (w2 ** 0.5 + 1e-12)
    # the deep blocks are identical weights in every arm: their gradient is a pure readout
    # of what blocks 0-8 hand them
    for b in [9, 10, 11]:
        g2 = sum(float(p.grad.float().pow(2).sum())
                 for n, p in P.items() if n.startswith(f"blocks.{b}.") and p.grad is not None)
        scal[f"G.gnorm_blk{b}"] = g2 ** 0.5
    for tag, pref in [("head", "head."), ("patch", "patch_embed."), ("pos", "pos_embed")]:
        g2 = sum(float(p.grad.float().pow(2).sum())
                 for n, p in P.items() if n.startswith(pref) and p.grad is not None)
        scal[f"G.gnorm_{tag}"] = g2 ** 0.5
    mw.zero_grad(set_to_none=True)
    return out, scal


# =========================================================================================
def main():
    ap = argparse.ArgumentParser(parents=[__import__("main").get_args_parser()],
                                 add_help=False, conflict_handler="resolve")
    ap.add_argument("--out", default="plots/cache/ckpt_diff.json")
    ap.add_argument("--n_images", type=int, default=128)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    args.nb_classes = 1000
    for k in ["skip_attn_segments", "weight_shuffle", "target_model_weight_shuffle",
              "init_method_copied_blocks", "attention_residual_scaling",
              "attention_out_scaling", "learning_rate_scaling_params"]:
        setattr(args, k, {})
    for k in ["random_blocks", "clip_outlier_blocks", "delete_blocks",
              "init_method_scaled_blocks", "freeze_blocks", "hold_back_blocks"]:
        setattr(args, k, [])
    for k in ["skip_load_blocks", "skip_load_block_attributes", "freeze_block_attributes"]:
        v = getattr(args, k, "")
        setattr(args, k, [z for z in v.split(",")] if isinstance(v, str) and v else [])
    args.distributed = False
    args.gpu = args.rank = 0
    args.world_size = 1
    torch.manual_seed(0)

    from datasets import build_dataset
    ds, _ = build_dataset(is_train=False, args=args)
    ld = torch.utils.data.DataLoader(ds, batch_size=args.n_images, shuffle=True,
                                     num_workers=8, pin_memory=True,
                                     generator=torch.Generator().manual_seed(0))
    x, y = next(iter(ld))
    x, y = x.cuda(), y.cuda()
    print(f"measuring on {tuple(x.shape)} real val images", flush=True)

    arms = [a for a in (args.arms.split(",") if args.arms else ARMS)]
    tgt = M._target_params(args)
    out = {}
    for arm in arms:
        torch.manual_seed(0)
        if arm == "ftb4o":
            mw = M.build_ftb4o(args, x, M._proc_model(args, random_blocks=[]))
        else:
            mw = M.build_arm(arm, args, tgt).cuda()
        rec = {"per_block": {}, "scalar": {}}
        rec["per_block"].update(weight_stats(mw))
        f, hs = forward_stats(mw, x)
        rec["per_block"].update(f)
        rec["scalar"].update(hs)
        g, gs = grad_stats(mw, x, y)
        rec["per_block"].update(g)
        rec["scalar"].update(gs)
        out[arm] = rec
        print(f"{arm:11} qk/v {sum(rec['per_block']['W.qk_over_v'])/9:6.3f}  "
              f"logit_scale {sum(rec['per_block']['W.logit_scale'])/9:8.4f}  "
              f"attn_ent {sum(rec['per_block']['F.attn_entropy'])/9:6.4f}  "
              f"tok_cos {sum(rec['per_block']['F.tok_cos'])/9:6.3f}  "
              f"g/w {sum(rec['per_block']['G.g_over_w'])/9:.3e}", flush=True)
        del mw
        torch.cuda.empty_cache()

    out["_acc"] = {k: v[0] for k, v in arm_acc().items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
