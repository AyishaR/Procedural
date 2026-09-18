"""Experiment 3: how much does each block's function change between adjacent epochs, on identical inputs?

For the pair e -> e+1 the residual stream h_l of epoch e is computed on a fixed diagnostic set (full token
sequence), and block l of epoch e and of epoch e+1 are both applied to the same h_l.  With the block update
F(h) = attn(norm1 h) + mlp(norm2(h + attn(norm1 h))):

    U_l(e) = mean_x ||F_{l,e+1}(h) - F_{l,e}(h)|| / (||h|| + eps)

reported for the CLS token, for the patch tokens (mean of the per-token ratio) and for the whole sequence
(Frobenius); likewise for the attention sublayer alone (identical input h) and the MLP sublayer alone
(identical input h + attn_e(h)); plus the write ratio ||F_e|| / ||h||, the relative change ||dF|| / ||F_e||
and the weight-space change ||theta_{e+1} - theta_e|| / ||theta_e|| per block (k-bias excluded).  Always
one-epoch pairs; the last two pairs (297->298, 298->299) come from the fp32 full checkpoints because the
fp16 dumps' rounding (~2e-4 relative per block) equals the true late-epoch weight change; for those pairs the
fp16-dump version is computed as well (`fp16_variant`: the actual two-checkpoint rounding error), and `--floor`
measures the single-checkpoint floor (fp16 dump vs fp32 checkpoint of the same epoch).  Strict fp32, TF32 off.

  python -m dynamics.funcchange --runs all --pairs all --floor
"""
import argparse
import os
import time

import torch

from dynamics import common
from dynamics import data as ddata
from dynamics import registry as dreg
from probes import registry as preg

EPS = 1e-6
KEYS = ("U_cls", "U_patch", "U_frob", "Ua_cls", "Ua_patch", "Ua_frob", "Um_cls", "Um_patch", "Um_frob",
        "W_cls", "W_patch", "W_frob", "R_frob", "U_cls_sq")


def load_from(model, path, zero_k_bias=True):
    sd = preg.load_state_dict_file(path)
    info = preg.zero_k_bias_(sd) if zero_k_bias else {}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (path, missing[:3], unexpected[:3])
    model.k_bias_removed = info
    return info


def ratios(d, h):
    """per-token ratio ||d|| / ||h|| -> (cls [B], patch-mean [B], frobenius [B])."""
    nd, nh = d.norm(dim=-1), h.norm(dim=-1)
    r = nd / (nh + EPS)
    frob = d.flatten(1).norm(dim=1) / (h.flatten(1).norm(dim=1) + EPS)
    return r[:, 0], r[:, 1:].mean(1), frob


@torch.no_grad()
def compare(mA, mB, X, chunk):
    L = len(mA.blocks)
    acc = {k: torch.zeros(L, dtype=torch.float64, device=X.device) for k in KEYS}
    N = X.shape[0]
    for s in range(0, N, chunk):
        x = X[s:s + chunk].float()
        hs = common.states(mA, x)
        for l in range(L):
            h = hs[l]
            A, B = mA.blocks[l], mB.blocks[l]
            aA, aB = common.attn_branch(A, h), common.attn_branch(B, h)
            mA_ = common.mlp_branch(A, h + aA)
            mB_iso = common.mlp_branch(B, h + aA)
            FA = aA + mA_
            FB = aB + common.mlp_branch(B, h + aB)
            for tag, d, ref in (("U", FB - FA, h), ("Ua", aB - aA, h), ("Um", mB_iso - mA_, h + aA), ("W", FA, h)):
                c, p, f = ratios(d, ref)
                acc[f"{tag}_cls"][l] += c.double().sum(); acc[f"{tag}_patch"][l] += p.double().sum(); acc[f"{tag}_frob"][l] += f.double().sum()
                if tag == "U":
                    acc["U_cls_sq"][l] += c.double().pow(2).sum()
            acc["R_frob"][l] += ((FB - FA).flatten(1).norm(dim=1) / (FA.flatten(1).norm(dim=1) + EPS)).double().sum()
    out = {k: (v / N).cpu().tolist() for k, v in acc.items()}
    mean_sq = out.pop("U_cls_sq")
    out["U_cls_sd"] = [max(0.0, q - u * u) ** 0.5 for q, u in zip(mean_sq, out["U_cls"])]
    return out


@torch.no_grad()
def weight_change(mA, mB):
    L = len(mA.blocks)
    num, den = [0.0] * L, [0.0] * L
    for (n, pa), (_, pb) in zip(mA.named_parameters(), mB.named_parameters()):
        if not n.startswith("blocks."):
            continue
        l = int(n.split(".")[1])
        a, b = pa.float(), pb.float()
        if n.endswith("attn.qkv.bias"):
            D = a.shape[0] // 3
            a, b = torch.cat([a[:D], a[2 * D:]]), torch.cat([b[:D], b[2 * D:]])
        num[l] += float((b - a).pow(2).sum()); den[l] += float(a.pow(2).sum())
    return [(n / d) ** 0.5 if d > 0 else None for n, d in zip(num, den)], [d ** 0.5 for d in den]


def pair_sources(e, i, use_fp32_last):
    """(pathA, pathB, source) for the pair i -> i+1."""
    fa, fb = dreg.full_ckpt(e, i), dreg.full_ckpt(e, i + 1)
    if use_fp32_last and fa is not None and fb is not None:
        return fa, fb, "fp32"
    return preg.ckpt_path(dreg.epoch_entry(e, i)), preg.ckpt_path(dreg.epoch_entry(e, i + 1)), "fp16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="all")
    ap.add_argument("--pairs", default="all", help="all = 0..298 (pair i -> i+1) | comma list of i")
    ap.add_argument("--sets", default="diag_train2k,diag_test2k")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--no-fp32-last", dest="fp32_last", action="store_false")
    ap.add_argument("--floor", action="store_true", help="fp16 dump vs fp32 checkpoint of the same epoch (297-299)")
    ap.add_argument("--limit", type=int, default=None, help="images per set (smoke)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.umask(0o002)
    dreg.ensure_dirs()
    dreg.check_subsets()
    sets = {s: ddata.load_resident(s, "cuda", torch.float16, limit=args.limit) for s in args.sets.split(",")}
    lr = dreg.lr_epoch_means()
    suf = f"__limit{args.limit}" if args.limit else ""
    mA, mB = preg.new_model().cuda().eval(), preg.new_model().cuda().eval()
    for m in (mA, mB):
        for p in m.parameters():
            p.requires_grad_(False)
    with common.precision("fp32"):
        for e in dreg.resolve_runs(args.runs):
            avail = set(dreg.available_files(e))
            pairs = list(range(299)) if args.pairs == "all" else [int(x) for x in args.pairs.split(",") if x]
            pairs = [i for i in pairs if i in avail and i + 1 in avail]
            jobs = [("pair", i) for i in pairs] + ([("floor", i) for i in dreg.FP32_FILES if dreg.full_ckpt(e, i) and i in avail] if args.floor else [])
            todo = [j for j in jobs if args.force or not (dreg.RES / "funcchange" / f"{e['name']}@{j[0] if j[0] == 'floor' else 'e'}{j[1]}{suf}.json").exists()]
            print(f"{e['name']}: {len(todo)} of {len(jobs)} jobs to do", flush=True)
            for kind, i in todo:
                t0 = time.time()
                if kind == "pair":
                    pa, pb, source = pair_sources(e, i, args.fp32_last)
                else:
                    pa, pb, source = preg.ckpt_path(dreg.epoch_entry(e, i)), dreg.full_ckpt(e, i), "fp16-vs-fp32"
                ka, kb = load_from(mA, pa), load_from(mB, pb)
                rec = dict(run=e["name"], arm=e["arm"], seed=e["seed"], role=dreg.ROLE[e["name"]], kind=kind, file_from=i,
                           file_to=i + 1 if kind == "pair" else i, E_from=dreg.epoch_of(i), E_to=dreg.epoch_of(i + 1) if kind == "pair" else dreg.epoch_of(i),
                           source=source, paths=[str(pa), str(pb)], lr_mean_from=float(lr[i]), k_bias_removed=[ka, kb],
                           chunk=args.chunk, limit=args.limit, sets={}, **dreg.stamp())
                for name, (X, y, meta) in sets.items():
                    rec["sets"][name] = dict(n=int(X.shape[0]), subset_sha256=meta.get("subset_sha256"), **compare(mA, mB, X, args.chunk))
                rec["weight_rel_change"], rec["weight_norm"] = weight_change(mA, mB)
                if kind == "pair" and source == "fp32":          # the same pair from the fp16 dumps: two-checkpoint rounding error
                    fa16, fb16 = preg.ckpt_path(dreg.epoch_entry(e, i)), preg.ckpt_path(dreg.epoch_entry(e, i + 1))
                    load_from(mA, fa16), load_from(mB, fb16)
                    rec["fp16_variant"] = {name: compare(mA, mB, X, args.chunk) for name, (X, y, meta) in sets.items()}
                    rec["fp16_variant"]["weight_rel_change"] = weight_change(mA, mB)[0]
                rec["seconds"] = time.time() - t0
                dreg.write_json(dreg.RES / "funcchange" / f"{e['name']}@{'floor' if kind == 'floor' else 'e'}{i}{suf}.json", rec)
                first = next(iter(rec["sets"].values()))
                print(f"  {kind} {i:3d} ({source:11s}) U_cls b0..b11: " + " ".join(f"{u:.1e}" for u in first["U_cls"]) + f"  ({rec['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
