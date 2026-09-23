# Norm-ratio scaling ablation: review, evidence, next steps

**Question:** is the up/down-scaling of the value and output-projection matrices implemented
correctly enough to support the claim that what proc pretraining contributes to the late
blocks is the *magnitude* of the attention write relative to the residual stream, and not
the learned weight structure?

**Verdict:** the implementation is correct and the claim is supported by the IN-100
dissociation (upscaled random recovers proc; downscaled proc falls back to random). What is
left is one missing control, one verification figure, and some reporting discipline — plus
small code fixes that affect precision, not conclusions.

Scope: `utils.scale_layer_weights` (`utils.py:1121`), the `*_match_*_delta_norms` init paths
(`main.py:832-1119`), model/target wiring (`main.py:559-606`), ratio definition
(`engine.py:1413-1420`). Measurements from jobs 29380866 / 29380885 / 29380909 and run log
`logs/ft_29377630_ftb6s.out`.

---

## Next steps, in priority order

**1. Run the profile control on IN-100.** *(highest value; one extra arm)*
Necessity + sufficiency *at proc's values* does not exclude "any sufficiently large
attention write helps, and proc's particular profile is incidental". Scale to 0.5x and 2x
proc's rho profile, or to a flat profile across blocks. If accuracy is flat across those,
the honest claim is "attention write magnitude above some threshold is what matters", not
"the proc-specific magnitude profile is what transfers". The IN-1k partials — where the `*s`
arm sits slightly *above* the proc arm — are exactly the pattern that makes a reviewer ask.

**2. Produce the "the match landed" figure.** *(minutes; needed for the paper)*
`main.py:1122` skips the post-scaling re-analysis exactly when `args.model == "vit_base"`, so
no log confirms the achieved ratios. The data already exists: `model_analyse` logs
`attn_delta_norm_ratio_layer{i}` at epoch **-1** and every 10 epochs for `vit_base`
(`main.py:1182-1184`). Plot proc vs scaled per layer at epoch -1 — they should coincide —
and optionally the trajectory, which also shows whether the intervention persists in training.

**3. Fix the bias scaling before the final numbers.** *(2 lines; see §5.1)*
`proj.bias` / `fc2.bias` are not scaled, so the match is inexact when they are non-zero:
+9.0% at the most extreme downscale factor. Does **not** invalidate the IN-100 result
(0.0083 achieved vs 0.0076 target, still ~150x below proc), so do not rerun on its account.

**4. Finish the IN-1k pairs and add seeds.** *(compute)*
Only `ftb1` reached epoch 300 (79.02). Batch 1 (29377xxx) died at ~epoch 122; batch 2
(29380xxx) is at ~150 and running. Single-seed gaps of +/-0.5 are within noise. Rest the
claim on IN-100 and use IN-1k as a scaling check once complete.

**5. Report the three quantities the intervention does not match** (§5.3). Doing this
proactively strengthens the result: the benefit survives despite large, measurable forward-
pass differences, which is stronger evidence that the learned structure is dispensable.

**6. Code hygiene** (§6): sort `init_method_scaled_blocks`, guard `torch.distributed.barrier()`,
avoid `--simultaneous_init_scaling`, watch CPU memory in `attention_residual_analysis`.

---

## 2. What is established

Both directions exist in the code and both were run on ImageNet-100:

| `--init_method` | model that trains | target (stats source) | v/proj | tests |
|---|---|---|---|---|
| `upscale_random_match_attn_delta_norms` | random (`pr_load_model(path="")`) | proc ckpt `--initialize` | x s, s>1 | magnitude **sufficient** |
| `downscale_pr_match_attn_delta_norms` | proc ckpt | random, copied from the same base weights | x s, s<1 | magnitude **necessary** |
| `upscale_pr_match_attn_delta_norms` | proc | proc ⇒ r=1, no-op unless `--target_model_weight_shuffle` | — | — |

Each has `_match_mlp_delta_norms` (scales `fc2`) and `_match_delta_norms` (both sublayers)
variants. The two models are identical everywhere except the blocks under study: the `pr`
checkpoint drops head/cls/pos/patch_embed, and `random_blocks` merely deletes checkpoint
keys, so both arms share the same random weights elsewhere.

The dissociation is real: scaling v/proj changes ‖Delta‖ but **not its direction**, so the
downscale arm holds the learned structure fixed and removes only the magnitude. Only the
SLURM scripts for the downscale direction are missing on this disk; every `*s` script passes
`upscale_random_match_attn_delta_norms`.

## 3. How to phrase the claim

"Proc init is about how much each attn layer writes to the residual stream relatively" is an
exact description of the *metric*, but false as a description of the *weights*: in the proc
model's own context those blocks sit at rho = 0.05-0.09, **below** random init's 0.15. Adding
"and the weight structure plays no role, only the magnitude" fixes it, because it turns a
description into an interventional hypothesis — rho need not be intrinsic to the weights,
only controllable, which it is.

Suggested wording:

> What proc pretraining contributes to the last blocks is the per-layer magnitude of the
> attention write relative to the residual stream, not the learned weight structure:
> installing that magnitude in a random-init model recovers the benefit, and removing it from
> a proc-init model destroys it.

Scope it to the attention sublayers of the last k blocks (`norm1`+`qkv`+`proj` are
transplanted; MLPs and blocks 0-5 are random in both arms). The k = 1..6 sweep is a depth
axis worth showing.

**Why rho = ‖Delta‖/‖r_in‖ and not something else.** A reviewer will ask. Pre-LN makes the
stream norm nearly invisible: every sublayer sees `norm1(r)` and the head sees `norm(r)`,
both scale-free. The only role of ‖r‖ is gain control — a fixed write rotates a big stream
less than a small one — and that effect *is* rho at the next layer. So matching rho per block
already handles the dilution chain, while matching ‖r_out‖/‖r_in‖ targets a number the
forward pass never reads. R = sqrt(1 + rho² + 2·rho·cos) also mixes the two variables the
experiment separates: if the hypothesis is about magnitude, rho names it and R does not.
R is additionally ill-conditioned where rho << 1 and has an unreachable floor sqrt(1-cos²).
**Keep ‖Delta‖/‖r_in‖.** (`check_activation_norms.py --target out_in` exists for comparison,
not as a recommendation.)

**What is actually scaled.** Only the V-slice of `qkv.weight` and `proj.weight`, each by
sqrt(r). q and k are multiplied by 1.0 and `norm1` by 1.0, so the attention pattern is
bit-identical — q and k enter the softmax, so scaling them would change *which* tokens are
attended to, not just how much is written. Only V and proj sit strictly downstream of the
softmax, which is what makes this a magnitude-only intervention. The MLP variant scales only
`fc2` (`fc1` is upstream of GELU, so not linear in Delta_mlp).

---

## 4. Evidence

### 4.1 The intervention is exact (except biases)

V-slice and `proj.weight` each x sqrt(r) ⇒ Delta_attn x r exactly; `fc2.weight` x r ⇒
Delta_mlp x r exactly. Verified numerically on `results/pr_vitb_n/pr_6066174_final.pth`,
block 11.

Biases are not scaled, so

    Delta_attn(scaled) = s²·(W_p A W_v x) + s·(W_p A b_v) + b_p

Only the first term carries the intended r. At random init all Linear biases are zero
(`init_weights_vit_timm`) ⇒ the **upscale direction is exact**. With proc biases:

| requested r | 0.00757 | 0.0102 | 0.0289 | 0.0771 | 0.25 |
|---|---|---|---|---|---|
| current recipe (v & proj weight x sqrt(r)) | **+9.0%** | +6.8% | +2.7% | +1.2% | +0.4% |
| `proj.weight` **and** `proj.bias` x r | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| current recipe, MLP (`fc2.weight` x r) | +8.5% | +6.3% | +2.1% | +0.8% | +0.2% |
| `fc2.weight` **and** `fc2.bias` x r | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

r ~ 0.0076 is the block-11 factor the downscale direction asks for. Exact recipe:

```python
block.attn.proj.weight.data *= r      # instead of v,proj *= sqrt(r)
block.attn.proj.bias.data   *= r
block.mlp.fc2.weight.data   *= r      # instead of fc2.weight only
block.mlp.fc2.bias.data     *= r
```

Downside: the whole factor lands on one matrix. Splitting sqrt(r) across v and proj while
also scaling `b_v` by sqrt(r) and `b_p` by r keeps the split and is equally exact.

### 4.2 What the norm ratios mean

Decompose each sublayer's write against the stream it is written into:

    alpha = 1 + rho·cos        surviving fraction of the incoming stream
    beta  = rho·sqrt(1-cos²)   newly written content, in units of ‖r_in‖
    R     = sqrt(alpha²+beta²) = ‖r_out‖/‖r_in‖

alpha ~ 1 = refine, alpha ~ 0 = erase, alpha < 0 = invert.

*Random init*: alpha = 1.00 everywhere, cos ~ 0, beta = 0.15-0.31; the stream creeps 17 -> 42
over 12 blocks.

*ftb6 init (proc norm1+attn in blocks 6-11) — what the logged ratios describe:*

| blk | rho | cos | **alpha** | beta | R | ‖r_in‖ |
|---|---|---|---|---|---|---|
| 6 | 1.01 | -0.07 | +0.93 | 1.01 | 1.38 | 33 |
| 7 | 1.25 | -0.20 | +0.75 | 1.22 | 1.44 | 47 |
| 8 | 1.37 | -0.27 | +0.63 | 1.31 | 1.47 | 68 |
| 9 | 2.31 | -0.45 | **-0.08** | 2.02 | 2.09 | 100 |
| 10 | 2.67 | -0.69 | **-0.82** | 1.93 | 2.10 | 210 |
| 11 | 1.19 | -0.46 | +0.43 | 1.04 | 1.17 | 407 |

These layers progressively **delete the stream that blocks 0-5 built** — block 9 cancels it,
block 10 inverts it — writing 1-2x the stream norm of new content in its place. The stream
explodes 33 -> 407 and the (identical, random) MLPs of blocks 6-11 are squeezed out
(rho_mlp 0.22 -> 0.02).

*The same weights among all 12 proc blocks behave oppositely:*

| blk | rho | cos | alpha | R | ‖r_in‖ |
|---|---|---|---|---|---|
| 1-8 | 0.05-0.09 | -0.02..-0.30 | +0.98..+1.00 | 0.98-1.00 | ~1600 |
| 9 | 0.23 | -0.60 | +0.86 | 0.88 | 1588 |
| 10 | 0.21 | -0.45 | +0.91 | 0.93 | 1372 |
| 11 | 0.67 | -0.65 | +0.57 | 0.77 | 1316 |

Blocks 1-8 are near-identity; blocks 9-11 *shrink* the stream. Cause: `norm1` strips the
input scale, so ‖Delta‖ is set by the weights and is nearly independent of ‖r_in‖. Across the
two contexts rho differs by 2-18x while ‖Delta‖ differs by only ~2-4x (LN removes the input's
scale but not its direction). Absolute write at block 11: ~880 (proc model), ~480 (hybrid),
~6 (random init).

Consequences: proc pretraining builds sublayers that **emit ~100x larger writes** than random
init and, in the late blocks, aim them **against** the stream — a norm-control / cleanup role
(R = 0.77-0.93 in its own model). The high ratios in the run logs are a property of **the
transplant**: blocks 6-11 land in a stream ~40x smaller than the one they were trained in.
Always report rho together with cos (or as alpha/beta) — rho = 1.2 with cos = -0.46 and
rho = 1.2 with cos = 0 are opposite operations.

Caveats: k-Dyck-trained weights fed ImageNet; the all-proc model runs with random embeddings
(its own `pos_embed` is [1,196,768], head is 128-way, **no CLS token** in pretraining), so the
block-0 blow-up in that table is likely a random-patch-embed artifact. The cosines and
absolute write magnitudes are the robust parts.

### 4.3 What the intervention does not match — disclose all three

**(a) Activation norms.** The per-block ratio is matched to 0.0%, and the stream still ends
3x too large:

| blk | attn ratio (matched) | ‖r_in‖ proc → scaled | cos(r_in, Delta) proc → scaled | mlp ratio |
|---|---|---|---|---|
| 6 | 1.011 / 1.011 (0.0%) | 33.3 → 33.3 (+0.0%) | **-0.07** → -0.00 | -3.7% |
| 7 | 1.248 / 1.248 (0.0%) | 46.7 → 48.4 (+3.6%) | **-0.20** → -0.04 | -11.4% |
| 8 | 1.370 / 1.370 (0.0%) | 67.9 → 76.6 (+12.7%) | **-0.27** → +0.01 | -26.1% |
| 9 | 2.312 / 2.312 (0.0%) | 100.1 → 130.3 (+30.1%) | **-0.45** → -0.01 | -41.5% |
| 10 | 2.665 / 2.665 (0.0%) | 210.4 → 325.5 (+54.7%) | **-0.69** → -0.01 | -57.8% |
| 11 | 1.189 / 1.189 (0.0%) | 406.6 → 922.2 (**+126.8%**) | **-0.46** → +0.01 | -66.5% |

Final block output: 465 → 1437 (**+209%**). Geometry accounts for all of it: block 10 proc
grows by sqrt(1 + 2.66² + 2·2.66·(-0.69)) ~ 2.1 (observed 1.93), scaled by sqrt(1 + 2.66²) =
2.84 (observed 2.83). The MLPs of blocks 6-11 hold *identical* random weights in both arms;
`norm2` makes Delta_mlp scale-invariant, so the inflated stream mechanically suppresses them.
This is largely invisible to the forward pass (pre-LN) but **not** to optimization:
LayerNorm's backward scales like 1/‖x‖, so gradients reaching earlier blocks shrink by
roughly the same factor.

**(b) Rotation — provably unmatchable.** Direction is all that propagates, and the rotation is
cos(r_in, r_out) = alpha / R:

| blk | proc (alpha/R) | reachable with random directions? |
|---|---|---|
| 6 | +0.67 | yes (rho = 1.10) |
| 7 | +0.52 | yes (rho = 1.64) |
| 8 | +0.43 | yes (rho = 2.13) |
| 9 | **-0.04** | **no** |
| 10 | **-0.39** | **no** |
| 11 | +0.37 | yes (rho = 2.51) |

With cos ~ 0 the rotation is 1/sqrt(1+rho²), strictly positive at any scale. Proc blocks 9
and 10 rotate the stream *past orthogonal* — partial reversal — which **no rescaling of
random weights can reproduce**. This is the cleanest statement of what "structure" buys.

**(c) Weight norms.** The multipliers look alarming next to random init (2.41, 2.94, 3.60,
5.86, 9.90, 11.49 ⇒ Delta_attn x5.8 ... x132), but proc weights are themselves 1.8-4.6x larger
than random init:

| blk | s | ‖W_v‖ upscaled-rand | ‖W_v‖ proc | overshoot | ‖W_p‖ upscaled-rand | ‖W_p‖ proc | overshoot |
|---|---|---|---|---|---|---|---|
| 6 | 2.41 | 37.0 | 27.3 | 1.36x | 37.0 | 31.7 | 1.17x |
| 7 | 2.94 | 45.2 | 29.7 | 1.52x | 45.2 | 34.0 | 1.33x |
| 8 | 3.60 | 55.3 | 31.2 | 1.77x | 55.3 | 35.9 | 1.54x |
| 9 | 5.86 | 89.9 | 33.8 | 2.66x | 89.9 | 38.7 | 2.32x |
| 10 | 9.90 | 152.0 | 48.0 | **3.17x** | 152.3 | 51.4 | 2.96x |
| 11 | 11.49 | 176.7 | 71.1 | 2.49x | 176.7 | 76.8 | 2.30x |

So the arms sit within 1.2-3.2x of each other, and the AdamW "stiffness" effect (per-step
updates are roughly scale-free) is mild. The overshoot is itself a finding: proc reaches the
same rho with ~2.5x smaller weights, i.e. it buys magnitude through structure. For the
downscale arm the mirror number is harsher — proc's block-11 v/proj land at ~0.4x of
*random-init* norm, so those weights move quickly relative to their own size and the
intervention may decay during training. Track it via the epoch-10 ratio logs.

---

## 5. Code items

| # | Issue | Criticality | Acts on results? |
|---|-------|-------------|------------------|
| a | Unscaled `proj.bias` / `fc2.bias` (§4.1) | Medium — precision for final numbers | ~9% off at the most extreme block; conclusion unaffected |
| b | No post-scaling confirmation for `vit_base` (`main.py:1122`) | Medium-high | Verification gap; data exists in wandb |
| c | Block order load-bearing but unchecked | Low (latent) | No — scripts pass ascending |
| d | `type=bool` flags, unguarded `barrier()`, CPU memory | Low | No |

**c — block order.** `main.py:545` iterates `--init_method_scaled_blocks` in the given order.
Sequential matching is only valid ascending, because rescaling block *i* changes the stream
entering blocks > *i*. Scripts pass `"6,7,8,9,10,11"`, so nothing is affected; `"11,10"` would
silently mismatch. `"all"` is documented in the help string but crashes the `int(x)` parse.
Fix: `sorted(...)` at parse time and accept `"all"`.

**d — footguns.**
- `--simultaneous_init_scaling` is `type=bool` (`main.py:339`), so passing `false` sets it
  **True** (`bool("false")` is truthy) — as for every `type=bool` flag in the repo. Harmless
  today because it is never passed. Its branch computes `current_stats` once for all blocks,
  so every block after the first gets a stale target. Do not enable it.
- `torch.distributed.barrier()` at `main.py:910/995/1115` is unguarded; a non-torchrun
  single-GPU run crashes there. Wrap in `if args.distributed:`.
- `attention_residual_analysis` accumulates full activations on CPU: ~3.0 GB per tensor,
  4 tensors ⇒ ~12 GB per block at 5000 samples for `vit_base`, ~70 GB for the 6-block target
  pass. It fits on your nodes; a streaming fold like `residual_stream_stats.py::_fold` would
  remove the risk if `k` or the block count grows.

Not an issue: the `if scale_qk == scale_v` branch in `scale_layer_weights` is mathematically
identical to the else branch (scaling the whole `qkv` by a common value == scaling q,k by
`scale_qk` and v by `scale_v`). It is a fast path, not a behaviour change.

---

## 6. Reproducing the checks

```
sbatch scaling_checks/run_scaling_checks.sh      # all of the below, ~5 min on one L40S
```

- `scaling_checks/check_scaling_exactness.py` — is the applied factor exact? (§4.1)
- `scaling_checks/check_weight_norms.py` — where do the weights land? (§4.3c)
- `scaling_checks/check_activation_norms.py` — do the activations match? (§4.3a);
  `--target out_in` matches ‖r_out‖/‖r_in‖ instead, for comparison
- `scaling_checks/interpret_norm_ratios.py` — alpha/beta decomposition (§4.2);
  `--tokens cls|patch` splits CLS from patch tokens

Jobs behind the numbers here: 29380866, 29380885, 29380909; run log
`logs/ft_29377630_ftb6s.out`. The check scripts use 256 ImageNet val images, so their solved
factors differ slightly from the run logs (which use 5000 augmented train images).
