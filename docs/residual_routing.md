# Residual routing on the ImageNet ViT: independently weighted residual inputs

Ported from the layer-scale repo (`arch/residual_routing.py`, spec in
`analysis/notes/residual_routing_handoff.md`), where the same architecture is training on ClimbMix
for GPT-2 124M and 350M. Lines 1-395 of `models/residual_routing.py` are that file verbatim; only
the ~60-line tail is ViT-specific.

## What the model computes

The single residual stream is replaced by an explicit per-token history. `history[0]` is the
tensor entering block 0 (`patch_embed -> _pos_embed -> norm_pre`, so cls token, position embedding
and `pos_drop` applied). `history[k]` for `k >= 1` is the k-th branch's output **including `ls1`/
`ls2` (LayerScale) and `drop_path1`/`drop_path2`** — exactly what the vanilla `Block` would have
added to the stream. Every consumer reads its own weighted sum of every earlier entry,

    mixed_i = sum_{j < i} a[i, j] * history[j],    i = 1 .. M+1,   M = 2 * depth

and the mixes **replace** the residual additions; nothing is added back. `Block.forward` is never
called.

For ViT-B/16 (depth 12): `M = 24` branches in execution order (`attn_0, mlp_0, ..., mlp_11`) plus
the readout, so **25 consumers** and `25*26/2 = 325` trainable scalars. The readout consumer's mix
goes through `norm -> pool -> fc_norm -> head`, exactly where the vanilla stream does.

`a = softplus(theta)` with `theta = log(expm1(1))`, so every coefficient is **exactly 1.0** at
init: the routed model *is* the vanilla model at init, and the router draws no RNG, so a routed
model built from the same seed shares the baseline's backbone bit for bit. Coefficients are
non-negative and never normalised (no softmax, no division by the number of sources). Exact zeros
come from `router.mask`, which multiplies the activated coefficient so a masked theta stays in the
graph with an exactly-zero gradient — DDP never sees an unused parameter. `--routing_bound A > 1`
switches to `A * sigmoid(theta)`, also 1 at init.

The thetas are 1-D, so `optim_factory.get_parameter_groups` puts them in the `no_decay` group at
the backbone LR with no edit to the optimizer.

## Two implementations

`--routing_impl loop` accumulates each mix through autograd in the vanilla summation order;
`buf` keeps the branch outputs in one contiguous buffer and computes each mix with a Triton
weighted row-sum, deferring each source's gradient until every consumer that reads it has run.
Same parameters, same checkpoints, same math.

**The run uses `buf`**: +22% per fwd+bwd micro-step against the eager baseline, versus +124% for
`loop`. `buf` is also the better-conditioned path under this repo's fp16 AMP — its mix and
backward accumulate in fp32 and cast once at the end, while `loop` performs a genuine fp16
multiply per edge.

`buf` requires exactly one backward per forward (no `create_graph`, no `retain_graph` + second
pass, no backward rooted at an intermediate — all raise) and no forward hooks on branch modules
during training. Gradient checkpointing is refused outright.

## Gate (job 29579068, one H200, per-GPU batch 128, ImageNet val probe)

`routing/gate.py`; artefacts in `kempfe/routing/gate/{gate.json,gate.png}`.

| check | result |
|---|---|
| weights | 25 consumers, 325 scalars, 26 router keys, every shared tensor `torch.equal`, `max abs(C-1) = 0` |
| fp32 logits | **loop bit-identical to baseline**; buf 4.4e-06 |
| fp16 AMP logits | **loop bit-identical under autocast too**; buf 2.0e-03 |
| fp32 gradients | **loop bit-identical (0.0)**; buf 7.1e-06 |
| fp16 gradients | loop 1.6e-03, buf 1.4e-03 — both ~1.5x fp16 eps (9.8e-04) |
| loop vs buf | 7.6e-06 max relative, real Triton kernels (reference measured 6e-06) |
| init profile | all 24 branch outputs equal to the baseline's; ratios 0.141..0.678 |
| fp16 range | 0 skipped optimizer steps on any arm; max abs(mix) = 0.01% of fp16 max; finite at 2x coefficients |
| cost | base 77.2 ms / 18.7 GB, loop 173.2 ms (+124%) / 20.5 GB, **buf 94.4 ms (+22%) / 22.2 GB** |

Two notes on the gradient rows. The fp32 result is stronger than the reference repo asserts: with
every coefficient exactly 1 and the same summation order, the routed backward reproduces the
baseline's gradients bit for bit. The fp16 row is **not** a correctness statement — under autocast
the matmul backward itself runs in half precision, so a mathematically identical but
differently-ordered backward lands ~1 ulp away. The first gate run tripped a 1e-6 bound at
1.61e-03 for exactly this reason; the check was split rather than loosened.

The framing that the summed gradient is a new fp16 overflow risk does not survive contact: with
`a == 1` the sum over consumers is *numerically identical* to the vanilla residual-stream
gradient, since the chain rule already accumulates it. The measured headroom (0.01% of fp16 max)
and the zero skip-step difference confirm it.

## What the routed arm logs

`Block.forward` never runs, so `utils.HookCollector` — which hooks the Block and reconstructs
`acts[i]['inp'/'attn'/'blk']` — cannot work; both collectors now refuse a routed model outright,
and `utils.RoutedHookCollector` + `engine.routed_layer_metrics` take over. The split is not a
matter of effort:

* **Unchanged**: the attention-map family (`attn_entropy`, `cls_attn_entropy`, `attn_mad`) —
  it depends only on `acts[i]['attn_map']`, which routing does not touch.
* **Exactly the same quantity as the baseline**: `attn_delta_norm_layer{i}` and
  `mlp_delta_norm_layer{i}` are `history[2i+1]` and `history[2i+2]`, i.e. literally the tensors
  the vanilla path calls `attn_out` and `blk - attn`.
* **Same definition, new denominator**: the two `*_delta_norm_ratio` series are still branch
  output over that branch's input, but the input is the consumer's mix rather than the block
  stream. They coincide exactly while every coefficient is 1 and drift apart as the coefficients
  move, so `attn_mix_norm_layer{i}` / `mlp_mix_norm_layer{i}` are logged alongside and the
  denominator stays recoverable.
* **Absent**: `blk_acc`, `attn_acc`, `ece`, `blk_act_norm`, `blk_act_rms`, `mlp_blk_act_norm`,
  `attn_blk_act_norm`, `inp_norm`. Every one is defined on "the tensor entering / leaving block
  i". Routing has no such tensor: block i's two consumers read *different* mixes and
  `mixed_{2i+2} != mixed_{2i+1} + history[2i+1]`. They would agree with the baseline at init and
  then silently come to mean something else, which is worse than not having them.

`test_stats` (`loss`, `acc1`, `acc5`) is identical either way, so `log.txt` — the authoritative
accuracy source — stays schema-compatible with the already-trained base arm. The routed arm adds
`routing/{mean,min,max,emb_col_mean,prev_diag_mean,readout_row_mean,n_edges}` per epoch, the full
25x25 coefficient matrix in `routing.jsonl`, and `amp_scale`.

## Smoke: the routed model tracks the baseline exactly at init

Run on the real 300/50 schedule, bounded by `--time` (job 29581444, 11 epochs), against the base
arm at the same seed:

| epoch | base loss | routed loss | delta | base acc1 | routed acc1 |
|---|---|---|---|---|---|
| 0 | 6.8686 | 6.8686 | -0.0000 | | |
| 4 | 6.2265 | 6.2239 | -0.0026 | 14.140 | 14.238 |
| 9 | 5.5949 | 5.5853 | -0.0096 | 32.350 | 33.188 |
| 10 | 5.4691 | 5.4526 | -0.0166 | | |

Epoch 0's training loss is **identical to four decimals on a real 4-GPU run** — the routed model
is the baseline at init, as designed — and the gap then grows smoothly and monotonically as the
coefficients move off 1. Nothing else misbehaved: `train_grad_norm` finite throughout
(0.911 -> 1.671), zero NaN/inf, and `amp_scale` monotone *non-decreasing* (65536 -> 131072 ->
262144), which only happens after thousands of consecutive overflow-free steps. Coefficients
drifted gently: `routing/max` 1.004 -> 1.58 over 16 epochs.

The resume rehearsal (job 29585630) restored 26 router tensors and continued without a
discontinuity: epoch 11's loss step (-0.1249) sits among the preceding steps (-0.133, -0.153,
-0.133) and its delta to base (-0.0098) matches the pre-resume trend. The half-precision
`-model.pth` dump keeps its 26 router tensors in fp32 while the backbone is fp16, as intended.

These early deltas are **not** a result: 11 of 300 epochs, one seed. The result is epoch-299
test top-1.

## Earlier false alarm: the warmup, not the routing

The first 5-epoch smoke hit a non-finite loss in epoch 2. It had compressed warmup from 50 epochs
to 1 while keeping `lr 2e-3`. The controlled re-run settles it: the **vanilla** arm at the same
compressed schedule diverges too (job 29579323, `grad_norm nan` in epoch 1), so this is the LR
schedule and not residual routing. The coefficient matrix at the point of failure was
unremarkable in any case — mean 1.01, range [0.65, 1.66], readout row all ~1.

There is no way to smoke-test "the first 5 epochs of the real recipe" by lowering `--epochs`:
`utils.cosine_scheduler` asserts `len(schedule) == epochs * niter_per_ep`, so `warmup_epochs`
must fit inside the run (job 29579324 died on that assert in 21 s). The smoke therefore runs the
real 300/50 schedule and is bounded by `--time`, ending in TIMEOUT; re-submitting it against the
same output directory is the `--auto_resume` rehearsal.

## The experiment

The base arm already exists: **`r`, slurm 29384839** — random-init `vit_base`, 300 epochs,
seeds 0/1/2 at **78.284 / 78.040 / 77.912** top-1 (mean 78.08, sd 0.19). The routed arm is that
same recipe at seed 0 with `--resid_routing true --routing_impl buf` and nothing else changed
(`routing/sbatch/train_route.sh`). A delta needs to clear roughly ±0.4 to mean anything against a
3-seed baseline spread of 0.19.

**Run**: `route_29590809` (seed 0, `routing_impl buf`), outputs under
`kempfe/routing/runs/route_29590809/s0`, driven by `routing/sbatch/chain.sh`. ~6 min/epoch on
4x H200, so ~30 h over 2-3 segments.

Transient-failure note: segment 1 died at epoch 21 with a `/dev/shm` allocation failure inside a
DataLoader worker on rank 1 -- the traceback never leaves torch's collate path, so it is a node
issue, not the model. The chain driver now retries a segment that made progress and stops only
when a segment logs no new epochs at all.

**Interim, epoch 202 of 300 (2026-09-13).** The routed arm tracked the baseline to within
noise through epoch ~110 (mean delta over the first 20 eval points: -0.007), then fell behind
systematically. Every eval point since epoch 114 is negative, and the gap widens:

| epoch | routed acc1 | base acc1 | delta |
|---|---|---|---|
| 99 | 75.04 | 75.14 | -0.10 |
| 149 | | | -1.05 |
| 169 | | | -1.96 |
| 199 | 75.18 | 77.26 | **-2.08** |

That is 5-6x the base arm's 3-seed spread (0.37), so it is not noise. The signature is
**overfitting, not instability**: routed *training* loss is lower than base (2.575 vs 2.723 at
epoch 202) while test accuracy is lower, `amp_scale` never once backed off in 202 epochs, and
every gradient norm matched base's. The router used its freedom heavily -- `routing/max` 6.62,
`routing/min` 0.003, embedding column mean 0.29, readout row mean 0.64 -- and the accuracy gap
grew as the coefficients moved away from 1.

Final epoch-299 numbers and heatmaps: *pending* (segment 3 running).

## Risks being watched

* **Unbounded coefficient drift.** The reference repo clips gradients at 1.0; this repo runs with
  `--clip_grad None`, and the thetas sit in the `no_decay` group, so nothing pulls them back
  toward 1. Watch `routing/max` and `routing/min` over the first 50 epochs; if `routing/max`
  exceeds ~5, reconsider `--routing_bound` (the bounded parameterisation is also exactly 1 at
  init). This is the largest scientific risk in the port.
* **GradScaler skip-rate divergence** would silently change the number of effective updates;
  `amp_scale` is logged for both arms and the gate measured zero difference at init.
* **`grad_norms.json` totals are not comparable across arms** — theta gradients are reductions
  over 19.4M elements. Compare per-bucket, never the total.
* **Downstream analysis tooling assumes the vanilla block loop.** `probes/extract_features.py`
  (on the `probes-lens` branch) re-implements it and would produce plausible but wrong features
  for a routed checkpoint; it needs its own routed walk before any probe/lens work on this arm.
* Consumer 1's coefficient gradient is **structurally ~0** — its mix has a single source, so the
  coefficient only rescales the input of `norm1`, and LayerNorm is scale-invariant. Expected, not
  a bug; `tests/test_residual_routing.py` records it.
