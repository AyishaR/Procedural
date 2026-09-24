# Handoff: progressive layer unfreezing for a ViT-S / ImageNet-100 run

Port the "progressive layer unfreezing" intervention from the layer-scale repo to a ViT-S
trained on ImageNet-100 and run four arms against the unmodified model. The reference
implementation is complete, reviewed (Codex), gated, smoke-tested at 124M/350M and has
produced results on ClimbMix; reuse it rather than re-deriving it. The model / dataset /
recipe side is yours; this note fixes the intervention, the arms, the checks and the
comparison so the two experiments stay comparable.

## Read these first (layer-scale repo, `main`)

- `nanoGPT/unfreeze.py` -- the module to copy. Pure parts: `partition`, `unfreeze_iters`,
  `lr_mult`, `parse_arm` / `arm_tag` / `train_flags`. Bound part: `UnfreezePlan`
  (`from_model`, `param_groups`, `build_optimizer`, `apply_lr`, `mask_grads`, `is_event`,
  `summary`, `wandb_scalars`, `describe`). Only `param_chunk(name)` (which parameter belongs
  to which block, which are "always trainable") is model-specific.
- `nanoGPT/train.py`, the six `unfreeze_*` insertions (search for `unfreeze_plan`): knobs,
  guard, optimizer build + resume assert, `apply_lr` at the top of the iteration, wandb
  scalars at eval, `mask_grads` between the backward/all-reduce and the clip/step.
- `tests/test_unfreeze.py` (module contract + the train.py source contract) and
  `tests/test_train_unfreeze_smoke.py` (the trainer end to end on CPU: snapshots pass the
  checker, exact resume, refusals, E = 0 == base bit for bit). Port the first; the second is
  the template for your own smoke.
- `analysis/unfreeze_gate.py` (CPU gate, no GPU) and `analysis/unfreeze_check.py` (post-smoke
  checker on `ckpt_latest_<t>.pt` snapshots). Port the checks, not necessarily the scripts.
- CLAUDE.md, section "Progressive layer unfreezing on ClimbMix c20" -- the design decisions
  and the outcome; results page (figures + table):
  https://claude.ai/artifact/DWDtnmEZV6UPXVd1dTsM1a

## The intervention, precisely

**Chunking.** The `depth` transformer blocks are split into `C = 4` contiguous chunks of
`depth / C` blocks (ViT-S: 12 blocks -> chunks of 3: blocks 0-2, 3-5, 6-8, 9-11). Everything
that is not a block is **always trainable**: for a timm-style ViT that is `patch_embed`,
`cls_token` (+ `reg_token`), `pos_embed`, `norm_pre`, the final `norm` / `fc_norm`, and the
`head`. (The LM analogue was wte/wpe/ln_f.) `param_chunk(name)` must return the block index
for `blocks.<i>.*`, `None` for the always-trainable tensors, and raise for anything else, so
an unexpected parameter (an EMA copy, an auxiliary head) is refused rather than silently
frozen or silently trained.

**Direction.** The chunks are listed in *unfreeze order*: `front` = blocks 0-2 first, then
3-5, 6-8, 9-11 (unfreezing towards the back); `back` = blocks 9-11 first, then 6-8, 3-5, 0-2.

**Schedules** (`T` = total optimizer steps of the run, `E` = fraction of the run at which the
LAST chunk in unfreeze order unfreezes, `u_k` = the step at which chunk `k` unfreezes):

- `lin`: `u_k = round(k / (C - 1) * E * T)` -- chunk 0 trains from step 0, the others come in
  evenly spaced, the last at `E * T`. With `E = 0.2`: `0, T/15, 2T/15, T/5`.
- `one`: `u_k = 0` for `k < C - 1`, `u_{C-1} = round(E * T)` -- only the last chunk in
  unfreeze order is ever frozen; the other three train from step 0. With `E = 0.2`:
  `0, 0, 0, T/5`.

`unfreeze_iters(sched, C, E, T)` is the single source of truth (Python's banker's `round`);
the trainer, the gate, the checker and the plots all call it, nobody re-derives it. Chunk `k`
is frozen at every step `it < u_k` and trainable from `it = u_k` on (the step at `u_k` is its
first update).

**Per-chunk lr ramp.** A chunk with `u_k > 0` gets the global lr times
`min(1, (it - u_k + 1) / (W + 1))` for `it >= u_k` (the shape of a linear warmup), chunks
alive from step 0 get the recipe's own warmup, so `E = 0` is the vanilla run for any `W`. In
the LM the ramp made no difference (0.005 nats at E = 0.5, inside seed noise), so pick
`W` = one epoch's worth of steps or the recipe's warmup length, whichever is shorter, and do
not spend a run on it.

## Mechanism ("cold moments") -- copy this exactly

1. **Every parameter keeps `requires_grad=True` and sits in the optimizer** from step 0.
   Never flip `requires_grad` and never rebuild the optimizer: DDP's reducer registers its
   hooks once at construction (a parameter that is not registered is never all-reduced, and
   the ranks drift apart silently), `torch.compile` guards on `requires_grad`, and
   `Optimizer.load_state_dict` refuses a changed group layout.
2. **Param groups:** one {decay, no-decay} pair per chunk plus one pair for the always-
   trainable tensors (2C + 2 groups), each carrying `chunk` (int or `None`) and `name` keys
   (extra keys survive optimizer state-dict round trips). Use your recipe's own decay rule
   inside each pair (timm: `ndim <= 1` and `no_weight_decay()` names -> no decay).
3. **Freezing = dropping the gradients before the step.** After the backward of the last
   micro-step (i.e. after DDP's all-reduce has written the reduced grads back) and BEFORE
   gradient clipping and `optimizer.step()`, set `p.grad = None` for every parameter of every
   chunk with `it < u_k`. AdamW (fused or not) collects only parameters whose grad is not
   `None` and creates their state lazily in that same branch, so a frozen parameter receives
   no update, no decoupled weight decay and no Adam moments: it stays **bit-identical to the
   init**, and a newly unfrozen chunk starts at Adam step 1 with fresh moments. The clip norm
   then covers the trainable grads only. Belt and braces: also set the frozen groups' `lr`
   to 0 in `apply_lr`. No backward FLOPs are saved (same step time as base).
4. **Do NOT implement it as "lr = 0 without masking".** That keeps the weights put but lets the
   moments accumulate the whole frozen period ("warm moments") -- a different intervention,
   deliberately not run here.
5. **AMP / GradScaler:** with bf16 autocast the scaler is a no-op. With fp16 + GradScaler,
   drop the grads after `scaler.unscale_` would be wrong -- drop them before `unscale_`
   (i.e. right after the backward), which is what the LM trainer does since `unscale_` sits
   inside the clip block. Non-finite checks of the scaler see only trainable grads.
6. **Resume is a pure function of the step counter:** no extra checkpoint state. Refuse a
   resume whose checkpoint carries different `unfreeze_*` knobs or a different `T` (the `u_k`
   depend on it), and refuse a vanilla checkpoint as an unfreeze run (group layouts differ).
7. **EMA of weights:** if the recipe keeps an EMA model, note that the EMA of a frozen chunk
   equals its init (the parameter never moved) and starts tracking only after the unfreeze;
   evaluate both the raw and the EMA model, or disable EMA for this experiment. Layer-wise lr
   decay, if any, composes with the ramp multiplicatively and is fine.
8. **drop_path / dropout / mixup / stochastic depth:** unaffected (a frozen block still runs
   forward and backward, its weights just do not move).

## The four arms (+ the untouched base), `E = 0.2`, `C = 4`

| tag | direction | schedule | frozen at step 0 | unfreeze steps (ViT-S, `T` steps) |
|---|---|---|---|---|
| `u4f-lin0.2` | front | lin | blocks 3-11 (three chunks) | 3-5 @ T/15, 6-8 @ 2T/15, 9-11 @ T/5 |
| `u4b-lin0.2` | back | lin | blocks 0-8 (three chunks) | 6-8 @ T/15, 3-5 @ 2T/15, 0-2 @ T/5 |
| `u4f-one0.2` | front | one | blocks 9-11 only | 9-11 @ T/5 |
| `u4b-one0.2` | back | one | blocks 0-2 only | 0-2 @ T/5 |

Same seed, same data order, same recipe (epochs, lr schedule, augmentations, EMA choice) as
the base; the arms train from scratch with the base's seed so the init is bit-identical to
the base's. If your recipe's `T` is not an integer number of these fractions, `round` decides
(see `unfreeze_iters`); record the resulting `u_k` in the run config.

The tag grammar `u<C><f|b>-<lin|geo|one><E>[-w<W>]` is what `parse_arm` / `train_flags`
implement; keeping it lets the two repos share plotting code.

## Checks (in this order, before the four runs)

1. **CPU gate on a toy ViT (4 blocks, tiny width, `C = 4`, a handful of steps):**
   (a) the frozen chunks are `torch.equal` to the init and have NO optimizer state until
   their `u_k`, then move and carry `state["step"] == t - u_k`; (b) the always-trainable
   tensors move from step 1; (c) the per-group lrs equal global lr times the ramp multiplier;
   (d) `E = 0` (any `W`) is **bit-identical** to the vanilla optimizer + plain lr loop over
   several steps; (e) a save / load mid-run + a fresh plan reproduces the uninterrupted run
   bit for bit; (f) another chunk count or the vanilla optimizer refuses the state;
   (g) `parse_arm` / `train_flags` round-trip. All of this is `tests/test_unfreeze.py`.
2. **GPU smoke through the real launcher** (a few hundred steps, snapshots every N steps via
   a `latest_ckpt_keep`-style hard link), with `E` chosen so every chunk unfreezes inside the
   smoke; then the checker on the snapshots: frozen chunks bit-identical to the seed init
   until `u_k`, Adam state exactly for the unfrozen chunks with `step = t - u_k`, every tensor
   and moment finite, saved group lrs == schedule x ramp, snapshots reach the end and bracket
   every unfreeze, step time == base. Run it once on the DDP configuration you will train on.
3. **E = 0 control on GPU vs a base smoke:** GPU runs are not bit-reproducible (atomics in the
   embedding / attention backward), so compare against a *second* base smoke and require the
   control-vs-base deviation to match base-vs-base at every snapshot (in the LM both grew
   from 2.5e-5 at step 10 to 8e-2 at step 60; identical curves). The bit-exact proof is the
   CPU gate.

## Logging and comparison

- Log at every eval: `unfreeze/trainable_layers`, `unfreeze/trainable_param_frac`,
  `unfreeze/n_unfrozen`, `unfreeze/lr_chunk<k>`; print a line at every unfreeze event.
- Primary metric: paired delta vs the same-seed base at the END of training (same step for
  every run): top-1 on the ImageNet-100 val set, and the val loss. Secondary: the top-1 / loss
  curve with the unfreeze steps marked (the LM curves show every frozen arm briefly *ahead* of
  base at ~5 % of training and behind from ~10-20 % on -- look for the same).
- One seed first for all four arms; the LM effects were 10-60x the seed noise, so a second
  seed is only needed for arms that land within ~1 sd of base (in the LM: only the
  "last chunk frozen" arm).

## What the LM found (ClimbMix c20, GPT-2 124M, paired val loss, final checkpoints)

Every arm was worse than base, monotonically in `E`, and back-to-front cost ~2x
front-to-back at every `E`. The mild arms isolated *which* layers may be frozen:

| arm (E = 0.1) | 2-seed delta (nats) |
|---|---|
| only the last chunk frozen (`u4f-one0.1`) | +0.004 +- 0.005 (null) |
| only the first chunk frozen (`u4b-one0.1`) | +0.043 +- 0.005 |
| front-to-back staged (`u4f-lin0.1`) | +0.048 +- 0.005 |
| back-to-front staged (`u4b-lin0.1`) | +0.129 +- 0.004 |

Full sweep, seed 1338: front E = 0.1/0.25/0.5/0.75 = +0.043/+0.088/+0.139/+0.170, back =
+0.125/+0.189/+0.263/+0.292; the ramp irrelevant (+0.134 vs +0.139). Interpretation: the
cost comes from the early layers being frozen while the rest trains -- later layers and the
embeddings adapt to random early features and do not recover in the remaining run; freezing
late layers early is free. The ViT question is whether the same asymmetry holds with patch
tokens and a classification head, where the "readout" is one token and the early blocks see
a fixed linear patch embedding.
