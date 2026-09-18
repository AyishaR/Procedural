# Post-hoc training dynamics: when do intermediate predictions become confident, and what do the late blocks do afterwards?

Handoff: `docs/posthoc_training_dynamics_handoff.md`. Code: `dynamics/` (+ small backward-compatible edits to
`probes/registry.py`, `probes/extract_features.py`, `probes/lens.py`), branch `training-dynamics`. Outputs:
`/work/dlc2workfs3/schrodi-procedural/kempfe/dynamics/` (`results/<stage>/`, `figures/`, `logs/`, `subsets/`).
Status: **implementation 2026-09-18; smoke and production pending** (this file is filled in as results land).

## 0. What can and cannot be measured

The training script writes a weights-only fp16 dump after every epoch (`checkpoint-<e>-model.pth`, e = 0..299 =
the 0-based index of the epoch just finished) and keeps the last three full fp32 checkpoints (297-299). On
2026-08-23 the collaborator pruned the per-epoch dumps of the older runs: the canonical **r** (29384839), **p**
(29377576), **ftb3b** (29388202/29406778/29406779) and **ftbrho** (29453944/29461252/29461253) keep only epochs
298-299. For those arms the training-time analysis survives (`results/imnet_base/accuracy_IMNET_BASE_<sid>.json`:
the raw CLS logit lens through the checkpoint's own norm+head on the full 50k val set at init and epochs 4, 9,
14, 19, 29, ..., 299; block activation norms; write ratios; for r/p/ftb3b the epoch-averaged training gradient
norm per block) plus `log.txt`. Experiment 1's raw-lens part is therefore available for the canonical arms;
experiments 2-4 need weights and run on **substitute runs with complete ladders** (decision 2026-09-18: no
retraining until the analysis shows patterns worth the GPU-days; dumps are read in place, never copied).

### 0.1 Runs

Proxy quality was measured on experiment 1's own quantity: the training-time raw-lens trajectory over blocks
b5-b11 and epochs 19-299, as the mean absolute difference (MAD) in top-1 points from the reference arm's seed
mean. Scale: r's seeds differ from each other by 2.4-6.0 points, p's by 1.7-2.2, ftb3b's by 0.7-2.7; r vs p = 24.4.

| role | arm | slurm id / seeds | MAD vs r / p / ftb3b | what differs from the canonical arm |
|---|---|---|---|---|
| random-like | `ftblrm` | 29545846 s0 | 5.8 / 18.6 / 19.0 | plain timm init; proc-matched per-tensor lr; k-bias hand-patched after file 227 |
| random-like | `ftbqu` | 29518360 s0-2 | 5.1-7.1 (epochs 189+) / 9-13 / 7-12 | random weights, q/k x2.18 in blocks 0-8; historical JSON only from epoch 189 (s0/s1), one epoch (s2) |
| random-like | `ftbrhos` | 29538140 s0 | 5.9 / 25.8 / 26.7 | front (0-8) rescaled to a shuffled proc target, 75.1 % final; k-bias patched after file 271; update_freq 4 |
| proc-like | `ftb4` | 29547831 s0 | 25.8 / 2.1 / 6.7 | plain proc init, all 12 blocks, from the kdyck-shuffled ckpt pr_6463456 (p used pr_6066174) |
| proc-like | `ftb4i` | 29448854 s1, s2 | 26-28 / 3.1-4.6 / 6.2-6.8 | proc 0-7 (pr_6066174) + random 8-11 |
| proc-like | `ftb4is` | 29547835 s0 | 24.9 / 3.1 / 5.6 | ftb4i recipe on the kdyck-shuffled ckpt (own arm) |
| intermediate | `ftbqmlnvo` | 29523316 s0-2 | 11.5-16.3 / 8.1-13.0 / 9-14 | proc marginals rank-mapped into 0-8 (section-0e arm) |
| rescaled | `ftbrhopl` | 29632673 s0 | 24.8 / 6.9 / 2.0 | checkpoint-free ftbrho lever + per-tensor lr multipliers (train_lr peaks 0.163), trained in bf16; block-11 weight norm ~9.6k |
| rescaled | `ftbrhop` | 29626634 s0 | 27.6 / 6.3 / 3.8 | checkpoint-free ftbrho lever (random, 9-11 proj/fc2 amplified to write ratio 1.4) |
| rescaled | `ftbcomp1` | 29498148 s0-2 | 23-24 / 5.1-5.9 / 2.9-3.6 | lever at ratio 0.25 + proc block 0 copied |
| history only | `r`, `p`, `ftb3b`, `ftbrho` | 3 seeds each | reference | training-time logs only |

All 17 seed dirs: 300 dumps, `log.txt` ends at 299, fp32 checkpoints 297-299. Shared recipe: lr 2e-3, 50-epoch
linear warmup, cosine to 1e-6 at epoch 300 (per iteration, 312 it/epoch), AdamW wd 0.05, batch 4096, mixup
0.8 / cutmix 1.0 / label smoothing 0.1, drop-path 0, no EMA, fp16 AMP (ftbrhopl: bf16). Every run was resumed
at least once. `vit_base` evaluates the val set every 5th epoch only.

### 0.2 Conventions

- **Depth** as in §0e of `docs/i100_late_block_scaling.md`: depth 0 = patch+pos embedding (`emb`), depth d =
  output of block d-1 (`b0`..`b11`); the after-attention state of block L is `aL` (the historical
  `layer_sub = L - 0.5` row). 25 half-depths in total. **Readout = CLS only**, `head(fc_norm(norm(h[:,0])))`
  with `fc_norm` = Identity, i.e. exactly the checkpoint's own classification path; slot `b11` is the model's
  output and is checked against the model forward on every checkpoint.
- **Epochs**: `E` = "after E epochs of training" (the handoff's numbering) ↔ file `checkpoint-(E-1)-model.pth`.
  Core grid E = 1, 2, 3, 5, 10, 20, 40, 80, 150, 225, 300; dense grid = every epoch over E = 1..20 (the expected
  transition window) ∪ every 5th file (4, 9, ..., 299 = the historical grid) ∪ core (76 files); bypass on the core
  grid + loss-matched epochs + the files bracketing each run's raw-lens transitions (±2); tuned lens / probes at
  E = 1, 3, 10, 40, 150, 300 plus the loss-matched epochs. The canonical r / p / ftb3b / ftbrho seeds get an
  endpoint pass on their surviving files 298-299 (raw lens, bypass, gradients, the single 298→299 pair).
  E = 0 (init) only where a verified init dump exists (ftbqu_s0, ftbqmlnvo_s0, ftbrhop_s0).
- **Sets**: `val` = the frozen probes `imnet_val` (20/class held out of the probe train subset, i.e. images the
  models were trained on → "training-set" numbers); `test` = the official 50k val ("validation" numbers);
  `diag_train2k/diag_test2k` (2/class) for the function-change measure; `diag_train4k/diag_test4k` (4/class,
  ≈ one optimizer batch) for gradients. All frozen with sha256 under `subsets/`; the ImageNet splits are the
  section-0e ones (identical sha). Eval transform = the training eval transform (Resize 256 bicubic,
  CenterCrop 224, ImageNet mean/std); one centre crop, no flip.
- **k-bias**: `attn.qkv.bias[768:1536]` (the key bias) adds q·b_k to every attention logit of a query, a
  constant over keys, so the softmax is exactly invariant to it. AdamW random-walks it (no gradient, no weight
  decay in the no-decay group, Adam normalisation), reaching norms of 90-680 in block 10 of every random-init
  run; three runs were hand-patched mid-run (`checkpoint-<e>.pth.orig_kbias`: ftblrm 227, ftbrhos 271,
  ftbrhopl 270 and 299). It is zeroed at load in every dynamics stage (recorded per checkpoint) and excluded
  from parameter norms and weight deltas.
- **Precision**: the probes feature cache stays bf16 (comparability with §0e). Raw lens and bypass run an fp32
  stream with TF32 matmuls; function change and gradients run strict fp32. bf16 deviates from fp32 by
  1e-2..1e-1 relative in the late stream, the size of a 0.9-gain effect, so no paired loss difference is ever
  taken in bf16. The smoke records the fp32 / TF32 / bf16 floors of the full-model loss.
- **fp16 dump floor**: a single dump's rounding is 2.05-2.08e-4 relative per block (measured on
  ftbqmlnvo_s0), equal to the true fp32 weight change at e298→299 (1.9-2.5e-4). The pairs 297→298 and
  298→299 are therefore computed from the fp32 checkpoints; fp16 values below 3x the measured floor are
  censored.
- Historical per-block training gradient norms reproduce two quirks of the training code's grouping (the q/k/v
  slices are added next to the whole qkv tensor, so totals count qkv twice; the per-level lambdas close over
  the loop variable, so no per-block bucket was ever written). `dynamics.common.hist_style_grad_norms`
  documents both; our own `grad_norms` are plain.

### 0.3 The four measurements

1. **Raw lens over training** (`dynamics/rawlens.py`, dense grid): per image and half-depth top-1/5, hard and
   label-smoothed CE, correct-class margin, confidence, agreement with the final prediction; on `val` and
   `test`. Historical raw lens of the canonical arms parsed by `dynamics/hist.py` into the same schema and
   validated against the recomputation where both exist. Tuned lens (CLS, KL to the checkpoint's own final
   distribution, step chosen on val) and linear probes (CLS / mean-patch / concat, standardised, (wd, step) per
   depth chosen on val) via the unchanged probes pipeline on epoch entries (`dynamics/lens_probe.py`).
2. **Block usefulness** (`dynamics/bypass.py`, core + matched epochs): block l replaced by its gain-scaled
   version (gain 0 = bypass = identity; 0.9 = attenuation of both residual branches), blocks l+1..11 recomputed
   from the cached prefix. `C_l(E) = L_bypass − L_full` (hard CE; also smoothed), accuracy lost, images
   corrected / harmed by the block; last-2 / last-3 bypass = readout on b9 / b8. Plus the exact local
   sensitivity dL/dg of each image's loss to each of the 24 branch gains at g = 1.
3. **Block-function change** (`dynamics/funcchange.py`, all 299 one-epoch pairs): on the epoch-e stream,
   `U_l(e) = mean ||F_{l,e+1}(h) − F_{l,e}(h)|| / (||h|| + ε)` for CLS, patch tokens and the whole sequence;
   attention sublayer alone (same h) and MLP sublayer alone (same h + attn_e(h)); write ratio ||F_e||/||h||;
   relative change ||ΔF||/||F_e||; the relative late share Σ_{l≥9} U_l / Σ_l U_l (a compositional quantity, read together with the absolute U and the lr schedule, never as an lr-invariant measure); weight-space Δθ/θ per block; the fp16 floor and, for the two fp32 pairs, the fp16-dump variant (actual two-checkpoint rounding error).
4. **Gradient snapshots** (`dynamics/grads.py`, dense grid): exact full-batch gradient on `diag_train4k` /
   `diag_test4k` under CE with label smoothing 0.1 (the training criterion without mixup/cutmix; q_y = 0.9001)
   and plain CE; per-block / per-tensor norms, relative gradient ||g_l||/||θ_l||, late share, mean p_y,
   p_y − q_y, ||p − q||₁, fraction with p_y > q_y. Newly evaluated gradients at saved weights, not the
   historical training gradients.

### 0.4 Sequence statistics, loss matching and the verdict rule (frozen before production, after design review B)

All timing statistics are computed on sampled epochs and come with a bracket: the crossing epoch `E` and the
previous sampled epoch `E_prev`; the event lies in `(E_prev, E]`. Two events are ordered only if their brackets
do not overlap; otherwise the order is **unresolved at the sampling resolution**. Differences already present at
E = 1 are reported as "unresolved within the first epoch".

- **Confidence time** `t_conf(d, τ)`: the smallest sampled E from which the raw-lens top-1 at depth d (test set)
  stays ≥ τ · (the same run's terminal top-1 at the same depth d) at every later sampled epoch, τ ∈ {0.5, 0.9},
  d ∈ {b8, b9}. Accuracy is a milestone, not confidence; the confidence variants use the measured quantities:
  median correct-class margin at d > 0 (sustained), mean max-prob at d ≥ 0.5 · terminal (sustained), and the
  final-output target error mean ‖p − q‖₁ under the smoothed target. A confidence-led statement requires the
  margin/max-prob variants, not the accuracy milestone alone. Reference time for the before/after splits:
  `t_conf(b9, 0.5)` (b9 precedes the last two blocks; b8 precedes the last three).
- **Usefulness plateau** `t90(C_l)`: the smallest sampled E such that |C_l(e) − C_l(300)| ≤ 0.1 · |C_l(300)| for
  every sampled e ≥ E (last crossing; re-entries reset it), for l ∈ {9, 10, 11} and the last-3 bypass. The
  fraction of terminal usefulness acquired after `t_conf`, `[C_l(300) − C_l(t_conf)] / C_l(300)` with log-E
  interpolation, is reported with its signed numerator and is **undefined** when |C_l(300)| < 0.02 (CE units).
  Both hard and smoothed CE, the accuracy lost, the corrected/harmed counts and the local attenuation (gain 0.9
  and dL/dg) must agree in sign before a change is read as reduced usefulness.
- **Functional change**: `U_l` values from fp16 pairs below 3× the run's measured per-block fp16 floor are
  censored; every derived quantity (cumulative path `A_l`, late share, plateau `t90(A_l)`) is reported twice, with
  censored terms set to 0 (lower) and to the floor (upper); a conclusion that differs between the two is
  unresolved. `A_l` accumulates the observed E = 1→300 intervals only (within-epoch cancellation is invisible;
  the init→E=1 movement is included only where an init dump exists). The late share is a relative measure;
  it is read next to the absolute late and early U, the lr schedule (50-epoch warmup, cosine decay) and the
  per-tensor multipliers of ftblrm / ftbrhopl. Shares before/after `t_conf` are ratios of sums.
- **Gradients** (supplementary): the same before/after ratio of summed squared per-block norms; the norm of the
  full-batch mean gradient can be small through cancellation, so the per-64-image chunk norms are reported next
  to it. Neither the gradient share nor ‖g_l‖/‖θ_l‖ removes the rescaling confound; they are not a stall test.
- **Loss matching**: level grid = the random-like role's seed-mean monotone (running-min) hard CE on the official
  val set at the core epochs; for every run and level the *evaluated* checkpoint (dense grid) whose monotone CE
  is closest is selected; the achieved CE, its relative deviation, E and lr are recorded; a match is valid within
  5 %; the primary matched comparison uses only the levels every run matches (`common_levels`); the selected
  files are then measured by bypass / lens / probes. Sensitivity: the same matching on the smoothed CE of the
  train-split set is reported. Neither is the historical mixup/cutmix training objective.
- **Uncertainty**: within-recipe seed variation only for the 3-seed arms (ftbqu, ftbcomp1, ftbqmlnvo; ftb4i has 2);
  example-level uncertainty by bootstrapping images jointly across epochs from the stored per-image arrays; the
  two are never pooled. Fixed-cohort analyses (images selected by earlier intermediate correctness / margin,
  then their later tail contribution and corrected/harmed outcomes) test whether aggregate timing reflects a
  changing image composition. Diagnostic-set stability: U and gradient profiles on `diag_train2k/4k` vs the
  independent `diag_test2k/4k`.
- **Verdict rule**, applied separately to (i) the canonical historical evidence (raw lens only, plus the 298-299
  endpoint pass) and (ii) each substitute recipe: **supported** = confidence brackets precede the late-block
  usefulness / functional-change plateaus with non-overlapping brackets in the random-like recipes, the ordering
  is absent or reversed in the proc-like / rescaled recipes, and the late-block change after `t_conf` is small
  relative to the early blocks' at matched loss (not only at matched epoch, i.e. not explained by global
  convergence); **weakened** = the late blocks keep acquiring substantial usefulness (C_l) and functional change
  after confidence in the random-like recipes, or the proc-like / rescaled recipes show the same sequence;
  **unresolved** = overlapping brackets, conclusions that flip under censoring bounds, under hard vs smoothed
  objective, or under epoch vs loss matching, or a bridge to the canonical arms that the endpoint pass does not
  support. A positive sequence never establishes suppression or causal mediation; the substitute design cannot
  establish that canonical r developed the sequence or that canonical p avoided it, only that the sequence
  occurs (or not) in recipes whose readout dynamics resemble theirs.

Implementation: `dynamics/summarize.py` (`sequence_stats`, `matched_epochs`, `transition_epochs`).

## 1. Smoke and validation (2026-09-18, jobs 29733757 / 29733764 / 29733765 / 29733775, `dynamics/sbatch/smoke.sh`, one H200)

Every stage ran end to end on `ftbqmlnvo_s0` (files 0, 149, 299 and the init dump, 512-image `--limit`) plus
the numerically extreme cases `ftblrm_s0` file 200 (k-bias norm 676 before zeroing) and `ftbrhopl_s0` file 299
(block-11 weight norm 9.6k). Findings:

- **Readout identity.** Slot `b11` of the raw lens equals the model's own forward bit for bit (top-1 agreement
  1.0000, max |Δlogit| = 0) on every checkpoint and precision.
- **Historical lens reproduced.** Full-size (50k) recomputation vs the training-time `accuracy_*.json` lens at
  files 99 and 299: max |Δtop-1| over the 24 historical half-depths = 0.06 pt (ftbqmlnvo_s0) / 0.06 pt
  (ftblrm_s0), mean 0.01; historical `b11` vs `log.txt` test_acc1 within 0.04 pt over 32 epochs. The
  canonical arms' historical heatmaps are therefore on the same footing as the recomputed ones.
- **Precision floors** (2048 test images, full-model CE at `b11`; max over all 25 half-depths in brackets):
  |L_tf32 − L_fp32| = 1.3e-5 … 1.9e-4 [≤ 2.4e-4]; |L_bf16 − L_fp32| = 1.4e-4 … 8.4e-4 [≤ 2.5e-3, the ftblrm
  file-200 case with the k-bias still present]. Smoke bypass effects at gain 0.9 are 5e-3 … 3e-2 in CE, so the
  TF32 floor is ≥ 25× below the smallest of them; bf16 would be within a factor 5. Raw lens and bypass run at
  TF32, function change and gradients at strict fp32, as planned.
- **fp16 dump floor** (`funcchange --floor`, 256 images/set, fp16 dump vs fp32 checkpoint of the same epoch):
  U_cls = 4e-5 … 9e-5 in blocks 1-8, 1.5e-4 … 3.5e-4 in blocks 0 and 9-11 (ftbqmlnvo_s0); 5e-4 … 9e-4 in
  blocks 10-11 of ftbrhopl_s0. True fp32 pair changes at 297→298 / 298→299 are 1.1e-4 … 2.5e-3, i.e. 1-3× the
  floor; at pair 149 the change is 2e-2 … 6e-1, ≥ 100× the floor. The last two pairs are taken from the fp32
  checkpoints and fp16 values below 3× the floor are censored, as planned.
- **k-bias.** Zeroing is applied through the entry flag (`dynamics.registry.entry` marks every run; the
  section-0e entries stay untouched for the probes pipeline) and recorded per checkpoint: ftbqmlnvo_s0 file 299
  removes norms 6-27 in blocks 0-9 and 137 in block 10; ftblrm_s0 file 200 removes 135 / 281 / 322 / 680 in
  blocks 7 / 8 / 9 / 10. With the k-bias removed the precision floors on that checkpoint drop to
  |L_tf32 − L_fp32| ≤ 8e-5 and |L_bf16 − L_fp32| ≤ 2.1e-3 (max over half-depths).
- **Sanity of the measures** (ftbqmlnvo_s0, 512 test images): bypassing block 0 destroys the model
  (C = 6.8, all 512 images lost); at file 299 bypassing block 9 costs 3.0 CE / 37 pt, block 10 1.0 / 9 pt,
  block 11 0.46 / 2.5 pt; attenuating blocks 10-11 by 10 % *lowers* the test CE (C(0.9) = −0.02 / −0.03) and
  the exact gain gradient agrees in sign (dL/dg = +0.08 / +0.39 for their MLP branches). Gradient late share
  (blocks 9-11) rises from 0.19 at file 0 to 0.44 at file 299; 7 % of training images have p_y > q_y at the end.
- **Throughput** (H200): raw lens 1.5-1.8k img/s with 2 resident checkpoints (decode-bound); bypass ~120 img/s
  per checkpoint including both gain sweeps and two gain-gradient backward passes; function change ~2.5 s per
  pair (256 images/set); gradients ~6 s per checkpoint (512 images/set × 2 sets × 2 objectives).

Bugs found and fixed by the smoke: a missing `no_grad` in the raw-lens batch loop; the function-change
accumulator adding every block's value to all 12 slots; the k-bias flag missing on the three section-0e runs.

## 2. Results

_pending_

## 3. Assessment

_pending_

## 4. Reviews

- **Codex A1 (implementation, core modules; 2026-09-18).** 9 findings, all applied: top-1 must be the argmax
  (tie convention), not the first of `topk`; the precision context must switch inherited autocast off and use
  `torch.set_float32_matmul_precision` rather than the legacy TF32 flag; per-chunk gradient norms are computed from
  each chunk's own gradient instead of by subtraction; the lens cache-agreement assert keeps a hard floor (0.9
  for file < 20, 0.98 otherwise) instead of a blanket warning; the section-0e entries of ftbqmlnvo / ftbrhos /
  ftblrm now get the k-bias flag through `dynamics.registry.entry` (the smoke had found the same); tests extended
  (chunk norms vs independent gradients, argmax ties, precision restoration, a mocked checkpoint load with
  k-bias exclusion, dyn-only exclusion from `resolve_models("all")`).
- **Codex B (experimental design; 2026-09-18).** Must-fix items and how they were resolved before production:
  "lr-invariant" late share renamed and treated as a relative, compositional quantity read next to absolute U
  and the lr schedule; censoring propagated into shares, cumulative paths and plateau times as lower / upper
  bounds, plus the fp16-dump variant of the two fp32 pairs as the actual two-checkpoint rounding error; loss
  matching on the achieved CE of evaluated checkpoints with a 5 % tolerance, recorded membership and a common
  level set for the primary comparison; canonical and substitute conclusions separated in the verdict rule, with
  an endpoint pass on the canonical files 298-299 as the bridge beyond the lens; sequence statistics fully
  defined (sustained crossings with brackets, terminal reference at the same depth, defined usefulness
  fractions, ratio-of-sums shares, measured confidence variants); every epoch over E = 1..20 added to the dense
  grid and ±2-file refinement around each run's transitions; the supported / weakened / unresolved rule
  prespecified (section 0.4). Recommended and adopted for the analysis: cohort analyses and image bootstraps
  from the per-image arrays, subset stability via the independent diagnostic sets, chunk-norm vs full-batch
  gradient comparison, smoothed-CE matching as a sensitivity check.
- **Codex A2 (implementation, stage modules; 2026-09-18).** 16 findings, applied: the canonical-run summary
  lookup that would have crashed the endpoint pass; NPZ written atomically before the JSON marker and stale
  markers removed before rebuilding; corrected/harmed counts rounded instead of truncated; gradient target
  statistics use the objective's own q (one-hot for hard CE); `epoch_entry` no longer mutates the probes
  registry; feature caches are re-extracted when the checkpoint file's mtime changed; summaries de-duplicate
  (run, file, split, tag) records and filter precision tags; incomplete series are flagged `complete=False`;
  function-change intervals are dated by their end epoch and split at `t_conf` by interval endpoints; all
  plateau times carry brackets; figures average seeds by epoch identity, merge the historical lens with the
  recomputed endpoints for the canonical arms, and apply the summary's censoring rule.
- **Codex C (results analysis):** _pending_.
