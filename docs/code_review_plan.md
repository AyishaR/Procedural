# Code review plan: the intervention experiments (prepared 2026-09-22)

Purpose: walk the code of the experiments that carry the paper's claims, with a coworker, one family per session.
Each section says what the experiment is, which arms and scripts realise it, the code path in reading order with
`file:line` pointers, what the verifiers and tests cover, and what to look at in the review. Line numbers refer
to the working tree of 2026-09-22 (HEAD `f2ac050` plus uncommitted changes, see section 0.1); function names are
the stable anchors if lines drift.

Sources: [`docs/proc_init_recipe.md`](proc_init_recipe.md) (recipe, sections 5, 9-15), [`docs/early_lever_mechanism_plan.md`](early_lever_mechanism_plan.md) (mechanism
study, sections 4, 5b, 5e, 5f, 5g), [`docs/i100_synthesis.md`](i100_synthesis.md) (results), and the code itself.

Suggested order (four sittings of about an hour):

| sitting | sections | why this order |
|---|---|---|
| 1 | 0 (common path), 1 (late lever 2x2) | the late lever is the smallest code path and exercises the lr-scaling machinery that the early-lever step-size arms reuse |
| 2 | 2 (freezing), 3 (aux loss, suppression loss) | three small, self-contained additions to the optimizer and the training loop |
| 3 | 4.1-4.3 (early lever: extraction, spec, realisation) | the largest path: [`extract_profile.py`](../extract_profile.py) -> JSON -> `apply_analytic_profile` |
| 4 | 4.4-4.6 (calibration, verifiers, factorial and compose arms), 5, 6 | the rank-one components, what the verifiers prove, and the open items |

---

## 0. Before starting: repo state and the common path

### 0.1 What is uncommitted

Every intervention below except the profile extractor's first version lives in the working tree, not in a commit.
`git status` on 2026-09-22 (outputs and caches left out):

| state | files |
|---|---|
| modified since `f2ac050` (2026-09-17) | [`main.py`](../main.py), [`engine.py`](../engine.py), [`utils.py`](../utils.py), [`optim_factory.py`](../optim_factory.py), [`custom_lr.py`](../custom_lr.py), [`extract_profile.py`](../extract_profile.py), [`plot_reconstruction.py`](../plot_reconstruction.py), [`plots/verify/verify_joint_statistics.py`](../plots/verify/verify_joint_statistics.py), [`plots/verify/wandb_layerwise.py`](../plots/verify/wandb_layerwise.py) |
| new, untracked | [`row_lr_mask.py`](../row_lr_mask.py), [`plots/verify/test_row_lr_mask.py`](../plots/verify/test_row_lr_mask.py), [`plots/verify/test_calibration_guards.py`](../plots/verify/test_calibration_guards.py), `plots/verify/verify_*.py` (compose, late_trio_v, recipe_statement, resume_checkpoint, step_compensation, structure_only), [`plots/verify/make_step_compensation.py`](../plots/verify/make_step_compensation.py), `plots/verify/mechanism_*.py`, `plots/verify/fp16_*.py`, all `vitbase_runs/profile_ftb{ana*b7*,c7*,ck7*,rhoplv}.json`, all `vitbase_runs/lrscale_ftb{c7*,ck7*,rhoplv,rhoslv}.json`, [`docs/proc_init_recipe.md`](proc_init_recipe.md), [`docs/early_lever_mechanism_plan.md`](early_lever_mechanism_plan.md) |

For the review, the diff to read is

```
git diff f2ac050 -- main.py engine.py utils.py optim_factory.py custom_lr.py extract_profile.py
git status --short | grep '^??' | grep -v 'plots/out\|profile_\|lrscale_'
```

Committing the state before the review (one commit per family, or one for everything) would give the coworker
stable line numbers. The launched runs recorded which code they ran only through their start dates (section 15 of
the recipe doc), so a commit also fixes the provenance.

### 0.2 The common training path (every arm goes through this)

| step | where | note |
|---|---|---|
| argument parsing | [`main.py:47`](../main.py#L47) `get_args_parser` | the intervention flags: [`--release_blocks/_start/_end`](../main.py#L342) 342-346, [`--aux_lens_block/_mode/_weight/_until`](../main.py#L347) 347-352, [`--lr_scale_json`](../main.py#L400) 400, [`--profile_spec`](../main.py#L404) 404, [`--init_method`](../main.py#L391) 391, [`--init_method_scaled_blocks`](../main.py#L471) 471, [`--amp_dtype`](../main.py#L369) 369, [`--stop_after_epoch`](../main.py#L339) 339, [`--analysis_ckpt_dense_until/_every`](../main.py#L353) 353-355 |
| per-rank seed | [`main.py:684`](../main.py#L684) `seed = args.seed + get_rank()` | every init edit must therefore be either deterministic in `args.seed` or broadcast afterwards |
| model build | [`main.py:786`](../main.py#L786) [`utils.build_model`](../utils.py#L733) (utils 733-750) | a plain timm ViT-B |
| checkpoint loading + DDP wrap | [`utils.pr_load_model`](../utils.py#L877) [`utils.py:877-1118`](../utils.py#L877) | loads `blocks.*` of a procedural checkpoint, drops random blocks (`--random_blocks`), applies `--freeze_blocks` (1087-1096), wraps in DDP (1113-1118, `find_unused_parameters=False`) and returns `model.module`. Called from the init branches at [`main.py:1047-1072`](../main.py#L1047) |
| init branches | [`main.py:1047-1160`](../main.py#L1047) and [`main.py:1675-2110`](../main.py#L1675) | `analytic_profile` at 1069 (section 4); `upscale_random_match_delta_norms` at 1963 (section 1.4); older edits (shuffle, clip, spectral, copied blocks, mute_mlp) in between |
| rank sync | [`main.py:733-751`](../main.py#L733) `sync_initialisation` | broadcasts every floating-point tensor from rank 0; called at 2117 (after the edits, before the pre-training analyses) and 2163 (before training). This is the fix for the DDP rank-sync bug; log marker `[init-sync] broadcast` |
| EMA re-copy | [`main.py:2118-2122`](../main.py#L2118) | after the init edits |
| optimizer | [`main.py:1576`](../main.py#L1576) -> [`optim_factory.create_optimizer`](../optim_factory.py#L225) 225-358 | the group builders for lr scaling, row masks and release are chosen here (sections 1.2, 2.1) |
| aux loss attach | [`main.py:2178-2179`](../main.py#L2178) | lazily, once, inside the epoch loop (section 3) |
| training loop | [`engine.py:42`](../engine.py#L42) `train_one_epoch` | lr / wd per step 103-117, release factor 106-110, aux loss 129-140, non-finite dump 146, logging 266-267 |
| the lens (block read-out) | [`engine.py:416`](../engine.py#L416) `model_analyse`, computed at 478-499 | `head(fc_norm(norm(block output)))[:, 0]`, logged as `blk_acc_layer{i}`; every mechanism reading ("lens7") is this number. The aux loss reuses the same map (section 3) |
| checkpoint provenance | [`utils.save_model`](../utils.py#L607) 607, [`auto_load_model`](../utils.py#L636) 636 | `lr_scale_spec` written at 618, refused on mismatch at 674-677; `keep_all=True` on every post-training reload ([`engine.py:709, 1095, 1162, 1181`](../engine.py#L709)) since the reload artefact fix of 2026-09-19 |

Run scripts: every `vitbase_runs/run_train_<arm>.sh` is one template (`torchrun ... main.py --model vit_base
--warmup_epochs 50 --epochs 300 --lr 2e-3 --use_amp true ...`); arms differ only in the init and intervention
lines quoted below and in `--notes`. [`vitbase_runs/run_train_r0.sh`](../vitbase_runs/run_train_r0.sh) is the plain random reference. The
verification pattern of every new arm is the same three steps: an init dump through [`main.py`](../main.py)
([`plots/dump_init.py`](../plots/dump_init.py)), an arm-specific verifier on the dump, a two-rank smoke training on the real schedule.

---

## 1. The late lever 2x2 (blocks 9-11: loud write versus step size)

### 1.1 What it is

Multiplying the write matrices of blocks 9-11 of a random init (proj, fc2, in the v-inclusive form also the v rows
of qkv) by fixed factors so that each sub-layer writes 1.4 times its incoming stream. The 2x2 asks whether the gain
comes from the loud write in the forward pass or from the slower relative Adam steps that large weights imply.
Answer: the loud write ([`docs/proc_init_recipe.md`](proc_init_recipe.md) section 13).

| | loud write (weights x m) | random write (timm) |
|---|---|---|
| slow steps (no lr file) | `ftbrhop` 79.93 (fp16) | `ftbrhosl` 78.31: lr x 1/m (fp16) |
| normal steps | `ftbrhopl` 80.13: lr x m (bf16) | random 78.08 +- 0.19 (n = 3) |

v-inclusive twins (row mask): `ftbrhoplv` 80.15 (bf16), `ftbrhoslv` 78.18 (fp16), parent `ftbrho` 79.69 +- 0.30 (n = 3).
Caveats to carry: one seed per cell; the `pl` cells are bf16 after an fp16 overflow at epoch 44; `ftbrhoslv` skipped
fp16 steps in 32 epochs. `ftbrhos` is NOT a cell of this 2x2 (it is an early-block arm: random blocks 0-8 matched
to a shuffled proc target, 75.07).

Measured-profile cell (set up and verified 2026-09-22, launched 2026-09-23: `ftb3bpl` 29765171, `ftb3bb` 29765222, `ftb3bsl` 29765177; `docs/proc_init_recipe.md` section 13, last paragraph):
`ftb3bpl` = `ftb3b`'s init (proc's measured write ratios, attention 1.380 / 2.098 / 0.777, MLP 4.757 / 4.459 / 0.526) replayed
from [`vitbase_runs/profile_ftb3bpl.json`](../vitbase_runs/profile_ftb3bpl.json) (`extra`: v = proj 2.988 / 9.515 / 17.747,
fc2 32.14 / 331.07 / 223.23) with lr x the same multipliers ([`lrscale_ftb3bpl.json`](../vitbase_runs/lrscale_ftb3bpl.json), v rows
via the row mask), bf16; `ftb3bb` = the same init in bf16 without an lr file (reference); `ftb3bsl` = timm + the reciprocal file
([`lrscale_ftb3bsl.json`](../vitbase_runs/lrscale_ftb3bsl.json), optional). Comparator `ftb3b` 80.00 +- 0.14 (n = 3, fp16).
Multipliers fitted by [`plots/verify/fit_late_multipliers.py`](../plots/verify/fit_late_multipliers.py) from the reproduced init
dump `results/init_dumps/ftb3b_s0.pth`; verifier [`plots/verify/verify_late_trio_3b.py`](../plots/verify/verify_late_trio_3b.py)
(job 29760562: PASS, checks A-E as for the v-inclusive twins plus the targets). Result: to be entered here.

### 1.2 Code path in reading order

| step | where | what to read |
|---|---|---|
| the multipliers | [`vitbase_runs/profile_ftbrhop.json`](../vitbase_runs/profile_ftbrhop.json) (key `extra`: proj 9.06 / 21.16 / 57.91, fc2 9.42 / 27.47 / 81.71), [`vitbase_runs/profile_ftbrhoplv.json`](../vitbase_runs/profile_ftbrhoplv.json) (`extra`: v = proj 3.0071 / 4.6048 / 7.6129, fc2 9.4186 / 27.4658 / 81.7061) | read off `ftbrho`'s init dump; v and proj carry the square root of the attention factor each |
| run scripts | [`run_train_ftbrhop.sh:86-88`](../vitbase_runs/run_train_ftbrhop.sh#L86), `run_train_ftbrhopl.sh:71,86-89`, [`run_train_ftbrhosl.sh:86-88`](../vitbase_runs/run_train_ftbrhosl.sh#L86), `run_train_ftbrhoplv.sh:71,86-89`, [`run_train_ftbrhoslv.sh:86-88`](../vitbase_runs/run_train_ftbrhoslv.sh#L86) | `--init_method analytic_profile --profile_spec <json> --init_method_scaled_blocks ""` for the loud cells, `--init_method default` for the random cells, `--lr_scale_json vitbase_runs/lrscale_<arm>.json` for the lr cells, `--amp_dtype bfloat16` in the two `pl` scripts |
| the lr files | [`lrscale_ftbrhopl.json`](../vitbase_runs/lrscale_ftbrhopl.json) (six scalars = the multipliers), [`lrscale_ftbrhosl.json`](../vitbase_runs/lrscale_ftbrhosl.json) (reciprocals), [`lrscale_ftbrhoplv.json`](../vitbase_runs/lrscale_ftbrhoplv.json) / [`lrscale_ftbrhoslv.json`](../vitbase_runs/lrscale_ftbrhoslv.json) (nine entries; qkv as `{"rows": [[1536, 2304, lambda]]}`) | `[1536, 2304)` is the v third of the fused qkv |
| weight scaling at init | [`utils.apply_analytic_profile`](../utils.py#L1472) [`utils.py:1472-1609`](../utils.py#L1472), the `extra` branch 1579-1586 | with `--init_method_scaled_blocks ""` the per-block loop 1532-1576 is skipped; only `weights[name].mul_(multiplier)` on `_weight_slices` (1410-1415: q, k, v are row views of `attn.qkv.weight`) runs. No data, no RNG, rank-safe. The joint-statistics branch in [`main.py:1086-1160`](../main.py#L1086) is inert because the JSON has no `qk_entropy` / `fc1_gate` / `write_ratio` key |
| lr scaling: parameter groups | [`optim_factory.build_lr_scaled_param_groups`](../optim_factory.py#L154) [`optim_factory.py:154-192`](../optim_factory.py#L154) | `lr_scale` per named tensor, `wd_scale = 1/lr_scale` on decayed tensors (179-186), names and values validated (168-174), `decay` flag per group |
| lr scaling: wiring | [`optim_factory.create_optimizer`](../optim_factory.py#L240) 240-262 and 353-356 | mutual exclusion with `--lr_match_ckpt`, `--release_blocks`, `--custom_lr_layer` (240-244, 270-271); `split_lr_scale_spec` separates scalars from row masks (254); `_lr_scale_spec` stored on the optimizer (354) |
| lr scaling: per step | [`engine.py:103-117`](../engine.py#L103) | `lr = schedule * lr_scale` (105), `weight_decay = schedule * wd_scale` on groups flagged `decay` (114-117) |
| row mask (the `v` cells) | [`row_lr_mask.py`](../row_lr_mask.py) | docstring 1-31 (the argument: AdamW's update is linear in lr per coordinate, so a post-step correction equals a per-row group); [`split_lr_scale_spec`](../row_lr_mask.py#L44) 44-65, [`row_mask_vector`](../row_lr_mask.py#L68) 68-80, [`install_row_lr_masks`](../row_lr_mask.py#L91) 91-143 (guards 97-113; pre-hook remembers the step counter, post-hook 126-141 adds `-(lambda-1) * adaptive step` to the masked rows, skipped when the counter did not advance), `lr_spec_of` / [`assert_same_lr_spec`](../row_lr_mask.py#L146) 146-164 |
| provenance | [`utils.py:618`](../utils.py#L618), `674-677` | spec written into every checkpoint, resume refused on mismatch |
| the write-ratio measurement behind the numbers | [`plots/measure_init_rho_arms.py`](../plots/measure_init_rho_arms.py) (rho definition 13-14, [`build_recipe`](../plots/measure_init_rho_arms.py#L65) 65-94, [`measure`](../plots/measure_init_rho_arms.py#L218) 218-247); [`utils.sublayer_write_ratios`](../utils.py#L1731) 1731-1745 | rho = mean over tokens of the sub-layer write norm over the stream norm it is added to |

### 1.3 Tests and verifiers

- [`plots/verify/test_row_lr_mask.py`](../plots/verify/test_row_lr_mask.py): exact equivalence with a model whose q/k/v are three separate tensors in
  lr-scaled groups (6e-16 over 60 steps in float64), lambda = 1 bit-identical, zero-gradient steps decay by exactly
  `1 - lr*wd`, scaler-skipped steps leave the parameters unchanged, `load_state_dict` mid-run continues exactly,
  ViT-B through `create_optimizer`, malformed specs raise (T1-T10).
- [`plots/verify/verify_late_trio_v.py`](../plots/verify/verify_late_trio_v.py): `ftbrhoplv` init equals the reconstruction bit for bit and `ftbrho`'s dump to
  6.5e-6; `ftbrhoslv` is timm seed 0; the lr files name exactly the nine tensors with the multipliers / reciprocals;
  write ratios at init equal `ftbrho`'s per block.
- [`plots/verify/check_ckpt_lr_spec.py`](../plots/verify/check_ckpt_lr_spec.py): the lr specification recorded in a checkpoint equals the launch file.

### 1.4 Where the multipliers come from (the checkpoint-based late lever, optional)

`ftbrho` (`run_train_ftbrho.sh:71,83-86`: `--init_method upscale_random_match_delta_norms --init_method_scaled_blocks
9,10,11 --target_ratio_absolute 1.4 --random_blocks 0-8`) is the measured version: [`main.py:1963-2086`](../main.py#L1963) measures the
current write ratios with [`engine.attention_residual_analysis`](../engine.py#L1622) (1622-1847; the live lines are 1800-1811, keys
`norm_ratio_attn_delta_in_mean` / `norm_ratio_mlp_delta_in_mean`), transforms the targets
(`transform_target_ratios` [`main.py:528-564`](../main.py#L528)), and scales block by block, sequentially re-measuring
([`main.py:2032-2086`](../main.py#L2032)) through [`utils.scale_layer_weights`](../utils.py#L1304) (1304-1342; the attention factor is split as a square root
over v and proj, 2001 / 2018). The 2x2 arms replay those factors from JSON so that no data-dependent step remains.

### 1.5 Review questions

1. `build_lr_scaled_param_groups`: is `wd_scale = 1/lr_scale` the intended semantics (relative decay unchanged)?
   Confirm on [`engine.py:117`](../engine.py#L117) that the schedule multiplies `wd_scale` and never the current coefficient.
2. Row mask: walk `_correct` ([`row_lr_mask.py:126-141`](../row_lr_mask.py#L126)) against the AdamW update formula once by hand; check
   the bias-correction exponents use the step count after the step; check the skip condition covers the
   gradient-scaler path used in training ([`utils.NativeScalerWithGradNormCount`](../utils.py#L460), utils 460).
3. The `extra` branch of `apply_analytic_profile` touches only the six slices; confirm biases and LayerNorms of
   blocks 9-11 stay timm's (verified in [`verify_late_trio_v.py`](../plots/verify/verify_late_trio_v.py), but worth reading).
4. Precision: the two `pl` cells are bf16, the rest fp16, no bf16 random baseline exists. Decide whether this
   caveat needs a run before the paper.

---

## 2. Freezing (learning-order intervention: freeze and release)

### 2.1 What it is

`r0frz`: plain timm random init, blocks 0-7 at learning-rate factor 0 for epochs 0-29, linear release to the
schedule value over epochs 30-49, normal from epoch 50. Final 79.16 (n = 1) against random `r0` 77.42 / old random
mean 78.08: a pure learning-order manipulation without any checkpoint statistic reproduces half to three quarters
of the committed init's gain, with the winners' dynamics (block-7 lens at chance throughout, class formed in blocks
10-11, fit deficit). `r0frzl` (blocks 8-11 released instead) is the reverse control, launched as a screen.

| step | where | what to read |
|---|---|---|
| flags | [`main.py:342-346`](../main.py#L342) | `--release_blocks` (empty = off), `--release_start 30`, `--release_end 50` |
| run scripts | [`run_train_r0frz.sh:88`](../vitbase_runs/run_train_r0frz.sh#L88) `--release_blocks "0,1,2,3,4,5,6,7" --release_start 30 --release_end 50`; [`run_train_r0frzl.sh:88`](../vitbase_runs/run_train_r0frzl.sh#L88) `--release_blocks "8,9,10,11" ...` | everything else as [`run_train_r0.sh`](../vitbase_runs/run_train_r0.sh) |
| the factor | [`optim_factory.release_factor`](../optim_factory.py#L195) [`optim_factory.py:195-201`](../optim_factory.py#L195) | 0 before start, linear to 1 at end, in fractional epochs |
| the groups | [`optim_factory.build_block_release_param_groups`](../optim_factory.py#L204) 204-222 | tensors of the listed blocks go into groups flagged `release`; decay / no_decay split as the stock grouping; weight decay NOT compensated during the ramp |
| wiring | [`create_optimizer`](../optim_factory.py#L243) 243-244 (exclusion with the lr-scale paths), 249-252 | |
| per step | [`engine.py:54-55`](../engine.py#L54) (config + epoch banner `[release]`), 106-110 | `lr = schedule * release_factor(it / steps_per_epoch)` on the released groups, after the ordinary `lr_scale` line |
| why lr = 0 freezes exactly | docstring at [`optim_factory.py:197-199`](../optim_factory.py#L197) | AdamW's decoupled decay is `lr * wd * w`, so lr = 0 changes nothing; the Adam moments keep following the gradient ("lr factor 0, moments running", not "requires_grad False") |

Legacy freeze flags, for completeness: `--freeze_blocks` / `--freeze_block_attributes` ([`main.py:66-75`](../main.py#L66), normalised
788-805) set `requires_grad = False` inside [`utils.pr_load_model`](../utils.py#L877) ([`utils.py:1087-1096`](../utils.py#L1087)) before the DDP wrap and
before the optimizer, so frozen tensors are absent from every parameter group ([`optim_factory.py:72, 130, 177, 213`](../optim_factory.py#L72)).
No ImageNet run script uses them. `--freeze_patch_embeddings` / `--freeze_pos_embeddings` ([`main.py:249-250`](../main.py#L249)) are
parsed by [`main.py`](../main.py) but act only in [`models/vitp.py:17-18`](../models/vitp.py#L17), the procedural pretraining path: inert here.

### 2.2 Tests and verifiers

- No test file for the release path is in the repository. The checks recorded in
  [`docs/early_lever_mechanism_plan.md`](early_lever_mechanism_plan.md) section 5e ("Freeze and release": grouping, factor, bit-exact freeze under
  AdamW with warm moments, flag survives the optimizer state dict, refusal with the other builders) were run as
  ad-hoc scripts and not saved. See section 6.
- Arm-level: [`results/init_dumps/verify_wave2_task.sh`](../results/init_dumps/verify_wave2_task.sh) (kind `release`): init dump equals `r0`'s; after a frozen
  epoch all 96 tensors of blocks 0-7 equal the init and the 56 others moved; after the release epoch all 96 moved.
  The independent review of 2026-09-22 (section 5f of the mechanism doc) confirmed 96/96 tensors bit-identical
  through epoch 29 on the real run.

### 2.3 Review questions

1. The ramp multiplies the warm-up lr: blocks 0-7 receive 35% of the other blocks' integrated lr up to epoch 50 and
   hit full lr exactly at the schedule peak. Is "learning order" separable from "less training"? (The doc proposes a
   longer-schedule control.)
2. Weight decay is 0 while frozen and ramps with the lr. Intended.
3. `release_factor` is evaluated per update from `it / num_training_steps_per_epoch`; the epoch banner at
   [`engine.py:55`](../engine.py#L55) prints the value at the epoch start. Check both agree with the logged `min_lr`.
4. Resume: the `release` flag lives in the param-group dict and survives `optimizer.state_dict()`; a resume with a
   changed `--release_*` is NOT detected (unlike `lr_scale_spec`). Decide whether that needs the same provenance.

---

## 3. Auxiliary lens loss (align) and suppression loss

### 3.1 What it is

One class, two modes, on the block-7 lens (the same read-out the evaluation logs):

- align (C1 arms): cross-entropy of `head(fc_norm(norm(block-7 output)))[:, 0]` against the mixup targets, weight 0.3,
  epochs 0-49. Forces the class to be readable at block 7 early, on the committed init (`ftbc7c1` kdyck,
  `ftbck7c1` ksd) and on the full procedural checkpoint (`ftb4c1` kdyck, `ftb4kc1` ksd). Tests whether the
  suppressed block-7 transient of the winners is a mediator or a marker. Finals were pending on 2026-09-22
  (`ftbck7c1` led C by +1.4 at epoch 49 and trailed by 0.3-0.4 at epochs 209-224).
- suppress (`r0sup`): cross-entropy of the lens against the uniform distribution minus log K, i.e. KL(uniform ||
  lens), final norm and head detached, on plain random. The 2026-09-22 review found it a logit-variance penalty
  that does not remove the ordering (lens top-1 stays 15-28%); the run is kept as a reference, not as a causal test.

| step | where | what to read |
|---|---|---|
| flags | [`main.py:347-352`](../main.py#L347) | `--aux_lens_block -1` = off; `--aux_lens_mode align|suppress`; `--aux_lens_weight 0.3`; `--aux_lens_until 50` (active for epoch < until) |
| run scripts | [`run_train_ftbc7c1.sh:86-89`](../vitbase_runs/run_train_ftbc7c1.sh#L86), [`run_train_ftbck7c1.sh:86-89`](../vitbase_runs/run_train_ftbck7c1.sh#L86) (committed init + `--aux_lens_block 7 --aux_lens_mode align --aux_lens_weight 0.3 --aux_lens_until 50`); `run_train_ftb4c1.sh:74,88`, `run_train_ftb4kc1.sh:74,88` (`--initialize <full checkpoint> --init_method default` + the same aux line); [`run_train_r0sup.sh:88`](../vitbase_runs/run_train_r0sup.sh#L88) (`--aux_lens_mode suppress`) | |
| the class | [`utils.AuxLensLoss`](../utils.py#L298) [`utils.py:298-331`](../utils.py#L298) | docstring 299-308 (semantics and the known weakness); constructor registers a forward hook on `blocks[block]` of the un-wrapped model (313); `_keep` stores the output only when active and in training mode (314-315); [`set_epoch`](../utils.py#L316) 316-318; [`_norm`](../utils.py#L320) 320-324 recomputes LayerNorm functionally with detached gain/bias in suppress mode; [`__call__`](../utils.py#L325) 325-331: align = `criterion(logits, targets)`, suppress = `-mean log_softmax - log K` |
| attach | [`main.py:2178-2179`](../main.py#L2178) | once, lazily, in the epoch loop; adds no parameters, so the optimizer and DDP are unaffected |
| per epoch / per step | [`engine.py:56-58`](../engine.py#L56) (gate + banner `[aux-lens]`), 129-134 (AMP) and 136-140 (fp32): `loss = loss + weight * aux` | one backward over main + aux; rides the same `loss /= update_freq` and the same clipping |
| logging | [`engine.py:266-267`](../engine.py#L266) (`loss` = main loss, `aux_lens_loss` its own meter -> `train_aux_lens_loss` in log.txt), 290 and 307 (tensorboard / wandb batch-wise `train_loss` = main loss; fixed 2026-09-22 09:40, before that the batch-wise value included the aux term) | |
| the evaluation lens it copies | [`engine.py:478-499`](../engine.py#L478) in `model_analyse` | `model.norm -> model.fc_norm -> model.head -> softmax[:, 0]`, meter `blk_acc_layer{i}` |

### 3.2 Tests and verifiers

- No test file for `AuxLensLoss` is in the repository. The unit tests described in the mechanism doc (value
  equals the lens cross-entropy by hand; gradients reach blocks <= 7, plus norm and head only in align, none
  into later blocks; finite-difference gradient; window and eval mode) exist only in an old session scratchpad
  (`test_auxlens.py`, `tiny_aux_test.py`, `ckpt_lens_check.py`, 164 lines together). See section 6.
- Arm-level: [`results/init_dumps/verify_wave2_task.sh`](../results/init_dumps/verify_wave2_task.sh) (kind `aux`, `ckpt`): init dump bit-identical to the
  committed arm / to `r0` / to the checkpoint in all 144 block tensors; two-rank smoke with the loss ON in epoch 0
  and off in epoch 1, `train_aux_lens_loss` only while on. The 2026-09-22 review: training-time lens equals the
  evaluation lens numerically (diff 0), active for epochs 0-49 only, survives resume and the 8-to-4-rank hand-over.

### 3.3 Review questions

1. align trains the final norm and the head on block-7 features, so lens-7 of the C1 arms is no longer a passive
   probe; cross-arm comparisons need a post-hoc held-out probe on the per-epoch checkpoints. Agree on which number
   goes into the paper.
2. Deep supervision confound: blocks 0-7 receive a second gradient during warm-up. Is "readability" separable
   from "extra gradient"?
3. suppress: read the review's argument ([`docs/early_lever_mechanism_plan.md`](early_lever_mechanism_plan.md) section 5f) and decide whether an
   ordering-based penalty (adversarial probe with gradient reversal) is worth building, or whether `r0sup` stays a
   reference only.
4. `main_value` in `engine.py:134/140` is bound only when the loss is active; every consumer guards on
   `aux_value is None`. Fragile but correct; a default at the top of the loop would remove the trap.
5. The hook fires on every forward in training mode while active. `model_analyse` calls `model.eval()`
   ([`engine.py:445`](../engine.py#L445)) before its forwards, so the analysis passes never feed the hook; `_keep` also checks
   `module.training`. Verified; no action.

---

## 4. The early lever (checkpoint-free procedural init of blocks 0-7)

### 4.1 What it is

The committed design C = P + S + G on blocks 0-7, blocks 8-11 timm random:

- P, profile: effective scales of q, k, fc1 (v, proj, fc2 at timm scale, the `i` variant) and 64 LayerNorm
  statistics (Gaussian-sampled gains and biases from per-block mean and std);
- S, sink: a rank-one component on q and k, calibrated so that the block's mean attention entropy equals the
  prefix's;
- G, gate: a rank-one component on fc1, calibrated so that the mean fc1 pre-activation equals the prefix's.

Arms: `ftbanapermb7i` 80.37 (kdyck), `ftbanakpermb7i` 79.90 (ksd); references kdyck prefix `ftb4i` 79.89, random
78.08 +- 0.19. The factorial (section 5e of the mechanism doc, n = 1 per cell): the profile is needed on both tasks
(S + G loses 0.8-0.9), the gate is needed on ksd, the sink is dispensable on ksd and worth at most 0.45 on kdyck.

Naming (from the JSON key sets): `p` kdyck / `k` ksd checkpoint; `e` exact per-block scales; `r` sink (`--qk_entropy`);
`g` gate; `a` active-unit gate target / `m` mean pre-activation target; `b7` blocks 0-7; `i` input side only
(v, proj, fc2 timm); `w` write-matched v/proj/fc2; `vw` v by scale, proj/fc2 write-matched. `ftbc7*` / `ftbck7*` are
the mechanism-study cells on the committed spec (`p`, `ps`, `pg`, `sg`, `s`, `g`, `a1`, `c1`, `l`).

### 4.2 Extraction: [`extract_profile.py`](../extract_profile.py) (checkpoint -> JSON)

| step | where | what to read |
|---|---|---|
| docstring (the recipe in prose) | [`extract_profile.py:1-139`](../extract_profile.py#L1) | |
| CLI | [`build_parser`](../extract_profile.py#L423) 423-490: `--blocks` 429, `--exact` 431, `--no_layernorm` 433, `--query_key` 435, `--gain_fold` 445, `--realise` 448, `--qk_entropy` 453, `--fc1_gate` 459, `--fc1_gate_target` 462, `--common_write` 467, `--scale_weights` 471, `--write_ratio` 474, `--joint_images` 483, `--target_seeds` 485; [`validate_arguments`](../extract_profile.py#L500) 500-528 (a weight is scaled or write-matched, never both, 514-516) | |
| effective scales | [`effective_scales`](../extract_profile.py#L171) 171-192, [`gain_folded_scale`](../extract_profile.py#L161) 161-168 | q, k, v as row groups of the fused qkv, folded with norm1's gain; fc1 with norm2's; proj and fc2 raw; `exact` = `rms(W diag gamma) / 0.02`, `product` = `rms(W) rms(gamma) / 0.02` |
| LayerNorm statistics | [`layernorm_statistics`](../extract_profile.py#L195) 195-207 | gain and bias mean / std of norm1 and norm2 per block |
| joint targets (data) | [`measure_joint_targets`](../extract_profile.py#L226) 226-314 | builds the prefix model as [`main.py`](../main.py) builds `ftb4i` (243-258), draws training images under the evaluation transform through [`utils.calibration_images`](../utils.py#L1673) (260-261), measures [`utils.joint_statistics_per_block`](../utils.py#L1749) over `--target_seeds` random contexts (248-270), drops block 0 (271); writes `qk_entropy` 286-288, `fc1_gate` 289-299, `write_ratio` 300-308, `common_write` 309-313 |
| spec assembly | [`build_profile_specification`](../extract_profile.py#L344) 344-393 (fit and corrections 319-339 for the ramp form) | records `realise` and `gain_fold` 382-384 |
| main | 542-566 | spec -> tables -> `json.dump` |
| the output | [`vitbase_runs/profile_ftbanapermb7i.json`](../vitbase_runs/profile_ftbanapermb7i.json) | keys `q, k, fc1` (`per_block`), `realise: exact`, `gain_fold: exact`, `ln` (`source: parametric`, `stats` per block), `qk_entropy` (`entropy` per block 1-7, `renormalize: true`, seeds, sd), `fc1_gate` (`pre_activation_mean` per block 1-7) |

### 4.3 Realisation at init: [`main.py`](../main.py) -> [`utils.apply_analytic_profile`](../utils.py#L1472)

| step | where | what to read |
|---|---|---|
| run script | [`run_train_ftbanapermb7i.sh:86-88`](../vitbase_runs/run_train_ftbanapermb7i.sh#L86) | `--initialize "" --init_method analytic_profile --profile_spec vitbase_runs/profile_ftbanapermb7i.json --init_method_scaled_blocks 0,1,2,3,4,5,6,7` (block list from the flag, joint components from the blocks named in the JSON) |
| branch | [`main.py:1069-1081`](../main.py#L1069) | `pr_load_model(path="")` wraps in DDP; `apply_analytic_profile(model_without_ddp, spec, blocks, seed=args.seed)` on every rank (deterministic in the run seed, not the rank); log `[analytic_profile] block b: ...` |
| the function | [`utils.apply_analytic_profile`](../utils.py#L1472) [`utils.py:1472-1609`](../utils.py#L1472) (docstring 1473-1509) | per block: LayerNorm vectors first ([`_write_layernorm_vectors`](../utils.py#L1430) 1430-1469, generator `1000 + 10 seed + block`), then one multiplier per slice ([`_declared_scale`](../utils.py#L1418) 1418-1427; `realise: exact` recomputes the multiplier from the actual tensors so the declared effective scale is met to 1e-7, 1548-1562), applied in place 1566-1568; `extra` 1579-1586; legacy `fc1_bias` 1592-1595 and `q_sink` 1601-1608 |
| what each tensor receives | | q, k, v: one scalar each on the row views of `attn.qkv.weight`; proj, fc1, fc2: one scalar; norm1/norm2 weight and bias overwritten by sampled vectors; linear biases untouched by this step |

### 4.4 Calibration of the rank-one components: [`utils.calibrate_joint_statistics`](../utils.py#L2086)

| step | where | what to read |
|---|---|---|
| call site | [`main.py:1082-1160`](../main.py#L1082) | joint keys collected 1088; the tensors the calibration may touch 1090-1101; rank 0 only draws 256 training images (1105-1106) and calls `calibrate_joint_statistics` (1107); report lines `[qk_entropy]`, `[fc1_gate]`, `[write_ratio]` 1110-1138; unmet-target count broadcast and raised on every rank 1139-1145; the calibrated tensors broadcast from rank 0 and a sha256 of their bytes compared across ranks 1146-1159 (`[joint statistics] ... identical on all R ranks: True`) |
| the constructions | comment block [`utils.py:1620-1671`](../utils.py#L1620) | `W_q <- s_q W_q + alpha P c1^T`, `W_k <- s_k W_k + alpha P r^T`; `W_fc1 <- s W_fc1 - beta (1/sqrt n) 1 c2^T`; order within a block: sink, attention write ratio, gate, common write, MLP write ratio (1669-1670) |
| driver | [`calibrate_joint_statistics`](../utils.py#L2086) 2086-2143 | rejects the old key `qk_sink` (2092-2094), refuses two gate targets (2103-2104), `{}` when nothing is requested, walks blocks in depth order advancing `stream = block(stream)` (2119-2140) |
| the stream | [`block_input_stream`](../utils.py#L1696) 1696-1714, [`fc1_input`](../utils.py#L1717) 1717-1727 (follows the block's own forward incl. layer scale and drop path), [`sublayer_write_ratios`](../utils.py#L1731) 1731-1745, [`joint_statistics_per_block`](../utils.py#L1749) 1749-1776, [`calibration_images`](../utils.py#L1673) 1673-1677 | |
| sink | [`_install_rank_one_sink`](../utils.py#L1834) 1834-1902 | `c` = unit mean of `norm1(stream)`, `r` seeded random unit, `P` = [`sink_directions`](../utils.py#L1612) 1612-1617; strength bisected on the block's attention entropy ([`_bisect`](../utils.py#L1809) 1809-1826, brackets checked at strength 0 since the guard fix); `s_q`, `s_k` from [`_rescale_to_keep_norm`](../utils.py#L1785) 1785-1806 when `renormalize: true`; report incl. `matched` (tolerance 0.02 nats, 1894) |
| gate | [`_install_fc1_gate`](../utils.py#L1905) 1905-1968 | component `-(1/sqrt n) 1 c^T` with `c` the unit mean of `fc1_input`; bisected on `active_units` or `pre_activation_mean` (1942-1950); tolerances 1958-1961 |
| write ratio (the `w` / `vw` arms, `ftbanac`) | [`_match_write_ratio`](../utils.py#L2057) 2057-2082 | scalar on v rows + proj (square-root split) or on fc2, weights and biases, 6 iterations |
| common write (the `g0` arms, block 0 only) | [`_install_common_write`](../utils.py#L1971) 1971-2005, [`_install_common_write_at_ratio`](../utils.py#L2008) 2008-2054 | not in the committed design |

### 4.5 Verifiers and tests

| tool | what it proves |
|---|---|
| `plots/verify/verify_recipe_statement.py CKPT SPEC` | the equations of recipe sections 3-4 (declared scales, LN statistics) hold on the checkpoint and the spec |
| `plots/verify/verify_joint_statistics.py --spec SPEC [--dump ARM.pth --base BASE.pth]` (docstring 1-15) | exactly the expected tensors changed (55-63); sink: v rows identical, effective q/k ratio 1, rank-one residual < 1e-3 (73-80); gate: effective fc1 ratio 1, rank one, left singular vector constant (81-88); write ratios exact scalars (98-111); targets met on the calibration images, an unseen draw and the training augmentation (114-141); `VERDICT: PASS` (153) |
| [`plots/verify/verify_structure_only.py`](../plots/verify/verify_structure_only.py) | the S + G / S / G cells: profile step a no-op, only qkv or fc1 of blocks 1-7 change, each change rank one, all targets matched |
| [`plots/verify/verify_extract_profile.py`](../plots/verify/verify_extract_profile.py) | extraction checked on kdyck and ksd (weights, LN statistics, untouched blocks, forward pass vs the arm dumps) |
| [`plots/verify/test_calibration_guards.py`](../plots/verify/test_calibration_guards.py) | G1-G2: `_bisect` bracket check, `matched` from the achieved statistic, unmet target aborts; G3: `keep_all=True` on the post-training reload |
| `plot_reconstruction.py CKPT SPEC` (docstring 1-24) | twelve-panel figure: checkpoint prefix vs specification vs second moments only vs random on images; `plots/out/reconstruction_<spec>.png` |
| [`plots/verify/target_stability.py`](../plots/verify/target_stability.py), [`structure_only_reachability.py`](../plots/verify/structure_only_reachability.py) | how the targets move with the random context; which cells are reachable with `renormalize: false` |

### 4.6 Arms built on the same path with JSON surgery or lr files

| family | arms | how | code |
|---|---|---|---|
| factorial cells (P, P+S, P+G, S+G, S, G) | `ftbc7p/ps/pg/sg/s/g`, `ftbck7*` | the committed spec with keys removed (`vitbase_runs/profile_ftbc7*.json`); structure-only cells carry `renormalize: false` in `qk_entropy` and `fc1_gate` | no new code; `renormalize` read in the installers |
| step compensation (A1/A2/A3, the early-lever 2x2 "normal steps" cell) | `ftbc7a1`, `ftbck7a1`, `ftbc7a1g`, `ftbck7a1g`, `ftbc7a2`, `ftbck7a3qk`, `ftbck7a3f` | committed init + `--lr_scale_json vitbase_runs/lrscale_<arm>.json` with `m = rms(launched init) / rms(timm)` per tensor; q and k rows through the row mask, fc1 as a scalar ([`run_train_ftbc7a1.sh:89`](../vitbase_runs/run_train_ftbc7a1.sh#L89)) | section 1.2 machinery; multipliers from [`plots/verify/make_step_compensation.py`](../plots/verify/make_step_compensation.py); checked by [`verify_step_compensation.py`](../plots/verify/verify_step_compensation.py) |
| compose (early + late lever) | `ftbc7l` (kdyck, running since 2026-09-22 14:35) | [`profile_ftbc7l.json`](../vitbase_runs/profile_ftbc7l.json) = C's spec + the `extra` section of [`profile_ftbrhoplv.json`](../vitbase_runs/profile_ftbrhoplv.json); [`run_train_ftbc7l.sh:86-88`](../vitbase_runs/run_train_ftbc7l.sh#L86) | [`apply_analytic_profile`](../utils.py#L1532) 1532-1576 for blocks 0-7 and the `extra` branch 1579-1586 for blocks 9-11; the calibration never sees blocks 9-11; [`verify_compose.py`](../plots/verify/verify_compose.py); figures `plots/out/reconstruction_ftbc7l*.png` |

### 4.7 Review questions

1. Rank safety: `apply_analytic_profile` runs on every rank and must be deterministic in `args.seed`; the
   calibration runs on rank 0 and is broadcast. Read the two generators (`1000 + 10 seed + block`,
   `2000 + 10 seed + block`, `3000 + 10 seed + index`) and the broadcast list at [`main.py:1090-1101`](../main.py#L1090): is every
   tensor the installers can write in that list (qkv, fc1, fc2, and the biases of write-ratio blocks)?
2. `renormalize: true` shrinks the token-specific part of q, k, fc1 (block 1: `s_q = 0.58`) so "structure eats
   into scale" in the committed arms. Agree on how the paper words the scale factor (memory: effective scale = matrix
   size, not forward scale).
3. Calibration images: 256 training images under the evaluation transform, drawn on rank 0's GPU; the
   calibration hash differs between GPU types (L40S vs H200) at the rounding level. Acceptable, but say so.
4. The order of edits: `apply_analytic_profile` -> calibration -> EMA construction ([`main.py:1496-1504`](../main.py#L1496)) -> the
   legacy edit block (1675-2110) -> `sync_initialisation` (2117) -> EMA re-copy (2118-2122). Check no legacy edit
   fires for the committed arms (all their flags are at defaults).
5. [`extract_profile.py`](../extract_profile.py) averages targets over 5 random contexts; [`main.py`](../main.py) calibrates on one draw with the run
   seed. The verifiers check the achieved statistic on the calibration draw, an unseen draw and the training
   augmentation. Enough for the claim "calibrated to the prefix"?
6. Unmet targets abort since 2026-09-19; the launched arms of sections 10-11 of the recipe doc predate that and
   were re-checked by the verifiers (all PASS). Say so in the paper's appendix.

---

## 5. Experiments not on your list that the review should probably include

- The compose arm `ftbc7l` (section 4.6): the paper's "the two levers combine" claim rests on it and on `ftbanac`
  (80.55, analytic blocks 0-8 + late lever). Same code path, one JSON.
- The step-compensation arms (A1: `ftbc7a1`, `ftbck7a1`; section 4.6): they are the early-lever counterpart of the
  late-lever 2x2 and reuse the lr-scale and row-mask code of section 1. The kdyck A1 run had loss spikes that grow
  with the q/k step size (mechanism doc 5b); worth a look at [`lrscale_ftbc7a1.json`](../vitbase_runs/lrscale_ftbc7a1.json) together with the row mask.
- The factorial cells (section 4.6): no new code, but the `renormalize: false` path in the installers is exercised
  only by them.
- The checkpoint-based late lever `upscale_random_match_delta_norms` (section 1.4): the origin of every late-lever
  number, sequential scaling with re-measurement; `ftbcomp11` (80.63, the best arm) uses it.
- The proc-prefix loading path itself ([`utils.pr_load_model`](../utils.py#L877) with `--random_blocks`): the reference arms
  `ftb3i` / `ftb4i` (prefix 0-8 / 0-7) that every early-lever number is compared to.
- The lens metric ([`engine.model_analyse`](../engine.py#L416), section 0.2): every mechanism reading is `blk_acc_layer7`; the C1 arms
  manipulate exactly this number.
- The DDP init sync (`sync_initialisation`) and the reload artefact fix (`keep_all=True`): both changed which
  numbers are trustworthy (memories `ddp-rank-sync-bug`, `post-training-reload-artefact`).
- Operational wrappers that decide what a "run" is: [`vitbase_runs/resume_zero_kbias.sh`](../vitbase_runs/resume_zero_kbias.sh) (k-bias fp16 overflow
  fix on resume), [`vitbase_runs/handover_to_h200.sh`](../vitbase_runs/handover_to_h200.sh) (L40S -> H200 moves; changes the per-rank data split from the
  hand-over epoch on), `--stop_after_epoch` / `--analysis_ckpt_*` (screens and checkpoint thinning).
- Readout tooling: [`plots/verify/arm_truth.py`](../plots/verify/arm_truth.py) (last-epoch accuracy from log.txt), [`plots/verify/wandb_layerwise.py`](../plots/verify/wandb_layerwise.py)
  (the `ARMS` map, lines 10-83, is the authoritative arm -> run-id table).

## 6. Gaps found while preparing this plan

1. No tests for `AuxLensLoss` or for the freeze-and-release path are in the repository; the ones described in the
   mechanism doc live in an old session scratchpad
   (`/tmp/claude-16853/-home-schrodi-Procedural/57ac4c02-f3c3-4598-85f2-c462137a5cec/scratchpad/{test_auxlens,tiny_aux_test,ckpt_lens_check}.py`)
   and will disappear with it. Move them to `plots/verify/` next to [`test_row_lr_mask.py`](../plots/verify/test_row_lr_mask.py) before the review.
2. All intervention code is uncommitted (section 0.1). Commit before the review so the line numbers here hold.
3. `--freeze_patch_embeddings` / `--freeze_pos_embeddings` are parsed by [`main.py`](../main.py) but inert there (section 2.1).
4. [`plots/measure_init_rho_arms.py`](../plots/measure_init_rho_arms.py): `ARMS` (lines 30-31) does not list `ftbrho` although `main()` branches on it;
   the recipe row of [`plots/cache/init_rho.json`](../plots/cache/init_rho.json) needs it added.
5. [`docs/proc_init_recipe.md`](proc_init_recipe.md) section 5 ("Information flow, for following the code") cites line numbers that are
   stale relative to the working tree (e.g. [`apply_analytic_profile`](../utils.py#L1411) 1411 -> 1472, `calibrate_joint_statistics`
   1729 -> 2086, the branch 1051 -> 1069). Section 4 above has the current ones.
6. A resume with changed `--release_*` values is not detected (section 2.3, item 4).
