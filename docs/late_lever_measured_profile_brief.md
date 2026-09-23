# Brief: the late-lever step-size test at proc's MEASURED late-block profile (2026-09-22)

## Why

The late-lever 2x2 (`docs/proc_init_recipe.md` section 13) says "the late lever is the loud write, not the step size",
but every cell was built on the flat recipe target 1.4. Proc's own late blocks do not write at 1.4. Measured on the
kdyck checkpoint (`results/imnet_base/results_IMNET_BASE_29388202/s0/target_res_stats.json`, the targets `ftb3b` used):

| block | attention write ratio | MLP write ratio |
|---|---|---|
| 9 | 1.380 | 4.757 |
| 10 | 2.098 | 4.459 |
| 11 | 0.777 | 0.526 |

`ftb3b` (random init, blocks 9-11 scaled to exactly these values by `upscale_random_match_delta_norms`) is 80.00 +- 0.14
(n = 3, fp16, seeds 29388202 / 29406778 / 29406779); `ftbrho` (flat 1.4) 79.69 +- 0.30. So the mechanism claim is
tested at the recipe's profile only. This brief adds the missing cell at proc's profile: `ftb3b`'s initialisation, tensor
for tensor, with the learning-rate multipliers that give the scaled tensors the random init's relative Adam steps. It is
the `ftbrhoplv` construction (section 13, v-inclusive table) applied to `ftb3b` instead of `ftbrho`.

Decision rule (n = 1 against an n = 3 reference, seed resolution 0.45): if `ftb3bpl` ends within 0.45 of `ftb3b`, the loud
write is the lever at proc's profile as well; if it ends 0.45 or more below, the step size matters at the measured level
and the section-13 sentence has to be restricted to the flat target.

## Arms

| arm | init | steps | precision | status |
|---|---|---|---|---|
| `ftb3b` | measured profile (exists, n = 3) | slow (large weights, no lr file) | fp16 | done, 80.00 +- 0.14 |
| **`ftb3bpl`** (required) | the same init, replayed from JSON (`extra` multipliers) | normal: lr x m on the v rows, proj, fc2 of blocks 9-11 (`--lr_scale_json`, row mask for v) | **bf16** (see precision) | to build |
| `ftb3bb` (recommended) | `ftb3b`'s own script, one seed | slow | bf16 | precision-matched reference for `ftb3bpl`; without it the comparison is bf16 against fp16, the same caveat as `ftbrhopl` vs `ftbrhop` |
| `ftb3bsl` (optional) | timm | slow: lr x 1/m on the same tensors | fp16 | random write + slow steps at the measured multipliers; the 1.4 version (`ftbrhoslv` 78.18) was flat, so only run if budget allows |

## Steps

1. **Reproduce `ftb3b`'s launched init as a dump** (no dump exists in `results/init_dumps/`). Use exactly the command
   that produced `results/init_dumps/ftbrho_s0.pth` (first line of `results/init_dumps/ftbrho_s0.log`), with
   `--target_ratio_absolute` REMOVED and `--target_ratio_scale 1.0 --target_ratio_flatten false` kept, i.e. the init lines
   of `vitbase_runs/run_train_ftb3b.sh`:
   `--initialize results/pr_vitb_n/pr_6066174_final.pth --random_blocks 0,1,2,3,4,5,6,7,8
   --init_method upscale_random_match_delta_norms --init_method_scaled_blocks 9,10,11 --seed 0`.
   Keep `--data_path /data/datasets/ILSVRC2012`: the calibration draws 5000 training images with the seeded Python RNG
   on rank 0 (`main.py:1752-1766`), so the launched data path and seed 0 on one GPU reproduce the launched images (memory
   `init-dumps-calibrate-on-val`: a different data path calibrates on different images). One GPU, ~10 min; the HDD path
   is slow but 5000 images are fine. Write `results/init_dumps/ftb3b_s0.pth` and keep the log.
   Check: the printed target statistics equal the table above (`target_res_stats.json` of the launched run) and the
   achieved ratios after scaling equal the targets. This validates the reproduction the same way the `ftbrho` dump was
   validated (matched the launched init to 6.5e-6 relative).
2. **Read the multipliers off the dump.** For blocks 9-11: the v rows `[1536, 2304)` of `attn.qkv.weight`, `attn.proj.weight`,
   `mlp.fc2.weight`. Fit `dump = m * timm` per tensor against the timm seed-0 model (`results/init_dumps/r0_s0.pth` is timm
   seed 0 through main.py); the fit residual must be ~1e-7 (it was 3e-8 for `ftbrho`). v and proj carry the same factor
   (the square root of the attention factor, `main.py:2001`), fc2 the MLP factor. Assert every other tensor of the dump
   (q/k rows, fc1, all biases, LayerNorms, blocks 0-8, embeddings, head) is bit-identical to timm seed 0
   (`--init_method_bias_scaling` is false in `ftb3b`). Expect fc2 factors larger than `ftbrho`'s 9.4 / 27.5 in blocks 9-10
   and smaller than 81.7 in block 11.
3. **Write the spec files**, same shapes as `profile_ftbrhoplv.json` / `lrscale_ftbrhoplv.json`:
   - `vitbase_runs/profile_ftb3bpl.json`: `{"extra": {"9": {"v": a9, "proj": a9, "fc2": f9}, "10": {...}, "11": {...}}}`
   - `vitbase_runs/lrscale_ftb3bpl.json`: `{"blocks.9.attn.qkv.weight": {"rows": [[1536, 2304, a9]]},
     "blocks.9.attn.proj.weight": a9, "blocks.9.mlp.fc2.weight": f9, ... for 10 and 11}` (nine entries, nothing else)
   - optional `vitbase_runs/lrscale_ftb3bsl.json`: the reciprocals in the same shape.
   Full precision of the fitted numbers (the 1.4 files carry 4-6 digits; relative deviation is checked at 1e-5).
4. **Run scripts.** `run_train_ftb3bpl.sh` = `vitbase_runs/run_train_ftbrhoplv.sh` with the two file names, job name and
   `--notes` changed; keep `--initialize "" --init_method analytic_profile --init_method_scaled_blocks ""` (the `extra`
   multipliers apply to their own blocks; the joint-statistics branch is inert because the JSON has no target keys),
   `--amp_dtype bfloat16`, and add `--analysis_ckpt_dense_until 60 --analysis_ckpt_every 10` as the mechanism arms carry
   (per-epoch model checkpoints for the autopsy; the disk note in the mechanism doc, section 5c). `run_train_ftb3bb.sh` =
   `run_train_ftb3b.sh` + `--amp_dtype bfloat16` (one seed). `run_train_ftb3bsl.sh` = `run_train_ftbrhoslv.sh` with the
   reciprocal file.
5. **Verify** before launching (GPU, minutes), by generalising `plots/verify/verify_late_trio_v.py` (its paths to the
   profile, the lr files and the reference dump are constants; make them arguments or copy it to
   `verify_late_trio_3b.py`). Required checks, as its A-E:
   - A: `utils.apply_analytic_profile` on timm seed 0 with the new spec changes only the v rows, proj, fc2 of blocks 9-11,
     each by one scalar; everything else bit-identical to timm.
   - B: that reconstruction equals `ftb3b_s0.pth` to the rounding of the multipliers.
   - C: `lrscale_ftb3bpl.json` scalar entries == the multipliers, row entries == `[[1536, 2304, a_b]]` exactly; the `sl` file
     the reciprocals; no other tensor named.
   - D: the main.py init dump of `ftb3bpl` (`plots/dump_init.py` with the run script's flags) == the reconstruction bit for
     bit; `ftb3bsl` == timm seed 0.
   - E: write ratios at init on 256 training images equal `ftb3b`'s dump per block and are close to the targets
     (1.38 / 2.10 / 0.78 attention, 4.76 / 4.46 / 0.53 MLP; a few percent because of the image draw).
   - Two-rank smoke on the real schedule (two epochs) for each arm: three `[row-lr]` and six `[lr-scale]` lines, finite loss,
     checkpoint carries the lr spec (`plots/verify/check_ckpt_lr_spec.py`), resume with the same file works and a changed
     file is refused (`plots/verify/verify_resume_checkpoint.py`).
   - Precision probe on the `ftb3bpl` init (`plots/verify/fp16_headroom_probe.py`): record the maximum attention logit and
     fc2 output under fp16 autocast; this documents why the cell runs in bf16 (`ftbrhopl` overflowed fp16 at epoch 44 with
     smaller factors).
6. **Launch** with the user's go: one seed each, `ftb3bpl` first, then `ftb3bb`, `ftb3bsl` last. 4 H200 (shared partition,
   continuation job) or 8 L40S with the H200 hand-over (`vitbase_runs/handover_to_h200.sh`) as in wave 2; ~22 h on H200,
   9.5 min per epoch on L40S. wandb project `vit base kdyck`, ids into the run log as usual.
7. **Read out** the last-epoch top-1 from log.txt (`plots/verify/arm_truth.py`, never the accuracy JSON top level), note
   loss spikes and skipped fp16 steps, and write the result as a new paragraph at the end of section 13 of
   `docs/proc_init_recipe.md` (a third 2x2 table: measured profile) and into section 1 of `docs/code_review_plan.md`.

## Caveats to carry into the write-up

- Precision: `ftb3bpl` is bf16; `ftb3b` is fp16. `ftb3bb` removes this; without it, quote the `ftbrhopl` vs `ftbrhop`
  experience (+0.20 across precisions, inside the seed resolution) and say so.
- Compounding: the measured profile puts fc2 factors in blocks 9-10 above `ftbrho`'s, i.e. closer to the divergence
  regime of `ftb4j` (factors up to 173 diverged). `ftb3b` trained 3/3 in fp16 with slow steps; `ftb3bpl` gives those
  tensors 100x-scale steps in relative terms, which is the point of the cell and also the risk. If it diverges in the
  warm-up, that is a result (record it), not a reason to lower the target.
- Nominal, not realised: the lr multiplier matches the optimiser's explicit scaling (section 12 qualifier); the gradients
  differ between inits, so the trajectories are not identical by construction.
- Do not touch the 1.4 arms or the compose arm `ftbc7l`; this adds a cell, it does not replace the trio.
