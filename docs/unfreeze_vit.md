# Progressive layer unfreezing on ViT-S / ImageNet-100

Port of layer-scale's progressive layer unfreezing (`docs/unfreeze_vit_handoff.md`) to the
`vit_small` / IN-100 recipe. Four arms at E = 0.2, C = 4, seed 0, paired against schrodi's
`i100_playground/r_random/s0`. Code is in `unfreeze.py` (`--unfreeze_arm`); the scripts,
checker and analysis are in `unfreeze/`. Outputs are under
`/work/dlc2workfs3/schrodi-procedural/kempfe/unfreeze/`.

## Setup

| | |
|---|---|
| recipe | schrodi's `run_train_i100.sh` arm r: `vit_small`, 300 ep, warmup 50, lr 2e-3, fp16, 2x H200, batch 128 x update_freq 2 = 512, `--skip_norm true`, stage-wise / detailed metrics; `NUM_WORKERS` fixed at 12 |
| T | 300 x 247 = 74,100 optimizer steps |
| chunks | blocks 0-2 / 3-5 / 6-8 / 9-11; patch_embed, cls_token, pos_embed, norm and head always train |
| mechanism | "cold moments": frozen chunks' grads are dropped after the all-reduce and before `unscale_` / step, and their group lr is 0, so they get no update, no weight decay and no Adam state; per-chunk lr ramp over one epoch (247 steps) |
| base | `r_random` s0-2 plus the identical-recipe `r_random_k4` s0-2: 84.37 +- 0.30 top-1, 0.7845 +- 0.0072 loss (6 runs); paired s0 = 84.30 / 0.7887 |

| arm | frozen at step 0 | unfreezes at |
|---|---|---|
| `u4f-lin0.2` | blocks 3-11 | 3-5 @ ep 20, 6-8 @ ep 40, 9-11 @ ep 60 |
| `u4b-lin0.2` | blocks 0-8 | 6-8 @ ep 20, 3-5 @ ep 40, 0-2 @ ep 60 |
| `u4f-one0.2` | blocks 9-11 | ep 60 |
| `u4b-one0.2` | blocks 0-2 | ep 60 |

## Checks

- **CPU gate:** `tests/test_unfreeze.py` has 13 tests, all passing. They cover handoff items
  (a)-(g):
  - frozen chunks stay bit-identical to init, with no Adam state, until u_k; after that the
    Adam step count equals t - u_k;
  - E = 0 is bit-identical to the vanilla optimizer;
  - a mid-run resume is exact, and a different layout is refused;
  - the isolation and resume guards refuse what they should;
  - the source contract holds.
- **GPU smoke:** job 29775577, 2x H200, the real schedule, 650 steps, run through `unfreeze/check.py`.
  Everything passed:
  - The mechanism holds on the real DDP / fp16 trainer, and no GradScaler step was skipped.
  - **The recipe is bit-deterministic on GPU.** A second base run and the E = 0 control both match
    the first base run bit for bit at all 16 snapshots.
  - Step time is the same as base.
  - The smoke's base run matches `r_random/s0`'s epoch-0/1 train loss to 4 decimals, which
    justifies reusing that base.
- **Post-run:** `check.py --run` passes for all four arms. The delayed chunks are identical to the
  epoch-0 dump until their unfreeze epoch, and the `unfreeze/*` scalars in `log.txt` match the plan.

## Results (seed 0, epoch 299)

| arm | top-1 | delta vs s0 | delta vs mean (z) | test loss | delta loss vs s0 | train loss |
|---|---|---|---|---|---|---|
| base (6 runs) | 84.37 +- 0.30 | | | 0.7845 +- 0.0072 | | 2.04-2.06 |
| `u4f-lin0.2` | 83.02 | -1.28 | -1.35 (-4.5) | 0.8840 | +0.095 | 2.369 |
| `u4b-lin0.2` | 82.66 | -1.64 | -1.71 (-5.7) | 0.8780 | +0.089 | 2.501 |
| `u4f-one0.2` (9-11 frozen) | 83.72 | -0.58 | -0.65 (-2.2) | 0.7920 | +0.003 | 2.114 |
| `u4b-one0.2` (0-2 frozen) | **84.92** | **+0.62** | +0.55 (+1.8) | **0.7303** | **-0.058** | 2.153 |

Figure: `kempfe/unfreeze/figures/unfreeze_curves_s0.png`, made by `python unfreeze/curves.py`.

1. **Staged unfreezing hurts in both directions.** Both `lin` arms fall 6-10 points behind by
   epochs 40-80, recover slowly, and end 1.3-1.6 top-1 and about 0.09 loss below base. Train loss
   ends +0.3 / +0.45 above base, so this is lost fit and not overfitting. Back-to-front is worse
   than front-to-back on top-1 (-1.64 vs -1.28), but only 1.3x worse, and the two tie on loss. The
   LM's roughly 2x asymmetry is mostly gone.
2. **Freezing the first chunk does not hurt: it helps.**
   - `u4b-one0.2` (blocks 0-2 held at init for 60 epochs) ends with test loss -0.058 vs s0, about
     8 base sd, and top-1 +0.62 paired (+0.55 vs the mean, 1.8 sd).
   - Its train loss is +0.10 *above* base, so it fits train worse and generalises better, i.e. it
     acts as a regulariser.
   - It sits inside the base band until about epoch 220, and the gap opens in the cosine tail. The
     readout-skip gain showed the same timing.
   - This is the opposite of the LM, where freezing the first chunk cost +0.04 nats.
3. **Freezing the last chunk is not free.** `u4f-one0.2` costs 0.58 top-1 (2.2 sd), but its loss
   is null (+0.003, 0.5 sd). In the LM this arm was the null one.
4. The LM's early "frozen arms ahead of base at ~5 % of training" shows only faintly: the `one`
   arms sit at the top of the base band around epochs 5-15.

**Caveat:** one seed. Both `one` arms are within about 2 sd on top-1, so the signs of findings 2-3
need seed 1, paired against `r_random/s1`. The `u4b-one` loss effect is large enough that it is
unlikely to be noise.
