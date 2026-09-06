# ImageNet-1k figure deck

Regenerate with `python plots/make_figures.py` (all) or `--only 1,3` (a subset).
Outputs land in `plots/out/`, 300 dpi PNG, alongside two CSVs of the underlying numbers.

All accuracies are **last-epoch test top-1** (epoch 299 of 300), never max-over-epochs —
the convention recorded at the top of `docs/i100_late_block_scaling.md`.

Source is each run's `log.txt`, not wandb: jobs are `--requeue`d and every resume opens a
**new** wandb run, so one (slurm_id, seed) is scattered across several wandb runs that would
have to be stitched by epoch. `log.txt` is written by the training loop and is continuous.
wandb is still the only place per-layer rho-during-training is recorded, which no figure here
uses yet.

> **[2026-08-30] fig5, fig10 and fig12 carry claims retracted by docs section 0.** They are
> correct about *what was run and what it scored* and wrong about *why*. The early-block
> mechanism is one arrangement-invariant scalar, the attention write magnitude
> `gamma * ||W_v|| * ||W_proj|| / d`, which accounts for 83% of the variance across 30 runs
> (F = 64.9, p = 5e-11). fig10's qk/v ratio and fig12's v slice are both proxies for it: held at
> a fixed write magnitude, qk/v varies 1.00-2.18 and the logit scale 0.0055-0.0081 with no effect
> on accuracy. **Lead with `fig13_value_write.png`.**

## Suggested order for the supervisor

| # | file | one-line claim |
|---|---|---|
| 1 | `fig1_loss_curves.png` | Calibrating blocks 9-11 at init improves **generalisation** without improving the training fit. (**the figure you asked for**) |
| 2 | `fig2_generalisation.png` | Across all 33 arms, lower training loss predicts *worse* test accuracy (r=+0.79); overfitting predicts it almost perfectly (r=−0.92, p=3e-14). |
| 3 | `fig3_headline.png` | All eight n=3 ladder arms, grouped by what the method *requires*: no procedural weights (left) vs procedural weights + calibration (right). |
| 4 | `fig4_depth.png` | Full k=1..11 sweep both ways. Paired at matched k, placing blocks early is worth **+0.66**, winning 10/11 (Wilcoxon p=0.003). |
| 5 | ~~`fig5_mechanism.png`~~ (superseded) | 13-row elimination matrix. The benefit is carried by **one slice of the attention weights**: matching proc's `v` alone gives +1.37, matching `qk` alone gives +0.50. |
| 6 | `fig6_stability.png` | Procedural weights destabilise training; shuffling them, or using the recipe, does not. |
| 7 | `fig7_shuffle.png` | Destroying the weight *arrangement* costs a point at ONE procedural block but is **free** at nine — the sharpest open puzzle. |
| 13 | `fig13_value_write.png` | **START HERE.** One arrangement-invariant scalar — the attention write magnitude at init — accounts for **83%** of the variance across all 30 early-block runs (F = 64.9, p = 5e-11). Every other 'mechanism' in this deck is a proxy for it. (docs §0) |
| 10 | ~~`fig10_qkv_ratio.png`~~ (superseded) | Of 205 statistics measured at init, **one** survives all four gates: the q/k-to-v weight-scale ratio (r = +0.96, p = 4e-7 over 13 arms). The three diamonds were run *after* the relation was fitted and land on it. (docs 3.10.9.11-.12) |

## The init-time screen (fig10)

Three scripts, run in this order:

```
sbatch vitbase_runs/run_ckpt_diff.sh            # 1 L40S, ~25 min -> plots/cache/ckpt_diff.json
.venv/bin/python plots/score_ckpt_features.py   # ranks all 205 statistics; --mode ushape for
                                                # "closeness to proc" instead of "more is better"
.venv/bin/python plots/analyse_training_traces.py   # the same screen on the wandb per-layer traces
.venv/bin/python plots/fig_qkv_ratio.py         # -> out/fig10_qkv_ratio.png
```

The screen's value is as much in what it kills as in what it keeps. Attention entropy, logit
spread, token collapse, effective rank and the gradient-to-weight ratio all move by 2-70x between
`ftb3i` and `ftb4e3` — two arms that differ only by a within-slice permutation and train to the
same accuracy — so none of them can be the mechanism. See docs 3.10.9.11.

## Caveats that are on the figures on purpose

* **fig4** — late-block (`h`) points are single seed and the ViT-B seed s.d. is ~0.3, so no
  individual pair should be read alone; the claim is the *paired* statistic over all 11 depths.
  Both series meet at k=12 (the same model), and that arm scores 80.09 — identical to the
  separately-trained procedural baseline, which is a free consistency check.
* **fig3** — only the `rho`=1.4 arm is fully checkpoint-free. The `a1` arm keeps no procedural
  weights but still reads its target ratios off a procedural checkpoint; the figure groups on
  "keeps procedural weights", so `a1`'s dependence is stated in the caption rather than the
  grouping.
* **fig5** — presented as an elimination matrix. All five rows share one base: random embeddings,
  head, final norm and blocks 9-11. `pr_load_model` deletes `cls_token`/`pos_embed`/`patch_embed.*`
  from any `pr_*` checkpoint and `--skip_norm` deletes the final norm, so **no arm inherits
  procedural embeddings** — an earlier draft said otherwise and was wrong. The remaining
  difference between the two middle rows is proc's **1-D parameters** (LayerNorm gains, biases)
  and the qkv pooling. The 1-D reading has since been superseded: the init screen (fig10, docs
  3.10.9.11) finds the LayerNorm gain correlates at -0.97 but fails the `ftb4o` gate, while the
  qk/v scale ratio passes all four. Secondary confound: the two rows randomise the arrangement
  differently (uniform `randperm` vs the rank order of the random tensor).
* **fig5** — `ftb4o` (rho-only) and the clipping arms are NOT on this matrix. An earlier draft
  said `ftb4o` keeps procedural blocks 8-11; that is **wrong** — `upscale_random_*` calls
  `pr_load_model(path="")` (main.py:795-808), so the model is fully random and the checkpoint is
  used only as the measurement target. What actually keeps `ftb4o` off the matrix is that it
  calibrates blocks **0-7**, not 0-8, and it changes weight *scale* rather than weight *values*.
  It is on fig10, where it is the load-bearing arm.
* **fig6/fig7** — at nine blocks the shuffled arm scores the *same* as intact (80.16 vs 79.99,
  Welch p=0.50) and as full procedural init (80.09, p=0.50) — not higher, an earlier caption
  overstated this. What it does do is remove all four loss spikes per run. At one block the
  shuffle costs 0.97 (p=0.01). Both cannot be explained by the same mechanism yet.
* **fig1/fig2** — `train_loss` is computed on mixup+cutmix+label-smoothed targets, so it is not
  on the same scale as `test_loss`. Only across-arm comparison of the *same* quantity is used;
  no train-test gap is ever computed.
* **fig2** — correlations are across arm means and are descriptive, not causal.
* `ftbqm1dqk` and `ftbqm1dvo` are **n=2**: the third seed of each is still training and is
  deliberately excluded from `ARMS` in `make_figures.py`, because `summary()` reads each seed's
  last logged epoch and the winners are behind until ~epoch 250 — a partial read is biased against
  exactly the arms under test. Add the seed to `ARMS` once its `log.txt` reaches 300 lines.
* **fig5** — the bottom four rows (+1.37 to +2.08) are not significantly different from one
  another (Welch p = 0.073 to 0.50). Read the matrix as two groups, ~+1.7 and ~+0.45, not as a
  ladder. The caption says so; do not let a talk track turn it back into a ladder.
* **fig10** — six of the ten arms are numerically identical on the surviving statistic, so it
  explains the split between groups and none of the 0.73-point spread within the middle pack.
  `ftb4o` is single-seed. r = 0.97 over ten points with six of them tied is a much weaker claim
  than the number suggests: the two groups separate, the line through them is decoration.

## Still missing before submission

The **LayerScale baseline** (`init_values=` in `models/vision_transformer.py`, currently
`nn.Identity` in every arm). Every gain here is measured against a ViT *without* the standard
modern residual-scaling mechanism, which is the first thing a reviewer will ask about.
Four arms: random and recipe, each ±LayerScale.
