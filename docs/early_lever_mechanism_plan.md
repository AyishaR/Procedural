# Early lever: what is the mechanism? Evidence so far and an experiment plan

Draft for review, 2026-09-20. Nothing in this file is launched or implemented; section 7 lists what would have to be built.

**Committed design** (decided 2026-09-20, `proc_init_recipe.md` section 11): blocks 0-7 get the effective scales of q, k, fc1
and the LayerNorm statistics (the *profile*, P), blocks 1-7 additionally the rank-one attention sink calibrated to the prefix's
attention entropy (S) and the rank-one fc1 gate calibrated to the prefix's mean pre-activation (G); v, proj, fc2 and blocks 8-11
stay timm. C = P + S + G: `ftbanapermb7i` 80.37 (kdyck), `ftbanakpermb7i` 79.90 (ksd); random 78.08 +- 0.19 (n = 3); prefixes
79.89 (kdyck, n = 3) and 80.05 (ksd, n = 1). One seed per arm unless stated; seed resolution 0.45.

**The question.** Why does C train to a better network than a random init? "The early blocks learn slowly" (section 12 of the
recipe doc) is one candidate. The plan below does not assume it. It first states what any explanation has to account for,
using only measurements, then lists competing explanations with what each predicts, then the experiments that separate them.

**How evidence is tagged.** [C] = measured on the committed design (blocks 0-7, mean gate, write side at timm).
[O] = measured on older recipes (blocks 0-8, scale-only or bias-based sink / gate, other gate target). [O] results are priors:
they motivate experiments but are not assumed to hold for C. Every number below is last-epoch accuracy or read off
the scripts named next to it.

Scripts written for this note: `plots/verify/mechanism_phase0.py` (training traces already logged: lens, entropy, write ratios,
losses), `plots/verify/mechanism_autopsy.py` (per-epoch checkpoints on fixed inputs, GPU job 29744556, 12 min),
`plots/out/mechanism_autopsy.json` (its output).

---

## 1. What has to be explained

### 1.1 The block-7 lens transient separates winners from the rest, but does not rank winners

Lens = final norm + head applied to the class token after block 7, top-1 on the validation set, measured at epochs 4, 9, 14, 19,
29, 39, ... During the first ~50 epochs of a random-init run, block 7 alone already classifies: 47.8% at epoch 29, when the full
network is at 64.2%. In every arm that gains this transient is suppressed.

| set of arms (last-epoch final available) | n | Spearman(lens7 peak, final) | Spearman(lens7 at epoch 29, final) |
|---|---|---|---|
| all cached arms, both tasks | 57 | -0.80 | -0.81 |
| kdyck | 39 | -0.81 | -0.83 |
| ksd | 18 | -0.92 | |
| early-lever family (no late-lever, no suffix arms) | 42 | -0.85 | -0.84 |
| ... of those, final >= 79 only | 21 | -0.29 | -0.31 |

- As a classifier it is sharp: "lens7 at epoch 29 <= 30" identifies final >= 79 in 39 of 42 early-lever arms (exceptions
  `ftbanaksw` 78.73 / 22.6, `ftbanakperab7` 78.64 / 23.9, `ftbanape` 79.44 / 30.1).
- As a ranker of winners it fails (-0.3): kdyck scale-only recipes peak at 23-29 and reach 79.8-80.7 [O], prefix-like arms
  peak at 4-16 and reach 79.6-80.4. Within matched pairs it does order: all four mean-gate arms have a lower peak and a higher
  final than their active-unit twins [C].
- So the transient marks a *regime* (with a border near 30), not a dose. Whatever the mechanism is, it should explain a
  threshold-like relation, and the lens should be treated as a readout whose causal status is open (section 4, C-experiments).

### 1.2 Where the class is computed: depth profile during training and at the end

Lens accuracy per block (`mechanism_phase0.py profile`), and a *trained* linear probe on block 7 / 9 features (class token + mean
patch token, 20 training images per class, 10k validation images; `mechanism_autopsy.py --probe`). `ftbqu` (timm with q, k x2.18,
78.05, n = 3) stands in for random because no plain random run kept its per-epoch checkpoints (section 3, R0).

| arm | epoch 29: lens b7 / b9 / b11 | end: lens b9 / b10 / b11 | probe b7, epoch 29 / end | probe b9, epoch 29 / end |
|---|---|---|---|---|
| random (`r`; probe: `ftbqu`) | 47.8 / 59.5 / 64.2 | 39.1 / 73.3 / 78.1 | 40.1 / 42.7 | 50.5 / 61.0 |
| C kdyck `ftbanapermb7i` [C] | 6.7 / 32.6 / 59.5 | 3.9 / 51.2 / 80.4 | 16.2 / 25.9 | 31.6 / 45.1 |
| kdyck prefix `ftb4i_kdyck` | 4.1 / 15.9 / 49.8 | | 11.4 / 20.9 | 18.4 / 40.3 |
| C ksd `ftbanakpermb7i` [C] | 3.8 / 40.2 / 61.4 | 7.0 / 61.2 / 79.9 | | |
| kdyck scale-only `ftbanap` [O] | 28.4 / 48.0 / 59.4 | | 28.4 / 40.0 | 38.4 / 62.1 |
| loser `ftbana` [O] | 41.2 / 53.6 / 58.8 | | | |

- In a random-init run the first seven blocks do three quarters of the classification early on, and even the final network is
  linearly class-decodable in the middle (probe 43% at block 7, 61% at block 9).
- In C and in the prefix the class information is *genuinely* lower in the middle of the final network (probe 21-26% at block 7,
  40-45% at block 9), not just misaligned with the head: the class is computed in blocks 10-11.
- The scale-only winner `ftbanap` [O] ends with a random-like depth profile (probe 40 / 62) and still gains +2.2. Late class
  formation is therefore a footprint of prefix-like winners, not a necessary condition for the gain. Any story built only on
  "the late blocks must learn the class" has to deal with this arm (or the arm has to be re-examined on the committed block range).
- Same picture in the attention maps at the end of training (mean attention distance per block, epoch 289): random 66 58 53 60 65
  76 86 97 107 110 109 111; C kdyck 62 73 56 68 71 49 59 58 77 83 106 120; kdyck prefix 61 87 74 71 65 58 60 65 66 78 98 121;
  `ftbana` 60 64 58 54 60 77 77 89 101 107 105 116. Winners keep attention local through block 7-8 and integrate globally only in the last two
  or three blocks; random and the losers go global from block 6 on.

### 1.3 Slow start, persistent fit deficit: the gain is regularisation, not a better fit

| arm | test acc at 19 / 49 / 99 | final | train loss at 299 |
|---|---|---|---|
| random `r` (seed 0) | 55.3 / 69.6 / 75.1 | 78.28 | 2.289 |
| C kdyck [C] | 49.8 / 66.7 / 74.0 | 80.37 | 2.457 |
| kdyck prefix | 38.5 / 57.1 / 71.0 | 79.91 | 2.616 |
| C ksd [C] | 52.9 / 68.6 / 75.0 | 79.90 | 2.374 |
| `ftbanap` [O] | 46.0 / 68.6 / 75.5 | 80.24 | 2.364 |
| `ftbana` [O] | 46.3 / 67.2 / 74.0 | 76.61 | 2.209 |
| `ftbanak` [O] (ksd scale-only) | 46.2 / 69.0 / 75.9 | 77.86 | 2.192 |

Winners are behind random until epoch ~100-120 and end with a *higher* training loss (mixup targets); the two losers fit best
and test worst. Over ~100 arms the final training loss correlates positively with test accuracy (memory note
`fit-loss-predicts-test-acc`). An explanation of the early lever has to produce a persistent fit deficit, not only a delay.

### 1.4 Who receives the gradient in the first epochs

Mean gradient norm per tensor, blocks 1-7 against blocks 8-11 (`grad_norms_*.json`, 82 runs with a final): the early / late ratio
over epochs 0-9 predicts the final with Spearman -0.67 (q), -0.65 (v), -0.59 (fc1), -0.50 (k), -0.42 (fc2).

- random: early blocks receive 2.8x the per-block gradient norm of late blocks at epochs 4-9 (1.6x at the end); full proc init:
  0.32x at epoch 4, 1.0x at epoch 29, 1.3x at epoch 49 (wandb `grad_norm`).
- C kdyck at epoch 0 [C]: early / late = 0.0009 / 0.0063 (q), 0.0045 / 0.102 (fc1), 0.011 / 0.102 (fc2); v and proj are comparable.
  Early fc1 reaches a third of the late value only at epoch 19. In the active-unit twin early fc1 is at 0.036 / 0.066 after ONE epoch.
- loser `ftbana` [O]: early q / k receive 1.5x (epoch 0) to 4x (epoch 4) the late gradient, like random.

Caveat that matters for interpretation: Adam normalises gradient magnitude. A small gradient does not make a tensor move slowly;
it tells that the loss currently does not depend on the tensor, and it lowers the signal-to-noise ratio of its steps.

### 1.5 Realised relative weight change per epoch (the quantity "slow steps" is about)

||W(e+1) - W(e)|| / ||W(e)||, mean over blocks 1-7, in percent, at epochs 0->1 / 4->5 / 9->10 / 19->20 / 49->50
(`mechanism_autopsy.py`, part B). Random-level reference = `ftbrhosl`, whose early blocks are untouched timm with normal steps.

| arm | raw rms q; fc1 (x timm) | q | fc1 | fc2 |
|---|---|---|---|---|
| random-level | 1.0; 1.0 | 6.9 / 18.6 / 26.3 / 29.7 / 29.0 | 4.4 / 11.1 / 21.6 / 26.8 / 26.3 | 4.1 / 9.6 / 20.5 / 25.7 / 24.6 |
| C kdyck [C] | 3.5; 0.86 | 1.9 / 7.4 / 11.8 / 13.3 / 21.1 | 5.9 / 9.5 / 16.0 / 22.5 / 24.4 | 2.3 / 3.2 / 9.5 / 20.3 / 18.5 |
| C ksd [C] | 4.7; 3.2 | 1.6 / 5.1 / 8.0 / 11.6 / 24.0 | 2.4 / 4.3 / 7.3 / 13.6 / 25.2 | 5.3 / 9.7 / 16.7 / 24.5 / 26.8 |
| ksd scale-only `ftbanak` [O], final = random | 4.7; 3.1 | 1.8 / 4.2 / 8.1 / 11.6 / 23.4 | 2.0 / 4.1 / 7.5 / 14.0 / 24.4 | 2.1 / 3.9 / 6.9 / 14.2 / 23.7 |
| `ftbana` [O], 76.61 | 1.3; 0.36 | 7.5 / 14.4 / 23.0 / 26.1 / 28.0 | 15.7 / 27.3 / 37.5 / 30.8 / 26.4 | 6.7 / 12.9 / 24.5 / 27.8 / 24.8 |
| `ftbanal` [O], 79.78 (= `ftbana` + lr scales) | 1.3; 0.36 | 3.3 / 6.5 / 10.0 / 14.2 / 25.3 | 7.7 / 11.7 / 18.3 / 25.4 / 27.1 | 7.5 / 11.8 / 17.9 / 27.5 / 25.5 |

- The slow steps are real where the raw matrix is large: q and k of C move 2.5-3.7x more slowly than random for the first ~20
  epochs. **They are also transient by themselves:** by epoch ~100 every arm moves at the same relative rate (18-21% per epoch),
  because AdamW drives the norms to an equilibrium that does not depend on the init. The "slow phase" is the first ~50 epochs.
- On kdyck fc1 is NOT slow in C (raw 0.86 x timm; its relative change equals random's). Only q and k are.
- **Slow steps are not sufficient on ksd:** `ftbanak` has the same realised slowness of q, k, fc1 as C on ksd (and slow v, proj,
  fc2 on top) and ends at random level. What separates it from C on ksd is sink + gate, i.e. the forward state.
- `ftbana`, the arm below random, is the *fast* one: its small raw fc1 (0.36 x timm) moves 2.5-3.6x faster than random's.
- The gate slows fc2 although fc2 is at timm scale with normal nominal steps (C kdyck fc2 2.3 / 3.2 against random 4.1 / 9.6):
  a closed MLP gives fc2 incoherent gradients. Dormancy can thus be produced by the forward state, not only by |W|.

### 1.6 The functional state of blocks 1-7 over training (fixed 256 training images)

D = how much a block changes the *differences between patch tokens* of an image, ||Pi(x_out - x_in)|| / ||Pi x_in||.
"att spec" = token-specific part of the attention write relative to the rms token norm (the part that mixes tokens differently
per query; the common part is a broadcast). Values at epochs 0 / 4 / 9 / 19.

| arm (final) | D | att spec | what the MLP does |
|---|---|---|---|
| random-level (78.1-78.3) | 0.39 / 0.42 / 0.41 / 0.47 | 0.04 / 0.11 / 0.21 / 0.22 | half the units active from step 0, GELU rms 0.33 |
| kdyck prefix (79.89) | 0.015 / 0.03 / 0.05 / 0.17 | 0.006 / 0.017 / 0.025 / 0.063 | closed for ~10 epochs (z mean -2.4 .. -3.1, GELU rms 0.04-0.06) |
| C kdyck (80.37) [C] | 0.05 / 0.08 / 0.13 / 0.27 | 0.008 / 0.014 / 0.070 / 0.157 | closed for ~10-20 epochs (z mean -2.3 .. -3.7, GELU rms 0.03-0.06) |
| active-unit twin (79.63) [C] | 0.07 / 0.18 / 0.30 / 0.38 | 0.013 / 0.065 / 0.177 / 0.259 | gate gone after ONE epoch (active 0.04 -> 0.28, z mean -0.44 -> -0.13) |
| `ftbanap` (80.24) [O] | 0.05 / 0.13 / 0.19 / 0.37 | 0.017 / 0.069 / 0.129 / 0.254 | never closed, but quiet: z std 0.2, GELU rms 0.11, grows over ~20 epochs |
| `ftbana` (76.61) [O] | 0.06 / 0.19 / 0.25 / 0.41 | 0.026 / 0.110 / 0.181 / 0.216 | same start as `ftbanap`, wakes ~2x faster (z std 0.70 against 0.29 at epoch 9) |
| C ksd (79.90) [C] | 0.12 / 0.31 / 0.37 / 0.40 | 0.012 / 0.031 / 0.095 / 0.157 | MLP token-specific write random-like from epoch ~2 on; attention stays common ~10 epochs |
| ksd prefix (80.05) | 0.22 / 0.32 / 0.33 / 0.33 | 0.076 / 0.082 / 0.098 / 0.078 | loud MLP (GELU rms 0.6), attention half common, flat for 50 epochs |
| `ftbanak` (77.86) [O] | 0.50 / 0.51 / 0.49 / 0.51 | 0.053 / 0.095 / 0.150 / 0.276 | loud random MLP from step 0 (GELU rms 0.56) |

- Random-init early blocks scramble token differences by 40% per block from the first step. Every winner starts far below that and
  opens over ~10-30 epochs; the sink holds the attention entropy near its initial value for ~4-9 epochs and is gone (random-like
  entropy, 3.4-3.9 nats) by epoch 19 in all of them.
- A transparent start is not sufficient: `ftbana` starts as transparent as `ftbanap` and ends 3.6 points lower.
- Slow steps are not sufficient: `ftbanak`, `ftblrm` [O].
- The one pattern that fits all reference arms is the conjunction: **start transparent AND open slowly** (random: neither, baseline;
  `ftbana`: transparent but fast, below random; `ftbanak` / `ftblrm`: slow but scrambling from step 0, baseline; prefixes, C,
  `ftbanap`, `ftbanal`: both, gain). It is a fit to ~10 single-seed arms across two recipes, hence a hypothesis, not a result.
- On ksd what stays "closed" for the first ~10 epochs is only the token-specific attention; the MLPs are active per token almost
  immediately. If a single common factor exists, delayed *token mixing* is a better candidate than delayed MLP opening.

### 1.7 Phase 0 on the whole zoo (run 2026-09-20, no training)

Four sharded GPU array jobs on the test partition (29744588 zoo, 29744596 depth, 29744597 fit gap, 29744623 class-token pathway;
28 single-GPU tasks, ~14 min wall-clock in total): `plots/verify/mechanism_phase0_gpu.py`, aggregation
`mechanism_phase0_aggregate.py`, figures `mechanism_phase0_plots.py` -> `plots/out/phase0/figs/f1..f8` (png + pdf). The zoo is every
finished run that kept per-epoch checkpoints: 107 runs, 70 arms (`plots/out/mechanism_run_index.json`). "Early-lever arms" =
blocks 9-11 untouched (q, k, fc1 at timm scale) and no loud-late signature (lens7 < 3 while the probe finds the class): 92 runs,
57 arms, 72 of them kdyck.

**(a) The gain is a smaller generalisation gap (f3).** Final checkpoints on 25k training images WITHOUT augmentation and 25k
validation images:

| arm | clean-train top-1 / CE | val top-1 / CE | gap CE |
|---|---|---|---|
| kdyck prefix | 96.56 / 0.263 | 80.20 / 0.930 | 0.667 |
| C kdyck | 97.56 / 0.223 | 80.59 / 0.947 | 0.724 |
| C ksd | 98.38 / 0.198 | 79.90 / 0.992 | 0.795 |
| scale-only `ftbanap` [O] | 98.95 / 0.176 | 80.35 / 0.987 | 0.811 |
| active-unit twins | 98.7-99.0 / 0.17-0.18 | 79.6-79.7 / 1.02-1.04 | 0.84-0.86 |
| write-matched | 99.08 / 0.167 | 79.06 / 1.088 | 0.921 |
| timm random `r` / random-level | 99.3-99.4 / 0.16 | 78.2-78.5 / 1.10-1.18 | 0.93-1.03 |
| `ftbanak` (ksd scale-only) [O] | 99.27 / 0.157 | 77.92 / 1.187 | 1.030 |
| `ftbana` [O] | 99.40 / 0.157 | 76.86 / 1.254 | 1.097 |

Spearman(gap, validation accuracy) = -0.94 over the 14 models. A random-init ViT-B fits the clean training images to 99.4%;
the winners fit them less and are better calibrated on validation. Whatever the early lever does, its end effect is less
memorisation. (Open tension: on IN-100, where overfitting is stronger, the early proc blocks cost accuracy [O].)

**(b) Which early-phase quantity predicts the final (f1, f6, f7).** Rank correlation with the last-epoch accuracy, and partial
correlation given the lens:

| quantity (blocks 1-7 unless stated) | all 107 | kdyck early-lever arms (72) | partial given lens, kdyck |
|---|---|---|---|
| final training loss | +0.81 | +0.75 | +0.62 |
| head lens at block 7, epoch 29 | -0.78 | -0.75 | |
| trained probe at block 7, epoch 29 | -0.72 | -0.68 | -0.16 |
| realised change of q, k, fc1 at epoch 9 (19) | -0.44 | -0.74 (-0.72) | -0.54 (-0.45) |
| GELU rms at epoch 19 (29) | -0.60 (-0.72) | -0.67 (-0.66) | -0.58 |
| raw rms of q after the first epoch | | +0.70 | +0.40 |
| D at epoch 9 | -0.44 | -0.58 | -0.31 |
| token-specific attention write, epoch 9 | -0.49 | -0.50 | -0.15 |
| realised change of fc1 in blocks 9-11 at epoch 19 | | -0.84 | -0.60 |
| norm of q in blocks 9-11 at epoch 19 | | -0.64 (ksd -0.87) | |

On kdyck the realised slowness of the early input matrices predicts the final about as well as the lens and *beyond* it; this
is the strongest existing argument for running A1. On ksd (18 runs) the lens, the block-7 probe and the attention entropy after
epoch 0 lead (-0.87, -0.87, -0.83): the sink dose-response. Among winners only (final >= 79) nothing but the fit deficit orders
the arms (+0.6). Two late-block quantities predict as well as any early one: the late blocks move MORE in epochs 0-4 in winners
(+0.45) and less at epoch 19 (-0.72 .. -0.84), and their q norm grows less. The late blocks' early dynamics are part of the
picture (M3), not only the early blocks'.

**(c) Counterexamples to "transparent start + slow opening".** (i) `ftbqmln` and `ftbqm1d` (78.2 / 78.0, n = 3 each) have the same
D, token-specific writes, raw scales and realised q / k / fc1 steps as their winning twins `ftbqmlnvo` / `ftbqm1dvo` (79.9 / 79.1,
n = 3 each); the pairs differ in the write side: timm-scale v / proj give a *common* attention write twice as loud in the first
epochs (0.24 against 0.13 of the token norm; C: 0.13) and a v that moves half as fast. (ii) The write-matched arms are the most
transparent of all (D 0.004 -> 0.05 at epoch 9) and end at 78.9 with a transient of 32, which appears late (epoch 39), after the
sink has gone. (iii) `ftbvd` [O], timm with v x 0.46 and nothing else, gains +0.66 (n = 3). So the conjunction of section 1.6 is
not sufficient; **the loudness of the common (broadcast) attention write in the first ~10 epochs is a further candidate**, and it
is invisible to D because a common write leaves token differences unchanged.

**(d) Depth (f2).** Probe per block of the final network, blocks 5 / 7 / 9: timm random `r` 32 / 48 / 69; C kdyck 17 / 25 / 44; kdyck
prefix 15 / 20 / 39; C ksd 15 / 26 / 48; but `ftbanap` 27 / 40 / 62 and `ftbana` 28 / 40 / 62. Late class formation separates the
sink-and-gate winners from everything else, and does NOT separate the scale-only winner from the loser with the same init.
There are at least two kinds of winners; one explanation may not cover both.

**(e) Not a class-token shortcut (f8).** At block 7, epoch 29, a probe on the mean patch token finds the class nearly as well as
a probe on the class token (random-level 27 against 32; C 10 against 11). The transient is early class alignment of the whole
token stream. A late-class-token control (CaiT-style) is therefore not expected to explain the effect; it stays a cheap optional arm.

**(f) The sink.** In C the most-attended key holds 52-62% of the attention mass after epoch 0, 26-28% at epoch 4, 12-13% at epoch
9. It is neither the class token (1% of images) nor a fixed position (2%): an image-dependent patch. In the kdyck prefix it is
the class token in 38% of the images. With the mean gate the sink lasts longer than with the active-unit gate (entropy at epoch 4:
0.71 against 2.69): a closed MLP keeps the stream direction the sink reads stable. Sink and gate interact; the D cells P + S
against P + S + G measure that.

---

### 1.8 Phase 0 with the real random reference (R0, 2026-09-21)

`r0` (plain timm random, all 300 per-epoch checkpoints, final 77.42) went through every Phase-0 analysis (5 one-GPU tasks, job
29751186; `--only r0 --tag r0`, results in `plots/out/phase0/*_shardr0.json`, autopsy merged into `mechanism_autopsy.json`). Figures
f1-f8 still show the stand-ins.

- *Which stand-in was faithful.* `ftbrhosl` (timm early blocks) matches `r0` in every early-stack quantity (D 0.386-0.489 against
  0.391-0.475, entropy 5.23 / 4.19 / 3.93 against 5.23 / 4.17 / 3.88, q drift 6.9 / 18.6 / 26.3 / 29.7 against 7.0 / 18.7 / 25.7 /
  28.9 x 1e-2): everything stated with it stands. `ftbqu` (q, k x 2.18) does not: q drift 2.6x lower, entropy 4.58 at init, final
  block-7 lens 5.9 against 25.5.
- *Realised slowness, now against the real random.* Relative weight change per epoch in blocks 1-7, C / `r0` at epochs 0, 4, 9, 19:
  kdyck q 0.27, 0.40, 0.46, 0.46; k 0.28, 0.27, 0.31, 0.37; fc1 1.39, 0.87, 0.76, 0.84 (not slow). ksd q 0.23-0.40, k 0.23-0.38, fc1
  0.35-0.55. Section 1.5 stands.
- *Depth.* Trained probe at block 7: `r0` 40.7 at epoch 29 and 48.9 at the end, C 16.6 and 25.3; C forms the class about two blocks
  later (block 9: 44.3 against 71.1). The transient is visible in `r0`'s lens by depth: block 7 48.5 -> 42.2 -> 25.2 and block 6
  39.3 -> 22.4 -> 9.4 at epochs 29, 99, 299, while the probe at the same blocks keeps rising (40.7 -> 48.6 -> 48.9): the information
  stays, the head stops reading it there.
- *Fit gap.* `r0` fits exactly like the old random run (clean-train 99.38% / CE 0.158 against 99.37% / 0.159) and generalises worse
  (val CE 1.186 against 1.128): its 0.66-point deficit is a generalisation difference at identical fit, not a protocol difference.
- *Class-token path: statement (e) of 1.7 needs softening.* With the real random the class token leads the mean patch token at
  block 7 (probe 18.0 against 11.8 at epoch 9, 36.2 against 22.9 at epoch 29; at block 11 53.0 against 30.1); the stand-in `ftbqu`
  had understated this (31.8 against 27.4). In every winner the two are equally low (C 10.7 / 10.4, prefix 7.0 / 6.9, P-only 22.7 /
  16.2). In `r0` the class token receives the larger attention update early (0.33 against 0.21 of its norm at epoch 4, 0.50 against
  0.33 at epoch 9, row entropy 3.3-3.6); in C its row sits on the sink (entropy 0.92 at epoch 4) and what it receives is the common
  write (cosine 0.97 with the patches' mean update). So in plain random the early class readability of block 7 is carried more by
  the class token than by the patches, and the winners remove that lead. A class-token pooling pathway is a candidate concrete form
  of M3; not tested yet.

## 2. Candidate mechanisms and what each predicts

They are not mutually exclusive. The experiments in section 4 are chosen so that each pair is separated by at least one arm.

| id | mechanism | core claim | existing evidence for | existing evidence against / open |
|---|---|---|---|---|
| M1 | **Slow relative steps** | large raw q, k (fc1) move slowly under Adam; that alone produces the gain | [O] `ftbana` 76.61 -> `ftbanal` 79.78 by lr scales alone; realised slowness confirmed (1.5) | not sufficient: `ftbanak` (measured equally slow, random level), `ftblrm` 77.73 [O]; irrelevant for the late lever (`ftbrhopl`); fc1 is not slow on kdyck in C; never tested on C |
| M2 | **Slow opening** (state preservation) | the early stack must start transparent for token differences and open over tens of epochs; slow steps matter only as the means that keep the state alive | the conjunction fits all arms of 1.6; mean gate (lasts ~15 epochs) beats active-unit gate (lasts 1 epoch) 4 / 4 [C] | which part has to stay closed (token mixing, MLP, both) is open; ksd suggests mixing; the lens does not rank winners |
| M3 | **Learning order / depth allocation** | because the early stack is dormant while the class computation is first formed, blocks 8-11 take it over; the final network computes the class late | depth profile, probes, attention distance (1.2); gradient allocation (1.4); `ftbrhosl` (slow LATE blocks) has the highest transient of all, 62.8, and no gain [O] | `ftbanap` gains with a random-like final depth profile [O] |
| M4 | **Direct-path attenuation** | the head reads block 7 through the identity path; whatever lowers the early blocks' gain on that path helps. Would unify both levers: loud late writes shrink the early contribution after the final LayerNorm (lens7 ~ 0 in every late-lever arm) | late-lever arms [O] | untested; could be a restatement of M3 |
| M5 | **Regularisation of early features** | restrained early plasticity lowers the capacity to fit the augmented training set with early, class-aligned features; the benefit is the persistent fit deficit itself | 1.3; `ftbana` = most plastic early fc1, best fit, worst test | does not say why it must happen early in training; M2 / M3 also produce a deficit |
| M6 | **Component-specific roles** | sink: shared broadcast vs sharpness vs frozen routing; gate: MLP off vs operating regime vs duration of dormancy | ksd dose-response with entropy [O]; mean vs active-unit twins [C] | not separated on C; sink and gate never separated with the rank-one constructions |
| M7 | **Lazy early / rich late** | raw scale sets the feature-learning regime: large raw matrices keep the early blocks' random features nearly fixed while timm-scale late blocks learn; a small raw fc1 (`ftbana`) learns greedily and overwrites its init | 1.5: `ftbana` fc1 changes 27-38% per epoch at epochs 4-9 | same data as M1 / M2; needs its own test (realised feature change on fixed inputs) |

**After Phase 0 (1.7).** M1 gains correlational support on kdyck (realised slowness predicts beyond the lens) and stays refuted as sufficient on ksd. M2 as stated is not sufficient (three counterexamples); it needs a third condition or a different variable, and the loudness of the common attention write is the candidate (**M2b: quiet broadcast**; `ftbvd` +0.66 [O] is its statistics-free data point). M3 is supported as a footprint of sink-and-gate winners and by the late-block predictors, and is not what separates `ftbanap` from `ftbana`. M5 is established as the *end effect* (gap CE -0.94 with accuracy) whatever the route. A strict class-token shortcut is ruled out.

Prediction matrix for the main experiments (section 4). "gain" = final >= 79.5; "lost" = at random level; T = lens transient.

| experiment | M1 slow steps | M2 slow opening | M3 learning order | M5 regularisation |
|---|---|---|---|---|
| A1: C init, steps compensated | lost, T up | lost only if the sink / gate state decays faster (measurable); else kept | kept if blocks 8-11 still learn first | partly lost |
| B1: timm + zero-init early branches (transparent, normal steps) | no gain | no or small gain (opens too fast) | small gain if T is suppressed | small |
| B2: B1 + restrained early blocks with release (no checkpoint statistics at all) | gain (steps are what matters) | gain | gain, T suppressed | gain |
| B3: timm + restrained early blocks only (not transparent) | gain | no gain (as `ftblrm`, `ftbanak`) | gain only if T is suppressed | some gain |
| C1: C init + auxiliary head loss at block 7 for 50 epochs | kept | kept | lost | kept or lost |
| S+G on timm (structure only) | no gain | gain only if it stays closed long enough without slow steps | gain if T suppressed | some |

---

## 3. Phase 0: no training (existing logs and per-epoch checkpoints)

Done: 1.1-1.7 (P0.1 mediator screen, P0.2 depth, P0.3 fit gap, P0.4 sink identity all ran on 2026-09-20; R0 is training). The list below is kept as the record of what was asked; remaining Phase-0 ideas: probe on training against validation images per block (where the memorisation sits), and the same screen once R0 and wave 1 have checkpoints.

- **P0.1 Mediator screen over the whole zoo.** Run part A / B of `mechanism_autopsy.py` on every run that kept per-epoch checkpoints
  (~80 runs x epochs 0, 4, 9, 19, 29; ~20 GPU-minutes, sharded over several GPUs of the test partition so the wall-clock is a
  few minutes) and ask which early-phase quantity predicts the final across arms and tasks: lens7,
  block-7 probe, D, token-specific attention write, token-specific MLP write, realised drift of q / k / fc1, gradient allocation.
  Partial correlations tell whether, e.g., D adds anything beyond the lens. This ranks M1-M3 on existing data before any new run
  and shows which arms each candidate mediator fails on. *Why first:* it costs nothing and section 1.6 is 12 arms, not 80.
- **P0.2 Probe against lens for all blocks and 6 epochs** (C, prefix, `ftbanap`, `ftbana`, random-level, both tasks): is "late class
  formation" present in every winner of the committed family, and absent in every loser? (1.2 has two epochs and two blocks.)
- **P0.3 Generalisation footprint.** Final checkpoints of C, random-level, `ftbana`, prefix on a training subset WITHOUT
  augmentation and on validation: is the fit deficit a smaller train / test gap (less memorisation), and at which depth does the
  training-set advantage of the losers sit (probe on train against probe on val per block)? Separates M5 from M3.
- **P0.4 Sink identity.** Which key carries the sink in C and in the prefixes (class token, a fixed position, an image-dependent
  patch) and what the broadcast vector contains (its probe accuracy). Needed for M6 before designing sink controls.
- **R0 (one training run, listed here because everything depends on it): plain timm random with per-epoch checkpoints, the init
  itself saved, fixed code.** The baseline `r` predates per-epoch checkpoints; all random-level references above are stand-ins.
  **LAUNCHED 2026-09-20 09:31, group partition, job 29744580 (continuation 29744581), seed 0**, `vitbase_runs/run_train_r0.sh`
  = the committed arm's script with only the init lines changed (`--init_method default`). Gated on verification job 29744579:
  init through main.py == timm seed 0 bit for bit (saved as `results/init_dumps/r0_s0.pth`), 2-rank smoke with checkpoint PASS.

---

## 4. Experiments (new training runs)

Common rules. Same training protocol as C. Every arm saves the init and per-epoch checkpoints and gets the same autopsy, so the
mediators (lens, probe, D, token-specific writes, realised drift) are compared uniformly. Exploratory arms start as 40-epoch jobs
with the *unchanged* 300-epoch schedule (~3 h) and are read out by "lens7 at epoch 29 <= 30" plus the mediators; promising arms
are promoted by resuming the same run, so nothing is wasted. The lens readout is valid for screening inside the early-lever
family (39 / 42) and must not be used for arms that suppress it by construction (C1, anything touching the late blocks) or to rank
winners; those run to 300. A contrast counts when it exceeds 0.45 against an n = 3 reference or replicates with the same sign on
both tasks; decisive contrasts get two more seeds (group F).

### Group A: is it the step size? Optimiser-only interventions on the bit-identical committed init

Motivation: the only direct evidence for slow steps is [O] (`ftbana` / `ftbanal`), on a recipe without sink and gate where fc1 was
*fast*, not where q / k were slow. In C the realised slowness exists only in q, k (and fc1 on ksd) and only for ~50 epochs (1.5).

- **A1 compensated steps.** lr x m (weight decay / m) with m = rms(W_init) / rms(W_timm) per tensor, read from the *launched* init
  (after sink and gate, which shrink the random part: s_q = 0.58 in block 1). q and k rows via the row mask, fc1 as a group.
  Two variants because the profile also makes the LayerNorm gains small and therefore fast: A1a = q, k (+ fc1), A1b = A1a + gains
  (m < 1). Both tasks. *Reads:* final; T; and above all whether entropy, token-specific attention write and D rise earlier than
  in C. If the gain is lost AND the state decays earlier -> M2 (steps as preservation). Lost with the state intact -> M1 / M7.
  Kept -> slow steps are not needed once sink and gate are there (as for the late lever, `ftbrhopl`).
- **A2 over-restrained.** lr x 0.5 on the same tensors (slower than natural). If slower opening is monotonically better (mean gate
  over active-unit gate points that way), A2 >= C. Distinguishes "threshold" from "dose"; one arm, kdyck.
- **A3 which tensor.** Compensate q, k only against fc1 (+ norm2 gain) only, on ksd where both are slow. One 40-epoch pair.
- (A4, optional, needs a lr-scale *schedule*: natural steps for 50 epochs, then compensated; and the reverse. Since the natural
  slowness fades by itself by epoch ~100, A4 is only worth it if A1 loses the gain.)

### Group B: is "transparent start + slow opening" enough, with no checkpoint statistics at all?

Motivation: section 1.6. If a statistics-free construction of the same two properties reproduces the gain and the footprints, the
procedural statistics are one implementation of a generic principle; if not, something specific to P, S, G is doing the work.
These are also the missing literature baselines (zero-init residual branches, LayerScale; `late-block-scaling-open-gaps` memory).

- **B1 transparent, normal steps.** timm init; proj = 0 and fc2 = 0 in blocks 1-7 (every early branch exactly off at step 0,
  D = 0), nothing else. Under Adam these layers leave zero within a few hundred steps: transparency without slow opening.
- **B2 transparent + restrained.** B1 plus lr x rho on all tensors of blocks 1-7 (rho ~ 0.3, the realised factor of 1.5; weight
  decay / rho). In C the slowness fades by itself, because the large norms relax to the AdamW equilibrium; an lr multiplier on
  timm-size weights does not fade. So the faithful version needs the schedule (rho released to 1 between epochs 30 and 60); a
  constant-rho arm is the cheap first look and doubles as the test of "persistent restraint" (M5).
- **B3 restrained only.** timm init, the same restraint, no zero-init. [O] predicts no gain (`ftblrm` 77.73); on the new
  protocol with a stronger and uniform restraint this is the cleanest test of M1 / M7 without any forward change.
- **B4 which sub-layer.** proj = 0 with restrained q, k, v, proj only (delayed token mixing) against fc2 = 0 with restrained fc1,
  fc2 only (delayed MLP). Motivated by ksd, where only the mixing is delayed. 40-epoch pair first.
- **B6 quiet broadcast.** timm init with v (and proj) x 0.5 in blocks 1-7, alone and on top of B1 / B2: the statistics-free version of what separates `ftbqmlnvo` from `ftbqmln` (1.7c); `ftbvd` [O] (v x 0.46 in blocks 0-8, +0.66, n = 3) is the existing data point. 40-epoch screen.
- **B5 LayerScale on blocks 1-7** (learnable per-channel branch scale, init 1e-4): the standard baseline; its opening speed is set
  by the LayerScale parameters' own steps. One arm.

### Group C: is the lens transient a mediator or a marker?

- **C1 force early class formation.** C init plus an auxiliary cross-entropy on the block-7 lens (weight ~0.3) during epochs
  0-50, then off. Everything else identical. If the gain disappears, early class alignment of the middle is causally harmful
  (M3); if it stays, the transient is a marker of dormancy and M2 / M5 carry the effect. Runs to 300 (the lens is manipulated).
- **C2 (exploratory) cut the direct gradient path on timm.** Forward unchanged; in blocks 8-11 the identity path is detached
  for the backward pass during epochs 0-30, so blocks 0-7 receive gradient only through the late branches. If T falls and a
  gain appears with no change of init or step size, the direct path is the culprit (M4).
- **C3 (exploratory, links the levers) boundary attenuation.** timm init, stream divided by a constant c after block 7. In a
  pre-norm network the late branches see the same inputs, their writes become c times louder relative to the stream, and the
  early blocks' direct-path gain drops by c: a late lever without touching a late weight. Prediction under M4: ~ `ftbrho`
  (79.7-80.0), lens7 ~ 0.

### Group D: what does each component do? (the factorial and replacement controls, on the committed definition)

Cells, mean gate throughout, each retained component calibrated on its own arm's stream; S, G in profile-off cells need
`renormalize: false` (measured: reachable 7 / 7 with the mean gate, recipe doc section 12).

| cell | kdyck | ksd | question |
|---|---|---|---|
| P | prepared (`ftbanapeb7`); [O] 79.2-80.7 on blocks 0-8 | prepared (`ftbanakpeb7`); [O] 77.9 | does the profile alone still win on kdyck on blocks 0-7? |
| P + S | new | new; [O] bias sink 79.58 | what the sink adds without gate |
| P + G | new | new; [O] bias gate 78.0-78.3 | what a *persistent* mean gate adds without sink ([C]: +0.4 .. +0.7 on kdyck over a gate that lasts one epoch) |
| S + G (structure only) | new | new | can the functional state replace the profile when nothing keeps it slow? (M2 predicts: only if it stays closed) |
| S + G + matched steps | later | later | lr scales that give S + G the committed arm's relative steps: can optimiser scaling replace the profile? |

Replacement controls, chosen after the factorial says which component matters where:

- sink -> *entropy-matched, query-dependent sharpening* (temperature on q, k to the same entropy; check that the common-query
  share stays low): sharpness against shared routing. And -> *uniform attention held for ~15 epochs* (q, k = 0 with restrained
  steps): "no token mixing" against "broadcast of one token's content". [O] kdyck scale-only winners have uniform attention.
- gate -> *fc2 = 0* (MLP off at step 0 but awake at full speed) against the gate (asleep for ~15 epochs): off-at-init against
  duration. -> *bias-only mean shift* (same operating point, no weight alignment). -> *branch attenuation matched to the gate's
  MLP write* (write less, same pre-activation regime). -> *dose*: mean target x 0.5 / x 1.5.

### Group E: regularisation (only if Phase 0 / group A point to M5)

C against random under weaker and stronger explicit regularisation (drop-path 0 / 0.2, no mixup): if the init is a regulariser,
its gain should shrink when explicit regularisation is strong and grow when it is weak. Four runs; deferred.

### Group F: seeds

Two more seeds of C per task and of the ksd prefix (n = 1 so far); two more seeds of whichever contrasts decide (expected: A1,
B2, C1, and S + G).

---

## 5. Proposed order

**Should group D come earlier? Yes, its four cells, not its replacement controls** (revised 2026-09-20). Three reasons. (i) Inside
the hypothesis that fits the data best (M2), the open question is *which part has to stay closed*, token mixing or the MLP
(section 1.6: on ksd only the mixing is delayed). P + S against P + G answers exactly that, on the committed definition, where
today only [O] analogues with bias-based constructions exist. (ii) The cells need no new training code: the extractor already
selects components; S + G needs `renormalize: false`, measured feasible with the mean gate. Groups B and C need new code (lr
schedule, auxiliary loss, detach), which can be written and tested while wave 1 trains. (iii) On ksd the component effects are
large (profile alone = random level [O], +2 with the structure), so single seeds resolve them; on kdyck the profile alone
already wins [O] and the component effects are <= 0.7, so there the cells start as 40-epoch screens and are promoted only if
the mediators move. The replacement controls (what *property* of sink or gate matters) stay after the cells, because the cells
decide which component deserves them.

| wave | runs (4 GPUs x ~22 h each unless "40 ep") | why this order |
|---|---|---|
| 0 | Phase 0 (GPU array jobs, no training); **R0 running** | free; R0 is the reference for every dynamics comparison |
| 1 (no new code) | A1a kdyck, A1a ksd; ksd cells P, P + S, P + G, S + G | steps (A1) and "which component, which part must stay closed" (D cells) where effects are resolvable at n = 1 |
| 1s (40 ep) | kdyck cells P, P + S, P + G, S + G; A1b, A2, A3 pair | screening; promoted by resume if the mediators move |
| 2 (new code, written during wave 1) | B1, B2 (constant rho, then release schedule), B3, C1 kdyck; B4 pair and B5 as 40-epoch screens | generality of "transparent + slow opening" and the mediator test |
| 3 | replacement controls for the component that matters; S + G with matched steps; C2 / C3 if M4 is alive | depends on waves 1-2 |
| 4 | seeds of the deciding contrasts; group E if M5 is alive | |

Wave 1 fills the ~6-7 concurrent 4-GPU slots; wave 1s was planned to share those slots (3 h each on H200) and runs on L40S
instead (section 5b).

## 5b. Wave 1 as launched (2026-09-20, the "middle path": 8 full runs + 7 screens)

The eight full runs wait on the shared H200 partition (4 GPUs each), submitted in priority tiers (nice 0 / 100 / 200). **Re-ordered
2026-09-20 11:55 on Simon's call that the structure-only cells (S + G, no effective scales) are the most interesting arms:**
`ftbck7sg` and `ftbc7sg` (and their continuations) now lead at nice 0, the two A1 runs follow at nice 50, the ksd cells at 100,
`ftbc7p` at 200. Both S + G cells are full 300-epoch runs, on both tasks. The group
H200 partition is blocked for a day or more by another group member's eight chained single-GPU jobs (account cap 13 GPUs /
603 CPUs). The shared H200 nodes were full as well (no node with 4 free GPUs + 192 CPUs + 750 GB), so the seven screens were
**moved to the L40S nodes** the same day (8 GPUs each, update frequency 4, so the batch of 4096 and the schedule are unchanged;
`NCCL_SHM_DISABLE=1` as for every 8-rank L40S job): four on the group L40S partition, which is not preemptable (the fifth hit the
account's CPU cap of 754), three on the shared L40S partition, which is preemptable (`--requeue` + auto-resume cover that). All
seven started within two minutes and train at 0.457 s per micro-step = 9.5 min per epoch, so epoch 39 is reached after ~6.9 h
(limit 9 h). Full runs have a continuation job; screens stop after epoch 39 of the unchanged 300-epoch schedule
(`--stop_after_epoch 39`) and are promoted by resubmitting `resume_zero_kbias.sh` with `STOP_AFTER_EPOCH=-1`. A promoted screen
would continue on other hardware (4 H200 instead of 8 L40S): same global batch and schedule, but a different per-rank data
split and reduction order from the promotion epoch on, i.e. not bit-comparable with an uninterrupted H200 run. That is inside the
seed noise, but it is a caveat the full runs do not carry. Every new run keeps per-epoch model checkpoints for epochs 0-59, then
every 10th and the last (`--analysis_ckpt_dense_until 60 --analysis_ckpt_every 10`): 84 instead of 300 files, because the
workspace stands at 9.15 TB of 10 TB (section 5c).

| arm | what | length, hardware | job (continuation) |
|---|---|---|---|
| `ftbc7a1` | A1, kdyck: committed init, lr x m on q rows, k rows, fc1 of blocks 0-7 | full, 4 H200 (shared) | 29745240 (29745241) |
| `ftbck7a1` | A1, ksd: the same | full, 4 H200 (shared) | 29745242 (29745243) |
| `ftbck7ps` | D cell, ksd: profile + sink | full, 4 H200 (shared) | 29745244 (29745245) |
| `ftbck7pg` | D cell, ksd: profile + mean gate | full, 4 H200 (shared) | 29745246 (29745247) |
| `ftbck7sg` | D cell, ksd: sink + gate on timm, size free, **fp16: DEAD at epoch 10** (non-finite loss, see below) | full, 4 H200 (shared) | 29745248 (continuation cancelled) |
| `ftbck7sgb` | the same cell restarted from scratch in **bf16** (init bit-identical: calibration hash `32e2e50a40b1b34f`) | full, 4 H200 (shared) | 29746035 (29746036) |
| `ftbck7p` | D cell, ksd: profile only | full, 4 H200 (shared) | 29745250 (29745251) |
| `ftbc7p` | D cell, kdyck: profile only | full, 4 H200 (shared) | 29745252 (29745253) |
| `ftbc7sg` | D cell, kdyck: sink + gate on timm, size free | full, 4 H200 (shared) | 29745254 (29745255) |
| `ftbc7ps` | D cell, kdyck: profile + sink | screen, 8 L40S (group) | 29745319 |
| `ftbc7pg` | D cell, kdyck: profile + mean gate | screen, 8 L40S (group) | 29745320 |
| `ftbc7a1g` | A1 + LayerNorm gains, kdyck | screen, 8 L40S (group) | 29745321 |
| `ftbck7a1g` | A1 + LayerNorm gains, ksd | screen, 8 L40S (group) | 29745322 |
| `ftbc7a2` | A2, kdyck: lr x 0.5 on q, k rows and fc1 | screen, 8 L40S (shared) | 29745327 |
| `ftbck7a3qk` | A3, ksd: q, k rows only | screen, 8 L40S (shared) | 29745324 |
| `ftbck7a3f` | A3, ksd: fc1 only | screen, 8 L40S (shared) | 29745325 |

Specs: the D cells are the committed specs with components removed by JSON surgery (`vitbase_runs/profile_ftbc7*.json`,
`profile_ftbck7*.json`), so scales and targets are identical across cells; each retained component is calibrated at init on its own
arm's stream. The structure-only cells carry `"renormalize": false`.

**Step multipliers** (`plots/verify/make_step_compensation.py`, job 29744814): m = rms(launched init) / rms(timm), init rebuilt on
the 256 training images of the launched protocol (it differs from the validation-image dump by <= 0.43%).

| | q | k | fc1 (block 0; blocks 1-7) | LayerNorm gains |
|---|---|---|---|---|
| kdyck | 3.04-4.32 | 3.95-4.46 | 3.43; **0.76-1.01** | 0.34-0.53 |
| ksd | 4.04-5.02 | 3.53-5.38 | 3.45; 2.30-3.54 | 0.27-0.61 |

On kdyck the mean gate takes 48-70% of fc1's size budget (s = 0.30-0.52), so the raw fc1 of blocks 1-7 sits at timm size: the
kdyck step-size arm is effectively a test of slow q and k. On ksd q, k and fc1 are all slow.

**Verification** (array 29744815, one task per arm, 2 L40S each, ~12 min): init dump through main.py; step-size arms: init ==
the committed arm's verified init bit for bit, lr file == measured m to 1e-5, rows [0, 768) and [768, 1536) masked, v untouched,
lr scale x wd scale = 1 in every decayed group, checkpoint carries the lr spec; profile cells: `verify_recipe_statement.py` and
`verify_joint_statistics.py` PASS; structure-only cells: `verify_structure_only.py` PASS (profile step is a no-op; only qkv and fc1
of blocks 1-7 change; every change rank one; all 14 targets matched; raw q, k up to 3.0x and fc1 1.10x timm); 2-rank smoke of every
arm; screening mechanics tested on `ftbc7ps` (stop after epoch 0 with exit code 0, thinned checkpoint not written, resume at
epoch 1 runs to the end). 13 of 15 tasks passed at once. The two structure-only arms failed only the smoke, which started at the
peak learning rate without warmup (loss NaN within a few steps; max |logit| 879 in block 1, first gradient norm 30-65). Under the
real schedule (300 epochs, 50 warmup epochs, stopped after epoch 1; job 29744861) both train in fp16 and in bf16, fp16 with some
skipped steps on kdyck. They are launched in fp16 like every other cell; if one diverges later in the warmup, bf16 is the
fallback and becomes a precision caveat for that cell.

**The ksd structure-only cell needed the fallback (2026-09-20).** `ftbck7sg` (fp16, started 12:10) lost a finite loss in epoch 9
at lr 3.9e-4 (20% of the peak), with no warning in loss or gradient norm (1.4-1.5); the in-job retry got through epoch 9 with
other augmentation draws and then died at the same iteration of epoch 10 on four resumes in a row (attempts 3-6), which ended
the job. Restarted from scratch in bf16 as `ftbck7sgb` (13:21, same node slot; `run_train_ftbck7sgb.sh` differs from the fp16
script in `--amp_dtype bfloat16`, job name and note only; its bf16 variant had passed the real-schedule warmup test). The kdyck
cell `ftbc7sg` runs on in fp16 (no event through epoch 15); if it fails, `run_train_ftbc7sgb.sh` is ready. Precision caveat: the
ksd S + G cell is bf16, its references (C-ksd, random, the other ksd cells) are fp16.

*What is known about the cause (not localised; autopsy deferred on Simon's call until it happens again).* Short GPU probes
(`plots/verify/fp16_headroom_probe.py`, `fp16_logit_tail.py`, `fp16_nan_hunt.py`; test partition, minutes each):
(i) at the epoch-8 checkpoint the only quantity anywhere near the fp16 range is the attention logit of blocks 1-2: max 6368 in
`ftbck7sg` and 3200-4080 in `ftbc7sg`, against 386-512 in the committed ksd arm (the structure-only sink sits on timm-size q, k
with a query side that is only 65-80% common, so the same mean entropy needs a ~10x larger rank-one logit); stream, fc1, GELU
and fc2 values are all < 30; (ii) the logit tail over 51,200 augmented + mixup training inputs is thin (median 3452, max 6189 =
9.4% of 65504); (iii) 1.28 M training inputs through the epoch-9 checkpoint under fp16 autocast give no non-finite output at
all. So it is not a rare input overflowing the logits at end-of-epoch weights; it happens at weights inside epoch 10. A 4-GPU
replay of the failing resume on L40S reproduced one event (rank 3 only, epoch 10 micro-step 1847) and the new crash-path dump
caught it: `results/init_dumps/tmp_nanrepro_ftbck7sg/s0/nan_dump_e10_it1847_{weights,rank3}.pt` (385 MB, kept);
`plots/verify/fp16_nan_autopsy.py <weights> <batch>` names the first non-finite module under fp16 and checks bf16 on the same
batch in about a minute on one GPU. Not run.

*New in the training code:* `utils.dump_nonfinite_state`, called in `engine.train_one_epoch` only when the loss is already
non-finite (before the existing assert): every rank that sees it writes its micro-batch, the first of them also the weights of
that moment (`<output_dir>/nan_dump_e<E>_it<S>_*.pt`; at most 2 weight and 8 batch dumps per run; no collectives, never raises,
unit-tested). Runs started or resumed after 13:19 carry it (`ftbck7sgb` does), so a second event documents itself. Note that
`print` is silenced on ranks > 0: a NaN on another rank shows in the log only as `Torchrun exited with code 1` and the next
`=== attempt`, which is what the watcher now keys on.

**First readings, epochs 4-14 (2026-09-20 13:50; n = 1 per arm, PROVISIONAL, no statement about finals; the first informative
point is lens7@29, which classifies winners but does not rank them).** From wandb (`plots/verify/wandb_layerwise.py`, wave-1 arms
added) and log.txt; ent = mean attention entropy of blocks 1-7 (nats), mlp = mean MLP write ratio of blocks 1-7.

| epoch 9 | test acc | lens7 | ent | mlp |
|---|---|---|---|---|
| random `r0` | 32.9 | 29.8 | 4.17 | 0.32 |
| C kdyck `ftbanapermb7i` | 26.1 | 7.1 | 2.22 | 0.09 |
| scale-only kdyck `ftbanap` [O] | 21.1 | 13.9 | 4.34 | 0.13 |
| **S + G kdyck `ftbc7sg`** | 30.1 | 11.0 | 2.01 | 0.32 |
| P + S kdyck `ftbc7ps` | 29.0 | 12.1 | 3.11 | 0.23 |
| P + G kdyck `ftbc7pg` | 29.5 | 9.1 | 3.73 | 0.12 |
| A1 kdyck `ftbc7a1` (lr x m) | 27.3 | 5.6 | 0.90 | 0.06 |
| A2 kdyck `ftbc7a2` (lr x 0.5) | 24.8 | 6.3 | 2.01 | 0.10 |
| C ksd `ftbanakpermb7i` | 33.0 | 3.9 | 2.73 | 0.34 |
| S + G ksd `ftbck7sg` (the dead fp16 run) | 29.8 | 12.5 | 2.14 | 0.28 |
| A1 ksd `ftbck7a1` (lr x m) | 36.1 | 5.0 | 1.80 | 0.35 |
| A3 ksd q, k only `ftbck7a3qk` | 33.3 | 3.4 | 1.65 | 0.33 |
| A3 ksd fc1 only `ftbck7a3f` | 35.4 | 5.3 | 2.72 | 0.36 |

At epoch 14, S + G kdyck: acc 45.4, lens7 11.3, ent 2.40, mlp 0.32 (random 47.2 / 39.2 / 4.04 / 0.31; C 40.4 / 8.7 / 3.23 / 0.12).
What can be said at this point: (i) the structure-only cell suppresses the block-7 lens like a winner (11 against 39 at epoch 14)
and keeps its sink longer than C, but its MLP write is random-like from epoch 9 and its accuracy tracks random, i.e. it has the
lens signature WITHOUT the slow start: whichever way its final goes separates the two; (ii) compensating the step size does not
wash the structure out, the opposite: with lr x m on q, k the attention stays sharper than in C (kdyck ent 0.90 against 2.22,
ksd 1.80 against 2.73), the lens stays suppressed and accuracy runs slightly ahead of C; lr x 0.5 looks like C; (iii) on ksd the
q, k compensation is what sharpens attention and the fc1 compensation is what speeds up accuracy; (iv) both kdyck cells suppress
the lens against random and against the scale-only arm, the gate cell more than the sink cell.

**Which group-D contrasts are meaningful (2026-09-20 15:00).** Cells per task: random, P, P + S, P + G, P + S + G (= C), S + G; S alone
and G alone do not exist. One run has sd 0.278, so a difference of two n = 1 cells has sd 0.39 (gaps >= 0.8 count) and an
interaction (difference of differences) sd 0.56 (>= 1.1).
- *ksd, finals (all full runs; effects of ~2 are resolvable at n = 1):* P - random (does the profile alone do anything; [O] says
  no), (P + S) - P and (P + G) - P (each component on the profile), C - (P + S) and C - (P + G) (each component given the other),
  (S + G) - random (is structure sufficient) and C - (S + G) (are the scales necessary). Caveat: `ftbck7sgb` is bf16.
- *kdyck:* finals only for P, S + G, C, random; P + S and P + G are 40-epoch screens. Component effects are expected <= 0.7 here, below
  what n = 1 resolves, so on kdyck group D is read through the mediators (lens, entropy, MLP write, depth profile), not finals.
- *S + G is not literally "C minus the profile":* same targets, but realised on timm-size weights with `renormalize: false`: rank-one
  strengths ~1.5-2x larger, query side 65-80% common instead of 97%, sink logits 10x larger, MLP write at init random-like (0.25-0.32
  against 0.08). A loss of S + G could therefore also come from the harsher realisation (its sink lives longer: entropy 3.14 at
  epoch 29 against 3.72 in C), not only from the missing scales.
- *The clean P cells have not started;* until then the only P rows are older recipes (`ftbanap`, `ftbanak` [O]): priors only.
- *Gap:* without S alone and G alone on timm, a win of S + G cannot be attributed. Both are reachable (sink always, mean gate 7/7) and
  need no new code (the `sg` specs minus one component). Proposed as four 40-epoch L40S screens; not prepared, awaiting the go.

Readings at epoch 19 (29 where logged), kdyck: block-7 lens C 9.2 ~ P + G 11.0 ~ S + G 11.0 (9.9 at epoch 29) < P + S 15.8 < P [O] 24.2
< random 44.3 (49.1): every structure component suppresses the lens beyond the profile, the gate more than the sink, and S + G gets
there without the profile. MLP write ratio: only C keeps it small (0.14); in P + G it is back at 0.26 by epoch 19 (0.12 at epoch 9), so
the gate stays closed only while the sink is there too: an interaction visible in the mediators. Accuracy: every cell short of C
learns faster early (51-54 against 49.8), S + G tracks random (63.1 against 64.2 at epoch 29; C 59.5). ksd S + G (bf16): lens 14.1,
16.1, 18.3 at epochs 9, 14, 19 (C 3.9 flat; P [O] 34.3; random 44.3): partly suppressed and rising, sink still sharp (entropy 2.39).

**All screens continue (Simon, 2026-09-20 17:50): no promotion decision, every run goes on.** The seven screens keep training on
their L40S nodes overnight (continuations 29748440-29748446 via `resume_zero_kbias.sh` with `STOP_AFTER_EPOCH=-1`, `afterany` on the
screen jobs, 23.5 h limit; ids in `launched_wave1.txt`), ~9.65 min per epoch, so they should be near epoch 125-130 by Monday 08:00;
then they move to H200 as slots open (same global batch and schedule; 8 -> 4 ranks changes the per-rank data split, not
bit-comparable from there). `ftbck7a3qk` was preempted on the shared L40S partition at 17:53 (epoch 38, requeued). Two full runs
were preempted on the shared H200 partition at 17:18 (run log); `ftbc7a1` resumed at 17:40, `ftbck7a1` waits.

**Readings at epoch 29 and beyond (2026-09-20 18:00; n = 1 per arm, PROVISIONAL).** Block-7 lens at epoch 29 (random 49.1, losers [O]
39.6-41.2, P-only kdyck [O] 28.4, C kdyck 6.7, C ksd 3.8):

| arm | lens7 @29 (@39, @49) | lens of blocks 8, 9 @29 | ent @29 | mlp @29 | test acc @29 |
|---|---|---|---|---|---|
| C kdyck / C ksd | 6.7 (3.6, 2.3) / 3.8 (3.4, 3.2) | 11.9, 32.6 / 18.4, 40.2 | 3.72 / 4.08 | 0.14 / 0.31 | 59.5 / 61.4 |
| random `r0` | 49.1 (49.3, 47.8) | 55.1, 59.9 | 3.89 | 0.21 | 64.2 |
| S + G kdyck | 9.9 (7.5, 5.3) | 26.4, 45.9 | 3.14 | 0.18 | 63.1 |
| S + G ksd (bf16) | 16.8 (13.0, 10.2) | 32.4, 48.3 | 2.98 | 0.21 | 62.2 |
| P + S kdyck / P + G kdyck | 14.2 / 7.8 | 23.8, 43.1 / 18.3, 39.3 | 3.42 / 3.64 | 0.21 / 0.51 | 60.4 / 61.1 |
| P + G ksd | 5.7 | 22.7, 45.2 | 4.09 | 0.29 | 62.2 |
| A1 kdyck (lr x m) / + gains / lr x 0.5 | 3.3 (0.7, 0.9) / 2.7 / 6.1 | 9.3, 33.2 / 8.5, 30.8 / 9.6, 27.3 | 3.18 / 2.07 / 4.09 | 0.10 / 0.09 / 0.14 | 59.6 / 58.5 / 57.7 |
| A1 ksd (lr x m) / + gains / q, k only / fc1 only | 5.2 (3.0, 2.5) / 5.0 / 3.1 / 6.3 | 28.3, 48.7 / 26.8, 48.4 / 19.5, 42.7 / 26.5, 47.6 | 2.94 / 2.56 / 2.70 / 3.89 | 0.17 / 0.15 / 0.26 / 0.22 | 62.9 / 62.4 / 61.9 / 62.6 |

(i) Every wave-1 arm is inside the winner class of the lens rule (<= 30): the rule does not separate them, as expected (it
classifies, it does not rank), and S + G lies outside the family it was fitted on. (ii) Compensating the steps does not move the
mediators towards random: on kdyck the lens is lower than C's and the depth profile is C's (9.3, 33.2 against 11.9, 32.6), attention
stays sharper (3.18, with gains 2.07, against 3.72). If slow steps act, it is not through these mediators. (iii) The kdyck A1 run had
an instability at the top of the warmup: train loss 4.36 -> 4.81 and test accuracy 60.6 -> 58.1 between epochs 34 and 39, then
recovery; it trails C by 2.4, 1.8, 1.4, 0.9 points at epochs 44, 49, 54, 59. The ksd A1 run has no such event and sits on C
(70.15 against 70.17 at epoch 54) after leading early. (iv) The two markers disagree on S + G: lens like a winner (and falling
like C's), but the class is readable earlier in depth than in C (block 8: 26 against 12), accuracy and train loss track random
(kdyck at epoch 69: 73.07 / 3.489 against random 73.33 / 3.512; C 71.05 / 3.644), i.e. no fit deficit so far. (v) kdyck cells: the
gate suppresses the lens more than the sink (7.8 against 14.2); in P + G the MLP write overshoots to 0.51 at epoch 29.

**Loss spikes belong to the kdyck committed init and grow with the q, k step size (2026-09-20 19:45; train loss per epoch, a spike
= a rise of more than 0.25 over the previous epoch).** None in random, in both S + G runs, in the kdyck cells P + S and P + G, and in
every ksd arm (C, A1, A1 + gains, A3 q/k, A3 fc1, P + G, P + S). On kdyck with the full P + S + G init:

| arm | q, k, fc1 step multiplier | spikes (epoch: loss reached) |
|---|---|---|
| `ftbc7a2` | x 0.5 | 37: 5.19 |
| C `ftbanapermb7i` (80.37) | x 1 | 40: 4.65, 64: 3.99, 80: 4.37, 111: 3.68 |
| `ftbc7a1` | x m (3.0-4.5 on q, k) | 31: 5.33, 37-38: 6.14, 6.66, 41: 4.93, 60-61: 5.04, 6.85, 72: 5.97, 82-83: 4.06, 4.89; back within 1-3 epochs each time |
| `ftbc7a1g` | x m, LayerNorm gains x m (< 1) | 41-42: 5.09, 6.90 = chance, NOT recovered (6.59 at epoch 48, test accuracy 1.38 at 44) |

So the full kdyck init is a fragile configuration, the step size of q and k sets how often and how hard it breaks, and with slowed
LayerNorm gains it does not come back. This is a candidate reading of what the slow steps are for on kdyck (protection of the
sharp-sink state at high learning rate), separate from the mediators, which the compensation did not move. Caveat: a constant
multiplier over-compensates once the weight norms have relaxed (all arms equalise by epoch ~100, section 1.5), so from epoch ~30 on
A1 gives q and k LARGER relative steps than a random init has, at the top of the learning-rate schedule; a multiplier that follows
the weight norm (group B's schedule) would be the clean test. `ftbc7a1g` was paused at epoch 48 (continuation cancelled, resumable;
per-epoch checkpoints 39-48 cover the collapse) and its protected L40S slot went to `ftbc7a2`, which had been waiting for a node.

**Group D at epoch 49: what each factor does (2026-09-20 20:00; n = 1 per cell, PROVISIONAL, no statement about finals).** acc = test
top-1, loss = train loss of that epoch, lens7 / lens9 = head read-out of blocks 7 / 9, ent and mlp as above. P rows are older
recipes [O]; the clean P cells and ksd P + S were preempted at epochs 3, 6 and 23 and wait for a slot.

| kdyck, epoch 49 | acc | loss | lens7 | lens9 | ent | mlp |
|---|---|---|---|---|---|---|
| random | 70.0 | 3.764 | 47.8 | 65.2 | 3.73 | 0.17 |
| P [O] `ftbanap` | 68.6 | 3.887 | 26.2 | 54.5 | 3.70 | 0.26 |
| P + S | 68.8 | 3.819 | 6.5 | 42.3 | 3.43 | 0.19 |
| P + G | 68.3 | 3.861 | 1.9 | 35.7 | 3.69 | 0.30 |
| P + S + G (C) | 66.7 | 3.926 | 2.3 | 27.5 | 3.79 | 0.13 |
| S + G | 69.5 | 3.748 | 5.3 | 43.5 | 3.27 | 0.14 |

| ksd, epoch 49 | acc | loss | lens7 | lens9 | ent | mlp |
|---|---|---|---|---|---|---|
| P [O] `ftbanak` | 69.0 | 3.758 | 36.5 | 57.2 | 3.48 | 0.34 |
| P + G | 68.9 | 3.837 | 3.1 | 45.3 | 3.98 | 0.25 |
| P + S + G (C) | 68.6 | 3.817 | 3.2 | 38.3 | 4.05 | 0.27 |
| S + G (bf16) | 69.2 | 3.752 | 10.2 | 48.4 | 3.08 | 0.15 |

- *Gate:* the strongest lever on the block-7 lens (given P: 26 -> 1.9 on kdyck; on ksd P + G = C) and on how late the class forms
  (lens9 -19 on kdyck). Without the sink its MLP closure does not hold: the MLP write overshoots (0.51 at epoch 29, 0.30 at 49).
- *Sink:* second on the lens (26 -> 6.5), adds to late class formation (lens9 -12 kdyck, -7 ksd given P + G), and is what keeps the
  MLP quiet together with the gate (C 0.13, S + G 0.14). It stays sharpest where there is no profile (S + G entropy 3.27, C 3.79).
- *Profile (scales + LayerNorm statistics):* almost irrelevant for the block-7 lens once structure is there (C 2.3, S + G 5.3), but
  it is the factor that carries the slow start and the fit deficit: every cell with P is behind random in accuracy (-1.2 to -3.3)
  and in train loss (+0.06 to +0.16), S + G is not (-0.5, -0.02; the same on ksd), and it deepens late class formation (lens9 43.5 ->
  27.5 kdyck, 48.4 -> 38.3 ksd). The three factors are roughly additive on lens9: C is the extreme on both tasks.
- *What this does not tell:* which marker carries the gain. In the 107 finished runs the epoch-49 train loss tracks the final only
  moderately (Spearman +0.49; +0.62 without the damaged arms), and the no-deficit band (< 3.78, where S + G sits) spans finals from
  77.5 to 80.1 (mean 78.3, n = 27; it contains the winner `ftbanaperab7i` 79.63). Test accuracy at epoch 49 is anti-correlated with the
  final (-0.32).

**Matched-epoch readings late in training (2026-09-21 08:40; finals pending, n = 1).** Test accuracy (train loss):

| epoch | random `r0` | old random (3 seeds) | C kdyck | S + G kdyck | C ksd | S + G ksd (bf16) | P + G ksd |
|---|---|---|---|---|---|---|---|
| 199 | 76.65 (2.721) | 77.26-77.52 | 78.55 (2.884) | 78.92 (2.747) | 78.68 (2.792) | 78.30 (2.728) | 78.89 (2.795) |
| 229 | 77.03 | 77.61-77.67 | 79.28 | 79.22 | 79.22 | 78.91 | 79.42 |
| 249 | 77.14 (2.402) | 77.68-78.17 | 79.85 (2.591) | 79.35 (2.462) | 79.54 (2.503) | 78.87 (2.427) | |
| 259 | 77.31 (2.348) | 77.69-77.97 | 79.94 (2.539) | 79.38 (2.407) | 79.63 | | |
| 299 | 77.42 | 77.91-78.28 | 80.37 | | 79.90 | | |

The structure-only cells left random behind after epoch ~100 (the "tracks random" reading of epochs <= 94 did not hold, as the
no-early-prediction rule warns): at epoch 259 S + G kdyck is +1.4 to +2.1 over random and 0.56 under C; S + G ksd at 249 is +0.7 to
+1.7 over random and 0.67 under C. Finished runs still gained +0.1 to +0.45 after epoch 259. On ksd, P + G is level with C at epoch
229 (79.42 against 79.22). `r0` ended at 77.42, 0.66 under the old random mean (run log).

## 5c. Disk

`/work/dlc2workfs3/schrodi-procedural` stood at 9.85 TB of the 10 TB limit on 2026-09-20 (all users' files; 6.13 TB of it per-epoch
model checkpoints under `results/imnet_base`). **Thinned on 2026-09-20 ~14:00 with Simon's go:** 23,885 per-epoch model
checkpoints (4.14 TB) of 108 finished seed directories deleted; workspace now 5.80 TB. Tool: `plots/verify/thin_epoch_checkpoints.py`
(dry run -> validated manifest -> `--execute`; manifest and deletion log in `results/init_dumps/thinning_manifest.json`,
`thinning_deleted.log`). Only `checkpoint-<E>-model.pth` files were removed; full checkpoints, `checkpoint-best`, logs and json
files are untouched.

- **Kept everywhere:** epochs 0, 1, 2, 4, 5, 9, 10, 14, 19, 20, 29, 39, 49, 50, 59, 99, 100, 199, 200, 299, every 25th in both
  conventions (E % 25 = 0 and (E + 1) % 25 = 0), and the last epoch present (aborted runs keep their end state): 36 files per
  complete run. 14 was added to the agreed list because the lens is logged at 4, 9, 14, 19. Every epoch the analysis scripts read
  (`mechanism_autopsy.py`, `mechanism_phase0_gpu.py`, `end_state_stats.py`) is still there for all 107 runs of the zoo index
  (checked after the deletion).
- **Reference arms keep the wave-1 set** (epochs 0-59, every (E + 1) % 10 = 0, plus the above; 98 files): `ftbanapermb7i`,
  `ftbanakpermb7i`, `ftbanap`, `ftbanak`, `ftbana`, `ftbanaperab7i`, `ftbanakperab7i`, `ftbanapermb7`, `ftbanakpermb7`, so they can be
  compared with the wave-1 runs epoch by epoch. Thin further if the space is needed.
- **Not touched, and why (1.3 TB of per-epoch files remain there):** (i) 17 seed directories whose per-epoch checkpoints, all
  300 epochs, are referenced by the collaborator's `kempfe/dynamics` analysis of 09-18/19 (`ftb4i` kdyck s1-2, `ftbcomp1`,
  `ftbqu`, `ftbqmlnvo` s0-2 each, `ftbrhos`, `ftblrm`, `ftb4` ksd, `ftb4i` ksd, `ftbrhop`, `ftbrhopl`); (ii) the seven ksd runs
  29547822-29547836 (`ftb4`, `ftb4b/e/f/h/i/j`), whose files belong to the collaborator's account; (iii) every run with a job in
  the queue, R0 and all wave-1 runs; (iv) `ftbrhoplv` / `ftbrhoslv`, which had finished less than 3 hours earlier; (v) one aborted
  seed directory with a truncated kept file (29529160/s1). (i) and (ii) need a word with the collaborator before thinning.
- Not part of this pass: the local `wandb` directory (1.65 TB), the second and third full checkpoint of finished runs (~0.5 TB),
  `checkpoint-best` files (0.29 TB; max-over-epochs is not reported anyway), `i100_playground` (0.19 TB).

## 5d. Interim reading against the mechanisms (2026-09-21 09:00; finals pending, n = 1 per cell, no seeds)

- **M1 (slow steps) is not the explanation.** Two independent lines: S + G has timm-size weights, hence ordinary steps, and carries most
  of the gain on both tasks (epoch 259 kdyck +1.4 to +2.1 over random, epoch 249 ksd +0.7 to +1.7); compensating the steps on the
  committed init (A1) has not removed the gain (kdyck -0.36 against C at epoch 199 in spite of repeated collapses, ksd level at epoch
  104). On kdyck the step size looks like a stability matter (spikes scale with the q, k multiplier), not like the source of the gain.
- **M5 (fit deficit as the carrier) is weakened.** S + G kdyck gets ~75-80% of C's gain with ~30% of its train-loss deficit (2.407
  against random 2.28-2.38 and C 2.539 at epoch 259), and no slow start.
- **M3 (learning order: the class is not readable from block 7 early, the late blocks take it) is the one that survived an
  out-of-family test.** The lens rule was fitted on arms with scales or prefixes; S + G (lens7@29 9.9 / 16.8) and ksd P + G (5.7) lie
  outside that family and both land among the winners so far, while everything with a transient of ~40 (ksd P-only, `ftbana`) lost.
  It still does not rank winners: kdyck P-only [O] has the weakest suppression (28) and the highest accuracy at epoch 259 (80.06).
- **No single component is necessary.** ksd: P + G without any sink is level with C (79.42 against 79.22 at epoch 229); S + G, whose gate
  is functionally gone by epoch 9, wins as well; P alone fails there [O]. kdyck: P alone wins [O], S + G wins; P + S and P + G are at epoch
  ~130, where no arm, winner or not, is separated from random yet.
- **The gain is made late.** Every winner is at or below random until epoch ~130 and separates during the learning-rate decay: random
  goes 76.3 -> 77.3 between epochs 129 and 259 while its train loss falls 3.14 -> 2.35; winners gain 3-4 points over the same stretch.
  Consequence for the method: a 40-epoch (or 130-epoch) screen reads mediators, never outcomes.
- **What this changes in the plan.** (i) Decompose S + G: sink alone and gate alone on timm (no new code) are now the most informative
  missing cells. (ii) Group C (is the lens transient a mediator or a marker: auxiliary head loss at block 7 on a winning init, and
  its converse on a random init) moves ahead of group B (restrained steps), which M1's decline makes less urgent. (iii) Whether the
  scales add anything on top of the structure (C - S + G ~0.5-0.7 at matched epochs) needs seeds; one run per cell cannot resolve it.
  (iv) The random reference needs a second seed (`r0` 77.42 against 78.08 before).

**FIRST WAVE-1 FINALS (2026-09-21, last epoch, n = 1).** References: random 78.08 +- 0.19 (old, n = 3) and `r0` 77.42; C kdyck 80.37, C ksd 79.90.

| cell | final | against C | against random (old / `r0`) |
|---|---|---|---|
| S + G kdyck `ftbc7sg` (fp16) | **79.57** | -0.80 | +1.49 / +2.15 |
| S + G ksd `ftbck7sgb` (bf16) | **79.01** | -0.89 | +0.93 / +1.59 |
| P + G ksd `ftbck7pg` | **80.27** | +0.37 | +2.19 / +2.85 |
| A1 kdyck `ftbc7a1` (committed init, steps compensated: lr x 3.0-4.5 on q, k rows, x m on fc1) | **80.01** | -0.36 | +1.93 / +2.59 |
| P + G kdyck `ftbc7pg` (profile + mean gate, no sink; 40-epoch screen continued, L40S -> H200) | **79.92** | -0.45 | +1.84 / +2.50 |
| A1 ksd `ftbck7a1` (committed ksd init, steps compensated) | **79.59** | -0.31 (C ksd 79.90) | +1.51 / +2.17 |
| P + S ksd `ftbck7ps` (profile + sink, no gate) | **79.04** | -0.86 (C ksd 79.90) | +0.96 / +1.62 |

Structure alone, on timm-size weights with ordinary steps, carries half to two thirds of the gain on both tasks; the profile adds
the rest: C - (S + G) = 0.80 and 0.89, each about 2 sigma for a difference of two single runs (sd 0.39), together 0.85 +- 0.28. On ksd
the sink is not needed once profile and gate are there (P + G >= C). `ftbc7a1` (steps compensated) ended at 80.01, 0.36 under C and inside the seed resolution, in spite of nine loss spikes, two of
them down to chance level: the slow q, k steps are not what produces the gain on kdyck; what they buy is stability.

## 5e. Wave 2 as launched (2026-09-21 09:30): single-component cells and the forced-readability test

| arm | what | precision | L40S job = results id (continuation) | H200 hand-over |
|---|---|---|---|---|
| `ftbc7c1` | **C1 kdyck:** committed init + auxiliary cross-entropy on the block-7 lens, weight 0.3, epochs 0-49, then off | fp16 | 29751052 (29751053) | 29751054 |
| `ftbc7s` | kdyck **sink alone** on timm (the S + G spec minus the gate) | fp16 | 29751055 (29751056) | 29751057 |
| `ftbc7g` | kdyck **gate alone** on timm (the S + G spec minus the sink) | fp16 | 29751058 (29751059) | 29751060 |
| `ftbck7s` | ksd sink alone | bf16, as `ftbck7sgb` | 29751061 (29751062) | 29751063 |
| `ftbck7g` | ksd gate alone | bf16 | 29751064 (29751065), waits on the per-user L40S CPU cap | 29751066 |
| `ftbck7c1` | C1 ksd | fp16 | 29751067 (29751068), waits as well | 29751069 |
| `r0sup` | converse of C1 on plain random, by penalty: block-7 lens pushed to the uniform distribution (norm and head detached), weight 0.3, epochs 0-49 | fp16 | 29751150 (29751151), launched 09:45 on Simon's go, waits on the L40S CPU cap | 29751152 |
| `r0frz` | converse of C1 on plain random, by learning order: blocks 0-7 frozen (lr factor 0) for epochs 0-29, linear release over epochs 30-49, normal from 50 | fp16 | 29751165 (29751166), launched 09:52, waits as well | 29751167 |
| `ftb4c1` | **C1 on the FULL kdyck checkpoint** (Simon, 2026-09-22): kdyck4 checkpoint in all 12 blocks as in `ftb4` (head, cls, pos, patch embedding, final norm timm random) + the same auxiliary block-7 loss, weight 0.3, epochs 0-49; reference `p` 80.09 +- 0.11 (n = 3) | fp16 | 29754117 (29754118), launched 07:55 on L40S | shared 29754129 (interim), group 29754119 after `ftbck7c1` (~21:15) |
| `ftb4kc1` | C1 on the full ksd checkpoint (reference `ftb4` ksd 80.21, n = 1, fp16, `skip_norm true`) | fp16 | 29754123 (29754124), launched 07:58 on L40S | shared 29754125 (interim), group 29754127 after `r0frz` (~15:45) |
| `ftb4kd` | plain kdyck full-checkpoint reference: NOT NEEDED, the `p` runs (29377576, seeds 0-2: 80.17 / 80.14 / 79.96, mean **80.09 +- 0.11**) loaded the kdyck4 checkpoint into all 12 blocks under an equivalent protocol (wandb check 2026-09-22; my earlier note that they used an older checkpoint was wrong) | fp16 | verified, not launched | |

Every run starts on 8 L40S (shared partition, preemptable) and carries an H200 hand-over job (`vitbase_runs/handover_to_h200.sh`): when
that job gets a slot it cancels the L40S jobs of the arm, makes the directory safe to resume
(`plots/verify/verify_resume_checkpoint.py`) and continues, or starts from scratch under the same results id if no checkpoint exists.

**Auxiliary lens loss** (`utils.AuxLensLoss`, `--aux_lens_block / _mode / _weight / _until`; off by default and then inert): the lens is
exactly the one the evaluation logs, head(fc_norm(norm(block output)))[:, 0], taken by a forward hook on the un-wrapped model, one
backward over main + weight x aux; the logged `loss` stays the main loss, the auxiliary term is logged as `aux_lens_loss`.
*align* = cross-entropy against the (mixup) targets, final norm and head receive its gradient. *suppress* = cross-entropy against
the uniform distribution minus log K (>= 0, zero iff the head reads nothing from the block), final norm and head detached, so only
blocks 0-7 and the embeddings move; bounded, unlike gradient ascent on the lens cross-entropy. Unit tests: value equals the lens
cross-entropy computed by hand; gradients reach blocks <= 7 (+ norm and head only in align); none into later blocks; gradient
checked against finite differences; window and eval mode.

**Full-checkpoint C1 arms** (verified 2026-09-22, tasks 29754075 / 29754108): the init dump through main.py equals the procedural
checkpoint bit for bit in all 144 block tensors (the checkpoint keeps its weights under `state`; qkv is stored fused), and head,
cls token, pos embedding, patch embedding and the final norm are timm seed 0 (`pr_load_model` drops them; `--skip_norm true`),
exactly the `ftb4` construction; 2-rank smoke on the real schedule with the auxiliary loss ON in epoch 0 and off in epoch 1. In
the existing full-checkpoint runs the block-7 lens is suppressed like in the prefix arms (ftb4 ksd 3.2, 5.6, 6.9, 6.0, 3.3 at epochs
4-49; old-checkpoint kdyck `p` 2.1-10.7), so C1 asks the same question of the whole-network procedural init: does forcing early
readability at block 7 remove its gain? The hand-over wrapper now cancels every other job of the arm (any partition) except its own
continuations, so a run can be moved L40S -> shared H200 -> group H200.

**Freeze and release** (`--release_blocks / --release_start / --release_end`, `optim_factory.build_block_release_param_groups`,
`release_factor`; off by default): the tensors of the chosen blocks sit in their own parameter groups whose learning rate is the
schedule value times a factor that is 0 before the start epoch and rises linearly (per update) to 1 at the end epoch. With lr = 0
AdamW changes nothing, decoupled decay included, so the blocks are frozen bit for bit while their Adam moments follow the
gradient; weight decay is not compensated during the ramp. Why 0-29 / 30-49: in a random run the block-7 lens rises during epochs
0-30 and peaks at epochs 29-39 (47.8-49.3%), then decays slowly to 15-25% at the end; winners peak at epochs 14-19 below 10-18% and
decay to ~0. The freeze covers the rise, the release ends with the warmup, and all three interventions (C1, `r0sup`, `r0frz`) act
on epochs 0-49 only. Tests: grouping (every tensor once, exactly the blocks' 96 tensors released, decay flags as in the default
grouping), factor, bit-exact freeze under AdamW with warm moments, flag survives the optimizer state dict, refusal with the other
group builders; GPU task 29751149: init == `r0`'s, after a frozen epoch all 96 tensors of blocks 0-7 are identical to the init and
all 56 others moved, after the release epoch all 96 moved.

**Verification** (array 29750978 + rerun 29751018, 2 L40S per task): cells: `verify_structure_only.py`, generalised to one component
(profile step a no-op, exactly the 7 qkv or the 7 fc1 tensors change, each change rank one, all targets matched; regression on the
launched S + G spec passes); sink alone costs raw q, k 1.07-3.04x timm, gate alone fc1 1.12-1.25x. C1 arms and `r0sup`: init dump
through main.py == the verified init of the committed arm / of `r0`, bit for bit. All: 2-rank smoke on the real schedule (two epochs);
for the auxiliary arms epoch 0 ON and epoch 1 off, `train_aux_lens_loss` in log.txt only while on. The first `ftbc7g` smoke died of an
uncorrectable ECC error on dlc2gpu04 (now excluded); the rerun passed. At launch: 8 ranks, all calibration targets reachable,
tensors identical across ranks. The calibration hash depends on the GPU type, not on the arm: every L40S run of the committed kdyck
init has `de9ca9e3...` (`ftbc7a1g`, `ftbc7a2`, `ftbc7c1`), every H200 run `62edac63...` (C, `ftbc7a1`): same init up to the rounding of
the calibration on another GPU. Each single component is calibrated on its own arm's stream, so the sink of `ftbc7s` has slightly
other strengths than the sink inside S + G (alpha 5.23 against 4.80 in block 3) for the same entropy targets.

**Intervention arms, first readings (2026-09-21 15:40; n = 1, PROVISIONAL).** Block-7 lens | model accuracy at epochs 9, 19, 29, 39, 49, 69:

| arm | lens7 | accuracy |
|---|---|---|
| random `r0` | 29.8, 44.3, 49.1, 49.3, 47.8, 47.1 | 32.9, 55.6, 64.1, 67.9, 70.0, 73.3 |
| C kdyck | 7.1, 9.2, 6.7, 3.6, 2.3, 1.6 | 26.2, 49.8, 59.5, 65.0, 66.8, 71.0 |
| `ftbc7c1` (C + align) | 19.9, 38.3, **50.1, 56.3**, collapse | 26.1, 48.6, 60.4, 66.0, collapse |
| `r0sup` (random + suppress) | 23.2, 31.4, **27.7, 23.4** | 34.9, 56.1, 64.4, 68.5 |
| `r0frz` (random + freeze) | **0.3, 0.4, 0.3, 0.5, 0.4, 0.4** | 26.7, 44.1, 52.0, 56.2, 59.3, 65.1 |

- *C1: the manipulation works.* The auxiliary loss makes block 7 of the winning init as class-readable as in a random run (50-56%
  against 3.6-6.7 in C) while accuracy runs 1 point ahead of C. In epochs 49-50 the run had one of the kdyck init's loss spikes
  (train loss 3.86 -> 4.37 -> 6.78, test accuracy 0.27 at epoch 49; it began while the auxiliary loss was still on) and is
  back within three epochs (train loss 4.88, 3.95, 3.87 at epochs 51-53; 68.79 at epoch 54 against C's 68.32), so the run stays
  interpretable, with the spike as a caveat.
- *The suppress penalty is a weak manipulation.* It drives the lens distribution to within 0.022 nats of uniform, but that
  flattens the read-out's SIZE, not its ORDER: the lens' top-1 accuracy only falls from 49 to 23-31%, the block-9 lens from 60 to 55,
  accuracy equals random. It lands at the rule's threshold (27.7 at epoch 29), not among the winners (<= 10). A margin- or
  probe-based penalty would be needed to really remove the information.
- *The freeze is a strong one.* Lens7 stays at chance, also after the release (0.4 at epoch 69), block 9 reads 1.5-7%: the class
  is computed entirely in blocks 8-11, more extreme than in C or the prefix. The price so far is a large accuracy deficit
  against random that is closing (-12.1, -11.7, -10.7, -9.8, -8.2 at epochs 29, 39, 49, 59, 69); the kdyck prefix was at 61.6 at
  epoch 49 and 69.4 at epoch 89 and still ended at 80.0, so nothing can be said about the final.

**Intervention arms, second reading (2026-09-22 07:20; n = 1, PROVISIONAL).**

| arm | lens7 @29 / 49 / 69 / 99 / 149 | lens9 @99 | acc @99 / 149 / 174 (train loss @174) |
|---|---|---|---|
| random `r0` | 49, 48, 47, 42, 36 | 69 | 75.43 / 76.74 / 76.80 (2.883) |
| C kdyck (80.37) | 6.7, 2.3, 1.6, 0.7, 0.5 | 18 | 73.96 / 76.72 / 77.44 (3.027) |
| kdyck prefix s1 (80.00) | 5.8, 1.7, 4.5, 1.0, 0.6 | 10 | 70.91 / 75.38 / 76.83 (3.132) |
| `r0frz` (freeze 0-29, release 30-49) | **0.3, 0.4, 0.4, 0.4, 0.2** | **1.6** | 69.27 / 73.79 / 75.30 (3.339) |
| `r0sup` (suppress penalty, epochs 0-49) | 28, -, **16**, -, - | 60 (@69) | 73.86 @69 (random 73.33) |
| C ksd (79.90) | 3.8, 3.2, 2.2, 1.6, 0.8 | 30 | 75.00 / 77.40 / 78.19 (2.934) |
| `ftbck7c1` (C ksd + align, epochs 0-49) | **53, 61, 45, 33**, - | 51 | **75.97** @99 (loss 3.254 against C's 3.320) |

- *Freeze:* the pure learning-order manipulation on a plain random init reproduces the winners' dynamics without any checkpoint
  statistic: block-7 lens at chance throughout, block 9 at 1-2% (the class forms in blocks 10-11 only, deeper than in C or the
  prefix), a fit deficit larger than any winner's (train loss 3.34 at epoch 174 against C 3.03, prefix 3.13, random 2.88), and an
  accuracy gap to random that closes monotonically: -12.1, -10.7, -8.2, -6.2, -4.4, -3.0, -1.5 at epochs 29-174. It trails the
  prefix by 1.5 at epoch 174, where the prefix had caught up with random and went on to 80.0. The final (~15:45) is the test of M3
  in its purest form.
- *Forced readability (ksd):* the auxiliary loss holds the block-7 lens at 53-61% while on, and it decays only slowly after epoch 50
  (45 at 69, 33 at 99 against C's 1.6); block 9 stays at 51 (C 30). Accuracy runs ~1 point AHEAD of C at epochs 29-99 with a
  smaller fit deficit. If M3 is right this run ends below C; if it ends level, the transient is a marker. Final ~21:15. The kdyck
  twin `ftbc7c1` (spike at 49-50, then idle overnight, resumed this morning) is at epoch 68 and uninformative yet.
- *Suppress penalty:* weak while on (lens 28 at epoch 29), but the suppression deepens AFTER it is switched off (16 at epoch 69,
  random 47; block 9: 60 against 68) at no accuracy cost so far (73.86 against 73.33 at epoch 69). Resumes this morning.
- *Single components at epochs 99-119:* sink alone is winner-like on the lens on both tasks (kdyck 8.8 / 6.5, ksd 4.8 / 3.9; S + G
  3.0 / 8.0), gate alone is not (kdyck 25 / 22, ksd 31; random 42 / 39) and its MLP write is at the random level from epoch 9 (0.37):
  without sink or scales the gate does not stay closed. Accuracies at 119: S kdyck 76.59, G kdyck 76.17, S ksd 76.51, random 75.82,
  S + G 76.2. The clean P-only kdyck cell `ftbc7p` suppresses the lens more than the old recipe did (15.2 against 28.4 at epoch 29).

**Factorial reading with six finals (2026-09-22 13:30; n = 1 per cell, two-cell difference sd 0.39, resolution ~0.8).**
Gain over the old random mean (78.08): kdyck C 2.29, ksd C 1.82.

| cell | kdyck | ksd |
|---|---|---|
| P + S + G (C) | 80.37 | 79.90 |
| P + G (no sink) | 79.92 (-0.45) | **80.27 (+0.37)** |
| P + S (no gate) | running (epoch 205) | 79.04 (-0.86) |
| S + G (no profile) | 79.57 (-0.80) | 79.01 (-0.89) |
| P alone | 80.24 [O, `ftbanap`, older realisation]; clean cell pending | 77.86 [O, `ftbanak`] = random; clean cell pending |
| S alone / G alone | running | running / pending |

"All three needed" is NOT what the cells say. Supported: the profile is needed on both tasks (S + G loses 0.8-0.9); the gate is
needed on ksd (P + S loses 0.86 and P alone is at random level [O]), not on kdyck (P alone wins [O]); the sink is dispensable on ksd
(P + G >= C) and worth at most 0.45 on kdyck (inside the resolution). Minimal winning sets so far: kdyck P (possibly alone), ksd P + G.
The sink's role is as a mediator (it keeps the gate closed and drives the lens suppression), not as a contributor to the final.
Pending for the full picture: clean P cells (`ftbc7p`, `ftbck7p`), P + S kdyck, S / G alone, and seeds for any gap below 0.8.

**Full-checkpoint C1 arms, first reading (2026-09-22 15:30; epochs 4-59, n = 1).** Block-7 lens | block-9 lens | test acc:

| epoch | `p` (kdyck full ckpt, 80.09) | `ftb4c1` = p + align | `ftb4` (ksd full ckpt, 80.21) | `ftb4kc1` = ftb4 + align | random |
|---|---|---|---|---|---|
| 9 | 4.4 / 5.5 / 12.3 | 10.3 / 10.5 / 11.3 | 5.6 / 8.3 / 15.1 | 12.6 / 12.5 / 15.1 | 30 / 32 / 33 |
| 19 | 9.6 / 13.0 / 25.1 | 25.4 / 26.5 / 28.4 | 6.9 / 13.0 / 31.1 | 24.8 / 24.8 / 30.9 | 44 / 52 / 56 |
| 29 | 10.7 / 20.9 / 44.4 | 41.4 / 43.2 / 46.8 | 6.0 / 19.8 / 47.2 | 37.7 / 38.2 / 46.6 | 49 / 60 / 64 |
| 39 | 10.5 / 24.3 / 54.5 | 50.6 / 53.1 / 56.8 | 4.7 / 22.8 / 56.6 | 47.2 / 47.4 / 56.3 | 49 / 63 / 68 |
| 49 | 8.5 / 25.2 / 61.0 | - | 3.3 / 22.2 / 61.9 | 53.4 / 54.2 / 63.1 | 48 / 65 / 70 |
| 59 | 7.1 / 24.4 / 65.1 | - | 2.4 / 22.4 / 66.3 | 45.2 / 51.2 / 66.7 (aux off) | 48 / 67 / 72 |

The manipulation takes on the whole-network procedural init exactly as on the prefix arms: the block-7 lens climbs to random's level
(41-53 against 5-11 in `p` / `ftb4`), block 9 with it (43-54 against 20-25), and the head reads block 7 about as well as block 11
(lens7 ~ lens11 - 5 in the aux window, i.e. the class becomes readable early and the later blocks add little). Accuracy runs 1-2
points AHEAD of the plain full-checkpoint runs during the window (`ftb4c1` 56.8 against 54.5 at 39; `ftb4kc1` 63.1 against 61.9 at 49)
with a slightly smaller fit deficit; the early attention entropy is unchanged by the loss (the sink state is not what it removes).
After switch-off the ksd lens-7 decays slowly (53 -> 45 in 10 epochs), as in the prefix twin. The finals (Wed ~08:35 / ~10:30) then
answer the same question as for the prefix arms, on the init the paper starts from.

**Committed-init C1 arms, late reading (2026-09-22 15:40).** ksd (`ftbck7c1`, epoch 224): the forced readability persists long after
the loss is off: block-7 lens 33 at epoch 99, 21 at 149, 15 at 219 (C ksd: 1.6, 0.8, 0.5; random 42, 36, 29), block 9 51 / 43 / 31
(C 30 / 23 / 14). Accuracy led C by +1.4 at epoch 49 and the lead shrank to zero by epoch 199 (78.66 against 78.68) and to -0.3 to
-0.4 at epochs 209-224 (78.93 against 79.28 at 219), with a smaller fit deficit throughout (train loss 2.58 against 2.67 at 219;
random 2.59). The crossing at ~epoch 200 is the signature the learning-order hypothesis predicts (early class alignment buys fit and
early accuracy, costs late generalisation); the final (~21:10) decides whether the gap ends inside or outside the 0.8 resolution.
kdyck (`ftbc7c1`, epoch 99, behind after its collapse and preemptions): lens 31 / 49 (blocks 7 / 9) at epoch 99 against C's 0.7 / 18;
accuracy +1.0 over C at epoch 99; too early for the crossing.

**ksd C1 FINAL (2026-09-23 ~01:25, last epoch): `ftbck7c1` = C ksd + auxiliary block-7 lens cross-entropy (weight 0.3, epochs 0-49):
79.19 (train loss 2.280).** Against C ksd 79.90 (train loss 2.374) that is -0.71, at the edge of the ~0.8 resolution of a one-seed pair;
against random `r0` 77.42 it is +1.77. The trajectory is the crossing predicted in the late reading: +1.4 at epoch 49 (70.02 against
68.60), +1.0 at 99, +0.2 at 149, 0.0 at 199, -0.31 at 249, -0.65 at 274, -0.71 at 299, with a smaller fit deficit at every point (2.28
against 2.37 at the end). Forcing the class into block 7 during the warm-up buys fit and early accuracy and costs late generalisation:
the sign the learning-order reading (M3) predicts, on the committed ksd init, one seed. The kdyck twin `ftbc7c1` (epoch 189, 78.24
against C kdyck's 78.6 at the same epoch; preempted overnight, next in the shared queue) and the two full-checkpoint C1 arms
(`ftb4kc1` epoch 214, `ftb4c1` epoch 188) complete the picture today.

**Two more finals (2026-09-23 09:45, last epoch).** `ftbc7ps` (P + S kdyck) 79.79 (train loss 2.353): -0.13 against P+G kdyck 79.92,
+0.22 against S+G kdyck 79.57, -0.58 against C kdyck 80.37, +2.37 over `r0`. On kdyck every pair of the three components ends at
79.6-79.9 and only the full C reaches 80.4 (all one seed; the pairwise spread is inside the resolution, the gap to C at its edge).
`ftbck7a1g` (A1 ksd + LayerNorm gains) 79.36 (2.330): -0.23 against `ftbck7a1` 79.59 and -0.54 against C ksd 79.90: the LayerNorm
gains add nothing on top of the step-compensated committed ksd init.

**ksd full-checkpoint C1 FINAL (2026-09-23 12:44, last epoch): `ftb4kc1` = ksd checkpoint in all 12 blocks + auxiliary block-7 lens
loss (weight 0.3, epochs 0-49): 79.30 (train loss 2.332).** Against the plain full ksd checkpoint `ftb4k` 79.77 (2.398) that is -0.47,
against C ksd 79.90 (2.374) -0.59; the crossing again: +4.25 over `ftb4k` at epoch 49, +1.48 at 99, -0.05 at 199, -0.53 at 249, -0.46
at 299, with the smaller fit deficit throughout. Forcing the class into block 7 during the warm-up costs about half a point on the
paper's own init as well (one seed; `ftb4c1`, the kdyck twin, is 1.6 below `ftb4` at epoch 280 and finishes ~14:40).

**Reverse freeze `r0frzl`, mid-run reading and a prediction (2026-09-23 11:10, epoch 164; Simon asked for one).** Two indicators
disagree, and the mechanistic one is the stronger. (i) *Accuracy and fit:* its gap to `r0` runs -2.43 / -0.94 / -0.26 / -0.01 / +0.24 at
epochs 49 / 99 / 124 / 149 / 164, a near-copy of C kdyck's (-3.27 / -1.47 / -0.76 / -0.02 / +0.21) and A1 kdyck's, with C's fit deficit
(train loss 3.068 at 164 against `r0` 2.947, C 3.082); C and A1 then gained +2.7 over `r0` from epoch 164 to 299. (ii) *Lens (wandb
per-block accuracies, `plots/cache/verify/wandb_layerwise.json`):* the block-7 lens of `r0frzl` is 61.0 at epoch 29 (= its block-11
readout, 61.1: with blocks 8-11 frozen the head reads block 7 through four near-identity blocks), 67-70 through epochs 49-99 and still
65.0 at 159, against `r0` 49 -> 35, C kdyck 7 -> 0.5, `r0frz` 0.3 throughout, and the C1 arms' 50-56 during their loss window. Block 9 is at
73 and block 11 at 77 at 159 (`r0`: 68.5 / 76.8; C: 11.5 / 77.2). So the reverse freeze produced the most extreme early-class-formation
order in the zoo: the class is complete in blocks 0-7 by epoch 29 and everywhere by 160, whereas the winners' late surge (C +2.7 from
164 on) is the late blocks forming the class after 160 (C's block-9 lens 11.5 at 159). `r0frzl` has no such reservoir; its remaining gain
should be `r0`-like (+0.6 from 164 to 299) plus whatever its closing fit deficit adds. **Prediction: random-like, about 78.0 +- 0.5 (77.7-78.4),
i.e. above the low random seed `r0` (77.42) but at the old random mean (78.08), not near `r0frz` (79.16) or C (80.37).** If it lands there,
the freeze arm's gain is learning order (late class formation), not the reduced early budget of the frozen blocks; if it lands at 79+,
the accuracy-shape analogy wins and the lens classification is wrong for this arm. Final ~20:50.

**Interim readings, 2026-09-23 06:40 (matched evaluated epochs; no prediction before ~270, drifts over the last 10 evaluated epochs in
brackets).** *Forced readability on the paper's own init:* `ftb4kc1` (full ksd checkpoint + C1) at epoch 214 is 78.47 against `ftb4k`
79.23 (-0.77, from -0.29 ten epochs earlier) and C ksd 79.10 (-0.63); `ftb4c1` (full kdyck checkpoint + C1) at 189 is 77.55 against
`ftb4` 78.26 (-0.71, flat) and C kdyck 78.40 (-0.85, from -0.52). Both led their references by 1-2 points during the loss window and
have crossed below them, the same signature as `ftbck7c1`; `ftbc7c1` (C kdyck + C1, epoch 189, preempted) is at -0.16 against C kdyck
(from -0.01), the crossing under way. *Compose:* `ftbc7l` (C + late lever, slow steps, fp16) at epoch 154 is 76.36: -0.23 against C kdyck
(76.59) and -2.2 / -2.4 against the two-lever arm `ftbanac` (78.51) and the late-lever arm `ftbrhoplv` (78.74) at the same epoch. The loud
write's mid-run lead over C (+2.2 at 154) is absent when it sits on top of C's blocks 0-7: the arm tracks C, not the late lever, so far;
the final (~21:00) says whether the levers add at the end (C's own late surge takes it from 76.6 at 154 to 80.37). fp16 incidents are
ordinary (scaler-skipped steps in every fp16 arm, no non-finite loss). *Single components, kdyck (epoch 249):* sink alone `ftbc7s` 78.82 =
+1.68 over `r0` (77.14 at 249; gap shrinking from +2.0), -0.53 below S+G, -1.03 below C; gate alone `ftbc7g` 78.04 = +0.90 over `r0`,
-1.31 below S+G. On kdyck the sink is the larger single piece and the two do not simply add (S+G +2.2 over `r0` at 249 against
+1.7 and +0.9 alone). *P + S kdyck* `ftbc7ps` at 284: 79.75, -0.20 against P+G (79.95), +0.14 against S+G (79.61), -0.58 against C (80.33),
drifts flat: on kdyck every pair of components lands at 79.6-79.9, only all three reach 80.4. *`ftbck7a1g`* (A1 ksd + LayerNorm gains) at 279:
79.38, -0.27 against `ftbck7a1` (79.64) and -0.47 against C ksd: the gains add nothing. *Reverse freeze* `r0frzl` at 104: 74.70, -0.92 against
`r0` (75.61) and +5.1 against `r0frz` at the same epoch (69.64, which then surged to 79.16): freezing the late blocks costs almost
nothing early, unlike freezing the early ones; whether it gains at the end is open. *P alone kdyck* `ftbc7p` at 84: 74.50, -0.24 against
`r0` and +2.1 against C (which starts slowly): too early.

**FREEZE ARM FINAL (2026-09-22 15:44, last epoch): `r0frz` = plain timm random, blocks 0-7 at lr 0 for epochs 0-29, linear release
over 30-49, then normal: 79.16 (train loss 2.768).** Against random `r0` 77.42 (train loss 2.258) that is +1.74, against the old
random mean 78.08 +- 0.19 it is +1.08; the committed kdyck arm C is 80.37, the kdyck prefix 79.89, the scale-only recipe 80.24.
So a pure learning-order manipulation with no checkpoint statistic of any kind, on a plain random init, produces 47-76% of C's gain
(n = 1; a second seed and a longer-schedule control for the 35% integrated-lr deficit of blocks 0-7 are the obvious follow-ups).
Its dynamics were those of the winners throughout: block-7 lens at chance for the whole run, block 9 at 1-2%, the class formed in
blocks 10-11 only, a fit deficit larger than any winner's (2.77 against C's 2.46), and a gap to random that closed monotonically from
-12 at epoch 29 to +1.7 at the end. This is the strongest single piece of evidence for M3 (learning order / late class formation
as the mechanism) and against M1 (slow steps) and the "statistics" reading of the early lever.

**REVERSE FREEZE `r0frzl` (control for `r0frz`; queued 2026-09-22 17:00, Simon: high priority on H200).** Plain timm random init,
blocks 8-11 at lr factor 0 for epochs 0-29, linear release over 30-49, normal from 50; blocks 0-7, embeddings, norm and head train
from the start (`run_train_r0frzl.sh` = `run_train_r0frz.sh` with `--release_blocks 8,9,10,11`). Reading: if the freeze arm's gain is
about learning ORDER (the late blocks must form the class before the early ones move), freezing the late blocks instead should give
nothing or a loss; if it is about a reduced early training budget of the frozen blocks, this arm should gain as well. Verification:
init dump through main.py == the `r0` init bit for bit (job 29759800); the freeze mechanism is the one verified bit-exactly on the
`r0frz` checkpoints (section 5f), only the block list differs; the 2-rank smoke could not be run because every L40S test node tried
today (dlc2gpu08, 11, 17) hung at the first NCCL collective, so the check moves into the live run: after its epoch 0 the per-epoch
checkpoint's blocks 8-11 must equal the init bit for bit (watcher). Jobs: results id 29760130 (shared-H200 primary, first of ours
in that queue with the other pending shared jobs held; the next slot frees ~21:10), continuation 29760131, group-H200 hand-over
29760132 (cap full until a group slot frees), continuation 29760133. STARTED 2026-09-22 22:38 on the shared partition (dlc2gpu22);
live check after epoch 0 PASSED: blocks 8-11 identical to the timm init in 48 of 48 tensors, all 104 other tensors moved, `[release]`
lines show lr factor 0 for blocks 8-11. Epoch 105 at 06:30 on 09-23, final ~21:00.

## 5f. Code review of the intervention arms (three independent reviewers, 2026-09-22 09:00-09:40)

**Freeze and release (`r0frz`): correct, no blocker.** Blocks 0-7 bit-identical to the timm init through epoch 29 (96/96 tensors,
against the init dump and between the epoch-0 and epoch-29 checkpoints); the 56 other tensors train from epoch 0; the released lr
follows schedule x factor exactly (logged min_lr = 2e-3 x E/50 x (E-30)/20); torch AdamW with lr 0 is a bit-exact no-op, decay
included; warm moments benign (epoch-30 movement of blocks 0-7 = 2-3% of a full-lr epoch, as the ramp predicts; no jump at release).
Write-up caveats: (i) the ramp multiplies the WARMUP lr, so blocks 0-7 get 35% of the other blocks' integrated lr up to epoch 50 and
reach full lr exactly at the schedule peak: a deficit mixes "less training" with "learning order"; (ii) weight decay is 0 while
frozen and ramped with the lr (wd/lr ratio unchanged); (iii) write "lr factor 0, moments running", not "frozen"; (iv) patch / pos /
cls / final norm / head train from epoch 0, so epochs 0-29 are a random-feature front end with a trainable embedding and a
4-block back end. Operational: an NCCL collective timeout in epoch 94 (09-21 17:35) left the job idle for ~7 h before it exited.

**Align loss / forced readability (C1 arms): correct, no blocker.** Training-time lens == evaluation lens (numerically, diff 0);
active for epochs 0-49 only, survives resume and the 8 -> 4-rank hand-over; scaled and accumulated once with the main loss; DDP
sums the two-path gradients before one all-reduce (no double count, no hang). One logging inconsistency (fixed 09:40): the batch-wise
wandb / tensorboard `train_loss` included the auxiliary term (~1.3 higher during epochs 0-49); the console line and log.txt always
used the main loss. Caveats: (i) the aux term trains the head and final norm on block-7 features, so the lens-7 of the C1 arms is
not the passive probe it is elsewhere: use a post-hoc held-out linear probe on the per-epoch checkpoints for cross-arm comparisons;
the decay after switch-off (ksd 61 -> 33) is partly the head forgetting block 7; (ii) blocks 0-7 receive a second gradient during
warmup (deep supervision), a confound with "readability" as such; (iii) `ftbc7c1`'s epoch-49 collapse (grad-norm NaNs, loss to
7.2, accuracy 0.27, recovered by epoch 54; LN biases and attn.proj of blocks 0-2 moved most) happened with the loss still on; the
committed arm itself had a smaller spike of the kind at epoch 64 and ended at 80.37; carry it as a caveat, n = 1.

**Suppress penalty (`r0sup`): implemented as specified, but NOT a meaningful test of causality.** (i) KL(u || lens) =
logsumexp(z) - mean(z) - log K is a logit-VARIANCE penalty, blind to ordering: satisfied at KL 0.02-0.03 while the lens top-1 stays
15-28% (chance 0.1%). (ii) A loophole: the LN normalisation is differentiated (shrinking the token does not help), but the early
stack inflates the class token along the dimensions the detached final-LN gain suppresses (dims 334 and 62, gamma 0.035 / 0.094):
at epoch 49 98.6% of the block-7 class-token energy sits in those two dims (r0 91.9% in its top two, second gain 0.445), class-token
rms 31.5 against 18.2 while the patch rms is equal, LN-output norm 2.9 against 6.0. The read-out is diluted through the LN, not
removed; patch tokens are never penalised. (iii) The pressure is tiny: ||0.3 grad_aux|| is 0.7% of ||grad_main|| at the block-7
class token and 0.4% on the parameters of blocks 0-7 + embeddings, cos(grad_aux, grad_main) ~ 0. What the arm can say at most: the
model-head read-out at block 7 is not necessary for the random-init trajectory (its block-7 lens stays at 16-18% after the
penalty is off, block 11 and accuracy unaffected). Reviewer's recommendation: an adversarial linear probe on the block-7 class token
(and the mean patch token) with its own optimiser and gradient reversal into blocks 0-7, weight chosen so the probe stays at
chance, the probe top-1 logged next to the lens top-1 (the eval metric must not be the penalised metric); or the KL form on the
pre-LN token whitened by fixed statistics with a weight that makes the block-7 gradient share O(10%); and the same penalty on a
procedural-init arm, since only that pairing addresses the causal claim about the gain. Docstring of `AuxLensLoss` corrected.

**Decision (Simon, 2026-09-22 11:00): `r0sup` is kept and finished as a REFERENCE** (what a lens-magnitude penalty with ~0.4% gradient
share does: block-7 head read-out at 16-18% for the rest of training, block-9 lens 8 points under random at epoch 69, accuracy
unchanged so far) against which a stronger, ordering-based penalty can be judged later; it is not counted as a causal test.

## 5g. Compose arm: early lever + late lever (`ftbc7l`, relaunched 2026-09-22 14:35 on the group H200 partition)

**Why (Simon):** the paper wants to say that the two levers combine. Decision: commit to the early lever as it is (C = P + S + G on blocks
0-7, the verified design; the factorial shows no subset beating it beyond noise, section 5e) and add the late lever with the
checkpoint's write statistics on all three write matrices of blocks 9-11, i.e. the `ftbrho` init reproduced: v = proj x 3.0071 /
4.6048 / 7.6129 (the square root of the attention factor on each), fc2 x 9.4186 / 27.4658 / 81.7061 (`profile_ftbrhoplv.json`'s
`extra`), with `ftbrhop`'s optimiser setting (slow steps, no lr file, fp16); block 8 timm. This replaces a first version that put the
whole attention factor on proj (`ftbrhop`'s proj / fc2-only multipliers; the same forward write, a different split over v and
proj); that run (results id 29756513) was cancelled at epoch ~3 on Simon's call and its directory set aside as
`results_IMNET_BASE_29756513_superseded_projfc2only`. The v-inclusive form is the one every checkpoint-based late-lever arm and
`ftbanac` carry; the two splits had matched within noise (`ftbrhoplv` 80.15 against `ftbrhopl` 80.13).

kdyck only: the ksd checkpoint's own late blocks give nothing (`ftb4h` ksd 77.88), so a ksd compose arm would test the late numbers
as a task-independent recipe, a different claim (one seed later, for the appendix, if wanted). One seed first; the expected margin
over the better single lever is +0.2 to +0.4 (`ftbanac` 80.55 against `ftbanap` 80.24 / `ftbrhop` 79.93; `ftbcomp11` 80.63 +- 0.18
with checkpoint weights), below the single-seed resolution, so seeds (compose x 3, C x 2 more) should follow. Wording: sub-additive
is the expectation; the levers are mechanistically distinct (the early lever carries a fit deficit, train loss 2.46 against 2.23 for
random; the late lever none, 2.27-2.29; `ftbanac` sat between, 2.34).

**Construction:** `vitbase_runs/profile_ftbc7l.json` = C's spec (`profile_ftbanapermb7i.json`, unchanged) + the `extra` section of
`profile_ftbrhoplv.json`; `--init_method_scaled_blocks 0-7` (the `extra` multipliers apply to their own blocks regardless; the profile
loader rejects any unknown key, so the spec carries no note); `run_train_ftbc7l.sh` = C's script with the spec swapped.

**Verification (`plots/verify/verify_compose.py`; job 29756561, 1 L40S; the earlier proj/fc2 version had passed the same checks and a
2-rank smoke on H200, 29756323):** spec == C + extra; the init built as main.py builds it equals C's init bit for bit in all 104
tensors of blocks 0-7 and outside the blocks (the joint calibration does not see the late blocks); blocks 9-11 v rows, proj, fc2 ==
timm x multiplier (max relative deviation 1.2e-7), q / k rows and every other tensor of blocks 8-11 == timm; the main.py init dump
equals the verified C dump bit for bit plus the multipliers; write ratios at init on 256 training images: blocks 9-11 attention 1.49 /
1.48 / 1.38, MLP 1.60 / 1.41 / 1.41 (timm 0.14-0.25; `ftbrho` 1.32-1.41 / 1.43-1.47: the same multipliers come out up to 13% louder
at block 9 on top of C's early stack than on timm's, because the stream entering block 9 differs); fp16 forward finite. The L40S test
nodes dlc2gpu11 and dlc2gpu17 hang at the first NCCL collective (unrelated to the arm).

**Figures for the check:** `plots/out/reconstruction_ftbc7l.png` and `reconstruction_ftbc7l_scales.png` (blocks 0-7 as for the
other recipe arms; the late multipliers show in the value / proj / fc2 panels) and `plots/out/reconstruction_ftbc7l_late.png` (write
ratios per block for timm, C, the compose init and the `ftbrho` init).

**Jobs:** results id 29756597 (the shared-H200 primary, cancelled by the group hand-over before it started); group-H200 hand-over
29756600 started at once on dlc2gpu19 from scratch, continuation 29756601.

## 6. How the outcomes would be read

- A1 loses the gain and the sink / gate state decays earlier; B2 gains; B1 and B3 do not -> **M2, slow opening**; section 12 of
  the recipe doc becomes "the scales keep a transparent early stack closed for ~20 epochs".
- A1 loses the gain with the state intact; B3 gains -> **M1 / M7**, step size as such.
- A1 keeps the gain; S + G gains without the profile -> the forward state suffices; the profile matters only where it creates
  the state (kdyck quiet fc1), and "slow learning" is not the explanation on C.
- C1 loses the gain -> the lens transient is a mediator (**M3**); C1 keeps it while A1 / B arms move it -> marker.
- B2 reproduces the gain AND the footprints (probe, attention distance, fit deficit) -> the mechanism is generic; the paper's
  claim shifts from "procedural statistics" to "what they implement", with P + S + G as the instance found by pretraining.
- None of B1-B5 gains while C does, and the factorial shows S or G specific effects -> component-specific roles (**M6**) and
  the replacement controls become the main line.

## 7. What would have to be built (small)

- lr-scale *schedule* (lr scale and row mask as functions of the epoch, weight-decay compensation following it); tests as for
  the row mask. Needed for A4, B2-release.
- `--save_init` by default for new runs (checkpoint of the init itself; today `checkpoint-0` is the state after epoch 0).
- zero-init of proj / fc2 per block: already expressible with the profile's `extra` multipliers (x 0).
- auxiliary lens loss at a chosen block with an epoch window (C1); backward-only detach of the identity path in chosen blocks (C2);
  constant stream divisor after a chosen block (C3); LayerScale restricted to chosen blocks (B5).
- profile extraction without scales / LayerNorm statistics for S + G, with `renormalize: false`; verifier support (the verifiers
  already report `matched`; training aborts on unmet targets).
- `mechanism_autopsy.py --all` (zoo-wide), probe for all blocks.

## 8. Open questions for you

1. Budget: how many full runs for this study, and is the 40-epoch screening-with-promotion protocol acceptable?
2. Both tasks for every experiment, or kdyck as the primary and ksd only for the deciding arms (or the reverse, since ksd is
   where the structure matters)?
3. If a statistics-free recipe (B2) works, is that a welcome result for the paper or a distraction from explaining the procedural
   init? It decides how much of group B runs before group D.
4. Should the late lever be folded in (M4, C3) or kept out of scope for now?
5. Seeds: spend them early on C itself (n = 3 per task) before anything else, or only on deciding contrasts?
