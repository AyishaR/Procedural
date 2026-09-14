# What procedural pretraining contributes to a ViT-B, and how it turns into accuracy

*Research note, 2026-09-12. All numbers are produced from the raw run records by
`plots/verify/gather_synthesis.py` and `plots/verify/matched_fit.py`: the construction of every arm is parsed
from its slurm `Namespace` dump, accuracies and losses come from `results/imnet_base/*/log.txt`, init-time forward
statistics from the init dumps, per-block training curves from wandb. The tables are in `docs/synthesis_data.md`
(T1-T7) and `plots/cache/verify/synthesis.json`. Figures: `plots/out/fig18_two_levers_paper.pdf`,
`plots/out/fig19_proc_suffix_paper.pdf`.*

## Summary

Initialising a ViT-B/16 from a checkpoint pretrained on procedural data (proc) and then training it on
ImageNet-1k gives 80.1% top-1 instead of 78.1% from a random init. We asked which part of the checkpoint
carries the two points and how that part acts during training. Three findings.

1. The benefit can be obtained without proc's weights, from either end of the network. Random weights whose
   per-tensor scales in blocks 0-8 follow proc's, together with proc's LayerNorm gain vectors, reach 79.6-80.7
   (the "early lever"). The recipe needs no checkpoint at all: 18 hand-written scale numbers plus LayerNorm gains
   and biases sampled from proc's per-block mean and std (36 numbers) give 80.2 (`ftbanap`), and proc's permuted
   gain vectors instead of sampled ones 80.7 (`ftbanag`). Random weights whose blocks 9-11 are amplified so that they write 6-10 times more into the
   residual stream reach 79.7-80.1 (the "late lever"). Each lever alone is worth as much as proc's own prefix;
   combined they are sub-additive (best arm 80.6).
2. What both levers change is the depth profile of how much each block writes into the residual stream: proc
   is a loud block 0, a nearly silent middle and a loud top, and every random init that ends up near 80 starts
   with part of that profile. But the profile alone is not sufficient: an init that reproduces the early lever's
   profile with isotropic LayerNorm gains is 1.5 points *worse* than random, and the same init with proc's
   permuted gain vectors is the best early-lever arm (80.7). The per-channel gain pattern matters, its channel
   identity does not, and the LayerNorm biases are not needed (`ftbanab`, 76.7).
3. The two levers reach 80 by a different route than proc's weights do. Proc's weights make the network fit the
   training set less (train loss 2.5-2.6 vs 2.2) and its test loss keeps falling to the end of training. The
   scale levers fit the training set as well as a random init does, are already ahead by epoch 150, and then
   overfit less in the last third of training, when the random init's test loss rises by 0.13 nats. A
   linear-probe measurement locates the difference in where class information forms: winners keep the middle
   blocks free of it from the first epochs; the random init commits class information to block 7 by epoch 30 and
   spends the rest of training undoing it.

The mechanism inside the last step, why a quiet, anisotropic early block or an amplified top block produces a
network that overfits less, is not established; Section 5 lists the arms that would settle it.

## 1. Setup and conventions

ViT-B/16, ImageNet-1k, 300 epochs, batch 4096, AdamW (lr 2e-3, weight decay 0.05, 50 warm-up epochs), RandAugment,
mixup 0.8, cutmix 1.0, label smoothing 0.1, random erasing 0.25; identical for every arm. The train loss quoted
below is the mixup/label-smoothing loss of the last epoch, the test loss is the cross-entropy on the validation
set, and accuracy is the last-epoch top-1 (never the best epoch). The random baseline `r` is timm's
trunc-normal init: 78.08 +/- 0.19 over three seeds. The proc checkpoint (`pr_vitb_n/pr_6066174_final.pth`,
15,000 epochs of procedural pretraining) in all twelve blocks gives 80.09 +/- 0.12. Two three-seed arms are
distinguishable at about 0.45 points; a single-seed arm against a three-seed arm at about 0.6.

Two measurements recur. The **residual-write ratio** of a sublayer is ‖f(x)‖/‖x‖, the norm of the attention or MLP
output over the norm of the residual stream it is added to, averaged over tokens; "loud" and "quiet" blocks refer
to it. The **block probe** is the trained classification head applied to the output of block ℓ; it measures how
much of the final class information is already present after that block, and is logged every ten epochs.

Runs from before the DDP synchronisation fix of 2026-08-31 are used only if their init is deterministic (block
copies, delta-norm scaling); inits that draw random numbers after the DDP wrap (shuffles, quantile matching) from
that era are excluded. Runs of a coworker that share the results directory are excluded.

## 2. Where the benefit lives

### 2.1 Proc blocks at the bottom or at the top

Replacing part of the proc checkpoint with random blocks (T6, fig. 19):

| proc blocks kept | 0-10 | 0-9 | 0-8 | 0-7 | 0-6 | 0-5 | 0-4 | 0-3 | 0-2 | 0-1 | 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 80.37 | 80.24 | 79.99 | 79.89 | 79.67 | 79.66 | 79.89 | 79.58 | 79.49 | 79.11 | 78.78 |
| seeds | 3 | 1 | 3 | 3 | 1 | 1 | 3 | 3 | 1 | 1 | 3 |

| proc blocks kept | 1-11 | 2-11 | 3-11 | 4-11 | 5-11 | 6-11 | 7-11 | 8-11 | 9-11 | 10-11 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 (1 seed each) | 79.85 | 78.68 | 78.82 | 79.11 | 79.67 | 78.72 | 78.84 | 79.69 | 78.89 | 78.65 | 78.55 |

A proc prefix is worth almost the whole effect however short it is: blocks 0-2 give +1.4, block 0 alone +0.7, and
the seven blocks above block 4 add half a point together. A proc suffix is worth +0.5 to +1.8 with no clear
dependence on its length; at one seed per arm the shape of that series cannot be read. The prefix series is also a
series in training loss, from 2.63 (blocks 0-8) down to 2.27 (block 0): the more proc weights at the bottom, the
less the network fits the training set.

### 2.2 The late lever: amplifying random top blocks

`upscale_random_match_delta_norms` starts from a random init and multiplies the write-side matrices (v, proj,
fc2) of the listed blocks, block by block on a batch of training images, until each sublayer's residual-write ratio
equals a target: proc's own ratio at that block (b-series) or a constant (1.4 for `ftbrho`). No proc weights enter.

| amplified blocks | 11 | 10-11 | 9-11 | 9-11 to 1.4 | 7-11 | 6-11 | 5-11 | 4-11 | 3-11 | 2-11 | 1-11 | 0-7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 79.16 | 79.78 | 80.00 | 79.69 | 80.10 | 79.04 | 79.20 | 78.93 | 78.94 | 78.67 | 78.68 | 77.27 |
| seeds | 1 | 3 | 3 | 3 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 |

Amplifying the top two to five blocks is worth as much as a proc prefix. Amplifying six or more blocks is worth
+0.6 to +1.2, and amplifying blocks 0-7 (`ftb4o`) is harmful. Amplifying only the attention sublayer (rattn
series) gives +0.7 to +1.3. Every b-series arm ends with a train loss within 0.1 of the random init's.

The converse manipulation, proc's own top blocks *downscaled* to the random init's write ratios (e-series), costs
nothing when the proc part is deep (proc 3-11 downscaled: 80.14; all twelve proc blocks downscaled: 80.21) and is
harmful when only one or two proc blocks are kept and quietened (proc 10-11: 76.74, proc 11: 76.37).

### 2.3 The early lever: what of proc's prefix is needed

All arms below act on blocks 0-8 and leave blocks 9-11 random (T1).

| what blocks 0-8 receive from proc | arm | top-1 (seeds) | vs random |
|---|---|---|---|
| the weights, intact | `ftb3i` | 79.99 (3) | +1.91 |
| the weights, entries permuted within each tensor; all 1-D vectors permuted | `ftb4e3fix` | 79.50 (3) | +1.42 |
| per-tensor value distribution rank-mapped onto random positions, v matched per slice; LN gains and biases permuted; linear biases zero | `ftbqmlnvo` | 79.93 (3) | +1.86 |
| as above, values replaced by a Gaussian with the tensor's mean and std | `ftbqmlnvog` | 79.55 (3) | +1.48 |
| as above, values replaced by a Student-t with the tensor's kurtosis | `ftbqmlnvot` | 80.36 (1) | +2.28 |
| as `ftbqmlnvo`, plus proc's linear biases permuted | `ftbqm1dvo` | 79.11 (3) | +1.03 |
| as `ftbqmlnvo`, without any 1-D vector (LN gains 1, biases 0) | `ftbqmvo` | 78.61 (3) | +0.53 |
| per-tensor distribution with q, k, v pooled (v loud); LN from proc | `ftbqmln` | 78.18 (3) | +0.10 |
| per-tensor norms only, random directions, LN gain uniform at proc's rms | `ftbnorm` | 78.28 (3) | +0.20 |
| random init, v scaled by 0.46, nothing else | `ftbvd` | 78.74 (3) | +0.66 |
| random init, v scaled by 1.74 | `ftbvu` | 77.65 (3) | -0.43 |
| random init, q and k scaled by 2.18 | `ftbqu` | 78.05 (3) | -0.03 |
| proc's write ratios imposed through v, proj, fc2 on random q, k, fc1; LN 1 | `ftbrhos` | 75.07 (1) | -3.01 |
| random forward pass, proc's relative Adam step sizes | `ftblrm` | 77.73 (1) | -0.35 |
| smooth 18-number std profile (block 0 + linear ramp), LN gains 1, biases 0 | `ftbana` | 76.61 (1) | -1.47 |
| `ftbana` with the top blocks' MLP write flattened | `ftbanaf` | 76.41 (1) | -1.67 |
| `ftbana` + proc's LN gains, permuted, effective scales unchanged | `ftbanag` | 80.70 (1) | +2.62 |
| `ftbana` + proc's LN biases, permuted | `ftbanab` | 76.70 (1) | -1.38 |
| `ftbana` + Gaussian-sampled LN gains and biases | `ftbanap` | 80.24 (1) | +2.16 |
| `ftbanap` with gains ~ N(1, 0.25) and weights left as `ftbana`'s (anisotropy only) | `ftbanau` | 77.35 (1) | -0.73 |
| `ftbanap` with q, k, fc1 at random effective scale, MLP write matched | `ftbanai` | epoch 124: 76.1 (`ftbanag` 76.3, `ftbana` 74.9) | |
| `ftbanap` with isotropic gains and `ftbanap`'s relative Adam steps (slow steps only) | `ftbanal` | epoch 99: 75.5 (= `ftbanap`, `ftbana` 74.0) | |
| `ftbanap` + blocks 9-11 amplified to write ratio 1.4 | `ftbanac` | prepared, waits for `ftbanap` | |

In decreasing order of confidence:

1. **The weights themselves are not needed.** Proc's per-tensor value distribution rank-mapped onto random
   positions (`ftbqmlnvo`) equals intact proc blocks (`ftb3i`) within 0.1.
2. **The shape of the value distribution is not needed, or is worth a few tenths at most.** Gaussian and Student-t
   donors with proc's per-tensor second moments give 79.55 +/- 0.74 and 80.36 against 79.93 +/- 0.39.
3. **The scale of the value path is the decisive scalar, and it is monotone.** Scaling v down by 0.46 in an
   otherwise random init gives +0.66; scaling it up by 1.74 gives -0.43. `ftbqmln` and `ftbqmlnvo` differ only in
   whether v carries the pooled qkv scale (loud) or its own (quiet), and give +0.10 against +1.86. Sharper attention
   logits alone (`ftbqu`) do nothing.
4. **The LayerNorm gain vectors are part of the mechanism.** Removing the LN vectors from `ftbqmlnvo` loses 1.3
   points. Replacing proc's per-channel gains by their mean while keeping every effective scale identical
   (`ftbana`) does not merely lose the effect, it makes the init 1.5 points worse than random. Restoring the
   permuted gain vectors (`ftbanag`) gives 80.70, the best early-lever arm; restoring the biases instead
   (`ftbanab`) gives 76.70, indistinguishable from `ftbana`. Anisotropy on its own is not the answer either:
   `ftbanau` (gains ~ N(1, 0.25), weights as `ftbana`) gives 77.35, so the per-channel gain pattern is worth about 0.7
   of the 4.1 points between `ftbana` and `ftbanag`; the other ingredient that `ftbanag` adds is q, k, v, fc1 matrices
   2.5 times larger in rms, which Adam moves 2.5 times more slowly relative to their size (`ftbanal`, running). Since the gains are permuted, what matters is that the gain varies across channels
   (mean 0.4, std 0.1 in proc), not which channel has which gain. Adding proc's linear biases on top hurts
   (`ftbqm1dvo`, -0.8).
5. **Two things that are individually inert.** Proc's residual-write budgets imposed on random q, k, fc1
   (`ftbrhos`, -3.0) and proc's relative Adam step sizes with an unchanged forward pass (`ftblrm`, -0.35).

Established by `ftbanap` (80.24): the LayerNorm vectors can be sampled from two numbers per vector; the early
lever is checkpoint-free. Not yet established: whether the input-side scales (q, k, fc1) are necessary (`ftbanai`,
running) and whether the slow input-side steps alone suffice (`ftbanal`, running, on `ftbanap`'s curve at epoch 99).

### 2.4 Combining the levers

`ftbcomp11`, proc blocks 0-10 with block 11 amplified to 1.4, is the best arm at 80.63 +/- 0.18, 0.26 above the
same prefix without amplification. `ftb4jd` (proc 0-7, blocks 8-11 amplified to half of proc's ratios) gives
80.11; `ftbcomp25` (proc 0-3, blocks 9-11 amplified to 0.25) 80.16; `ftbcomp1` (proc block 0 only, blocks 9-11
to 0.25) 79.98. Each lever alone is worth +1.6 to +2.3; both together +2.5.

## 3. What the winning initialisations share

### 3.1 The write profile at initialisation, and why it is not sufficient

Residual-write ratios measured on 64 validation images before any training (T4):

| | block 0 | blocks 1-8 | blocks 9-11 |
|---|---|---|---|
| random | 0.31 / 0.70 | 0.19 / 0.36 | 0.15 / 0.25 |
| proc, all blocks | 3.88 / 29.3 | 0.08 / 0.02 | 0.44 / 0.50 |
| early lever `ftbqmlnvo` | 1.43 / 2.66 | 0.13 / 0.04 | 0.09 / 0.13 |
| late lever `ftbrho` | 0.31 / 0.70 | 0.19 / 0.36 | 1.39 / 1.42 |
| `ftbana` (harmful) | 1.36 / 2.63 | 0.13 / 0.04 | 0.09 / 0.14 |

(attention / MLP.) Proc is a loud first block, an almost silent middle, and a loud top. The early lever reproduces
the first two, the late lever the third. Every trained network, the random init included, ends with this shape:
at epoch 289 the random init's MLP write ratios are 1.9 for block 0, 0.41 for blocks 1-8 and 0.96 for blocks
9-11 (T5). So training builds the profile by itself, and the winning inits start closer to where training goes.

The profile is nevertheless not what makes an init work. `ftbana` matches the early lever's profile in every block
to within ten percent and is harmful. `ftbrhos` carries proc's write budgets and is the worst arm. Proc's own top
three blocks, as loud in attention as the late lever and twice as loud in the MLP, are worth +0.8 (`ftb3h`)
where the late lever is worth +1.6 to +1.9.

### 3.2 Where class information forms during training

The block probe over training (T5; seed means; fig. 18 and 19) separates winners from losers more sharply than
any initialisation statistic:

| arm | final top-1 | block-7 probe peak (epoch) | blocks 6-9 probe at epoch 289 |
|---|---|---|---|
| random | 78.1 | 48% (29) | 21% |
| late lever `ftbrho`, `ftb3b` | 79.7, 80.0 | 1.4%, 0.4% | 0.4% |
| early lever `ftbqmlnvo`, twins | 79.9, 79.6, 80.4 | 26%, 26%, 23% | 7-9% |
| proc prefix `ftb3i`, `ftb1i` (1 seed), full proc | 80.0, 80.4, 80.1 | 9%, 10%, 10% | 1%, 2%, 2% |
| both levers `ftbcomp11` | 80.6 | 0.4% | 0.2% (block 10 at 5.5%, block 11 at 80.6%) |
| `ftbanag` (gains restored) | 80.7 | 25% | 11% |
| `ftbana`, `ftbanaf`, `ftbanab` | 76.6, 76.4, 76.7 | 42%, 42%, 43% | 33%, 35%, 34% |
| `ftbrhos` | 75.1 | 44% | 39% |
| `ftblrm` | 77.7 | 39% | 17% |
| proc 5-11 `ftb7h` | 79.7 | 46% | 27% |

In a random init, block 7 becomes 48% class-decodable within 30 epochs and then loses most of that over the
remaining 270, ending at 15%; the final readout is spread over blocks 6-11. Every scale-lever and proc-prefix
winner suppresses this early commitment, partly (early lever) or completely (late lever), and ends with the
readout in blocks 10-11. The combined arm `ftbcomp11` is the extreme case: no transient at all, and at the end
the class information sits in block 11 alone. Every damaged arm shows the transient at full strength and ends with
*more* class information in the middle blocks than the random init. Within the analytic family the two go together exactly:
`ftbana`, `ftbanaf` and `ftbanab` have the write profile and the transient; `ftbanag` has the gain vectors and
not the transient. The one exception is the proc-suffix arm `ftb7h`, which gains +1.6 with a full transient and a
spread readout; it is a single seed, and the suffix series has no replication.

### 3.3 Fit and generalisation: two routes to the same accuracy

Over the 112 valid arms, final accuracy rises with final training loss (Pearson +0.61, Spearman +0.71), and the
final test loss is linear in the final training loss (r = -0.80; test loss = 2.65 - 0.66 x train loss, residual sd
0.06). Arms more than two residual sd above this line, that is with a worse test loss than their fit predicts,
are exactly the damaged ones (`ftbrhos`, `ftb1e`, `ftb11isfix`, two older damaged arms; `ftbana` is at 1.7 sd).

The residual from that line separates the winners into two groups (T1):

| group | arms | final train loss | test-loss residual | reading |
|---|---|---|---|---|
| proc weights in the prefix | `p`, `ftb3i`, `ftb1i`, `ftb7i`, `ftbcomp11`, `ftb4jd` | 2.47-2.64 | -0.03 to +0.04 | on the line: the whole gain comes from fitting less |
| scale levers on random weights | `ftbrho`, `ftb3b`, `ftb2b`, `ftb5b`, `ftbqmlnvo`, twins, `ftbanag`, `ftbanap`, `ftb4e3fix`, `rattn3` | 2.23-2.38 | -0.07 to -0.10 | below the line: better test loss at the same fit |
| proc weights elsewhere or rescaled | `ftb11h`, `ftb4l`, `ftb4m`, `ftbcomp1`, `pds12`, `ftb9e` | 2.32-2.49 | -0.04 to -0.08 | between the two |
| damaged | `ftbrhos`, `ftbana`, `ftbanaf`, `ftbanab`, `ftbanau`, `ftb1e`, `ftb2e`, `ftb11isfix` | 2.20-2.29 | +0.10 to +0.15 | above the line |
| inert | `ftblrm`, `ftbqu`, `ftbqmln`, `ftbnorm`, `ftbvd`, clipped-random controls | 2.19-2.28 | -0.01 to +0.04 | on the line, next to random |

Proc's weights act as a regulariser in the usual sense: the network fits the augmented training set less, and its
test loss is what the fit predicts. The scale levers, early or late, fit the training set about as well as a random
init and still end with a lower test loss. `ftbqmlnvo` has a little of both (+0.10 fit deficit, -0.07 residual);
`ftbcomp11` combines proc's prefix with the late lever and ends on the line at the lowest test loss of any arm
(0.913).

## 4. From initialisation to accuracy: the sequence of events

The training curves (T3, T7, and the test-loss table below) let the process be described stage by stage. Each
stage names what is observed, in which arms, and where the chain breaks for the arms that fail.

**Stage 1, step 0.** The init sets the forward-pass statistics described in 3.1. This is verified by construction
for every arm through the dump-and-measure pipeline and involves no uncertainty.

**Stage 2, the first 50 epochs: where class information first appears.** Two things do *not* matter here. The
speed of early progress does not: at epoch 9 the late lever is at 36% top-1, random at 33%, the early lever at 21%,
proc prefixes at 12-15%, and all of them end near 80 or above. Generalisation does not differ yet either: at
epochs 29-49, test loss at matched training loss is within 0.03 of random's for every winner. What does happen is
the block-probe transient of 3.2: by epoch 30 the random init has committed class information to its middle blocks,
and every winner has not. Damage is also visible this early: `ftbana` and `ftbrhos` already have a test loss 0.05
to 0.09 above random's at the same training loss by epoch 29-49.

**Stage 3, epochs 50-150: the scale levers get ahead while fitting normally.** By epoch 149 the scale-lever
arms lead the random init in accuracy by 0.6 to 1.3 points (77.7-78.4 vs 77.1) with a training loss within 0.1 of
it. The proc-weight arms are *behind* at this point (74.8-76.1) with a training loss 0.25-0.40 higher, and their
test loss is higher than random's. `ftbana` and `ftbrhos` have stopped improving (76.1, 74.9) and stay there.

**Stage 4, epochs 150-300: the learning-rate decay, where the final ordering is set.** Between epoch 150 and 300
the random init gains one point of accuracy while its test loss *rises* from 1.06 to 1.19; it is fitting the
training set further (train loss 2.96 to 2.23) at the expense of calibration. The arms differ in how much of this
loss-overfitting they undergo:

| arm | test loss at 149 | at 299 | change | top-1 at 149 → 299 |
|---|---|---|---|---|
| random | 1.059 | 1.187 | +0.13 | 77.1 → 78.1 |
| `ftblrm` (inert) | 1.058 | 1.194 | +0.14 | 76.9 → 77.7 |
| `ftbrho` / `ftb3b` (late lever) | 0.987 / 0.991 | 1.059 / 1.037 | +0.07 / +0.05 | 78.4 → 79.7 / 78.2 → 80.0 |
| `ftbqmlnvo` / Gaussian / Student-t twin (early lever) | 1.016 / 1.003 / 0.984 | 1.033 / 1.044 / 0.983 | +0.02 / +0.04 / 0.00 | 77.8 → 79.9 / 77.7 → 79.6 / 77.9 → 80.4 |
| `ftbanag` (analytic scales + proc gain vectors) | 1.032 | 0.988 | -0.04 | 77.8 → 80.7 |
| `ftbanap` (analytic scales + sampled LN statistics) | 0.997 | 1.005 | +0.01 | 78.1 → 80.2 |
| `ftb3i` / `p` / `ftbcomp11` (proc weights) | 1.136 / 1.085 / 1.106 | 0.951 / 0.979 / 0.913 | -0.19 / -0.11 / -0.19 | 74.8 → 80.0 / 76.1 → 80.1 / 75.5 → 80.6 |
| `ftbana` / `ftbrhos` (damaged) | 1.120 / 1.180 | 1.292 / 1.342 | +0.17 / +0.16 | 76.1 → 76.6 / 74.9 → 75.1 |

Three regimes. The random init and the inert arms overfit in loss by 0.13-0.14 nats. The scale levers reach the
same training loss and overfit by 0.00-0.07 (`ftbanag`, with a +0.16 fit deficit, does not overfit at all). The proc-weight arms never overfit: their test loss falls to the last
epoch, and they overtake everyone in the final third. The damaged arms overfit more than random, from an earlier
minimum (epoch 129-149), and their accuracy is flat.

**What the chain says, and what it does not.** The observable sequence is: an initialisation that keeps the
middle blocks from committing class information early (Stage 2) leads to a network that is ahead by mid-training
at the same fit (Stage 3) and that degrades less, or not at all, when the learning rate decays (Stage 4). Proc's
weights add a second, independent mechanism, a persistent fit deficit that removes the loss-overfitting entirely
at the price of slower early training. What the chain does not contain is the step from "middle blocks stay
non-decodable" to "less overfitting at the end": we have no measurement of what the winners' early blocks compute
that a random init's do not, and no arm that manipulates the transient directly. Also, the proc-suffix arm
`ftb7h` gains without suppressing the transient, so the stage-2 signature is sufficient in the arms we can
construct but has not been shown to be necessary.

## 5. Open questions and the runs that would close them

1. **The carrier inside the early lever: channel anisotropy or step size.** `ftbanag` restores the gain vectors and
   the full effect (80.70), but keeping the effective scales fixed made its q, k, v and fc1 matrices 2.5 times larger in rms,
   so under Adam they move 2.5 times more slowly relative to their size. `ftblrm` showed that slow steps on a
   random forward pass do nothing; it did not show that fast steps on the correct forward pass are harmless. One
   arm separates the two: `ftbana` with a random gain vector of mean 1 and 25% spread (anisotropy without any
   change of weight rms). That arm, `ftbanau`, finished at 77.35: anisotropy alone recovers 0.7 of the 4.1 points.
   The complementary arm `ftbanal` (`ftbanap` with isotropic gains and per-tensor learning-rate scales that
   reproduce `ftbanap`'s relative steps, verified to match its weight dynamics to 0.5% over four steps) is running
   on L40S; if it reaches ~80 the early lever is profile + slow input-side steps, if it lands near 78 both halves
   are needed.
2. **Whether the gain vectors can be sampled: yes.** `ftbanap` ends at 80.24 with LayerNorm gains and biases drawn
   from proc's per-block mean and std. The checkpoint-free recipe is 18 scale numbers plus 36 LayerNorm statistics.
   One seed; two more are the next runs.
3. **The input-side scales (q, k, fc1).** Untested except through the confounded `ftbrhos`. `ftbanai`, launched
   2026-09-12: q, k, fc1 at random effective scale with fc2 re-tuned so the MLP write budget stays that of
   `ftbanap` (without that correction, removing the quiet fc1 alone raises the blocks-1-8 MLP write from 0.04 to
   0.18, which would confound the input side with the write budget).
4. **Replication of the proc-suffix series.** The only counterexample to the stage-2 signature (`ftb7h`) and the
   whole h- and e-series are single seeds. Proc 5-11 and proc 4-11 at three seeds would settle whether the
   counterexample is real.
5. **What the early blocks compute.** Three measurements on existing checkpoints: CKA between the early blocks of
   `ftbqmlnvo`, `ftbana` and the random init at matched epochs; the effective rank of block outputs over training
   (logged for a subset of arms); attention locality per block. If the winners' early blocks are random blocks that
   stay near-identity longer, the mechanism is regularisation and the question ends; if they compute different
   features, that is the result.
6. **Why proc's own top blocks underperform the late lever** (`ftb3h` 78.9 against 79.7-80.0), while proc's own
   prefix equals the early lever. Downscaling them makes it worse (e-series, 76.4-77.0 for one to three proc blocks),
   so it is not simply loudness. No arm proposed yet.
7. **The checkpoint-free combination.** `ftbanac`, `ftbanap` plus blocks 9-11 amplified to a write ratio of 1.4
   (tuned to 1.40 / 1.38 / 1.45 attention, 1.40 / 1.50 / 1.36 MLP), prepared and verified; launches once `ftbanap`
   is final. It asks whether an init with no checkpoint anywhere reaches `ftbcomp11`'s 80.6 or stays sub-additive.
8. **A twin for the late lever.** The early lever has been reduced to its ingredients; the late lever has not. Is
   the write magnitude of v, proj, fc2 specifically required, or would amplifying q, k and fc1 as well, or scaling
   the residual branch itself, do the same? And the boundary at five amplified blocks rests on single seeds.

Established as negative results, not missing: the distribution shape, the weights, the channel identities, the
step sizes alone, the write budgets alone, the readout position as a cause, and the write profile by itself.

## 6. The early lever after the analytic family (status 2026-09-13)

The analytic family takes the early lever apart on a random init, one ingredient at a time. All arms act on
blocks 0-8, one seed each, last-epoch top-1:

| arm | what blocks 0-8 get | top-1 | train loss | test-loss residual |
|---|---|---|---|---|
| random init | nothing | 78.08 (3) | 2.225 | +0.01 |
| `ftbana` | 18-number std profile in the weights; LN gains 1, biases 0 | 76.61 | 2.209 | +0.11 |
| `ftbanaf` | `ftbana`, top blocks' MLP write flattened | 76.41 | 2.207 | +0.10 |
| `ftbanab` | `ftbana` + proc's LN biases, permuted | 76.70 | 2.194 | +0.10 |
| `ftbanau` | `ftbana` + gains ~ N(1, 0.25), weights unchanged | 77.35 | 2.204 | +0.05 |
| `ftbanag` | same profile, but the 0.4 lives in proc's permuted gain vectors; q,k,v,fc1 2.5x larger | 80.70 | 2.384 | -0.08 |
| `ftbanap` | as `ftbanag` with gains and biases sampled from proc's per-block mean and std | 80.24 | 2.364 | -0.08 |
| `ftbanal` | `ftbanap` with isotropic gains and `ftbanap`'s relative Adam steps via lr scales | running | | |
| `ftbanai` | `ftbanap` with q, k, fc1 at random effective scale (MLP write matched) | running | | |
| `ftbanac` | `ftbanap` + blocks 9-11 amplified (checkpoint-free early + late) | running | | |

**What this establishes.**

1. *The early lever is a checkpoint-free recipe.* Eighteen scale numbers (block 0 plus a linear ramp over blocks
   1-8 for q, k, v, proj, fc1, fc2) and thirty-six LayerNorm statistics (mean and std of the gain and bias per
   block) reproduce the full effect: `ftbanap` 80.24 against the proc prefix 79.99 +/- 0.36, `ftbqmlnvo` 79.93
   +/- 0.39 and the twins 79.6-80.4. Nothing is read from the checkpoint but those 54 numbers, and the 54 were
   themselves smoothed by hand.
2. *Where the factor 0.4 lives decides everything.* `ftbana` and `ftbanag` compute the same function at
   initialisation to within a few percent in every block (identical effective scales, write ratios and attention
   entropy) and end 4.1 points apart. The only difference is whether proc's average LayerNorm gain of 0.4 is
   folded into the weight matrices or kept as a gain vector with the matrices 2.5 times larger. Any account of the
   effect in terms of the forward pass at initialisation is therefore incomplete.
3. *The gain vector contributes two things, and the pattern is the smaller one.* Its channel-to-channel spread
   (`ftbanau`: anisotropy with the weights untouched) is worth 0.7 of the 4.1. Its biases are worth nothing
   (`ftbanab`). What remains is the raw scale of the input-side matrices: under Adam the relative change of a
   matrix per step is about lr / rms(W), so `ftbanag`'s and `ftbanap`'s q, k, v, fc1 move 2.5 times more slowly
   relative to their size than `ftbana`'s. `ftbanal` isolates that ingredient and is on `ftbanap`'s curve at epoch
   99 (75.5 against 74.0 for `ftbana`).
4. *The losing and winning arms differ in the same training signature as the proc-based arms.* `ftbana`, `ftbanab`,
   `ftbanau` show the mid-block transient at full strength (block-7 probe 42-43% at epoch 30), fit the training set
   *better* than random (train loss 2.19-2.21) and overfit in loss by 0.15-0.17 in the last third. `ftbanag` and
   `ftbanap` damp the transient to 25%, fit less (2.36-2.38) and their test loss does not rise. So the analytic
   family reproduces, inside one construction, the two routes of Section 3.3.

**The working hypothesis.** The early lever is the conjunction of a forward-pass profile (quiet writes in
blocks 1-8 behind a loud block 0) and slowly moving input-side weights in those blocks. Either alone fails:
proc's step sizes on a random forward pass do nothing (`ftblrm`, 77.7), the profile with fast steps is harmful
(`ftbana`, 76.6). A mechanism consistent with everything measured: with quiet writes the early blocks contribute
little to the output, but Adam normalises their gradients, so with small weights they still change as fast as in
a random init and quickly build class-specific features that the rest of the network then fits to; with slow
steps they stay close to their initialisation while the readout forms at the top, and the features that are
learned in the early blocks are learned under a working readout. Proc's LayerNorm gain of 0.4 enforces both
halves at once, which is why every proc-derived init has them together and why the coupling was only visible
once the profile was written by hand. This is a hypothesis; `ftbanal` is its direct test.

**What is still unclear.**

- Whether slow input-side steps are *sufficient* given the profile (`ftbanal`, final 2026-09-14 early morning),
  and whether the sharper logits and quiet fc1 are needed at all (`ftbanai`, same time).
- Everything above is one seed per cell. The recipe (`ftbanap`) and whichever of `ftbanal`/`ftbanau` carries the
  mechanism need three seeds before any of it is a claim; the twins' spread (79.2 to 80.4) shows how wide a single
  seed can sit.
- Whether the late lever is the same mechanism. Amplifying v, proj, fc2 of blocks 9-11 also makes those matrices
  3 to 10 times larger, i.e. slows their relative Adam steps by the same factor. No arm has separated "louder top"
  from "slower top". Two cheap arms would: the late lever with its learning rate raised by the amplification
  factor on the amplified tensors (loud but not slow), and a random init with the learning rate of blocks 9-11's
  write matrices lowered by that factor (slow but not loud). If the first loses the effect and the second gains it,
  both levers are one mechanism, slow steps in the right place, and the depth profile is the visible side effect.
- The step from "early blocks stay near their init" to "less overfitting in the last third" is still a story, not a
  measurement. What would test it, on checkpoints we have: CKA between the early blocks of `ftbanap`, `ftbana`
  and the random init at matched epochs (do the winners' early blocks compute something different, or the same
  thing later), the effective rank of block outputs over training, and per-block attention locality.

**Final reading (2026-09-14; `ftbanal` 79.78, `ftbanai` 78.11, one seed each).** `ftbanal` (isotropic gains, `ftbanap`'s
relative Adam steps through per-tensor lr scales) ends 0.46 below `ftbanap` (80.24) with the identical final training
loss (2.364) and a slightly higher test loss (1.028 vs 1.005): the slow input-side steps carry most of the early lever
(+1.7 of +2.2; the remainder is within one seed resolution of the gain's channel pattern's 0.7 in `ftbanau`).
`ftbanai` (input-side effective scales reset to timm's, MLP write and raw matrix sizes kept) ends at the random init's
level (78.11 vs 78.08) with a *lower* training loss (2.314): it fits better and generalises worse, off the fit line on
random's side. Together with `ftbana` (profile without slow steps, 76.61) and `ftblrm` (slow steps without profile,
77.73) the four arms close the decomposition: the early lever is the conjunction of the input-side effective-scale
profile (attention logits 1.75x, fc1 pre-activations 0.36x) and slow relative Adam steps on the same matrices; either
alone is harmful, the pair is worth +1.7, and the gain's anisotropy adds at most 0.5 on top. Proc's LayerNorm gain of
~0.4 is the device that provides both at once (it sets the effective scales while the raw matrices stay 2.5x larger).
What remains open is *why* slow steps on a sharp-logit, quiet-MLP input side generalise better; the measurements in §5
(CKA, effective rank, attention locality over training) are the next step, and `ftbanac` (early + late lever) is at
epoch 110.

**Next runs, in order.** (i) `ftbanal` and `ftbanai` have landed (above); `ftbanac` is at epoch 110 on L40S. (ii) Two more seeds of
`ftbanap`, and of `ftbanal` if it holds, four runs. (iii) The late-lever step-size trio (2026-09-13/14): `ftbrhop` (proj/fc2-only base at `ftbrho`'s per-tensor products,
write ratio 1.4) FINAL 79.93 against `ftbrho` 79.69 +/- 0.30: the late lever is the two writing matrices, v is not
needed. `ftbrhosl` (random weights, lr / multiplier: slow not loud) FINAL 78.31 = random (78.08): slow steps on the writing
matrices alone do nothing. `ftbrhopl` (scaled weights, lr x multiplier: loud not slow; bf16 because the compensated steps push the
top-block activations past fp16) is on `ftbrho`'s curve at epoch 144 (78.51 vs 78.36 at 149), final 2026-09-15 evening.
If it holds, the late lever is the loud write itself and the step size is irrelevant -- the mirror image of the early
lever, where the slow steps carry the effect and the scale profile alone is harmful. (iv) The measurements, which need no training. (v) Seeds of `ftbanac` if it reaches the combined level.
After (ii) and (iii) the early- and late-block results are either one story or demonstrably two, at three seeds.
(vi) Generality across checkpoints (wandb project "vit base kdyck shuffle"; docs/i100_late_block_scaling.md 0d.11,
"Generality test" and "The input side at the function level"). The second procedural checkpoint (`pr_vitb_ksd`; prefix
arms `ftb4i` 80.05, `ftb4` 80.21) shares the early-lever *functional* state with kdyck -- block 0 loud, middle attention
sharp (logit std 12-500 vs random 0.31), middle MLP effectively silent (fc1 pre-activations shifted to a mean of -2 to -3
rms so the GELU is off), large input-side matrices that Adam moves slowly -- but reaches it through structure: the fc1
rows are anti-aligned with the normalised stream, a rank-one relation no per-tensor statistic carries. kdyck's recipe
worked because kdyck's fc1 is small (0.36x), so random matrices at that scale silence the GELU by scale instead; ksd's
fc1 is loud (0.9-2.3x), so every second-moment copy gives a loud, half-on MLP, the `ftbanai` state. Accordingly
`ftbanak` (the `ftbanap` procedure applied verbatim to ksd) FINAL 77.86 = random, and `ftbanakw` (write ratios matched,
GELU still on) is below random at epoch 239. Registered prediction: `ftbqmlnvok` (twin recipe) and `ftbanakg` (exact
scales + permuted LN vectors) also end near random. Consequences: the functional mechanism generalises across the two
checkpoints and ksd confirms it; the parametric extraction procedure ("read 54 second moments off the checkpoint")
does not, and the paper must state the lever as the functional state with `ftbanap` as one checkpoint-free
instantiation. The late lever is untouched (a single number, never read off a checkpoint). Decisive test launched
2026-09-14: `ftbanakb` = `ftbanak` + one fc1 bias per block (8 numbers) gating the MLP to ksd's fraction of positive
pre-activations; ~80 => "scales + gate" is the general checkpoint-free form.

## Appendix: data status

Dynamics (Section 3.2, T5) are in the wandb cache for every arm named in this note; for `ftb1i` only seed 0
has per-block curves in wandb (the resumed seeds 1 and 2 logged none), so its dynamics row is a single seed.
`ftbanag` finished 2026-09-12 07:33 (80.70) and its per-block curves are complete. `ftbanab` finished 2026-09-12 11:23 (76.70). `ftbanap` (epoch 219, 79.59) was preempted on the shared partition
and is queued to resume, `ftbanau` finished 2026-09-13 07:15 (77.35), `ftbanap` 2026-09-13 14:39 (80.24); `ftbanai` (epoch 128) and
`ftbanal` (epoch 101) run on the shared H200 partition, `ftbanac` (epoch 29) on 4x L40S; rows of running arms are read from their partial logs at the epoch
stated. Every other number is final.
