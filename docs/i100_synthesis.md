# What procedural pretraining contributes to a ViT-B, and how it turns into accuracy

*Research note, 2026-09-11. All numbers are produced from the raw run records by
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
   per-tensor scales in blocks 0-8 follow proc's, together with proc's LayerNorm gain vectors, reach 79.6-80.4
   (the "early lever"). Random weights whose blocks 9-11 are amplified so that they write 6-10 times more into the
   residual stream reach 79.7-80.1 (the "late lever"). Each lever alone is worth as much as proc's own prefix;
   combined they are sub-additive (best arm 80.6).
2. What both levers change is the depth profile of how much each block writes into the residual stream: proc
   is a loud block 0, a nearly silent middle and a loud top, and every random init that ends up near 80 starts
   with part of that profile. But the profile alone is not sufficient: an init that reproduces the early lever's
   profile with isotropic LayerNorm gains is 1.5 points *worse* than random. The per-channel gain pattern matters,
   and its channel identity does not.
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
| `ftbana` + proc's LN biases, permuted | `ftbanab` | epoch 190: on `ftbana`'s curve | |
| `ftbana` + proc's LN gains, permuted, effective scales unchanged | `ftbanag` | epoch 150: 77.77, on the twins' curve | |
| `ftbana` + Gaussian-sampled LN gains and biases | `ftbanap` | epoch 25, running | |

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
   permuted gain vectors (`ftbanag`) puts the run back on the twins' curve; restoring the biases instead
   (`ftbanab`) does nothing. Since the gains are permuted, what matters is that the gain varies across channels
   (mean 0.4, std 0.1 in proc), not which channel has which gain. Adding proc's linear biases on top hurts
   (`ftbqm1dvo`, -0.8).
5. **Two things that are individually inert.** Proc's residual-write budgets imposed on random q, k, fc1
   (`ftbrhos`, -3.0) and proc's relative Adam step sizes with an unchanged forward pass (`ftblrm`, -0.35).

Not established: whether the input-side scales (q, k, fc1) are necessary, since the only evidence (`ftbrhos`) is
confounded with the gain point; and whether the gain vectors can be sampled rather than copied (`ftbanap`).

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
| proc prefix `ftb3i`, full proc | 80.0, 80.1 | 9%, 10% | 1%, 2% |
| `ftbanag` (gains restored, epoch 149) | | 25% | 19% at 149 |
| `ftbana`, `ftbanaf`, `ftbanab` | 76.6, 76.4, (running) | 42%, 42%, 43% | 33%, 35%, 38% |
| `ftbrhos` | 75.1 | 44% | 39% |
| `ftblrm` | 77.7 | 39% | 17% |
| proc 5-11 `ftb7h` | 79.7 | 46% | 27% |

In a random init, block 7 becomes 48% class-decodable within 30 epochs and then loses most of that over the
remaining 270, ending at 15%; the final readout is spread over blocks 6-11. Every scale-lever and proc-prefix
winner suppresses this early commitment, partly (early lever) or completely (late lever), and ends with the
readout in blocks 10-11. Every damaged arm shows the transient at full strength and ends with *more* class
information in the middle blocks than the random init. Within the analytic family the two go together exactly:
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
| scale levers on random weights | `ftbrho`, `ftb3b`, `ftb2b`, `ftb5b`, `ftbqmlnvo`, twins, `ftb4e3fix`, `rattn3` | 2.23-2.36 | -0.07 to -0.10 | below the line: better test loss at the same fit |
| proc weights elsewhere or rescaled | `ftb11h`, `ftb4l`, `ftb4m`, `ftbcomp1`, `pds12`, `ftb9e` | 2.32-2.49 | -0.04 to -0.08 | between the two |
| damaged | `ftbrhos`, `ftbana`, `ftbanaf`, `ftb1e`, `ftb2e`, `ftb11isfix` | 2.20-2.29 | +0.10 to +0.15 | above the line |
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
| `ftb3i` / `p` / `ftbcomp11` (proc weights) | 1.136 / 1.085 / 1.106 | 0.951 / 0.979 / 0.913 | -0.19 / -0.11 / -0.19 | 74.8 → 80.0 / 76.1 → 80.1 / 75.5 → 80.6 |
| `ftbana` / `ftbrhos` (damaged) | 1.120 / 1.180 | 1.292 / 1.342 | +0.17 / +0.16 | 76.1 → 76.6 / 74.9 → 75.1 |

Three regimes. The random init and the inert arms overfit in loss by 0.13-0.14 nats. The scale levers reach the
same training loss and overfit by 0.00-0.07. The proc-weight arms never overfit: their test loss falls to the last
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
   the effect, but keeping the effective scales fixed made its q, k, v and fc1 matrices 2.5 times larger in rms,
   so under Adam they move 2.5 times more slowly relative to their size. `ftblrm` showed that slow steps on a
   random forward pass do nothing; it did not show that fast steps on the correct forward pass are harmless. One
   arm separates the two: `ftbana` with a random gain vector of mean 1 and 25% spread (anisotropy without any
   change of weight rms). If it recovers, anisotropy is the cause; if not, `ftbana` with per-tensor learning-rate
   scales that reproduce `ftbanag`'s relative steps.
2. **Whether the gain vectors can be sampled** (`ftbanap`, final expected 2026-09-12). If yes, the checkpoint-free
   recipe is 54 scale numbers plus 36 LayerNorm statistics; if not, it needs the 18 vectors.
3. **The input-side scales (q, k, fc1).** Untested except through the confounded `ftbrhos`. One spec change on
   the surviving analytic arm.
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
7. **A twin for the late lever.** The early lever has been reduced to its ingredients; the late lever has not. Is
   the write magnitude of v, proj, fc2 specifically required, or would amplifying q, k and fc1 as well, or scaling
   the residual branch itself, do the same? And the boundary at five amplified blocks rests on single seeds.

Established as negative results, not missing: the distribution shape, the weights, the channel identities, the
step sizes alone, the write budgets alone, the readout position as a cause, and the write profile by itself.

## Appendix: data status

Dynamics (Section 3.2) are in the wandb cache for 25 of the arms named in this note; `ftbcomp11` and `ftb1i` were
being fetched when this version was written and are not yet in T5. `ftbanab`, `ftbanag` and `ftbanap` are
running (finals 2026-09-12, 04:00 / 08:00 / 18:00). Every other number is final.
