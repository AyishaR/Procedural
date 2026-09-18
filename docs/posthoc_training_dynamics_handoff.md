# Post hoc analysis of layer learning and the shortcut hypothesis

## Goal and scope

Use existing checkpoints to test whether early predictive success in the randomly initialized baseline is followed by reduced learning or usefulness of its late layers, and whether procedural initialization or rescaling changes this sequence.

Keep this investigation **post hoc**. Frozen-backbone diagnostic probes may be fitted; do not retrain or fine-tune backbones. Recommend additional training only if these results identify a specific unresolved question. Resolve architecture, checkpoint contents, data access, and evaluation conventions from the existing project; this handoff does not prescribe an implementation.

## Context and hypotheses

The models undergo **300 epochs of ImageNet-1k training**, with a checkpoint saved after every epoch. Compare the available random-initialization, procedural-initialization, and rescaled-random runs, including available seeds.

Existing final-checkpoint analyses suggest:

- Procedural and rescaled models develop compatibility with the final classifier later across depth; their raw logit lenses rise sharply near the output.
- Their independent linear probes and tuned lenses also improve later across depth. The difference therefore extends beyond compatibility with the final head.
- Transfer and other probe evaluations mostly overlap when plotted against ImageNet decodability instead of depth. Better general early/middle features are not currently supported.

**Historical feature inputs:** All previously shown experiments except NYUv2 depth and the two lenses used both CLS and mean-pooled image tokens. Establish from the project whether these were combined or evaluated separately, including spatial-feature handling for dense tasks. Retain those results as broader representation-quality evidence. They do not establish that each token source is independently worse, or that information absent from pooled features is absent from the full spatial sequence.

**Primary focus for this investigation:** The classification head receives CLS, so use CLS-only decoding to assess the proposed shortcut and readout alignment. Pooled-token and combined-feature probes are selected controls for information accessible outside CLS.

The hypothesis to test is: **intermediate predictions become confident, then baseline late-layer learning or contribution stalls; this sequence is weaker or absent under procedural initialization/rescaling.** Alternatives include ordinary convergence, a redistribution across depth without suppressed learning, or primarily changing readout alignment. Existing endpoint plots do not establish temporal order or causal mediation of the accuracy gain.

## Shared setup

- Start with epochs **1, 2, 3, 5, 10, 20, 40, 80, 150, 225, 300**. Include image-training initialization if saved: after warm-up/rescaling for those conditions. Expand to intervening epochs around interesting transitions.
- Use the same fixed diagnostic training and validation subsets across runs/checkpoints, with consistent preprocessing and repeatable evaluation. Distinguish training-set and validation results.
- Keep probe fitting, hyperparameter selection, and final evaluation separate. Use comparable probe fitting effort and appropriate regularization across representations.
- Compare equal epochs and, where possible, checkpoints with similar full-model diagnostic loss under the same evaluation objective. Record learning-rate schedules or other known run differences when interpreting changes.
- Document actual layer indexing, feature extraction, pooling, and normalization. Verify the final-layer lens reproduces the checkpoint's model output. Use model identities rather than relying on plot colors.
- Compare decoders receiving the same CLS representation, documenting normalization differences. At a few representative checkpoints, add mean-pooled-image-only and existing combined-feature controls; preserve their documented conventions and distinguish them from the new CLS-only results.
- CLS-only refers to diagnostic readout inputs. All forward passes retain the complete token sequence. Block bypass, functional-change, and gradient diagnostics continue to include patch-token computation and its interaction with CLS.

## 1. When do intermediate predictions become confident?

**Experiment:** At each selected checkpoint, apply that checkpoint's own final normalization/readout to intermediate CLS representations. Measure ground-truth accuracy, cross-entropy, and correct-class margins, together with the full model's performance. Produce training-time-by-depth heatmaps. At a common smaller selection of checkpoints, compare this raw lens with both a tuned lens and an independently fitted linear probe, all receiving CLS only. Use the pooled-token controls at a few representative checkpoints to assess information accessible outside CLS.

**Support:** Baseline intermediate predictions through its own head become reliable and confident before the late-layer changes/contributions in experiments 2–3 plateau; procedural/rescaled runs show a different sequence. Strong fitted-probe accuracy alone does not establish use of the proposed shortcut.

**Weakens the hypothesis:** Intermediate predictions become reliable only after late layers have already developed, or early confidence appears without a subsequent reduction in late-layer learning.

**Interpretation:** Correct classification alone does not imply a small gradient. Track final-output error relative to the target distribution, especially when label smoothing or mixed targets are used. Reconstructed diagnostic targets are not the original historical training targets. A tuned-lens/linear-probe improvement establishes decodability, not that the model uses that particular decoding rule.

| Decoder comparison | Interpretation |
| --- | --- |
| Strong CLS-only probe, weak raw lens | Predictive information is accessible in CLS but poorly aligned with the trained head. |
| Adding pooled image tokens substantially improves probing | Useful predictive information is accessible outside CLS; aggregation into CLS becomes a candidate bottleneck. Account for probe capacity and fitting differences. |
| CLS-only and combined-feature probes both improve mainly late | Delayed development of linearly accessible task information extends beyond CLS; this does not prove information is absent earlier. |

## 2. How useful are late layers at different training stages?

**Experiment:** Evaluate the full checkpoint, then bypass individual residual blocks and the final two or three blocks together. Define bypass consistently with the actual architecture. Measure

\[
C_\ell(e)=\mathcal L_{\text{bypass }\ell}(e)-\mathcal L_{\text{full}}(e).
\]

Report loss changes, accuracy changes, and counts of examples corrected or harmed by including the block(s). Complement full bypass with small reductions in residual-branch gain, such as 1 to 0.9, to assess local sensitivity.

**Support:** Baseline late-layer usefulness stops increasing after intermediate confidence rises, while procedural/rescaled late-layer usefulness continues developing.

**Weakens the hypothesis:** Baseline late layers continue acquiring substantial usefulness despite confident intermediate predictions. This challenges the strong claim that the tail becomes obsolete or fails to train.

**Interpretation:** Bypass can create unfamiliar activations, especially across models with different norm ratios. Greater disruption need not imply more useful feature computation; it can reflect readout alignment or scale dependence. Local attenuation is a complementary sensitivity measurement, not a complete remedy. Block effects need not add because blocks interact.

## 3. How much does each block's function change between epochs?

**Experiment:** Compare adjacent checkpoints on identical block inputs. For an input representation from epoch e, evaluate the block from e and from e+1 on that same representation. For residual update function F, one useful relative measure is

\[
U_\ell(e)=\mathbb E_x\left[
\frac{\|F_{\ell,e+1}(h_\ell^e(x))-F_{\ell,e}(h_\ell^e(x))\|}
{\|h_\ell^e(x)\|+\epsilon}
\right].
\]

Document the actual block/update definition. Report functional change separately for CLS and spatial tokens, retaining the complete token sequence as block input. Always use one-epoch intervals, even when displaying only selected epochs; for the final checkpoint use 299 to 300.

**Support:** Baseline late-layer functional change declines or becomes small relative to earlier layers after intermediate confidence rises, while procedural/rescaled runs sustain more late-layer change. Combine this with experiment 2 to establish whether the changes are useful.

**Weakens the hypothesis:** Baseline late layers keep changing and contributing, or their apparent plateau tracks global convergence/learning-rate decay rather than the proposed sequence.

**Interpretation:** Holding inputs identical separates a block's own changes from changes propagated by earlier layers. This measures net change over an epoch, not total optimization activity: within-epoch changes can cancel. Large movement is not necessarily beneficial, and this measure is not invariant to every reparameterization. Raw parameter-gradient or update norms are particularly confounded by the rescaling intervention.

## 4. Supplement: diagnostic gradient snapshots

**Experiment:** Recompute gradients on fixed diagnostic examples under a documented objective. Inspect depth-wise gradient magnitudes alongside loss, confidence, and final-logit error p−q, where q is the diagnostic target distribution. These measurements require no backbone updates.

**Purpose:** Check whether a late-layer plateau coincides with reduced available training signal. Interpret alongside functional change and contribution, not as standalone evidence of starvation. Small gradients throughout a converged model are not evidence of specifically neglected late layers.

These are newly evaluated gradients at saved weights, not recovered historical gradients. Individual AdamW updates cannot be reconstructed from epoch weights alone. Do not interpret an epoch-wise block swap as an isolated optimizer-step contribution; other blocks have co-adapted.

## Outputs and decision

Provide plots and underlying measurements for experiments 1–3, with experiment 4 as supplementary evidence. Include available seed variation, evaluation/probe conventions, and a concise assessment of whether the predicted sequence is supported, weakened, or unresolved. Distinguish variation across examples from variation across training runs.

Observing the sequence supports the hypothesis but does not prove suppression or causal mediation. If it is absent, revise the explanation. If relevant differences are already established at epoch 1, report that temporal ordering is unresolved within the first epoch. Any proposed follow-up training should target a specific gap found here; do not launch it as part of this task.
