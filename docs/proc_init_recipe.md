# What the procedural initialisation does, as statistics

Status 2026-09-20 (gate target decided: mean pre-activation, section 11). Every formula below was read off the code and then checked numerically against what the code
produces (`plots/verify/verify_recipe_statement.py`, section 5). Section 3 was corrected on 2026-09-19: the effective scale is
the size of the matrix the forward pass applies, not the checkpoint's forward (output) scale; no run is affected. Sections 3-6 describe the first reconstruction arms
(mean-pre-activation gate, block-0 common write, single-context targets; never launched); section 8 the write-ratio component
(launched, harmful); section 9 the final family; **sections 10-12 the results (2026-09-18/19) and what follows from them.** The reference prefix is **blocks 0..7** (prefix arms
`ftb4i`: kdyck 79.89 over three seeds, ksd 80.05 one seed), the split most late-lever arms use. Arms `ftbanaperg0b7`
(kdyck) and `ftbanakperg0b7` (ksd) are set up and verified, not launched. The earlier blocks-0..8 variants
(`ftbanaperg0`, `ftbanakperg0`, reference `ftb3i`) remain; section 7 states exactly how the two differ.

## 1. Claim

A ViT-B initialised from a procedurally pretrained checkpoint in blocks 0..7 gains about +1.8 points of ImageNet top-1
(79.89 against 78.08 +- 0.19 for random).
The claim is that this prefix can be replaced by a **random** network that shares with the checkpoint only

1. one scale per weight matrix and block (48 numbers) and the mean and standard deviation of each LayerNorm vector
   (64 numbers), and
2. three **rank-one alignments with one shared direction of the residual stream** (15 numbers),

and that (1) alone already yields the gain on kdyck (measured with blocks 0..8: `ftbanap` 80.24, quantile twin
79.93 +- 0.39 over three seeds, prefix `ftb3i` 79.99) but not on ksd, where the sink of (2) is what carries the gain
(sections 10-11; the final recipe there has 102 numbers, section 11 (ii)), while (2) makes the random network match a small selected set of the prefix's initial statistics -- much of the kdyck prefix's
initial behaviour, a partial reconstruction on ksd (section 6).

What the prefix is doing, in one sentence: block 0 writes one large vector into every token; in blocks 1..7 the
attention reads only that shared part, so every query attends to the same key, and the MLP's input weights point
against it, so the GELU is off; the middle of the network therefore starts by adding the same vector to every token and
nothing token-specific, and it moves slowly because its input matrices are large behind LayerNorm gains of about 0.4.

## 2. Notation

Block $b$, width $d=768$, $H=12$ heads of size $d_h=64$, MLP width $n=3072$, $\sigma_0 = 0.02$ (timm's initialisation
std). $W_q, W_k, W_v \in \mathbb{R}^{d\times d}$ are the row groups $[0,d)$, $[d,2d)$, $[2d,3d)$ of `attn.qkv.weight`;
$W_{proj}\in\mathbb{R}^{d\times d}$, $W_{fc1}\in\mathbb{R}^{n\times d}$, $W_{fc2}\in\mathbb{R}^{d\times n}$. LayerNorms
$\mathrm{norm}_\nu(x) = \gamma_\nu \odot \hat x + \beta_\nu$, $\nu\in\{1,2\}$, $\hat x$ the standardised stream.
$\mathrm{rms}(A)$ is the root mean square over all entries. $\nu(s)=1$ for $s\in\{q,k,v\}$, $\nu(s)=2$ for $s=fc1$.

## 3. What is read from the checkpoint

`extract_profile.py CKPT OUT.json --blocks 0-7 --gain_fold exact --exact --qk_entropy --fc1_gate --common_write`

**Effective scales** (`gain_folded_scale`, `effective_scales`), for $b=0..7$:

$$e_{b,s} = \frac{\mathrm{rms}\big(W^{ck}_s\,\mathrm{diag}(\gamma^{ck}_{\nu(s)})\big)}{\sigma_0}\quad (s\in\{q,k,v,fc1\}),\qquad
e_{b,s} = \frac{\mathrm{rms}(W^{ck}_s)}{\sigma_0}\quad (s\in\{proj,fc2\}).$$

$W\,\mathrm{diag}(\gamma)$ is the matrix the forward pass applies to the standardised $\hat x$, hence "effective"; $e$ is its
Frobenius size per entry, in units of timm's initialisation. It is the size of that matrix, not the size of its output in the
checkpoint ("What the effective scale is, and is not" below).

**Why the gain is folded in.** The blocks are pre-norm (`timm` `Block.forward`: $x \leftarrow x + \mathrm{attn}(\mathrm{norm}_1 x)$,
$x \leftarrow x + \mathrm{mlp}(\mathrm{norm}_2 x)$) and each norm is an `nn.LayerNorm(768, eps=1e-6)` with an elementwise affine
part: for every token, $\mathrm{norm}(x) = \gamma \odot \hat x + \beta$ with $\hat x = (x - \bar x)/\sqrt{\mathrm{var}(x) + \epsilon}$
standardised over the 768 channels (mean 0, variance 1 per token; checked numerically). A linear layer behind it therefore computes

$$q = W_q\,\mathrm{norm}_1(x) = \big(W_q\,\mathrm{diag}(\gamma_1)\big)\,\hat x + W_q\,\beta_1 .$$

Two things follow. (i) *The input-dependent part of the layer depends on $W$ and $\gamma$ only through the product.* The full affine
map is $z = W\,\mathrm{diag}(\gamma)\,\hat x + W\beta + b$: its input-dependent term is set by $W\,\mathrm{diag}(\gamma)$, its constant
term by $W\beta + b$. So $(W, \gamma, \beta)$ and $(W\,\mathrm{diag}(\gamma), \mathbf 1, \beta)$ agree on the input-dependent term but
not on the offset ($W\,\mathrm{diag}(\gamma)\beta \neq W\beta$ in general; an exact fold to $\gamma' = \mathbf 1$ would need $\beta' =
\mathrm{diag}(\gamma)^{-1}\beta$). A recipe that copied $\mathrm{rms}(W)$ alone would describe a different input-dependent map
whenever $\gamma \neq 1$. The checkpoints' gains are about 0.4 (kdyck norm1 0.31-0.41, norm2 0.39-0.52; ksd 0.25-0.60), so their raw
input-side matrices are about 2.5 times larger than the matrix the forward pass applies: $\mathrm{rms}(W_q)/\sigma_0 \approx 3.5$ against an
effective 1.4. The effective scale is invariant under the split between $W$ and $\gamma$ (the raw $\mathrm{rms}(W)$ is not), and
imposed on a random matrix it fixes that matrix's output size exactly (below), which is why it is what the specification
records (`extract_profile.gain_folded_scale`). The second term, $W_q\beta_1$, is the same vector for every token and is not a scale;
its size is reproduced by the bias statistics (mean, std of $\beta$), its direction relative to the rows of $W$ is not (inert on
kdyck, part of the gate on ksd: `extract_profile.py` docstring).
proj and fc2 have no LayerNorm in front (they read the attention output and the GELU output), so for them the raw
$\mathrm{rms}(W)/\sigma_0$ is the effective scale.

(ii) *The optimiser sees the split.* Adam updates $W$ and $\gamma$ as separate tensors with per-coordinate normalised steps of size
$\approx \eta$, so the relative change of $W$ per step is $\propto \eta/\mathrm{rms}(W)$: a matrix that is 2.5 times larger behind a gain
of 0.4 moves 2.5 times more slowly in relative terms although the function is the same. The recipe therefore has to reproduce
both the size of the product (the matrix the forward pass applies) and how it is split between $W$ and $\gamma$ (the dynamics). That is what Step A does:
$\gamma$ is sampled with the checkpoint's mean and std and $W$ is set to $e\,\sigma_0/\mathrm{rms}(\gamma)$ times timm's matrix
(`utils.apply_analytic_profile`; with `"realise": "exact"` rescaled until $\mathrm{rms}(W\,\mathrm{diag}\gamma) = e\,\sigma_0$ holds to
float precision), so the *global second-moment scale* of the input-dependent map matches the checkpoint's (not the map itself: $W^0$ is random and
$\gamma$ a fresh sample) and the raw matrices come out about 2.5 times larger than timm's, as in the checkpoint. The evidence that
the split is the carrier on kdyck: `ftbana` puts the effective
scales into the raw weights with gains 1 (same second-moment profile, normal steps) and loses (76.61); `ftbanag` is the same arm with the
checkpoint's permuted gains and the weights divided by $\mathrm{rms}(\gamma)$ and wins (80.70); `ftbanal` keeps gains 1 but reproduces
the slow relative steps through per-tensor learning-rate scales and reaches 79.78 (sections 10-12).

*Product versus exact fold.* $\mathrm{rms}(W\,\mathrm{diag}\gamma) = \mathrm{rms}(W)\,\mathrm{rms}(\gamma)$ only if the column norms of
$W$ are uncorrelated with $\gamma_j^2$. In both checkpoints they are mildly positively correlated: the exact fold is larger by 2-11%
in blocks 1-8 and by up to 15% for q of block 0. In the recipe $\gamma$ is a fresh sample independent of $W$, so the two agree to
0.2% (section 6); realising the *exact* fold with an independent $\gamma$ therefore reproduces the size of the checkpoint's folded
matrix $W\,\mathrm{diag}\gamma$ but overshoots its raw scale by that correlation (q/k raw 3.62/4.29 in `ftbanape` against the prefix's
3.49/4.03), while the *product* convention reproduces the raw scale and understates the folded size -- with independent $\gamma$
and $W$ no recipe matches both. The specifications since 2026-09-16 use the exact fold (`--gain_fold exact`). Neither convention
reproduces the checkpoint's *output* size, which alignment raises by factors of 2-11 in rms (next paragraph); against that the
2-15% between the two conventions is a choice of definition, not of function.

**Why the rms: the recipe as a fitted initialisation distribution.** timm initialises every linear weight as
`trunc_normal_(weight, std=.02)` with biases 0 and LayerNorm $\gamma = 1$, $\beta = 0$ (`models/vision_transformer.py`,
`init_weights_vit_timm`; the truncation bounds are the absolute $\pm 2$, i.e. $\pm 100\sigma$, so the draw is a plain
$\mathcal N(0, \sigma_0^2)$): a data-free distribution with one parameter per tensor. Ingredient (1) of the claim keeps that form
and refits its parameters to the prefix: a zero-mean Gaussian per matrix slice, a Gaussian with free mean per LayerNorm vector,
then one sample. For a zero-mean i.i.d. Gaussian the maximum-likelihood estimate of the standard deviation *is* the rms of the
entries (the checkpoint's scalar means are at most 0.017 of the rms), and the mean and standard deviation of a LayerNorm vector
are the fit of its Gaussian, so nothing about the input is assumed when the rms is used: it is the parameter of the family the
initialisation is drawn from, exactly as in the baseline it is compared with. What such an initialisation does to its input then
follows from independence, the argument of every variance-preserving initialisation: for $W$ independent of its input $y$,
$\mathbb E\,\|Wy\|^2 = \sigma_W^2\, d_{out}\, \|y\|^2$ for any fixed $y$, isotropic or not (the next paragraph measures it).
Four qualifications. (a) The refit is of the triple $(W, \gamma, \beta)$, not of $W$ alone: the same effective scales with
$\gamma = 1$ are a different, harmful initialisation (`ftbana` 76.61, below random), because the split between $W$ and $\gamma$
sets Adam's relative step ((ii) above). (b) The fit fixes the rms as the statistic but not the fold: the literal maximum-likelihood
fit of the raw $W$ is the *product* convention, while the *exact* fold matches the second moment of the matrix the forward pass
applies; a model with independent $W$ and $\gamma$ cannot match both (previous paragraph), and the specifications match the second.
The `"realise": "exact"` rescaling makes the sample's own rms exact instead of its expectation, a 0.1-0.2% matter. (c) Not
fitted: the biases of the linear layers, which stay at timm's 0 (checkpoint rms 0.05-0.12 for qkv, 0.006-0.09 for the others, blocks 0-7),
and everything the family cannot express -- correlations between entries, which is where the checkpoint keeps most of its size
(next paragraph). The rms is the best fit within the family, not a good description of the checkpoint. (d) Scope: this is
ingredient (1) only. The sink, the gate and the common write take a direction from the model's own stream on training images and
are a data-dependent initialisation outside this family; on kdyck (1) alone yields the gain, on ksd it does not (section 11).
"Data-free" also refers to the initialisation step, not to where the numbers come from: they are distilled from a pretrained
checkpoint, and the claim is that this much of it suffices, not that the table could be written down a priori.

**What the effective scale is, and is not** (measured 2026-09-19, `plots/verify/effective_scale_meaning.py`: kdyck, 32 validation
images, seed 0; the recipe is `ftbanaperab7i`'s initialisation under the launched protocol). LayerNorm fixes the norm of every
token, $\|\hat x\|^2 = d$, and nothing about the distribution of directions, so

$$\mathbb E\,\|M\hat x\|^2 = \mathrm{tr}\big(M\,\Sigma\,M^\top\big),\qquad \Sigma = \mathbb E[\hat x\hat x^\top],\quad \mathrm{tr}\,\Sigma = d,\qquad
R := \frac{\mathbb E\,\|M\hat x\|^2}{\|M\|_F^2}.$$

$R = 1$, i.e. $e$ sets the output size, in two cases: $\Sigma = I$, which never holds (the largest eigenvalue of $\Sigma$ carries
0.35-0.47 of the energy in blocks 1-7 of a plain timm network, 0.34-0.39 in the recipe, 0.83-0.85 in the kdyck prefix; isotropic: 0.0013), or
**$M$ independent of $\hat x$**, because a random matrix treats every direction alike and only $\|\hat x\|^2 = d$ enters. The second
case is the justification on the initialisation side: timm's matrices give $R = 0.95$-$1.04$ in every block and slice despite
that anisotropy, so an effective scale imposed on a random matrix fixes its output rms at $e\,\sigma_0\sqrt d$. It is not true
of the checkpoint, whose matrices are aligned with the stream (blocks 1-7; block 0 reads the raw embeddings and has $R \approx 1$
for q, k, v):

| blocks 1-7 | kdyck prefix | recipe (`ftbanaperab7i`) | timm |
|---|---|---|---|
| $R$ of q / k / v / fc1 | 20-39 / 5.0-8.8 / 6.7-11.5 / 97-124 | 25-72 / 0.9-1.3 / 1.0 / 10-17 | 0.95-1.04 |
| random matrix behind the model's own $\gamma_1$ / $\gamma_2$ | 0.53-0.68 / 1.7-2.2 | 1.0 / 1.0 | 1.0 / 1.0 |
| share of $\|M\|_F^2$ in the top 1 / top 8 singular directions, q | 17-31% / 48-63% | | 0.5% / 3.7% |
| the same for k; v; fc1 | 9-14% / 50-64%; 5-9% / 26-35%; 35-42% / 43-50% | | 0.5% / 3.7% |
| median singular value relative to a Gaussian matrix of std $\sigma_0$ (q, k, v, fc1) | 0.20-0.25 | | 1.0 |

Three consequences. (a) **$e$ describes the matrix, not the checkpoint's forward pass.** The checkpoint's output energy is 5-124
times what $e$ predicts (2-11 in rms); the difference is alignment, which no per-tensor moment carries, and which the sink and
the gate put back for q and fc1 (their $R$ in the recipe is the rank-one component's doing; k and v stay unaligned, $R \approx 1$).
The checkpoint's gains are aligned too: norm1's suppress the stream's high-energy channels, norm2's amplify them (second row),
which the Gaussian-sampled gains drop. (b) **The recipe replaces a strongly anisotropic matrix by an isotropic one of equal
size.** In the checkpoint half of the size sits in eight directions over a bulk at about 0.22 of timm's scale, the same in every
slice and block; the recipe spreads that size evenly over all 768 directions. That is the claim of section 1, not a defect of the
measurement: a robust bulk scale (the median row) would describe the checkpoint better and give a six to eight times quieter q/k with raw
norms below timm's, against what the winning arms share (raw q/k of 3.2 and more); the earlier full-rank twin already showed that
rank moves the trajectory, not the endpoint. (c) **Scale and rank-one component are the two halves of one output.** Splitting each
output into the part common to all tokens and the token-specific part: the token-specific rms is what the scale sets, and it lands
where the prefix has it (fc1 0.20-0.21 against 0.14-0.23, q 0.78-1.57 against 0.92-1.08); the common part is what the rank-one
components set (common share of q 0.96-0.97 against 0.93-0.96; fc1 pre-activation mean -0.6..-0.8 against -2.0..-2.7, the
active-unit target needing a shallower shift than the prefix's). Not reproduced: k and v carry no aligned part in the recipe
(common share of k 0.18-0.46 against 0.58-0.81, of v 0.19-0.33 against 0.82-0.90; "write side at timm" is the raw timm weight, behind
the sampled gains an effective v of 0.38-0.45 (blocks 1-7) with token-specific rms 0.18-0.23 against the prefix's 0.39-0.49), and the stream
itself is less common-mode (0.20-0.34 of the LayerNorm output's energy against 0.68-0.76), so the sink spends 13-62% of
$\|W_q\|_F^2$ on the mean direction where the prefix has 8-12%. Because the renormalisation keeps $e$ fixed, the specification's
number then covers the random part plus the aligned part; the token-specific scale is $s\,e$ with the logged $s_q$, $s_k$, $s$
(block 1: $s_q = 0.58$). One pooled number per slice also hides the heads: per-head $e_q e_k$ spans 1.7-3.5 in blocks 1-7 (pooled
2.3-2.8) and 0.6-3.0 in block 0.

**LayerNorm statistics** (`layernorm_statistics`): mean and standard deviation (`torch.std`, unbiased) of each of
$\gamma_1,\beta_1,\gamma_2,\beta_2$.

**Joint-statistic targets** (`measure_joint_targets` -> `utils.joint_statistics_per_block`). They are functionals of the
*prefix model*: a fresh timm ViT-B (seed 0) whose tensors of blocks 0..7 are the checkpoint's, everything else random
(this equals the init dump of the prefix arm `ftb4i` tensor for tensor, 152 of 152). Measured on $N=256$ ImageNet **training**
images under the evaluation transform, chosen by `torch.randperm` with generator seed $4000+\text{seed}$
(`utils.calibration_images`); no labels.

- $h_b$, $b=1..7$: attention entropy in nats, averaged over images, heads and queries.
- $m_b$, $b=1..7$: fc1 pre-activation, averaged over images, tokens and hidden units.
- $\kappa_0$: cosine between the patch tokens of block 0's **output**, averaged over images and all ordered token
  pairs (the diagonal is included, as `_token_cosine` computes it).

kdyck: $h = 0.41 .. 1.01$, $m = -1.95 .. -2.68$, $\kappa_0 = 0.912$. ksd: $h = 0.22 .. 2.67$, $m = -1.91 .. -3.17$,
$\kappa_0 = 0.643$. Two weights-only readings are stored for reference and not used: the energy share of fc1's mean row
(kdyck 0.32-0.41 gain-folded, random $1/n = 0.0003$) and of block 0's fc2 mean column (0.026).

## 4. The initialisation

**Step A, second moments** (`utils.apply_analytic_profile`). Start from timm's initialisation $\theta^0$: every block
weight has $\mathrm{rms} = \sigma_0$ (measured 0.9985-1.0018 $\sigma_0$), $\gamma=1$, all biases 0. For $b=0..7$, with a
generator seeded $1000+10\cdot\text{seed}+b$ (draw order: norm1 permutation [unused], $\gamma_1$, $\beta_1$, then the
same for norm2):

$$\gamma_\nu \sim \mathcal N(\mu_\gamma,\sigma_\gamma^2)^{d},\qquad \beta_\nu\sim\mathcal N(\mu_\beta,\sigma_\beta^2)^{d},$$
$$W_s = c_{b,s}\,W^0_s,\qquad c_{b,s} = \frac{e_{b,s}\,\sigma_0}{\mathrm{rms}\big(W^0_s\,\mathrm{diag}(\gamma_{\nu(s)})\big)}\ \ (s\in\{q,k,v,fc1\}),\qquad
c_{b,s} = \frac{e_{b,s}\,\sigma_0}{\mathrm{rms}(W^0_s)}\ \ (s\in\{proj,fc2\}),$$

so that $\mathrm{rms}(W_s\,\mathrm{diag}\gamma) = e_{b,s}\sigma_0$ holds to float precision (`"realise": "exact"`, the default since
2026-09-17; with `"gain_fold": "product"` the denominator is $\mathrm{rms}(W^0_s)\,\mathrm{rms}(\gamma)$ and the product is what
holds). The specifications written before that date carry no `realise` key and use the legacy multiplier
$W_s = \big(e_{b,s}/\mathrm{rms}(\gamma_{\nu(s)})\big) W^0_s$ (proj, fc2: $e_{b,s} W^0_s$), which meets the declared scale only as far as
$\mathrm{rms}(W^0) = \sigma_0$ and $W^0$, $\gamma$ are uncorrelated: within 0.2% (section 6). With $\mathrm{rms}(\gamma)\approx 0.4$ the raw
input-side matrices are about 2.5 times larger than timm's, which is what makes Adam's relative step on them small.
Linear biases, blocks 8..11, embeddings and head are untouched.

**Step B, three rank-one components** (`utils.calibrate_joint_statistics`), on $N$ training images $X$ (same chooser,
seed = run seed), blocks in depth order, $x_b$ = the stream entering block $b$ with all earlier changes in place.
Each component has the form $W \leftarrow s\,W + a\,\ell\, r^\top$ with $\|r\| = 1$ and $\|\ell\| = 1$ for the gate and the common
write; for the sink $\ell = P$ stacks $H$ unit vectors, so $\|P c^\top\|_F = \sqrt H$, and the bound below uses the actual norms, the strength $a$ found by bisection (at most 40 halvings of $[0,\ 0.98\,\|W\mathrm{diag}\gamma\|_F/\|\ell r^\top\mathrm{diag}\gamma\|_F]$,
for the sink the smaller of the q and k bounds; tolerance $10^{-4}$ for the sink and the common write, $10^{-5}$ for the gate) as the
smallest value that meets the target on $X$, and $s>0$ solving

$$\big\|(sW + a\,\ell r^\top)\,\mathrm{diag}(\gamma)\big\|_F = \big\|W\,\mathrm{diag}(\gamma)\big\|_F$$

($\gamma = 1$ for fc2), so every effective scale of Step A is preserved exactly (`_rescale_to_keep_norm`: with
$A = W\,\mathrm{diag}\gamma$, $B = \ell r^\top \mathrm{diag}\gamma$, $\alpha = \|A\|_F^2$, $\beta = \langle A, B\rangle_F$,
$\delta = \|B\|_F^2$, the positive root $s = \big(-a\beta + \sqrt{a^2\beta^2 + \alpha(\alpha - a^2\delta)}\big)/\alpha$ of
$\alpha s^2 + 2a\beta s + a^2\delta - \alpha = 0$; the bracket $a < 0.98\,\|A\|_F/\|B\|_F$ keeps $a^2\delta < \alpha$, so the root
exists and is positive). Since 2026-09-17 the
kept norm follows the specification's convention: $\mathrm{diag}(\gamma)$ for `"gain_fold": "exact"`, the identity (raw norm)
for `"product"`, where $\mathrm{rms}(W)\,\mathrm{rms}(\gamma)$ is the declared quantity (checked: product 1.000000 raw, exact 1.000000 folded).

| component | tensor, block | $\ell$ | $r$ | target |
|---|---|---|---|---|
| common write (`_install_common_write`) | $W_{fc2}$, $b=0$ | $u$, a unit Gaussian direction, seed $5000+10\cdot\text{seed}+b$ | $\mathbf 1_n/\sqrt n$ | token cosine of $\mathrm{block}_0(x_0)$ $= \kappa_0$ |
| sink (`_install_rank_one_sink`) | $W_q$, $b=1..7$ | $P$: the $H$ unit vectors $p_h\in\mathbb R^{d_h}$ stacked, seed $2000+10\cdot\text{seed}+b$ | $c_1 = \mathrm{normalise}\big(\mathrm{mean}_{X,\text{tokens}}\ \mathrm{norm}_1(x_b)\big)$ | attention entropy $= h_b$ |
| | $W_k$, same $a$ | $P$ | a unit Gaussian direction, seed $3000+10\cdot\text{seed}+b$ | |
| gate (`_install_fc1_gate`) | $W_{fc1}$, $b=1..7$ | $-\mathbf 1_n/\sqrt n$ | $c_2 = \mathrm{normalise}\big(\mathrm{mean}_{X,\text{tokens}}\ \mathrm{norm}_2(x_b + \mathrm{attn}(\mathrm{norm}_1 x_b))\big)$ | mean pre-activation $= m_b$, or (default since 2026-09-17, `--fc1_gate_target active_units`) fraction of pre-activations $> 0$ $= a_b$; both were monotone in $a$ over the bisection bracket in every verified run (not a theorem) |

Within a block the order is sink, gate, common write, because norm2 sees the attention output and fc2 reads fc1.
Why these produce the behaviour: head $h$'s query gets $a\,p_h\,(c_1^\top y_i)$ and its key $a\,p_h\,(r^\top y_j)$, so
the rank-one/rank-one contribution to the logit is $a^2 (c_1^\top y_i)(r^\top y_j)/\sqrt{d_h}$ (the base/base and base/rank-one
cross terms remain; the measured sink share and common-query share say the displayed term dominates after calibration); $c_1^\top y_i$ is nearly the same for all tokens (the
common part of the query is 99% or more of its energy after calibration, measured), which leaves a function of the key $j$
alone: a sink. The gate shifts every hidden unit by $-\tfrac{a}{\sqrt n}\,c_2^\top y$. The common write adds
$\tfrac{a}{\sqrt n}(\mathbf 1^\top \mathrm{GELU}(\cdot))\,u$, one direction for all tokens.

Data enters Step B only through $c_1$, $c_2$ and the three strengths. Everything else is a seeded random draw or a
number from section 3.

## 5. Information flow, for following the code

1. **Statistics**: `extract_profile.py` `main` (434) -> `build_profile_specification` (286): `effective_scales` (152)
   and `layernorm_statistics` (176) per block; with the joint flags `measure_joint_targets` (207) builds the prefix
   model, draws images with `utils.calibration_images` (utils 1494) and calls `utils.joint_statistics_per_block`
   (utils 1552). Output: one JSON, e.g. `vitbase_runs/profile_ftbanaperg0b7.json`, keys `q,k,v,proj,fc1,fc2`
   (`per_block`), `ln.stats`, `qk_entropy.entropy`, `fc1_gate.pre_activation_mean`, `common_write.token_cosine`.
2. **Run script** `vitbase_runs/run_train_ftbanaperg0b7.sh` (ksd: `run_train_ftbanakperg0b7.sh`; second moments only:
   `run_train_ftbanapeb7.sh`, `run_train_ftbanakpeb7.sh`): `--initialize "" --init_method analytic_profile
   --profile_spec <json> --init_method_scaled_blocks 0,...,7`. The block list is taken from this flag, the joint
   components from the blocks named in the JSON; nothing in the code fixes the range.
3. **main.py**: seeds per rank (666-670), builds `dataset_train` (675), enters the `analytic_profile` branch (1051):
   `utils.pr_load_model(path="")` wraps the timm model in DDP (whose constructor broadcasts rank 0's parameters), then
   `utils.apply_analytic_profile` (call at 1060, definition utils 1411) runs on **every** rank: it multiplies by scalars and
   samples the LayerNorm vectors from a generator seeded with the run seed, not the rank, so the ranks stay identical. Then, if the
   JSON has joint keys, rank 0 draws the calibration images from `dataset_train.samples` under
   `build_transform(False, args)` and calls `utils.calibrate_joint_statistics` (call at 1079, definition utils 1729,
   installers at utils 1614 / 1658 / 1692); the changed tensors are broadcast from rank 0 (1098) and their equality is
   asserted across ranks. `sync_initialisation` (715; called at 2063 and 2109) broadcasts all floating-point tensors
   again before the analyses and before training.
4. **Logs to look for**: `[analytic_profile] block b: ... (raw rms/0.02 = ..., effective = ...)`, `[common_write]`,
   `[qk_entropy]`, `[fc1_gate]` with target, reached value and `reachable`, then
   `[joint statistics] 15 calibrated tensors identical on all R ranks: True`.
5. **Checks**: `plots/verify/verify_recipe_statement.py CKPT SPEC` (the equations of sections 3-4),
   `plots/verify/verify_joint_statistics.py --spec SPEC [--dump ARM.pth --base BASE.pth]` (structure and targets, also
   on dumps made through `main.py` by `plots/dump_init.py`), `plot_reconstruction.py CKPT SPEC` (figures).
6. **Write ratio (section 8)**: `extract_profile.py --scale_weights ... --write_ratio ...` -> key `write_ratio` in the JSON ->
   `utils.calibrate_joint_statistics` -> `utils._match_write_ratio`; `main.py` prints `[write_ratio] block b attention|mlp: ...`
   and broadcasts qkv, proj, fc2 (weights and biases) of those blocks with the other calibrated tensors.

## 6. What was checked, and what is not reproduced

`verify_recipe_statement.py`, both checkpoints, all PASS: timm rms within 0.2% of $\sigma_0$; $W_s = m\,W^0_s$
elementwise (spread $6\cdot10^{-7}$); $\mathrm{rms}(W_s)\mathrm{rms}(\gamma)/\sigma_0 = e_{b,s}$ and the exactly folded value
within 0.2%; sampled LayerNorm moments within 5% (768 draws); specification numbers equal an independent recomputation
from the checkpoint ($10^{-4}$); each changed tensor equals $sW + a\,\ell r^\top$ with the vectors **as defined above**
(relative residual $4\cdot10^{-8}$); effective scales unchanged ($10^{-7}$); all 15 targets met ($\le 4\cdot10^{-5}$) and
reachable; nothing else changed. Through `main.py`: dumps identical in structure, two-rank fp16 smoke training, tensors
identical on both ranks. On images the calibration did not see (dumps of the 0..7 arms, 256 images): with another draw the entropy moves by at
most 0.12 nats (kdyck) / 0.18 (ksd), the mean pre-activation by 0.003 / 0.01, the token cosine by 0.003 / 0.014; under
the training augmentation by at most 0.21 / 0.32 nats, 0.008 / 0.12 and 0.012 / 0.05.

Function of the initialised model against the prefix, blocks 1..7 (`plots/out/reconstruction_<arm>.png`):

| | kdyck prefix / `ftbanaperg0b7` | ksd prefix / `ftbanakperg0b7` |
|---|---|---|
| attention entropy | 0.43-1.11 / 0.57-1.12 | 0.17-2.81 / 0.27-2.71 |
| mass on the top key | 0.66-0.84 / 0.66-0.81 | 0.31-0.72 / 0.35-0.90 |
| token-specific share of the attention write | 0.000-0.003 / 0.000 | 0.005-0.188 / 0.000-0.001 |
| attention write ratio | 0.059-0.090 / 0.050-0.093 | 0.089-0.578 / 0.104-1.189 |
| mean fc1 pre-activation | -1.96..-2.69 / -1.95..-2.68 | -1.91..-3.19 / -1.89..-3.15 |
| active fc1 units | 0.000-0.003 / 0.000 | **0.065-0.154 / 0.000-0.006** |
| GELU output rms | 0.024-0.063 / 0.013-0.054 | **0.38-0.69 / 0.04-0.09** |
| MLP write ratio | 0.010-0.028 / 0.002-0.008 | 0.086-0.297 / 0.031-0.060 |
| token cosine of the block input | 0.914-0.916 / 0.916-0.920 | **0.66-0.74 / 0.66-0.91** |

kdyck is reproduced except for the MLP write ratio (3-5 times too small) and block 0's own write ratios (attention
1.5 against 3.9, MLP 6.2 against 28.8, by design: the target is the token cosine, since matching the ratio 28.8 with a
pure rank-one write drives the cosine to 0.997 and leaves the sink nothing to resolve). **ksd is reproduced only in the
two calibrated quantities.** Its prefix is a partly open gate (6-15% active units, GELU rms 0.4-0.7) and a partly
token-specific attention (up to 19% of the write); one direction with all units shifted alike closes the gate
completely, and the fully common-mode writes of blocks 1-2 push the tokens to cosine 0.9 where the prefix stays at
0.7. The three rank-one statistics are a good model of kdyck and a coarse one of ksd.

**Not reproduced, and visible only in the random tail: the size of the stream.** The rms of the stream entering block 8
is 53.0 under the kdyck prefix and 5.9 under `ftbanaperg0b7` (second moments only 2.9, random 1.35); ksd 30.9 against
4.3 (6.7, 1.35). LayerNorm hides this from blocks 1..7, whose normalised input matches, but the random blocks 8..11
write with a fixed size, so their write is 0.5-0.6% (attention) and 0.7% (MLP) of the stream under the kdyck prefix and
4.9-5.2% and 6.1-6.4% under the reconstruction (random network 15-16% and 24-27%). This follows from block 0: its target is the token
cosine, reached at an MLP write ratio of 6.2 where the prefix has 28.8. The second-moment arm `ftbanap` (stream 2.8,
80.24) shows that the large stream is not needed for the gain on kdyck, so this is recorded as a difference, not as a
defect; a fourth, ablatable statistic (rms of block 0's output) would be the way to close it.

Not established by any of this: that the functional state *causes* the gain. The accuracy evidence is the arms listed
in `docs/i100_synthesis.md`, mostly single seeds; the reconstruction arms test whether a network built this way trains
like the prefix.

## 7. Blocks 0..7 against the earlier blocks 0..8

Why 0..7: `ftb4i` is the prefix arm with this split for both tasks (kdyck 79.91 / 80.00 / 79.75, ksd 80.05), several
late-lever and combination arms treat 8..11 as the tail (`ftb4jd`, `ftb4h`), and for ksd no 0..8 prefix run exists, so
the earlier ksd arms (0..8) had been compared with a prefix one block shorter.

What changes, computed (2026-09-17):

- **Specification.** `extract_profile.py --blocks 0-7` writes exactly the 0..8 specification without its block-8
  entries: 148 numbers, none of the shared ones differs (kdyck and ksd; also the second-moment bases `ftbanapeb7`,
  `ftbanakpeb7` against `ftbanape`, `ftbanakpe`). The targets of blocks 1..7 cannot depend on block 8, and they do not.
- **Initialisation through `main.py`** (job 29727574, dumps by `plots/dump_init.py`): all 96 tensors of blocks 0..7 are
  bit-identical to those of the 0..8 arm; block 8 is timm's (rms 1.000 $\sigma_0$, gains 1); everything outside blocks
  0..7 (56 tensors) is identical to the init dump of the prefix arm `ftb4i`, so reconstruction and prefix differ at
  initialisation in blocks 0..7 and nowhere else.
- **Checks repeated on the 0..7 arms**: `verify_recipe_statement.py` PASS (15 targets, "only blocks 0-7 touched"),
  `verify_joint_statistics.py` PASS in specification mode and on the dumps against the bases, two-rank fp16 smoke
  training with `[joint statistics] 15 calibrated tensors identical on all 2 ranks: True`. Both verifiers and
  `plot_reconstruction.py` now read the block range from the specification instead of assuming 0..8.
- **Evidence that was measured with 0..8 and is not transferred**: `ftbanap`, `ftbanal`, `ftbanac`, `ftbanape`,
  `ftbanapx` and all ksd recipe arms. `run_train_ftbanapeb7.sh` / `run_train_ftbanakpeb7.sh` are the second-moment-only
  counterparts on 0..7, set up, not launched.

## 8. Input side by scale, output side by write ratio: three sufficiency arms (blocks 0..7)

Set up and verified 2026-09-17, not launched. They ask which part of the second-moment recipe carries "random weights ->
behaves like the prefix", separating the matrices behind a LayerNorm (q, k, v, fc1: their effective scale fixes their
function in any network) from the matrices that write into the un-normalised stream (proj, fc2: the same weights write 0.26
of their own stream and 1.45 of a random prefix's, so only the write ratio carries over).

| arm (kdyck / ksd) | effective scale + LN statistics | write-matched to the prefix | left at timm |
|---|---|---|---|
| (1) `ftbanapeb7i` / `ftbanakpeb7i` | q, k, fc1 | - | v, proj, fc2 |
| (2) `ftbanapeb7w` / `ftbanakpeb7w` | q, k, fc1 | v, proj (sqrt of the attention factor each), fc2: as the late lever | - |
| (3) `ftbanapeb7vw` / `ftbanakpeb7vw` | q, k, v, fc1 | proj (whole attention factor), fc2 | - |
| base `ftbanapeb7` / `ftbanakpeb7` | all six | - | - |

`extract_profile.py CKPT OUT.json --blocks 0-7 --gain_fold exact --exact --scale_weights q,k,fc1[,v] [--write_ratio
v,proj,fc2 | proj,fc2]`. A weight named in `--scale_weights` gets $e_{b,s}$ as in section 4; one named in `--write_ratio`
gets no scale but a target. Targets (16 numbers, `utils.sublayer_write_ratios` on the prefix model, 256 training images,
evaluation transform), for $b = 0..7$:

$$\rho^{att}_b = \mathrm{mean}_{X,\text{tokens}} \frac{\|\mathrm{attn}(\mathrm{norm}_1 x_b)\|}{\|x_b\|},\qquad
\rho^{mlp}_b = \mathrm{mean}_{X,\text{tokens}} \frac{\|\mathrm{mlp}(\mathrm{norm}_2 x'_b)\|}{\|x'_b\|},\quad x'_b = x_b + \mathrm{attn}(\mathrm{norm}_1 x_b).$$

kdyck: $\rho^{att} = 3.48, 0.082, 0.058, 0.090, 0.073, 0.090, 0.089, 0.079$; $\rho^{mlp} = 29.8, 0.028, 0.021, 0.017, 0.011,
0.0098, 0.0100, 0.0110$. ksd: $2.91, 0.28, 0.58, 0.12, 0.10, 0.13, 0.11, 0.087$ and $18.0, 0.14, 0.30, 0.11, 0.086, 0.098,
0.125, 0.122$. Calibration (`utils._match_write_ratio`, inside `calibrate_joint_statistics`, blocks in depth order, attention
before MLP): $W_{proj} \leftarrow f\,W_{proj}$ (or $W_v, W_{proj} \leftarrow \sqrt f\,W$) with $f = \rho^{att}_b / \text{current ratio}$,
then $W_{fc2} \leftarrow f' W_{fc2}$; exact in one step because the write is linear in these tensors while their biases are 0.
No direction is added. A block that carries both an MLP write ratio and the common write of section 4 (both set fc2) is solved
jointly (`utils._install_common_write_at_ratio`: one scalar for the write ratio and one rank-one strength for the token cosine, so
fc2's scale is an outcome of the two targets rather than a preserved input); sink and gate compose with it (order in a block:
sink, attention write, gate, common write, MLP write).

Checked (`verify_recipe_statement.py`, `verify_joint_statistics.py`, all six PASS; through `main.py` job 29728161):
weights without a scale are timm's bit for bit; targets equal an independent recomputation ($3\cdot10^{-5}$); every
write-matched tensor is its previous value times one scalar (residual $3\cdot10^{-8}$), v and proj carry the same factor in
(2); ratios met to $2\cdot10^{-7}$; nothing else changes; dumps: (2), (3) differ from (1) only in those tensors, (1) from the
base only in v, proj, fc2, everything outside blocks 0..7 equals `ftb4i`'s init; two-rank fp16 smoke runs train
(loss 7.07 -> 6.94), `[joint statistics] 48 / 32 calibrated tensors identical on all 2 ranks: True`.

What the figures show (`plots/out/reconstruction_<arm>.png`, now twelve panels incl. stream rms), kdyck, blocks 1..7:

| | prefix | (1) | (2) = (3) | base |
|---|---|---|---|---|
| attention / MLP write ratio | 0.06-0.09 / 0.010-0.028 | 0.07-0.09 / 0.11-0.13 | as prefix | 0.09-0.15 / 0.03-0.05 |
| stream rms entering block 8 | 53.0 | 1.13 | 53.8 | 2.9 |
| write of the random tail (attention / MLP) | 0.5% / 0.7% | 16-18% / 27-32% | 0.5% / 0.7% | 8% / 12% |
| token cosine | 0.915 | 0.39-0.41 | 0.74 | 0.61-0.64 |
| attention entropy, fc1 pre-activation mean | 0.4-1.1, -2..-2.7 | 5.1, 0 | 5.2, 0 | 5.2, 0 |

(2) and (3) show the same measured behaviour at initialisation (all twelve quantities agree to the printed three decimals, both tasks); they differ only in where the attention factor
sits: v 1.9-2.7 and proj 5.2-6.7 in (2), v 0.59-0.76 (the prefix's) and proj 19-27 in (3). Write matching reproduces the
stream size and hence the quiet random tail, which no effective-scale arm does. It does not give the sink or the gate. The
price is large output matrices: random weights are not aligned with the stream's common direction, so proj needs 19-27
times timm's scale (prefix 1.6-2.2) and fc2 4-13 times, 63-66 times in block 0 (prefix 3.1). Adam's relative step on
these tensors is smaller by the same factors, so (2)/(3) against (1) changes loudness AND step size of the output side, as
`ftbrhop` did; the lr-scale files of the late-lever trio are the way to separate the two if an effect shows.
The earlier necessity result is the e-series: the full checkpoint with v, proj, fc2 rescaled to the random init's write
ratios keeps its accuracy (80.21), so the prefix's write profile is not necessary; these arms test sufficiency.

**Outcome (2026-09-19).** Write-matching was launched as the `w` arms of section 9 and lost about a point on kdyck (78.90 and
~78.9 against 79.74 / 80.12 without it) and the whole gain on ksd (~78.0 against 79.65); the two earlier write-matched arms
`ftb4o` (77.27) and `ftbanakw` (76.65) had shown the same. The component stays in the code as an ablation; it is not part of
the recipe (section 11).

## 9. The recipe as four separately controlled quantities (2026-09-17; launched the same evening, results in section 10)

After review the recipe is stated as: **input-side effective scales + LayerNorm statistics**, plus three functional targets read
off the checkpoint prefix, each with its own knob and calibrated per block in this order (later steps do not disturb earlier ones):

1. q/k: rank-one component, target = mean **attention entropy** (`--qk_entropy`; key renamed from `qk_sink`: that all queries pick
   one key is a property of the construction, reported as top-key mass, not of the target);
2. v / proj: scalar(s), target = **attention write ratio** (`--write_ratio`, optional);
3. fc1: rank-one component, target = **fraction of fc1 pre-activations > 0** (`--fc1_gate`, `--fc1_gate_target active_units`, the
   default; `pre_activation_mean` is the target of the section-4 arms and stays available; mean, std and GELU rms are reported);
4. fc2: scalar, target = **MLP write ratio** (optional).

No block-0 common write in this family. Other changes of the day: targets are **means over 5 random contexts** (`--target_seeds 5`:
fresh timm embeddings + fresh image draw each; the spread is stored next to every target; `plots/verify/target_stability.py`:
image draw irrelevant, embeddings move entropy / write ratios by 3-17% and the tiny kdyck active fractions by up to 56%);
`"realise": "exact"` (declared scales met to 1e-7 instead of 2e-3; specs without the key are bit-identical to before); the
rank-one components keep the norm the spec declares (`gain_fold`: folded for exact, raw for product); `--query_key pooled` is now
really flat (1.32); `--common_write` and the block-0 exclusion refer to block 0, not to the first listed block.

Arms (kdyck / ksd), all blocks 0-7: `ftbanaperab7` / `ftbanakperab7` (all six weights by effective scale), `...b7i` (v, proj, fc2
timm), `...b7w` (v, proj, fc2 write-matched), `...b7vw` (v by scale, proj and fc2 write-matched). All 16 verifier runs PASS
(`logs/pera7/`), dumps through `main.py` and two-rank fp16 smoke runs fine (job 29728975), figures in `plots/out/`.

**Init signatures next to finished arms** (`plots/verify/init_signature_table.py`, `plots/out/init_signatures.json`; means over
blocks 1-7, 256 training images):

| arm | top-1 | entropy | active fc1 | pre-act. mean | GELU rms | attn / MLP write | stream rms b8 | tail write % | raw q / k |
|---|---|---|---|---|---|---|---|---|---|
| kdyck prefix 0-7 | 79.89 | 0.69 | 0.0008 | -2.43 | 0.03 | 0.081 / 0.015 | 52.9 | 0.6 / 0.7 | 3.49 / 4.03 |
| `ftbanag`, `ftbanap`, twin, `ftbanape` | 80.70, 80.24, 79.93, 79.44 | 5.2 | 0.50 | 0 | 0.10 | 0.12 / 0.037 | 2.5-2.8 | 9 / 13 | 3.2-3.6 / 3.2-4.3 |
| `ftbana`, `ftbanau` (same second-moment profile, gains 1) | 76.61, 77.35 | 5.2 | 0.50 | 0 | 0.10 | 0.12 / 0.039 | 2.5 | 9 / 14 | 1.32 / 1.32 |
| `ftbanai`; `ftb4o` | 78.11; 77.27 | 5.3 | 0.50 | 0 | 0.33 | 0.12 / 0.037; 0.08 / 0.016 | 2.7; 51.7 | 9 / 13; 0.5 / 0.7 | 2.48; 1.0 |
| `ftbanaperab7` | - | 0.70 | 0.0012 | -0.69 | 0.16 | 0.145 / 0.052 | 2.9 | 8 / 12 | 3.49 / 4.29 |
| `ftbanaperab7i` | - | 0.68 | 0.0011 | -0.75 | 0.16 | 0.110 / 0.159 | 1.2 | 17 / 27 | 3.53 / 4.29 |
| `ftbanaperab7w` = `vw` | - | 0.75 | 0.0013 | -0.70 | 0.16 | 0.077 / 0.015 | 54.3 | 0.5 / 0.7 | 3.44 / 4.29 |
| ksd prefix 0-7 | 80.05 | 1.37 | 0.116 | -2.76 | 0.57 | 0.198 / 0.140 | 30.8 | 0.9 / 1.3 | 4.5 / 4.6 |
| ksd scale-only arms | 77.6-77.9 | 5.15 | 0.50 | 0 | 0.55 | 0.30-0.33 / 0.47-0.51 | 6 | 4 / 5 | 4.1-4.7 |
| gate only; weak sink; sink; sink + gate (`ftbanaksg`) | 78.0-78.3; 78.73; 79.58; 79.92 | 5.2; 1.82; 1.37; 1.40 | 0.114; 0.50; 0.50; 0.114 | | 0.20; 0.55; 0.55; 0.20 | ... / 0.24; 0.49; 0.48; 0.21 | 3.4-6.1 | | 4.7 |
| `ftbanakperab7` | - | 1.44 | 0.126 | -1.05 | 0.23 | 0.390 / 0.188 | 4.6 | 6 / 8 | 4.58 / 4.89 |
| `ftbanakperab7i` | - | 1.42 | 0.123 | -1.07 | 0.22 | 0.107 / 0.240 | 1.2 | 15 / 28 | 4.67 / 4.89 |
| `ftbanakperab7w` = `vw` | - | 1.50 | 0.124 | -1.05 | 0.22 | 0.208 / 0.150 | 34.7 | 0.8 / 1.0 | 4.57 / 4.89 |

Reading. kdyck: the winners span entropy 0.7-5.2, active units 0.001-0.5 and streams 2.5-53; what they share is raw q/k >= 3.2 behind
small gains and a quiet fc1 (GELU rms <= 0.16), and `ftbana` / `ftbanag` differ in nothing but the raw q/k scale. ksd: the scale-only
arms have all of that and stay random; accuracy follows the attention entropy (5.15 -> 1.82 -> 1.37: 77.9 -> 78.7 -> 79.6), the gate
adds the rest. `ftbanakperab7` is at initialisation the weights-only twin of `ftbanaksg` (79.92). The active-unit target is met on
kdyck with a mean of -0.5 to -0.8 (every unit near GELU's minimum, GELU rms 0.16 against the prefix's 0.03 and the recipes' 0.10):
the same fraction from a different distribution; the mean target matches the GELU output there (0.013-0.054), the active-unit
target is the closer one on ksd (0.22 against 0.05 for the mean target; prefix 0.57).

## 10. Results of the blocks-0-7 family (2026-09-18/19, one seed each, last epoch)

Random init 78.08 +- 0.19 (n = 3); kdyck prefix 0-7 (`ftb4i`) 79.89 (n = 3); ksd prefix 0-7 (`ftb4i`, coworker's run) 80.05 (n = 1).
All arms: exact effective scales + 64 LayerNorm statistics + entropy component (blocks 1-7) + fc1 gate (blocks 1-7), targets
averaged over 5 contexts, `"realise": "exact"`, blocks 8-11 and everything outside the blocks bit-identical to the random init
(checked on the init dumps, section 9). "write side" = how v, proj, fc2 are set.

| arm | task | gate target | write side | top-1 | vs random | lens peak (epoch) |
|---|---|---|---|---|---|---|
| `ftbanapermb7i` | kdyck | mean pre-activation | timm (random) | **80.37** | +2.29 | 9.2 (19) |
| `ftbanapermb7` | kdyck | mean pre-activation | checkpoint effective scales | **80.12** | +2.04 | 11.4 (14) |
| `ftbanaperab7` | kdyck | active-unit fraction | checkpoint effective scales | **79.74** | +1.66 | 18.4 (19) |
| `ftbanaperab7i` | kdyck | active-unit fraction | timm (random) | **79.63** | +1.55 | 15.5 (19) |
| `ftbanaperab7w` | kdyck | active-unit fraction | write-matched | 78.90 | +0.82 | 32.6 (39) |
| `ftbanapermb7w` | kdyck | mean pre-activation | write-matched | 78.91 | +0.83 | 31.6 (39) |
| `ftbanakpermb7i` | ksd | mean pre-activation | timm (random) | **79.90** | +1.82 | 4.0 (14) |
| `ftbanakpermb7` | ksd | mean pre-activation | checkpoint effective scales | **79.54** | +1.46 | 15.9 (19) |
| `ftbanakperab7i` | ksd | active-unit fraction | timm (random) | **79.65** | +1.57 | 7.3 (19) |
| `ftbanakperab7` | ksd | active-unit fraction | checkpoint effective scales | 78.64 | +0.56 | 23.9 (29) |
| `ftbanakperab7w` | ksd | active-unit fraction | write-matched | 78.09 | +0.01 | 37.7 (49) |
| `ftbanakbs` (older) | ksd | fc1 bias, persistent (lr x0.02), no sink | scale recipe | 78.27 | +0.19 | |

For reference, earlier arms: kdyck scale-only recipes 79.18-80.70 (lens peaks 25-30), losers `ftbanai` 78.11 / `ftbana` 76.61 /
`ftb4o` 77.27 (lens 34-42); ksd scale-only 77.6-77.9 (lens 39.6), bias sink `ftbanaks` 79.58 (lens 13.8), bias sink + persistent
bias gate `ftbanaksg` 79.92; random's lens peak 47.8 (29), both prefixes 7 (14).

**The block-7 logit lens** (the intermediate read that ordered the finals). `engine.evaluate` applies the model's own final
norm, `fc_norm` and head to the class token of block 7's OUTPUT and reports the top-1 accuracy of that softmax on the validation
set (`engine.py` 462-470, meter `blk_acc_layer7`, logged as `Epoch-wise/acc_layer7` -- the block row, which `plots/verify/
wandb_layerwise.py` keeps because it is written after the attention row of the same key); evaluated at epochs 4, 9, 14, 19 and
then every 10 epochs (29, 39, ..., 299): `model_analyse` runs when $(epoch+1) \bmod 10 = 0$ or $epoch < 20$ inside the 5-epoch
validation cadence (`main.py`), so lens peaks are located on that grid, not on a 5-epoch grid.
Formally, with $x_7$ the output of block 7 and $z = W_{head}\,\mathrm{fc\_norm}(\mathrm{norm}(x_7))_{cls}$,
$\mathrm{lens}_7 = \Pr[\arg\max z = y]$ over the validation images. Its peak over training against the final: 7 -> 79.9-80.1
(prefixes), 11-18 -> 79.7-80.1 (this family without write-matching), 25-30 -> 79.2-80.7 (earlier scale recipes), 32-38 ->
76.7-78.9 (every write-matched arm), 35-48 -> 76.6-78.1 (losers, random). One seed each; the zone around 30 holds both a 79.4 and
a 76.7.

## 11. What matters, by task

| ingredient | kdyck | ksd |
|---|---|---|
| input-side effective scales + LN statistics (raw q, k 3-4x timm behind gains ~0.4; fc1 effective 0.35-0.45) | supported as the carrier by the ablations: `ftbana` 76.61 vs `ftbanag` 80.70 differ only in the raw q/k scale; every winner has it | not sufficient (all scale-only arms at random); necessity under a sink untested (section 12) |
| attention entropy (rank-one sink) | not needed: scale-only recipes reach the prefix; harmless when added (79.74, 80.12) | supported as the carrier by the dose-response: entropy 5.15 / 1.82 / 1.37 -> 77.9 / 78.7 / 79.6 (`ftbanak`, `ftbanaksw`, `ftbanaks`) |
| fc1 gate | no resolved difference between the two targets (80.12 vs 79.74, inside the 0.45 seed resolution) | small addition on top of the sink (bias version 79.58 -> 79.92); nothing alone (78.0-78.3, persistent or not) |
| v, proj, fc2 | checkpoint effective scales (proj 1.6-2.2, fc2 0.75-0.95) or timm: both fine (79.74 / 79.63); write-matched: 78.90 and 78.91 under either gate target, -1.2 against the best arm | timm: 79.65; checkpoint effective scales (proj ~3, fc2 ~2.2, random content then writes 2x the prefix's ratio): 78.64; write-matched: ~78.0 |

Consequences. (i) The write side must not be scaled up to the prefix's write ratios: random content at the prefix's loudness
hurts on both tasks (section 8). (ii) Leaving v, proj, fc2 at the random init has reached the prefix on ksd (`ftbanakperab7i`) and is the only write-side
choice not contradicted on either task (kdyck `ftbanaperab7i` 79.63 at the last epoch, 2026-09-19 13:15: 0.26 below the 0-7 prefix's 79.89, inside the 0.45 seed resolution; lens 15.5); it removes the
ingredient that hurt on ksd; with it the recipe is 24 scales (q, k, fc1) + 64 LayerNorm statistics + 7 entropy targets + 7 gate targets =
102 numbers, no checkpoint at initialisation. (iii) The current ablations are consistent with the two tasks being carried by different ingredients -- kdyck by the input-side
scales (slow steps, quiet fc1), ksd by the sink; one recipe containing both now reaches the prefix on both tasks with the write side
at timm: `ftbanaperab7i` 79.63 on kdyck (prefix 79.89, n = 3) and `ftbanakperab7i` 79.65 on ksd (prefix 80.05, n = 1), both gaps
inside the seed resolution; all of this is one seed per arm.
(iv) The gate target is decided (2026-09-20): the **mean pre-activation**. The mean target closes the MLPs much further than the
prefix (on ksd GELU rms 0.05 against the prefix's 0.57 and the active-unit gate's 0.22), and that is not a defect: all four
mean-gate arms finish above their active-unit twins, one seed each, last epoch --

| task, write side | mean gate | active-unit gate | difference |
|---|---|---|---|
| kdyck, timm | `ftbanapermb7i` **80.37** | `ftbanaperab7i` 79.63 | +0.74 |
| kdyck, checkpoint effective scales | `ftbanapermb7` 80.12 | `ftbanaperab7` 79.74 | +0.38 |
| ksd, timm | `ftbanakpermb7i` **79.90** | `ftbanakperab7i` 79.65 | +0.25 |
| ksd, checkpoint effective scales | `ftbanakpermb7` 79.54 | `ftbanakperab7` 78.64 | +0.90 |

Two of the four differences exceed the 0.45 seed resolution on their own and all four have the same sign (mean +0.57); the
lens peak moves the same way in every pair (9.2 / 11.4 / 4.0 / 15.9 against 15.5 / 18.4 / 7.3 / 23.9), so the more strongly
closed gate suppresses the block-7 transient further. The mean gate also repairs the ksd arm with checkpoint write scales, which
the active-unit gate had left near random. **Committed design: q, k, fc1 effective scales + LayerNorm statistics + entropy
targets + mean-pre-activation gate, v / proj / fc2 at timm** -- `ftbanapermb7i` 80.37 on kdyck (prefix 79.89, n = 3) and
`ftbanakpermb7i` 79.90 on ksd (prefix 80.05, n = 1): at or above the prefix on both tasks with one recipe, 102 numbers, no
checkpoint at initialisation. Still one seed per arm; seeds of the two committed arms are the next launch.

## 12. Next: three factors, two contrasts (planned, not set up)

**Three factors, not two.** The committed design changes (i) the *structure*: sink and gate, the rank-one parts common to all
tokens; (ii) the *scale as size*: how large the random, token-specific part of q, k, fc1 is, together with the LayerNorm
statistics; and, as a side effect of (ii), (iii) the *scale as slow steps*: a raw matrix $m$ times larger moves $m$ times more
slowly under Adam relative to its own size. A single 2x2 cannot separate all three. Two contrasts do, and they share cells.

**Contrast 1: scale versus structure, at natural steps.** Three of the four cells exist (last epoch, one seed unless noted):

| | structure off | structure on |
|---|---|---|
| scale on | scale-only recipes (blocks 0-8): kdyck 79.18-80.70, ksd 77.86 / 77.68 (`ftbanak`, `ftbanakx`) | committed arms (blocks 0-7): kdyck 79.74 / 79.63 / 80.12, ksd 79.65 |
| scale off | random 78.08 +- 0.19 (n = 3) | **missing: structure only** (sink + gate on a timm init) |

Read so far: on kdyck scale alone reaches the prefix and structure adds nothing resolvable; on ksd scale alone is random level
and structure recovers it (within scale on, the sink is the carrier: gate only 78.0-78.3, sink 79.58, sink + gate 79.92,
section 10). The missing cell says whether structure alone suffices, which matters most on ksd. Caveat: the scale-only column
was measured on blocks 0-8; its blocks-0-7 versions (`ftbanapeb7`, `ftbanakpeb7`) are prepared and were never run.

**Contrast 2: size versus slow steps.** This question exists only where the scale is on, so it is asked inside the top-right
cell: the committed initialisation bit for bit, once with natural steps and once with the lr multipliers that restore the random
init's relative steps (the step-size 2x2 below; its top row is the clean test, because nothing but the optimiser's scaling moves).

**Minimal plan: two new arms per task**, (a) structure only, (b) committed init with normal steps. Optional: the two prepared
scale-only arms on blocks 0-7, to put contrast 1 on one block range, and the scale-off / slow-steps corner of the 2x2 below.

**The structure-only cell, measured** (`plots/verify/structure_only_reachability.py`, job 29740448: timm seed 0, the committed
arms' own targets, 256 training images, `utils.calibrate_joint_statistics`; "size kept" = `renormalize: true`, the convention of
every launched arm, "size free" = `renormalize: false`). Blocks 1-7, number of targets met:

| targets of | gate target | size kept: entropy / gate | size free: entropy / gate | size free: raw rms of q, k; fc1 against timm |
|---|---|---|---|---|
| kdyck (`ftbanaperab7i`) | active-unit fraction | 3 / 0 of 7 | 7 / **0** of 7 (at the strength cap 64: 0.002-0.125 against 0.0003-0.004) | 1.29-3.02; 2.31 |
| ksd (`ftbanakperab7i`) | active-unit fraction | 5 / 6 of 7 | 7 / 5 of 7 (blocks 1, 4 at the cap) | 1.06-3.04; 1.00-2.31 |
| kdyck (`ftbanapermb7i`) | mean pre-activation | 2 / 7 of 7 | **7 / 7** | 1.25-3.02; 1.09-1.11 |
| ksd (`ftbanakpermb7i`) | mean pre-activation | 5 / 7 of 7 | **7 / 7** | 1.07-3.60; 1.07-1.15 |

Consequences. (1) With the size kept the cell does not exist: the sink cannot reach the entropy of blocks 1-2 (1.83 / 1.35-1.56
nats against 0.25-0.61) on either task, and since 2026-09-19 an unmet target aborts the run (section 15). (2) With the size free
the sink always reaches its target; the price is a raw q/k matrix up to 3.0-3.6 times timm in blocks 1-2, all of it in the
rank-one common part: the random, token-specific part stays exactly timm's, and because Adam's step is absolute per coordinate
that part also keeps timm's relative steps, so "scale off" holds in both senses for it. (3) The rank-one gate reaches the
*mean pre-activation* target everywhere (fc1 grows by 7-15%), but not the *active-unit* target: on a timm init the pre-activation
std is about 0.5 (0.2 behind the profile), so closing all but $3\cdot10^{-4}$ of the units needs a shift the component cannot deliver
within the cap, on kdyck in no block. The structure-only cell is therefore available as designed only under the mean-gate
target, which is the one decided on 2026-09-20 (section 11 (iv)), so the cell is available as designed; under the
active-unit target it would need a calibrated fc1-bias gate (the existing `"fc1_bias"` key is a fixed constant per block, not
calibrated to a target). (4) The same entanglement is present, mildly, in the committed arms: their components keep the matrix
size, so installing the structure shrinks the token-specific part to $s\,e$ (block 1: $s_q = 0.58$, section 3); structure "eats
into" scale there, and the structure-only arm is the one cell where the two are independent by construction. Report the
resulting raw sizes next to its accuracy.

**The step-size 2x2 (contrast 2 in full).** The scales do two things at once: they set the output size of the random input-side matrices in the forward pass (the
effective-scale profile; exact for a random matrix, section 3) and, through the large raw matrices
behind small gains, they slow Adam's steps on the input side. The earlier 2x2 on the scale-only recipe (`ftbana` = the profile's
weight scales with gains 1 and normal steps, 76.61; `ftbanal` = the same forward pass with per-tensor lr scales reproducing the
slow steps, 79.78 -- neither carries the LayerNorm anisotropy; `run_train_ftbanal.sh` loads `profile_ftbanal.json` plus
`lrscale_ftbanal.json`) is not transferable: it sat on the blocks-0-8 ramp recipe
with the pooled q/k value and the fc2 correction, without sink and gate, and its lr scales reproduced `ftbanap`'s multipliers.
The same 2x2 on the committed design, sink and gate in every cell:

| | slow steps | normal steps |
|---|---|---|
| profile on | the arms of section 10 | the section-10 initialisation tensor for tensor, only the per-tensor lr multipliers changed ($\lambda = m$ on the scaled tensors), so nothing but the optimiser's scaling moves |
| profile off | weights at timm scale, lr scales reproducing the section-10 arm's relative steps (the `ftbanal` construction, applied here to timm-scale weights) | **structure only**: sink + gate on a plain random init |

Mechanism (`optim_factory.build_lr_scaled_param_groups`, `engine.py` 99 / 104): a JSON {tensor name: $\lambda$} puts each named
tensor into a parameter group with $\eta_g(t) = \eta(t)\,\lambda$ and, for decayed tensors, $\mathrm{wd}_g(t) = \mathrm{wd}(t)/\lambda$.
Adam's step is $-\eta_g\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$, of magnitude $\approx \eta_g$ per coordinate once the moment estimates
have settled, so the *nominal* relative step of a tensor with $\mathrm{rms}(W) = m\,\sigma_0$ is $\propto \eta\lambda/m$ (a different
initialisation also changes the gradients and hence $\hat m_t, \hat v_t$; the construction matches the optimiser's explicit
scaling, not the realised trajectory): choosing $\lambda = 1/m$ (profile off, slow steps) gives a timm-scale tensor the relative steps of the
scaled one, and $\lambda = m$ (profile on, normal steps) gives a scaled tensor the random init's relative steps while its
initialisation stays exactly that of section 10 (a cell with gains set to 1 instead, as `ftbana` was built, would also change the
LayerNorm statistics and, through $W\beta$, the forward map); this is the `ftbrhopl` construction of section 13; the per-step
decoupled decay factor $1 - \eta_g\,\mathrm{wd}_g = 1 - \eta\,\mathrm{wd}$ is unchanged in both cases. $m$ is read per tensor
off the arm's init dump as $\mathrm{rms}(W)/\mathrm{rms}(W_{timm})$; the fused `attn.qkv.weight` is one tensor and one group,
so with per-tensor scales alone q, k and v would share one $\lambda$; the per-row learning-rate mask of section 13 removes this
(row ranges in the same JSON), so each of q, k, v can get its own $\lambda$ read off its own rows.
In the "profile off" cells q and k keep the random init's effective scale, and the entropy target is then not reachable within
the sink's norm budget (measured above: 2-5 of 7 blocks), so those cells use `renormalize: false`. The gate target of every cell is the mean pre-activation (decided 2026-09-20,
section 11 (iv)), the same on both tasks. After the 2x2: two more seeds of the committed arm per task and two more
seeds of the ksd prefix (n = 1 so far).

## 13. The late-lever 2x2 (step-size trio), checked 2026-09-19

The same loud-versus-slow question was asked of the late lever on 2026-09-14/15, with the same mechanism as section 12, and the
files and flags were re-read for this note. Base arm `ftbrhop`: timm random everywhere, proj and fc2 of blocks 9-11 multiplied by
$m = 9.06 / 21.16 / 57.91$ (proj) and $9.42 / 27.47 / 81.71$ (fc2) = `ftbrho`'s $v \cdot proj$ and fc2 products, read off `ftbrho`'s init
dump (`profile_ftbrhop.json`, key `extra`; write ratio 1.4 in blocks 9-11 at init). Only proj and fc2 are touched: v sits inside
the fused `attn.qkv.weight` (`models.vision_transformer.Attention`, the repo's copy of timm's, holds one `nn.Linear(d, 3d)` whose
output is reshaped into q, k, v), and a learning-rate scale is a property of a
parameter group, i.e. of a whole tensor, so v could not get its own scale without also scaling q and k -- which is why the trio
was built on proj/fc2 alone rather than on `ftbrho`'s v/proj split (every other late-lever arm scales v and proj together, each by
the square root of the attention factor). With only proj and fc2 scaled, the per-tensor scale is exact per tensor here.
*Removing the limitation (done 2026-09-19, `row_lr_mask.py`).* Two ways were open: (a) split the fused tensor into three
parameters (exact, but every checkpoint, dump and analysis reads `attn.qkv.weight`); (b) keep the tensor and let the optimiser
scale rows. (b) is implemented as an exact post-step correction rather than a new optimiser: AdamW's update is linear in the
learning rate per coordinate, $\Delta p = -\eta\,\mathrm{wd}\,p - \eta\,\hat m/(\sqrt{\hat v} + \epsilon)$, so after the stock
step has run with the plain $\eta$ a hook (`optimizer.register_step_post_hook`) adds $-\eta(\lambda_{row} - 1)\,\hat m/(\sqrt{\hat v}
+ \epsilon)$ to the masked rows from the moment estimates the optimiser has just stored. The result equals a parameter group
with $\eta\lambda$ and $\mathrm{wd}/\lambda$ row by row: the decay $\eta\,\mathrm{wd}\,p$ is never touched, the moments are the
stock ones, and a step skipped by the AMP scaler runs no hook. Specification: in the lr-scale JSON a tensor may map to
`{"rows": [[start, end, lambda], ...]}`; `optim_factory.create_optimizer` splits such entries off (`split_lr_scale_spec`),
keeps the tensor in an ordinary group and installs the hook (`install_row_lr_masks`; AdamW without amsgrad only).
The mask is opt-in and off by default: `--lr_scale_json` defaults to empty, a JSON without a `rows` entry installs no hook at all
(every launched lr-scale file is scalar-only), and an installed mask announces itself with a `[row-lr]` line per tensor at
optimizer creation. It is meant for the step-size cells of the section-12 2x2 only, never for a default run.
Tests (`plots/verify/test_row_lr_mask.py`): exact equivalence with the same network built from three separate tensors in
groups with $\eta\lambda$, $\mathrm{wd}/\lambda$ ($6\cdot10^{-16}$ over 60 steps, float64, weight decay on, a learning rate
that changes every step, both `foreach` settings, and $2\cdot10^{-2}$ without the mask); $\lambda = 1$ bit-identical to no mask;
zero-gradient steps decay by exactly $1 - \eta\,\mathrm{wd}$ regardless of $\lambda$; a scaler-skipped step leaves the
parameters unchanged; `load_state_dict` mid-run continues the reference trajectory exactly; on ViT-B through
`create_optimizer` the masked rows' first update equals decay $+\ \lambda\cdot$(adaptive step of the unmasked run) to
$9\cdot10^{-6}$ relative while every other tensor's update is bit-identical and the moment estimates are untouched; malformed
specifications raise. On the GPU (2026-09-19, test partition): the fused AdamW kernel agrees with the reference to
$6\cdot10^{-16}$; a 2-rank fp16-AMP smoke run of `ftbanaperab7i` with the test mask (q rows 0.29, k rows 0.23, v rows 1.0 in block 1;
q/k rows 0.25 in block 2; fc1 of block 2 at a scalar 1.1) installs the mask on both ranks, trains epoch 0, saves, resumes with
optimizer and scheduler, re-installs the mask, trains epoch 1, and leaves the masked tensors finite with the optimizer state
intact (`results/init_dumps/test_rowlr2.sbatch`, job 29737359). Scaling the *gradient* of the v rows instead would not work: Adam normalises each coordinate's step by
its own second moment, so a constant gradient scale cancels (up to $\epsilon$).

| | loud write (proj, fc2 at $m$) | random write (timm scale) |
|---|---|---|
| slow steps ($\eta/m$ relative) | `ftbrhop` 79.93: weights at $m$, no lr scale | `ftbrhosl` 78.31: timm weights, $\lambda = 1/m$ (`lrscale_ftbrhosl.json`) |
| normal steps ($\eta$ relative) | `ftbrhopl` 80.13: weights at $m$, $\lambda = m$ (`lrscale_ftbrhopl.json`), **bf16** | random 78.08 +- 0.19 (n = 3) |

Checked: `lrscale_ftbrhopl.json` equals the six multipliers of `profile_ftbrhop.json` and `lrscale_ftbrhosl.json` their reciprocals
(max relative deviation $1.3\cdot10^{-4}$, rounding); `ftbrhosl` runs `--init_method default` (timm init) with the lr file, `ftbrhopl`
the `ftbrhop` profile with the lr file; `optim_factory.build_lr_scaled_param_groups` gives the six tensors $\eta\lambda$ and
$\mathrm{wd}/\lambda$, so in every cell the decoupled decay factor per step is $1 - \eta\,\mathrm{wd}$ and the nominal relative step
is $\eta/(m\sigma_0)$ in the top row (slow steps) and $\eta/\sigma_0$ in the bottom row (normal steps), as intended (the section-12 qualifier applies:
nominal, not the realised trajectory). Reading: the late lever is the loud write -- 80.13 and 79.93 against 78.31 and 78.08;
slowing the steps of random-scale proj and fc2 does nothing (+0.23, inside the 0.45 seed resolution), and giving the loud
weights normal steps costs nothing (+0.20, inside it as well). Caveats that stay: one seed per cell except random; `ftbrhopl`
trained in bf16 after its fp16 run overflowed at epoch 44 (docs `i100_late_block_scaling.md`; the k-bias null direction, fixed for
resumes by `resume_zero_kbias.sh`), the other three cells in fp16, and no bf16 random baseline exists, so the 0.20 between
`ftbrhopl` and `ftbrhop` carries a precision difference as well as a seed. Nothing in this 2x2 needs re-running; the
"profile on / normal steps" cell planned in section 12 is this `ftbrhopl` construction applied to the early lever.

*v-inclusive cells (launched 2026-09-19 ~12:35, shared partition).* With the row mask, v can carry its own multiplier, so the
trio can be built on `ftbrho`'s own init (v and proj each $\times\sqrt{a_b}$, fc2 $\times f_b$) instead of the proj/fc2-only base:

| | loud write (v, proj $\times \sqrt{a_b}$, fc2 $\times f_b$) | random write (timm scale) |
|---|---|---|
| slow steps | `ftbrho` 79.69 +- 0.30 (n = 3, fp16; the arm itself) | **`ftbrhoslv`** 29737718 **78.18** (fp16 as `ftbrhosl`): timm init, $\lambda = 1/\sqrt{a_b}$ on the v rows (row mask) and on proj, $1/f_b$ on fc2 |
| normal steps | **`ftbrhoplv`** 29737716 **80.15** (bf16 as `ftbrhopl`): `ftbrho` init via `profile_ftbrhoplv.json` (`extra`: v = proj 3.0071 / 4.6048 / 7.6129, fc2 9.4186 / 27.4658 / 81.7061), $\lambda$ = the same multipliers on the v rows, proj, fc2 | random 78.08 +- 0.19 (n = 3, fp16) |

The multipliers are `ftbrho_s0.pth` / timm per tensor (fit residual $3\cdot10^{-8}$; v and proj carry the same factor, the square
root of the attention factor). Verified (`plots/verify/verify_late_trio_v.py`, job 29737705): main.py's own init of `ftbrhoplv`
(plots/dump_init.py) equals the reconstruction bit for bit and `ftbrho`'s dump to $6.5\cdot10^{-6}$ relative; q/k rows, fc1,
biases, LayerNorms and every tensor outside blocks 9-11 are timm's; `ftbrhoslv`'s init is timm seed 0 bit for bit; both lr files
name exactly the nine tensors (v rows [1536, 2304) via the row mask, proj and fc2 as scalar groups), with $\lambda$ equal to the
multipliers / their reciprocals ($< 1.4\cdot10^{-6}$); write ratios at init on 64 training images equal `ftbrho`'s per block
(attention 1.345 / 1.385 / 1.406, MLP 1.450 / 1.426 / 1.417); the 2-rank one-epoch smoke of each arm (job 29737710) prints the
three `[row-lr]` and six `[lr-scale]` lines, trains, and writes a checkpoint whose recorded lr specification equals the launch
file. Precision is paired with the comparator of each cell (`ftbrhoplv` vs `ftbrhopl` both bf16, `ftbrhoslv` vs `ftbrhosl` both
fp16), so the v-inclusion question is asked within one precision per cell; the fp16-vs-bf16 caveat between the rows stays.

**Finals (2026-09-20, last epoch, n = 1 per cell): `ftbrhoplv` 80.15, `ftbrhoslv` 78.18.** Against their v-exclusive twins that is
+0.02 (`ftbrhopl` 80.13) and -0.13 (`ftbrhosl` 78.31), both far inside the 0.45 seed resolution: carrying v in the multipliers
and in the row mask changes nothing. The reading of the late-lever 2x2 stands on `ftbrho`'s own init as well: the loud write
is the lever (80.15 and 79.69 against 78.18 and 78.08), the step size is not (slow steps on a random-scale write: +0.10 over
random; normal steps on the loud write: +0.46 over `ftbrho`, at the edge of the seed resolution and confounded with bf16).
`ftbrhoslv` skipped updates to fp16 overflows in 32 of its epochs from epoch 40 on (gradient-norm windows with a non-finite
entry), without a non-finite loss; both runs completed in one job, their continuations were cancelled as no-ops.


*Verifiers (2026-09-19).* `verify_joint_statistics.py` and `verify_recipe_statement.py` now run the model on the GPU when one is
present (cudnn deterministic, as `main.py`; the calibration in a training run happens on rank 0's GPU, so this is also the more
faithful path) and cap their threads at the job's CPU allotment. Before, they were CPU-only with 32 hard-coded threads while the
verification jobs ran two or three of them at once in a 32-CPU allocation: 40 min per arm. Measured on `ftbanapermb7i`
(256 images): joint verifier 107 s on the GPU, 442 s on 16 uncontended CPU threads; statement verifier 85 s on the GPU. The
verdicts and every checked number agree with the CPU logs to the printed digits (rank-one residuals are smaller on the GPU;
the augmentation column of the behaviour table is RNG-dependent and not checked).

*Measured-profile cell (set up and verified 2026-09-22, brief `docs/late_lever_measured_profile_brief.md`; LAUNCHED 2026-09-23 06:40 with
Simon's go, one seed each: `ftb3bpl` results id 29765171, `ftb3bb` 29765222, `ftb3bsl` 29765177; shared-H200 primaries with group-H200
hand-overs and group-L40S fallbacks, run log).* Every cell above was built at the flat target 1.4; proc's own late blocks write at 1.380 / 2.098 / 0.777 (attention) and
4.757 / 4.459 / 0.526 (MLP) in blocks 9 / 10 / 11 (`target_res_stats.json` of `ftb3b`, results id 29388202), and `ftb3b` (random init,
blocks 9-11 scaled to exactly these values by `upscale_random_match_delta_norms`; n = 3, fp16) is 80.00 +- 0.14 against `ftbrho`'s
79.69 +- 0.30 at the flat 1.4. The step-size question is therefore asked again at the measured profile, with the `ftbrhoplv`
construction applied to `ftb3b`:

| | loud write (`ftb3b` init: v, proj $\times \sqrt{a_b}$, fc2 $\times f_b$ at the measured ratios) | random write (timm scale) |
|---|---|---|
| slow steps | `ftb3b` 80.00 +- 0.14 (n = 3, fp16; the arm itself); `ftb3bb` = the same init on the current script in bf16, one seed, the precision-matched reference (29765222) | `ftb3bsl` (optional, 29765177): timm init, $\lambda = 1/\sqrt{a_b}$ on the v rows (row mask) and proj, $1/f_b$ on fc2, fp16 |
| normal steps | **`ftb3bpl`** (required, 29765171): the `ftb3b` init replayed from `profile_ftb3bpl.json` (`extra`), $\lambda$ = the same multipliers on the v rows, proj, fc2 (`lrscale_ftb3bpl.json`), bf16 | random 78.08 +- 0.19 (n = 3, fp16) |

Multipliers, read off `results/init_dumps/ftb3b_s0.pth` (the launched init reproduced with `ftb3b`'s flags on one L40S GPU, 5000
calibration images from `/data/datasets/ILSVRC2012`, seed 0; job 29760435, `plots/verify/fit_late_multipliers.py`): v = proj
2.988053 / 9.514924 / 17.746902 (the square roots of the attention factors 8.93 / 90.53 / 314.95), fc2 32.141033 / 331.073303 /
223.230820 for blocks 9 / 10 / 11 (fit residual $2.5 \cdot 10^{-8}$; q/k rows, fc1, biases, LayerNorms and everything outside blocks
9-11 are timm seed 0 bit for bit). Against `ftbrho`'s 3.01 / 4.60 / 7.61 and 9.42 / 27.47 / 81.71 the measured profile is 3.4x /
12x / 2.7x louder in fc2 and 1.0x / 2.1x / 2.3x in v and proj -- block 11 included, although its targets are below 1.4: each
block's ratio is taken against a stream that the louder writes of blocks 9-10 have already inflated (peak entering block 11: 626
against 58 for `ftbrhoplv`, probe below). The reproduced dump's printed targets agree with the launched run's table to 0.2-1.3 %
(block-9 MLP 4.817 vs 4.757; the training augmentation of the 5000 calibration images depends on the loader-worker count), and no
init-time record of the launched run survives (console log not on wandb, only epochs 298/299 kept), so the reproduction rests on the
identical flags and the target agreement, not on a tensor comparison as for `ftbrho`.

Verified (`plots/verify/verify_late_trio_3b.py`, job 29760562, PASS): A the reconstruction on timm seed 0 changes only the v rows,
proj, fc2 of blocks 9-11, each by one scalar (spread $6 \cdot 10^{-8}$); B it equals the `ftb3b` dump to $1.1 \cdot 10^{-7}$
relative; C both lr files name exactly the nine tensors, $\lambda$ = the multipliers (0 deviation) and their reciprocals
($4 \cdot 10^{-7}$); D main.py's own init of `ftb3bpl` (`plots/dump_init.py` with the run script's flags) equals the reconstruction
bit for bit and `ftb3bsl`'s is timm seed 0 bit for bit; E write ratios at init on 64 training images equal `ftb3b`'s per block
(attention 1.328 / 2.046 / 0.770, MLP 4.989 / 4.558 / 0.524; within 4.9 % of the 5000-image targets). Precision
(`plots/verify/fp16_headroom_probe.py`, same job): the `ftb3bpl` init peaks at 626 in the block-11 stream (1.0 % of the fp16
limit; block-10 fc2 output 600) against 134 (0.2 %) for `ftbrhoplv`; the fp16 forward is finite at init, but `ftbrhopl` overflowed
fp16 at epoch 44 from 5x smaller activations, so the cell runs in bf16 and `ftb3bb` supplies the bf16 reference. The 2-rank smokes
(`results/init_dumps/resume_3b.sbatch`, job 29763105, dlc2gpu05, `NCCL_P2P_DISABLE=1`; two epochs on a compressed schedule, since
`utils.cosine_scheduler` refuses a warm-up longer than the run: 2 warm-up epochs to the real peak lr, then cosine) pass for both arms:
three `[row-lr]` and six `[lr-scale]` lines, finite loss through the ramp to lr 2e-3 with the x331 group (`ftb3bpl` 7.20 -> 6.94,
grad norm 6.8 -> 0.9), both checkpoints carry the lr specification (`check_ckpt_lr_spec.py`: equal to the launch file), resuming with
the same file continues at epoch 2, resuming with a file whose block-9 fc2 entry is changed by 1 % is refused
(`RuntimeError: lr specification mismatch on resume`), and `verify_resume_checkpoint.py` accepts the directory. `ftb3bb`'s init through main.py with its run script's flags (SSD data copy, bf16; job 29765170) is
bit-identical to `ftb3b_s0.pth` in the nine scaled tensors and timm seed 0 elsewhere.

Caveats to carry into the read-out (brief): precision (bf16 against fp16 unless `ftb3bb` finishes; `ftbrhopl` vs `ftbrhop` was +0.20
across precisions); compounding (fc2 factors up to 331 sit near the divergence regime of `ftb4j`, whose factors up to 173 diverged;
`ftb3b` trained 3/3 with slow steps, `ftb3bpl` gives those tensors ~100x larger relative steps -- a warm-up divergence is a result,
not a reason to lower the target); nominal step match, not the realised trajectory. Decision rule (n = 1 against n = 3, seed
resolution 0.45): `ftb3bpl` within 0.45 of `ftb3b` -> the loud write is the lever at proc's profile as well; 0.45 or more below -> the
step size matters at the measured level and the sentence above is restricted to the flat target. Result: to be entered here (last
epoch from log.txt, `plots/verify/arm_truth.py`).

## 14. Which blocks the late lever scales: 9-11, kept (decided 2026-09-19)

With the early lever moved from blocks 0-8 to 0-7 (section 7) the question arose whether the late lever should move from 9-11
to 8-11 so that no block is left untouched. It stays at 9-11; block 8 stays timm random. Evidence (last epoch, `arm_truth.json`):

| arm | early blocks | late blocks | late target | n | top-1 |
|---|---|---|---|---|---|
| `ftbrho` | random 0-8 | 9-11 | write ratio 1.4 | 3 | 79.69 +- 0.30 |
| `ftbcomp11` | proc 0-10 | 11 | 1.4 | 3 | 80.63 +- 0.18 |
| `ftbcomp25` | proc 0-3, random 4-8 | 9-11 | 0.25 | 3 | 80.16 +- 0.12 |
| `ftbcomp1` | proc 0, random 1-8 | 9-11 | 0.25 | 3 | 79.98 +- 0.23 |
| `ftb4jd` | proc 0-7 | 8-11 | proc's own ratios x 0.5 | 3 | 80.11 +- 0.04 |
| `ftb4j` | proc 0-7 | 8-11 | proc's own ratios | 3 | diverged three times identically (2.47 at epoch 54) |
| `ftb4jc` | proc 0-7 | 8-11 | 1.4 | 1 | cancelled at init: block-11 fc2 factor 8738 |
| kdyck prefix 0-7, random 8-11 | proc 0-7 | none | | 3 | 79.89 |

*Stability is the argument.* A write-ratio target is measured against the stream each block actually receives, and a scaled block
inflates the stream the next scaled block is measured against, so the factors compound with depth. Three blocks at 1.4 end at
fc2 factors 9.4 / 27.6 / 81.9 (`ftbrho`) and train cleanly; four blocks in a cascade reach 4.9 / 23.4 / 94.9 / 173 at proc's own
profile (`ftb4j`, diverged) and 8738 at the constant 1.4 (`ftb4jc`), and only the halved profile (2.4 / 11.4 / 44.6 / 59.9,
`ftb4jd`) trained. `i100_late_block_scaling.md` records the same conclusion: raw factor magnitude does not predict divergence,
compounding across blocks does. Extending the cascade downward by one block is the direction that destabilises it.
*And there is nothing to gain.* `ftb4jd` is +0.22 over the 0-7 prefix, inside the 0.45 seed resolution; block 11 alone is the
strongest late-lever number; and untouched random blocks between the prefix and the lever are harmless (`ftbcomp25`,
`ftbcomp1`: five and eight of them). *Comparability.* Every late-lever result this document relies on -- the `ftbrho` seeds, the
2x2 of section 13, the combined arm `ftbanac` (analytic 0-8 + key `extra` on 9-11) -- scales 9-11; the 8-11 factors would have to
be re-derived and the section-13 cells re-run. The committed combined design is therefore: early lever on 0-7 (section 9),
block 8 bit-identical to the random init, late lever on 9-11 with the `ftbrhop` multipliers (section 13). If the boundary is
ever to be closed, the cheap test is block 8 added alone to the committed design, not a four-block cascade.

## 15. External review of the code (2026-09-19): what it touched, what was fixed

A reviewer went through main.py, engine.py, utils.py, optim_factory.py, custom_lr.py, row_lr_mask.py and extract_profile.py and
ran synthetic CPU tests. Every finding was checked against the code and against the launched runs before anything was changed.

**Does any finding change a reported number? No.** Every last-epoch accuracy in this document and in
`i100_late_block_scaling.md` is `test_acc1` of the training loop's own validation pass on the in-memory model, read from the
run's `log.txt` (`plots/verify/arm_truth.py`) or from the same numbers on wandb; the block-7 lens peaks come from
`model_analyse` on the in-memory model during training and lie at epochs 7-48. The findings and their reach:

| finding | reach into completed runs | fix (all 2026-09-19) |
|---|---|---|
| post-training analysis (`attention_analyse_final`) reloaded the final checkpoint through `ft_load_model`, which drops head, class token, position embedding and patch projection when `--initialize_as_pr` (default true): trained blocks under a random head | real. The accuracy JSON's top-level `acc1/acc5/ece` (last writer) and its epoch-299 per-layer `stats` entries read ~0.1% for every finished run (e.g. `accuracy_IMNET_BASE_29729541_s0.json`: 0.104%), and the epoch-299 point of the prefix-less `Layer-wise` wandb rows is that broken model. Nothing reported uses either; the "head probe ~0 at the end" seen earlier was this artefact. | `ft_load_model(..., keep_all=True)` at all four "Loading fine-tuned model" sites; `test_calibration_guards.py` G3 |
| calibration could miss a target and train anyway: `_bisect` returned the bracket end with `reachable False`, the installer still wrote it, `main.py` only printed the flag; and `reachable True` only meant a one-sided threshold crossing | none: no launched log contains `reachable False`, and both verifiers check the achieved statistics (entropy within 0.02, active units within 3%, write ratios within 1e-3) and PASS for every arm of sections 10-11 | `_bisect` checks the bracket at strength 0 (a target already passed returns (0, False)); every installer reports `matched` from the achieved statistic with that tolerance; `main.py` broadcasts the count of unmet targets and raises on every rank; G1-G2 |
| cross-rank check compared sums of absolute values | none (the broadcast is the mechanism; the check was weak, not wrong) | sha256 of every calibrated tensor's bytes, gathered and compared |
| `--gain_fold product --realise multiplier` wrote no `gain_fold`, so joint calibration defaulted to exact | none: every launched spec is exact / exact | `gain_fold` recorded whatever the realisation |
| `shuffle_weights` accepted `attn.qk.bias` / `attn.v.bias` and did nothing | none: no run script uses them | resolved against `attn.qkv.bias` with bias spans |
| full-precision branch: `parameter_norm` unbound, no clipping | none: every run uses AMP | initialised on both branches; clips when `--clip_grad` is given, as the AMP path |
| lr-scale JSON: unknown names ignored, NaN / inf / negative accepted, fractional row bounds, grouping by 6 decimals, `--lr_match_ckpt` + `--lr_scale_json` silently combined | none: the four launched files (`lrscale_ftbanal/ftbrhopl/ftbrhosl/slowgate.json`) name only trainable parameters and hold finite positive values | names validated, values validated, exact grouping key, the combination raises; `test_row_lr_mask.py` T7 |
| weight-decay schedule used the current coefficient as the eligibility flag (a schedule reaching 0 never comes back) | none: `--weight_decay_end` is unset in every run, so the schedule is constant 0.05 | persistent `decay` flag per group |
| row mask: native fused AdamW runs `optimizer.step()` on an overflow and skips inside the kernel, so the post-step hook would add a correction from stale moments | none: the training path builds a non-fused AdamW, and no launched run carries a mask | the hook corrects only when the tensor's step counter advanced during the step (pre/post hooks); T9 on the GPU |
| row mask: the mask was not part of the optimizer state, so a resume with a changed JSON went unnoticed | none (no run with a mask) | `lr_scale_spec` (scalar scales + row masks) written into every checkpoint, refused on mismatch at resume; T10 and run C of the GPU smoke |
| ECE accumulated over every lens layer | none: no ECE is reported | classifier only |
| lr-step diagnostic subtracted the assumed decay from a zero change on a skipped step | none: `LR Scaling/*` panels are diagnostics, unused in any table | skipped steps detected by the step counter and left out |
| doc: rows/columns in section 13, "code raises" in section 8, lens cadence in section 10, `ftbanal` description in section 12 | text only | corrected above |
| `custom_lr.py` is a layer-wise lr schedule without decay compensation | none: not used by any 2x2 arm; `create_optimizer` refuses combining it with the lr-scale paths | documented here |

What the reviewer asked for the early-lever 2x2 and what the code now guarantees: (A) $m_s$ is read per tensor from the *final*
init dump of the arm, after sink and gate (the exact fold keeps $\mathrm{rms}(W\,\mathrm{diag}\gamma)$, so the raw rms of q, k, fc1
moves by $s_q, s_k, s$; section 12 already reads the dump, this is why); (B) q, k, v get separate multipliers through the row mask;
(C) the profile-off cells are extracted without `--scale_weights` and with `--no_layernorm`, i.e. neither matrix scales nor
LayerNorm statistics, sink and gate targets kept; (D) an unmet target now aborts the run instead of training a different arm;
(E) the interpretation stays "profile bundle x explicit optimiser scaling", not identical trajectories. The precision caveat of
section 13 (`ftbrhopl` bf16) stands.

The accuracy JSON top-level fields of runs finished before 2026-09-19 stay wrong on disk; they are not read by any table. A
correct post-training analysis of a finished run can be regenerated from its saved final checkpoint with `--analyse_only`.
