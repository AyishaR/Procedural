# The readout interventions, precisely

Code: `models/readout_routing.py` (`ReadoutMixer`, `readout_init`) and
`VisionTransformer._forward_readout_mixed` in `models/vision_transformer.py`. Results:
`docs/residual_routing.md` (readout-only routing, follow-up 2).

**In one sentence:** the residual stream is *not* cut. Every block, including the last ones, runs
exactly as in vanilla ViT-B/16 on the full stream. Only the tensor handed to the classifier is
replaced by a fixed (or learned) weighted sum of the embedding and the individual branch outputs.

## 1. Vanilla ViT-B/16 (as configured in this repo)

$L = 12$ blocks, width $d = 768$, $N = 196$ patches plus one CLS token, so every stream tensor is
$(N{+}1) \times d$. Pre-LN blocks, no LayerScale, no stochastic depth, no `norm_pre`, no `fc_norm`,
`global_pool='token'` (the head reads the CLS row).

**Embedding.** Patches $p_1, \dots, p_N$ are embedded by $E$, and the learned CLS token $t$ and
position embeddings $P_0, \dots, P_N$ are added:

$$
x_0 \;=\; \begin{bmatrix} t + P_0 \\ E p_1 + P_1 \\ \vdots \\ E p_N + P_N \end{bmatrix}
\;\in\; \mathbb{R}^{(N+1)\times d}.
$$

**Blocks** $\ell = 0, \dots, L-1$ ($x_\ell$ is the stream *entering* block $\ell$):

$$
\begin{aligned}
a_\ell &= \mathrm{Attn}_\ell\big(\mathrm{LN}_{\ell,1}(x_\ell)\big), & x_\ell' &= x_\ell + a_\ell,\\
m_\ell &= \mathrm{MLP}_\ell\big(\mathrm{LN}_{\ell,2}(x_\ell')\big), & x_{\ell+1} &= x_\ell' + m_\ell .
\end{aligned}
$$

Unrolled, the final stream is the embedding plus every branch output:

$$
x_L \;=\; x_0 + \sum_{\ell=0}^{L-1} \big(a_\ell + m_\ell\big).
$$

**Head.** A final LayerNorm, then the CLS row, then a linear classifier:

$$
\text{logits} \;=\; W\,\mathrm{LN}_f\!\big(x_L\big)_{\text{cls}} + b .
$$

LayerNorm acts per token, so $\mathrm{LN}_f(x_L)_{\text{cls}} = \mathrm{LN}_f\big(x_{L,\text{cls}}\big)$.
**The head only ever sees the CLS row of whatever it is given.**

## 2. The intervention: a different readout, same blocks

Collect the $2L+1 = 25$ *sources* in execution order:

$$
s_0 = x_0,\qquad s_{2\ell+1} = a_\ell,\qquad s_{2\ell+2} = m_\ell \qquad (\ell = 0,\dots,L-1),
$$

i.e. `emb, attn_0, mlp_0, attn_1, …, mlp_11`. The branch outputs are captured by forward hooks
on `drop_path1` / `drop_path2`, the last module of each branch. The block loop itself is untouched
(`Block.forward` runs as in vanilla), so

$$
x_{\ell+1} = x_\ell + a_\ell + m_\ell \quad\text{exactly as before, for every } \ell .
$$

Only the head's input changes, from $x_L$ to a weighted sum over the sources:

$$
\boxed{\;z = \sum_{j=0}^{2L} w_j\, s_j, \qquad
\text{logits} = W\,\mathrm{LN}_f(z)_{\text{cls}} + b\;}
$$

With all $w_j = 1$, $z = x_L$ and the model is vanilla bit for bit; the gate checks this. The
weights $w$ are either a fixed buffer or an `nn.Parameter` (linear, no decay, backbone LR). The sum
is accumulated in fp32 in source order.

## 3. The arms

Using $x_{\ell+1} - x_{k} = \sum_{i=k}^{\ell}(a_i + m_i)$:

| arm | mode | $w$ (1 at, 0 elsewhere) | head input $z$ | trainable |
|---|---|---|---|---|
| base | — | all | $x_{12}$ | — |
| A | `fixed_last4` | 17..24 | $x_{12} - x_8 = \sum_{\ell=8}^{11}(a_\ell + m_\ell)$ | no |
| B | `learn_last4` | init as A | $\sum_j w_j s_j$ | yes |
| C | `learn_ones` | init all | $\sum_j w_j s_j$ | yes |
| **D** | `fixed_emb_last3` | 0, 19..24 | $x_0 + (x_{12} - x_9) = x_0 + \sum_{\ell=9}^{11}(a_\ell + m_\ell)$ | no |
| **E** | `fixed_b0_last3` | 0, 1, 2, 19..24 | $x_1 + (x_{12} - x_9) = x_0 + a_0 + m_0 + \sum_{\ell=9}^{11}(a_\ell + m_\ell)$ | no |

## 4. What the head actually receives (CLS row)

Because the head reads only the CLS row, the relevant quantity is $z_{\text{cls}}$. For D:

$$
z^{D}_{\text{cls}} \;=\; \underbrace{t + P_0}_{=:\,c\ \text{(image-independent)}}
\;+\; \sum_{\ell=9}^{11}\big(a_{\ell,\text{cls}} + m_{\ell,\text{cls}}\big).
$$

**The "input-token skip" in D carries no image information to the head.** At the CLS position
the embedding is the learned constant $c = t + P_0$; the patch embeddings $E p_i$ live in the
other rows, which the head never reads. D is therefore

$$
\text{logits}^D = W\,\mathrm{LN}_f\!\Big(c + \textstyle\sum_{\ell=9}^{11}(a_\ell + m_\ell)_{\text{cls}}\Big) + b,
$$

i.e. **A's construction, cut to the last 3 blocks, plus a learned offset $c$ inside the final
LayerNorm.** $\mathrm{LN}_f$ centres and rescales, so $c$ is not an ordinary bias: it shifts the
vector before normalisation, and so changes which directions survive the centring and how the
features are scaled. The vanilla readout contains the same $c$, since
$x_{L,\text{cls}} = c + \sum_{\ell}(a_\ell + m_\ell)_{\text{cls}}$; A removed it.

For E, the extra terms $a_{0,\text{cls}} + m_{0,\text{cls}}$ *do* depend on the image: block 0's
attention at the CLS row mixes all patch tokens. So E adds genuine image information from
block 0 on top of D, and $\text{E} \approx \text{D}$ (79.20 vs 79.29) says that information buys
nothing measurable.

Consequence for the reading of the D/E result (+1.2 over base): the gain is **not** from routing
image content from the input to the head. It comes from some combination of

1. cutting the head's input to blocks 9-11 instead of 8-11, and
2. the learned constant $c$ inside $\mathrm{LN}_f$, which A did not have.

The missing `fixed_last3` run is exactly the control that separates these two.

## 5. What changes in training: the gradient paths

Let $G = \partial\mathcal{L}/\partial z$. It is non-zero only in the CLS row, because the head
reads nothing else. Branch $k$ reads the stream $u_k = \sum_{i<k} s_i$ and produces
$s_k = f_k(u_k)$. The total gradient reaching source $j$ is

$$
\frac{\partial \mathcal{L}}{\partial s_j}
\;=\; \underbrace{w_j\, G}_{\text{direct path to the head}}
\;+\; \underbrace{\sum_{k>j} \Big(\frac{\partial f_k}{\partial u_k}\Big)^{\!\top}
\frac{\partial \mathcal{L}}{\partial s_k}}_{\text{through every later branch that reads it}} .
$$

* **Vanilla** ($w \equiv 1$): every branch output gets the head's gradient $G$ directly; this is
  the identity shortcut of the residual stream.
* **A, D, E** ($w_j = 0$ for the cut sources): blocks 0-8 in D (1-8 in E, 0-7 in A) lose the direct
  term. They are trained **only** through their effect on the inputs of the later branches, in the
  end through blocks 9-11 (8-11 for A). What their outputs contribute to the stream is still read
  by every later block; it just no longer counts at the output directly.
* In D the direct term survives only for $s_0 = x_0$. Since $G$ lives in the CLS row, that means
  $t$ and $P_0$ get a direct gradient, and the patch embeddings do not.

So the intervention (a) leaves the forward computation of every block unchanged, given its
input, (b) removes the early blocks' direct contribution to the logits, and (c) removes their
direct gradient. After training, the block weights and hence all $x_\ell$ of course differ from a
vanilla-trained model's.

## 6. Diagnostics caveat

Everything computed on the stream (`blk_acc_layer{i}`, ECE, rank/CKA, the logit lens) sees
$x_\ell$, which for A/D/E is **not** the tensor the head was trained on: the head reads $z$, not
$x_{12}$. `test_acc1` is the model's real accuracy.
