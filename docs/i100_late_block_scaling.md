# Why the blocks 9-11 scaling result does not survive proc-initialised early blocks

> **Reporting convention (applies to every number in this document).** All accuracies are
> **last-epoch test top-1** — the value at epoch 299 of a 300-epoch run — **not**
> max-over-epochs. Taking the max is a selection on the test set and inflates arms unevenly.
>
> Migrated from max to last on **2026-08-26**; every figure was recomputed directly from each
> run's `log.txt`. The conventions differ by ~0.02-0.15 on most arms but **not uniformly** —
> IN-1k random is 78.08 last vs 78.16 max while proc is 80.09 vs 80.10, so the old convention
> inflated the random baseline ~8x more than proc and **understated every gain over random by
> ~0.08**. No sign or ordering changed; several margins grew slightly.
>
> When adding an arm, take the final `test_acc1` from `log.txt`, never `Max accuracy` from stdout.

## 0c. CURRENT UNDERSTANDING: the early blocks carry the VALUE DISTRIBUTION (2026-09-02, rev 2)

> **2026-09-05: read §0d first.** A run-by-run verification pass (configs re-read from every run's own
> `Namespace` dump / checkpoint `args`, inits rebuilt and measured, end states measured) corrects
> several statements below: `ftbclip*` are random-init controls, the resolution is 0.45 not 0.41,
> `ftbvd` (+0.66, checkpoint-free) and `ftbslice` were missing from the ladder, `ftbcfg` is not the
> Gaussian twin of `ftbqmlnvo`, and the accuracy of every family tracks the final *training* loss.

> **This replaces the first version of §0c, written earlier the same day, which claimed the
> transfer is weight RANK. That claim is wrong and the text has been deleted rather than
> deprecated.** §0c.5 records what it said and what killed it. The rank observation itself
> survives, but it explains the *trajectory*, not the endpoint — see §0c.2.
>
> **UPDATE 2026-09-04 (§0c.8-0c.10): §0 is now RETRACTED as a mechanism.** All five re-runs and
> `ftbcfg` finished clean. The early-block effect is a **+1.26 super-additive interaction** between
> proc's LayerNorm gains and its v-slice — neither is worth anything alone — which no single scalar
> can represent. `ftbcfg` also settles sufficiency: proc's three attention scale factors, from a
> random init, land **0.59 BELOW random**. §0c.1's claim (what transfers is the value distribution)
> survives and is strengthened. The late-block results (§3.10.3, §3.12.3) were never affected.

### 0c.1 The anchor result: `ftbqmlnvo`

A random model, with proc's **sorted per-tensor values rank-mapped** into blocks 0-8 (the `v`
slice matched against proc's own `v`; `q,k` on the pooled map) plus **proc's LayerNorm gains**.
Blocks 9-11 random. No proc arrangement of any kind — the ordering inside every tensor is
inherited from the random tensor it was written into, so it is **full-rank** (fc1 stable rank
343.9 against random's 346.9).

**Clean.** All three seeds log `[init-sync] broadcast 152 tensors from rank 0`, so the §0a DDP
bug did not touch it. This is the only uncontaminated arm in the whole `ftbqm*` family.

**COMPLETE 2026-09-03, all three seeds at epoch 299:**

| arm | n | final top-1 | vs random 78.08 |
|---|---|---|---|
| **`ftbqmlnvo`** | 3 | **79.93 ± 0.39** (79.48, 80.14, 80.18) | **+1.85** |
| `ftb3i` (proc intact) | 3 | 79.99 ± 0.36 (79.64, 79.97, 80.35) | +1.91 |

**A gap of 0.06, against a 0.41 pp resolution.** Destroying the arrangement of proc's early-block
weights costs nothing measurable *by this comparison*.

> **Scope note added 2026-09-05 (§0c.11).** `ftbqmlnvo` differs from `ftb3i` in more than the
> arrangement: its q,k are narrower (50.9 vs 55.1/61.9) and its non-LayerNorm biases are zero. The
> arm that isolates arrangement alone is `ftbqks`, which matches `ftb3i` on every init statistic
> measured — gamma, all per-slice norms, write magnitude, logit scale and biases. If it lands at
> ~79.5 rather than ~80.0, destroying the arrangement costs **~0.5**, just above resolution, and
> "arrangement is free" holds only to within that.

*(The projection made on 2026-09-02 from partial runs — 79.94, 95% interval [79.87, 80.01] — came
in at **79.93**, an error of 0.007. It ran each seed through the empirical distribution of
`acc(299) − acc(e)` over 168 completed seed-runs: drift from ep 274 is +0.089 ± 0.076, from ep 284
+0.019 ± 0.055, from ep 289 +0.008 ± 0.043, never exceeding ±0.34. That method is reliable; the
epoch-49 proxy of §0b is not — see §0c.4.)*

**Why this matters.** §3.10.9.5 recorded the base rate: *"nine arms built as 'random model + proc
statistics' span −0.80 to +0.78, while four arms that start from the checkpoint span +1.91 to
+2.29. Nothing has ever landed in between."* `ftbqmlnvo` is a random-model-plus-statistics arm
sitting at +1.85, level with proc intact. **The base rate is broken, by a clean arm.**

**So what transfers is the value distribution.** Specifically: per-tensor sorted value multisets
in blocks 0-8, with `v` sliced from `qk`, plus proc's LayerNorm gains. **Not** the arrangement,
the structure, the rank, the singular directions, or the network. That is strictly less
information than the checkpoint and it is enough to recover essentially all of the effect.

### 0c.2 Rank governs the TRAJECTORY, not the endpoint — note this for later

Stable rank splits the arms perfectly on *early speed* and not at all on the *final score*.

| arm | fc1 stable rank | ep49 | ep74 | ep99 | ep124 | qk std | final |
|---|---|---|---|---|---|---|---|
| `ftb3i` | **5.2** | 60.28 | 43.98\* | **70.50** | 73.14 | 0.0763 | 79.99 |
| `ftb4e3` (buggy) | 337.9 | 60.75 | 67.69 | **71.39** | 73.59 | 0.0763 | 80.16 |
| **`ftb4e3fix`** | 337.9 | 68.51 | 73.55 | **75.40** | 76.51 | 0.0763 | running |
| `ftbqmln` | 344.4 | 66.65 | 72.34 | 74.71 | 76.25 | 0.0663 | 78.86 |
| `ftbqm1dvo` | 343.9 | 66.11 | 71.98 | 74.50 | 76.06 | 0.0663 | 79.44 |
| `ftbqmlnvo` | 343.9 | 68.01 | 73.19 | 75.53 | 76.73 | 0.0663 | ~79.94 |
| `ftbcfg` | 343.0 | 69.47 | 73.83 | 75.52 | 76.42 | n/a | running |

\* anomalous one-epoch dip.

**The only low-rank arm is the only slow starter.** Every full-rank arm sits at 74.5-75.5 at
epoch 99; `ftb3i` at 70.50 — a 4-5 point gap with a perfect split on rank.

**`ftb4e3fix` flips sides, and that is the clean test.** `ftb4e3` and `ftb4e3fix` are the same
construction; the DDP bug is the only difference. Buggy it starts at 71.39 (slow family); fixed
it starts at **75.40** (fast family). Permuting proc's weights therefore moves epoch-99 accuracy
by **+4.0**, and §3.13.x's *"permuting the weights destroys the rank and changes early speed not
at all"* is a third conclusion that rested entirely on the contaminated arm.

**It is not weight scale.** §3.13.x explained the slow start as *"larger weights take smaller
relative steps"* under AdamW. `ftb4e3fix` carries proc's widest qk (std 0.0763) and starts
*faster* than `ftbqmln` at 0.0663. The scale account is contradicted; rank is not.

**Status: a live idea, not a result.** The mechanism — "low rank at init constrains the effective
degrees of freedom, so training must grow capacity rather than prune it" — is plausible and
unestablished. Nothing here says it matters for anything you care about, since both routes reach
~80. What it does buy is methodological (§0c.4) and it is what `ftbsv`/`ftbsb` now test (§0c.6).

### 0c.3 What §0 now has, and what it still cannot explain

`ftbqmlnvo` sits at `value_write` **0.4917**, identical to `ftb3i`, in §0's top group — which
predicted **+1.69 ± 0.41** against a measured **+1.86**. That is §0's **first clean, out-of-sample
confirmation**; every other member of that group is contaminated. §0 is the best account we have.

It is still not a mechanism, and four things remain unexplained:

* **Non-monotonicity.** Random init sits at 0.307 — *below* the winning group's 0.49-0.54 — and
  scores 0.00. `ftb4o` at 3.949 scores −0.80. Three measured groups is a grouping, not a
  dose-response curve, and there is no account of the shape or of where the optimum is.
* **Sufficiency is untested.** Every arm in the top group carries proc's actual per-tensor value
  multisets. `ftbcfg` carries three scalars and no checkpoint, and is the only test of whether
  the scalar alone suffices. It has not landed (§0c.4 for a within-family projection).
* **The depth ramp.** Monotone from +0.70 (1 proc block) to +2.29 (11). A per-block scalar
  averaged over blocks does not predict why adding blocks keeps helping (§3.13.3).
* **`ftb4n` = 75.67**, proc's blocks 9-11 on random early blocks — *worse* than random — and the
  **IN-100 sign flip** (e4 = +0.16, negative direction at the other scale, §3.10.5), still the
  document's main open problem.

### 0c.4 The epoch-49 proxy has now failed TWICE, in opposite directions

| arm | §0b.1 predicted | measured | error | sigma |
|---|---|---|---|---|
| `ftbqmvo` | 78.71 | 78.69 | −0.02 | 0.0 |
| `ftb4i` | 80.33 | 79.96 | −0.37 | 0.7 |
| **`ftbqmlnvo`** | 78.66 | **79.93** | **+1.27** | **2.4** |
| **`ftb11isfix`** | 78.36 | **76.15** | **−2.21** | **4.2** |

Two of the four resolved predictions are catastrophic, **and they miss in opposite directions**,
so this is not a correctable bias — the proxy is simply uninformative for these arms. A 4.2 sigma
miss on a fit with a 0.53 pp residual sd means the residual sd is not describing the error
distribution at all.

**Why.** §0b fitted one line across **two trajectory families** (§0c.2). Slow starters end high,
so the fit encodes "high train loss at ep49 → good final" and therefore misprices every fast
starter. `ftbqmlnvo` is the first arm to start fast *and* finish high, so it is exactly the case
the proxy cannot represent.

**Consequence: treat every remaining §0b.1 prediction as worthless**, `ftb4e3fix` (78.58) and
`ftbcfg` (78.24) included. Do not use the ep-49 proxy on any arm in this document.

**What IS reliable: late-epoch drift extrapolation.** Projecting `ftbqmlnvo` from its epoch
274/284/289 reads through the empirical `acc(299) − acc(e)` distribution over 168 completed
seed-runs gave **79.94 [79.87, 80.01]** against a measured **79.93** — an error of 0.007. Past
~epoch 270 that method is accurate to ~0.1 pp. Before ~epoch 200 nothing here is trustworthy
except a within-family comparison (below).

**What replaces it: compare within a trajectory family, later.** Among fast starters the epoch-164
read orders correctly — `ftbqmln` 77.50 → 78.86 and `ftbqm1dvo` 77.96 → 79.44. On that basis:

| arm | ep164 | within-family projection |
|---|---|---|
| `ftb4e3fix` | 77.79 | **~79.2-79.3** |
| `ftbcfg` | 76.43 (n=1) | **~77.8-78.0**, at or below random |

Both are extrapolations from two calibration points, and `ftbcfg`'s late reads are n=1-2. If they
hold: arrangement costs ~0.7 pp (not zero, not the 1.4 the rank hypothesis wanted), and the
three-scalar checkpoint-free recipe **fails**.

### 0c.5 Retracted: the rank hypothesis for the endpoint (held for ~6 hours, 2026-09-02)

It claimed the early-block benefit is proc's low rank (stable rank 5-22 vs 194-343 for any
permutation), on the grounds that `value_write` is Frobenius-based and blind to the spectrum, and
that *"every arm that failed is full-rank and the one arm that won is low-rank."*

**`ftbqmlnvo` is full-rank and wins**, so the necessity claim is false, and it is clean, so it
cannot be explained away as contamination. `ftb4e3fix` was never needed to settle it.

The supporting *measurements* were all correct and are retained: proc's early matrices really are
stable rank 5-22, a permutation or a quantile rank-map really does put them on Marchenko-Pastur,
`value_write` really is blind to that, and no init in this repo had ever controlled rank. The
error was reading a real init-time difference as the cause of a final-accuracy difference, on the
strength of two proxy-predicted numbers that §0c.4 now shows were unreliable.

### 0c.6 Queued, and what is still open

**Re-runs of the contaminated `ftbqm*` arms, queued 2026-09-02** — tiers 1 and 2 of §0a.1, 15
jobs, fresh SLURM_IDs so `--auto_resume` cannot reach the contaminated checkpoints:

| arm | new SLURM_ID | gives `ftbqmlnvo` |
|---|---|---|
| `ftbqmln` | 29529147 | a clean single-variable control on the **qkv** axis |
| `ftbqm1dvo` | 29529150 | a clean single-variable control on the **1-D** axis |
| `ftbqm1d` | 29529154 | the pooled / all-1D corner anchoring the 2x2 |
| `ftbqm1dv` | 29529157 | the both-sliced corner; also `ftb4e3`'s claimed twin (§0.5) |
| `ftbqm1dqk` | 29529160 | the qk-only corner |

Not queued: `ftbqmbias`, `ftbqm1dpar` (tier 3, both near-zero effects).

**`--custom_init_type spectral`** (`main.spectral_reinit`, added 2026-09-02) still exists and is
verified (`plots/verify_spectral_init.py --fig`, all checks pass, `plots/out/fig14_spectra.png`).
Its motivation has changed: `ftbsv` (proc's spectrum, random directions) and `ftbsb` (proc's
directions, random spectrum) no longer test the endpoint — they test **§0c.2's trajectory claim**,
by moving rank while holding ‖W‖_F and `value_write` exactly fixed. Neither is queued.

**Still the largest uncontrolled factor: the LayerScale baseline.** Every number in this document
is measured against a no-LayerScale random init.


### 0c.7 The block-0 anomaly: the SAME operation is free at nine blocks and catastrophic at one

`ftb11isfix` finished 2026-09-03 and does not fit §0c.1.

| arm | proc content | arrangement | n | final | vs random |
|---|---|---|---|---|---|
| `ftb11i` | block **0**, rest random | intact | 3 | 78.78 ± 0.18 | **+0.70** |
| **`ftb11isfix`** | block **0**, rest random | **destroyed** | 2 | **76.15 ± 0.35** | **−1.93** |
| `ftb3i` | blocks **0-8**, 9-11 random | intact | 3 | 79.99 ± 0.36 | +1.91 |
| **`ftbqmlnvo`** | blocks **0-8**, 9-11 random | **destroyed** | 3 | **79.93 ± 0.39** | +1.85 |

**Destroying the arrangement costs 0.06 across blocks 0-8 and 2.63 at block 0 alone** — and at
block 0 it lands **2 points BELOW random init**, i.e. proc's block 0 with its values scrambled is
much worse than having no proc block at all. Both `ftb11isfix` seeds agree to 0.35, so this is
not noise: 2.63 against a 0.41 pp resolution.

**This is clean and it is not the DDP bug.** `ftb11isfix` is the §0a re-run; the pre-fix `ftb11is`
scored 77.81, so the fix made the effect **larger**, not smaller (−0.97 → −2.63). §0a already
flagged the direction — *"1 block costly, 9 blocks free"* — but at a size that looked like it
could be noise. It cannot now.

**Neither §0 nor §0c predicts this.** `value_write` is permutation-invariant, so §0 puts
`ftb11i` and `ftb11isfix` in the same group. §0c.1 says the value distribution is what transfers,
and `ftb11isfix` carries proc's block-0 value multisets exactly (a shuffle of proc's own tensor).
Both accounts say the pair should score the same. They differ by 2.63.

**Candidate readings, none tested:**

* **Block 0 is structurally special.** It is the only block whose input is the raw patch
  projection rather than a learned residual stream. Scrambling it may corrupt the
  patch-to-token map in a way no deeper block can, and that nothing downstream can repair when
  every deeper block is random.
* **Isolated scale without matching content is harmful, and this is that failure mode again.**
  `ftb11isfix` puts proc's large block-0 norms (‖W‖_F 55/62/29 against random's 15.4) into a
  single block of an otherwise random network with random arrangement. The other arms that
  landed *below* random do the same thing: `ftb4o` (write magnitude only) −0.80 and `ftb4n`
  (proc 9-11 on scaled random early blocks) −2.41. What makes `ftbqmlnvo` different may be that
  it changes nine blocks coherently rather than one in isolation.
* **Extent, not depth.** The 1-vs-9 contrast confounds *which* block with *how many*. A
  shuffled `ftb9i`/`ftb8i` (blocks 0-2 / 0-3) would separate them and is the obvious next arm.

**What it does NOT overturn.** §0c.1's claim is about blocks 0-8 as a set, and `ftbqmlnvo` vs
`ftb3i` measures exactly that at n=3 each. This section narrows the claim's scope rather than
contradicting it: *the value distribution is what transfers when proc occupies the early stack;
it is not a per-block statement.*

**`ftb4e3fix` has now landed and the asymmetry is confirmed, at n=3 on both sides.** It is proc
0-8 entrywise-shuffled — `ftb11isfix`'s operation at nine blocks instead of one — and it scores
**79.50 +/- 0.35**, i.e. **-0.49** against `ftb3i`. `ftb11isfix` scores **76.36 +/- 0.43**, i.e.
**-2.42** against `ftb11i`. Same operation, **five times the cost at one block**, and only the
one-block version lands below random. The block-0 anomaly is a measured result, not a projection.

`ftbqks` (queued 2026-09-04) tests whether the 0.49 is even real, or an artifact of `ftb4e3`'s
shuffle pooling q and k — see §0c.10.


### 0c.8 The clean 2x3 closes: it is an INTERACTION, not two main effects (2026-09-04)

All five tier-1/2 re-runs and `ftbcfg` have finished under the §0a fix. **Three standing claims
die here.** Everything below is n=3, clean, blocks 0-8 with 9-11 random, against random 78.08.

**The grid.** Every cell is a random model with proc's sorted per-tensor values rank-mapped into
blocks 0-8. Two knobs: how the fused `qkv` is matched, and which 1-D params are transplanted.

| 1-D params | qkv **pooled** | qkv **v sliced** | v-slice effect |
|---|---|---|---|
| **none** | `ftbqm` **78.16** | `ftbqmvo` **78.61** | +0.45 |
| **LayerNorm only** | `ftbqmln` **78.22** | `ftbqmlnvo` **79.93** | **+1.71** |
| **all 8** | `ftbqm1d` **78.01** | `ftbqm1dvo` **79.11** | +1.10 |

Taking `ftbqm` as this family's baseline:

```
LayerNorm gains ALONE   +0.06        (ftbqmln,  0.3 sigma -- nothing)
v-slice ALONE           +0.45        (ftbqmvo,  ~1 sigma  -- barely)
BOTH                    +1.77        (ftbqmlnvo)
sum of singles          +0.51
INTERACTION             +1.26        strongly super-additive
```

**Neither knob does anything on its own; together they are worth +1.77.** The whole pooled column
is flat at 78.0-78.2, i.e. at random, regardless of the 1-D treatment.

**Three claims retracted.**

1. ~~"proc's LayerNorm gains are worth +0.78"~~ (§0.2, header point 3a). Clean, `ftbqmln` is
   **78.22, +0.14 over random, 0.7 sigma**. The gains alone are worth **nothing**. The +0.78 came
   from the §0a-contaminated run (78.86) and the bug's inflation (-0.64) is most of the gap.
2. ~~"the 1-D parameters are not the mechanism"~~ (§0.2, point 3a-ii). They are half of it — but
   only the LayerNorm ones, and only jointly with the v slice. Adding the other five 1-D params on
   top **costs 0.82** (`ftbqmlnvo` 79.93 -> `ftbqm1dvo` 79.11), so proc's attention/MLP biases are
   actively harmful rather than inert.
3. ~~"one scalar (`value_write`) explains 83% of the variance"~~ (§0.1). A single scalar cannot
   represent a +1.26 interaction. `ftbqmln` and `ftbqmlnvo` differ by **1.71** while their
   `value_write` differs only through `||W_v||`; `ftbqm` and `ftbqmln` differ by 0.06 while their
   `value_write` differs 2.6-fold. §0's grouping is a coincidence of this arm set.

**And the DDP bug inflated every affected arm by about half a point**, consistently:

| arm | contaminated | clean | shift |
|---|---|---|---|
| `ftbqmln` | 78.86 | **78.22** | **-0.64** |
| `ftb4e3fix` (vs `ftb4e3`) | 80.16 | **79.50** | **-0.66** |
| `ftbqm1d` | 78.50 | **78.01** | **-0.49** |
| `ftbqm1dvo` | 79.44 | **79.11** | **-0.33** |

Four arms, all negative, mean **-0.53**. Anything in this document quoting a pre-fix `ftbqm*` or
shuffle number is roughly half a point too high.

### 0c.9 SUFFICIENCY ANSWERED: the scale factors alone are worse than nothing

`ftbcfg` finished: **77.49 +/- 0.54 (n=3), i.e. 0.59 BELOW random.** It is a plain random init
given proc's three attention scale factors and no checkpoint anywhere
(`--custom_init_type slice_scale --slice_scale_qk 1.587 --slice_scale_v 0.729
--slice_scale_proj 2.395`), hitting `value_write` 0.536, logit scale 0.00806 and qk/v 2.18 — all
three of proc's numbers at once.

**This closes §0.4's central open question with a No.** Matching proc's attention statistics from
a random init is not merely insufficient, it is *harmful*. Together with `ftbqmlnvo` (79.93, which
carries proc's actual value multisets and nothing else) the split is clean:

* proc's **value distributions** transfer (+1.85)
* proc's **scale factors** do not (-0.59)

It also joins the list of arms that land below random by importing an isolated scale without the
matching content: `ftb4o` -0.80, `ftb4n` -2.41, `ftbvu` -0.43, `ftb11isfix` -1.72, `ftbcfg` -0.59.

### 0c.10 The clean ladder, all n=3

Blocks 0-8, blocks 9-11 random, against random init **78.08 +/- 0.19**. Resolution **0.45 pp** (§0d.1; the 0.41 quoted before came from a smaller pool). Rows added 2026-09-05: `ftbvd`, `ftbslice`.

| what blocks 0-8 receive | arm | final | delta |
|---|---|---|---|
| proc **intact** | `ftb3i` | **79.99 +/- 0.36** | **+1.91** |
| proc's values rank-mapped, v sliced, LN gains | `ftbqmlnvo` | **79.93 +/- 0.39** | **+1.85** |
| proc's values, entrywise shuffled (qk pooled) | `ftb4e3fix` | **79.50 +/- 0.35** | **+1.42** |
| ... + all 8 1-D params | `ftbqm1dvo` | 79.11 +/- 0.26 | +1.03 |
| proc's values, v sliced, no 1-D | `ftbqmvo` | 78.61 +/- 0.14 | +0.53 |
| **random, v x0.459, no checkpoint** | `ftbvd` | **78.74 +/- 0.10** | **+0.66** |
| random, qk x3.795 v x1.745, no checkpoint | `ftbslice` | 78.57 +/- 0.28 | +0.49 |
| per-tensor norms only | `ftbnorm` | 78.28 +/- 0.32 | +0.20 |
| proc's values, pooled qkv, LN gains | `ftbqmln` | 78.18 +/- 0.15 | +0.10 |
| proc's values, pooled qkv, no 1-D | `ftbqm` | 78.16 +/- 0.12 | +0.08 |
| qk scaled up, no checkpoint | `ftbqu` | 78.05 +/- 0.09 | -0.03 |
| proc's values, pooled qkv, all 1-D | `ftbqm1d` | 78.01 +/- 0.13 | -0.07 |
| v scaled up, no checkpoint | `ftbvu` | 77.65 +/- 0.06 | -0.43 |
| **three scale factors, no checkpoint** | `ftbcfg` | **77.49 +/- 0.54** | **-0.59** |
| **proc's block 0 only, shuffled** | `ftb11isfix` | **76.36 +/- 0.43** | **-1.72** |

Open, and unchanged by any of this: the **block-0 anomaly** (§0c.7 — shuffling costs 0.49 across
nine blocks and 2.42 at block 0 alone, now both n=3), the **depth ramp**, **`ftb4n`**, the
**IN-100 sign flip**, and the missing **LayerScale baseline**.

**Queued 2026-09-04: `ftbqks` (SLURM_ID 29535764, 3 seeds).** `ftb4e3fix` with `q` and `k`
shuffled as SEPARATE pools, so proc's per-block `||W_q||`/`||W_k||` are exact rather than collapsed
toward their common mean (block 4: proc 53.62/62.39, `ftb4e3`'s pooled shuffle 58.14/58.20, up to
13.5% off).

> **Prediction corrected 2026-09-05 — see §0c.11.** The sentence that stood here said `ftbqks`
> would reach ~79.9 if the qk asymmetry explained the `ftb4e3fix`/`ftbqmlnvo` gap. That was the
> wrong reference: `ftbqks` shuffles `attn.qkv.bias`, `attn.proj.bias` and `mlp.fc*.bias` too, so
> it sits in the **all-1D** row beside `ftb4e3fix` (79.50), not beside `ftbqmlnvo` (79.93, whose
> biases are zero). The gap decomposes as **-0.82 (biases) + 0.39 (wider q,k)**, both measured.
> **`ftbqks` should land near 79.5**, and it changes only the q-vs-k asymmetry at fixed mean width,
> which moves neither the write magnitude nor the logit scale.


### 0c.11 Why the three "arrangement destroyed" arms differ: an additive decomposition (2026-09-05)

`ftbqmlnvo` (79.93), `ftb4e3fix` (79.50) and `ftbqm1dvo` (79.11) all destroy the arrangement of
proc's blocks 0-8 and all carry proc's per-slice value multisets. They are **not** the same
experiment, and the differences between them are fully accounted for by two variables.

Measured at init (`plots/measure_init_rho_arms.build_arm`, blocks 0-8, mean over blocks):

| arm | gamma | ‖W_q‖ | ‖W_k‖ | ‖W_v‖ | **write** | logit | qkv.bias | proj.bias | fc1.bias | final |
|---|---|---|---|---|---|---|---|---|---|---|
| `ftb3i` (intact) | 0.384 | **55.1** | **61.9** | 28.8 | **0.479** | 0.082 | 2.89 | 0.65 | 1.15 | **79.99** |
| `ftbqmlnvo` | 0.384 | 50.9 | 50.9 | 28.8 | **0.479** | 0.062 | **0.00** | **0.00** | **0.00** | **79.93** |
| `ftb4e3fix` | 0.384 | 58.6 | 58.6 | 28.8 | **0.479** | 0.082 | 2.89 | 0.65 | 1.15 | **79.50** |
| `ftbqks` | 0.384 | **55.1** | **61.9** | 28.8 | **0.479** | 0.082 | 2.89 | 0.65 | 1.15 | *running* |
| `ftbqm1dvo` | 0.384 | 50.9 | 50.9 | 28.8 | **0.479** | 0.062 | 2.89 | 0.65 | 1.15 | **79.11** |

**gamma, ‖W_v‖ and the write magnitude are identical (0.479) across all five.** The two variables
that move are the **q,k width** and **whether proc's non-LayerNorm biases came along**.

**The decomposition**, from `ftbqmlnvo`:

```
ftbqmlnvo                                    79.93
  + proc's biases        (zero -> proc)      -0.82   ->  79.11 = ftbqm1dvo    MEASURED 79.11
  + wider q,k            (50.9 -> 58.6)      +0.39   ->  79.50 = ftb4e3fix    MEASURED 79.50
  + q/k asymmetry        (58.6/58.6 -> 55.1/61.9)  ?  ->  ~79.5 = ftbqks      pending
```

Both steps land on the measured value exactly. So **`ftbqmlnvo`'s 0.43 advantage over `ftb4e3fix`
is not the qk pooling** — it is -0.82 from the biases, partly cancelled by +0.39 from the wider
q,k. Two effects of opposite sign that nearly annul.

**Where the qk-pooling hypothesis came from, and why it was wrong.** §0c.10 proposed that
`ftb4e3`'s `attn.qk.weight` shuffle (rows [0:2e] as one pool) explained the gap, because it
collapses proc's per-block ‖W_q‖/‖W_k‖ toward their common mean. That is a real effect on the
weights but the wrong attribution: the two arms *also* differ in their 1-D treatment
(`quantile_1d_mode layernorm` vs a shuffle list that includes `attn.qkv.bias`, `attn.proj.bias`,
`mlp.fc1.bias`, `mlp.fc2.bias`), and that is the larger term.

**Consequence for `ftbqks`.** It is in the **all-1D row**, not `ftbqmlnvo`'s row. Its correct
reference is `ftb4e3fix` (79.50), and it changes only the q-vs-k asymmetry at a fixed mean width,
which moves neither the write magnitude nor the logit scale. **The prediction is ~79.5, not
~79.9.** Its partial reads on 2026-09-05 (s0 79.03@284, s1 79.10@259, s2 79.71@254, the latter two
still climbing) are consistent with that.

**Two further observations from the table.**

* `ftbqmlnvo` has the **lowest logit scale of the group** (0.062 against 0.082) and **zero biases**,
  and still beats every arm except intact proc. Neither the logit scale nor proc's biases are
  doing useful work.
* **Proc's biases cost 0.82.** They are not inert — transplanting them is worse than leaving the
  random model's zeros. This is the same conclusion §0c.8 reached from the 3x2 grid, now confirmed
  on a second, independent pair (`ftbqmlnvo` vs `ftbqm1dvo` in the grid; `ftb4e3fix` vs a
  bias-free counterfactual here).

**What this does NOT explain.** `ftb3i` (79.99) and `ftbqks` (predicted ~79.5) have **identical**
init statistics on every column of the table above — same gamma, same per-slice norms, same write,
same logit, same biases. The only difference is the arrangement within each tensor. If `ftbqks`
lands at 79.5 that is a **0.5 pp cost of destroying the arrangement**, just above the 0.41
resolution, and nothing measured at init distinguishes the two. §0c.1's "arrangement is free"
would then hold only to within ~0.5, not exactly.


## 0d. VERIFICATION PASS (2026-09-05): what was actually run, re-derived from the runs themselves

> Written before anything in it was used for a decision. Nothing below trusts an arm *name*, a
> table in this document, or memory: every configuration was read back from the run's own
> `Namespace(...)` dump in `logs/ft_<job>_<name>.out`, or -- for the 53 completed runs launched
> before those logs existed (the `b_vitb_*` chain launchers, Aug 8-14) -- from the `args` object
> stored in the run's own `checkpoint-299.pth`. Every accuracy is the last-epoch `test_acc1`
> re-read from `log.txt` (`max(epoch)`, not line count). Scripts: `plots/verify/` (README there).
> The train-loss / test-loss numbers are epoch-299 values from the same logs.

### 0d.1 Corrections to statements made earlier in this document

1. **`ftbclip01/1/5` never touched the checkpoint.** All three ran with
   `--random_blocks 0..11`; the log says `Loading state dict with 0 keys`. They winsorised a
   *random* init in blocks 0-8. §3.10.6 describes them correctly ("no checkpoint is involved");
   the `LABEL` in `plots/make_figures.py` ("proc, extreme tail clipped") and every later sentence
   that treated them as a test of proc's tails are wrong. Pooled, the nine seeds give
   **77.69 +/- 0.45** against `r`'s 78.08 +/- 0.19 (Welch p = 0.068): either lightening a Gaussian
   init's tails costs ~0.4, or `r` is a high draw. Clipping *proc's* tails has never been run on
   IN-1k.
2. **Resolution is 0.45 pp, not 0.41.** Pooled within-arm sd over every arm with >= 2 completed
   seeds is **0.278 (df = 80, 40 arms)**; 0.286 over clean-status arms only. Two n = 3 arms
   therefore resolve at ~0.45 (2 sigma). `ftbqmlnvo` 79.93 vs `ftb4e3fix` 79.50 (0.43) is
   borderline, not established.
3. `ftbqmln` (clean) is **78.18**, not 78.22 (78.12 / 78.06 / 78.35).
4. **Two checkpoint-free arms were missing from the §0c.10 ladder**: `ftbvd` (random, v x0.459)
   **78.74 +/- 0.10, +0.66, n = 3** and `ftbslice` (random, qk x3.795, v x1.745) 78.57 +/- 0.28,
   +0.49, n = 3. §0.4 lists them with stale n. `ftbvd` is the best checkpoint-free early-block arm
   we have, it clears the resolution, and it falsifies the sentence "every non-value arm <= +0.20"
   used in the 2026-09-05 mechanism discussion.
5. **`ftbcfg` is not the Gaussian analogue of `ftbqmlnvo`.** Its three factors were fitted to a
   blocks-0-8 *average* that includes block 0, whose every matrix is 3-4x random. Measured on the
   actual init (§0d.5): `ftbcfg`'s attention write in blocks 1-8 is 0.537 vs proc's 0.44
   (+22%), its MLP write is 1.23 vs proc's 0.36 (3.4x), its LayerNorm gains are 1.0 vs 0.4, and
   block 0 gets nothing special. It tested "proc's average attention scales on a Gaussian net",
   not "proc's per-block scale profile". §0c.9's "SUFFICIENCY ANSWERED" is therefore overstated.
6. **Launcher names of old runs can be stale.** `vitbase_runs/b_vitb_7e.sh` carries
   `SLURM_ID=29408623`, but that run's stored args are the `ftb7b` construction
   (`upscale_random_match_delta_norms`, blocks 5-11). The inventory below names every old run by
   its stored args, never by its launcher.
7. **The rank-sync bug inflated every affected arm and did so by making it fit *worse*:**
   `ftb4e3` +0.66 (train loss 2.583 vs 2.299 clean), `ftbqmln` +0.68 (2.296 vs 2.231), `ftbqm1d`
   +0.49, `ftbqm1dvo` +0.33, `ftb11is` +1.45 (block 0). Four replicas with different permutations
   sharing one gradient acted as a regulariser -- consistent with §0d.4, and a reminder that a
   fit deficit is not proof of a mechanism.

### 0d.2 Complete inventory of finished ViT-B / IN-1k runs

Random baseline `r` = 78.08 +/- 0.19; "vs r" is the difference of seed means. "train loss" is the
epoch-299 training loss (mixup/cutmix soft targets), seed-mean. Contaminated (pre-fix, RNG-edit)
runs are listed separately in D and used nowhere else.


**A. proc VALUES in blocks 0-8**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `ftb3i` | proc intact | 3 | 79.99 ± 0.36 | +1.91 | 2.626 |
| `ftbqmlnvo` | proc marginals rank-mapped, v sliced, LN gains+biases (permuted); linear biases 0 | 3 | 79.93 ± 0.39 | +1.86 | 2.328 |
| `ftb4e3fix` | proc tensors permuted within slice (qk pooled); all 1-D permuted | 3 | 79.50 ± 0.35 | +1.42 | 2.299 |
| `ftbqks` | as ftb4e3fix, q and k permuted separately (running) | - | - | - | - |
| `ftbqm1dvo` | v sliced, ALL 1-D permuted | 3 | 79.11 ± 0.26 | +1.03 | 2.277 |
| `ftbqmvo` | v sliced, no 1-D | 3 | 78.61 ± 0.14 | +0.53 | 2.271 |
| `ftbqmln` | pooled qkv, LN gains+biases | 3 | 78.18 ± 0.15 | +0.10 | 2.231 |
| `ftbqm` | pooled qkv, no 1-D | 3 | 78.16 ± 0.12 | +0.08 | 2.247 |
| `ftbqm1d` | pooled qkv, all 1-D | 3 | 78.01 ± 0.13 | -0.07 | 2.224 |

**B. NO checkpoint values in blocks 0-8**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `r` | timm random | 3 | 78.08 ± 0.19 | -0.00 | 2.225 |
| `ftbvd` | random, v x0.459 | 3 | 78.74 ± 0.10 | +0.66 | 2.284 |
| `ftbslice` | random, qk x3.795, v x1.745 | 3 | 78.57 ± 0.28 | +0.49 | 2.239 |
| `ftbnorm` | random directions, proc per-tensor norms (qkv pooled), LN gain uniform at proc RMS | 3 | 78.28 ± 0.32 | +0.20 | 2.257 |
| `ftbqu` | random, qk x2.177 | 3 | 78.05 ± 0.09 | -0.03 | 2.192 |
| `ftbclip01` | random, top 0.1% |w| winsorised | 3 | 77.76 ± 0.69 | -0.32 | 2.229 |
| `ftbclip1` | random, top 1% | 3 | 77.73 ± 0.16 | -0.35 | 2.217 |
| `ftbclip5` | random, top 5% | 3 | 77.59 ± 0.55 | -0.49 | 2.225 |
| `ftbvu` | random, v x1.745 | 3 | 77.65 ± 0.06 | -0.43 | 2.192 |
| `ftbcfg` | random, qk x1.587, v x0.729, proj x2.395 | 3 | 77.50 ± 0.54 | -0.57 | 2.201 |
| `ftb4o` | random blocks 0-7 rho-matched to proc (8-11 random) | 1 | 77.27 | -0.80 | 2.228 |

**C. block 0 only (1-11 random)**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `ftb11i` | proc block 0 intact | 3 | 78.78 ± 0.18 | +0.70 | 2.274 |
| `ftb11d` | proc block 0, v/proj/fc2 downscaled to random rho | 3 | 77.71 ± 0.12 | -0.37 | 2.200 |
| `ftb11s` | random block 0 upscaled to proc rho | 1 | 76.59 | -1.49 | 2.263 |
| `ftb11isfix` | proc block 0 permuted | 3 | 76.36 ± 0.43 | -1.72 | 2.286 |

**D. contaminated (pre rank-sync fix) versions**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `ftb4e3_PRE` |  | 3 | 80.16 ± 0.10 | +2.08 | 2.583 |
| `ftbqm1dv_PRE` | v sliced (qk_v), all 1-D | 3 | 79.48 ± 0.36 | +1.40 | 2.325 |
| `ftbqm1dvo_PRE` |  | 3 | 79.44 ± 0.29 | +1.36 | 2.322 |
| `ftbqmln_PRE` |  | 3 | 78.86 ± 0.24 | +0.78 | 2.296 |
| `ftbqm1dqk_PRE` | qk sliced, all 1-D | 3 | 78.66 ± 0.17 | +0.58 | 2.272 |
| `ftbqm1d_PRE` |  | 3 | 78.50 ± 0.12 | +0.42 | 2.263 |
| `ftbqm1dpar_PRE` | pooled, 1-D Gaussian-matched | 3 | 78.35 ± 0.32 | +0.28 | 2.253 |
| `ftbqmbias_PRE` | pooled, linear biases only | 3 | 78.13 ± 0.17 | +0.05 | 2.245 |
| `ftb11is_PRE` | block 0 permuted | 3 | 77.81 ± 0.30 | -0.27 | 2.223 |
| `ftb0l_PRE` | all 12 blocks permuted + downscaled | 1 | 76.49 | -1.58 | 2.251 |

**E. proc prefix intact (ftbKi = K proc blocks at the start), scaling of the prefix**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `p` | all 12 proc | 3 | 80.09 ± 0.12 | +2.01 | 2.469 |
| `ftb1i` |  | 3 | 80.37 ± 0.12 | +2.29 | 2.563 |
| `ftb2i` |  | 1 | 80.24 | +2.17 | 2.595 |
| `ftb3i` |  | 3 | 79.99 ± 0.36 | +1.91 | 2.626 |
| `ftb4i` |  | 3 | 79.89 ± 0.13 | +1.81 | 2.607 |
| `ftb5i` |  | 1 | 79.67 | +1.59 | 2.596 |
| `ftb6i` |  | 1 | 79.66 | +1.59 | 2.560 |
| `ftb7i` |  | 3 | 79.89 ± 0.35 | +1.81 | 2.492 |
| `ftb8i` |  | 3 | 79.58 ± 0.25 | +1.50 | 2.403 |
| `ftb9i` |  | 1 | 79.49 | +1.41 | 2.362 |
| `ftb10i` |  | 1 | 79.11 | +1.03 | 2.323 |
| `ftb11i` |  | 3 | 78.78 ± 0.18 | +0.70 | 2.274 |
| `ftb4m` | proc 0-7 downscaled to random rho, 8-11 random | 1 | 80.00 | +1.92 | 2.426 |
| `ftb4l` | proc all, 0-7 downscaled to random rho | 1 | 79.85 | +1.77 | 2.345 |
| `ftb4g` | proc 0-7, random 8-11 downscaled (v,proj,fc1,fc2) | 1 | 80.00 | +1.92 | 2.480 |
| `ftb4n` | random 0-7 upscaled to proc rho + proc 8-11 | 1 | 75.67 | -2.41 | 2.306 |
| `ftb4k` | random all, all 12 blocks rho-matched to proc | 1 | 79.77 | +1.69 | 2.398 |
| `ftb0a` | proc attention sublayers only, all blocks | 1 | 79.54 | +1.46 | 2.336 |
| `ftb0m` | proc MLP sublayers only, all blocks | 1 | 78.99 | +0.91 | 2.410 |

**F. late-block families (prefix random)**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `ftbrho` | random, blocks 9-11 to rho=1.4 (recipe) | 3 | 79.69 ± 0.30 | +1.61 | 2.271 |
| `ftbrho07` | rho=0.7 | 1 | 78.88 | +0.80 | 2.221 |
| `ftb1b` | random + rho-matched last 1 | 1 | 79.16 | +1.08 | 2.235 |
| `ftb2b` | last 2 | 3 | 79.78 ± 0.02 | +1.70 | 2.259 |
| `ftb3b` | last 3 (= a1) | 3 | 80.00 ± 0.14 | +1.92 | 2.292 |
| `ftb4b` | last 4 | 3 | 80.02 ± 0.15 | +1.94 | 2.356 |
| `ftb5b` | last 5 | 1 | 80.10 | +2.02 | 2.326 |
| `ftb6b` | last 6 | 1 | 79.04 | +0.96 | 2.251 |
| `ftb7b` | last 7 | 2 | 79.20 ± 0.03 | +1.12 | 2.247 |
| `ftb8b` | last 8 | 1 | 78.93 | +0.85 | 2.234 |
| `ftb9b` | last 9 | 1 | 78.94 | +0.86 | 2.227 |
| `ftb10b` | last 10 | 1 | 78.67 | +0.59 | 2.231 |
| `ftb11b` | last 11 | 1 | 78.68 | +0.60 | 2.229 |
| `ftb1h` | proc last 1 (rest random) | 1 | 78.55 | +0.47 | 2.214 |
| `ftb2h` |  | 1 | 78.65 | +0.58 | 2.317 |
| `ftb3h` |  | 1 | 78.89 | +0.81 | 2.300 |
| `ftb4h` |  | 1 | 79.69 | +1.61 | 2.318 |
| `ftb5h` |  | 1 | 78.84 | +0.76 | 2.328 |
| `ftb6h` |  | 1 | 78.72 | +0.64 | 2.337 |
| `ftb7h` |  | 1 | 79.67 | +1.59 | 2.332 |
| `ftb8h` |  | 1 | 79.11 | +1.03 | 2.250 |
| `ftb9h` |  | 1 | 78.82 | +0.74 | 2.247 |
| `ftb10h` |  | 1 | 78.68 | +0.60 | 2.244 |
| `ftb11h` |  | 1 | 79.85 | +1.77 | 2.322 |
| `ftb1e` | proc last 1 downscaled to random rho | 1 | 76.37 | -1.71 | 2.281 |
| `ftb2e` |  | 1 | 76.74 | -1.34 | 2.265 |
| `ftb3es1` | (= a2 seed) | 1 | 78.20 | +0.12 | 2.320 |
| `ftb4e` |  | 1 | 77.30 | -0.78 | 2.282 |
| `ftb5e` |  | 1 | 78.00 | -0.08 | 2.356 |
| `ftb6e` |  | 1 | 78.90 | +0.82 | 2.424 |
| `ftb7e` |  | 1 | 79.39 | +1.31 | 2.459 |
| `ftb8e` |  | 1 | 79.48 | +1.40 | 2.495 |
| `ftb9e` |  | 1 | 80.14 | +2.06 | 2.412 |
| `ftb10e` |  | 1 | 79.73 | +1.65 | 2.417 |
| `pds2` | full proc, last 2 downscaled | 1 | 80.15 | +2.08 | 2.474 |
| `pds3` |  | 1 | 80.05 | +1.97 | 2.467 |
| `pds4` |  | 1 | 79.89 | +1.81 | 2.581 |
| `pds5` |  | 1 | 79.84 | +1.76 | 2.512 |
| `pds12` | all 12 downscaled | 1 | 80.21 | +2.14 | 2.442 |
| `ftb0g` | all 12 downscaled incl. fc1 | 1 | 79.91 | +1.83 | 2.661 |
| `ftb0h` |  | 1 | 80.09 | +2.01 | 2.594 |
| `ftbcomp1` | proc block 0 + blocks 9-11 to rho 0.25 | 3 | 79.98 ± 0.23 | +1.90 | 2.352 |
| `ftbcomp25` | proc 0-3 + 9-11 to rho 0.25 | 3 | 80.16 ± 0.12 | +2.08 | 2.491 |
| `ftbcomp11` | proc 0-10 + block 11 to rho 1.4 | 3 | 80.63 ± 0.18 | +2.55 | 2.583 |
| `ftb4jd` | proc 0-7 + 8-11 rho x0.5 | 3 | 80.11 ± 0.04 | +2.03 | 2.640 |

**G. attention-only suffix (MLP of those blocks random)**

| arm | blocks 0-8 receive (9-11 random unless noted) | n | last-epoch top-1 | vs r | train loss |
|---|---|---|---|---|---|
| `pattn1` | proc attention of last 1 block | 1 | 79.02 | +0.94 | 2.228 |
| `pattn2` |  | 1 | 78.46 | +0.38 | 2.221 |
| `pattn3` |  | 1 | 78.59 | +0.51 | 2.208 |
| `pattn4` |  | 1 | 77.95 | -0.12 | 2.194 |
| `pattn5` |  | 1 | 78.11 | +0.03 | 2.206 |
| `pattn6` |  | 1 | 78.61 | +0.53 | 2.227 |
| `rattn1` | random, rho_attn-matched last 1 | 1 | 78.86 | +0.78 | 2.223 |
| `rattn2` |  | 1 | 78.78 | +0.70 | 2.217 |
| `rattn3` |  | 1 | 79.40 | +1.32 | 2.232 |
| `rattn4` |  | 1 | 79.38 | +1.30 | 2.230 |
| `rattn5` |  | 1 | 79.19 | +1.11 | 2.226 |
| `rattn6` |  | 1 | 79.28 | +1.20 | 2.227 |
| `pattn1d` | proc attention last 1, downscaled | 1 | 77.07 | -1.01 | 2.254 |
| `pattn2d` |  | 1 | 77.45 | -0.63 | 2.274 |
| `pattn3d` |  | 1 | 78.20 | +0.12 | 2.327 |
| `pattn4d` |  | 1 | 76.04 | -2.04 | 2.281 |
| `pattn5d` |  | 1 | 77.32 | -0.75 | 2.283 |
| `pattn6d` |  | 1 | 77.41 | -0.66 | 2.217 |

Three runs on the *old* checkpoint (`pr_27267764`, jobs 29236813-15, July) are excluded as not
comparable (memory note "IN-100 default checkpoint is kdyck4").

### 0d.3 The one regularity that holds across all 103 arms: the final TRAINING loss predicts the test accuracy

`plots/out/fig15_fit_vs_gen.png` (`plots/verify/fit_vs_gen.py`). Final train loss vs final test
top-1 over 103 arms (clean seeds): **Pearson +0.61, Spearman +0.68**. It holds *within* every
family, which is what makes it more than a confound:

| family | n | r(train loss, acc) |
|---|---|---|
| proc values in blocks 0-8 (A) | 8 | +0.73 |
| no checkpoint values, blocks 0-8 rescaled (B) | 11 | +0.66 |
| blocks 0-8 arms A+B together | 19 | +0.75 (Spearman **+0.88**) |
| proc prefix intact (E) | 15 | +0.57 |
| b-series (random + rho-matched last k) | 11 | **+0.88** |
| e-series (proc last k, downscaled) | 9 | **+0.90** |
| rattn-series | 6 | +0.96 |
| late-block families pooled (F) | 47 | +0.50 |

Test loss is linear in train loss (r = -0.85, residual sd 0.052-0.057 over all arms). The arms
more than 2 sd *above* that line -- worse test loss than their fit predicts -- are exactly the
damaged ones: `ftb4n` (+0.19), `pattn4d`, `ftb1e`, `ftb11isfix` (+0.13), `ftb11s`, `ftb2e`.
Everything else, including every winning early-block arm, every late-block recipe arm and the
depth sweeps, sits on one line: **an init helps to the extent that it leaves the network fitting
the augmented training set *worse* at epoch 299, unless it damages the network outright**.

Two readings of the same fact, and they are not the same:

* **When the deficit appears.** All proc-scale early-block arms share the same *slow start*
  (train loss +0.4 to +0.5 above `r` at epoch 9, +0.12 to +0.18 at epoch 49) -- `ftbqm`,
  `ftbnorm`, `ftbqmln` included. In the failing arms the gap closes to <= 0.02 by epoch 99 and
  stays there. In the winning arms it *persists*: `ftbqmlnvo` +0.10, `ftb4e3fix` +0.07,
  `ftbqm1dvo` +0.05 from epoch 99 to 299; `ftb3i` +0.40; `p` +0.24; `ftb4m` +0.20. `ftbvd` has
  *no* slow start (-0.05 at epoch 9) but a persistent +0.05 -> +0.66. The slow start is
  irrelevant; the persistent deficit is what tracks accuracy.
* **Fit part vs frontier part.** Regressing accuracy on train loss over the 65 non-damaged arms
  gives 5.2 pp per nat. Splitting each arm's gain into the part predicted by its train-loss gap
  to `r` and the residual: `ftb3i` +1.91 = +2.06 fit -0.16 residual; `ftb4i` likewise
  (+1.97 / -0.16). The arrangement-destroyed arms are the opposite: `ftbqmlnvo` +1.86 =
  +0.53 fit **+1.33** residual, `ftb4e3fix` +0.38 / **+1.04**, `ftbqm1dvo` +0.27 / +0.77 -- like
  the late-block recipe (`ftbrho` +0.24 / **+1.37**). The intact and the shuffled proc prefix
  reach the same accuracy by different routes: intact proc is a pure regulariser (it fits much
  worse), permuted proc values fit almost as well as random yet generalise better. The failures
  are all frontier losses (`ftbcfg` -0.45, `ftbvu` -0.25, `ftb4o` -0.82, `ftb11isfix` -2.04).

### 0d.4 What proc's early blocks look like, slice by slice

`plots/verify/proc_shape.py`, weight std as a multiple of timm's 0.02 (LN gains = mean):

| block | q | k | v | proj | fc1 | fc2 | gamma1 | gamma2 | attn write (x rand) | MLP write (x rand) | logit (x rand) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.86 | 3.81 | 3.42 | 3.61 | 3.29 | 3.11 | 0.31 | 0.52 | **4.3** | **5.6** | 2.0 |
| 1 | 3.09 | 3.89 | 1.22 | 1.57 | 0.72 | 0.75 | 0.44 | 0.44 | 0.85 | 0.24 | 2.3 |
| 2 | 3.48 | 3.95 | 1.37 | 1.65 | 0.73 | 0.77 | 0.41 | 0.43 | 0.93 | 0.24 | 2.3 |
| 4 | 3.49 | 4.06 | 1.70 | 2.01 | 0.83 | 0.82 | 0.39 | 0.42 | 1.4 | 0.32 | 2.5 |
| 6 | 3.61 | 4.07 | 1.78 | 2.07 | 0.90 | 0.88 | 0.37 | 0.39 | 1.4 | 0.31 | 2.3 |
| 8 | 3.97 | 4.28 | 2.03 | 2.34 | 1.22 | 1.19 | 0.38 | 0.34 | 1.8 | 0.50 | 2.6 |
| 11 | 4.86 | 4.83 | 4.62 | 5.00 | 3.71 | 3.84 | 0.44 | 0.45 | 10 | 6.4 | 4.6 |

(attn write = gamma1 * v * proj, MLP write = gamma2 * fc1 * fc2, logit = gamma1^2 * q * k, all as
multiples of random init; exact LayerNorm-composed values per arm are in §0d.5.)

Three things this table settles:

* **Block 0 is a different regime from blocks 1-8.** Everything in block 0 is 3-4x random, with
  the LN gain only partly compensating: its attention write is 4.3x random and its MLP write
  5.6x. Blocks 1-8 are *quiet*: attention write 0.85-1.8x, MLP write 0.24-0.50x, with attention
  logits 2.3-2.6x sharper. That is the block-0 anomaly in numbers, and it is why `ftbcfg`'s
  blocks-0-8 *average* mis-specified blocks 1-8.
* **The end state of training looks like proc.** At epoch 299 a randomly initialised ViT-B has
  q, k, v, proj, fc1, fc2 all at ~4x std 0.02 and gamma1 ~0.4 in blocks 1-8 (§0d.7). Proc's
  attention *input* scales are where training ends up anyway; proc's *MLP* scale (0.3x) is not.
* **Shape.** Kurtosis of q/k/v/proj in blocks 1-8 is 3.1-4.9 (mild), of fc1/fc2 4.3-11.8
  (heavy, and fc2 is skewed, 0.1-0.4). A Student-t fitted to (norm, kurtosis) -- what
  `--quantile_source parametric` draws -- removes 50-90% of the W1 distance between proc's
  marginal and a Gaussian of the same norm for most slices, but only 7-26% for the skewed
  early-block fc2. **Code gap:** with `--quantile_qkv_mode` other than `pooled` the parametric
  branch is never reached for `attn.qkv.weight` (the slice branch `continue`s first,
  `main.py` ~1078-1113), so the parametric arm as configured on top of `ftbqmlnvo` would give a
  Student-t to proj/fc1/fc2 and proc's *empirical* v. Fix before running it.

The cleanest *existing* test of shape is `ftbqmln` vs `ftbnorm`: identical effective scales at
init (§0d.5: logit 0.065, attention write 0.81, MLP write 0.36 in block 4 for both), proc's exact
marginals in one and Gaussians in the other, 78.18 vs 78.28. **Shape contributed nothing there.**
The 1-D analogue (`ftbqm1dpar` vs `ftbqm1d`, contaminated era, 78.35 vs 78.50) says the same.

### 0d.5 The initial state of every arm, measured on the tensors that were trained

Each arm's init was rebuilt with `plots/dump_init.py` -- main.py's own path, flags copied
verbatim from the run's `Namespace` dump, seed 0, after the rank-0 broadcast -- and measured
(`plots/verify/init_dist_stats.py`, `init_forward_stats.py`). Scales are LayerNorm-composed
(`||W diag(gamma)||`), averaged over blocks 1-8 and given as **multiples of random init**:
logit = gamma1^2 |Wq||Wk|, attn write = gamma1 |Wv||Wproj|, MLP write = gamma2 |Wfc1||Wfc2|. The
forward columns are on 64 val images: rho = ||sublayer output|| / ||stream||, attention entropy
(uniform = 5.28), and the residual-stream norm after block 8 (random: 39).

| arm | vs r | logit | v_eff | attn write | MLP write | gamma1 | block 0: write / MLP | rho_attn 1-8 | rho_mlp 1-8 | entropy 1-8 | stream after b8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ftb3i` | +1.91 | 2.59 | 0.71 | **1.42** | 0.35 | 0.39 | 4.3 / 5.6 | 0.08 | 0.02 | 0.7 | 1420 |
| `ftbqmlnvo` | +1.86 | 1.75 | 0.68 | **1.37** | 0.32 | 0.39 | 4.1 / 5.4 | 0.13 | 0.04 | 5.2 | 75 |
| `ftb4e3fix` | +1.42 | 2.39 | 0.68 | **1.37** | 0.32 | 0.39 | 4.1 / 5.4 | 0.15 | 0.04 | 5.2 | 73 |
| `ftbqks` | running | 2.36 | 0.68 | **1.37** | 0.32 | 0.39 | 4.1 / 5.4 | 0.15 | 0.04 | 5.2 | 73 |
| `ftbqm1dvo` | +1.03 | 1.75 | 0.68 | **1.37** | 0.32 | 0.39 | 4.1 / 5.4 | 0.14 | 0.04 | 5.2 | 73 |
| `ftbqmvo` | +0.53 | 10.69 | 1.69 | **3.41** | 0.79 | 1.00 | 12.4 / 10.2 | 0.15 | 0.05 | 3.4 | 162 |
| `ftbnorm` | +0.20 | 1.75 | 1.32 | **2.63** | 0.32 | 0.41 | 4.5 / 5.4 | 0.24 | 0.04 | 5.2 | 84 |
| `ftbqmln` | +0.10 | 1.75 | 1.32 | **2.62** | 0.32 | 0.39 | 4.5 / 5.4 | 0.25 | 0.04 | 5.2 | 88 |
| `ftbqm` | +0.08 | 10.69 | 3.26 | **6.53** | 0.79 | 1.00 | 13.4 / 10.2 | 0.26 | 0.04 | 3.5 | 196 |
| `ftbqm1d` | -0.07 | 1.75 | 1.32 | **2.63** | 0.32 | 0.39 | 4.5 / 5.4 | 0.26 | 0.04 | 5.2 | 87 |
| `ftb4m` | +1.92 | 2.35 | 0.29 | **0.22** | 0.18 | 0.47 | 0.3 / 0.1 | 0.19 | 0.34 | 2.9 | 105 |
| `r` | +0.00 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.0 / 1.0 | 0.19 | 0.36 | 5.3 | 39 |
| `ftbvd` | +0.66 | 1.00 | 0.46 | **0.46** | 1.00 | 1.00 | 0.5 / 1.0 | 0.08 | 0.38 | 5.3 | 36 |
| `ftbslice` | +0.49 | 14.40 | 1.74 | **1.74** | 1.00 | 1.00 | 1.7 / 1.0 | 0.34 | 0.28 | 2.5 | 50 |
| `ftbqu` | -0.03 | 4.74 | 1.00 | **1.00** | 1.00 | 1.00 | 1.0 / 1.0 | 0.20 | 0.35 | 4.7 | 39 |
| `ftbclip01` | -0.32 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 1.0 / 1.0 | 0.19 | 0.36 | 5.3 | 39 |
| `ftbvu` | -0.43 | 1.00 | 1.74 | **1.74** | 1.00 | 1.00 | 1.7 / 1.0 | 0.33 | 0.30 | 5.3 | 47 |
| `ftbcfg` | -0.57 | 2.52 | 0.73 | **1.75** | 1.00 | 1.00 | 1.7 / 1.0 | 0.33 | 0.30 | 5.2 | 47 |
| `ftb4o` | -0.80 | 1.00 | 3.59 | **13.91** | 2.05 | 1.00 | 12.2 / 135.4 | 0.07 | 0.01 | 5.3 | 1431 |
| `ftb11i` | +0.70 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 4.3 / 5.6 | 0.01 | 0.01 | 5.3 | 1462 |
| `ftb11d` | -0.37 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 0.3 / 0.1 | 0.22 | 0.36 | 5.3 | 39 |
| `ftb11s` | -1.49 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 12.2 / 135.4 | 0.01 | 0.01 | 5.3 | 1391 |
| `ftb11isfix` | -1.72 | 1.00 | 1.00 | **1.00** | 1.00 | 1.00 | 4.5 / 5.4 | 0.10 | 0.14 | 5.3 | 74 |

What the table shows, read row against row:

* **`ftbqmlnvo` (+1.86) and `ftbqmln` (+0.10) differ in exactly one number.** Logit scale 1.75
  vs 1.75, MLP write 0.32 vs 0.32, LayerNorm gains 0.39 vs 0.39, block 0 identical, attention
  entropy 5.2 vs 5.2, rho_mlp 0.04 vs 0.04 -- and attention write **1.37 vs 2.63** (v_eff 0.68
  vs 1.32; forward rho_attn 0.13 vs 0.25). That one difference is worth +1.75. The same pair at
  gamma = 1 (`ftbqmvo` 3.41 vs `ftbqm` 6.53) is worth +0.45, and on a plain Gaussian net
  (`ftbvd` 0.46 / `r` 1.00 / `ftbvu` 1.74) it is +0.66 / 0 / -0.43. **Within every family the
  attention write of blocks 1-8 orders accuracy, and lower is better.** The three winning
  permuted-proc arms have rho_attn *below* random's (0.13-0.15 vs 0.19); the three failing ones
  have it above (0.24-0.26).
* **Across families the level differs by a point and a half at the same write.** Write 1.37 on
  the permuted-proc profile gives +1.86; write 1.75 on a Gaussian net gives -0.5. The profile
  differs in MLP write (0.32 vs 1.0), LayerNorm gains (0.4 vs 1.0), the block-0 regime (4 / 5
  vs 1 / 1) and the marginal shape. Which of these sets the level is *not* determined by any
  existing arm -- see §0d.8.
* **`ftbnorm` = `ftbqmln` to two decimals in every scale**, Gaussian vs proc's heavy tails, and
  78.28 vs 78.18. The shape of the marginal did nothing there.
* **`ftbcfg` went the wrong way.** Its attention write is 1.75 (up, like `ftbvu`), its MLP write
  1.0, its block 0 ordinary. It sits in the worst cell of the Gaussian family and scored like
  `ftbvu`. It does not test whether proc's *values* are needed.
* **Intact proc is a different regime, not a stronger version of the same one.** `ftb3i`, `ftb4i`,
  `p`: block 0's MLP output is **29x** the incoming stream (weight norms predict 5.6x -- the
  arrangement makes it coherent), the stream leaves block 8 at 1420 (36x random), and attention
  in blocks 1-8 is near one-hot (entropy 0.7). Its write can be anything from 0.10 (`ftb0h`) to
  1.42 (`ftb3i`) at +1.8 to +2.0. It reaches the same accuracy as the permuted arms by
  under-fitting (§0d.3), not by the same route.
* **Every damaged arm reproduces block 0's explosion with random content.** `ftb4o`, `ftb11s`,
  `ftb4n`: rho-matching a random block 0 to intact proc gives write 12x and MLP write **135x**
  (stream 1400 after block 8) -- that is what rho-matching to proc's *intact* profile means, and
  it is why every rho-matched early arm lost. `ftb11isfix` (permuted block 0, random 1-11) is
  loud (4.5 / 5.4, stream 74) with nothing behind it and loses 1.72; the same block 0 followed by
  the quiet permuted blocks 1-8 (`ftb4e3fix`) gains 1.42. The block-0 anomaly is now a
  quantified interaction between a loud random block 0 and what follows it, still unexplained.

### 0d.6 The end of training

`plots/verify/end_state_stats.py` on the epoch-299 weights of all 201 completed runs, seed-mean;
same conventions, now relative to **random init's end state** (`plots/out/fig16_end_state.png`).

| arm | vs r | logit | attn write | MLP write | gamma1 | gamma2 | gamma2 blocks 6-8 | rho_mlp 6-8 | entropy 9-11 | train loss |
|---|---|---|---|---|---|---|---|---|---|---|
| `ftbcomp11` | +2.55 | 0.66 | 0.46 | **0.55** | 0.24 | 0.35 | **0.39** | 0.37 | 3.89 | 2.583 |
| `ftb1i` | +2.29 | 0.73 | 0.51 | **0.60** | 0.26 | 0.39 | **0.40** | 0.37 | 3.84 | 2.563 |
| `p` | +2.01 | 0.75 | 0.56 | **0.61** | 0.27 | 0.39 | **0.42** | 0.40 | 3.75 | 2.469 |
| `ftb3i` | +1.91 | 0.70 | 0.47 | **0.60** | 0.25 | 0.39 | **0.44** | 0.34 | 3.76 | 2.626 |
| `ftb4m` | +1.92 | 0.82 | 0.56 | **0.62** | 0.28 | 0.40 | **0.46** | 0.44 | 3.68 | 2.426 |
| `ftbqmlnvo` | +1.86 | 0.75 | 0.78 | **0.42** | 0.31 | 0.37 | **0.42** | 0.42 | 3.71 | 2.328 |
| `ftb4e3fix` | +1.42 | 0.76 | 0.79 | **0.45** | 0.31 | 0.40 | **0.49** | 0.42 | 3.50 | 2.299 |
| `ftbqm1dvo` | +1.03 | 0.77 | 0.80 | **0.49** | 0.31 | 0.43 | **0.56** | 0.43 | 3.46 | 2.277 |
| `ftbqmvo` | +0.53 | 1.21 | 0.92 | **0.75** | 0.39 | 0.63 | **0.83** | 0.45 | 3.36 | 2.271 |
| `ftbnorm` | +0.20 | 0.74 | 0.81 | **0.53** | 0.31 | 0.45 | **0.61** | 0.44 | 3.32 | 2.257 |
| `ftbqmln` | +0.10 | 0.75 | 0.82 | **0.62** | 0.32 | 0.50 | **0.76** | 0.46 | 3.26 | 2.231 |
| `ftbqm1d` | -0.07 | 0.74 | 0.81 | **0.62** | 0.31 | 0.51 | **0.77** | 0.46 | 3.33 | 2.224 |
| `ftbqm` | +0.08 | 1.28 | 1.00 | **0.85** | 0.41 | 0.71 | **1.02** | 0.46 | 3.27 | 2.247 |
| `r` | -0.00 | 1.00 | 1.00 | **1.00** | 0.38 | 0.80 | **1.38** | 0.42 | 3.26 | 2.225 |
| `ftbvd` | +0.66 | 0.97 | 0.99 | **1.08** | 0.37 | 0.85 | **1.55** | 0.45 | 3.53 | 2.284 |
| `ftbslice` | +0.49 | 1.16 | 0.95 | **0.78** | 0.39 | 0.64 | **0.92** | 0.45 | 3.34 | 2.239 |
| `ftbcfg` | -0.57 | 0.97 | 0.99 | **1.02** | 0.37 | 0.82 | **1.46** | 0.42 | 3.12 | 2.201 |
| `ftbvu` | -0.43 | 1.01 | 1.00 | **1.02** | 0.38 | 0.82 | **1.43** | 0.41 | 3.01 | 2.192 |
| `ftbrho` | +1.61 | 1.20 | 0.94 | **0.74** | 0.39 | 0.62 | **0.84** | 0.49 | 3.45 | 2.271 |
| `ftb3b` | +1.92 | 1.11 | 0.93 | **0.80** | 0.38 | 0.67 | **1.02** | 0.50 | 3.29 | 2.292 |
| `ftb4b` | +1.94 | 1.07 | 0.91 | **0.99** | 0.37 | 0.79 | **1.34** | 0.83 | 3.43 | 2.356 |
| `ftb11i` | +0.70 | 1.30 | 0.94 | **0.75** | 0.40 | 0.63 | **0.79** | 0.44 | 3.31 | 2.274 |
| `ftb11isfix` | -1.72 | 1.05 | 1.03 | **1.01** | 0.39 | 0.80 | **1.29** | 0.38 | 3.30 | 2.286 |
| `ftb4o` | -0.80 | 1.60 | 1.14 | **0.97** | 0.47 | 0.82 | **1.20** | 0.41 | 3.51 | 2.228 |

* **Training erases the Gaussian arms' init scales.** `ftbvd`, `ftbvu`, `ftbcfg`, `ftbqu` all end
  within 3% of `r` in every scale and gain, yet span 1.2 pp. Their effect is on the trajectory,
  not on where the weights end up.
* **Random init ends up looking like proc's attention side.** `r` at epoch 299 has q, k, v, proj
  at ~4x their init std and gamma1 ~0.4 in blocks 1-8 -- proc's early-block *attention* scales
  are the ones training converges to anyway. Its MLP side does not: proc's early MLPs are 0.3x
  and stay there.
* **Within the permuted-proc family the winners end with quieter MLPs and nothing else.**
  Attention scales are identical (logit 0.75, write 0.80 in all six); MLP write ends at
  0.42-0.49 in the three winners and 0.53-0.62 in the three losers; gamma2 over blocks 6-8 at
  0.42-0.56 vs 0.61-0.77 (seed-stable, all three seeds each). A v-slice difference at init turns
  into an MLP difference at the end.
* **Across all 104 arms, the end-state gamma2 of blocks 6-8 is the strongest single end-state
  correlate of accuracy** (Spearman -0.62, p = 2e-12; gamma1 of blocks 0-8: -0.53; late-block
  attention entropy: +0.51). It is a *signature of the proc-prefix route*, not a requirement:
  the b-series reaches +1.9 with blocks 6-8 as loud as random (`ftb3b` 1.02, `ftb4b` 1.34).
  There are two routes to +1.9 -- a proc-valued prefix (quiet middle MLPs, low gamma1) and a
  calibrated suffix (loud calibrated blocks) -- and they share only the fit deficit and a more
  diffuse late attention.

### 0d.7 What the early blocks are doing, as far as the data go

1. **The benefit is regularisation-shaped.** Every family sits on one train-loss / test-accuracy
   line; the winning early-block inits leave a persistent +0.05 to +0.10 nat fit deficit from
   epoch 100 on (intact proc: +0.40), and the failures either close the deficit (proc-scale arms
   with a wide v) or damage the net outright (loud random block 0).
2. **Arrangement is not needed** (`ftb4e3fix`, `ftbqmlnvo` within resolution of `ftb3i`), but it
   is what produces the one-hot attention, the 36x stream and the heavy under-fitting of the
   intact arms. Two routes, one endpoint.
3. **Shape is not needed where it was tested** (`ftbqmln` = `ftbnorm`; `ftbqm1dpar` = `ftbqm1d`).
   The heavy-tail hypothesis has no supporting arm.
4. **Given proc's per-block scale profile, the attention write of blocks 1-8 is the switch**:
   below random's (v_eff ~0.7, write ~1.4x) gives the full +1.86; above it (pooled v, 2.6x) gives
   nothing. Lower write helps in every family, in the same direction, by a family-dependent
   amount.
5. **Nothing has yet been run that carries proc's scale profile without proc's values**: `ftbnorm`
   had the profile but the wide pooled v (the losing cell); `ftbcfg` had the wrong write, no MLP
   scaling, no gains, no block 0. The claim "proc's values are necessary" rests on these two arms
   and is therefore untested. The claim "the write magnitude is sufficient" (§0) is false as
   stated (`ftbcfg`, `ftbslice`) and true only within a family.
6. **The end state says the MLP side is where the winners differ**, which the init-time analysis
   could not see (MLP write is 0.32 in winners and losers alike at init). How a narrower v at init
   ends in quieter MLPs at epoch 299 is the mechanistic question that remains.

### 0d.8 The decisive experiments, in order of information per GPU-day

1. **`ftbnorm` with sliced qkv** -- Gaussian directions, proc's per-block norms for q, k, v
   *separately*, proj, fc1, fc2, LayerNorm gains at proc's RMS (and its LN biases). This is the
   exact Gaussian twin of `ftbqmlnvo` (§0d.5: every scale equal, shape Gaussian). Needs a
   `--norm_match_qkv_mode` in `match_target_block_norms` (~15 lines). ~+1.8 => the early half is
   a per-block scale profile and checkpoint-free; ~+0.2 => something in the values matters that
   `ftbqmln` vs `ftbnorm` did not expose, and the Student-t arm becomes the next question.
2. **rho-match a random net to the *permuted* proc profile**: `upscale_random_match_delta_norms`
   on blocks 0-8 with `--target_model_weight_shuffle` (the flag exists, never used in any run).
   Targets rho_attn ~0.14, rho_mlp ~0.04, block 0 at 1.5 / 2.5 instead of intact proc's
   3.9 / 29 -- i.e. the forward profile of `ftbqmlnvo`, three numbers per block, no values.
3. **`ftbqmlnvo` with gamma1 only vs gamma2 only** (`quantile_1d_mode` split): separates the
   attention-side from the MLP-side gain, which §0d.6 says is where the winners differ.
4. **`ftbvd` sweep** (v x0.3 / x0.6 / x0.46 with fc2 x0.3): the only checkpoint-free arm above
   resolution, never followed up; tells whether the Gaussian family can be pushed toward +1.
5. **Parametric (Student-t) values on top of `ftbqmlnvo`** -- only after fixing the code gap in
   §0d.4, and only if (1) fails.

**Queued 2026-09-05 (after pre-launch verification, `plots/verify/verify_new_arms.py`):**

| arm | SLURM_ID | construction | verified at init |
|---|---|---|---|
| `ftbqmlnvog` | 29538122 (s0-s2) | `ftbqmlnvo` with `--quantile_source gaussian`: every 2-D slice replaced by a Gaussian with the slice's mean and norm (v sliced, permuted LN gains/biases, zero linear biases), main.py `make_donor` | all q/k/v/proj/fc1/fc2 norms and logit / attn-write / MLP-write scales within 0.2% of `ftbqmlnvo_s0`; kurtosis 3.00 in every slice (was 3.6-11.8); identical LayerNorm value multisets; forward profile identical (rho_attn 1.42 / 0.09-0.18, rho_mlp 2.67 / 0.03-0.06, entropy 5.2, stream 75) |
| `ftbrhos` | 29538140 (s0-s2) | random net, `upscale_random_match_delta_norms` on blocks 0-8 against a target = proc with blocks 0-8 permuted (`--target_model_weight_shuffle`, the 13 `ftb4e3` attributes); only v, proj, fc2 rescaled, gamma = 1, q/k/fc1 at random scale | rho profile within 0.025 (attn) / 0.007 (mlp) of `ftb4e3fix_s0` at every block: block 0 at 1.63 / 2.34, blocks 1-8 at 0.09-0.19 / 0.03-0.07, stream 73; entropy uniform (5.26); factors v,proj x2.28 and fc2 x5.76 at block 0, v,proj x0.97-1.40 and fc2 x0.21-0.47 in blocks 1-8; kurtosis 3.00 |

Registered expectations (final, last-epoch, n = 3, resolution 0.45): if the early half is a scale
profile, `ftbqmlnvog` ~ 79.9 (= `ftbqmlnvo`); if proc's values matter beyond second moments,
~ 78.3 (= `ftbnorm`). `ftbrhos` ~ 79.5 if two write budgets per block suffice (= `ftb4e3fix`,
whose profile it copies); ~ 78.7 (= `ftbvd`) if the input-side scales (sharper logits, quiet MLP
pre-activations, gains) are needed as well; below random if a loud random block 0 in front of
quiet random blocks is what `ftb11isfix` says it is. Per §0b, no reading before epoch ~270.

The verification dumps used `results/init_dumps/imnet_small` (val images symlinked as the training
root) to skip the 1.28M-file NFS index, which was crawling (7.5 s per directory listing) at the
time; the rho-matching only needs 5000 reference images, and its train-vs-val agreement is ~3% in
blocks 1-8 (`tmp_ftb4o_s0/target_res_stats.json` vs the val forward pass). The queued runs use the
real training set as every rho arm did.

Dequeued the same day as no longer informative: the parked clean re-runs of `ftbqm1dv` (29529157-9)
and `ftbqm1dqk` (29529160-2); their checkpoints are kept and they are blacklisted in
`sweep_stalled.py`. `ftb11e` (SLURM_ID 29520494, at epoch 242) was dequeued and then re-queued the
same day to finish the e-sweep. The three `ftbqks` resumes (285 / 277 / 274 epochs done) stay queued.

**Paused 2026-09-06 08:45.** The ImageNet NFS server (`dlc-hdd1:/hdd1/datasets`) crawled from
2026-09-05 ~10:00 through the night (3 s per directory listing; first batch of an epoch waited
284 s); the six new-arm jobs got nodes at 02:49-04:20 but reached only epochs 1-5 by 08:40 (3 h to
epoch 0, then 25-90 min per epoch, ~2 epochs/h across the six), while burning full GPU-hours and
fair-share. All ten jobs (six new arms, three `ftbqks` resumes, `ftb11e`) were cancelled; every run
sits at a checkpoint (`ftbqmlnvog` s0/s1/s2 at 3/3/5, `ftbrhos` at 3/1/2, `ftb11e` 248, `ftbqks`
285/277/274). **To resume once the NFS is healthy** (a class-directory listing under ~0.1 s):
`python sweep_stalled.py --submit` resubmits every incomplete run with a fitted time limit, or per
run `sbatch --requeue --export=SLURM_ID=<id>,SEED=<s> vitbase_runs/run_train_<arm>.sh`
(IDs: `ftbqmlnvog` 29538122, `ftbrhos` 29538140, `ftb11e` 29520494, `ftbqks` 29535764). If NFS
stalls recur, the durable fix is a node-local copy of ImageNet (`localtmp` gres is 3.6 TB on the H200
nodes) staged at job start.

**SSD copy done 2026-09-06 13:08.** Workspace `imagenet` on the SSD filesystem (`ws_allocate -F dlcsmall
imagenet 40`; `/work/dlcsmall2/schrodi-imagenet`; expires 2026-10-16, extend with `ws_extend imagenet 40`,
3 extensions left). Copied with `vitbase_runs/copy_imagenet_to_dlcsmall_cpu.sbatch` (64 parallel
per-class rsyncs on a CPU node, 4 h 06 at ~12 MB/s -- the HDD server, not the client, was the
limit: 16 and 64 streams gave the same rate). Verified: 1000 classes, 1,281,167 train files,
50,000 val JPEGs + the source's stray `valprep.sh` (17 rsync temp files from the cancelled first copy attempts were removed; their real files match the source), 200 random
files size-identical and 5 md5-identical to the source. Throughput measured from the login node:
SSD share 1585 files/s / 237 MB/s single-stream (during the copy) vs HDD share 9 files/s that day.
Expected effect on training (from the Sep 1-4 healthy-HDD logs: mean 0.13 s/it, median 0.10,
compute 0.085; first epoch after every (re)start 20-40 min): ~9 -> ~12 epochs/h steady state and
~5 min instead of 20-40 min cold starts, i.e. ~35 h -> ~25 h per 300-epoch run, and immunity to
the HDD server. Node-local NVMe (2.5 GB/s) adds nothing in steady state (decode-bound) and costs
160 GB of staging per start. `run_train_{ftbqmlnvog,ftbrhos,ftbqks,ftb11e}.sh` now use
`--data_path /work/dlcsmall2/schrodi-imagenet`; new scripts should too (identical class/file sets,
so resuming a run from the copy is equivalent). **Resumed 2026-09-06 19:55** (`sweep_stalled.py --submit --max-submit 10`): `ftbqmlnvog` jobs 29539557-9, `ftbrhos` 29539560-2, `ftb11e` 29539556 (9.5 h), `ftbqks` 29539553-5 (4-6 h), all reading the SSD copy.

Not worth running now: more shuffle variants (`ftbqks` will land ~79.4 and changes nothing), more
rho-matched-to-intact-proc arms (all reproduce the block-0 explosion), LayerScale baselines
before (1) is known.

### 0d.9 Experimental setup of the two decisive arms, and the follow-ups that hang on them

**Common protocol** (identical to every other arm in §0d.2). ViT-B/16 (`vit_base`, 12 blocks, d = 768,
12 heads, MLP 3072), ImageNet-1k train (1.28M) → val top-1 (50k). 300 epochs, 50 warm-up, lr 2e-3
cosine to 1e-6, batch 4096 as 4 × H200 × 128 × `update_freq` 8, AdamW (β 0.9/0.999, wd 0.05 on 2-D
weights only — biases and LayerNorm params are not decayed, `optim_factory.py:73`), mixup 0.8 /
cutmix 1.0, RandAugment m9, label smoothing 0.1, random erasing 0.25, drop-path 0, AMP. Seeds 0-2,
`seed = args.seed + rank` for the random init, DDP with the rank-0 broadcast after all init edits
(§0a). Readout: **last-epoch (299) top-1**, never max-over-epochs; the training loss at 299 and the
end-state statistics (§0d.6) are recorded for both arms because the mechanism claims are about
them. No reading before epoch ~270 (§0b). Results dirs `results/imnet_base/results_IMNET_BASE_<SLURM_ID>/s<seed>`,
seed-0 init dumps `results/init_dumps/ftbqmlnvog_s0fast.pth`, `ftbrhos_s0fast.pth`.

**Reference points** (n = 3 unless noted): `r` 78.08, `ftbqmlnvo` 79.93, `ftb4e3fix` 79.50,
`ftbqm1dvo` 79.11, `ftbvd` 78.74, `ftbnorm` 78.28, `ftbqmln` 78.18, `ftbcfg` 77.50; resolution 0.45.

#### Arm 1 — `ftbqmlnvog`, SLURM_ID 29538122, `vitbase_runs/run_train_ftbqmlnvog.sh`

```
--initialize results/pr_vitb_n/pr_6066174_final.pth --skip_norm true
--init_method quantile_match_target_blocks --init_method_scaled_blocks 0,1,2,3,4,5,6,7,8
--quantile_source gaussian --quantile_qkv_mode v_only --quantile_1d_mode layernorm
```
(`run_train_ftbqmlnvo.sh` with `--quantile_source gaussian` added; the script diff is that line, the
job name and the wandb note.)

What `main.py` does with it, in order (`quantile_match_target_blocks`, ~line 1001):

1. `model` = timm random init of all 12 blocks (trunc-normal std 0.02, LayerNorm gains 1, biases 0),
   DDP-wrapped. Blocks 9-11 are never touched, so they are random by construction (no
   `--random_blocks` needed).
2. `target` = the procedural checkpoint loaded into a second model; `cls_token`, `pos_embed`,
   `patch_embed.*`, `head.*` and (via `--skip_norm`) the final norm are dropped.
3. For every 2-D weight of blocks 0-8, `make_donor` builds the values written into the random
   tensor **by rank map** (sorted donor → the random tensor's sort order, so the arrangement is the
   random tensor's): `attn.qkv.weight` rows `[2e:3e]` (v) get a Gaussian with **proc's v-slice mean
   and norm**; rows `[0:2e]` (q, k) get a Gaussian with the **pooled q+k+v** mean and norm, exactly
   as the pooled path of `ftbqmlnvo` gives them; `attn.proj`, `mlp.fc1`, `mlp.fc2` get Gaussians with
   proc's tensor mean and norm. The donor is rescaled to the target norm exactly, so every norm is
   proc's to float precision.
4. 1-D: `norm1.{weight,bias}` and `norm2.{weight,bias}` of blocks 0-8 receive **proc's values in a
   uniform random permutation** (`quantile_1d_mode layernorm`); `attn.qkv.bias`, `attn.proj.bias`,
   `mlp.fc*.bias` stay at zero (as in `ftbqmlnvo`).
5. Rank-0 broadcast of the whole state dict; training starts.

The **only** difference to `ftbqmlnvo` is the marginal shape of the 2-D weights: Gaussian
(kurtosis 3.00, no skew) instead of proc's (kurtosis 3.6-11.8, fc2 skew up to 0.4). Verified on the
rebuilt init (§0d.8 table): all slice norms and composed scales within 0.2%, identical LayerNorm value
multisets, zero biases, identical forward profile.

#### Arm 2 — `ftbrhos`, SLURM_ID 29538140, `vitbase_runs/run_train_ftbrhos.sh`

```
--initialize results/pr_vitb_n/pr_6066174_final.pth --skip_norm true --random_blocks ""
--init_method upscale_random_match_delta_norms --init_method_scaled_blocks 0,1,2,3,4,5,6,7,8
--target_model_weight_shuffle "0[norm1.weight,norm1.bias,attn.qk.weight,attn.v.weight,attn.qkv.bias,
    attn.proj.weight,attn.proj.bias,norm2.weight,norm2.bias,mlp.fc1.weight,mlp.fc1.bias,
    mlp.fc2.weight,mlp.fc2.bias];1[...same...];...;8[...same...]"
```
(`init_method_scaled_attributes` at its default `v,proj,fc2`.)

What `main.py` does with it, in order (`upscale_random_*` branch ~line 913, target shuffle ~1385,
matching ~1672):

1. `model` = timm random init, all 12 blocks, DDP-wrapped. Nothing from the checkpoint is ever
   loaded into it.
2. `target` = the checkpoint loaded into a second model (all 12 blocks), then blocks 0-8 permuted
   entrywise within each of the 13 listed tensors — `attn.qk.weight` as one pool, `attn.v.weight`
   separately, LayerNorm gains/biases and all biases permuted too — i.e. **the `ftb4e3fix`
   construction**. Its blocks 9-11 are intact proc but do not enter any target for blocks 0-8.
3. ρ targets: one forward pass of the target over **5000 random training images** (train transforms)
   gives, per block b in 0-8, `norm_ratio_attn_delta_in_mean` = mean over tokens of
   ‖attn(norm1(x))‖ / ‖x‖ and the MLP analogue on the post-attention stream (`engine.attention_residual_analysis`).
4. Sequential matching, block 0 → 8: measure the model's current ρ_attn(b) on the same 5000
   images, multiply `v` and `proj` (weight and, if `--init_method_bias_scaling`, bias — off here)
   each by (target/current)^(1/2), so the attention write moves by target/current; re-measure,
   then multiply `fc2` by target/current for ρ_mlp(b). Blocks are matched in order because block b's
   stream depends on the rescaled blocks before it. Targets are measured on rank 0 and broadcast.
5. Rank-0 broadcast; training starts.

What the init carries: **two numbers per block** (the attention and MLP write budgets of the
permuted-proc profile), Gaussian weights, γ = 1, q/k/fc1 at random scale — so attention is uniform
at init (entropy 5.26) and MLP pre-activations are at random scale. On the verification dump the
factors were block 0: v, proj ×2.28 (write ×5.2), fc2 ×5.76; blocks 1-8: v, proj ×0.97-1.40, fc2
×0.21-0.47; the resulting profile matched `ftb4e3fix` to 0.025 (attn) / 0.007 (mlp) at every block.
In the queued runs the 5000 reference images are drawn from the real training set with the run's
seed, so the factors vary slightly across seeds, as in every ρ arm before.

#### What each outcome decides, and the follow-up it triggers

| | `ftbrhos` ≈ 79.5 (budgets suffice) | `ftbrhos` ≈ 78.7 or below |
|---|---|---|
| **`ftbqmlnvog` ≈ 79.9** (shape irrelevant) | **A.** The early half is checkpoint-free: two write budgets per block. Next: (a) one checkpoint-free init for all 12 blocks — the permuted-proc ρ profile for 0-8 plus the late-block recipe for 9-11 — 3 seeds, the paper's method; (b) block-0 ablation: `ftbrhos` with block 0 left at random budgets (is the loud block 0 needed, or merely harmless?); (c) profile ablations: flatten the ramp, halve/double all budgets; (d) the LayerScale / ReZero baseline, now the natural competitor; (e) can the ρ profile be written down without a checkpoint (it is ~0.15 / 0.04 in 1-8 with a 1.6 / 2.4 block 0)? | **B.** The scale profile suffices but the input-side scales matter (sharper logits 2.4×, quiet MLP pre-activations 0.36×, gains 0.4). Next: (a) `ftbrhos` with the LayerNorm gains at proc's RMS (`norm1,norm2` added to the scaled attributes, or a gain init of 0.4) and, separately, with q/k ×1.6; (b) the γ1-only / γ2-only split of `ftbqmlnvo` to say which side; then A(a)-(e). |
| **`ftbqmlnvog` ≈ 78.3** (values needed beyond 2nd moments) | **C.** Contradiction: budgets suffice with Gaussian weights but the Gaussian rank-map does not. First re-verify both dumps against each other (they share every measured scale except the pooled-qk logit 1.75 vs 1.0 and the MLP pre-activation 0.36 vs 1.0 — which would make the *sharper logits* or the *quieter MLP pre-activations* harmful, not helpful); then `ftbrhos` + gains as in B(a). | **D.** Proc's values matter beyond second moments given the profile. Next: (a) the Student-t twin (`--quantile_source parametric`, now applied per slice) — 4 moments; (b) if that fails, per-block marginal swaps (block b's empirical marginal written into block b′) and the fc2 skew specifically; (c) `ftbsv` / `ftbsb` (built, verified, Gaussian-shaped with proc's biases) as the rank / singular-vector controls, read against `ftbqm1dvo`. |

Independent of the cell: `ftb4m` to n = 3 (the single-seed "intact proc is insensitive to the
write" point that the scale story leans on), and `ftbqks` closes the q/k-asymmetry question when its
resumes finish.

## 0b. Can a short run act as a proxy? Yes -- but NOT the obvious one (2026-08-31)

> **THE FIT BELOW HAS FAILED TWICE — 2.4 sigma and 4.2 sigma, in OPPOSITE directions. Do not use
> it. See §0c.4.** It predicted `ftbqmlnvo` at 78.66 (measured **79.93**) and `ftb11isfix` at 78.36
> (measured **76.15**). The cause is structural, not noise: the line is fitted across
> **two trajectory families** (§0c.2) and therefore encodes "slow start → good finish", so it
> misprices every fast starter. Treat the remaining §0b.1 predictions — `ftb4e3fix` 78.58 and
> `ftbcfg` 78.24 especially — as weak evidence, and do not apply the ep-49 proxy to the `ftbqm*`
> family at all.
>
> The framing sentence below (*"every arm that ends ahead starts far behind"*) is also false as
> stated: it is read off `ftb4e3`, the contaminated arm of §0a, which is 9.4 points down at ep49.
> `ftb4e3fix` is ~2 points down and starts in the FAST family. Compare within a family and read
> later (~ep 164); §0c.4 has the calibration.

Motivating question: could a 10-epoch run screen arms and save the ~21 h each full run costs?

**A naive short screen is worse than useless here.** Early test accuracy is *anti*-correlated with
final accuracy, because every arm that ends ahead starts far behind and crosses late (3.10.9.8:
`ftb4e3` is 9.4 points down at epoch 49 and does not overtake until ~214).

| epoch | Pearson r of test acc with FINAL test acc |
|---|---|
| 4 | **-0.46** |
| 24 | **-0.77** |
| 49 | **-0.81** (strongest anti-correlation) |
| 149 | -0.51 |
| 174 | -0.09 (crossover) |
| 224 | +0.92 |
| 249 | +0.99 |

Spearman with the final ranking only reaches 0.9 at **epoch 214** -- 71% of the run. Ranking 24
arms by accuracy at epoch 9 gives a top-5 of `ftbrho, ftbvd, ftbclip5, r, ftbclip1` against the
true `ftbcomp11, ftb1i, ftb4e3, p, ftbrho`: **1/5 overlap**, and it promotes the random baseline.

**What does work: train loss at epoch 49.** It is a DIRECT relationship (higher training loss ->
better final accuracy, the regulariser story of 3.13.x), so there is no sign to remember.

| proxy | \|spearman\| with final | direction | epochs |
|---|---|---|---|
| **train loss @ ep 49** | **0.80** | direct | **50** |
| train loss @ ep 149 | 0.83 | direct | 150 |
| train loss @ ep 24 | 0.74 | direct | 25 |
| test acc @ ep 49 | 0.75 | INVERTED | 50 |
| train loss @ ep 9 | 0.58 | direct | 10 |
| test acc @ ep 9 | 0.54 | INVERTED | 10 |
| init `value_write`, no training | 0.44 | inverted | 0 |

Fitted on the 24 arms with a complete 300-epoch run:

```
final_top1 = 3.218 * train_loss@49 + 66.163
r = +0.834   R^2 = 0.696   residual sd = 0.53 pp
```

**Use it to screen, never to decide.** The effects here are 0.4-2.1 pp against a 0.41 pp
resolution, and a residual sd of 0.53 pp will reorder anything closely spaced. Kill clear
losers, promote clear winners, run the survivors to 300. At 50 epochs a screen costs ~3.6 h/seed
against ~21 h.

Two traps. **Do not screen in the middle**: test-accuracy rho collapses to 0.13 at epoch 149 as
the curves cross, so mid-run is the worst place to look. And the zero-training `value_write`
proxy scores only 0.44 here against the R^2 = 0.83 of section 0 -- that figure was fitted within one
arm family sharing a base, and it does not generalise across families. Section 0's scope caveat is
the same caveat.

### 0b.1 Registered predictions (made 2026-08-31, before these runs finished)

Each prediction uses only that arm's OWN epoch-49 train loss through the equation above. None of
these arms had finished when the predictions were written, so this is a genuine out-of-sample test
of the proxy -- check them off as the runs land.

| arm | train_loss@49 | seeds | predicted final | 95% interval | epochs at registration |
|---|---|---|---|---|---|
| `ftb4i` | 4.4023 | 2 | **80.33** | [79.29, 81.37] | [300, 28, 238] |
| `ftb11e` | 3.9889 | 1 | **79.00** | [77.96, 80.04] | [100, 0, 0] |
| `ftb9e` | 3.9716 | 1 | **78.94** | [77.90, 79.99] | [298, 0, 0] |
| `ftb10e` | 3.9633 | 1 | **78.92** | [77.87, 79.96] | [52, 0, 0] |
| `ftb7e` | 3.9231 | 1 | **78.79** | [77.75, 79.83] | [278, 0, 0] |
| `ftbqmvo` | 3.8982 | 3 | **78.71** | [77.67, 79.75] | [281, 300, 300] |
| `ftbqmlnvo` | 3.8849 | 3 | **78.66** | [77.62, 79.71] | [242, 248, 233] |
| `ftb11b` | 3.8704 | 1 | **78.62** | [77.58, 79.66] | [172, 0, 0] |
| `ftb4e3fix` | 3.8577 | 3 | **78.58** | [77.53, 79.62] | [100, 165, 159] |
| `ftb10b` | 3.8169 | 1 | **78.45** | [77.40, 79.49] | [198, 0, 0] |
| `ftb11isfix` | 3.7899 | 3 | **78.36** | [77.32, 79.40] | [244, 260, 258] |
| `ftb9b` | 3.7902 | 1 | **78.36** | [77.32, 79.40] | [200, 0, 0] |
| `ftbcfg` | 3.7514 | 3 | **78.24** | [77.19, 79.28] | [181, 137, 137] |
| `ftbvu` | 3.7128 | 3 | **78.11** | [77.07, 79.15] | [294, 295, 294] |

Stored machine-readably in `plots/cache/proxy_predictions.json`.

**Early partial evidence, already in:** `ftbqmvo` predicted **78.71**, and its two finished seeds
average **78.69** (off by 0.02). `ftb4i` predicted **80.33**, its one finished seed is **79.91**
(off by 0.42, inside the 0.53 pp residual sd). Encouraging but n=2 arms; the real test is the
full table.

## 0a. A DDP BUG CONTAMINATES THE SHUFFLE ARMS (found 2026-08-31)

> **`ftb4e3`, `ftb11is` and `ftb0l` did not run the experiment their configs describe.** Read
> this before using any shuffle result, including fig5's top rows, the "shuffling is free"
> claim and fig7's shuffle-count sweep.

**The bug.** `main.py:527` sets `seed = args.seed + utils.get_rank()`, so every rank has a
different torch RNG. `utils.pr_load_model` constructs DDP at `utils.py:968` and returns
`model.module` -- so **all** of the init surgery runs on each rank's own replica, *after* DDP's
one-time construction broadcast. DDP all-reduces gradients and never re-syncs weights. Any edit
drawing from the torch RNG therefore leaves the four ranks holding four different models for the
entire run, trained with averaged gradients.

**Verified**, two ranks, real `utils.shuffle_weights` (`plots/verify_rank_fix.py`):

| | rank0 checksum | rank1 checksum | |
|---|---|---|---|
| `randperm` path, before fix | -92487.539 | **-109416.297** | diverged |
| `randperm` path, after fix | -92487.539 | -92487.539 | agree |
| `argsort` path, before/after | -86603.906 | -86603.906 | never diverged |

Norms identical throughout: same values, different arrangement per rank.

**Blast radius** (`plots/audit_rank_bug.py`, 201 runs audited):

* **SEVERE, all weights diverge** -- `ftb4e3` (29451642), `ftb11is` (29472466), `ftb0l` (29472465).
  These use `--weight_shuffle`, i.e. `torch.randperm` on every listed tensor.
* **PARTIAL, 1-D params only** -- `ftbqm1d`, `ftbqmbias`, `ftbqm1dpar`, `ftbqmln`, `ftbqm1dv`,
  `ftbqm1dqk`, `ftbqm1dvo` (`quantile_1d_mode != skip` -> randperm; `parametric` -> randn).
* **CLEAN** -- everything else, 168 runs: the 2-D quantile rank map (`argsort` of the synced
  tensor, no RNG), rho-matching (scale factors broadcast from rank 0), norm matching, outlier
  clipping, `slice_scale`, and every plain checkpoint load (`r`, `p`, `ftb3i`, `ftb1i`, `ftb11i`,
  `ftbqm`, `ftbnorm`, `ftb4o`, `ftbrho`).

**What it costs.** `ftb4e3` is the +2.08 arm at the top of fig5 and the reference point of §0's
grouping; its partner `ftbqm1dv` is only PARTIALLY affected, so **the two are not like-for-like**
-- which is the unexplained 0.68 gap chased at length on 2026-08-31. `ftb11is` being SEVERE while
its control `ftb11i` is clean undercuts fig7 the same way.

**The fix** (`main.py`, immediately before `Start training`): broadcast every floating-point
tensor of `model_without_ddp.state_dict()` from rank 0, then barrier. This covers every current
and future edit site at once and is a no-op when the ranks already agree.

**Telling fixed runs from buggy ones** -- four independent markers:

1. arm name `ftb4e3fix` / `ftb11isfix`
2. results dir `29523306` / `29523309`
3. wandb config `notes = "rank-sync-fix"`
4. the run log contains `[init-sync] broadcast N tensors from rank 0` -- the only marker that
   proves the fix *executed*; `plots/audit_rank_bug.py` checks it first.

**Reruns queued 2026-08-31**: `ftb4e3fix` (SLURM_ID 29523306) and `ftb11isfix` (29523309),
3 seeds each, experiment args verified byte-identical to the originals.

### 0a.1 Re-run queue: the seven PARTIAL arms (not yet started)

**Status 2026-08-31.** Three arms have been re-run under the fix and are queued or running:

| arm | new SLURM_ID | replaces |
|---|---|---|
| `ftb4e3fix` | 29523306 | `ftb4e3` (29451642), SEVERE |
| `ftb11isfix` | 29523309 | `ftb11is` (29472466), SEVERE |
| `ftbqmlnvo` | 29523316 | its own pre-fix run (29520502, 350 epochs, moved to `..._prefix_contaminated`, 69 GB -- prune when convenient) |

`ftb0l` (29472465, SEVERE, n=1) has NOT been re-run; it is a minor arm (full proc, all 12 blocks
downscaled AND shuffled) and nothing in section 0 rests on it.

**Still contaminated, still quoted throughout section 0 and fig5.** These seven are PARTIAL: their
2-D weights were rank-consistent (the quantile rank map uses `argsort`, no RNG) but their 1-D
parameters diverged across the four ranks. That is not obviously negligible -- LayerNorm gains
multiply everything downstream, and in several of these arms the 1-D parameters ARE the variable
under test.

| arm | old SLURM_ID | `quantile_1d_mode` | n | delta vs random |
|---|---|---|---|---|
| `ftbqm1d` | 29501773 | shuffle | 3 | +0.42 |
| `ftbqmbias` | 29501780 | bias | 3 | +0.05 |
| `ftbqm1dpar` | 29502416 | shuffle + parametric | 3 | +0.28 |
| `ftbqmln` | 29504032 | layernorm | 3 | +0.78 |
| `ftbqm1dv` | 29507368 | shuffle | 3 | +1.40 |
| `ftbqm1dqk` | 29511670 | shuffle | 3 | +0.58 |
| `ftbqm1dvo` | 29511673 | shuffle | 3 | +1.36 |

**Cost:** 21 jobs x 4 GPUs x ~23 h. Do not queue this on top of a loaded cluster without checking.

**How to queue** (each script needs the same three infra edits first -- they are what the already
re-run arms got):

```bash
cd /home/schrodi/Procedural
for a in ftbqm1d ftbqmbias ftbqm1dpar ftbqmln ftbqm1dv ftbqm1dqk ftbqm1dvo; do
  f=vitbase_runs/run_train_$a.sh
  grep -q 'SBATCH --requeue' $f || sed -i 's|#SBATCH --nodes 1|#SBATCH --requeue\n#SBATCH --nodes 1|' $f
  sed -i 's|--notes ""|--notes "rank-sync-fix"|' $f
  sed -i 's|accuracy_IMNET_BASE_$SLURM_ID.json|accuracy_IMNET_BASE_${SLURM_ID}_s${SEED}.json|' $f
  sed -i 's|grad_norms_IMNET_BASE_$SLURM_ID.json|grad_norms_IMNET_BASE_${SLURM_ID}_s${SEED}.json|' $f
  jid=$(sbatch --parsable --export=SEED=0 $f)
  for s in 1 2; do sbatch --export=SLURM_ID=$jid,SEED=$s $f; done
  echo "$a -> SLURM_ID $jid"
done
```

Do NOT reuse the old SLURM_IDs: a fresh id gives a fresh results directory, and `--auto_resume`
would otherwise pick up the contaminated checkpoints.

**Verify afterwards** with `python plots/audit_rank_bug.py --refresh`, which classifies a run by the
`[init-sync] broadcast` marker in its log -- proof the re-sync executed -- ahead of any flag.

**What cannot be concluded until then.** `ftb4e3fix` against the existing `ftbqm1dv` is still not
like-for-like: one side fixed, the other not. Section 0's write-magnitude grouping mixes CLEAN
arms (`r`, `ftbqm`, `ftbnorm`, `ftb3i`) with PARTIAL ones (every `ftbqm1d*`), so its R^2 = 0.83 is
computed over a mixed population. Re-running these seven is what makes fig5 and section 0 clean.

**Unrelated issue found alongside**: `main.py`'s init/analysis path **deadlocks on 2 GPUs** with a
600 s NCCL ALLREDUCE timeout. All the arms here ran at 4 GPUs; anything at a different GPU count
needs checking.

## 0. CURRENT UNDERSTANDING (2026-08-30) — read this before anything below

> **RETRACTED AS A MECHANISM 2026-09-04 — see §0c.8.** This section is now a lab notebook, not a
> conclusion. Every `ftbqm*` arm it groups has been re-run clean, and the result is a **+1.26
> super-additive interaction** (LayerNorm gains alone +0.06, v-slice alone +0.45, both +1.77) that
> a single scalar cannot represent. `ftbqmln` and `ftbqmlnvo` differ by **1.71** at nearly the same
> `value_write`; `ftbqm` and `ftbqmln` differ by 0.06 at a 2.6-fold difference in it. §0.4's
> sufficiency question is answered No by `ftbcfg` (**77.49, 0.59 below random**), and §0.2's
> "LayerNorm gains are worth +0.78" is dead (clean: **+0.14, 0.7 sigma**).
>
> The accuracy numbers below are still correct as measurements, but every pre-fix `ftbqm*` value is
> about **0.53 too high** (§0c.8's contamination table).
>
> Two standing qualifications. The R^2 = 0.83 is computed over a population in which every arm but
> `ftb3i` and `ftbqmlnvo` carries the §0a DDP bug — tiers 1-2 of the re-runs are queued (§0c.6) and
> will settle it. And §0.4-0.6's own caveats are unresolved: sufficiency (`ftbcfg`) is untested and
> the relation is **not monotone** in `value_write` (random sits at 0.307, below the winners, and
> scores 0.00).

> **This section supersedes every mechanistic claim about the EARLY blocks (0-8) in the rest of
> this document.** The accuracy numbers below are all still correct — they come from each run's
> `log.txt` and nothing about them has changed. What was wrong is the *interpretation*: sections
> 3.10.5 through 3.10.9.12 chase a succession of quantities (the value distribution, the
> LayerNorm parameters, the attention logit scale, the qk/v ratio, the v slice) that all turn out
> to be **proxies for one scalar**. Treat those sections as a lab notebook of how we got here,
> not as conclusions. The late-block results (§3.10.3, §3.12.3) are a separate finding and are
> NOT affected.

### 0.1 The result

Define the **attention write magnitude** at init, per block, averaged over blocks 0-8:

```
value_write  =  gamma_norm1 * ||W_v|| * ||W_proj||  /  d
```

It is a product of norms, so it is **invariant to any permutation of the weights**. Sorting every
early-block arm that carries the checkpoint's 2-D values by this one number:

| value_write | arms | delta vs random | runs |
|---|---|---|---|
| **~0.52** | `ftbqm1dvo`, `ftbqm1dv`, `ftb4e3`, `ftb3i` | **+1.69** (sd 0.41) | 12 |
| **~0.87** | `ftbqm1d`, `ftbqm1dqk`, `ftbqmln`, `ftbqm1dpar` | **+0.51** (sd 0.27) | 12 |
| **~2.24** | `ftbqm`, `ftbqmbias` | **+0.07** (sd 0.13) | 6 |

One-way ANOVA over the three groups: **F = 64.9, p = 4.9e-11, R^2 = 0.828.** One scalar measured
at init accounts for **83% of all variance across 30 runs**.

Figure: `plots/out/fig13_value_write.png` (`plots/fig_value_write.py`). Reproduce the numbers with
`plots/analyse_ckpt_differences.py`; the feature is `W.value_write` in `plots/cache/ckpt_diff.json`.

### 0.2 Every earlier "mechanism" was this scalar in disguise

* **The LayerNorm gains (§3.10.9.2, header point 3a).** `ftbqm` and `ftbqm1d` differ *only* in
  1-D parameters, and `2.237 x 0.384 = 0.859 ~ 0.869`. proc's gains matter **exactly and only**
  as a multiplier on the write magnitude — not as learned per-channel values. The +0.78 for
  `ftbqmln` was never about the gains being special.
* **The qkv slicing / the v slice (§3.10.9.5, §3.10.9.12).** Pooling inflates `||W_v||` from 28.8
  to 50.9, and `0.869 x (28.8/50.9) = 0.492 ~ 0.514`. The same scalar again.
* **The attention logit scale (§3.10.9.10, header point 3a-ii).** Within a fixed write magnitude
  it varies 0.0055 -> 0.0081 and accuracy does not track it. `ftbqm1dvo` (0.00552) and `ftbqm1dv`
  (0.00738) differ by **0.04**.
* **The qk/v ratio (§3.10.9.11).** Within a fixed write magnitude it varies 1.00 -> 1.15 -> 1.87
  -> 2.18 and accuracy does not track it either. The r = +0.955 in that section is real but it is
  a proxy correlation: in every checkpoint-derived arm, moving v moves both quantities together.
* **The 1-D partition (§3.10.9.6).** `ftbqmln` +0.78, `ftbqmbias` +0.05 and `ftbqm1dpar` +0.28 all
  sit at the same write magnitude and are within resolution of each other. There is no 1-D story.

### 0.3 Why this quantity works where rho did not

`value_write` is a product of weight norms and is **arrangement-invariant**. rho is a
forward-pass measurement and is not: it also sees how the weights are arranged.

* `ftb3i` (proc intact) and `ftb4e3` (proc permuted within each slice) have rho 0.471 vs 0.297 —
  a 1.6x difference — and score +1.91 and +2.08, the same run. rho cannot be the mechanism.
* Their `value_write` is 0.536 vs 0.514. Nearly identical, as it must be under permutation.

This also explains the rho arms' failures without any extra hypothesis. `ftb4o` matches proc's
rho_attn (0.462 vs 0.471) while its `value_write` is **3.949, i.e. 7.4x proc's**, because its
LayerNorm gains are 1.0 where proc's are 0.38. Matching rho and matching the write magnitude are
different targets, and only the latter tracks accuracy. Same for `ftb11d` (-0.37) and `ftb11s`
(-1.49): both move the write magnitude by 78-129x in one block.

### 0.4 What is NOT yet established

**Whether the write magnitude is sufficient on its own, or needs the checkpoint's values too.**
Every arm in the table above carries proc's 2-D value multisets. The checkpoint-free arms are:

| arm | value_write | logit_scale | delta | n |
|---|---|---|---|---|
| `r` | 0.307 | 0.0032 | 0.00 | 3 |
| `ftbqu` | 0.307 | 0.0152 | -0.01 | 2 |
| `ftbslice` | 0.536 | **0.0461** | +0.48 | 2 |
| `ftbnorm` | 0.869 | 0.0055 | +0.20 | 3 |
| `ftbvd` | 0.141 | 0.0032 | +0.78 | 1 |
| `ftb4o` | 3.949 | 0.0032 | -0.80 | 1 |

`ftbslice` was built to answer this and **is mis-specified**: it matches proc on `value_write`
and on the qk/v ratio, but with gamma pinned at 1.0 you cannot match the ratio without inflating
q and k in absolute terms, so its logit scale lands at **0.0461 — 5.7x proc's 0.00806, and higher
than `ftbqm`'s 0.0353, which scored +0.08**. Its +0.48 is therefore not evidence against the
scale hypothesis.

**`ftbcfg` (queued 2026-08-30, 3 seeds, SLURM_ID 29520505) is the corrected test.** On a plain
random init with no checkpoint anywhere: `--custom_init_type slice_scale --slice_scale_qk 1.587
--slice_scale_v 0.729 --slice_scale_proj 2.395`, which hits **all three** of proc's attention
numbers at once (`value_write` 0.536, `logit_scale` 0.00806, `qk/v` 2.18 — verified on CPU
against the checkpoint). If it reaches ~+1.7, the early-block effect is three scale factors and
the checkpoint is unnecessary. If it lands near 0, proc's actual values matter and the scalar is
necessary but not sufficient.

### 0.5 Resolution, and what cannot be read

**Pooled within-arm seed sd = 0.247 (df = 24, all n=3 arms).** The se of a 3-vs-3 difference is
0.202, so the smallest resolvable gap is **0.41 pp at p < 0.05**. Any difference below that in
this document is not a result. This replaces the "0.68 noise floor" quoted in 3.10.9.12, which
was estimated from a single pair.

**One gap is genuinely unexplained and bounds everything.** `ftb4e3` (+2.08) and `ftbqm1dv`
(+1.40) were verified **in code** to be the same construction: all 108 tensors in blocks 0-8 have
identical sorted value multisets (including the qk and v slices separately), all 24 tensors
outside blocks 0-8 are identical, and the training hyperparameters match. No init feature can
separate them, so their 0.68 difference is run-level noise exceeding the seed-level estimate.
It is also the entire residual spread inside the top group. **Do not read the ordering within a
value_write group.**

### 0.6 Caveats on the headline claim

* Only **three distinct values** of `value_write` appear among the content arms (0.52, 0.87,
  2.24). An R^2 of 0.83 over three groups is a strong grouping, not a fitted dose-response curve.
  Nothing here establishes the shape of the relation or locates an optimum.
* `value_write` may itself be a proxy for something it is collinear with in this sample. The
  claim that survives is narrower: *every intervention that changed early-block accuracy did so
  while changing this scalar, and interventions that left it fixed did not change accuracy.*
* Random init sits at 0.307 — **below** the best group's 0.52 — and scores 0.00. So the relation
  is not monotone in `value_write` alone across all arms; content and scale are entangled, and
  `ftbcfg` is what separates them.
* This says nothing about the late blocks. §3.10.3 and §3.12.3 stand.

> **Where this ended up.** The investigation started as "why does the 9-11 scaling result die
> when blocks 0-8 are proc?" The answer inverted the question, and produced a method.
>
> 1. **Late blocks want *write magnitude*, not learned weights.** Setting `rho ~1.4-2.0` in the
>    last few blocks at init is worth **+2.47 over random on IN-100** and **+1.61 on
>    ImageNet-1k**, with **no checkpoint involved** (§3.12.3, §3.10.3). This is the practical
>    output and it replicates across two checkpoints, two datasets and two model scales.
>
> 2. **The best model is proc everywhere *except* the last block, with that block calibrated —
>    and this replicates at both scales.** `q2` reaches **87.25 +/- 0.09** on IN-100 against proc
>    init's 86.15 (**+1.10**, 3 seeds spanning 0.18) and `ftbcomp11` reaches **80.63 +/- 0.18**
>    on IN-1k against 80.09
>    (**+0.54**, 4.4 sigma). It is the best arm in this document at both scales (§3.10.4).
>    The *decomposition* differs, though: calibrating the last block is worth **+1.36** on IN-100
>    but only **+0.26** on IN-1k, and the claim that proc's weights are *worse than random* in
>    block 11 holds only at ViT-B (on IN-100 that step is **-0.26**).
>
> 3. **Early blocks help on IN-1k and not on IN-100 — and the *mechanism* differs, not just the
>    size.** On IN-1k the contribution is the weight **value distribution**: matching proc's
>    per-tensor norms is worth nothing (+0.16, 0.7 sigma) while proc's shuffled values are worth
>    +2.11. On IN-100 the reverse — norms fully account for it (e2 = e3 to 0.01) and the sign is
>    negative. Same intervention, opposite mechanism and opposite sign (§3.10.5). This is the
>    main open problem.
>
> 3a. ~~**One component of the early-block benefit now transplants, and it is the LayerNorm
>    parameters.**~~ **RETRACTED 2026-09-04 — the clean re-run of `ftbqmln` scores +0.14 (0.7
>    sigma), not +0.78; the gains are worth NOTHING alone and only matter jointly with the v slice
>    (§0c.8). Previously marked superseded by §0** — the gains act only as a multiplier on the write
>    magnitude (2.237 x 0.384 = 0.859). Original text: Giving a random model proc's LayerNorm gains and biases in blocks 0-8 —
>    6,912 gain values, nothing else — is worth **+0.78 (p = 0.013, 3 seeds)**. It is the only
>    positive result in an elimination of nine transplant arms, and it is specific: proc's weight
>    values are worth +0.08, its per-tensor norms +0.20, its non-LayerNorm biases +0.05,
>    moment-matched gains +0.28, and matching its write magnitude rho is *harmful* at -0.80.
>
> 3a-ii. ~~**The gap is localised to a single variable: how the fused qkv is matched.**~~
>    **SUPERSEDED by §0** — the qkv split and the logit scale are both proxies for
>    `value_write`. Original text: `ftbqm1d`
>    (all 8 1-D params, qkv **pooled**) scores **+0.42**; `ftb4e3` (the same 8, qkv **sliced**)
>    scores **+2.08** — identical 1-D content, **1.66 apart at p = 0.0001**. The 1-D parameters
>    are therefore NOT the mechanism: all eight are worth *less* than the four LayerNorm ones
>    alone (+0.78). The first quantity that groups the arms correctly is the
>    **attention logit scale**, `gain^2 * ||W_q|| ||W_k||`, which no per-tensor summary captures
>    because it combines the weight scale with the LayerNorm gain (§3.10.9.10). proc's Q and K are
>    ~4x random while V is only ~1.5x, and proc's 0.31 gains cancel most of it; arms that take the
>    big Q,K *without* the small gains run at **11x** random's logit scale and are worth nothing.
>    `ftbqm1dv` sits at the winners' value exactly, and is the arm that closes or breaks the
>    elimination (§3.10.9.5-10).
>
> 3b. **Early blocks want the learned weights — the depth picture.** On
>    ImageNet-1k proc's early blocks are a monotone ramp worth up to +2.29 (§3.13.3); on IN-100
>    nine proc blocks are worth **+0.16**. `ftb3i` ruled out a shuffle artifact. This
>    contradiction is real and unexplained, and it is the main open problem (§3.13.3).
>
> 4. **Proc's specific magnitude profile is fungible** — matching it is not required; a uniform
>    target does as well (§3.12.1). The stronger "uniform is *better*" claim was
>    original-checkpoint only and did not replicate (§3.11.1).
>
> 5. **Depth, not count, decides** — the same 3-block intervention is -1.23 at blocks 3-5,
>    -0.45 at 6-8, +1.93 at 9-11 on IN-100 (§3.12.4), and calibrating *early* blocks is harmful
>    at both scales (ftb4o 77.27 < r 78.08).
>
> **Two standing caveats before any of this is written up — see §3.14.2.** (i) every arm here
> runs with **LayerScale disabled**, so gains "over random init" are over a no-LayerScale
> baseline; (ii) two literature claims from earlier drafts were retracted on 2026-08-23 and the
> direction of the finding is less novel than those drafts implied. No measured result changed.
> The **composition result** (claim 2 above) is unaffected by both and is what to lead with.
>
> **Before porting**, two things. (a) §3.10.1: the rho target is calibrated against a
> *random-init* residual stream and does not survive a change in what precedes the scaled
> blocks — reaching 1.4 costs a x81.9 `fc2` factor from random init but x2930 with four proc
> blocks in front. Measure rho on the exact model you will train. (b) §3.14: nanochat
> **zero-inits both output projections**, so rho there is exactly 0 and the recipe cannot be
> applied as a multiplier; it already implements a depth-dependent rho schedule through
> `resid_lambdas`. The novel content of this study is the *measured band* and the *depth at
> which the sign flips*, not the direction itself.

## 1. Setup

| | |
|---|---|
| model / data | `vit_small`, IN-100 (`/home/schrodi/Procedural/imagenet100`) |
| pr checkpoint | `results/pr_vits/pr_27291166_final.pth` |
| schedule | 300 epochs, 50 warmup, lr 2e-3, AMP, `--skip_norm true` |
| batch | `total_batch_size 512` (batch 128, update_freq 2 on 2 GPUs) |
| seeds | 0, 1, 2 |
| wandb | `procedural_pretraining / i100-playground` |
| scaling | attn V-slice + `attn.proj` by `sqrt(r)`, `mlp.fc2` by `r` (`*_match_delta_norms`) |

Scripts: `i100_playground/run_train_i100.sh` (one arm per `ARM=`), `launch.sh`,
`run_posthoc_ratios.sh`, `launch_trajectory.sh`.

Metric throughout is `rho = ||Delta_sublayer|| / ||r_in||`, the per-block write magnitude
relative to the residual stream entering that block, averaged over 5000 training images.

### Arm design

In every arm the **target differs from the trained model only in the blocks being scaled**.
This is what makes the comparison clean, and it is why arm b1 was rebuilt (see §5).

| arm | blocks 0-8 | blocks 9-11 | scaled toward | `--init_method` |
|---|---|---|---|---|
| r | random | random | — | `default`, `--initialize ""` |
| p | proc | proc | — | `default` |
| a1 | random | random, **up** | random 0-8 + proc 9-11 | `upscale_random_match_delta_norms` |
| a2 | random | proc, **down** | random 0-11 | `downscale_pr_match_delta_norms` |
| b1 | proc | proc, **down** | proc 0-8 + random 9-11 | `downscale_mixed_match_delta_norms` |
| b2 | proc | random, **up** | proc 0-11 | `upscale_random` + `--init_method_copied_blocks "0;1;...;8"` |

**`--random_blocks` does not tell you what the trained model contains.** It is interpreted
differently by each init method, so composition cannot be read off the launch flags:

| init method | `--random_blocks` applies to | trained model |
|---|---|---|
| `default` | the model | proc except those blocks |
| `upscale_random` | the **target** only (model is `pr_load_model(path="")`) | fully random, unless `--init_method_copied_blocks` copies proc blocks in afterwards |
| `downscale_pr` | the model; target is fully random | proc except those blocks |
| `downscale_mixed` | the **target** only (cleared for the model) | proc in **all** blocks |
| `match_target_block_norms` | — | random, listed blocks rescaled to proc norms |
| `clip_outlier_weights` | — | proc in all blocks, listed blocks winsorised |

So b1 and c2 pass `--random_blocks` yet train an all-proc model, and b2/g2 pass none yet end up
proc-in-0-8 via the block copy. The arms whose trained model is **proc 0-8 + random 9-11** are
**b2, e3 and g2** — not b1, and not the f-arms, which are proc in all 12 blocks with only
0-8 clipped. Verify from the run log (`Load initialization from`, `Removing key blocks.N`,
`Copying weights for layer N`) rather than from the flags.

---

## 2. Results

Best top-1, mean over 3 seeds (c2: 2 complete). All arms, final.

| arm | blocks 0-8 | blocks 9-11 | mean | sd | vs r | vs p |
|---|---|---|---|---|---|---|
| r | random | random | 84.20 | 0.09 | — | -1.17 |
| p | proc | proc | 85.37 | 0.21 | +1.17 | — |
| **a1** | random | random, **up** | **86.26** | 0.27 | **+2.06** | +0.89 |
| a2 | random | proc, **down** | 84.51 | 0.39 | +0.31 | -0.86 |
| b1 | proc | proc, **down** | 84.87 | 0.24 | +0.67 | -0.51 |
| b2 | proc | random, **up** | 85.53 | 0.32 | +1.33 | +0.16 |
| c1 | random, rho -> proc | random | 84.42 | 0.19 | +0.22 | -0.95 |
| c2 | proc, rho -> random | proc | 85.40 | 0.21 | +1.20 | +0.03 |
| e1 | proc **shuffled** | proc | 85.15 | 0.27 | +0.95 | -0.23 |
| **e2** | random, \|\|W\|\| -> proc | random | **83.10** | 0.68 | **-1.10** | -2.27 |
| e3 | proc **shuffled** | random | 83.22 | 0.45 | -0.98 | -2.15 |
| **f1** | proc, top 0.1% **clipped** | proc | **85.78** | 0.26 | +1.58 | **+0.41** |
| **f2** | proc, top 1% **clipped** | proc | **86.01** | 0.09 | +1.81 | **+0.63** |
| **f3** | proc, top 5% **clipped** | proc | **85.76** | 0.71 | +1.56 | **+0.39** |
| g1 | proc clipped 1% | proc, **down** | 84.68 | 0.16 | +0.48 | -0.69 |
| **h3** | random | random, **flat rho 1.39** | **86.91** | 0.59 | **+2.71** | **+1.54** |
| h1 | random | random, rho x0.5 | 85.87 | 0.33 | +1.67 | +0.50 |
| h2 | random | random, rho x2.0 | 85.22 | 0.64 | +1.02 | -0.15 |
| m1 | random | 3,4,5 calibrated | 82.88 | 0.87 | -1.32 | -2.49 |
| m2 | random | 6,7,8 calibrated | 83.91 | 0.42 | -0.29 | -1.46 |
| n07 | random | 9-11, abs rho 0.7 | 85.99 | 0.19 | +1.79 | +0.61 |
| n10 | random | 9-11, abs rho 1.0 | 86.53 | 0.52 | +2.33 | +1.15 |
| **n14** | random | 9-11, **abs rho 1.4** | **87.02** | 0.24 | **+2.82** | **+1.65** |
| **n20** | random | 9-11, **abs rho 2.0** | **86.99** | 0.15 | **+2.79** | **+1.61** |
| n25 | random | 9-11, abs rho 2.5 | 86.79 | 0.18 | +2.59 | +1.41 |
| n26 | random | 9-11, abs rho 2.6 | 86.72 | 0.44 | +2.52 | +1.35 |
| n28 | random | 9-11, abs rho 2.8 | 86.08 | 0.99 | +1.88 | +0.71 |
| x0 | **xavier** init, no calibration | — | 81.73 | 0.33 | — | — |
| x14 | **xavier** init | 9-11, abs rho 1.4 | 82.56 | — | — | *(n=1, +0.80 vs x0)* |
| e4 | proc | random (no scaling) | 85.05 | 0.19 | +0.85 | -0.32 |
| g2 | proc clipped 1% | random, **up** | 85.64 | 0.62 | +1.44 | +0.27 |
| d0 | proc **frozen** | proc | 76.45 | 0.33 | -7.75 | -8.92 |
| d1 | proc **frozen** | proc, down | 75.95 | 0.67 | -8.25 | -9.42 |
| d2 | proc **frozen** | random, up | 76.19 | 0.14 | -8.01 | -9.18 |

Seed noise is 0.15-0.61, so the comparisons below sit well outside it. The a-b arms are §2.1;
the c-e arms probe *what proc's blocks 0-8 contribute* (§3.6-3.8); the d arms are the freeze
test (§3.4).

### 2.1 The two contrasts that started this

**Random 0-8: magnitude is decisive.** Installing proc-like write magnitudes into an
all-random model gains +1.93 over random and beats proc itself. Stripping magnitude from
genuinely transplanted proc blocks returns to random level. Swing a1 - a2 = **1.58**.

**Proc 0-8: magnitude barely matters.** The same swing b2 - b1 = **0.56**, ~3x smaller.
b2 is the sharp one: with proc 0-8 you can discard proc's blocks 9-11 entirely, put back
random weights at matched magnitude, and lose nothing (85.53 vs 85.37).

**a1 contains no proc weights at all** — it is `pr_load_model(path="")`, fully random, with
only blocks 9-11 rescaled; the checkpoint is used solely to measure target ratios. It beats
full proc init by ~1 point, while every arm with proc 0-8 lands at 85.1-85.6 regardless of
what is done to 9-11.

---

## 3. What is actually going on

### 3.1 A hypothesis that the data killed

Initial ratios suggested proc's late blocks were near-identity: at init, proc blocks 5-11 sit
at rho ~0.015-0.030 against random init's ~0.11, i.e. they write 5-8x less. Weight norms are
also ~2.4x larger in proc's late blocks (92.4 vs 38.4), and relative drift over training is
about half that of random init (1.6-2.3 vs 3.0-3.6), while *absolute* drift is comparable
(87-148 vs 116-137) — consistent with AdamW moving weights a roughly fixed distance
regardless of `||W||`.

That suggested "proc's large weights cannot move, so the blocks stay near-identity and the
network is effectively shallow". **This is wrong.** Measured on the epoch-299 checkpoints,
proc's late blocks moved more than any other arm's:

| blk | r: init -> final | p: init -> final | a1: init -> final |
|---|---|---|---|
| 5 | 0.124 -> 0.246 | **0.015 -> 0.180** | 0.123 -> 0.261 |
| 8 | 0.111 -> 0.167 | **0.019 -> 0.380** | 0.110 -> 0.257 |
| 11 | 0.107 -> 0.246 | **0.030 -> 0.510** | 1.126 -> 0.410 |

Proc's blocks 5-11 climb 12-25x and end as the *largest* writers of any arm. There is no
effective-depth collapse, and reduced plasticity is not the mechanism.

### 3.2 What the trajectory shows

`rho_attn`, block 11, seed 0, measured from the per-epoch checkpoints:

| epoch | r | p | a1 | b1 | b2 |
|---|---|---|---|---|---|
| 0 | 0.115 | 0.212 | **1.051** | **0.016** | **0.085** |
| 2 | 0.094 | 0.619 | 0.900 | 0.087 | 0.107 |
| 10 | 0.087 | 0.821 | 0.607 | 0.202 | 0.231 |
| 20 | 0.085 | 0.598 | 0.545 | **0.323** | **0.284** |
| 40 | 0.064 | 0.325 | 0.426 | 0.271 | 0.224 |
| 70 | 0.061 | 0.279 | 0.239 | 0.267 | 0.209 |
| 299 | 0.246 | 0.510 | 0.410 | 0.507 | 0.476 |

- **b1 and b2 start 5.3x apart and are within 14% of each other by epoch 20**, and sit on
  top of untouched proc by epoch 70. Block 9 behaves the same, starting 9x apart
  (0.008 vs 0.075) and converged by epoch 40. All of this happens *before* the LR schedule
  peaks at epoch 50.
- **a1 never converges onto r.** It starts 9x above and still ends 1.7x above
  (0.410 vs 0.246).
- Proc 0-8 drives a large early transient: p's block 11 jumps 0.212 -> 0.821 by epoch 10
  before settling. Blocks 9-11 are violently reshaped in the first few epochs.
- Every arm dips to a minimum around epoch 40-70 and then climbs, so the final profile is
  set by late training rather than by initialisation.

### 3.3 First hypothesis (superseded by §3.12 and §3.13)

With proc-initialised blocks 0-8, the residual stream entering block 9 admits essentially one
solution for the late blocks, and optimisation finds it within ~20 epochs regardless of where
those blocks start. The blocks 9-11 intervention is therefore **erased**, which is why
b1 ~ b2 ~ p in accuracy. With random 0-8 the constraint is absent, the initialisation
persists, and a magnitude-tuned random model (a1) reaches a better optimum than proc init
does.

Two sub-mechanisms were candidates:

- **(i) co-adaptation** — blocks 0-8 reshape the stream during the first ~20 epochs, and the
  late blocks follow.
- **(ii) stream statistics at init** — the stream proc 0-8 produces at epoch 0 already fixes
  what blocks 9-11 must do, independent of any subsequent change in 0-8.

### 3.4 The freeze test rules out co-adaptation

d1 and d2 repeat b1 and b2 with **blocks 0-8 frozen**, so the early blocks cannot reshape the
stream at all. `rho_attn`, block 11 (checkpoint at epoch N = after N+1 epochs of training):

| epoch | d1 | d2 | d2/d1 | b1 | b2 | b2/b1 |
|---|---|---|---|---|---|---|
| 0 | 0.033 | 0.071 | **2.16** | 0.016 | 0.085 | **5.31** |
| 2 | 0.076 | 0.095 | 1.26 | 0.087 | 0.107 | 1.23 |
| 5 | 0.105 | 0.110 | **1.04** | 0.123 | 0.159 | 1.29 |
| 20 | 0.201 | 0.195 | 0.97 | 0.323 | 0.284 | 0.88 |
| 40 | 0.425 | 0.349 | 0.82 | 0.271 | 0.224 | 0.83 |

**Freezing does not preserve the intervention.** d1 and d2 converge onto each other by epoch 5,
at least as fast as the unfrozen pair. Mechanism (i) is refuted: blocks 0-8 do not need to
change for the attractor to operate. **Mechanism (ii) holds** — the residual stream that proc
0-8 produces *at initialisation* is what determines the late-block solution.

### 3.5 The erasure timescale follows the LR schedule

Same measurement with `--warmup_epochs 5` instead of 50:

| epoch | p (warm5) | p (warm50) | b1 (warm5) | b1 (warm50) |
|---|---|---|---|---|
| 2 | 0.436 | 0.619 | 0.175 | 0.087 |
| 5 | 0.278 | 0.795 | **0.305** | 0.123 |
| 10 | 0.274 | 0.821 | 0.250 | 0.202 |
| 40 | 0.254 | 0.325 | 0.166 | 0.271 |

With warmup 5, b1 reaches p's level by **epoch 5** (0.305 vs 0.278); with warmup 50 it takes
~70 epochs. The transient in p is also much smaller (peak 0.436 vs 0.821). So the *speed* of
erasure is set by how fast the LR ramps — it is not an intrinsic timescale — while the
*existence* of the attractor is unaffected. Shortening warmup makes erasure faster, not weaker.

### 3.6 Early-block magnitude is not the cause

c1 and c2 apply the same delta-norm matching to blocks **1-8** (block 0 excluded, §5.1).
Both are null:

| arm | design | mean | reference |
|---|---|---|---|
| c1 | random model, early blocks matched to proc's ratios | 84.42 | r = 84.20 (**+0.22**) |
| c2 | all-proc model, early blocks matched to random's ratios | ~85.40* | p = 85.37 (**+0.03**) |

\* seeds at epoch 287-298, provisional.

Neither installing proc's early write profile into a random model nor removing it from a proc
model changes anything, despite c1 scaling attention writes by 0.60-0.92x and block 4's `fc2`
by 12.65x. This gives a **depth asymmetry**: late-block magnitude is decisive in the random
context (a1: +1.93 over r), early-block magnitude is irrelevant in both contexts. Whatever
proc's blocks 0-8 contribute is carried by the **learned structure / direction** of those
weights, not by their write magnitude.

The freeze arms agree from the accuracy side: d0 = 76.45, d1 = 75.95, d2 = 76.19. With blocks
0-8 frozen the 9-11 intervention again makes no difference (d1 vs d2 = 0.20, inside noise),
matching §3.4. Freezing itself costs ~8.9 points, as expected with 3/4 of the network fixed.

### 3.7 Early-block structure is not the cause either

e1 keeps proc init but permutes every tensor in blocks 0-8 in place (`--weight_shuffle`), so
each per-tensor norm is preserved exactly while the learned structure is destroyed. qk and v
slices are shuffled separately so the v-slice norm, which sets the attention write magnitude,
is not redistributed across qkv.

| arm | mean | sd |
|---|---|---|
| r | 84.20 | 0.09 |
| **e1** (proc, blocks 0-8 shuffled) | **85.15** | 0.27 |
| p | 85.37 | 0.21 |

e1 lands with p, not r: shuffling costs at most 0.26 (~1 sd) and retains the full proc
advantage over random (+0.71, ~3 sd). **The learned structure of blocks 0-8 is not what proc
init contributes.**

### 3.8 The weight-norm profile is not the cause either

Comparing what the two "structure-destroyed" arms preserve isolates a single variable:

| | proc weight norms in 0-8 | proc structure in 0-8 | result |
|---|---|---|---|
| e1 | **yes** (35 -> 92, rising with depth) | no (shuffled) | 85.15 ~ p |
| c1 | no (random's flat 38.4) | no (random) | 84.42 ~ r |

Both destroy structure; only e1 keeps proc's per-block `||W||` profile, and only e1 keeps the
proc advantage. The remaining candidate is therefore the **per-block weight-norm profile** —
`||W||` growing with depth — not the weight directions and not the delta-norm ratios.

Note this is *not* the same as the write ratio rho: c1 matched rho per block and gained
nothing, so the two are dissociable. `||W||` also sets how far AdamW moves a block in relative
terms (§3.1), which is a plausible route by which it could matter without changing the
function at init.

**Result (e2): refuted, and not by a null.** Giving a random model proc's per-tensor `||W||`
in blocks 1-8 scores **83.10 +/- 0.68** — **1.24 points BELOW random** (84.52), nowhere near p.

| arm | blocks 0-8 | blocks 9-11 | mean | vs r |
|---|---|---|---|---|
| r | random | random | 84.52 | — |
| c1 | random, **rho** matched to proc | random | 84.42 | +0.22 |
| **e2** | random, **\|\|W\|\|** matched to proc | random | **83.10** | **-1.10** |
| e1 | proc **shuffled** (norms exact) | proc | 85.15 | +0.95 |
| p | proc | proc | 85.37 | +1.17 |

So installing proc's weight-norm profile is *actively harmful*, not merely insufficient. Note
also that c1 and e2 come apart: matching what the early blocks **write** (rho) is exactly
neutral, while matching **how large their weights are** costs 1.24. The two are dissociable,
and neither reproduces proc.

**What this leaves.** e2 differs from e1 in two ways — random directions instead of proc's
shuffled values in 0-8, *and* random instead of proc weights in 9-11 — so e1's near-p result
cannot yet be attributed. The remaining candidate is the property shuffling preserves and
Gaussian rescaling destroys: the **marginal distribution of proc's weight values**
(heavy-tailed vs Gaussian at matched norm).

The arm that separates these is **e1 with random blocks 9-11** (`--weight_shuffle` on 0-8 plus
`--random_blocks 9,10,11`; no new code):

- vs **e2** (same random 9-11): isolates value distribution from norm profile.
- vs **e1** (same shuffled 0-8): isolates whether proc's real blocks 9-11 were doing the work.

Because permuting preserves the multiset of values, **e3 and e2 have identical per-tensor
norms**; they differ only in the shape of the distribution inside each tensor. That makes
e3 - e2 a single-variable contrast.

### 3.8.1 Proc's early weights are strongly heavy-tailed

Measured directly on `pr_27291166_final.pth` (CPU only, no training):

| tensor | kurtosis | \|max\|/std |
|---|---|---|
| blocks.0.mlp.fc2.weight | 5.27 | 14.5 |
| blocks.4.attn.qkv.weight | 3.82 | 8.9 |
| **blocks.4.mlp.fc1.weight** | **34.16** | 40.3 |
| **blocks.4.mlp.fc2.weight** | **323.27** | 64.0 |
| blocks.8.mlp.fc2.weight | 8.57 | 13.1 |

Over the 36 weight matrices of blocks 0-8: mean kurtosis **38.69**, max **368.22**, with 20/36
above 5 — against Gaussian's 3.0. So proc's early blocks are sparse with a few very large
entries, and e2 replaced exactly that with Gaussian mass at matched norm. That is a far larger
change than "same norms" suggests, and is a plausible reason e2 *hurt* rather than being
neutral.

Note `blocks.4.mlp.fc2` (kurtosis 323, max 64 sd) is the same tensor behind the `rho_mlp` =
2.55 outlier that forced c1's scale factor of 12.65 (§5.1) — two independent measurements
pointing at the same handful of weights.

### 3.8.2 Does the contribution live in a few extreme weights? (f1)

f1 winsorises the top **0.1%** of weights per tensor in blocks 0-8 and then rescales each
tensor back to its **original norm** — the rescale matters, since §3.8 showed that changing
`||W||` is itself harmful, and without it this would repeat e2's confound.

Clipping fraction vs resulting distribution (dry run on the checkpoint):

| clip | mean kurtosis | max |
|---|---|---|
| none | 38.69 | 368.22 |
| **0.1%** | **6.68** | **14.34** |
| 1.0% | 4.97 | 9.16 |
| 5.0% | 3.43 | 5.16 |

0.1% removes most of the heavy tail while leaving 99.9% of the weights exactly as trained, so
it is the sharpest available version of the test. Every previous arm manipulated *aggregate*
statistics; f1 targets a specific handful of parameters.

All three fractions are running as a **dose-response sweep**:

| arm | clip | mean kurtosis after | interpretation if p is retained |
|---|---|---|---|
| f1 | 0.1% | 6.68 | the extreme tail is incidental |
| f2 | 1.0% | 4.97 | most of the heavy tail is incidental |
| f3 | 5.0% | 3.43 (near-Gaussian) | the value distribution is irrelevant altogether |

A sweep is more informative than any single point: if the advantage decays monotonically with
clip fraction, the contribution is spread through the tail; if it survives even f3 — where the
weights are essentially Gaussian at proc's norms — then the distribution is ruled out
completely, and f3 becomes the direct complement of e2 (same near-Gaussian distribution, but
reached by clipping proc rather than by rescaling random draws).

Note f3 and e2 end up in a similar distributional place from opposite directions, so
**f3 >> e2 would be informative on its own**: it would mean what matters is not the shape of
the distribution but which specific weights hold the mass.

### 3.8.3 Results: the distribution is irrelevant, and the outliers are *harmful*

| arm | blocks 0-8 | blocks 9-11 | mean | sd | vs r | vs p |
|---|---|---|---|---|---|---|
| e2 | random, \|\|W\|\| -> proc | random | 83.10 | 0.68 | -1.10 | -2.27 |
| **e3** | proc **shuffled** | random | **83.22** | 0.45 | -0.98 | -2.15 |
| e1 | proc **shuffled** | proc | 85.15 | 0.27 | +0.95 | -0.23 |
| **f1** | proc, top 0.1% clipped | proc | **85.78** | 0.26 | +1.58 | **+0.41** |
| **f2** | proc, top 1% clipped | proc | **86.01** | 0.09 | +1.81 | **+0.63** |
| **f3** | proc, top 5% clipped | proc | **85.76** | 0.71 | +1.56 | **+0.39** |

**(a) The value distribution is not the mechanism.** e3 lands on e2 (83.22 vs 83.10, both sd
~0.5), so proc's own shuffled values and Gaussian draws at matched norms are
indistinguishable. The last candidate from §3.8 is ruled out.

**(b) e1's result was carried by its blocks 9-11, not its early blocks.** e1 and e3 differ
*only* in blocks 9-11 (proc vs random), and that difference is worth **1.93**. §3.7's reading
— "shuffling 0-8 keeps the proc advantage, so structure is irrelevant" — was therefore too
generous: what it actually showed is that shuffled early blocks are harmless *given* proc
blocks 9-11.

**(c) There is an interaction.** e3 (83.22) sits *below* random (84.20): proc-like early
blocks actively hurt when the late blocks are random. Proc's blocks 0-8 are only beneficial
paired with proc-like blocks 9-11. Compare b2 (real proc 0-8 + random-but-magnitude-matched
9-11) at 85.64 — the pairing survives if the random late blocks are rescaled, which is exactly
the intervention a1/b2 apply.

**(d) The outliers are mildly harmful, not load-bearing.** All three clip fractions **beat**
proc init, with f2 at 86.01 +/- 0.09 (+0.65 over p, the tightest spread of any arm here;
seeds 86.14 / 86.20 / 86.06). The effect is flat across 0.1% / 1% / 5%, so it is not
dose-dependent — even minimal clipping captures it. Clipped proc (86.01) is within noise of
a1 (86.26), the best arm, reached from the opposite direction.

This is the first *positive*, actionable result of the investigation: **winsorising the top
~1% of weights in proc blocks 0-8, with norms restored, improves proc init by ~0.65 on
IN-100/ViT-S.** It is also a practical recipe rather than an explanation, and it should be
checked on IN-1k before being relied on.

> **Superseded by §3.11.** On a second proc checkpoint (kdyck4) all three clip fractions land
> *below* proc init (-0.32, -0.08, -0.17). The +0.65 is a property of `pr_27291166`'s
> unusually heavy tail (kurtosis 38.7, max 368) and does not generalise. Read this section as
> "proc's extreme weights are mildly harmful when it has them", not as a recipe.

**Corroboration from existing runs.** e1 and the f-arms are a clean control pair: both destroy
proc's early-block structure and both keep proc blocks 9-11 and the original norms, but only
clipping *removes* the extreme values (shuffling merely relocates them).

| arm | blocks 0-8 | outliers | mean |
|---|---|---|---|
| e1 | values permuted | **kept** | 85.15 |
| f1 | top 0.1% winsorised | **removed** | 85.78 |
| f2 | top 1% winsorised | **removed** | 86.01 |

The +0.63 to +0.86 gap is attributable to the outliers alone. Note also that a1 (86.26, no
proc weights at all) and f2 (86.01, proc minus outliers) converge on ~86.0-86.3 from opposite
directions, while every arm with *intact* proc early blocks sits at 84.9-85.5 — as if the
outliers impose a ceiling.

### 3.8.4 rho is blind to the clipping benefit

Running the trajectory analysis on the clip arms shows they converge to the **same** late-block
attractor as untouched proc. `rho_attn` block 11, seed 0:

| epoch | r | p | e1 | f1 | f2 | f3 | e3 |
|---|---|---|---|---|---|---|---|
| 5 | 0.073 | 0.795 | 0.300 | 0.846 | 0.946 | 0.845 | 0.075 |
| 70 | 0.061 | 0.279 | 0.370 | 0.295 | 0.336 | 0.355 | 0.114 |
| 299 | 0.246 | **0.510** | 0.624 | **0.499** | **0.604** | **0.587** | 0.302 |

Block 8 is tighter still (p 0.380 vs f-arms 0.334-0.343). So clipping blocks 0-8 leaves the
attractor essentially untouched while gaining +0.37 to +0.65 in accuracy.

**The erasure result and the clipping result are therefore separate effects.** Erasure is a
rho phenomenon and reproduces on IN-1k; the clipping gain is invisible to rho. Further
rho-based probes cannot explain it — that would need a different observable (linear-probe or
CKA feature quality on the early blocks, gradient noise, loss-curvature). `layer_wise_stats.py`,
`linear_probe_utils.py` and `cka_utils.py` are already in the repo.

e3 behaves as a sanity check should: with random blocks 9-11 it lands at 0.302, near random's
0.246 rather than proc's 0.510 — the attractor requires proc-like late blocks, consistent with
the interaction in §3.8.3.

### 3.8.5 Do the outliers cause the erasure? (g1/g2)

If proc's outliers dominate the residual stream that pins blocks 9-11, removing them should
restore the late-block intervention's leverage. g1/g2 repeat b1/b2 on a **clipped** proc model
(1%, applied to model *and* target so they still differ only in 9-11):

| context | swing |
|---|---|
| random 0-8 | a1 - a2 = **1.72** |
| proc 0-8 | b2 - b1 = **0.56** |
| clipped-proc 0-8 | **g2 - g1 = ?** |

**Result: the outliers do not cause the erasure.**

| context | swing | |
|---|---|---|
| random 0-8 | a1 - a2 | **+1.75** |
| proc 0-8 | b2 - b1 | **+0.66** |
| **clipped-proc 0-8** | **g2 - g1** | **+0.96** |

g1 = 84.68 +/- 0.16, g2 = 85.64 +/- 0.62. The swing rises from 0.56 to 0.87, but with g2's
sd at 0.55 that +0.31 is inside one standard deviation and far from the random context's
1.73. Both arms also land on their unclipped counterparts (**g1 - b1 = -0.14**,
**g2 - b2 = +0.17**), so clipping barely moves either arm once the late-block scaling is
applied — even though the same clipping gives +0.65 over p when blocks 9-11 are left alone
(f2 = 86.01).

The rho trajectories agree. `rho_attn` block 11, ratio between the paired arms:

| epoch | b2/b1 (unclipped) | g2/g1 (clipped) |
|---|---|---|
| 0 | 5.39 | **13.10** |
| 20 | 0.88 | 1.69 |
| 70 | 0.78 | **1.05** |
| 150 | — | 1.19 |

g1/g2 start 13x apart — wider than b1/b2 — and converge to 1.05x by epoch 70. The attractor
forms regardless. **The outliers are not what pins blocks 9-11**, and the erasure and clipping
results remain two separate phenomena.

**Methodological warning: early accuracy gaps are not predictive.** At epoch 194 the gap was
+1.65, and extrapolating from the completed pairs (whose gaps *grew*: b2-b1 0.23 -> 0.56,
a1-a2 1.05 -> 1.73) suggested a final 1.8-2.7. The gap instead **shrank** to +0.87. The
early->final ratio is unstable in magnitude (1.1x-2.4x) *and* in direction. Do not read
partial-epoch comparisons, even between arms at matched epochs.

**A caveat this creates for the erasure story.** g1/g2 converge in rho (1.05x) while differing
by +0.87 in accuracy; b1/b2 converge in both. So **rho convergence does not imply functional
equivalence** — two arms can write equally loudly and still encode different things. The
inference "rho converged, therefore the intervention was erased" (used in §3.4 and §3.10) is
therefore not valid on its own. Those conclusions still stand because they have accuracy
agreement behind them (IN-1k: seven proc-context runs within 79.84-80.21 of p's 80.09), but
the rho argument should not be stated alone.

Implementation note: clipping here is `--clip_outlier_blocks`, orthogonal to `--init_method`
(the `clip_outlier_weights` method used by f1-f3 cannot compose with the scaling methods).
It runs at the top of the `start_epoch==0` block — before `init_method_copied_blocks` so g2's
copied blocks arrive already clipped, and before the rho measurement so the scaling sees the
clipped model.

### 3.9 Conclusion

The blocks 9-11 intervention fails in the proc context because the residual stream produced by
proc-initialised blocks 0-8 determines what the late blocks must compute, and gradient descent
finds that solution as soon as the learning rate permits. Interventions on blocks 9-11 in the
proc context are therefore not expected to persist however they are applied — which is why
b1 ~ b2 ~ p.

Ruled out as the cause:

| candidate | ruled out by |
|---|---|
| reduced plasticity of proc's large-norm weights | §3.1 — proc's late blocks move 12-25x, more than any other arm |
| effective-depth collapse | §3.1 — those blocks end as the *largest* writers |
| co-adaptation of blocks 0-8 during training | §3.4 — freezing 0-8 does not preserve the intervention |
| the LR / warmup schedule | §3.5 — shorter warmup makes erasure faster, not weaker |
| early-block **write magnitude** | §3.6 — c1 and c2 are both null |
| early-block **learned structure** | §3.7 — e1 (blocks 0-8 shuffled) stays at p |
| early-block **weight-norm profile** | §3.8 — e2 scores 1.24 *below* random |
| early-block **value distribution** | §3.8.3 — e3 lands on e2, not on e1 |
| early-block **extreme weights** | §3.8.3 — clipping them *improves* on proc (f1/f2/f3) |

Every isolated variable is ruled out, and two of the last three came back with the *opposite*
sign to the hypothesis: imposing proc's early-block norms hurts (e2), and removing proc's
extreme weights helps (f1-f3).

What §3.8.3 leaves is an **interaction rather than a property**: proc's blocks 0-8 are
beneficial only when blocks 9-11 are proc-like, and harmful otherwise (e3 < r). No
single-block-set statistic explains the gap because the effect does not live in either half
alone. The productive next question is therefore not "which statistic of blocks 0-8?" but
"what does the 0-8 / 9-11 *pairing* provide that neither half does?" — for which the
mid-training intervention (§4) is the sharper probe than any further init-time ablation.

Two results are worth carrying forward regardless of how that resolves:

1. **a1 beats proc init by ~0.9 (86.26 vs 85.37) while containing no proc weights at
   all** — random init with blocks 9-11 rescaled. Every arm with proc blocks 0-8 lands in
   85.1-85.6 whatever is done to 9-11, so proc's early blocks look like a ceiling rather than
   a benefit on IN-100/ViT-S.
2. **A depth asymmetry.** Late-block write magnitude is decisive in the random context
   (a1 - a2 = 1.72) and inert in the proc context (b2 - b1 = 0.56); early-block write
   magnitude is inert in both (c1, c2).

---

## 3.10 Does it transfer to ImageNet-1k / ViT-B?

Measured post-hoc on the **existing** IN-1k runs (all 300 per-epoch checkpoints were retained),
so this cost no training. Single seed each.

| arm | slurm_id(s) | n | top-1 | sd | vs p | rho blk11 @0 | @299 |
|---|---|---|---|---|---|---|---|
| r | 29384839/s0-s2 | 3 | 78.08 | 0.19 | -2.01 | — | — |
| p | 29377576/s0-s2 | 3 | 80.09 | 0.12 | — | 0.667 | 0.404 |
| a1 | 29388202, 29406778, 29406779 | 3 | 80.00 | 0.14 | **-0.09** | 0.652 | 0.342 |
| a2 | 29407014/s0-s2 | 3 | 77.37 | 0.72 | -2.72 | 0.131 | 0.415 |
| b1 | 29409958 | 1 | **80.05** | — | **-0.04** | **0.154** | **0.395** |

Seeds are split across slurm_ids for some arms (a1's three seeds live under three different
job ids), so group by **config**, not by job id, when collecting these.

**Transfers:**

1. **Erasure**, and it reproduces *quantitatively*. `rho_attn` block 11 at epoch 299:

   | context | pair | IN-100 | IN-1k |
   |---|---|---|---|
   | proc 0-8 | b1 / p | 0.507/0.510 = **0.99** | 0.395/0.404 = **0.98** |
   | random 0-8 | a1 / r | 0.410/0.246 = **1.67** | 0.342/0.610 = **0.56** |

   In the proc context the intervention is erased to within 2% on both datasets, despite b1
   starting 4.3x from p. In the random context the arms stay separated on both (~1.7x apart
   on IN-100, ~1.8x on IN-1k). The *direction* of the separation flips — a1 ends above r on
   IN-100 and below it on IN-1k — but presence vs absence of separation is what the mechanism
   predicts, and that is identical. This holds across a sign flip in proc's late-block profile
   (proc above random init on IN-1k, below on ViT-S/IN-100), so the attractor is not a ViT-S
   artifact.
2. **Late-block sufficiency.** a1 recovers +1.90 over random (80.06 vs 78.08) against proc's
   +1.94 — i.e. it reproduces proc init *exactly* (3 seeds each, sd 0.12).

**Does not transfer:**

3. **"a1 beats proc" is IN-100/ViT-S specific.** a1 is +0.96 above p on IN-100 but level with
   it on IN-1k (-0.04, sd 0.12 both). The §3.9 claim that proc's early blocks act as a ceiling
   should be scoped to ViT-S/IN-100. On IN-1k the honest statement is that late-block
   magnitude is *sufficient* to replace proc init, not that it beats it.

### 3.10.0 Which IN-1k runs count

**Never initialise from `results/pr_vitb_old/`.** That directory holds the deprecated
`pr_27267764_*` checkpoints and is retired; `results/pr_vitb_n/pr_6066174_final.pth` is the only
checkpoint to use for ImageNet-1k from here on.

77 of the 81 runs in `results/imnet_base/` already use it. Grouping every run by the checkpoint
it *recorded at runtime*:

| recorded checkpoint | runs | status |
|---|---|---|
| `results/pr_vitb_n/pr_6066174_final.pth` | 77 | **in scope** |
| `results/pr_vitb/pr_27267764_final.pth` | 2 (29236814, 29236815) | **deprecated**, since moved to `results/pr_vitb_old/` |
| none recorded | 2 (29236813, 29384839) | random-init baselines, expected |

The two deprecated-checkpoint runs are **not failures** — both completed 300 epochs. They are
`s[11] r[0-10] sba[norm2, mlp...]`, i.e. block-11 scaling with an attention-only proc load, and
**that arm already exists three times on the current checkpoint** (29377585 = 79.02,
29377600 = 78.89, 29388182 = 77.07). There is nothing to re-run.

**Read the checkpoint from the run, not from the script.** `run_train_ftb5.sh` and
`run_train_ftb6.sh` name `results/pr_vitb/pr_6066174_final.pth`, which does not exist — and *no
completed run ever recorded that path*. Those scripts are stale files, not evidence about run
history. The authoritative source is `pr[0].path` in
`results/imnet_base/accuracy_IMNET_BASE_<jobid>.json`.

#### Most IN-1k runs are scaled arms, not placement arms

46 in-scope runs have no surviving log. Recovering their configs from `wandb/run-*/files/
config.yaml` (matched on `slurm_id`) shows what they are:

| init_method | runs |
|---|---|
| `default` (no scaling) | 20 |
| `upscale_random_match_delta_norms` | 20 |
| `downscale_pr_match_delta_norms` | 17 |
| `upscale_random_match_attn_delta_norms` | 6 |
| `downscale_pr_match_attn_delta_norms` | 6 |

**Only 12 runs are pure proc-placement** (no `init_method`, no `skip_load_blocks`), and 10 of
those are arms already tracked here.

**A near-miss worth recording.** Scanning by `--random_blocks` alone made job 29388227 look like
a duplicate of `ftb5h` — both show `random 0-6` — while scoring 80.12 against ftb5h's 78.84.
That looked like a 1.3-point contradiction of §3.13.4. It is not: 29388227 is an
`upscale_random` arm, where `--random_blocks` describes the **target model**, not the trained
one. The same trap as §1, now confirmed for the ViT-B scripts too.

It also resolved a false alarm: apparent 2.8-point spreads at "identical" configs
(proc 8-11: 80.17 vs 77.30) are `upscale` vs `downscale` arms, **not seed variance**. The IN-1k
noise floor is not 2.8 points, and single-seed comparisons are not invalidated.

**Method for identifying any unlogged run:**

```bash
# slurm_id -> full arg set, from the local wandb cache
grep -rl "value: <jobid>" wandb/run-*/files/config.yaml
```

#### 3.10.1 The rho target is not portable across model compositions

`ftbrho` reaches rho 1.4 on a **fully random** ViT-B with a worst `fc2` factor of 81.9. Asking
for the same 1.4 on a model with only **four** proc blocks in front (`ftbcomp`: proc 0-3 copied,
blocks 9-11 scaled) needs **36x more**:

| block | current rho (mlp) | factor for target 1.4 |
|---|---|---|
| 9 | 0.0040 | 352 |
| 10 | 0.0013 | 1067 |
| 11 | 0.0005 | **2930** |

Cancelled at init. Proc's early blocks inflate the residual stream enough that the late blocks
write almost nothing relative to it, so a constant rho target that is reachable from random init
is unreachable from a proc-seeded one.

**Consequence for the recipe and for the nanochat port:** rho ~1.4 was calibrated against a
random-init residual stream. It is not a property of the architecture, and it does not survive a
change in what precedes the scaled blocks. Any port must measure rho *on the model it will
actually train*, not adopt 1.4 as a constant (§3.13.5 already says measure-then-set; this is the
sharper version — measure on the exact composition).

#### 3.10.2 Runaway factors flatten when the target is lowered

Three independent cases now show the same thing: a large target makes each scaled block inflate
the stream the next one is measured against, so factors compound with depth. Lowering the target
unwinds the compounding **superlinearly**, and the factors go flat:

| arm | change | worst `fc2` factor | profile across blocks |
|---|---|---|---|
| ftbrho | rho 1.4 -> 0.7 | 81.9 -> 7.34 | 9.4 / 27.6 / 81.9 -> 3.4 / 4.9 / 7.3 |
| ftb4jd | proc profile x0.5 | 173 -> 59.9 | 4.9 / 23.4 / 94.9 / 173 -> 2.4 / 11.4 / 44.6 / 59.9 |
| ftbcomp25 | rho 1.4 -> 0.25 | 2930 -> 40.6 | 352 / 1067 / 2930 -> 36.9 / 39.0 / **40.6** |

ftbcomp25 is the cleanest illustration: at 1.4 the factors span 8x across three blocks; at 0.25
they are flat to within 10%. **When an arm's factors run away with depth, the target is too high
for that composition** — lower it rather than accepting the largest factor.

#### 3.10.3 Seeded IN-1k comparison: the recipe recovers most, but not all, of proc init

All three quantities now have 3 seeds, so the ImageNet-1k comparison can be made properly.
Baselines were **already seeded** and are not single-seed as earlier drafts of this document
claimed: `p` is job 29377576 (80.20 / 80.14 / 79.96) and `r` is job 29384839
(78.28 / 78.04 / 77.91).

| arm | seeds | mean |
|---|---|---|
| **p** proc init | 80.20, 80.14, 79.96 | **80.09 +/- 0.12** |
| **ftbrho** recipe, rho 1.4, no checkpoint | 80.03, 79.48, 79.55 | **79.69 +/- 0.30** |
| **r** random | 78.28, 78.04, 77.91 | **78.08 +/- 0.19** |

| comparison | delta | Welch |
|---|---|---|
| recipe vs random | **+1.61** | **7.9 sigma** |
| proc vs random | **+2.01** | 15.7 sigma |
| **recipe vs proc** | **-0.41** | **2.2 sigma** |

**The correct claim: the checkpoint-free recipe recovers ~80% of procedural pretraining's gain
over random (1.61 of 2.01), and is ~2 sigma short of matching it.** Earlier drafts reported
"within 0.03 of proc init" from ftbrho seed 0 (80.03) against p's mean; seed 0 was the best of
three draws and that comparison was not like-for-like.

The seed std of **0.29** for a ViT-B arm also calibrates every single-seed number in §3.10:
differences below ~0.6 are not interpretable, differences above ~1.0 are.

#### 3.10.4 The composition beats procedural pretraining, at both scales

§3.13.4 splits the effect in two — early blocks want proc's *learned weights*, late blocks want
*write magnitude* — and predicts the two should compose. Tested at both scales:

| | IN-100 / ViT-S | IN-1k / ViT-B |
|---|---|---|
| **proc 0-10 + block 11 calibrated** | **87.25 +/- 0.09** (n=3) | **80.63 +/- 0.18** (n=3) |
| proc 0-10 + block 11 random | 85.89 +/- 0.18 (n=3) | 80.37 +/- 0.12 (n=3) |
| **p** proc init | 86.15 +/- 0.17 | 80.09 +/- 0.12 |
| r random | 84.55 +/- 0.36 | 78.08 +/- 0.19 |
| arms | `q1` / `q2` | `ftb1i` / `ftbcomp11` |

| comparison | IN-100 | IN-1k |
|---|---|---|
| **calibrated composition vs proc init** | **+1.10** (n=3) | **+0.54** (4.4 sigma) |
| block-11-random vs proc init | **-0.26** (n=3) | **+0.28** (2.9 sigma) |

**"Proc everywhere except the last block, with that block calibrated" beats procedural
pretraining at both scales**, and more strongly at ViT-S. `q2` at 87.25 is the best IN-100 arm in
this document — above `a1` (86.59) and `n14` (87.02). The composition is therefore a property of
the architecture, not of ViT-B or ImageNet-1k.

**But the two steps do not decompose the same way, and one sub-claim does not replicate:**

| step | IN-100 | IN-1k |
|---|---|---|
| remove proc from block 11 | **-0.26** | **+0.28** |
| then calibrate that block | **+1.36** | **+0.26** |

An earlier draft claimed from `ftb1i` that proc's learned weights are *worse than random* in
block 11. That holds at ViT-B (+0.30, 3.5 sigma) and **not** at ViT-S, where the same step is
0.23 *below* proc init. **Calibration does nearly all the work on IN-100 and about half on
IN-1k** — consistent with everything else in this document: IN-100 is the dataset where late-
block magnitude is the whole story and proc's weights contribute little (§3.13.3).

The `ftbcomp25` pair says the same thing from the other end on IN-1k: proc 0-3 alone
(79.58 +/- 0.25) and the recipe alone (79.69 +/- 0.30) both fall short of proc init, and
composing them reaches it (80.16 +/- 0.12, +0.59 over its own control at 3.8 sigma).

**Both `q` arms are now n=3**, with q2's three seeds spanning only 0.10 (87.34 / 87.28 / 87.24).
Several `q` seeds failed repeatedly en route; the cause was the rendezvous-port collision in
§5.12, not the arms. `ftb4jd` at 80.11 +/- 0.04 settles the arm that diverged
three times identically as `ftb4j` until its target was halved.

**Divergence note: ftbcomp11's block-11 `fc2` factor was 267**, above `ftb4j`'s 173 which
diverged three times — and it trained cleanly on all three seeds. **Raw factor magnitude does not
predict divergence; compounding across blocks does.** ftb4j scaled four blocks in a cascade, each
inflating the stream the next was measured against; ftbcomp11 and q2 scale one block, so there is
nothing to compound. This supersedes the caution in §5.4 and §3.10.2 for the single-block case.

#### 3.10.5 What proc's early blocks contribute: not rho, not norms — and the two datasets differ in kind

> **[SUPERSEDED by §0, 2026-08-30]** The elimination in this section is sound as a record of
> which arms were run and what they scored, but its framing — that the early-block benefit
> lives in the weight *value distribution* — is a proxy. Every arm here differs in
> `value_write`, and that scalar accounts for 83% of the variance across all of them. Read §0.


A weight **shuffle** (`randperm` over the flattened tensor, `utils.py:1210`) is a near-perfect
control: it preserves the per-tensor value multiset **exactly** — norm, variance, kurtosis, every
moment — and destroys only the arrangement (row/column organisation, singular-value spectrum,
effective rank, any learned feature detector). Four transplants separate the candidates:

| what is transplanted into blocks 0-8 | IN-1k arm | score | vs r 78.08 +/- 0.19 |
|---|---|---|---|
| **rho only** — random values, proc's write magnitude | ftb4o (0-7) | 77.27 | **-0.80** |
| **per-tensor norms only** — random directions | **ftbnorm** | **78.28 +/- 0.32** | **+0.20 (0.9 sigma)** |
| **values, structure destroyed** — proc's weights shuffled | ftb4e3 | 80.16 | **+2.08** |
| values + structure — proc intact | ftb3i | 79.99 | +1.91 |

**On IN-1k the benefit is none of rho, norms, or structure.** Calibrating early blocks to proc's
write magnitude is harmful. Matching proc's per-tensor norms does **nothing** (+0.20, 0.9 sigma,
3 seeds). Destroying proc's structure costs nothing (+2.08 vs +1.91). What is left is the only
thing a shuffle keeps and a norm-match discards: **the within-tensor value distribution**, worth
the **+1.88** between ftbnorm and ftb4e3. Note this is *not* "proc adds heavy tails" — at ViT-B
random init is the heavier-tailed of the two (kurtosis 9.93 vs 5.42, §3.10.6).

**IN-100 gives the opposite decomposition:**

| arm | transplanted into blocks 0-8 | IN-100 |
|---|---|---|
| **e2** | per-tensor **norms only** | **83.57** |
| **e3** | proc's **values, shuffled** | **83.43** |
| e4 | proc intact | 84.71 |
| r | random | 84.55 |

e2 and e3 agree to **0.01**: on IN-100 the norm profile fully accounts for shuffled-proc, and the
distribution adds nothing. On IN-1k norms account for nothing and the distribution accounts for
everything.

**So the datasets differ in kind, not degree.** Same intervention, opposite mechanism *and*
opposite sign: shuffled-proc costs -1.07 on IN-100 and gains +2.11 on IN-1k. This is the sharpest
form of the contradiction in §3.13.3 and it remains unexplained.

**The shuffle contradiction, now seeded on both sides.** A shuffle destroys structure while
preserving every per-tensor statistic. Its cost depends entirely on how many blocks are shuffled:

| | intact | shuffled | shuffle effect |
|---|---|---|---|
| **9 blocks** (ftb3i / ftb4e3) | 79.99 +/- 0.36 | 80.16 +/- 0.10 | **+0.17 (0.8 sigma)** — free |
| **1 block** (ftb11i / ftb11is) | 78.78 +/- 0.18 | 77.81 +/- 0.30 | **-0.97 (4.8 sigma)** — costly |

All four arms are n=3, so neither side is noise. **Destroying the learned structure of one proc
block costs ~1 point and drops it below random (77.81 vs 78.08); destroying it in nine costs
nothing.**

The natural reading is **redundancy**: with nine proc blocks the network can rebuild what the
shuffle destroys, because the surrounding blocks provide enough scaffolding; with one block among
eleven random ones there is nothing to rebuild from, so its structure is the whole contribution.

Note also that **`ftb4e3` shuffled (80.16 +/- 0.10) is statistically indistinguishable from proc
init itself (80.09 +/- 0.12)**. Nine blocks of structurally destroyed proc weights match full
procedural pretraining on ImageNet-1k.

**A third axis, under test.** Everything above varies *what* is transplanted and *where by
depth*. `ftb0a` / `ftb0m` (batch 13) split by **sublayer** — proc's attention only, or proc's MLP
only, across all 12 blocks. The rho calibration acts on `v`/`proj` and `fc2` together, so if
proc's contribution sits in one sublayer the recipe may be scaling the wrong half.

#### 3.10.6 Offline probes: what a shuffle actually changes, and what proc's weights look like

A shuffle is useful precisely because it moves **nothing** per-tensor. Every scalar statistic —
norm, variance, kurtosis, every moment — is preserved by construction, so the only measurables it
can move are structural. Measuring directly on `pr_vitb_n` (blocks 0-2, 12 weight tensors):

| stable rank (participation ratio of singular values, max 768) | |
|---|---|
| proc intact | **267** |
| proc shuffled | **673** |
| random init | **673** |

**Proc's early weight matrices are strongly low-rank, and shuffling puts them exactly on random.**
Structure is therefore real and measurable *offline*, with no training — a far cheaper screen for
candidate interventions than a 22h run.

**Correction: proc is NOT heavier-tailed than random at ViT-B — the reverse.** Earlier sections
carried "proc's weights are heavy-tailed" from §3.8.1, which measured **ViT-S on the original
checkpoint** (kurtosis 38.7). On `pr_vitb_n`, blocks 0-8, 36 weight tensors:

| | proc | random (timm) |
|---|---|---|
| kurtosis | **5.42** | **9.93** |
| % of \|w\| > 3 sd | 0.86% | 0.27% |

Random init has the *rare extreme spikes* driving kurtosis up; proc has fatter shoulders (3x the
mass past 3 sd) but shorter extreme tails. So the +1.95 between `ftbnorm` (78.32) and `ftb4e3`
(80.25) in §3.10.5 is **not** "proc adds heavy tails" — if anything it is the opposite.

**Answered — see §3.10.8. The outliers are not the mechanism; clipping them costs ~0.37.** The
original framing is kept below for the record.

**Under test (`ftbclip01` / `ftbclip1` / `ftbclip5`, 3 seeds each, queued 2026-08-21).** A fully
random ViT-B with `--clip_outlier_blocks 0-8` at fractions 0.1% / 1% / 5%. `clip_block_outliers`
winsorises the extremes and restores each tensor's original norm, so only the tail changes, and
**no checkpoint is involved**. Against r = 78.08 +/- 0.19:

- **~80** at any fraction => random init's extreme outliers are the mechanism, and the *early*
  half of the recipe becomes checkpoint-free — the outcome that matters most for the nanochat
  port
- **~78.2** => outliers are not it, and what remains is the low-rank structure above (267 vs 673)

Note IN-100's `f1`/`f2`/`f3` clip *proc's* weights, not random's, so they do not answer this.

#### 3.10.7 Sublayer ablation: attention and MLP each carry part of it

Everything else in this document varies *what* is transplanted and *where by depth*. `ftb0a` and
`ftb0m` split proc's contribution by **sublayer** instead, using `--skip_load_block_attributes`
to drop half of every block from the checkpoint load:

| arm | proc weights loaded | random | score |
|---|---|---|---|
| **ftb0a** | attention (`norm1`, `attn.*`) | `norm2`, `mlp.*` | **79.54** (295/300) |
| **ftb0m** | MLP (`norm2`, `mlp.*`) | `norm1`, `attn.*` | **78.99** |
| p | everything | — | 80.09 +/- 0.12 |
| r | nothing | everything | 78.08 +/- 0.19 |

**Neither sublayer alone accounts for proc's +2.01, and both are worth most of the way there.**
Attention carries somewhat more (+1.46) than MLP (+0.91), but the contribution is **split across
both** rather than concentrated in one. Both single seed, and the 0.47 gap between them is ~1.6
sigma at the measured ViT-B seed std of 0.29 — suggestive of an attention lead, not established.

For the recipe this is mildly unwelcome: there is no single sublayer to target. The rho
calibration already acts on `v`/`proj` and `fc2` together, and this says that is the right choice
rather than an over-broad one.

#### 3.10.8 Random init's outliers are NOT the mechanism — clipping makes it worse

§3.10.6 measured random init as *more* heavy-tailed than proc at ViT-B, and the tails sit
entirely in the MLP:

| tensor | proc kurtosis | random kurtosis | proc max\|z\| | random max\|z\| |
|---|---|---|---|---|
| attn.qkv | 4.17 | **3.00** | 7.7 | 5.1 |
| attn.proj | 3.83 | **3.00** | 7.2 | 5.0 |
| **mlp.fc1** | 6.37 | **21.57** | 20.2 | **36.6** |
| **mlp.fc2** | 7.29 | **17.00** | 28.1 | **36.7** |

Attention is *exactly* Gaussian at random init; the MLP carries **36 sigma** outliers. Proc is
the reverse: mildly heavy attention, moderate MLP. So the two distributions differ specifically
as "tight with rare extreme spikes" (random) versus "broader shoulders, no extreme spikes"
(proc, 0.86% of mass beyond 3 sd against random's 0.27%).

**Prediction and result.** A fully random ViT-B with blocks 0-8 winsorised (norm restored, so
only the tail changes), at three fractions. The prediction recorded before the runs was
78.5 / 78.3 / 77.8 with a monotone decline, clip01 possibly above random.

| arm | clip | seeds | mean | vs r 78.08 +/- 0.19 |
|---|---|---|---|---|
| ftbclip01 | 0.1% (>3.29 sd) | 78.32, 77.97, 76.98 | **77.76 +/- 0.69** | **-0.32** |
| ftbclip1 | 1% (>2.57 sd) | 77.62, 77.95, 77.65 | **77.73 +/- 0.16** | **-0.35** |
| ftbclip5 | 5% (>1.96 sd) | 3 seeds | **77.59 +/- 0.55** | **-0.49** |

**Both completed arms came in below the predicted range, and below random.** Removing random
init's extreme outliers *costs* ~0.37; there is no monotone ordering between 0.1% and 1%. The
"random's outliers are harmful, proc simply has fewer" hypothesis is dead, and the sign is
opposite to what it predicted.

**What this closes off.** Every low-order summary of the weight distribution now fails to
reproduce the early-block benefit:

| property transplanted into blocks 0-8 | result | vs random |
|---|---|---|
| write magnitude rho (ftb4o) | 77.27 | **-0.80** |
| per-tensor norms (ftbnorm) | 78.28 | **+0.20** |
| extreme tail removed (ftbclip01/1) | 77.76 | **-0.32** |
| learned structure destroyed, values kept (ftb4e3) | 80.16 | **+2.08** |

Only the actual values work, and they work with their **arrangement destroyed**. By elimination
the necessary and sufficient property is the **full weight distribution** — not its norm, not its
tails, not its structure. Making the early half checkpoint-free would therefore require
*parameterising and sampling* that distribution, not applying a moment-matching correction: a
substantially larger lift than the late-block recipe, which is why the checkpoint-free method
stays late-block-only for now.

*Caveat 1: ftbclip01's spread is 0.66, well above the 0.29 ViT-B seed s.d., driven by s2 at
77.07. Its -0.36 is softer than ftbclip1's, where three seeds span 0.28.*

*Caveat 2: the random-init kurtosis figures above are **averages over blocks 0-8 and the tails are
not spread uniformly**. Block 0's random `mlp.fc1` measures kurtosis **3.00** — perfectly
Gaussian — against the 21.57 block-average (observed in the `ftbqm` init log). So random init's
36 sd MLP outliers are concentrated in particular, deeper blocks, and the clip arms were
therefore doing very different amounts of work block to block. This does not change the
conclusion (clipping costs ~0.37 at every fraction) but it does mean "random is heavy-tailed in
the MLP" is a statement about some blocks, not all.*

#### 3.10.9 Two arms testing the elimination (launched 2026-08-24)

> **[SUPERSEDED by §0, 2026-08-30]** Everything in 3.10.9.x is a lab notebook of a search that
> converged on the wrong variable four times: the value distribution, then the LayerNorm
> parameters, then the attention logit scale, then the qk/v ratio, then the v slice. All five
> are proxies for the attention write magnitude (§0). The **numbers** in these subsections are
> correct and still worth reading; the **conclusions** are not. Two specific retractions are
> flagged inline at 3.10.9.11 and 3.10.9.12.


**`ftbqm` — the positive control for "the full distribution is sufficient".** New init method
`quantile_match_target_blocks` (main.py): take a random model, sort each 2-D weight tensor in
blocks 0-8, and write the checkpoint's sorted values into the random tensor's **rank order**. The
result carries proc's exact value multiset — hence norm, variance, kurtosis, full histogram — in
an arrangement inherited from the random init, so none of proc's structure survives. Unit-tested
before launch: multiset, norm and kurtosis reproduce the donor exactly.

It should land on `ftb4e3`'s **80.16 +/- 0.10**. If it does not, the elimination argument above is
wrong somewhere.

A `--quantile_source parametric` flag substitutes a Student-t fitted per tensor to the
checkpoint's kurtosis and rescaled to its norm. That is the *informative* variant — it asks
whether a **parameterised** distribution suffices, which is what decides whether the early half
can ever be checkpoint-free. Not yet run.

**`ftbcomp1` — composition with a single proc block.** proc **block 0 only** copied in, blocks
9-11 calibrated, against `ftbcomp25`'s proc 0-3 with the same treatment (80.16 +/- 0.12).

If the two mechanisms compose additively over random (78.08): proc block 0 alone is worth +0.69
(ftb11i) and the recipe alone +1.58 (ftbrho), predicting **80.43**. Landing there would mean the
early contribution **saturates at the first block** for composition purposes, making the method
far cheaper — one block of proc weights rather than four. Landing nearer 79.5 would mean the
extra blocks do real work and the §3.13.3 ramp governs the composition too.

**Verified at launch** (both arms, before committing 22h x 6):

- `ftbqm`: 36 tensors matched (9 blocks x 4 weight matrices), blocks **0-8 only**, `|W|` equal to
  the target to 4 d.p. (98.4554 -> 98.4554), kurtosis moved from random's 3.00 to proc's
  per-tensor values. The core operation was also unit-tested standalone: multiset, norm and
  kurtosis reproduce the donor exactly with the arrangement inherited from random.
- `ftbcomp1`: `copied_blocks='0'`, worst `fc2` factor **40.5** — half of ftbrho's 81.9 which
  trained cleanly, far below ftb4j's diverging 173 — and flat across blocks 9-11
  (36.4 / 38.7 / 40.5), the signature of a target low enough that compounding does not build.

**A bug caught before launch, worth repeating.** The first `ftbqm` script was derived from the
clip template and inherited `--random_blocks "0..11"`. For `quantile_match_target_blocks` that
flag randomises the **target** as well, so the donor values would have been random rather than
proc. It would have run for 22h and produced a plausible-looking, meaningless number. Rebuilt
from `ftbnorm`, which has the correct structure (no `--random_blocks`; the model is made random
via `pr_load_model(path="")` inside the init method). **When deriving an arm from a template,
check every inherited flag against the new init method's semantics** — §1's flag-semantics trap
again.

**Note on why rho = 0.25 and not 1.4.** At rho 1.4 the block-11 `fc2` factor is **2884** with a
single proc block, against **2930** with four (§3.10.1). **Block 0 alone inflates the residual
stream as much as blocks 0-3 do** — the inflation is essentially all from the first block, and
rho 1.4 is unreachable either way. 0.25 also makes this a clean one-variable comparison against
ftbcomp25.

##### 3.10.9.1 Results (2026-08-26). Both predictions missed; one of them matters.

All 300-epoch, ViT-B, IN-1k, 3 seeds each. (`ftbqm` seed 1 completed 2026-08-26 and is now
included; the two-seed reading was 78.09 +/- 0.00.)

> **Convention: LAST-EPOCH test top-1** (see the banner at the head of this document). The
> whole document was migrated from max-over-epochs to last-epoch on 2026-08-26.

| arm | per-seed (last-epoch) | top-1 | predicted | vs prediction |
|---|---|---|---|---|
| `ftbqm` | 78.094, 78.294, 78.088 | **78.16 +/- 0.12** | 80.17 | **−2.01** |
| `ftbcomp1` | 80.186, 79.734, 80.018 | **79.98 +/- 0.23** | 80.43 | −0.45 |
| *r* random | 78.284, 78.040, 77.912 | 78.08 +/- 0.19 | — | — |
| *p* proc | 80.174, 80.144, 79.958 | 80.09 +/- 0.12 | — | — |

**`ftbqm` lands on the random baseline: 78.16 +/- 0.12 against 78.08 +/- 0.19, a difference of
+0.08 at Welch p = 0.57.** Proc's
full value multiset, written into blocks 0-8 in a random arrangement, is worth **nothing**. This
directly contradicts §3.10.5/§3.10.6, whose conclusion — "shuffled values reproduce the benefit
entirely, so the value distribution is necessary *and sufficient*" — rested on `ftb4e3`'s 80.17.
Two constructions intended to be the same thing differ by 2.01 points, so **the sufficiency claim
is retracted pending resolution.** The elimination table in §3.10.5 stands as far as it *rules
out* mechanisms (rho-only, norm-only, clipping all fail); what it identified as the surviving
mechanism does not survive.

**Where the two arms actually differ.** Both give blocks 0-8 proc's multiset in a random
arrangement and leave 9-11 random. Checked in the scripts, `main.py`, `utils.pr_load_model` **and
the run logs** (2026-08-26):

| | `ftb4e3` (80.16) | `ftbqm` (78.09) |
|---|---|---|
| base model | proc checkpoint, 9-11 re-randomised | random (`pr_load_model(path="")`) |
| blocks 0-8, 2-D weights | proc multiset, `randperm` order | proc multiset, random's rank order |
| blocks 0-8, **1-D params** (LayerNorm gains/biases, `qkv.bias`, `proj.bias`, `fc*.bias`) | **proc, shuffled** | **random** (`p.dim() < 2` is skipped) |
| `patch_embed`, `pos_embed`, `cls_token` | **random** | **random** |
| final `norm` | **random** | **random** |
| `head` | **random** | **random** |

> **Correction (2026-08-26). The embeddings hypothesis below was wrong and is withdrawn.**
> An earlier version of this section claimed `ftb4e3` inherits proc's frozen patch/pos/cls
> embeddings while `ftbqm` does not, and named that the leading explanation for the 2.07 gap.
> It does not. `utils.pr_load_model` (utils.py:753) **deletes** `head.weight`, `head.bias`,
> `cls_token`, `pos_embed`, `patch_embed.proj.weight` and `patch_embed.proj.bias` from any
> checkpoint whose filename contains `pr`, and `--skip_norm true` (set in every arm) additionally
> deletes `norm.weight` / `norm.bias` (utils.py:860). Confirmed in the stdout of `ftb4e3`,
> `ftbqm` and `ftb3i`, all of which print the six `Removing key ...` lines. **No arm in this
> document has ever inherited procedural embeddings or a procedural final norm.** Every arm
> trains with randomly-initialised, and then frozen, patch/pos embeddings.

That leaves **exactly one** difference (the second entry below was withdrawn on inspection):

1. **The 1-D parameters in blocks 0-8** — 2 LayerNorm gain/bias pairs and 4 biases per block,
   54 tensors over nine blocks. `ftb4e3` carries proc's values for these (shuffled); `ftbqm`
   leaves them at random init because `quantile_match_target_blocks` skips `p.dim() < 2`.
   This is now the leading candidate by elimination, and it points at a **normalisation-gain**
   effect — much closer to the late-block rho story, which would unify the two halves of the
   document rather than splitting them.
2. ~~**How the arrangement is randomised.**~~ **Withdrawn 2026-08-26 — this is not a difference.**
   `ftb4e3` uses a uniform `randperm`; `ftbqm` writes proc's sorted values into the rank order of
   the random tensor. Those are the *same operation in distribution*: the random tensor is iid, so
   by exchangeability its rank order **is** a uniformly random permutation. Verified numerically on
   `blocks.0.attn.qkv.weight` over 6 draws — stable rank 310.9 +/- 2.0 (rank map) vs 310.8 +/- 1.3
   (uniform), row-norm variance 0.00414 vs 0.00418, both inside seed noise, against proc intact at
   stable rank 11.5 and row-norm variance 0.266. The planned "uniform-shuffle 2-D only" control is
   therefore unnecessary and has been cancelled before launch, saving 66 GPU-hours.

**Next experiment (not yet run, cheap and decisive):** `ftbqm` plus proc's 1-D parameters in
blocks 0-8, nothing else changed. Landing near 80.16 isolates (1) and makes the LayerNorm gains
the mechanism; staying near 78.09 leaves (2) — the permutation type — as the explanation, which
would be a much stranger result. One arm, 3 seeds.

This is also the figure in `plots/out/fig5_mechanism.png`, which presents the four comparable
arms as a preserved-property matrix.

##### 3.10.9.2 `ftbqm1d` — the decisive arm (launched 2026-08-26)

`ftbqm` plus proc's **1-D parameters** in blocks 0-8, nothing else changed. Jobs
**29501773/s0, 29501774/s1, 29501775/s2** (`SLURM_ID` pinned to 29501773), ViT-B, IN-1k,
300 epochs, 3 seeds.

**New flag `--quantile_1d_mode skip|shuffle`** (main.py). Default `skip` reproduces the original
`quantile_match_target_blocks` behaviour exactly, so no existing arm changes.

**Why the 1-D params are shuffled and not quantile-matched.** Quantile matching is unusable on
them: at random init every LayerNorm weight is 1.0 and every bias 0.0, so the tensor is
**constant**, `argsort` is degenerate, and a rank map would write the donor's values in **sorted**
order — inventing a monotone structure present in neither model. Unit-tested before launch: a
rank map onto a constant returns the donor sorted ascending, while a uniform `randperm` preserves
the multiset and the norm exactly. `shuffle` therefore uses a uniform permutation, which is also
precisely how `ftb4e3` treats these tensors, making the comparison one-variable.

**Verified at launch** (all three seeds, before committing 22h x 3):

- **72 1-D tensors** shuffled = 8 per block x 9 blocks (`norm1.weight/bias`, `norm2.weight/bias`,
  `attn.qkv.bias`, `attn.proj.bias`, `mlp.fc1.bias`, `mlp.fc2.bias`), every `|W|` equal to the
  target to 4 d.p. (e.g. `blocks.0.norm1.weight` 9.2751 -> 9.2751).
- **36 2-D tensors** quantile-matched, identical to `ftbqm`.
- Blocks **0-8 only**; no `--random_blocks` inherited (the trap of 3.10.9); seeds write to
  `s0/s1/s2` under the pinned id.

**Prediction, recorded before results.**

| outcome | reading |
|---|---|
| lands near **80.16** (`ftb4e3`) | the 1-D parameters — LayerNorm gains and biases — ARE the early-block mechanism. The weight matrices contribute nothing beyond their norms, and the effect is a normalisation-gain effect, which unifies it with the late-block rho story. |
| stays near **78.09** (`ftbqm`) | the 1-D params are not it either, and the remaining difference is the *permutation type* — uniform `randperm` versus the rank order of the random tensor. That would be a strange result and would need its own arm. |
| lands **in between** | both contribute; the split would need a 1-D-only arm (no quantile matching on the 2-D weights) to apportion. |

The third outcome is the one to watch: it is the only one that needs a further arm.

##### 3.10.9.3 The 1-D decomposition — three arms (launched 2026-08-26)

Once `ftbqm1d` was designed it became clear one arm cannot answer the question, because taking
proc's 1-D params changes **two** things at once: the values become proc's, *and* they stop being
degenerate (random init sets every bias to exactly 0 and every LayerNorm gain to exactly 1). Three
arms now separate that, all `quantile_match_target_blocks` on blocks 0-8, 3 seeds each:

| arm | jobs | 1-D treatment | isolates |
|---|---|---|---|
| `ftbqm1d` | 29501773/4/5 | all 72 1-D tensors, proc's **exact values**, uniform permutation | do the 1-D params matter at all? |
| `ftbqmln` | 29504032/3/4 | the 36 **LayerNorm** tensors only (added 2026-08-26) | completes the partition; rho-matched to `ftb4e3` |
| `ftbqmbias` | 29501780/1/2 | the 36 **non-LayerNorm biases** only | biases vs LayerNorm gains |
| `ftbqm1dpar` | 29502416/7/8 | all 72, but a **Gaussian matched to proc's mean and std** per tensor | proc's *values* vs merely *non-degenerate* 1-D params |

**Two new flags, orthogonal by design.** `--quantile_1d_mode skip|shuffle|bias|layernorm` selects
*which* 1-D tensors are touched (`bias` and `layernorm` partition `shuffle` exactly: 36 + 36 = 72,
verified disjoint). `--quantile_1d_source empirical|parametric` selects *what* is written into
them. Both default to the pre-existing behaviour (`skip` / `empirical`), so no earlier arm changes.

**Why `ftbqm1dpar` is the arm that decides what the paper claims.** If it reproduces `ftbqm1d`,
the early-block effect is **not transfer at all**: it says the ViT default of zero biases and unit
LayerNorm gains is simply a poor init, and roughly any non-degenerate 1-D params of the right
scale beat it. That is a bigger result but a different paper. If it fails while `ftbqm1d`
succeeds, proc's actual 1-D values carry something a moment-matched draw does not — verified at
launch that the draw destroys exactly the higher-order structure: `blocks.0.mlp.fc1.bias` has
kurtosis **12.36** in proc and **3.16** in the parametric draw, with mean and std matched to 3
decimal places.

**A three-point ladder already exists for the LayerNorm gains**, which makes this sharper than it
looks. Proc's `blocks.0.norm1.weight` has mean 0.310, std 0.126:

| arm | LayerNorm gains in blocks 0-8 | result |
|---|---|---|
| random init | all exactly 1.0 (mean 1.0, std 0) | 78.08 |
| `ftbnorm` | all exactly 0.335 — norm-matched, **zero spread** | 78.28 (+0.20) |
| `ftbqm1dpar` | mean 0.310, std 0.126, Gaussian | 78.35 (+0.28) |
| `ftbqm1d` | proc's exact 768 values, shuffled | 78.50 (+0.42) |

`ftbnorm` already rules out the overall *scale*: setting every gain to proc's norm-implied 0.335
is worth nothing. So if the 1-D params are the mechanism, it is the **spread across channels**,
and `ftbqm1dpar` vs `ftbqm1d` then asks whether Gaussian spread suffices or proc's specific
per-channel values are needed.

###### First result: `ftbqm1dpar` is a null (2026-08-27, FINAL, n=3)

| seed | last-epoch top-1 |
|---|---|
| s0 | 77.986 |
| s1 | 78.558 |
| s2 | 78.518 |

**`ftbqm1dpar` = 78.35 +/- 0.32**, all three seeds complete at epoch 299.

| comparison | delta | Welch |
|---|---|---|
| vs random (78.08 +/- 0.19) | +0.28 | p = 0.28, **n.s.** |
| vs `ftbqm` (78.16 +/- 0.12) | +0.20 | p = 0.41, **n.s.** |
| vs `ftb4e3` (80.16 +/- 0.10) | **-1.80** | p = 0.006, significant |

Moment-matched 1-D params are indistinguishable from doing nothing, and land 1.80 short of the
arm they were built to reproduce.

**What this kills.** Had the parametric draw worked, the entire early-block effect would have
reduced to "the ViT default of zero biases and unit LayerNorm gains is simply a bad init, and
any non-degenerate 1-D params of roughly the right scale beat it". That would have been a larger
result but a different paper, and unrelated to procedural pretraining. **That reading is now
dead**: the right first two moments are not enough.

**What it leaves.** The draw matched proc's mean and std to three decimals while flattening
kurtosis (`blocks.0.mlp.fc1.bias` 12.36 -> 3.16), and it landed at rho 1.84 against `ftbqm1d`'s
1.86 (3.10.9.4). So whatever the 1-D parameters contribute is **not** their scale (ruled out by
`ftbnorm`), **not** their spread, and **not** the write magnitude. It would have to be the
specific per-channel values.

**This is not yet interpretable on its own.** `ftbqm1dpar` is the control for `ftbqm1d`, which is
its empirical-values twin — rho-matched, differing only in whether the 1-D values are proc's or
drawn. `ftbqm1d` was at 134-163/300 epochs when this was written, roughly half a day behind. Two ways
the pair can resolve:

| `ftbqm1d` outcome | reading |
|---|---|
| lands near **80.16** | the 1-D params ARE the mechanism, and specifically their per-channel values, since matched moments fail. The cleanest form of the result. |
| also lands near **78.1** | the 1-D params are not the mechanism either, and the +2.07 between `ftbqm` and `ftb4e3` is **unexplained again** — 3.10.9.1's elimination would have removed every candidate it proposed. |

##### 3.10.9.6 The 1-D partition results — all arms n=3, and the 1-D hypothesis is DEAD

`ftbqmln` and `ftbqmbias` partition `ftbqm1d`'s 72 tensors exactly (36 + 36, disjoint, verified).
**Both arms are now complete at n=3** (`ftbqmbias` finished 2026-08-27: 77.952, 78.158, 78.284).
The n=2 reading of `ftbqmbias` was 78.12 +/- 0.23; it moved by 0.01.

| arm | blocks 0-8 get | top-1 | vs random | Welch |
|---|---|---|---|---|
| `r` | nothing | 78.08 +/- 0.19 (n=3) | — | — |
| `ftbqm` | 2-D multiset only | 78.16 +/- 0.12 (n=3) | +0.08 | n.s. |
| `ftbqmbias` | + the 36 **biases** | 78.13 +/- 0.17 (n=3) | **+0.05** | p = 0.74, n.s. |
| `ftbqm1dpar` | + all 72, moments only | 78.35 +/- 0.32 (n=3) | +0.28 | p = 0.28, n.s. |
| `ftbqmln` | + the 36 **LayerNorm** tensors | **78.86 +/- 0.24** (n=3) | **+0.78** | **p = 0.013** |
| **`ftbqm1d`** | **+ all 8 1-D tensors** | **78.50 +/- 0.12** (n=3) | **+0.42** | **p = 0.039** |
| `ftb4e3` | same 8, **qkv sliced** | 80.16 +/- 0.10 (n=3) | +2.08 | — |

**`ftbqm1d` completed 2026-08-27 (78.362, 78.568, 78.572) and rules the 1-D parameters out.**
Taking all eight 1-D tensors is worth **+0.42**, which is *less* than taking only the four
LayerNorm ones (+0.78). The partition is **sub-additive**, not super-additive:

| | |
|---|---|
| non-LayerNorm biases alone | +0.05 |
| LayerNorm tensors alone | +0.78 |
| predicted if additive | +0.83 |
| **both together (`ftbqm1d`)** | **+0.42** |

`ftbqm1d` - `ftbqmln` = **-0.36** at Welch p = 0.11, so the decrement itself is suggestive rather
than established. What *is* established is `ftbqm1d` - `ftb4e3` = **-1.66 at p = 0.0001**, with
**identical 1-D content**. The two arms differ in one thing only: `ftb4e3` slices the fused qkv
and `ftbqm1d` pools it. **So the entire remaining 1.66 is attributable to proc's q,k >> v
asymmetry, and the 1-D parameters are not the mechanism.**

This also retires the super-additivity hypothesis that this section previously carried. The two
halves do not combine to +2.08; they combine to less than either the sum or the larger half.

**Within the 1-D parameters it is still the LayerNorm tensors that carry what little transfers**:
**+0.78 at p = 0.013, n = 3**, grown from +0.64 at n = 2 rather than regressing. But it is a
0.78-point effect inside a 2.08-point phenomenon, and adding the remaining 1-D tensors destroys
part of it.

**Naming precision, checked against the code and the run log.** `--quantile_1d_mode layernorm`
selects `norm1.weight`, `norm1.bias`, `norm2.weight`, `norm2.bias` — so it is the LayerNorm
gains **and** the LayerNorm biases, not gains alone. `--quantile_1d_mode bias` selects the four
*non-LayerNorm* biases `attn.qkv.bias`, `attn.proj.bias`, `mlp.fc1.bias`, `mlp.fc2.bias`. The two
sets partition the eight 1-D tensors per block exactly, which is what makes the additivity test
below valid; earlier text called them "gains" and "biases", which implied a different and
overlapping split.

Neither the *scale* of the gains (`ftbnorm`, +0.20) nor their *distribution shape*
(`ftbqm1dpar`, +0.28) substitutes for the actual per-channel values. So the transferable
component is specifically proc's 6,912 LayerNorm gain values plus their biases.

**But the halves do not sum to the whole.** They partition the tensor set exactly, so if the 1-D
params were the mechanism the two should add to `ftb4e3`'s +2.08:

| | |
|---|---|
| non-LayerNorm biases alone | +0.05 |
| LayerNorm tensors alone | +0.78 |
| **sum** | **+0.83** |
| `ftb4e3` | **+2.08** |

**They recover 40% of it.** Two possibilities were open when this was written: the halves
interact super-additively, or something depresses every quantile-matched arm.

**`ftbqm1d` settled it: the first is wrong.** Taking both halves gives **+0.42**, *below* either
the sum (+0.83) or the larger half alone (+0.78). The interaction is sub-additive, so the
remaining 1.66 to `ftb4e3` cannot come from the 1-D parameters in any combination.

**That leaves the second, which is concrete and was not anticipated.** `ftbqm`, `ftbqmln`,
`ftbqmbias`, `ftbqm1dpar` and `ftbqm1d` all pool the fused `qkv` when quantile-matching, which
erases proc's qk/v asymmetry entirely — v goes from 49% of qk's width to 100% (3.10.9.5).
`ftb4e3` does not. **The confound is therefore not specific to
`ftbqm1d` — it is shared by every arm in the fig5 elimination that uses quantile matching**, and
could be suppressing all of them by a common amount. That would make the elimination
systematically biased rather than wrong in one place.

**The gap now localises to ONE variable.** With `ftbqm1d` complete, the tightest pair in the
document is `ftbqm1d` (+0.42) against `ftb4e3` (+2.08): **identical 1-D content, identical 2-D
value multisets, differing only in whether the fused qkv is matched as one tensor or as qk and v
separately.** That single difference is worth **1.66 points at p = 0.0001**.

Everything else is eliminated by experiment: 2-D value distribution (+0.08), per-tensor norms
(+0.20), non-LayerNorm biases (+0.05), moment-matched 1-D params (+0.28), all eight 1-D params
(+0.42), write magnitude rho (**-0.80**). The qk/v asymmetry is what is left, and `ftbqm1dv`
tests it directly.

Note this is a hypothesis about v's *value distribution*, not about rho: 3.10.9.7 shows rho does
not track accuracy in the early blocks, so the mechanism by which widening v might hurt — if it
hurts at all — is unidentified. `ftbqm1dv` measures the effect without needing to name it.

**Update (3.10.9.9): the confound is larger than "a difference in v's spread".** The qkv pooling
is what holds the four pooled arms at rho 0.22-0.33 for the entire run against the sliced arms'
0.16, so it dominates their training dynamics rather than perturbing their init. Any reading of
the 3.10.5 elimination has to carry that: those four arms were run at ~1.6x the early-block write
magnitude of `ftb4e3`.

~~A sharp version of the problem: `ftbqmln` matches `ftb4e3` on rho at init...~~ **Withdrawn
2026-08-27** — that comparison used block 0 only, and the `ftb4e3` figure was itself wrong (see
3.10.9.7). Corrected, `ftbqmln` and `ftb4e3` do **not** match on rho: over blocks 1-8 they are
0.259 vs 0.156. The v-pooling inflates rho across the whole early stack, which is the substance
of 3.10.9.7.

##### 3.10.9.12 The slice arms land: it is v, not qk — and the "+2 group" is one group (2026-08-29)

> **[RETRACTED by §0]** "It is v, not qk" is right about which knob was being turned and wrong
> about why. Matching the v slice narrows `||W_v||` from 50.9 to 28.8, which multiplies the
> write magnitude by 0.57 — that, not the slice identity, is what the +0.88 main effect
> measures. The 2x2 factorial itself is valid and is reproduced in §0's grouping.


The three arms pre-registered in 3.10.9.11 finished. Figures: `fig5_mechanism.png` (now 13 rows,
with a fifth matrix column for the qk/v split), `fig10_qkv_ratio.png` (the three arms are the
diamonds, run after the relation was fitted).

| arm | what blocks 0-8 get | qk/v | logit scale | value write | vs random |
|---|---|---|---|---|---|
| `ftb4e3` | proc values, permuted within slice | 2.177 | 0.00738 | 0.514 | **+2.08** |
| `ftb3i` | proc values, not permuted | 2.177 | 0.00806 | 0.536 | **+1.91** |
| `ftbqm1dv` | both qkv slices matched | 2.177 | 0.00738 | 0.514 | **+1.40** |
| `ftbqm1dvo` | **v slice only** | 1.875 | 0.00552 | 0.514 | **+1.36** |
| `ftbqmln` | 4 LayerNorm tensors | 1.001 | 0.00552 | 0.869 | +0.78 |
| `ftbqm1dqk` | **qk slice only** | 1.155 | 0.00738 | 0.869 | **+0.58** |
| `ftbqm1d` | all 8 1-D, qkv pooled | 1.001 | 0.00552 | 0.869 | +0.42 |
| `r` | nothing | 1.001 | 0.00320 | 0.307 | 0.00 |
| `ftb4o` | rho only | 0.348 | 0.00321 | 3.949 | **-0.80** |

###### The logit-scale reading is dead

`ftbqm1dqk` and `ftbqm1dvo` were built to move the attention logit scale and the qk/v ratio in
**opposite** directions, so they discriminate. 3.10.9.11 recorded the two predictions in advance:

* logit scale is the mechanism -> `ftbqm1dqk` ~ +2, `ftbqm1dvo` ~ +0.4
* qk/v (or the write magnitude) is the mechanism -> the reverse, +1.56 and +0.47 off the fit

**Observed: `ftbqm1dvo` +1.36, `ftbqm1dqk` +0.58** (both now n=3). The ordering is the reverse of what the
logit-scale reading requires, and within 0.2 of the qk/v fit's numbers on both arms. Two further
tests sharpen it:

* `ftbqm1dvo` vs `ftbqm1dv` (v alone vs both slices): **-0.03, p = 0.94.** Matching the v slice
  alone recovers everything matching both slices does.
* `ftbqm1dqk` vs `ftbqm1d` (qk alone vs pooled): **+0.07, p = 0.57.** The qk slice buys nothing.

So the operative variable is **the v slice** — how wide proc keeps `W_v` relative to q and k, and
hence the attention write magnitude. The qk slice, which sets the logit scale, is inert here. That
is consistent with 3.10.9.10's finding that the LayerNorm gains cancel proc's 4x larger Q and K:
the logit scale never reaches the forward pass, so it cannot matter.

Note `qk_over_v` and `value_write` make the *same* prediction on this pair and are not separated by
it — v is the numerator of one and the denominator of the other. What separates them is `r`, which
has the best `value_write` (0.307, nearest proc's 0.514 of any non-winner) and scores 0.00. Over
all 13 arms `qk_over_v` gives **r = +0.955, p = 3.7e-7** (Spearman +0.921, p = 7.8e-6) against
`value_write`'s r = -0.684 and `logit_scale`'s -0.197 (n.s.).

###### Final numbers, all four cells at n = 3 (2026-08-29)

The third seed of `ftbqm1dqk` and `ftbqm1dvo` finished. Read as the 2x2 factorial it is
(`plots/fig_qkv_factorial.py`, fig12):

| | v slice POOLED | v slice MATCHED |
|---|---|---|
| **qk POOLED** | `ftbqm1d` +0.42 | `ftbqm1dvo` **+1.36** |
| **qk MATCHED** | `ftbqm1dqk` +0.58 | `ftbqm1dv` **+1.40** |

* main effect of matching **v**: **+0.88**
* main effect of matching **qk**: **+0.10**
* interaction: **-0.12**
* v-matched (n=6) vs v-pooled (n=6): **p = 2.3e-4**

Additive, one factor doing essentially everything. The extra seeds moved `ftbqm1dqk` +0.50 ->
+0.58 and `ftbqm1dvo` +1.37 -> +1.36, i.e. nothing changed qualitatively.

**Why qk is inert is a fact about the baseline, not about attention.** The fused qkv is 2/3 q and
k, so pooling all three values into one multiset hands every row roughly the q/k distribution:
pooled q and k come out at 92% and 82% of proc's, while pooled v comes out **176%** of proc's.
"Matching qk" was therefore close to a no-op by construction. The result is that proc's
transferable early-block content is **a narrow v against wide q and k** -- attention that reads
broadly and writes little.

###### The "one third missing" framing is probably wrong

`ftbqm1dv` came in at +1.40, not the ~+2.08 that 3.10.9.11 said was required — and that section's
"anything else falsifies the whole family at once" was overstated. But it is **not** a 0.68 residual
waiting to be explained:

| pair | difference | Welch p |
|---|---|---|
| `ftbqm1dv` vs `ftb4e3` | -0.68 | 0.073 |
| `ftbqm1dv` vs `ftb3i` | -0.51 | 0.158 |
| `ftbqm1dvo` vs `ftb4e3` | -0.71 | 0.233 |
| `ftbqm1dvo` vs `ftb3i` | -0.54 | 0.264 |
| `ftb3i` vs `ftb4e3` | -0.17 | 0.498 |

**No pair inside {`ftb4e3`, `ftb3i`, `ftbqm1dv`, `ftbqm1dvo`} is significantly different.** Pooled
they are **79.80 +/- 0.42 (n = 11), i.e. +1.72**, and they separate from {`ftbqm1d`, `ftbqm1dqk`}
at **p < 1e-4**. The matrix therefore reads as two groups — ~+1.7 and ~+0.45 — with the v slice
moving an arm between them, not as a ladder with a missing rung.

`ftb4e3` and `ftbqm1dv` are also *constructed* to be statistically equivalent: a rank map onto an
i.i.d. tensor is a uniform permutation (verified numerically, 3.10.9.5), so both give blocks 0-8
proc's exact per-slice value multisets in a uniformly random arrangement. Diffing their full
205-feature init vectors, nothing separates them but sampling noise — the largest relative gaps are
in per-head dispersion terms whose absolute values are ~0.004 in both.

**This reading is provisional.** The within-group ordering (2.08, 1.91, 1.40, 1.37) is monotone in
exactly the way a small real residual would be, and n = 3 cannot resolve a 0.5-0.7 effect at
sd 0.1-0.4. Distinguishing "one group" from "a shallow ladder" needs more seeds on `ftb4e3` and
`ftbqm1dv`, not a new arm. Until then, do not claim the elimination is closed, and do not claim a
third of the effect is unexplained either.

###### Caveats

* `ftbqm1dqk` and `ftbqm1dvo` are **n = 2**: the third seed of each was still training and is
  excluded from every number here rather than read early (the winners are behind until ~epoch 250,
  3.10.9.8, so a partial read is biased against exactly the arms under test). `ftbqm1dvo` vs
  `ftbqm1dqk` on its own is +0.87 at p = 0.178 — the direction is unambiguous, the magnitude is not.
* `ftb4o` remains n = 1 and is load-bearing for the `4o_ok` gate.
* Six of the thirteen arms are tied at qk/v = 1.00, so the ratio explains the split between groups
  and none of the 0.73-point spread inside the middle pack (`ftbqmln` +0.78 down to `ftbqmbias`
  +0.05). Whatever orders the middle pack is a second, smaller effect this screen does not see.

##### 3.10.9.11 A 205-statistic screen of the init: only the qk/v scale ratio survives (2026-08-28)

> **[RETRACTED by §0]** `qk_over_v` survived the four gates and reaches r = +0.955, but it is a
> proxy: in every checkpoint-derived arm, changing `||W_v||` moves the ratio and the write
> magnitude together. Held at fixed write magnitude the ratio varies 1.00 -> 2.18 with no
> effect on accuracy. The screen's METHOD (four gates, especially the ftb3i-vs-ftb4e3 shuffle
> test) is still the right tool and is what eventually found the arrangement-invariant answer;
> its winner was wrong because `value_write` was not in the 205 features as a composite.


Scripts: `plots/analyse_ckpt_differences.py` (measures), `plots/score_ckpt_features.py` (scores),
`plots/analyse_training_traces.py` (the same screen on the wandb per-layer traces),
`plots/fig_qkv_ratio.py` (fig10). Cache: `plots/cache/ckpt_diff.json`. Run:
`sbatch vitbase_runs/run_ckpt_diff.sh` (one L40S, ~25 min, 14 arms).

Everything is measured **at init on the reconstructed arm** via `measure_init_rho_arms.build_arm`,
which is the construction that is actually trained. Three families: weight-space (norms, per-slice
norms, per-head dispersion, stable rank, kurtosis, and every quantity **composed with the preceding
LayerNorm gain**, since attention sees `W_q diag(gamma_1)` and not `W_q`); forward-pass on 128 real
val images (attention logit spread, entropy, max prob, CLS mass, spatial attention distance, token
cosine similarity, participation-ratio effective rank, channel kurtosis, rho); and one backward
pass (per-block gradient norm and `||g||/||W||`). 205 scalars in total.

###### The four gates

A statistic that merely correlates is worthless here — with ten arms almost anything clears
`|rho| = 0.7`. So the screen adds three filters that no previous candidate has passed:

1. **`4o_ok`** — does it put `ftb4o` on the *far* side of random init from the winners? `ftb4o` is
   the one arm that finished **below** random (-0.80), so "more of X is better" requires ftb4o to
   have overshot X, not undershot it.
2. **`sep`** — does it separate the two +2 arms from the middle pack?
3. **`3i~4e3`** — this one is free, needs no accuracy fit, and is the strictest. `ftb4e3` is
   `ftb3i` with every tensor randomly permuted **within its slice**, and they score 80.16 and
   79.99 — the same run (p = 0.50). **Any statistic that moves between them cannot be the cause.**

###### Result: one survivor out of 205

| statistic | at init | spearman | `4o_ok` | `3i~4e3` | verdict |
|---|---|---|---|---|---|
| **`W.qk_over_v` = `\|\|W_qk\|\| / \|\|W_v\|\|`** | — | **+0.86** | **yes** | **0%** | **survives all four** |
| `W.ln1_gain_mean` | LayerNorm gain | -0.97 | no | 0% | fails 4o |
| `W.value_write` = `gain * \|\|W_v\|\| * \|\|W_proj\|\|` | — | -0.50 | no | 4% | fails |
| `W.logit_scale` = `gain^2 \|\|W_q\|\|\|\|W_k\|\|` | — | +0.30 | no | 8% | fails |
| `F.logit_std` | attention logit spread | +0.53 | no | **99%** | killed by the shuffle |
| `G.g_over_w` = `\|\|g\|\|/\|\|W\|\|` | one backward pass | -0.53 | no | **90%** | killed by the shuffle |
| `F.attn_entropy` | attention entropy | -0.45 | no | **80%** | killed by the shuffle |
| `F.eff_rank` | token effective rank | -0.39 | no | 51% | killed by the shuffle |
| `F.tok_cos` | token cosine similarity | +0.37 | no | 27% | killed by the shuffle |
| `F.rho_attn` | rho | -0.21 | no | 37% | fails (as 3.10.9.7 already found) |

`W.qk_over_v` over the ten scored arms: **Pearson r = +0.966, p = 5.4e-6; Spearman rho = +0.86,
p = 0.0014.**

| arm | `\|\|W_qk\|\|/\|\|W_v\|\|` | vs random |
|---|---|---|
| `ftb4e3`, `ftb3i`, (`p`) | **2.177** | +2.08, +1.91 |
| `ftbqmln`, `ftbqm1d`, `ftbqm1dpar`, `ftbnorm`, `ftbqm`, `ftbqmbias` | 1.001 | +0.78 … +0.05 |
| `r` | 1.001 | 0.00 |
| `ftb4o` | **0.348** | **-0.80** |

The six middle arms are numerically identical on this statistic (pooled quantile matching gives
every slice the same value distribution), so it explains the split between groups and **nothing**
of the 0.73-point spread inside the middle pack. That is a real limitation, not a rounding issue.

###### Why this one is hard to explain away

* **LayerNorm cannot undo it.** The same `gamma_1` multiplies q, k and v, so it cancels exactly
  in the ratio. Every other attention statistic measured here is contaminated by the gain — which
  is precisely why `ftbqm` (gain 1.0, so proc's large Q/K undamped) has an 11x logit scale and a
  0.62 entropy while `ftbqm1d` (gain 0.384) sits at 0.0055 and 0.989, with the two only 0.34
  points apart.
* **It is permutation-invariant**, so it passes the shuffle gate by construction — as does any
  pure scale. What makes it non-trivial is that the *other* pure scales (`value_write`,
  `logit_scale`, per-tensor norms) all fail one of the accuracy gates.
* **`ftb4o` inverts it**, and that is not an artifact of the screen: `ftb4o` scales `v` and `proj`
  up to match proc's rho while leaving q and k at random scale, which pushes v to ~3x q/k's width
  — the opposite direction from proc. Its block 8 sits at exactly 1.0 because that arm calibrates
  blocks 0-7 only, a free internal check.

###### What the screen kills

The negative results are as useful as the survivor, and several retire hypotheses that were live
in 3.10.9.8-.10:

* **Attention sharpness is irrelevant.** `ftb3i` has normalised entropy 0.1995 and logit spread
  53.1; `ftb4e3` has 0.983 and 0.745 — a 70x difference — and they train to the same accuracy.
  Proc's actual q/k arrangement produces extremely peaked attention at init and it buys nothing.
* **Token collapse is not it.** `ftb3i` has the *highest* token cosine similarity of any arm
  (0.912) and the *lowest* effective rank (2.53), the same corner as `ftb4o` (0.748), with
  opposite outcomes.
* **Small relative AdamW steps are not sufficient.** `||g||/||W||` at init: `ftb3i` 3.9e-4,
  `ftb4o` 7.3e-4, `r` 3.2e-2. The two arms with the smallest relative steps are the best and the
  worst arm on the matrix. The slow-start account of 3.10.9.8 survives as a description of the
  first 50 epochs but **cannot be the reason `ftb4o` ends up worse**.
* **rho again.** `ftb3i` 0.471 and `ftb4o` 0.462 at init, opposite outcomes — an independent
  reproduction of the 3.10.9.7 retraction on a different measurement path.

The training-trace screen (`analyse_training_traces.py`, on `acc_layer`, `attn_entropy_layer`,
`blk_act_rms_layer`, `delta_norm_ratio_layer`, `grad_norm_layer` across all recorded epochs)
returns **nothing** that passes all gates. `acc_layer` reaches spearman -0.95 but fails `4o_ok`,
consistent with the caveat already recorded in 3.10.9.9. Note `grad_norm_layer` is logged as the
sentinel -1 for every arm except `r`, so per-layer gradient norms during training do not exist as
data; the init-time backward pass above is the only gradient measurement available.

###### The three running arms are a pre-registered test

`ftbqm1dqk` and `ftbqm1dvo` were queued to separate the logit scale from the value write, and the
init measurements now show they separate `qk_over_v` too — in **opposite** directions from
`logit_scale`. Recorded before the results land:

| arm | `qk/v` | `logit_scale` | `value_write` |
|---|---|---|---|
| `ftbqm1dv` (both slices) | 2.177 | 0.00738 | 0.514 |
| `ftbqm1dvo` (v slice only) | **1.875** | 0.00552 | **0.514** |
| `ftbqm1dqk` (qk slice only) | **1.155** | **0.00738** | 0.869 |
| `ftbqm1d` (pooled) | 1.001 | 0.00552 | 0.869 |

* If **`qk_over_v`** is the mechanism: `ftbqm1dv` ~ +2, `ftbqm1dvo` clearly positive, `ftbqm1dqk`
  near `ftbqm1d`. The fitted line gives +1.6 and +0.5 respectively.
* If **`logit_scale`** is the mechanism: the ordering **reverses** — `ftbqm1dqk` ~ +2 and
  `ftbqm1dvo` ~ +0.4.
* If **`value_write`** is the mechanism: `ftbqm1dvo` ~ +2 and `ftbqm1dqk` ~ +0.4, i.e. the same
  ordering as `qk_over_v` but a larger gap.

`ftbqm1dv` is the consistency check: it matches `ftb4e3` to four decimals on all three. **Read
nothing before ~epoch 250** — `ftb4e3` did not overtake `ftbqmln` until epoch 214 (3.10.9.8).

> **Resolved in 3.10.9.12 (2026-08-29).** The observed order is `ftbqm1dvo` +1.37 > `ftbqm1dqk`
> +0.50, i.e. the reverse of what the logit-scale reading required: **the v slice carries the
> effect and the qk slice is inert.** `ftbqm1dv` landed at +1.40 rather than +2.08; the sentence
> originally here — that anything else "falsifies the whole family at once" — was too strong, and
> is withdrawn. It is not significantly below `ftb4e3` (p = 0.073) and the four top arms are
> indistinguishable from one another. See 3.10.9.12.

###### Caveats

* Ten arms, one of them (`ftb4o`) with a single seed. r = 0.966 on ten points with six of them
  tied at the same x is a much weaker claim than the number suggests; the two groups separate, the
  line through them is decoration.
* The screen is a filter over measurements, not a causal test. It says which quantities are worth
  an ablation and which are already dead. The ablations are the three running arms.
* Nothing here touches the LayerScale baseline, which remains the largest missing control.

##### 3.10.9.10 proc's Q and K are ~4x random while V is not — and the LayerNorm gain cancels it

> **[RETRACTED by §0]** The attention logit scale does not track accuracy once the write
> magnitude is held fixed: `ftbqm1dvo` (0.00552) and `ftbqm1dv` (0.00738) differ by 0.04. What
> is correct here is the *observation* that proc's gains cancel its large Q,K — but the reason
> that matters is that the same gain multiplies v, which is what sets the write.


Prompted by the observation that proc's Q and K matrices look unusually large. They are, and the
asymmetry against V is systematic across every block (std of each slice of `attn.qkv.weight`,
against timm's `trunc_normal_(std=0.02)` for random init):

| block | q / rand | k / rand | **v / rand** | LN gain |
|---|---|---|---|---|
| 0 | 3.86 | 3.81 | 3.42 | 0.310 |
| 1 | 3.09 | 3.89 | **1.22** | 0.444 |
| 4 | 3.49 | 4.06 | **1.70** | 0.389 |
| 8 | 3.97 | 4.28 | **2.03** | 0.384 |
| 11 | 4.86 | 4.84 | 4.63 | 0.353 |

**Q and K sit at 3.1-4.9x random in every block. V is only 1.2-2.0x through blocks 1-8**, and
only catches up at the two ends. So proc does not simply have "larger weights" — it has a
specific q,k >> v structure that random init does not.

**The LayerNorm gain cancels most of it.** Pre-norm means the branch sees `gain * norm(x)`, and
proc's gains average 0.31-0.44 rather than 1.0, so `q * gain` lands at only **1.2-1.7x** random.
**proc's large Q,K and small gains are a matched pair**, which is why 3.10.9.5's whole-tensor
statistics missed the structure: they aggregate q, k and v together and never combine the weight
scale with the gain.

###### The derived quantity: attention logit scale

Attention logits scale as `gain^2 * ||W_q|| ||W_k||` and the value write as `gain * ||W_v||`.
Neither had been computed for any arm. Relative to random init:

| arm | logit scale | v write | top-1 vs r |
|---|---|---|---|
| `r` | 1.00 | 1.00 | 0.00 |
| `ftbqm` | **10.98** | 3.31 | +0.08 |
| `ftbqmbias` | **10.98** | 3.31 | +0.05 |
| `ftbqmln` | 1.61 | 1.27 | +0.78 |
| `ftbqm1d` | 1.61 | 1.27 | +0.42 |
| **`ftbqm1dv`** | **2.14** | **0.72** | **+1.40** |
| `ftbqm1dvo` (v slice only) | 1.61 | **0.72** | **+1.36** |
| `ftbqm1dqk` (qk slice only) | **2.14** | 1.27 | **+0.58** |
| `ftb4e3` | 2.14 | 0.72 | **+2.08** |
| `ftb3i` | 2.14 | 0.72 | **+1.91** |

**This is the first quantity that explains a null rather than just recording it.** `ftbqm` and
`ftbqmbias` take proc's large Q,K but leave the LayerNorm gains at 1.0, so their attention runs at
**11x** the logit scale of random init — wildly oversharp. That is a concrete reason why arms
carrying proc's exact value multisets are worth nothing, which the elimination in 3.10.5 could
only state as a fact.

It also separates the two arms that win from the one that half-wins. `ftbqmln` recovers the gains
and drops to 1.61, worth +0.78. `ftb4e3` and `ftb3i` reach **2.14 with a v write of 0.72** —
*sharper* attention combined with *smaller* value writes — and are the only arms above +1.9.

**Prediction, with its limit stated.** `ftbqm1dv` sits at exactly 2.14 / 0.72, identical to both
winners, because slicing qkv restores proc's q,k >> v asymmetry. It should therefore land near
+2.08. **But it cannot attribute a win**: slicing raises the logit scale *and* lowers the value
write simultaneously, so a good result would be consistent with either factor. That is what the
`qk_only` / `v_only` arms are for (below).

**Caveat, and it is a real one.** `ftb4o` has logit scale **1.00** — it scales only `v`, `proj`
and `fc2`, never q or k — yet scores **-0.80** against random's 0.00. So this quantity does not
order every arm either. What it does do, for the first time, is group the fig5 family correctly
and give a mechanism for the two largest nulls in the elimination.

###### It does NOT explain the learning curves, and it is not monotone in accuracy

The obvious follow-up — does the logit scale account for the slow starts of 3.10.9.8? — is **no**.
Against epoch-49 accuracy over the six arms with completed runs:

| quantity | vs epoch-49 acc | vs final acc |
|---|---|---|
| rho (blocks 1-8) | **r = +0.78** | not monotone (3.10.9.7) |
| v write, `gain * \|\|W_v\|\|` | r = +0.43 | r = -0.69 |
| **attention logit scale** | **r = +0.23** (log-scale +0.06) | not monotone |

This is coherent rather than disappointing. rho measures the whole residual write including the
MLP, so it is the better account of how much signal propagates and how fast features can form.
Attention logits set the *pattern* of attention, not its magnitude, so there is no reason for them
to govern early learning speed — and they do not.

**Against final accuracy the logit scale is an inverted U, not a correlation:**

| logit scale | arms | final vs random |
|---|---|---|
| 1.00 | `r`, `ftb4o` | 0.00, **-0.80** |
| 1.62 | `ftbqmln` | +0.78 |
| **2.14** | `ftb4e3`, `ftb3i` | **+2.08**, **+1.91** |
| 10.98 | `ftbqm`, `ftbqmbias` | +0.08, +0.05 |

An optimum near **2.14**, with both too-blurry (1.0) and wildly oversharp (11.0) failing. That
shape echoes the late-block rho optimum of 1.4-2.0 in 3.12.3 — a band rather than a direction —
which is at least suggestive that the same kind of story applies at both ends of the network,
about different quantities.

**Two cautions.** Six arms is thin for asserting a curve shape. And `r` and `ftb4o` sit at
*identical* logit scale 1.00 with **0.81 points** between them, so the quantity is plainly not
sufficient on its own.

**What the new arms test.** `ftbqm1dqk` (`qk_only`, jobs 29511670/1/2) sits at logit **2.52** with
a v write of **1.37**; `ftbqm1dvo` (`v_only`, jobs 29511673/4/5) at logit **1.88** with write
**0.78**. With `ftbqm1d` (1.88 / 1.37) and `ftbqm1dv` (2.52 / 0.78) they complete a 2x2 in which
the two factors move independently — necessary because `--quantile_qkv_mode qk_v` moves **both**
at once (pooling averages q, k and v to a common width; slicing pushes q,k up *and* v down), so
`ftbqm1dv` on its own could not have attributed a win to either.

##### 3.10.9.9 Per-layer statistics DURING training — and what drives rho (2026-08-27)

Pulled from wandb (`Epoch-wise/*_layerN`), which records per-block statistics roughly every 10
epochs and which `log.txt` does not carry. Stitched across requeue fragments — `ftbqm1d` seed 0
alone spans **7 wandb runs** — by merging non-null values into the epoch marker
(`plots/cache/wandb_epochwise.json`, figure `plots/out/fig9_training_dynamics.png`). Seed 0 only.

**Two metric caveats, both of which bit an earlier draft of this analysis:**

* `grad_norm_layerN` is the **-1 sentinel** for every arm except `r` (0/384 sentinel for `r`,
  384/384 for the others). It is not that the metric is broken — it is simply not populated for
  any arm carrying procedural weights, so it cannot be used for these comparisons.
* `acc_layerN` is a **per-block read-out probe**. Early blocks score near zero by construction,
  so it must be read as a depth profile. Averaging it over blocks 0-8 — which the first pass did —
  produces a meaningless number and led to it being wrongly discarded as unusable.

###### The rho trace: what actually holds the early write down

The observation that prompted this: on the rho-over-training plot, `ftbqmln` leaves `ftb4e3`
after the first few epochs. Both start together (**epoch 4: 0.089 vs 0.084, ratio 1.07**), separate
progressively, and peak at ratio **1.67 by epoch 39**. So the initial rho is overwritten within
~4 epochs — both collapse to ~0.086 regardless of where they started — and the arms then *re-grow*
rho at different rates. **The persistent rho gap is a difference in where training drives rho, not
the init difference surviving.**

The natural hypothesis was that proc's full 1-D parameter set holds the early stack quiet while
LayerNorm tensors alone let it drift up. **`ftbqm1d` falsifies this.** It carries all eight 1-D
tensors — everything `ftb4e3` has — and tracks `ftbqmln` at every single epoch:

| epoch | `ftbqmln` (LN only, pooled) | `ftbqm1d` (all 8, **pooled**) | `ftb4e3` (all 8, **sliced**) |
|---|---|---|---|
| 4 | 0.089 | 0.090 | 0.084 |
| 19 | 0.221 | 0.229 | 0.161 |
| 49 | 0.216 | 0.224 | 0.130 |
| 99 | 0.212 | 0.234 | 0.140 |
| 199 | 0.248 | 0.329 | 0.209 |
| **plateau ep40-199** | **0.222** | **0.263** | **0.160** |

**What separates the low-rho arms is the qkv slicing, not the 1-D parameters.** `ftb4e3` slices
qk from v; every arm that pools sits high for the whole run. That is mechanically consistent:
pooling widens the v slice ~2x (3.10.9.5), v scales the attention output directly, so the write
stays large.

**The full trace set, including `ftb4o` (2026-08-27).** rho_attn, mean over blocks 1-8, seed 0:

| epoch | `r` | `ftb4o` | `ftbqm` | `ftbqmln` | `ftbqm1d` | `ftb4e3` | `ftb3i` |
|---|---|---|---|---|---|---|---|
| 4 | 0.382 | **0.047** | — | 0.089 | 0.090 | 0.084 | **0.021** |
| 19 | 0.272 | 0.119 | — | 0.221 | 0.229 | 0.161 | 0.059 |
| 49 | 0.192 | 0.121 | — | 0.216 | 0.224 | 0.130 | 0.104 |
| 99 | 0.262 | 0.153 | 0.203 | 0.212 | 0.234 | 0.140 | 0.164 |
| 149 | 0.320 | 0.207 | 0.272 | 0.226 | 0.283 | 0.170 | 0.205 |
| 199 | 0.365 | 0.260 | 0.324 | 0.248 | 0.329 | 0.209 | 0.232 |
| **plateau 40-199** | 0.288 | **0.183** | 0.239 | 0.222 | **0.263** | **0.160** | 0.211 |
| **final top-1 vs r** | 0.00 | **-0.80** | +0.08 | +0.78 | **+0.42** | **+2.08** | **+1.91** |

**This kills the rho story a second time, and more cleanly than 3.10.9.7 did.** Two pairs settle it:

* `ftb3i` (plateau **0.211**, **+1.91**) against `ftbqmln` (plateau **0.222**, **+0.78**) —
  indistinguishable write magnitude across the whole run, **1.13 points apart** in accuracy.
* `ftb4o` (plateau **0.183**, **-0.80**) against `ftb4e3` (plateau **0.160**, **+2.08**) —
  `ftb4o` runs *lower* than every arm except `ftb4e3` and finishes **worst of all nine**.

So the rho trajectory does not order the arms by outcome. An earlier draft of this section said the
qkv slicing "is what separates the low-rho arms"; that is true as a statement about rho — the four
pooled arms sit at 0.22-0.33 and the sliced `ftb4e3` at 0.160 — but it does **not** carry over into
accuracy, because `ftb3i` also slices and still plateaus at 0.211. **rho is a consequence of the
init, not a route to the mechanism, at init and throughout training alike.**

**What the `ftbqm1d` trace does establish.** The 1-D parameters are not what holds the early write
down: `ftbqm1d` carries all eight and plateaus *above* `ftbqmln`, which carries four. Whatever the
1-D params contribute to the +0.78, it is not via write magnitude.

###### The depth profile: the first quantity that orders the arms

`acc_layerN` is a read-out probe at each block. Summed over the 12 blocks it gives a
"depth to solve" score — **low means the task is only solved late in the stack**. At epoch 149,
65 epochs before the accuracy curves cross:

| arm | sum over blocks | blk 9 | final vs random |
|---|---|---|---|
| `ftb4e3` | **127.1** | 6.5 | **+2.08** |
| `ftb3i` | **127.1** | 6.0 | **+1.91** |
| `ftbqmln` | 215.8 | 40.2 | +0.78 |
| `ftbqm1d` | 225.8 | 42.1 | +0.42 |
| `ftbqmbias` | 228.3 | 43.2 | +0.05 |
| `ftbqm` | 233.4 | 42.7 | +0.08 |
| `ftb4o` | 275.8 | 54.1 | **-0.80** |
| `r` | 330.1 | 66.4 | 0.00 |

**Pearson r = -0.878 (n = 8, p = 0.004)**, and it is stable across training: -0.87 at epoch 49,
-0.88 at 99 and 149, -0.87 at 199, -0.84 at 249. Not a single-timepoint artifact.

**It separates the pair that rho cannot.** `ftb3i` and `ftbqmln` have indistinguishable rho
(plateau 0.211 vs 0.222, tail 0.275 vs 0.277 — 3.10.9.9) and are **1.13 points apart** in
accuracy. Their depth profiles differ **6.7x** at block 9 (6.0 vs 40.2). So this is not a
restatement of rho; it resolves the case rho fails on. `ftb4e3` and `ftb3i` also land at
**127.1 identically**, matching their statistically indistinguishable accuracies (p = 0.50).

**Reading.** Arms whose early blocks do *not* prematurely solve the task generalise better. That
is consistent with the regulariser picture of 3.13.x and with the winners' slow starts (3.10.9.8):
random init has effectively solved ImageNet by block 8, while the two winning arms are still at
6% at block 9 and only commit in the last two blocks.

**Two caveats.** `ftb4o` breaks monotonicity at the bottom — it solves *later* than random
(275.8 vs 330.1) yet scores 0.81 worse. Dropping it barely moves the correlation (-0.888), so it
is neither driving nor rescuing the result, but the relation is not clean across all eight. And
the epoch-299 measurement collapses to ~0.1 for every arm at every block, so the final logged
point measures something different and is excluded throughout.

**This is a correlation over 8 arms, not a mechanism.** It does not say why the early blocks stay
uninformative, only that they do in the arms that win. `ftbqm1dv` is a live test: its construction
(sliced qkv, restoring proc's q,k >> v asymmetry) predicts a low profile near 127.

###### Two other statistics

**Residual stream scale** (`blk_act_rms`, blocks 1-8) starts equal (ep49: 14.0 vs 14.5) and
diverges from ~epoch 149, with `ftb4e3` at roughly half `ftbqmln`'s by epoch 199 (6.9 vs 12.2) —
65 epochs before the accuracy crossover.

**Attention entropy** does **not** separate the arms (ratio 1.01-1.10 throughout), so this is not
an attention-sharpness effect.

**Where accuracy lives, at epoch 149** (`acc_layerN`, read-out accuracy at each block), 65 epochs
ahead of the accuracy crossover:

| block | `r` | `ftb4o` | `ftbqm` | `ftbqmln` | `ftbqm1d` | `ftb4e3` | `ftb3i` |
|---|---|---|---|---|---|---|---|
| 9 | 66.4 | 54.1 | 42.7 | 40.2 | 42.1 | **6.5** | **6.0** |
| 10 | 74.4 | 68.3 | 68.4 | 66.6 | 68.6 | **37.0** | **40.6** |
| 11 | 76.8 | 76.2 | 77.3 | 77.0 | 77.6 | 75.9 | 75.0 |

Every arm reaches ~76% at the final block by a completely different route. Random init has
effectively solved the task by **block 9** (66.4%); the two arms that win are at **6%** there and
only converge at block 11. **They defer the classification decision to the last two blocks and use
the early stack for something else.**

**How to read this, updated once `ftbqm1d` finished.** An intermediate draft said the profile
"groups the arms by final outcome rather than by construction"; a later one said the opposite,
that it "splits the arms the same way rho does". **Both were wrong**, and the resolution is below
(see *The depth profile: the first quantity that orders the arms*): summed over blocks it
correlates with final accuracy at **r = -0.88**, and it separates `ftb3i` from `ftbqmln` — the
pair rho cannot tell apart. `ftbqm1d`'s completion is what settled this: at 42.1 it profiles with
the pooled arms and duly scored +0.42, near them.

**Caveat.** All of these are seed 0 only, and all are *consequences* of the init difference rather
than independent causes. The rho gaps (0.16 vs 0.26) are far larger than the accuracy seed spread,
but that has not been verified directly.

##### 3.10.9.8 The gap is latent for 200 epochs — training curves (2026-08-27)

Every arm in 3.10.9.5-7 differs **only in its initial weights**. Verified by diffing the parsed
`Namespace` from the run logs of `ftb4e3` s0 and `ftbqmln` s0: lr, total batch, epochs, warmup,
weight decay, mixup, cutmix, smoothing, RandAugment, random-erasing, drop-path, optimizer, layer
decay and AMP are identical; the only differing keys are inert init-method fields. **So the 1.25
unexplained points are a deterministic consequence of the init.** They are not caused by anything
that happens during training, and the property we are missing must be a property of the initial
weights that our per-tensor summaries discard.

**But the difference does not appear until late.** Test accuracy, mean over 3 seeds:

| epoch | `r` | `ftbqm` | `ftbqmln` | `ftb4e3` | `ftb4e3` - `ftbqmln` |
|---|---|---|---|---|---|
| 49 | 70.18 | 66.89 | 66.65 | **60.75** | **-5.89** |
| 99 | 75.64 | 75.17 | 74.71 | 71.39 | -3.32 |
| 149 | 77.08 | 77.14 | 77.06 | 75.39 | -1.67 |
| 179 | 77.32 | 77.78 | 77.89 | 76.91 | -0.98 |
| 199 | 77.43 | 77.85 | 78.12 | 77.94 | -0.18 |
| 229 | 77.64 | 77.97 | 78.50 | 79.06 | +0.55 |
| 299 | 78.08 | 78.16 | 78.86 | **80.16** | **+1.30** |

`ftb4e3` is **5.89 points behind at epoch 49**, crosses `ftbqmln` at **epoch 214**, and then gains
**+3.25 over the final 120 epochs** against `ftbqmln`'s +0.97 and `ftbqm`'s +0.38. The winning
init is the slowest starter. Plotted as panel (b) of `plots/out/fig5_mechanism.png`.

###### Why the good inits start slow

Two properties, both measured, both pointing the same way.

**1. Low rho means weak residual writes.** Across the seven arms with a completed run,
rho(blocks 1-8) at init vs epoch-49 accuracy gives **Pearson r = +0.78**:

| arm | rho 1-8 | ep 49 | ep 299 |
|---|---|---|---|
| `ftb3i` | 0.099 | **60.28** | 79.99 |
| `ftb4e3` | 0.156 | **60.75** | 80.16 |
| `ftb4o` | 0.086 | **62.60** | 77.27 |
| `ftbqmln` | 0.259 | 66.65 | 78.86 |
| `ftbnorm` | 0.257 | 67.14 | 78.28 |
| `ftbqm` | 0.274 | 66.89 | 78.16 |
| `r` | 0.214 | **70.18** | 78.08 |

Blocks that write less are closer to identity, so less signal propagates and features form more
slowly. This is the Fixup/ReZero regime — except here it is not a design choice, it falls out of
the checkpoint's own LayerNorm gains (mean 0.31 rather than 1.0). It also reframes 3.14: the
near-identity literature predicted this *dynamics*, which we do reproduce, without predicting the
final ordering, which we do not.

**2. Larger weights take smaller relative steps.** proc's matrices are **3.3-3.8x wider** than
random init (`qkv[qk]` std: `r` 0.0200, `ftbqm`/`ftbqmln` 0.0663, `ftb4e3`/`ftb3i` 0.0763). AdamW's
update is roughly `lr` per parameter regardless of weight scale, so the same step is a ~3.7x
smaller *relative* change and reconfiguration takes proportionally longer. This is why `r` is the
fastest starter despite only mid-range rho — it is the one arm with small weights.

> **RETRACTED 2026-09-02 (rev 2) — see §0c.2.** An earlier banner here got this backwards. The
> paragraph's premise is false: it reads *"both start at 60.28 and 60.75"* off `ftb4e3`, the SEVERE
> arm of §0a. Fixed, `ftb4e3fix` starts at **68.51** at ep49 and **75.40** at ep99, against
> `ftb3i`'s 70.50 — so permuting the weights moves early speed by **+4.0** and **rank is exactly
> what governs the slow start**. The paragraph's other claim, that rank is irrelevant to the FINAL
> score, is correct and now independently confirmed by `ftbqmlnvo` (§0c.1).
>
> The neighbouring *"larger weights take smaller relative steps"* account is also contradicted:
> `ftb4e3fix` carries the widest qk in the study (std 0.0763) and starts fastest of the
> checkpoint-derived arms.

**Ruled out: low-rank structure.** `ftb3i` sits at stable rank 5-22 and `ftb4e3` at ~340
(3.10.9.5), yet they start at 60.28 and 60.75. Permuting the weights destroys the rank and changes
early speed not at all, so a restricted-subspace account of the slow start is wrong.

**The honest caveat.** Epoch-49 accuracy vs final accuracy is **r = -0.58** — slow starters tend to
finish better, consistent with the regulariser picture of 3.13.x — but `ftb4o` breaks it: lowest
rho of any arm, a slow start, and the **worst** final score at 77.27. A slow start is therefore
*associated* with the arms that win and is not sufficient to produce a win. Same lesson as
3.10.9.7: rho is a correlate of the mechanism, not the mechanism.

##### 3.10.9.7 A measurement bug, and what the corrected rho shows (2026-08-27)

**The bug.** `plots/measure_init_rho_arms.py` reconstructed `ftb4e3` with a single flat
`randperm` over the whole `attn.qkv.weight`. The trained arm uses `utils.shuffle_weights`, which
shuffles `attn.qk.weight` (rows 0:2e) and `attn.v.weight` (rows 2e:3e) as **separate pools**. The
script was therefore pooling — the very confound 3.10.9.5 had just identified — so every rho
figure previously reported for `ftb4e3` described a model that was never trained. **Training is
unaffected**; only the measurement scripts were wrong. Found because `ftbqm1dv`, which is
supposed to be structurally identical to `ftb4e3`, came out 2.5x lower at block 1.

Corrected (`rho_attn`, mean over **blocks 1-8**, 256 real val images, no training):

| arm | rho_attn 1-8 | qkv handling | top-1 vs random |
|---|---|---|---|
| `ftb3i` proc intact | **0.099** | — | **+1.91** |
| `ftb4e3` | **0.156** | sliced | **+2.08** |
| `ftbqm1dv` | **0.150** | sliced | pending |
| `r` random | 0.214 | — | 0.00 |
| `ftbqmln` | 0.259 | pooled | +0.78 |
| `ftbqm1d` | 0.266 | pooled | pending |
| `ftbqm` | 0.274 | pooled | +0.08 |
| `ftbqmbias` | 0.277 | pooled | +0.05 |

**1. `ftbqm1dv` now matches `ftb4e3` to 3 decimals** (0.150 vs 0.156; block 0 1.733 vs 1.736;
0.095 vs 0.099 at block 1). The reconstruction is faithful, and `ftbqm1dv` is confirmed as the
arm with no measured difference from `ftb4e3`.

**2. The v-pooling inflates rho across the entire early stack**, not just block 0: every pooled
arm sits at 0.26-0.28 against 0.15 for the sliced ones, ~1.8x. This is a *systematic* offset
affecting every quantile-matched arm in the 3.10.5 elimination.

**3. rho still does NOT predict accuracy in the early blocks — an intermediate draft of this
section claimed it did, and that was wrong.** Over six selected arms the correlation looked
strong (r = -0.86). Adding the two arms that most directly test it destroys it:

| arm | rho_attn 1-8 | rho_mlp 1-8 | vs random |
|---|---|---|---|
| **`ftb4o`** rho matched to proc, random values | **0.086** | 0.015 | **-0.80** |
| `ftb3i` proc intact | 0.099 | 0.019 | **+1.91** |
| `ftbqm1dv` | 0.150 | 0.041 | pending |
| `ftb4e3` | 0.156 | 0.041 | +2.08 |
| `r` random | 0.214 | 0.337 | 0.00 |
| `ftbnorm` | 0.257 | 0.036 | +0.20 |
| `ftbqmln` | 0.259 | 0.035 | +0.78 |
| `ftbqm1d` | 0.266 | 0.036 | pending |
| `ftbqm` | 0.274 | 0.044 | +0.08 |
| `ftbqmbias` | 0.277 | 0.044 | +0.05 |

Pearson r = **-0.27 (n = 8, p = 0.52)**, i.e. nothing. The -0.86 was an artifact of leaving
`ftb4o` and `ftbnorm` out.

**The decisive pair is `ftb4o` vs `ftb3i`.** They sit at rho 0.086 and 0.099 — the two lowest
values measured, indistinguishable from each other — and score **-0.80 and +1.91**, a 2.7-point
spread. `ftb4o` is a random model whose blocks 0-7 were scaled to proc's own measured ratios and
nothing else; `ftb3i` is proc's actual blocks 0-8. Same early write magnitude, opposite outcomes.
**Matching proc's early rho with random weights is actively harmful.** 3.10.9.4's original
conclusion stands and this section's contrary claim is withdrawn.

**What survives from this section.** The measurement bug is real and the corrected numbers above
replace the earlier ones. `ftbqm1dv` matching `ftb4e3` to 3 decimals is real and important. The
v-pooling offset is real — every pooled arm sits ~1.8x above the sliced ones in blocks 1-8 — but
it can no longer be argued to act *through* rho, since rho does not track accuracy here. Whether
widening v costs anything is now an open question that `ftbqm1dv` answers directly.

**Prediction, recorded before the runs land.** `ftbqm1dv` is structurally indistinguishable from
`ftb4e3` on everything measured, so it should reach ~+2.08; if it does not, the elimination has a
hole that no measurement so far has found. `ftbqm1d` differs from it only in v's spread, so the
pair isolates what widening v costs. Neither prediction rests on rho.

##### 3.10.9.5 Init-time structure of every arm — measured, not inferred (2026-08-27, RERUN)

> **CONFIRMED 2026-09-02 (rev 2) — see §0c.1.** The stable-rank table below is correct, and so is
> its verdict for the FINAL score: *"the low-rank structure, the learned singular directions and
> the per-neuron scale profile are all unnecessary."* It was argued from `ftb4e3` = 80.16, the
> contaminated arm of §0a, so the reasoning was unsound — but `ftbqmlnvo` (full-rank, clean,
> 79.94) reaches the same conclusion on solid ground.
>
> The verdict does **not** extend to the learning trajectory: rank is what separates slow starters
> from fast ones (§0c.2), which the contaminated `ftb4e3` hid.

`plots/analyse_init_structure.py`, one L40S, no training. Stable rank = `||s||^2 / s_max^2`,
averaged over blocks 0-8; low means structured, 200-340 means unstructured. **The first version
of this table was generated with a buggy `ftb4e3` reconstruction that pooled the fused `qkv`
instead of slicing it (see 3.10.9.7); these are the corrected numbers, and the `qkv` slices are
now reported separately because aggregating them is exactly what hid the bug.**

| arm | qkv[qk] | qkv[v] | proj | fc1 | fc2 | top-1 vs r |
|---|---|---|---|---|---|---|
| `r` | 265.6 | 194.5 | 194.6 | 343.3 | 343.6 | 0.00 |
| `ftbqm` | 265.5 | 194.4 | 194.2 | 339.6 | 334.5 | +0.08 |
| `ftbqmbias` | 265.5 | 194.4 | 194.2 | 339.6 | 334.5 | +0.05 |
| `ftbqmln` | 265.5 | 194.4 | 194.2 | 339.6 | 334.5 | +0.78 |
| `ftbqm1d` | 265.5 | 194.4 | 194.2 | 339.6 | 334.5 | pending |
| `ftbqm1dv` | 265.6 | 194.5 | 194.2 | 339.6 | 334.5 | pending |
| `ftb4e3` | 266.0 | 193.0 | 194.5 | 337.9 | 335.0 | **+2.08** |
| `ftb3i` | **9.4** | **20.0** | **22.5** | **5.2** | **5.7** | **+1.91** |

**1. Proc's weight matrices are drastically low-rank, and the structure is entirely disposable.**
`fc1` and `fc2` come in at stable rank **5.2 and 5.7 out of 3072** — proc's MLP matrices are
close to rank-5 objects. `qkv[qk]` is 9.4 and `proj` 22.5. A uniform permutation takes all of
them to 194-343, i.e. indistinguishable from random init, and costs nothing: `ftb4e3` +2.08
against `ftb3i` +1.91, p = 0.50. **The low-rank structure, the learned singular directions and
the per-neuron scale profile are all unnecessary.**

**2. The v-slice measurement, which is the point of the rerun.** Proc keeps v much narrower than
q and k; the pooled quantile match destroys that asymmetry completely:

| arm | qk std | v std | v / qk |
|---|---|---|---|
| `r` random | 0.02000 | 0.01999 | 1.00 |
| `ftbqm`, `ftbqmbias`, `ftbqmln`, `ftbqm1d` (**pooled**) | 0.06628 | 0.06624 | **1.00** |
| `ftbqm1dv` (**sliced**) | 0.07634 | 0.03755 | **0.49** |
| `ftb4e3` (**sliced**) | 0.07634 | 0.03755 | **0.49** |
| `ftb3i` proc intact | 0.07634 | 0.03755 | **0.49** |

Earlier drafts described this as "pooling widens v by up to 2.3x". That understates it: pooling
**erases the qk/v asymmetry outright**. Proc holds v at 49% of qk's width and every pooled arm
sits at exactly 1.00. Four of the arms in the 3.10.5 elimination carry this.

**3. `ftbqm1dv` and `ftb4e3` are now identical on every structural measure** — v std to four
decimals, qk std to four decimals, stable rank across all five tensors. Together with rho
(0.150 vs 0.156, 3.10.9.7) that closes the audit: **no measurement in this document separates
them.** The consistency test is therefore valid, and its outcome is informative either way:

* `ftbqm1dv` reaches ~+2.08 -> the constructions are equivalent and the early-block benefit
  reduces to per-tensor value multisets plus the qk/v asymmetry.
* `ftbqm1dv` lands low -> the difference between +0.08 and +2.08 is invisible to stable rank,
  row-norm variance, slice std, rho and 1-D moments alike.

**Base rate is against the first outcome** and should be stated plainly: nine arms built as
"random model + proc statistics" span **-0.80 to +0.78**, while four arms that start from the
checkpoint span **+1.91 to +2.29**. Nothing has ever landed in between, and `ftbqm1dv` is in the
first family.

The 1-D params split the arms exactly as designed (`norm1.weight` over blocks 0-8):

| arm | mean | std | kurtosis |
|---|---|---|---|
| `r`, `ftbqm`, `ftbqmbias` | 1.000 | 0.000 | 0.00 |
| `ftbqm1d`, `ftbqm1dv`, `ftbqmln`, `ftb4e3`, `ftb3i` | 0.384 | 0.104 | 3.09 |

and `mlp.fc1.bias` mirrors it (`r`/`ftbqm`/`ftbqmln` at 0.0000/0.0000; the rest at
-0.0134 / 0.0156, kurtosis **15.84**).

###### A residual difference between `ftbqm1d` and `ftb4e3`, found by this measurement

The two arms were designed to be identical in distribution, and on every aggregate statistic above
they are. **They are not identical**, and the difference is in the fused `qkv` tensor:

* `ftb4e3` shuffles `attn.qk.weight` and `attn.v.weight` as **separate pools**
  (utils.shuffle_weights:1195), so the v slice keeps v's own value distribution.
* `ftbqm1d` quantile-matches `attn.qkv.weight` as **one tensor**, so the v slice is drawn from the
  pooled q+k+v distribution.

In proc, v is much narrower than q and k in every block past the first:

| block | proc v std | `ftb4e3` v | `ftbqm1d` v | ratio |
|---|---|---|---|---|
| 0 | 0.0684 | 0.0684 | 0.0740 | 1.08x |
| 2 | 0.0274 | 0.0274 | 0.0628 | **2.29x** |
| 4 | 0.0340 | 0.0340 | 0.0649 | **1.91x** |
| 8 | 0.0407 | 0.0407 | 0.0714 | **1.76x** |

**`ftbqm1d` systematically widens v by up to 2.3x.** This matters because v and `proj` are exactly
the tensors the late-block recipe scales to set the attention write — so the arms differ in the
one quantity the rest of this document is about. It also shows up in rho: `ftbqm1d` 1.856 vs
`ftb4e3` 1.768 at block 0 (3.10.9.4), a gap I had read as noise.

**Consequence for reading the result.** If `ftbqm1d` reproduces 80.16, the conclusion stands and
this difference was immaterial. **If it comes back null, it is a confound and not evidence** that
the 1-D params fail: the arm would differ from `ftb4e3` in both the 1-D params *and* v's spread.

**`ftbqm1dv` — the slice-matched variant, launched 2026-08-27** (jobs 29507368/9/70, 3 seeds).
Rather than wait to see which way `ftbqm1d` falls, the clean arm is now running alongside it.
New flag `--quantile_qkv_mode pooled|qk_v`; `pooled` is the default and reproduces the existing
behaviour exactly, so no running or completed arm changes. `qk_v` matches rows [0:2e] and
[2e:3e] as two independent pools, mirroring `ftb4e3`'s `attn.qk.weight` / `attn.v.weight` split.

Verified before launch on `blocks.8.attn.qkv.weight`:

| | v std | qk std | full \|W\| |
|---|---|---|---|
| proc | 0.0407 | 0.0825 | 94.907 |
| `pooled` (`ftbqm1d`) | **0.0714** | 0.0713 | 94.909 |
| `qk_v` (`ftbqm1dv`) | **0.0407** | 0.0825 | 94.907 |

`qk_v` restores v's spread exactly and preserves each slice's value multiset. With it,
`ftbqm1dv` differs from `ftb4e3` in **nothing** that has been measured — same 2-D multisets per
slice, same 1-D values, same arrangement statistics (3.10.9.5). It is the arm that actually tests
whether the 1-D parameters carry the +2.07, and `ftbqm1d` becomes a secondary datapoint on
whether widening v costs anything.

The second outcome is live and should be planned for. Since 3.10.9.1's "permutation type" confound
was withdrawn (the rank map onto an iid tensor and a uniform `randperm` are the same operation in
distribution), the 1-D params are the *only* remaining difference between the two arms. If they
are not it, then either the elimination has a hole not yet identified, or one of the two arms
differs in a way that has not been audited.

##### 3.10.9.4 rho measured at init for every arm — and rho does NOT predict accuracy

Measured 2026-08-26 on one L40S, 256 real ImageNet val images, no training
(`plots/measure_init_rho_arms.py`, job 29504037). rho matches
`engine.attention_residual_analysis` exactly, so these are comparable to every rho quoted above.

| arm | what blocks 0-8 get | rho blk 0 | blk 1-8 | blk 9-11 | top-1 |
|---|---|---|---|---|---|
| `r` | nothing | 0.37 | 0.214 | 0.160 | 78.08 |
| `ftbnorm` | norms rescaled | 1.67 | 0.257 | 0.085 | +0.20 |
| `ftbqm` | 2-D values only | **6.54** | 0.274 | 0.034 | +0.01 |
| `ftbqmbias` | + biases only | **6.57** | 0.277 | 0.034 | pending |
| `ftbqmln` | + LayerNorm only | 1.78 | 0.259 | 0.083 | pending |
| `ftbqm1d` | + all 1-D, proc values | 1.86 | 0.266 | 0.085 | pending |
| `ftbqm1dpar` | + all 1-D, moment-matched | 1.84 | 0.267 | 0.085 | pending |
| `ftb4e3` | proc values, all shuffled | 1.77 | 0.274 | 0.088 | **+2.08** |
| `ftb3i` | proc intact | 4.09 | 0.099 | 0.007 | +1.91 |
| `p` | proc, all 12 blocks | 4.09 | 0.099 | 0.513 | 80.09 |

**1. rho does not predict accuracy in the early blocks.** The cleanest pair is `ftbnorm`
(rho 1.67, **+0.20**) against `ftb4e3` (rho 1.77, **+2.08**) — near-identical write magnitude,
1.9 points apart. In the other direction `ftbqm` sits at rho 6.54 for +0.01 while `ftb3i` sits at
4.09 for +1.91. Both the working and the failing arms span the whole rho range. **The early-block
effect is therefore not the late-block recipe at a different depth; it is a different phenomenon,
and the document should stop implying one mechanism covers both.**

**2. The LayerNorm gains are exactly the rho knob, as pre-norm predicts.** `x = x + attn(norm1(x))`
normalises away the input scale, so the gain sets the branch input and hence the write, while the
residual stream it is added to is untouched. Arms without proc's gains sit at rho ~6.5; adding them
drops block 0 to ~1.8, a 3.5x fall matching the 0.31 mean gain. Biases do not move rho at all
(`ftbqmbias` 6.57 vs `ftbqm` 6.54). The knob works — it just is not the knob that buys accuracy.

**3. This is what makes the four-arm decomposition readable**, and it is why `ftbqmln` was added
after the fact:

| arm | rho blk 0 | rho-matched to |
|---|---|---|
| `ftbqmln` | 1.78 | `ftb4e3` (1.77) — the arm that **works** |
| `ftbqmbias` | 6.57 | `ftbqm` (6.54) — the arm that **fails** |
| `ftbqm1dpar` | 1.84 | `ftbqm1d` (1.86) — differs only in *values* |

So `ftbqmln` is a like-for-like test against `ftb4e3`: same write magnitude, and the only
remaining difference is whether proc's LayerNorm gains are enough without its biases. A null from
`ftbqmbias` alone would have been ambiguous — it could have meant "biases do not matter" or
"biases cannot help while rho is 6.5" — which is precisely the gap `ftbqmln` closes.

**A caution recorded for the next person.** The first version of the measurement script reported
`ftbqm1d` with rho identical to `r` in all twelve blocks. That was a bug in the *script*, not in
any training run: the `r` branch had no early return, so the baseline fell through into the
quantile-matching loop and was silently measured as `ftbqm1d`. Training is unaffected — `r` trains
under `init_method="default"`, a different code path in main.py — but it is a reminder that the
script re-implements the arms rather than reusing main.py's code, which it must do because
`--attention_residual_analysis` returns at main.py:1086, before `shuffle_weights` at 1225. Two
implementations mean they have to be checked against each other; identical rows are the signal.

**`ftbcomp1`: 79.98 +/- 0.23 — statistically indistinguishable from proc init (80.09 +/- 0.12),
and +1.90 over random.** The additive prediction fails, but the result is arguably the stronger
one: **one proc block plus a calibrated last block matches full proc pretraining.** It is also
within noise of `ftbrho`'s recipe-alone 79.69 +/- 0.30 (max-based, not yet recomputed), so on this
evidence block 0 may be contributing little or nothing on top of the recipe.

Two caveats that keep this from being a headline. `ftbcomp1` runs the recipe at **rho 0.25**, not
the 1.4 used by `ftbcomp11` (80.63) — rho 1.4 is unreachable here (factor 2884, §3.10.9 above), so
this is *not* a like-for-like weakening of ftbcomp11 and the two cannot be differenced. And the
comparison that would separate "block 0 helps" from "the recipe is doing all of it" is
`ftbcomp1` vs a rho-0.25-only arm, which does not exist — `ftbrho` is at a different target.

### Splitting proc across depth (IN-1k)

`ftb4h` and `ftb4i` give proc weights to only half the network, seed 0:

| run | blocks 0-7 | blocks 8-11 | top-1 |
|---|---|---|---|
| r | random | random | 78.08 |
| **ftb4h** | random | **proc** | **79.69** |
| **ftb4i** | **proc** | random | **79.91** |
| p | proc | proc | 80.09 |

Either half alone recovers most of proc's +1.94, and neither reaches it: late-only +1.53,
early-only +1.77.

**This contradicts IN-100.** There, e3 (proc early blocks, random uncalibrated late blocks)
scored **83.22 — below random** (§3.8.3), whereas the IN-1k analogue ftb4i lands at 79.91,
essentially level with p. The two are not exact analogues (e3 also shuffles the proc blocks,
and the split is 0-8/9-11 vs 0-7/8-11), but the sign difference is large and unexplained. It
weakens the claim in §3.12 that proc early blocks are harmful without a matched late half —
that may be IN-100/ViT-S specific. Single seed, so worth confirming before leaning on it.

### Depth sweep in the RANDOM context (IN-1k)

The depth sweep below is in the *proc* context, where everything lands on p. The complementary
sweep in the **random** context — a1 repeated with different numbers of late blocks calibrated,
`--random_blocks` always the complement of `--init_method_scaled_blocks` — is the one that
shows structure. All are `upscale_random_*`, i.e. a fully random model with the listed blocks
scaled to the (random + proc-in-those-blocks) transplant's ratios. r = 78.08, p = 80.09:

| blocks calibrated | attn+mlp | attn only |
|---|---|---|
| 11 | 79.16 | 78.89 |
| 10,11 | **79.84** (n=3) | 78.90 |
| **9,10,11** | **80.06** (n=3) | 79.40 |
| **8-11** | **80.09** (n=3) | 79.38 |
| **7-11** | **80.12** | 79.19 |
| 6-11 | 79.04 | 79.28 |
| 5-11 | 79.20 (n=2) | — |
| 4-11 | 78.93 | — |

**1. There is an optimum at 3-5 blocks, and it is non-monotonic.** 9-11 through 7-11 recover
proc init almost exactly (80.06-80.12 vs 80.09). Going deeper *degrades*: 6-11 → 79.04,
4-11 → 78.93, back toward random. More calibration is not better; a specific late-block window
is.

**2. One block gets most of the way.** Block 11 alone gives 79.16 = +1.08 of proc's +2.01;
two blocks +1.68; three saturate.

**3. The MLP matters at every depth.** attn-only is 0.3-0.9 worse throughout and never reaches
p (best 79.40 vs 80.06). The effect is not purely an attention phenomenon, which is worth
noting since the review doc frames it around the attention write.

**Constraint on the calibration hypothesis (§3.12).** If *any* self-consistent calibration
sufficed, calibrating more blocks should be neutral or better. It is not. The early blocks
have to be left alone — consistent with c1's null (§3.6) and with e3 falling *below* random
(§3.8.3). The hypothesis must therefore be "the late blocks must be calibrated to the stream
the early blocks produce", not "calibration everywhere helps".

#### Why does the window close? Not stream inflation

The obvious explanation is compounding: the targets here are proc blocks measured in a *random*
stream, where rho is 1.0-1.5 rather than proc's native 0.02-0.03, so each calibrated block
roughly triples the stream (review doc: ftb6 runs 33 -> 407). Measured at **init**, with no
training, using `POST_HOC=true` on arms k1-k8 (k late blocks calibrated, `--random_blocks` the
complement):

| arm | blocks calibrated | final \|\|r_out\|\| | IN-1k top-1 |
|---|---|---|---|
| k1 | 1 | 72.4 | 79.16 |
| k2 | 2 | 131.9 | 79.84 |
| k3 | 3 | 284.6 | **80.06** |
| k4 | 4 | **442.6** | **80.09** |
| k6 | 6 | 462.3 | 79.04 |
| k8 | 8 | **238.8** | 78.93 |

**Refuted.** Accuracy and inflation are uncorrelated past k=3: k4 has the *most* inflation and
the best accuracy; k6 has the same inflation and is a point worse; k8 has *less* inflation than
k3 and is the worst arm. (k1-k3 alone looked like clean geometric growth — 72 -> 132 -> 285 —
and would have supported the story. k8 is what kills it.)

**What the data points to instead:** the degradation tracks *which* blocks are calibrated, not
how much the stream grows. Calibrating block 4 or 6 forces an **early** block to write ~1x the
stream it reads. That may suit late blocks, which aggregate, and damage early blocks, which
refine gradually — consistent with c1 and e3. The window would then reflect **depth-specific
roles**, not a scale budget.

**Direct test (running): the middle-band control.** Three arms calibrate exactly **3** blocks
at different depths, every other block left random and untouched, so block *count* is held
fixed and only *depth* varies:

| arm | band calibrated | status |
|---|---|---|
| m1 | 3,4,5 | **82.88 +/- 0.87 (-1.23 vs random)** |
| m2 | 6,7,8 | running |
| m3 | 9,10,11 | **= a1, already measured: 86.26 +/- 0.27** |

- **depth matters** => m1, m2 fall well below m3. Confirms the constraint-setter/satisfier
  picture (§3.12) and explains k6/k8 as "the sweep started including early blocks".
- **count matters** => m1 ~ m2 ~ m3. Any three calibrated blocks work, and k6/k8 fail simply
  from calibrating too many — reviving a budget-style account, though the inflation
  measurement above already rules out the obvious version.

m2 is the informative middle case: blocks 6-8 are exactly the ones pulled in when the sweep
goes from k3 (fine, 80.06) to k6 (bad, 79.04). If depth is what matters, m2 should sit clearly
below m3 and m1 below that.

m3 was not relaunched — it is bit-identical to a1 (same init method, same scaled and random
blocks). All three bands verified disjoint and covering blocks 0-11 exactly.

Note this also enabled `rin_norm_mean` / `rout_norm_mean` / `attn_delta_norm_mean` in
`attention_residual_analysis` — they were computed and discarded. The review doc's 4.3
activation-norm table needed them too.

### Replication across intervention depth (IN-1k, free)

Rather than add seeds to b1, the same claim is better tested by the four **sibling** all-proc
arms already on disk, each rescaling a different set of blocks. `rho_attn` block 11:

| scaled blocks | @0 | @70 | @299 | ratio to p @299 | top-1 |
|---|---|---|---|---|---|
| none (p) | 0.667 | 0.273 | 0.404 | — | 80.09 |
| 10,11 | 0.158 | 0.293 | 0.383 | 0.95 | 80.15 |
| 9-11 (b1) | 0.154 | 0.302 | 0.395 | 0.98 | 80.05 |
| 8-11 | 0.154 | 0.340 | 0.461 | 1.14 | 79.91 |
| 7-11 | 0.151 | 0.320 | 0.437 | 1.08 | 79.84 |
| **0-11** | 0.153 | 0.225 | 0.383 | **0.95** | **80.21** |

All five start at **0.23x of p** and end at **0.95-1.14x**, converged by epoch 70. Block 9
behaves identically (all end 0.403-0.498 vs p's 0.492). The `0-11` arm is decisive: rescaling
*every* block's write magnitude at init still lands on p and scores highest of the family, so
there is no intervention depth that escapes the attractor.

**Widening the intervention does not escape it either.** Two further runs
(`ftb0g` = 29435775, `ftb0h` = 29435780) scale **all 12 blocks** *and* extend the scaled
components from the default `v,proj,fc2` to **`v,proj,fc1,fc2`** via
`--init_method_scaled_attributes`:

| run | scaled blocks | attributes | top-1 |
|---|---|---|---|
| p | — | — | 80.09 +/- 0.12 |
| 29413395 | 0-11 | v,proj,fc2 | 80.21 |
| **ftb0g** | **0-11** | **v,proj,fc1,fc2** | **79.91** |
| **ftb0h** | **0-11** | **v,proj,fc1,fc2** | **80.09** |

Both land on p. So the erasure survives widening along *both* axes — every block, and more
matrices within each block — bringing the IN-1k proc-context replication to **seven runs**
spanning 79.84-80.21 against p's 80.09 +/- 0.12.

Note `fc1` sits upstream of the GELU, so scaling it is not linear in `Delta_mlp` (§3 of the
review doc): these arms perturb the forward pass *more* than the delta-norm matching intends,
and still converge to p. That strengthens rather than weakens the reading.

**a2 on IN-1k, now at 3 seeds.** The one anomalous number is confirmed but weaker than the
single seed implied: 76.90 / 78.20 / 77.02 = **77.37 +/- 0.72**, i.e. **-0.71** against the
random baseline (78.08 +/- 0.19), where the single seed suggested -1.18. On IN-100 the same
arm sits **+0.31 above** random. So the sign difference between datasets is real, but at ~1 sd
— and a2's spread is the largest of any arm — it is suggestive rather than solid.

**Caveats.** These are seven configs at one seed each (ftb0g/ftb0h being two of the same
config), not seven seeds of one config: they
corroborate the general claim (magnitude interventions do not stick in the proc context)
rather than pinning any individual arm's accuracy. r, p and a1 have 3 seeds each.
**a2 is the arm worth seeding** — at 76.90 it is 1.17 *below* the random baseline on IN-1k
while the same arm sat +0.21 *above* random on IN-100, and nothing else in the set is that far
out of line. That sign reversal is either real or a bad draw, and it is currently single-seed.

An older random baseline (29236813, 2026-07-03) reaches 78.48 — consistent with the August
3-seed mean — but showed an anomalous `rho` = 20.5 at block 11 decaying over ~70 epochs. That
was a stale-code artifact: the same-batch baseline (29384839) starts at **0.140**, in line with
IN-100's 0.115. All rho numbers above use the same-batch baseline. The lesson is to match
run *batches*, not just configs, when comparing against measurements taken from checkpoints.

### Does the recipe transfer to ViT-B? (ftbrho / ftbrho07)

Everything establishing the recipe is IN-100 / ViT-S. `ftbrho` and `ftbrho07` are the direct
port: random init, blocks 9-11 set to an **absolute** rho (1.4 and 0.7), no proc values used.
References on IN-1k: r = 78.08, p = 80.09, a1 = 80.06.

**The target rho is not scale-free — this is a caveat the IN-100 sections do not carry.**
ViT-B's random init writes far less relative to its stream than ViT-S's, so reaching the same
rho needs much larger surgery:

The scaled tensors default to `v, proj, fc2`, and the factor printed per block is the
**per-tensor** multiplier: `fc2` is multiplied by it directly, while `v` and `proj` each take it
(so the attention delta is amplified by its square). Full factors, both arms:

| block | ftbrho (rho 1.4) attn / **fc2** | ftbrho07 (rho 0.7) attn / **fc2** |
|---|---|---|
| 9 | 3.01 / **9.43** | 2.13 / **3.37** |
| 10 | 4.61 / **27.57** | 2.44 / **4.93** |
| 11 | 7.62 / **81.89** | 2.88 / **7.34** |

versus ~3-4 for the same blocks on IN-100 / ViT-S (n14).

Two things stand out. The factors **grow with depth within the arm** — scaling blocks 9 and 10
inflates the stream entering block 11, so block 11 must work that much harder to hit the same
ratio against a now-larger denominator. And halving the target rho cuts the worst factor by
**11x** (81.9 -> 7.34), not 2x, because that compounding unwinds. Cost is strongly superlinear
in the target, which makes rho 0.7 a far cheaper hedge than "half of 1.4" suggests.

**That prediction was wrong, and the recipe is working at scale.** Reasoning from §5.4 —
largest clean MLP factor 10.15 (h1), ftb4j diverged at 173 — ftbrho's worst block of **81.9**
was called as likely to collapse around epoch 50. It did not.

**Both arms finished 300 epochs:**

| arm | final | vs r (78.08) | vs p (80.09) |
|---|---|---|---|
| **ftbrho** (rho 1.4, no checkpoint) | **80.03** | **+1.95** | **-0.06** |
| ftbrho07 (rho 0.7) | 78.88 | +0.80 | -1.21 |

**Superseded by §3.10.3** — with all three seeds in, ftbrho is **79.69 +/- 0.30**, i.e. **-0.41
against proc init (2.2 sigma)** rather than level with it. The 80.03 below is seed 0, the best of
the three. The durable claim is +1.61 over random at 7.9 sigma, recovering ~80% of proc's gain
with no checkpoint.

rho 1.4 beats rho 0.7 by **+1.15**, so the IN-100 optimum transferred and the hedge was not
needed. Note ftbrho07 has the *lower* training loss (2.30 vs 2.35) but the lower accuracy — the
larger late-block writes generalise better rather than merely fitting harder.

So **raw factor magnitude does not predict divergence**. 81.9 trains cleanly at ViT-B scale
while 173 did not, leaving two possibilities the data does not separate: the threshold sits
between them, or ftb4j failed for a different reason — it calibrated random late blocks against
a *proc* stream (§5.4), a mismatch ftbrho does not have. The second is the better hypothesis,
since it also explains why ftb4j's factors compounded so hard across four blocks.

Practical consequence: the pre-launch rule in §5.4 ("anything above ~10 on the MLP is a
divergence risk") is too conservative by at least 8x and should be read as "inspect", not
"abort".

Outcomes:

- **both survive** => the recipe transfers, with two points of the dose-response at scale;
- **1.4 diverges, 0.7 survives** => the **safe band shrinks with model scale**, so the target
  rho must be chosen against the model's own init rather than copied from a smaller model —
  directly relevant to the nanochat port;
- **both diverge** => rho-targeting does not transfer to ViT-B as implemented, and the IN-100
  result is scale-limited.

Note both arms still load the proc checkpoint to build the *target model's structure*, even
though `--target_ratio_absolute` discards its values. That is cosmetic, not a dependency, but
it means the command line does not by itself demonstrate "no checkpoint required".

## 3.11 A second proc checkpoint (kdyck4)

A second IN-100 checkpoint, `results/pr_vits_kdyck4/pr_4159803_final.pth`, is being run
through the same arms. Both are 12-block, 15000-epoch, but their weight statistics differ
substantially:

| | \|\|W\|\| blk1 | \|\|W\|\| blk11 | mean kurtosis (0-8) | max kurtosis |
|---|---|---|---|---|
| `pr_vits/pr_27291166` (all results above) | 35.6 | 83.2 | **38.69** | **368.2** |
| `pr_vits_kdyck4/pr_4159803` | 54.0 | 85.0 | **9.66** | **54.9** |

kdyck4's early blocks are far less heavy-tailed (4x lower kurtosis, 7x smaller extreme) and
its norm profile is flatter. Runs use `PR_CKPT=`, `RUN_TAG=_k4` and wandb project
**`i100-playground-kdyck4`**.

### Results (3 seeds each)

| arm | old ckpt | kdyck4 | old: vs p | **kdyck4: vs p** |
|---|---|---|---|---|
| r random | 84.20 +/- 0.09 | 84.55 +/- 0.36 | -1.17 | -1.60 |
| **p proc** | 85.37 +/- 0.21 | **86.15 +/- 0.17** | — | — |
| a1 rand + up 9-11 | 86.26 +/- 0.27 | 86.59 +/- 0.57 | +0.89 | **+0.44** |
| a2 transplant + down | 84.51 +/- 0.39 | 84.43 +/- 0.40 | -0.86 | -1.72 |
| b1 proc + down | 84.87 +/- 0.24 | 85.39 +/- 0.49 | -0.51 | -0.76 |
| b2 proc + random up | 85.53 +/- 0.32 | 85.61 +/- 0.41 | +0.16 | -0.54 |
| c1 random, early rho | 84.42 +/- 0.19 | 84.39 +/- 0.30 | -0.95 | -1.76 |
| c2 proc, early rho | 85.40 +/- 0.21 | 85.67 +/- 0.15 | +0.03 | **-0.48** |
| **f1 clip 0.1%** | 85.78 +/- 0.26 | 85.84 +/- 0.24 | **+0.41** | **-0.31** |
| **f2 clip 1%** | 86.01 +/- 0.09 | 86.16 +/- 0.37 | **+0.63** | **+0.01** |
| **f3 clip 5%** | 85.76 +/- 0.71 | 85.89 +/- 0.13 | **+0.39** | **-0.26** |
| **proc advantage (p - r)** | **+1.17** | **+1.60** | | |

**1. kdyck4 is a materially better initialiser.** r moves only +0.35, inside the ~0.3
run-to-run variation measured for identical configs (§5.2), so random is unchanged as it must
be. The +0.78 on p is real and checkpoint-specific: proc init is worth **+1.60** here rather
than +1.17.

**2. `a1 > p` shrinks but survives**: +0.89 on the old checkpoint, **+0.44** on kdyck4. A
random model with rescaled blocks 9-11 still edges out proc init, but the gap halves as the
proc checkpoint improves. Whether it survives a third checkpoint is now the open question —
and §3.9's framing of proc's early blocks as a *ceiling* should be read as checkpoint-specific
until it does.

**3. The clipping result does NOT transfer — §3.8.3's headline was checkpoint-specific.**
All three clip fractions collapse from clearly beating proc to level with or below it:

| arm | old ckpt vs p | kdyck4 vs p |
|---|---|---|
| f1 (0.1%) | **+0.41** | **-0.31** |
| f2 (1%) | **+0.63** | **+0.01** |
| f3 (5%) | **+0.39** | **-0.26** |

This is exactly what §3.8.2 predicts if the benefit comes from removing extreme weights:
kdyck4 has 4x lower kurtosis and a 7x smaller extreme, so there is little tail to remove.
**"Winsorising the top ~1% improves proc init by ~0.63" is a property of
`pr_27291166`'s outliers, not of procedural pretraining.** The mechanism reading in §3.8.3
survives (proc's extreme weights are mildly harmful *when it has them*); the practical recipe
does not generalise.

**4. The context asymmetry — the core result — replicates and strengthens.**

| | old ckpt | kdyck4 |
|---|---|---|
| swing, random context (a1 - a2) | +1.75 | **+2.16** |
| swing, proc context (b2 - b1) | +0.66 | **+0.22** |
| ratio | 2.7x | **9.8x** |

Late-block magnitude is decisive when blocks 0-8 are random and nearly inert when they are
proc, on a second independently trained checkpoint. c1 also reproduces its null
(-1.65 vs p, tracking r at -1.51). `a1 > p` survives at +0.50.

**5. c2 moved from +0.08 to -0.47.** Matching proc's early write ratios to random's was
neutral before and is mildly harmful here — consistent with kdyck4's proc init simply being
better, so perturbing it costs more.

**6. The cause-side arms replicate too.** e1/e2/e3 on kdyck4, completing the sweep:

| arm | old ckpt | kdyck4 | reads as |
|---|---|---|---|
| e1 proc shuffled 0-8 + proc 9-11 | 85.15 (-0.23 vs p) | 85.15 (**-1.00** vs p) | shuffling costs *more* on the better checkpoint |
| e2 random \|\|W\|\|->proc | 83.10 (-1.10 vs r) | 83.57 (**-0.98** vs r) | replicates |
| e3 proc shuffled + random 9-11 | 83.22 (-0.98 vs r) | 83.43 (**-1.12** vs r) | replicates |

**e2 ~ e3 again** (83.57 vs 83.43, versus 83.10 vs 83.22 before) — the value-distribution null
of §3.8.3 holds on a second checkpoint. And **e3 is below random again**, which sharpens rather
than resolves the contradiction with IN-1k's ftb4i (79.91, level with p): the IN-100 result is
now 2-checkpoint replicated while the IN-1k counter-example is single-seed.

**Status.** All IN-100 arms complete on both checkpoints. The general lesson: **every "vs p"
number in this document is checkpoint-specific**, and comparisons must be made within a
checkpoint, never across. The clip sweep is the concrete demonstration — a +0.65 result on one
checkpoint and -0.08 on another. Note p itself moved +0.81, so "vs random" is the more stable
frame for anything meant to generalise.

### 3.11.1 Full kdyck4 results and what replicates

**Full kdyck4 results** (3 seeds each unless noted; `r` = 84.55, `p` = 86.15):

| arm | top-1 | | arm | top-1 |
|---|---|---|---|---|
| a1 calibrate 9-11 | **86.59 +/- 0.57** | | e1 pr shuffled 0-8 | 85.15 +/- 0.60 |
| h3 flat rho | 86.64 +/- 0.35 | | e4 pr 0-8 + rand 9-11 | 84.71 +/- 0.49 |
| g2 clip + upscale | 86.41 +/- 0.18 | | m2 band 6-8 | 84.66 +/- 0.65 |
| **p proc init** | 86.15 +/- 0.17 | | **r random** | **84.55 +/- 0.36** |
| h1 rho x0.5 | 86.05 +/- 0.39 | | c1 rand early matched | 84.39 +/- 0.30 |
| f2 clip 1% | 86.16 +/- 0.37 | | a2 hybrid down 9-11 | 84.43 +/- 0.40 |
| f3 clip 5% | 85.89 +/- 0.13 | | m1 band 3-5 | 84.01 +/- 0.27 (n=3) |
| f1 clip 0.8 | 85.84 +/- 0.24 | | e3 shuffled + rand 9-11 | 83.43 +/- 0.21 |
| b2 pr 0-8 + up 9-11 | 85.61 +/- 0.41 | | e2 rand norm-matched | 83.57 +/- 0.18 |
| c2 pr early matched | 85.67 +/- 0.15 | | d2 frozen 0-8 | 76.16 +/- 0.17 |
| b1 pr down 9-11 | 85.39 +/- 0.49 | | d0 frozen 0-8 | 75.79 +/- 0.20 |
| h2 rho x2 | 85.29 +/- 0.61 | | d1 frozen 0-8 | 75.51 +/- 0.40 |
| g1 clip + downscale | 85.25 +/- 0.98 | | | |

Cross-checks against the original checkpoint:

- **`a1 - r` = +2.01** (old +1.93) — the core effect, solidly replicated.
- **`g2 - g1` = +1.15** (old +0.87) — clipping direction replicates.
- **m1 -0.53, m2 -0.03 against `r`** (old -1.23, -0.45) — early-band calibration is still at or
  below random, so the **depth asymmetry replicates**, with smaller magnitude.
- **d0/d1/d2 at 75.6-76.3**, ~9 points below random — freezing blocks 0-8 is catastrophic on
  both checkpoints.
- **f1/f2/f3 at 85.84-86.16 vs `p` 86.15** — the clipping *gain* seen on the old checkpoint
  (+0.65) is **absent** here, as §3.11 first reported.


First complete kdyck4 arms (3 seeds unless noted, best-epoch top-1):

| arm | old ckpt | kdyck4 | change |
|---|---|---|---|
| r (random) | 84.20 +/- 0.09 | 84.55 +/- 0.36 | +0.26 |
| p (proc) | 85.37 +/- 0.21 | **86.15 +/- 0.17** | **+0.80** |
| a1 (calibrate 9-11) | 86.26 +/- 0.27 | 86.59 +/- 0.57 | +0.34 |
| h1 (rho x0.5) | 85.87 +/- 0.33 | 86.05 +/- 0.39 | +0.16 |
| h2 (rho x2) | 85.22 +/- 0.64 | 85.29 +/- 0.61 | +0.12 |
| **h3 (flat rho)** | **86.91 +/- 0.59** | **86.64 +/- 0.35 (n=3)** | **-0.65** |

**What replicates.** The core effect is solid: `a1 - r` is +1.93 on the old checkpoint and
**+2.01** on kdyck4. Calibrating the late blocks against a random baseline is a real,
checkpoint-independent gain. h1 and h2 also track their old values to within 0.16.

**The rho explanation is REFUTED. h3 does not replicate on kdyck4, and the reason is unknown.**

The hypothesis was that h3 inherits the checkpoint's own mean rho (it flattens to it rather than
targeting a constant), so kdyck4's lower mean should score lower. That predicted seed 0 — which
drew mlp rho **0.72**, the lowest of the three — to finish worst, around n07's 85.99. It
finished **best**:

| seed | attn rho | mlp rho | final |
|---|---|---|---|
| **s0** | 1.82 | **0.72** | **86.96** |
| s1 | 1.65 | 1.07 | 86.26 |
| s2 | 1.64 | 1.07 | 86.70 |

The seed with the *lowest* mlp rho scored *highest*, by +0.28 over the next best. And s1 and s2
have effectively identical rho (1.073 vs 1.065) yet differ by **0.42** — larger than the entire
effect the rho curve predicts across this range. Within this arm, rho explains nothing; seed
noise dominates completely.

So h3 at n=3 is **86.64 +/- 0.35**, against a1's **86.59 +/- 0.57** — a gap of 0.06, tied.

| | h3 | a1 | h3 - a1 |
|---|---|---|---|
| old ckpt | 86.91 +/- 0.59 | 86.26 +/- 0.27 | **+0.77** |
| kdyck4 | 86.64 +/- 0.35 | 86.59 +/- 0.57 | **-0.06** |

**§3.12.1's conclusion does not survive the second checkpoint.** "Proc's specific profile is
suboptimal, and flattening beats it" held by +0.77 on the original checkpoint and vanishes to
zero here. The rescue attempt above was wrong, and the honest position is that the +0.77 was
either checkpoint-specific or a favourable draw — not that flattening is better in general.

This matters for §3.12.3, which cited h3 as motivation for the checkpoint-free recipe. The
recipe itself is unaffected: it stands on the `n*` sweep, which is checkpoint-independent by
construction (n14 87.02, n20 86.99, 3 seeds each) and needs no argument about profiles at all. What
is gone is the claim that a *flat* profile is intrinsically better than proc's own.

**The test was underpowered, which should have been caught before it was stated as clean.** s1
and s2 have effectively identical mlp rho and finished 0.42 apart, so seed noise at fixed rho is
~0.4 — while the predicted effect (n07 85.99 vs n10 86.53, ~0.54) is barely larger. One seed
could never have separated those. A *confirming* result would also have been weak evidence.
Lesson for the next prediction: check the noise floor against the predicted effect before
calling something falsifiable.

**Why rho varies by seed at all:** the target model's blocks 0-8 are randomised
(`--random_blocks`), so the stream entering blocks 9-11 is seed-dependent and the measured
target rho moves with it. kdyck4's spread is much wider than the old checkpoint's (mlp 0.72 vs
1.07 across seeds, against 1.23-1.51). That spread is real but, per the table above, does not
translate into score — which is itself the evidence against the rho account.

**One margin genuinely does compress.** `p` improved by +0.80 while `r` barely moved, so every
"vs p" number shrinks: `a1 - p` falls from +0.89 to +0.44, and the recipe's headline goes from
**+1.65 over proc init to +0.87** (n14's 87.02 against p's 86.15). Against random it holds up:
+2.82 becomes **+2.47**. The recipe still wins on both checkpoints; the claim that needs
restating is how much it beats *proc init* by, which was always the checkpoint-specific half.

## 3.12 The early/late calibration hypothesis

Every property of proc's blocks 0-8 that could be isolated has been ruled out (§3.9). What
survives is an **interaction**:

| arm | blocks 0-8 | blocks 9-11 | result |
|---|---|---|---|
| e3 | proc | random, **uncalibrated** | 83.22 — **below random** |
| b2 | proc | random, **magnitude-matched** | 85.64 — above p |
| a1 | random | random, **magnitude-matched** | 86.26 — best arm |

Proc early blocks with *uncalibrated* late blocks are worse than an all-random network. The
same early blocks with *calibrated* late blocks beat proc init. Random early blocks with
calibrated late blocks are best of all.

**Hypothesis: what proc pretraining supplies is a self-consistent scale calibration between
early and late blocks, not the weights themselves.** The early blocks establish a
residual-stream scale; the late blocks are calibrated to write at the right relative magnitude
into it. Neither half carries value alone.

This accounts for the whole elimination table:

- every early-block ablation was null (rho, structure, norms, value distribution) because each
  preserved the stream scale the late blocks were calibrated to. e2 is the exception that
  fits — it *changed* the scale, and was the one harmful ablation;
- `a1 >= p` because the calibration can be manufactured directly, and manufacturing it against
  random early blocks works at least as well as proc's own;
- erasure, because in the proc context the calibration is already correct, so perturbing the
  late blocks is undone by training; in the random context the intervention *creates* a
  calibration, which training then keeps;
- the checkpoint-specificity of clipping, because outliers matter only insofar as they shift
  the stream scale, which varies by checkpoint.

**Constraints already on it.** The random-context depth sweep (§3.10) shows the effect peaks
at 3-5 late blocks and *degrades* when early blocks are included, so the claim is specifically
about late-block calibration to a fixed early stream — not calibration in general. The
inflation measurement in §3.10 rules out a scale-budget account of that window.

**Sharper statement: early blocks set a constraint, late blocks satisfy it.** Calibrating a
late block installs a valid solution to the constraint the early blocks impose. Calibrating an
early block *overwrites the constraint* that the downstream blocks were matched to. This covers
the window closing (§3.10) and c1's null (§3.6), and explains why random early blocks admit a
*better* solution than proc's own (a1 > p): they impose a weaker constraint. m1/m2 test the
"which blocks" half directly (§3.12.2).

> **Correction (e4, §3.12.6).** An earlier version of this section also claimed "proc early
> blocks are worse than random unless something downstream is matched to them", citing e3.
> That was wrong. e3 *shuffles* the early blocks; the unshuffled version (e4) scores +0.71 over
> random. Proc early blocks with random late blocks are fine. The real effect is a
> superadditive interaction between the two perturbations — see §3.12.6.

### 3.12.1 The profile control: proc's profile is fungible (the 'suboptimal' claim did not replicate)

> **Superseded in part by §3.11.1.** The +0.77 margin for h3 below is **original-checkpoint only**.
> On kdyck4 h3 and a1 tie (86.64 vs 86.59). Read the "profile is fungible" conclusion as
> supported on both checkpoints, and the stronger "flat is actively better" conclusion as **not
> replicated**.

h1/h2/h3 repeat a1 with the target profile rescaled or flattened. This is the control the
review doc listed as its highest-value missing experiment.

| arm | target rho, blocks 9/10/11 | top-1 | vs a1 | vs r |
|---|---|---|---|---|
| r random | 0.107 (untouched) | 84.20 +/- 0.09 | — | — |
| p proc init | — | 85.37 +/- 0.21 | — | +0.97 |
| a1 | 1.501 / 1.537 / 1.129 | 86.26 +/- 0.27 | — | +1.93 |
| h1 (x0.5) | 0.750 / 0.769 / 0.564 | 85.87 +/- 0.33 | -0.37 | +1.57 |
| h2 (x2.0) | 3.002 / 3.075 / 2.257 | 85.22 +/- 0.64 | -1.13 | +0.81 |
| **h3 (flat)** | **1.389 x3** | **86.91 +/- 0.59** | **+0.77** | **+2.70** |

**h3 is the best arm in the study**: +2.70 over random and **+1.73 over procedural init**,
using a *uniform* target. So the proc-specific magnitude profile does not transfer — matching
its shape is *worse* than ignoring it. The answer to the review doc's question is therefore
"attention write magnitude in a band, in the last few blocks", not "proc's particular profile".

Halving costs 0.37, doubling costs 1.13: there is an interior optimum near rho ~1.4 with slack
downward and less upward. h2 trained healthily (loss falling, 36.7% at epoch 18), so its
deficit is a real effect and not the AMP divergence flagged in §5.1.

### 3.12.2 Depth decides, not block count

m1/m2/m3 calibrate exactly **3** blocks at different depths, everything else untouched:

| arm | band | top-1 | vs random |
|---|---|---|---|
| **m1** | blocks 3,4,5 | **82.88 +/- 0.87** | **-1.23** |
| m2 | blocks 6,7,8 | *running* | |
| **m3** (= a1) | blocks 9,10,11 | **86.26 +/- 0.27** | **+1.93** |

Identical intervention, identical block count: **+1.93 at the end of the network, -1.23 in the
middle** — a 3.2-point swing from depth alone, and m1 lands *below* random. This settles the
§3.10 window: it closes because early blocks must be left alone, not because too many blocks
were touched. Same conclusion as c1 (§3.6) and e3 (§3.8.3), now isolated from every other
variable.

### 3.12.3 A checkpoint-free recipe

n07/n10/n14/n20 set an **absolute** rho (`--target_ratio_absolute`), using no proc values at
all. Random init sits at rho ~0.107 in these blocks.

| arm | target rho | top-1 | vs random | vs proc |
|---|---|---|---|---|
| r | 0.107 (untouched) | 84.20 +/- 0.09 | — | -0.97 |
| n07 | 0.70 | 85.99 +/- 0.19 | +1.69 | +0.73 |
| n10 | 1.00 | 86.53 +/- 0.52 | +2.18 | +1.21 |
| **n14** | **1.40** | **87.02 +/- 0.24** | **+2.68** | **+1.71** |
| **n20** | **2.00** | **86.79 +/- 0.18** | **+2.68** | **+1.71** |
| h3 *(control)* | 1.39, from proc's mean | 86.91 +/- 0.59 | +2.70 | +1.73 |

**n14 reproduces h3 to within 0.02** — the built-in control passes, so the absolute path is
equivalent to the flattened-target path and the result is not an artefact of the new code path.

**The recipe: set rho ~1.4-2.0 in the last ~3 blocks at init.** One number, one depth fraction,
no procedural checkpoint anywhere in the pipeline. Worth **+2.68 over random init and +1.71
over procedural pretraining** on IN-100/ViT-S. Safe up to rho ~2.6; diverges by 2.8-3.0
(§3.12.7); below 1.0 the benefit tapers. The gain scales with how far the base init sits below
the band (§3.12.8), which is the check to run before porting.

The response is monotone up to 1.4 and then **plateaus** — 1.4 and 2.0 are identical to 0.00.
Combined with h2 (a *profile* averaging 2.78, which scored 85.22), the usable band is roughly
`rho` in [1.4, 2.0] with degradation somewhere above it. Note h2 vs n20 is another case where
the *shape* matters and not just the level: uniform 2.0 is fine, the 3.00/3.07/2.26 profile is
not.

### 3.12.4 Depth response is monotone

With block count held at 3 (§3.12.2), all three bands now complete:

| band | top-1 | vs random |
|---|---|---|
| m1: blocks 3,4,5 | 82.88 +/- 0.87 | **-1.23** |
| m2: blocks 6,7,8 | 83.91 +/- 0.42 | **-0.45** |
| m3: blocks 9,10,11 (= a1) | 86.26 +/- 0.27 | **+1.93** |

Monotone in depth: harmful early, mildly harmful mid, strongly helpful late. The same
intervention spans a 3.2-point range purely by where it is applied. This is the cleanest
statement of the constraint-setter/satisfier picture (§3.12) and it fully explains the §3.10
window — the sweep degrades past 5 blocks because it starts including blocks whose calibration
is harmful.

Gains over random init, for reference: **+1.93** (IN-100 ViT-S, old ckpt), **+2.01** (IN-100,
kdyck4), **+1.90** (IN-1k ViT-B), rising to **+2.70** with the flat target. On IN-1k the method
*matches* proc init (-0.04) rather than beating it, so the defensible claim is "as good as
procedural pretraining, without pretraining".

**Before calling it a method:** (i) the constant works without the proc reference — done,
§3.12.3; (ii) it has been tuned only on ViT, 300-epoch, one LR schedule; (iii) it appears to run
*opposite* to Fixup / ReZero / SkipInit / T-Fixup, which scale residual branches **down** at
init — but that tension largely dissolves on inspection, since those methods target
trainability at extreme depth and apply one rule at all depths. See **§3.14**, and §3.12.5 for
the cross-init measurements.

### 3.12.5 Is rho ~0.1 a timm quirk? No — and GPT-style init is further out

`i100_playground/measure_init_rho.py` measures rho at init for different weight inits on the
same ViT-S architecture (seconds, no data, no training). Blocks 9-11, `rho_attn`:

| init | mean rho (blocks 9-11) | amplification to reach 1.4 |
|---|---|---|
| **nanoGPT-style** *(our reconstruction, NOT nanochat — see §3.14)* | **0.0053** | **x262** |
| timm (used throughout this study) | 0.0512 | x27 |
| small (normal 0.02, no out-proj scaling) | 0.0476 | x29 |
| Xavier uniform | 0.2953 | x4.7 |

Low late-block rho is **not** specific to timm — it is what most standard inits produce, and
**GPT-2 style init is 10x lower still**. The `1/sqrt(2L)` output-projection scaling is exactly
the "downscale residual branches" convention of the literature family, and it puts late-block
writes furthest below the useful band of anything tested.

Two consequences:

- **The recipe is not correcting a timm-specific defect**, which was the main deflationary
  explanation left after the literature check. Xavier is the exception at only 4.7x away, so
  the effect may be weaker under Xavier-initialised models — a cheap prediction to test.
- **A GPT-style port has more headroom than the ViT where the effect was found.** For the
  planned nanochat test this is the best possible pre-check outcome.

Caveat: measured with random Gaussian images and an untrained patch embed, so absolute values
differ from the training-time numbers (0.051 here vs 0.107 in the runs). Only the *cross-init*
comparison is apples-to-apples, since nothing but the init changes.

### 3.12.6 e4 explains e3's deficit — but does NOT close the IN-1k contradiction

> **Corrected by §3.13.3.** e4 resolved why *e3* scored below random (the shuffle, not the split
> boundary). It did not reconcile IN-100 with IN-1k: e4 lands on the random baseline while the
> IN-1k arms ftb4i/5i/6i sit ~1.6 above it. e4 confirmed the IN-100 side of the disagreement
> rather than removing the disagreement.

e3 (proc 0-8 **shuffled** + random 9-11) scored 83.22, *below* random, while IN-1k's ftb4i
(proc early, random late, **unshuffled**) scored 79.91, level with p. Those differ in two ways
— the shuffle and the split boundary — so ftb4e3 (running on IN-1k) cannot separate them. e4 is
ftb4i's design at e3's boundary: proc 0-8 **unshuffled**, random 9-11, no scaling.

| arm | blocks 0-8 | blocks 9-11 | top-1 | vs random |
|---|---|---|---|---|
| r | random | random | 84.20 +/- 0.09 | — |
| **e4** | **proc** | **random** | **85.05 +/- 0.19** | **+0.71** |
| e3 | proc **shuffled** | random | 83.22 +/- 0.45 | -1.10 |
| p | proc | proc | 85.37 +/- 0.21 | +0.97 |

**e4 lands near p, so the shuffle was responsible and the datasets agree.** ftb4i is unshuffled,
e4 is unshuffled, both sit at ~p. The contradiction is closed, and the §3.12 claim built on e3
is withdrawn.

**What is actually there is a superadditive interaction:**

| perturbation | arm | cost vs p |
|---|---|---|
| shuffle blocks 0-8 only | e1 | **-0.26** |
| randomise blocks 9-11 only | e4 | **-0.26** |
| **both** | **e3** | **-2.07** |

Either perturbation alone is nearly free; together they cost ~8x their sum. The network
tolerates degradation in one half and not in both — which is a statement about redundancy
between the halves, not about proc early blocks requiring a matched partner.

### 3.12.7 The rho band has a soft ceiling

> **Should we push rho above 1.4 on ImageNet-1k?** Expected answer: no gain, real risk.
> Three reasons. (a) The IN-100 sweep below is flat from 1.4 to 2.0 (87.02 / 86.99), so there is
> nothing to win. (b) Fitting ftbrho's two ViT-B points (rho 0.7 -> worst `fc2` 7.34;
> rho 1.4 -> 81.9) gives factor ∝ rho^3.5, so rho 2.0 would need a factor of **~283** against
> the only ViT-B divergence on record at 173 (ftb4j) — see §3.10.2. (c) The 0.36 shortfall
> against proc init (§3.10.3) has a structural explanation that rho cannot fix: the recipe
> rebuilds only the **late-block** half, and IN-1k additionally has an **early-block** half worth
> ~+1.56 (§3.13.4) that no rho setting touches. That also explains why the recipe *beats* proc
> on IN-100, where the early-block half is worth +0.07.
>
> If the question is to be settled empirically at ViT-B, **rho 1.7** is the informative point:
> predicted factor ~161, right at the known-diverging value, testing the plateau and the
> stability boundary in one run.



Extending §3.12.3 upward:

| rho | 1.0 | 1.4 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| top-1 | 86.53 | **87.02** | **86.99** | 86.79 | 85.90 *(n=1)* | *running* |

Plateau through 2.0, -0.20 at 2.5, then a real drop by 3.0. **Usable band ~[1.0, 2.5], optimum
1.4-2.0.** That is a wide tolerance — roughly a factor of 2.5 either side of the optimum still
beats proc init — which matters for porting the recipe to a setting where the right value
cannot be tuned.

**The upper limit is a stability boundary, not a gradual decline.** The high-rho arms did not
merely score worse, they *diverged*: `assert math.isfinite(loss_value)` after a healthy start
(n40 seed 0 was at loss 4.12 / acc 24.2 by epoch 7, then collapsed).

| rho | 1.0 | 1.4 | 2.0 | 2.5 | 2.6 | 2.8 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| seeds trained | 3/3 | 3/3 | 3/3 | 3/3 | **3/3** | **2/3** | **1/3** | **0/3** |
| top-1 | 86.53 | **87.02** | **86.99** | 86.79 | 86.72 | 86.08 | 85.90 | — |

So the threshold sits between **2.6 (3/3) and 2.8 (2/3)**. Accuracy also declines gently across
the stable range, so 1.4-2.0 is both the accuracy optimum and comfortably clear of instability. This also explains h2 (a profile averaging 2.78 that
survived but degraded to 85.22) and IN-1k's ftb4j, whose block-11 `fc2` multiplier reached
**173** and which collapsed at epoch 50 — §5.4.

### 3.12.8 The safe band is init-dependent (Xavier)

§3.12.5 predicted a *smaller gain* under Xavier init, which starts at rho ~0.295 rather than
timm's ~0.051 and so is already 6x closer to the useful band. **Confirmed, quantitatively:**

| init | baseline rho (blocks 9-11) | distance to 1.4 | baseline acc | + rho 1.4 | **gain** |
|---|---|---|---|---|---|
| timm | 0.051 | x27 | 84.20 +/- 0.09 | 87.02 +/- 0.24 | **+2.68** |
| **Xavier** | **0.295** | **x4.7** | 81.73 +/- 0.33 | 82.56 *(n=1)* | **+0.80** |

The gain shrinks from +2.68 to +0.80 as the baseline moves closer to the band. **Distance from
the target predicts the benefit**, which is direct evidence that the method *fixes
under-initialised late blocks* rather than adding a new capability. It also predicts where the
recipe will and will not pay: a lot for GPT-style init (rho ~0.005, §3.12.5), little for an init
that already writes hard in the late blocks.

**Unexpected second finding: the safe band is init-dependent.**

| init | rho 1.4 seeds trained |
|---|---|
| timm | **3/3** |
| **Xavier** | **1/3** (two diverged with `isfinite`, §5.4) |

The same target rho that is completely safe under timm diverges under Xavier. Xavier's larger
absolute weights make the resulting writes unsafe even though the *ratio* is identical, so
**rho alone does not determine stability** — the underlying weight scale matters too. For
porting: headroom and safety are separate questions, and a pre-check must verify both.

Caveats: the surviving x14 seed gives n=1, so +0.80 is soft. Xavier re-initialises all 52
weight tensors including patch-embed and head, and is simply a worse init here (81.73 vs
84.52) — so only the **within-family gains** are comparable, never x14 against the timm
baseline directly.

## 3.13 Synthesis: where the IN-100 results leave us

All numbers here are kdyck4 (§3.11.1), the current default checkpoint.

### 3.13.1 On IN-100, proc's benefit is late-block write magnitude and nothing else

Four arms, each attacking the question from a different side, agree:

| arm | what it does | result | reading |
|---|---|---|---|
| **e4** | proc 0-8 + random 9-11 | 84.71 vs r 84.55 | proc's early blocks alone are worth **zero** |
| **c1** | random model, early rho matched to proc | 84.39 ~ r | early rho is not the mechanism |
| **c2** | proc model, early rho matched to random | 85.67 ~ p | destroying it costs nothing |
| **a2** | proc late blocks downscaled to random rho | 84.62 ~ r | remove late rho and the entire proc gain goes |

e4 is the decisive one: transplant proc's first nine blocks wholesale, leave the last three
random, and you land on the random baseline. Whatever proc learned in its early blocks is not
what makes proc init better than random init.

### 3.13.2 The original question, answered: proc-early is *worse* than random-early

The study opened by asking why calibrating blocks 9-11 recovers the proc advantage when blocks
0-8 are random but "does not hold" when they are proc. With both arms at 3 seeds:

| | early blocks random | early blocks proc | difference |
|---|---|---|---|
| blocks 9-11 calibrated | **a1 86.59 +/- 0.57** | b2 85.61 +/- 0.41 | **-0.94** |

Same late-block treatment; the only difference is what sits in front of it. Proc costs **1.28**
(~3.9 sigma, and -0.96 on the original checkpoint, so it replicates). Supporting evidence points
the same way: m1 (84.01) and m2 (84.66) calibrate early bands and land at or below r = 84.55,
and freezing proc's blocks 0-8 (d0/d1/d2, 75.5-76.2) is catastrophic.

So the premise of the original question is inverted. Proc's early blocks do not fail to *behave
like* random ones — **they are actively a liability once the late blocks are calibrated.** There
is nothing to repair. The move is to stop using them: random init plus late-block calibration
(n14/n20, **87.02**) beats proc init (86.15) by **+0.87** and random by **+2.47**.

### 3.13.3 ImageNet-1k flatly contradicts §3.13.1, and this is the live problem

The identical experiment — proc early, random *uncalibrated* late — does not agree across scales:

| | proc blocks | score | vs random |
|---|---|---|---|
| IN-100 **e4** | 0-8 | 84.71 | **+0.16** |
| IN-1k **ftb11i** | 0 only | 78.78 +/- 0.18 (n=3) | +0.70 |
| IN-1k **ftb10i** | 0-1 | 79.11 | +1.03 |
| IN-1k **ftb9i** | 0-2 | 79.49 | +1.41 |
| IN-1k **ftb8i** | 0-3 | 79.58 +/- 0.25 (n=3) | +1.50 |
| IN-1k **ftb7i** | 0-4 | 79.89 +/- 0.35 (n=3) | +1.81 |
| IN-1k **ftb6i** | 0-5 | 79.66 | +1.58 |
| IN-1k **ftb5i** | 0-6 | 79.67 | +1.59 |
| IN-1k **ftb4i** | 0-7 | 79.91 | +1.83 |
| IN-1k **ftb3i** | 0-8 | **79.99 +/- 0.36** (n=3) | **+1.91** |
| IN-1k **ftb2i** | 0-9 | **80.24** | **+2.16** |
| IN-1k **ftb1i** | 0-10 | **80.37 +/- 0.12** (n=3) | **+2.29** |
| IN-1k **p** | all 12 | 80.09 +/- 0.12 (n=3) | +2.01 |

**The series is a monotone ramp across the whole depth range**, not a saturating curve. Earlier
drafts twice mis-read it — first as "saturates at ~4 blocks" from the endpoints, then as
"flattens above ~5" before 0-8/0-9/0-10 were run. It climbs continuously from 78.78 at one proc
block to **80.37 at eleven**.

**ftb1i (proc 0-10) at 80.50 is the highest IN-1k number recorded**, above full proc init
(80.09 +/- 0.12). Leaving the *last* block random appears to beat using proc there — which is
what §3.10.4 predicts, since the final block wants write magnitude rather than learned weights.
Single seed, and +0.40 is ~1.4 sigma, so this is suggestive rather than established.

**ftb3i settles the shuffle question.** It is proc 0-8 **unshuffled** — e4's exact composition at
IN-1k scale — and scores 79.97 against its shuffled twin `ftb4e3`'s 80.25. Statistically the
same, so the IN-1k advantage is **not** a shuffle artifact.

That makes the contradiction real and unexplained: **nine proc blocks are worth +0.07 on IN-100
and +1.88 on ImageNet-1k.** §3.12.6 treated e4 as closing this contradiction; it did not — e4 only confirmed the IN-100
side of it.

**`ftb3i` (launched 2026-08-18) is the missing arm.** It is proc 0-8 **unshuffled** with random
9-11 — e4's exact composition at ImageNet-1k scale, which had never been run. Its shuffled twin
`ftb4e3` already scored **80.25**, so the pair isolates the shuffle here the way e4 vs e3 did on
IN-100 (§3.12.6). If ftb3i lands near 80.3 like ftb4e3, proc's early blocks carry ~+2.1 at this
scale against +0.07 on IN-100 and the contradiction is real and unexplained; if it lands near
random, the IN-1k advantage was the shuffle all along and the two datasets agree.

**The decisive arm is missing because it diverged.** What separates "proc-early is neutral" from
"proc-early is a liability" is the a1-vs-b2 pair, and on IN-1k only the a1 side exists
(`ftbrho`, 79.98). The b2 analogue is **ftb4j**, which died at epoch 50 with a block-11 `fc2`
factor of 173 (§5.4). **Relaunched unchanged on 2026-08-17 and it diverged identically** — the
scale factors printed bit-identical (block-11 `fc2` x172.93 vs x173), it failed at the same
epoch 55 with the same 60.33 best accuracy, and the loss broke the same way
(4.35 -> 4.33 -> 5.32 -> 6.85). The init is fully deterministic, so this arm cannot be recovered
by rerunning.

**`--target_ratio_absolute 1.4` does NOT cap this arm — it makes it far worse.** That was the
obvious-looking fix and it was wrong. With proc 0-7 in front, proc's *own* rho in the blocks
being scaled is tiny, so 1.4 is an increase, not a ceiling:

| block | proc's own rho here (attn / mlp) | vs the absolute target 1.4 |
|---|---|---|
| 8 | 0.090 / **0.035** | **40x higher** |
| 9 | 0.264 / 0.161 | ~9x higher |
| 10 | 0.259 / 0.629 | ~2x higher |
| 11 | 0.879 / 0.766 | ~2x higher |

The launched arm (`ftb4jc`) printed a block-11 `fc2` factor of **8738** against the uncapped
run's 173, and was cancelled at init. Always read the `Absolute target` lines before assuming a
constant target is a reduction.

**This is itself a mechanism for §3.13.2.** The same proc late blocks measure rho 1.38 / 4.68 at
block 9 when *random* blocks precede them (from ftbrho's target) but 0.26 / 0.16 when *proc*
blocks do — 5-29x lower. Proc's early blocks inflate the residual stream so much that the late
blocks fall relatively silent, which is exactly the regime the recipe exists to escape; and the
surgery needed to escape it from there is large enough to destabilise training. That is a
concrete reason proc-early costs 0.94 against random-early on IN-100.

**What runs instead: `--target_ratio_scale 0.5`** (`ftb4jd`, launched 2026-08-17). Same design —
proc 0-7 copied, blocks 8-11 scaled, proc-matched profile — with the target profile halved. The
factors land in the demonstrated-safe band:

| block, mlp | ftb4j (diverged 3x) | **ftb4jd** |
|---|---|---|
| 8 | 4.9 | 2.4 |
| 9 | 23.4 | 11.4 |
| 10 | 94.9 | 44.6 |
| **11** | **173** | **59.9** |

59.9 sits below ftbrho's 81.9, which trained cleanly for 300 epochs. Caveat when reading the
result: the targets are *half* proc's ratios, so this is "proc early + late blocks calibrated to
half of proc's profile", not a strict b2 replica.

**Note also that ftb4j was never the b2 analogue.** IN-100's b2 copies proc 0-**8** and scales
9-11; ftb4j copies 0-7 and scales 8-**11**. The extra scaled block is exactly where the factor
explodes (block 8 needs `fc2` x335 under an absolute target), and four blocks compound harder
than three. No ViT-B arm has ever run the true b2 configuration — it remains available if the
0-7/8-11 boundary proves unusable.

Candidate explanations for the split, none tested: 1000 classes and 10x data give generic early
features more to do; ViT-B vs ViT-S; or e4's uncalibrated late blocks (rho ~0.107) throttle the
network so hard that no upstream quality can express itself, an effect that may be weaker at
ViT-B scale.

### 3.13.4 Position: proc weights are worth more early than late

The `h` series puts proc in the **late** blocks with random early — the exact inverse of the `i`
series. The full curve, now complete:

| proc blocks | arm | score |
|---|---|---|
| 3 (9-11) | ftb3h | 78.89 |
| 4 (8-11) | ftb4h | 79.69 |
| 5 (7-11) | ftb5h | 78.84 |
| 6 (6-11) | ftb6h | 78.72 |
| 7 (5-11) | ftb7h | 79.67 |

r = 78.08 +/- 0.19, p = 80.09 +/- 0.12. **The whole `h` series sits between 78.8 and 79.7** — it
never approaches proc init, and it is flat-to-noisy rather than a ramp. Contrast the `i` series
(§3.13.3), which climbs monotonically from 78.78 to 80.37. Proc weights placed late are worth
~+0.6 to +1.5; the same weights placed early are worth up to +2.29.

Matched-count pairs, early vs late:

| proc blocks | early (`i`) | late (`h`) | early advantage |
|---|---|---|---|
| 4 | ftb8i 79.58 +/- 0.25 | ftb4h 79.69 | -0.11 (tie) |
| 5 | ftb7i 79.89 +/- 0.35 | ftb5h 78.84 | **+1.05** |
| 6 | ftb6i 79.66 | ftb6h 78.72 | **+0.94** |
| 7 | ftb5i 79.67 | ftb7h 79.67 | 0.00 (tie) |

The matched pairs are **inconsistent** — two clear gaps and two ties — so "position beats count"
is not a law. All `h` arms are single seed, where the ViT-B seed std of 0.29 (§3.10.3) makes
differences under ~0.6 uninterpretable, which covers both ties.

**The robust version of the claim is §3.10.4's**, which does not depend on this table: proc's
weights are worth having in blocks 0-10 and are *worse than random* in block 11, and that
single-block effect is +0.28 at 2.9 sigma with 3 seeds on each side.

**`ftb11is` — block 0's structure matters, not its statistics.** Shuffling the weights of the
single proc block in `ftb11i` (78.78) lands at **77.81**, at or just below the random baseline
(78.08 +/- 0.19). Even one
early block contributes through its learned structure; its weight statistics alone are worth
nothing. This is the IN-1k counterpart of e2 on IN-100 (§3.8).

**`ftb0l` at 76.49** — proc everywhere but shuffled, biases skipped, all 12 blocks recalibrated —
lands **1.4 below random**. Destroying structure everywhere while keeping magnitudes is worse
than not using proc at all.

### 3.13.5 The method, and what must be said with it

**Measure rho at init, then scale the last ~25% of blocks to rho 1.4-2.0** — `v` and `proj` by
sqrt(r), `fc2` by r. No checkpoint, no pretraining, no data.

| | random | **recipe** | proc init |
|---|---|---|---|
| IN-100 / ViT-S | 84.55 | **87.02** | 86.15 |
| IN-1k / ViT-B (3 seeds) | 78.08 +/- 0.19 | **79.69 +/- 0.30** | 80.09 +/- 0.12 |

Three conditions, all learned the hard way:

- **rho is not scale-free.** ViT-B needed `fc2` x81.9 to reach 1.4 where ViT-S needed ~3-4. The
  recipe is *measure-then-set*; never copy a factor across architectures (§3.10).
- **The ceiling is ~2.6 and is init-dependent** — rho 1.4 trained 3/3 under timm but 1/3 under
  Xavier (§3.12.7, §3.12.8).
- **Late blocks only.** Early or all-block calibration is neutral-to-harmful at both scales
  (m1/m2 on IN-100; ftb4o 77.27 < r 78.08 on IN-1k).

~~For the nanochat port, GPT-style init sits at rho 0.0053 — x262 to reach 1.4, beyond anything
tested~~ — **superseded by §3.14**: that 0.0053 came from a nanoGPT-style *reconstruction* here,
not from nanochat, which zero-inits both output projections and therefore sits at rho **exactly
0**. Do not use the x262 figure. The remaining measurement is **trained** rho:
`nanochat/measure_init_rho.py --ckpt <dir> --step <n>`, which needs a checkpoint the clone does
not yet have. Whether late-block rho converges into 1.4-2.0 decides whether the port is worth
doing at all.

### 3.13.6 Open

- **Why the window is late-only.** Stream inflation was refuted as the explanation (§3.10).
- **The IN-100 / IN-1k early-block split** in §3.13.3.
- **Why h3's +0.77 did not replicate** on kdyck4 (§3.11.1) — the rho account was tested and refuted.

## 3.14 Relation to Fixup / ReZero / NormFormer — and what is actually novel here

Earlier drafts asserted this work "runs opposite to" the residual-downscaling literature. That
framing is too crude. Checked against the papers (2026-08-23), most of the tension dissolves,
one of the citations was being used backwards, and what remains genuinely novel is narrower than
claimed.

### The methods, and what they were for

| method | intervention at init | stated goal |
|---|---|---|
| **Fixup** (arXiv 1901.09321) | zero-init the **last layer of each residual branch**; scale remaining branch weights by `L^(-1/(2m-2))` | train **10,000-layer** ResNets without normalisation |
| **ReZero** (arXiv 2003.04887) | a single scalar `alpha = 0` gating each branch | train **120-layer** Transformers; 56% faster convergence at 12 layers |
| **SkipInit / T-Fixup** | scalar 0 on the branch; Fixup adapted to transformers | remove normalisation / warmup |

All start the network at the identity map. All are **trainability-at-depth** results, not
accuracy results. **nanochat implements exactly this** — `zeros_(attn.c_proj)`,
`zeros_(mlp.c_proj)` — which is why rho measures **0** there (`nanochat/measure_init_rho.py`).

### Why the conflict is mostly apparent

- **Different objective.** They prevent divergence in very deep, normalisation-free networks.
  The ViTs here are 12 layers with LayerNorm and 50-epoch warmup, where trainability was never
  the binding constraint. Note our *failure* mode is divergence from scaling **up** too far
  (rho >= 2.8, §3.12.7) — precisely what these methods exist to prevent. We operate inside the
  regime they made safe.
- **They are uniform; our result is that the sign flips with depth.** The same 3-block
  intervention is **-1.23** at blocks 3-5, **-0.45** at 6-8 and **+1.93** at 9-11 (§3.12.4). No
  single global down-scaling rule can express that, and none of these papers varies its rule by
  depth.
- **Init is not the destination.** ReZero's `alpha` is *learned*; zero is a starting point. Our
  recipe sets at init what they let training discover. Whether those agree is exactly what a
  trained-checkpoint rho measurement would settle.

### NormFormer — traced to source, and the direction flips with model scale

**Provenance note.** An earlier version of this section quoted NormFormer as saying "the optimal
weighting of residuals is larger at earlier than at later layers". **That sentence is not in the
paper** — checked against arXiv v1, the current arXiv abstract, and the ar5iv full text. It is a
search-engine paraphrase that appeared in two independent snippets, which is what made it look
verified. Do not cite it as a quote.

The verbatim abstract establishes only a **gradient** asymmetry, remedied by **normalisation**:

> "During pretraining, the Pre-LayerNorm transformer suffers from a gradient magnitude mismatch:
> gradients at early layers are much larger than at later layers. These issues can be alleviated
> by our proposed NormFormer architecture, which adds three normalization operations to each
> layer..."

**But the underlying phenomenon is real and is in the body.** The paper has a **ResScale**
ablation, and §4.2 reports that the learned residual weights vary with depth *and that the
pattern reverses with model size*:

> "At 125M and 355M parameters, the weights in the later layers are lower ... whereas at the
> largest scale, 1.3B, the weights are larger deeper into the network."

with the accompanying finding that scaling residuals **"helps at small scale and hurts large
scale"**.

So the paraphrase captured only the small-model half. At their **largest** scale, later layers
learn **larger** residual weights — the same direction as our depth asymmetry. At 125M-355M the
direction is the opposite.

Two things follow:

- **The relationship to this study is scale-conditional, not a clean agreement or a clean
  conflict.** Cite it as: NormFormer observes depth-varying learned residual weights whose sign
  depends on model size, agreeing with our direction only at 1.3B.
- **It independently corroborates the scale-dependence we keep hitting.** We find the
  early-block mechanism differs *in kind* between IN-100 and IN-1k (§3.10.5); they find residual
  scaling reverses between 355M and 1.3B. Residual-magnitude effects appear not to be
  scale-transferable in either modality — which is the strongest argument for the
  measure-on-your-own-model rule in §3.10.1.

*Caveat: the §4.2 lines above were extracted from the ar5iv rendering rather than read in the
PDF. Worth eyeballing directly before they go in a paper.*

### nanochat schedules residual magnitude by depth — verified from source

Independent of the literature, one directly checkable fact stands. In `nanochat/gpt.py`
`init_weights`:

```
resid_lambdas[i] = 1.15 - 0.10 * i / (n_layer - 1)     # 1.150 -> 1.050
x0_lambdas[i]    = 0.20 - 0.15 * i / (n_layer - 1)     # 0.200 -> 0.050
```

and the block input is `resid_lambdas[i] * x + x0_lambdas[i] * x0`. Both coefficients **decay
with depth**, so the residual stream receives progressively less amplification the deeper you
go, and each block's write matters relatively more with depth. Measured directly
(`nanochat/measure_init_rho.py`): `||r_in||` grows 37 -> 161 across a depth-12 trunk.

That is a depth-dependent residual-magnitude schedule, arrived at independently and pointing the
same way as our depth asymmetry. It is *suggestive convergent evidence*, and it is read off the
code rather than a paper — but it is not a claim by the nanochat authors about why, and should
not be presented as one.

### What is actually novel here

Possibly not the *direction* itself: nanochat's `resid_lambdas` schedule already leans that way
(verified from source), NormFormer sees it at 1.3B (§3.14), and the general area is occupied by
Fixup / ReZero / LayerScale. What this study adds:

1. **A measured optimum band**, rho **1.4-2.0**, with a soft ceiling at ~2.6 and divergence by
   2.8-3.0 (§3.12.7).
2. **The depth at which the sign flips** — roughly the last 25% of blocks — and direct evidence
   that applying the same intervention earlier is *harmful* at both scales (§3.12.4, ftb4o).
3. **That it is worth 1.5-2.4 points at fixed, moderate depth**, i.e. as an accuracy
   intervention rather than a trainability one.
4. **That compounding, not factor magnitude, drives divergence** — one block at x267 trains
   cleanly, four cascading at x173 do not (§3.10.4).

### Consequence for the nanochat port

Expectations should come down. Earlier notes quoted **x262 headroom** from "GPT-style init"; that
figure came from a nanoGPT-style *reconstruction* in `measure_init_rho.py`, not from nanochat.
nanochat is at **exactly zero** on the branch while *already* applying the qualitative insight
through `resid_lambdas`. A zero cannot be multiplied, so porting the recipe means **replacing**
a deliberate, well-tuned init rather than adjusting it.

The honest test is therefore not "how much can we scale up" but **where trained rho settles**:

- late-block rho converging to ~1.4-2.0 => the two results agree on the destination and differ
  only on the starting point; the recipe becomes "arrive faster"
- settling far below => language genuinely prefers small late-block writes and the vision
  finding does not transfer

`nanochat/measure_init_rho.py --ckpt <dir> --step <n>` performs that measurement. It needs a
trained checkpoint, which does not exist in the clone yet.


### 3.14.1 A missing baseline: LayerScale is disabled in every arm here

`models/vision_transformer.py` supports LayerScale (`init_values=`, line 271) but every arm in
this study runs with it **off** — verified directly: `blocks[0].ls1` and `ls2` are both
`nn.Identity`.

That matters because LayerScale (CaiT / DeiT-III) is the standard modern mechanism for exactly
the quantity this study manipulates: a learned per-channel scale on each residual branch,
initialised at ~1e-5. **Every number here is therefore measured against a ViT without the
standard residual-scaling mechanism**, which is the first objection any reviewer will raise.

The argument that it does not close the gap — LayerScale scales *down* and is *uniform* across
depth, so it cannot express the sign flip in §3.12.4 — is an argument, not a measurement. Four
arms would settle it (random and recipe, each +/- LayerScale) and the flag already exists.
**Until that is run, "+1.58 over random init" should be read as "over a no-LayerScale
baseline".**

### 3.14.2 Standing of the claims, 2026-08-23

A day of literature checking moved **no experimental number**. What it moved was the story around
them, all in the same direction. Recorded here so the corrections are not silently re-absorbed.

| claim | standing |
|---|---|
| "the literature supports our direction" | **retracted** — NormFormer's depth pattern *reverses with model size*, and at our model scale it points the other way (below) |
| "GPT-style init is x262 from the band, best possible port target" | **retracted** — that was our own nanoGPT reconstruction; nanochat is at rho **exactly 0** (§3.14) |
| "+1.58 over random init" | **asterisked** — the baseline has LayerScale disabled (§3.14.1) |
| every measured result in §2, §3.10-3.13 | **unchanged** |

**The scale mismatch matters.** NormFormer runs 125M-2.7B; ViT-S is ~22M and ViT-B ~86M, i.e.
*below their smallest model*. In the 125M-355M range they report later layers learning **lower**
residual weights — the opposite of our finding — and the reversal appears only at 1.3B. So the
regime that brackets our models does not support us. Whether that is a real conflict depends on
whether their ResScale multiplies the branch or the skip, which is not resolvable from the ar5iv
extraction and needs the PDF.

**What is genuinely strengthened.** Residual-magnitude effects appear **not to transfer across
scale** in either modality: they reverse between 355M and 1.3B, and we find the early-block
mechanism differs *in kind* between IN-100 and IN-1k (§3.10.5). Two independent studies, two
modalities. That is a better-earned argument for the measure-on-your-own-model rule (§3.10.1)
than the agreement an earlier draft asserted.

**What to lead with.** The composition result is untouched by all of this and depends on no
literature claim: proc everywhere *except* the last block, that block calibrated —
**87.25 +/- 0.09 vs 86.15** on IN-100 and **80.63 +/- 0.18 vs 80.09** on IN-1k, 3 seeds each,
replicated at two scales (§3.10.4). Frame the work around that, with the depth sign-flip
(§3.12.4) as the mechanism, rather than around the recipe as a general init method.

**Process note.** The retracted NormFormer sentence was believed because *two independent search
snippets contained it verbatim*. They were the same summariser. Agreement between search results
is not verification — check the source before a claim enters this document.

## 4. Tests

### Done

| arm | question | answer |
|---|---|---|
| c1 | does proc-like early rho alone create the attractor? | **no** — lands exactly on r (§3.6) |
| c2 | does random-like early rho remove the cap? | **no** — lands on p (§3.6) |
| d0/d1/d2 | co-adaptation vs stream statistics | **stream statistics** — freezing 0-8 does not preserve the intervention (§3.4) |
| warmup probe | is the erasure window an LR artifact? | **no** — shorter warmup erases *faster* (§3.5) |
| e1 | is proc's early-block structure the cause? | **no** — shuffling 0-8 keeps the proc advantage (§3.7) |
| e2 | is proc's early-block weight-norm profile the cause? | **no** — 1.24 *below* random (§3.8) |
| e3 | is it the **value distribution** of proc's early weights? | **no** — lands on e2, not e1 (§3.8.3); and e4 later showed the *shuffle*, not the value distribution, drove e3 low (§3.12.6) |
| e4 | was e3's deficit the shuffle or the random late blocks? | **both, superadditively** — shuffle alone -0.26, random late alone -0.26, together -2.07 (§3.12.6) |
| f1/f2/f3 | does the contribution live in a **few extreme weights**? | **no, the opposite** — clipping them beats proc init (§3.8.3), though this does *not* replicate on kdyck4 (§3.11) |

Verified at launch: e3 shuffles 9 blocks x 13 tensors and holds 9-11 random (36 key removals);
f1 clips 36 tensors with norms preserved to 4 d.p. (`blocks.0.attn.qkv.weight`
55.7273 -> 55.7273, kurtosis 4.56 -> 4.34).

**All IN-100 arms are complete.** Note "on both checkpoints" would overstate it: the kdyck4
pass covers the 26 arms in §3.11.1, while the `n*` rho sweep and `x0`/`x14` exist on the original
checkpoint only — they set `--target_ratio_absolute` (or load no checkpoint) and are therefore
bit-identical across checkpoints by construction, so re-running them would reproduce the same
numbers exactly.

### Historical: how the IN-1k arms were commissioned

The 2026-08-16 batch (ftb4j/4k/4l/4m/4n/4o, ftb5i/6i) and the 2026-08-17 batch
(ftb8i/11i/7i/5h/6h) all use `results/pr_vitb_n/pr_6066174_final.pth`, the same checkpoint as
every prior IN-1k arm, so they compare directly against r = 78.08 and p = 80.09.
(`results/pr_vitb/pr_27267764_final.pth` also exists and is *not* used; §3.11 shows swapping
proc checkpoints moves p by ~0.8 and can flip a result outright.) **Results are in §3.10** —
this section records only the commissioning rationale.

The tier-3 arms (4l/4m/4n/4o) all calibrate the **early** blocks 0-7, the region IN-100 says is
harmful (m1, m2, §3.12.4). That prediction was **confirmed**: ftb4o 77.27 against random's
78.08 (§3.10).

Launched via the `b_vitb_*.sh` chain launchers, so **no `--requeue`** (§5.2). `num_workers` must
be CPU-derived — the 2026-08-17 scripts arrived without it and would have run at 10 instead of
48 (§5.3).

Earlier proposals now resolved: **e4** (launched, §3.12.6), **n25/n30/n40** rho upper end
(launched, §3.12.7), **rho at init across standard inits** (done, §3.12.5), **Xavier variant**
(done, §3.12.8). Still open: **depth fraction on a non-12-block model** — whether "last ~3 of
12" is ~25% of depth or an absolute count is untested and matters for any port.

### Still running

#### Answered 2026-08-21

| question | arm | answer |
|---|---|---|
| does the last-block composition replicate at ViT-S? | q1 / q2 | **yes**, +1.10 over proc init (§3.10.4) |
| is the early-block contribution the **norm profile**? | ftbnorm | **no** — 78.28 +/- 0.32 vs random 78.08, 0.9 sigma (§3.10.5) |
| does **structure** matter at a single block? | ftb11i / ftb11is | **yes** — the shuffle costs 0.97 at 4.8 sigma (§3.10.5) |

#### Running

8 jobs. The clip question is **answered** (§3.10.8: random's outliers are not the mechanism —
removing them costs ~0.37). Two arms now test what the elimination leaves.

| group | seeds | question | ETA |
|---|---|---|---|
| **ftbqm** | 3 | positive control: does proc's **value multiset**, in a random arrangement, reproduce ftb4e3's 80.17? (§3.10.9) | ~22h |
| **ftbcomp1** | 3 | composition with **one** proc block instead of four — does the early contribution saturate at block 0? | ~22h |
| ftbclip5 | 2 running | completes the clip sweep at 5% | ~1h |

**Predictions on record** (stated before the runs, for calibration):

| arm | predicted | actual |
|---|---|---|
| ftbclip01 | 78.5 (78.2-79.0) | **77.76** — below range |
| ftbclip1 | 78.3 (77.9-78.8) | **77.73** — below range |
| ftbclip5 | 77.8 (77.0-78.3) | tracking ~77.6 — in range |
| ftbqm | **80.17**, i.e. reproduces ftb4e3 | pending |
| ftbcomp1 | **80.43** if the two effects are additive | pending |

The clip direction was right (clipping does not recover the gap) and the magnitude was wrong
(clip01 was expected at or above random; it is below).

### Considered and dropped

- **Per-block LR scaling on 0-8** (`c3`, set up but not launched). Premise was reduced
  plasticity; §3.1 refutes it. Note `--learning_rate_scaling_params` is dead code in
  `8c7ec84` — the working path is `--custom_lr_layer` + `--custom_block_targets_scale`.
- **Norm-matched proc** — rescale proc 0-8 weights to `||W||=38.4`. Confounded: rescaling
  changes the function, not only the plasticity. Needs new code.
- **Depth control** (freeze/delete late blocks) — refuted by §3.1; there is no effective-depth
  collapse to reproduce.

### Worth running next

- **The true b2 analogue** — proc 0-**8** copied, blocks **9-11** scaled, proc-matched target.
  Never run on ViT-B (§3.13.3). `ftb4jd` currently substitutes for it on the 0-7/8-11 boundary
  with a halved target; if that result is hard to read because of the halving, this is the clean
  version.
- ~~a2 seeds on IN-1k~~ — **done** (job 29407014): 76.90 / 78.20 / 77.02 = **77.37 +/- 0.72**,
  i.e. -0.71 against random, where the single seed had suggested -1.18. Still the one IN-1k
  number that does not fit, but less extreme than it looked.
- **rho trajectory for c1/c2/e1/e2/e3** once they are needed — `launch_trajectory.sh`,
  ~1 min per checkpoint, no training.
- **Probe *when* the attractor can still be perturbed**: intervene at epoch 20/50/100 rather
  than at init. Erasure completes by ~epoch 20-70 (§3.2), so an intervention applied after
  the transient may behave differently from one applied at init.
- **Port the cause-side arms (c1/c2/e1/e2/e3) to IN-1k** — e3 has resolved (negative, and
  §3.12.6 shows the shuffle confound explains it), so this no longer narrows to one or two arms.
  Low priority: the cause-side eliminations all came back negative, and the productive thread is
  now the recipe, not the erasure.
- **The nanochat port** (deferred, not scheduled). The prerequisite is a pre-check, not a
  training run: measure rho at init per block in nanochat's own model. **Done** —
  `nanochat/measure_init_rho.py` reports rho **exactly 0** in every block, because nanochat
  zero-inits `attn.c_proj` and `mlp.c_proj`. The x262 figure in §3.12.5 is a nanoGPT-style
  reconstruction, not nanochat. See §3.14: the recipe cannot be applied as a multiplier, and the
  remaining measurement is **trained** rho via `--ckpt/--step`.
- If *every* candidate is exhausted including e3, the next move is not another arm but a
  different measurement: compare the full weight *and* activation statistics of e1's blocks
  0-8 against c1's and e2's, and find what actually differs. e1 and c1 bracket the effect
  (85.15 vs 84.42) with both structures destroyed.

---

## 5. Notes and caveats

### 5.1 Block 0 must be excluded from the scaled set

Proc's block 0 sits at `rho_attn = 5.43`, versus 0.015-0.106 for blocks 1-8. This is the
random-patch-embed artifact: the pr checkpoint has no usable `patch_embed`/`pos_embed`/
`cls_token` (they are dropped on load), so block 0 reads an essentially random embedding and
its ratio says nothing about proc pretraining.

The first c1/c2 attempt scaled blocks 0-8. **All three c1 seeds diverged**, and the reason
generalises to any use of `--init_method_scaled_blocks` that includes block 0:

1. Matching proc's block-0 ratio multiplies a *random* block 0's write by 17.5x
   (`sqrt` -> 4.20 on v/proj, 5.78 on `fc2`).
2. That inflates the residual stream entering block 1 by ~6x.
3. The scaler is **sequential and re-measures `current_stats` per block**
   (`main.py`, non-`simultaneous_init_scaling` path), so every downstream block now measures
   a ~6x smaller rho and is scaled **up** to compensate — block 1: current 0.151 -> 0.025
   after inflation, target 0.056, giving 1.50 (observed 1.578).
4. An arm intended to scale *down* therefore applied factors of **1.5-10.5** across blocks
   0-8, and activations overflowed under AMP: `train_grad_norm=nan` from epoch 37, accuracy
   decaying 16.7 -> 5.5, `assert math.isfinite(loss_value)` (`engine.py:124`) at epoch 50.

Two lessons. **(a)** Exclude block 0 from any early-block matching; c1/c2 now scale blocks
1-8. **(b)** Because the scaler re-measures after each block, errors in early blocks cascade
into every later factor. Always read the printed factors and check their direction matches
the arm's intent before letting a run proceed — the sign is not guaranteed by the method name
(see the `downscale_pr` note below).

Note this is a property of *imposing* rho=5.43 on a random block 0 by scaling its weights, not
of the ratio itself: `p` trains fine with its own block 0 at 5.43.

**A second, smaller outlier sits in the MLP.** Proc's `rho_mlp` for blocks 1-8 is
0.231, 0.496, 0.666, **2.547**, 0.152, 0.083, 0.028, 0.029 — block 4 is ~10x its neighbours
(0.67, 0.15) and ~10x random init's flat ~0.25. So the early-block MLP ratios are *not*
uniformly below random's, and matching them scales some blocks up even in the "downscale"
direction. With block 0 excluded, c1's attn factors are all < 1 (0.60-0.92) as intended, but
the mlp factors run 0.61-3.56, and block 4's 3.56 means `fc2 x 12.65` (the printed value is
`sqrt(scale_sq)`; `fc2` takes the square). **This turned out to be harmless** — with block 0 excluded, c1 and c2
train normally (61.9% at epoch 63 and 57.2% at epoch 55 respectively). The block-0 cascade was
the whole problem.

**Do not use `grad_norm: nan` as a divergence signal.** Occasional non-finite grad norms are
routine under AMP — the scaler skips those steps — and the fully healthy 300-epoch runs logged
56-107 of them (r: 56, a1: 84, p: 107) while reaching 84-86% top-1. The real signature of the
block-0 failure was a **flat loss (~4.32) with decaying accuracy** (16.7 -> 5.5), not the NaN
lines. Judge divergence from the loss and accuracy curves.

### 5.2 Do not combine `--requeue` with a resubmitting chain launcher

The `b_vitb_*.sh` chain pattern resubmits inside a `while true` loop whenever the job leaves
`RUNNING`, and it predates `--requeue`. Adding `#SBATCH --requeue` (or the sbatch flag) gives
the *same* restart behaviour a second time: on preemption Slurm requeues the job **and** the
launcher submits a new one. Both then train into the same `output_dir`.

This happened to a2 seed 2 (`results_IMNET_BASE_29407014/s2`): two jobs resumed from the same
checkpoint at epoch ~170 and ran in parallel to ~254, producing 340 log lines for a 300-epoch
run and 85 duplicated epochs. A third job spawned while the launcher was being killed.

**How bad was it?** Much less than it looked. `main.py` sets `manual_seed`,
`cudnn.deterministic=True`, `benchmark=False` and `use_deterministic_algorithms(True,
warn_only=True)`, so the two trajectories are near-identical:

| | |
|---|---|
| train_loss abs diff over 85 dup epochs | mean **1.9e-3**, max 8.8e-3 |
| test_acc1 at ep 174 / 254 | 76.39 vs 76.64 / 76.87 vs 76.56 |
| bit-identical epochs | 0/85 — `warn_only=True` permits non-deterministic ops, and 4-way NCCL all-reduce is not order-deterministic |
| **inflation of best_top1 from taking max over both** | **+0.00** (both peak at 76.87) |

So the accuracy number is unaffected and no restart was needed — the fix was to cancel the
duplicates and let one trajectory finish. **Residual caveat:** s2's per-epoch checkpoints
between epochs 170-254 may alternate between the two runs, so s2 is unsuitable for
`launch_trajectory.sh`-style per-epoch analysis. Use s1 for that.

**Rule:** pick one restart mechanism. `--requeue` alone is preferable — it preserves the job
id, needs no supervising shell, and cannot double-submit. If using the chain launcher, drop
`--requeue`.

### 5.3 Other notes

- **b1's target was corrected mid-experiment.** It originally scaled toward the fully random
  model, so target and model differed in blocks 0-8 as well as 9-11, confounding the
  comparison. It now targets (proc 0-8 + random 9-11). The corrected version scales *down*
  (factors 0.27-0.42); the original scaled *up* (1.5-2.5x). Only the corrected runs are in §2.
- `downscale_mixed_match_delta_norms` (`main.py`) was added for b1 and is **uncommitted** —
  a `git pull` will drop it. b2 needs no new code, using
  `--init_method_copied_blocks "0;1;...;8"`, whose bracket-less form was added in `8c7ec84`.
  The two constructions were cross-checked: b2's scale factors are bit-identical to the
  earlier custom branch.
- On ViT-S the method names invert: because proc's late blocks sit *below* random here,
  `downscale_pr` toward a random target scales **up**. Read the printed factors, not the name.
- a2 seed 1 is still finishing (epoch 273/300); its mean is over 2 seeds.
- **`--grad_norms_json` defaults to a relative path (`grad_norms.json`) shared by every job**
  and is read-modify-written each epoch, so concurrent jobs corrupt it and crash with
  `JSONDecodeError` at `main.py:1233`. All arms here pass a per-run path. This affects any
  concurrent runs, vitbase included.
- `--enable_wandb false` crashes on non-`vit_base` models: `wandb_logger.update_config` is
  called unguarded at `main.py:1114` and `main.py:1154`. All runs here log to wandb.
- `main.py:1223` passes `custom_lr_transition_end=args.custom_lr_transition_start`, so the
  transition end is taken from the start argument. Harmless at 0; matters otherwise.
- Cluster noise: recurring `/dev/shm` exhaustion from job packing (mitigated with
  `--num_workers 5`, and 2 for one stubborn job) and preemption on `alldlc2`. Both recover
  via per-epoch checkpoints and `--auto_resume`.
- Accuracies are best-epoch top-1 on IN-100 at batch 512 and are not directly comparable to
  the IN-1k ViT-B numbers.

### 5.4 Divergence is a real failure mode of this intervention

Distinct from the `/dev/shm` and preemption noise (§5.2), several arms **diverged**: healthy
training followed by `grad_norm=inf`, then `nan`, then rising loss and collapsing accuracy,
ending at `assert math.isfinite(loss_value)` (`engine.py:124`).

| arm | where | scale factors involved |
|---|---|---|
| c1 (first attempt) | IN-100, epoch ~37-50 | block-0 cascade, 1.5-10.5 across blocks (§5.1) |
| n30 / n40 | IN-100 | target rho 3.0 / 4.0 (§3.12.7) |
| x14 | IN-100, 2/3 seeds | target rho 1.4 under **Xavier** init (§3.12.8) |
| ftb4j | IN-1k, epoch 55 (**twice, identically**) | block-11 `fc2` x **173** |
| ~~ftbrho~~ | **did NOT diverge** | block-11 `fc2` x 81.9 — trained clean past epoch 260 |

ftb4j is the clearest case: `upscale_random` with proc blocks 0-7 copied in, so blocks 8-11
were calibrated against a *proc* stream while the model's own late blocks were random — a huge
ratio, compounded across four blocks by the sequential re-measurement. Factors were
4.0 / 6.9 / 7.0 / 13.6 (attn) and 4.9 / 23.4 / 94.9 / **173** (mlp).

**How the printed factor maps onto weights.** `--init_method_scaled_attributes` defaults to
`v, proj, fc2`, and the printed `Scale for layer N` is `scale_sq ** (1/count)` where `count` is
how many of that sublayer's tensors are in the set (`main.py:1140`, `:1227`). So the MLP has
count 1 and the printed number **is** the `fc2` multiplier — it is *not* squared. Attention has
count 2, so `v` and `proj` each take the printed number and the attention delta moves by its
square. An earlier version of this doc had that backwards in both directions.

**Diagnostic rule:** isolated `grad_norm: nan` lines are normal under AMP — healthy 300-epoch
runs logged 56-107 of them. Divergence is *flat or rising loss with decaying accuracy*. Check
the loss curve, never the NaN count (§5.2).

**Before launching any arm with large scale factors**, read the printed `Scale for layer N`
lines — *all* of them, including the last block. Treat a large MLP factor as a reason to
*inspect*, not to abort: ftbrho trained cleanly at **81.9** (see §3.10), so the old
"above ~10 is a divergence risk" rule was too conservative by ~8x. The failures share a
different feature — calibrating blocks against a stream from a *different* model (ftb4j) or
pushing the target rho itself past ~2.6 (n30/n40, x14). The factors grow with depth inside a scaled
window, because each scaled block inflates the stream the next one is measured against, so the
final block is the one that decides. Reading a partial log and stopping early is how ftbrho got
recorded here at 27.6 when its real worst case was 81.9.

### 5.5 Do not submit a large IN-100 batch onto nodes running IN-1k jobs

**This killed a healthy run.** On 2026-08-16 a 41-job IN-100 batch went to `alldlc2`; four of
those 2-GPU jobs landed on `dlc2gpu24`, where `ftbrho07` had been training cleanly for 4h46m.
Roughly a minute later it died at epoch 63 with
`RuntimeError: Caught RuntimeError in DataLoader worker process 8` — the same `/dev/shm`
exhaustion the in-job retry loop exists to absorb.

The timeline reads as coincidence and is not:

```
19:21:02  i100h3 starts on dlc2gpu24   (2 GPUs)
19:21:39  i100e4 starts on dlc2gpu24   (2 GPUs)
19:22:08  ftbrho07 DIES; its 4 GPUs free and two more i100 jobs start that same second
```

The two jobs at 19:22:08 started *because* it died, but the two at 19:21:02/19:21:39 preceded
the death by a minute — those are the cause, not the effect. Read the co-location times before
concluding a run failed on its own.

Two properties made the damage worse than it needed to be:

- **The `vitbase_runs/run_train_ftb*.sh` scripts have no in-job retry loop** — unlike
  `run_train_i100.sh`, which retries up to 6 times. A single worker crash is terminal.
- **`--output_dir` is keyed to `$SLURM_ID`**, which defaults to `$SLURM_JOB_ID`. A plain
  resubmit gets a new job id, hence a new directory, and `--auto_resume` (default `True`) finds
  nothing and **silently restarts from epoch 0**. Resume with the original id explicitly:

```bash
sbatch --export=ALL,SLURM_ID=29453948 \
       --exclude=dlc2gpu18,dlc2gpu19,dlc2gpu21,dlc2gpu22,dlc2gpu23,dlc2gpu25 \
       run_train_ftbrho07.sh
```

**Rule:** before submitting an IN-100 batch, list the nodes hosting IN-1k jobs and pass them as
`EXCLUDE=` (launch.sh) or `ExcNodeList` (`scontrol update` on already-queued jobs — note the
field is `ExcNodeList`, not `excludenodelist`, which fails silently and reports success).

### 5.6 A 0-byte JSON file can kill a run permanently

`m1` seed 2 died at epoch 275 and every retry died identically:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
  main.py:1462   current_grad_norms_list = json.load(f)
```

`m1_band_k4/s2/grad_norms.json` was **0 bytes**. The old code did a read-modify-write:

```python
if os.path.exists(args.grad_norms_json):        # true for a 0-byte file
    with open(args.grad_norms_json, 'r') as f:
        current_grad_norms_list = json.load(f)  # raises, every single time
...
with open(args.grad_norms_json, 'w') as f:      # truncates BEFORE the dump
    json.dump(current_grad_norms_list, f, indent=4)
```

A job killed between the truncate and the dump leaves an empty file, and `os.path.exists` does
not test that the contents parse. The failure is therefore **permanent, not transient** — the
in-job retry loop (`MAX_RETRIES=6`) burned all six attempts in 9 minutes on the same exception.
This is a different failure class from preemption: retrying cannot fix it.

Fixed in `main.py` on 2026-08-17, both halves:

- the read is wrapped in `try/except (json.JSONDecodeError, ValueError)` and falls back to an
  empty history with a warning. Grad norms are diagnostic; they must not be able to take a
  275-epoch run down.
- the write goes to `<path>.tmp` and then `os.replace()`, which is atomic within a filesystem,
  so an interrupted write can no longer truncate an existing history.

**Scanning for others: use `find -L`.** `results/` is a symlink, so a plain
`find results -name "grad_norms*.json"` traverses nothing and reports zero files — which reads
exactly like "no corruption found". With `-L` there are 195 files, of which this was the only
bad one.

```bash
find -L results -name "grad_norms*.json" -empty
```

### 5.7 The chain launcher misreads a fast crash as completion

`b_vitb_*.sh` stops the chain when a job runs for under ~5 minutes, on the assumption that a
short run means training was already finished. A `/dev/shm` DataLoader crash also takes under a
minute, and is indistinguishable to that check:

```
RuntimeError: could not unlink the shared memory file /torch_..._4 : No such file or directory
  ... in DataLoader worker process 43
Runtime 62s too short. Stop chain.
```

The arm then looks finished while sitting at epoch 1. This has cost `ftbrho07` (epoch 63),
`ftb11i` (epoch 289), and `ftb4jd` twice, and nearly took `ftb7i` at epoch 295.

**Fixed 2026-08-18** by giving the ViT-B scripts the in-job retry loop that
`run_train_i100.sh` already had — six attempts, 30s apart, `--auto_resume` continuing from the
last epoch checkpoint, so a transient crash costs only the epoch in flight instead of the chain.
Applied to `run_train_ftb{4jd,5h,6h,7i,8i,9i,10i,11i,rho,comp*}.sh`; older ViT-B scripts still
lack it.

**A requeued job looks like a fresh one.** `--requeue` puts the job back under the *same* job id,
so `sacct` shows the current attempt's elapsed time and **no `PREEMPTED` row**. A job that has
been preempted twice and is 285 epochs in is indistinguishable from one started three hours ago
unless you look at the log. The evidence is the `Auto resume checkpoint` line:

```bash
grep -c "Auto resume checkpoint" logs/ft_<jobid>_<arm>.out   # number of resumes
grep -c '^{' results/imnet_base/results_IMNET_BASE_<slurm_id>/s0/log.txt   # true progress
```

**Judge progress by epochs in `log.txt`, never by `sacct` elapsed.**

**Diagnostic:** a chain that stopped with `Runtime <N>s too short` is only evidence of
completion if the job also exited **0**. Check the exit code, and check `log.txt` for the epoch
count, before believing an arm is done:

```bash
grep -c '^{' results/imnet_base/results_IMNET_BASE_<slurm_id>/s0/log.txt   # should be 300
```

### 5.8 Mid-training rank does not predict final rank

Peeking at a running arm is close to worthless on ImageNet-1k, and the p-vs-r pair shows why.
At epoch 168 of 300 they are effectively tied; their finals differ by 1.9 points. **The entire
separation appears in the cosine tail**, and arms carrying proc weights gain far more of it:

| arm | best by ep 168 | final | tail gain |
|---|---|---|---|
| p proc init (s0) | 77.08 | 80.17 | **+3.09** |
| ftb8i proc 0-3 | 77.61 | 79.70 | +2.09 |
| r random (s0) | 76.89 | 78.28 | +1.39 |
| ftbrho s0 (random + rho 1.4) | 78.67 | 80.03 | +1.36 |

A random-init arm calibrated at init leads at epoch 168 and finishes *behind* proc init. So a
mid-training lead does not survive, and a mid-training deficit is not fatal: the plausible
landing zone for an arm at 77.2 spans roughly 78.7 to 80.3 depending only on which tail profile
it follows.

**Rule: do not rank IN-1k arms before epoch ~290.** Where an early read is unavoidable, compare
against the *same arm family* at matched epoch and state the tail-gain range explicitly. This
document has twice recorded early signals that did not survive — g2-g1 at epoch 194 (+1.65,
final +0.87) and ftb4l reading exactly the random baseline's final at epoch 175 by coincidence.

### 5.9 Concurrency is capped by CPUs, not GPUs

The `alldlc2_gpu-h200` QOS caps the account at `cpu=2304, gres/gpu=48`. The partition sets
`DefCpuPerGPU=48`, and every ViT-B arm takes 4 GPUs, so **each job claims 192 CPUs**:

```
2304 / 192 = 12 concurrent ViT-B jobs, hard ceiling
```

The GPU cap is not what binds — 12 jobs use 48 GPUs and 2304 CPUs simultaneously, but in
practice the CPU total is reached first whenever any job is smaller than 4 GPUs. Jobs beyond
that queue with reason `MaxCpuPerAccount`, which is **expected throttling, not a failure**
(§4 lists the reason codes worth distinguishing).

This is a direct consequence of the CPU-derived `num_workers` policy: maximising workers per job
makes CPUs the scarce resource rather than GPUs. It is the right trade — CPU-derived workers gave
a 1.6x speedup on IN-100 (§5.3) — but the wall is 12 concurrent ViT-B jobs, and lowering
`num_workers` is the only way to raise it.

```bash
# current usage against the cap
squeue -u $USER -h -t R -o '%C' | awk '{s+=$1} END{print s" CPUs of 2304"}'
sacctmgr show qos where name=alldlc2_gpu-h200 -P -n format=Name,MaxTRESPA
```

### 5.10 Two IN-100 seeds hang DDP deterministically — unexplained

`q1` seed 1 and `q2` seed 2 have each failed **four times**, always at exactly **1:09**, across
**four different nodes** (dlc2gpu09 / 10 / 13 / 32). Crucially, dlc2gpu10 ran `q1` seeds 0 and 2
to completion while failing seed 1, so it is **not a bad node** — an earlier note in this document
said it was, and that was wrong.

```
[rank0] Watchdog caught collective operation timeout:
        WorkNCCL(SeqNum=1, OpType=ALLREDUCE, ...) ran for 600067 ms before timing out
```

What is ruled out:

- **Not a corrupt resume** — the output directories are empty; nothing was ever written.
- **Not our init code** — the hang is at DDP's *first* collective (`SeqNum=1`), reached before
  any scaling logic. `main.py`'s barriers (1221 / 1309 / 1434) live inside init-method branches
  that `q1`'s `default` init never enters.
- **Not transient** — six in-job retries x 10 min NCCL timeout produce the identical 1:09 runtime
  every time, which is exactly what distinguishes a deterministic fault from a flaky one.

**No explanation** for why a seed value would deterministically hang DDP's first collective.
Worked around by launching a fresh **seed 3** for each arm, so both reach n=3 regardless.

**General lesson:** an identical failure *duration* across attempts is strong evidence of
determinism. Repeated ~1:09 failures were misread here as bad luck twice before the pattern was
noticed; the giveaway is that retries cost a fixed timeout each, so the total is constant.

### 5.11 The 10 TB workspace quota — the cause of a whole day of "mysterious" failures

On 2026-08-23 every running job died inside a 90-second window (09:33:26-09:34:38) across six
nodes, and every resubmission then failed in 2-3 seconds with exit `0:53`. The cause was simply
that the workspace hit its **10 TB limit**:

```
results  8.9 T      wandb  800 G      logs  1.7 G      = 9.7 T of 10 T
touch logs/.wtest  ->  Disk quota exceeded
```

**The signatures are worth recognising, because none of them looks like a disk problem:**

| symptom | what it actually was |
|---|---|
| log stops **mid-epoch, healthy loss, no traceback** | job could not write its checkpoint |
| job fails in **2-3 s**, exit `0:53`, no log file at all | slurm could not create the log |
| `ALLREDUCE SeqNum=1` timeout, always exactly **1:09** | 6 in-job retries x 10 min against a persistent fault |

**A failure with no traceback should prompt `quota`/`df` before any hypothesis about nodes,
seeds or NCCL.** Note the third row above was *not* caused by the quota: it is a separate fault
(§5.12), and attributing it to the quota was itself a misdiagnosis. Two independent problems were
active the same day, which is why each fix appeared to only partly work.

**Root cause: unbounded per-epoch checkpoints.** Every run writes a full ~1 GB optimizer
checkpoint each epoch and never prunes — 300 epochs x ~1 GB x ~130 runs.

**The cleanup rule.** Keep `checkpoint-best.pth`, any final, and the **newest two epochs** per run
directory; delete the rest. This freed **8.2 TB across 92,476 files**, 9.7 T -> 1.4 T.

**Keep two, not one.** `ftbclip5` seed 2's newest checkpoint (ep49) was **truncated at 0.71 GB**
against 0.97 GB for its peers — written as the disk filled. Its ep48 fallback survived intact. A
keep-newest-only rule would have made that run unrecoverable.

```bash
# verify before deleting: no protected files, nothing outside results/, no run left bare
grep -cE 'best|final' delete_list.txt                        # must be 0
grep -vcE '^results/(imnet_base|i100_playground)/' delete.txt # must be 0
# and confirm every resume-critical run keeps a full-size newest checkpoint
```

**Not yet fixed:** `--save_ckpt_freq` is still 1. Setting it to ~25 would cut checkpoint growth
~25x while bounding resume loss to 25 epochs. Until then this will recur — the 18 jobs relaunched
after the cleanup will themselves write ~5 TB.

### 5.12 `torchrun --standalone` collides on a fixed rendezvous port

`--standalone` pins the rendezvous store to **`localhost:29400`**. Any second torchrun on the
same node — one of ours, or another user's — binds the same port, the two jobs' ranks
cross-connect, and both deadlock at the first collective:

```
[rank0] Watchdog caught collective operation timeout:
        WorkNCCL(SeqNum=1, OpType=ALLREDUCE, ...) ran for 600067 ms before timing out
```

**The decisive evidence:** jobs 29490166 and 29490167 ran on the *same node* (dlc2gpu03) — one
completed in 5:44, the other hung. Earlier a job alone on its node also hung, which fits, since
the collision is with *any* co-tenant, not only our own jobs. That is why it looked
node-independent and seed-independent.

**Recognising it:** the runtime is always **exactly 1:09**. Six in-job retries x a 600 s NCCL
timeout is a constant, so an identical failure *duration* across attempts is the giveaway. This
was misdiagnosed four times here — bad node, seed-keyed, then attributed to the 10 TB quota
(§5.11) — before the same-node pair made it unambiguous. Two independent faults were active on
2026-08-23 and the quota masked this one.

**Fix, applied to all 90 run scripts on 2026-08-23:**

```bash
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$MASTER_PORT --nproc_per_node=...
```

**Note slurm snapshots the batch script at submission**, so already-queued jobs keep the old
copy. After patching, pending jobs must be cancelled and resubmitted to pick up the fix; running
jobs are unaffected either way.

## 6. Reproducing

```bash
cd i100_playground

# training arms, 3 seeds each (~6h on 2x H200)
PARTITION=alldlc2_gpu-h200 NUM_WORKERS=5 ./launch.sh r p a1 a2 b1 b2   # the original grid
PARTITION=alldlc2_gpu-h200 NUM_WORKERS=5 ./launch.sh c1 c2 e1 e2 e3    # cause-side arms
PARTITION=alldlc2_gpu-h200 NUM_WORKERS=5 ./launch.sh f1 f2 f3          # outlier-clip sweep
PARTITION=alldlc2_gpu-h200 NUM_WORKERS=5 ./launch.sh d0 d1 d2          # freeze test
WARMUP_EPOCHS=5 RUN_TAG=_warm5 SEEDS=0 ./launch.sh p b1                # warmup probe

# rho over training, from checkpoints already on disk (~1 min/job, no training)
./launch_trajectory.sh                                  # default arms and epochs
EPOCHS="0 5 20 70 299" ./launch_trajectory.sh e1 e2 e3   # specific arms/epochs

# IN-1k post-hoc on the existing vit_base runs (~9 min/job, no training)
for arm in r p a1 a2 b1 s1011 s811 s711 sall; do
  for ep in 0 5 20 70 150 299; do
    sbatch --export=ALL,ARM=$arm,EPOCH=$ep run_posthoc_ratios_in1k.sh
  done
done

# IN-1k a2 seeds 1-2 (vitbase style, ~21h each)
./../vitbase_runs/b_vitb_3es.sh
```

Adding an arm means editing the `case` block in `run_train_i100.sh`; adding it to the
trajectory tooling means also editing the `case` in `run_posthoc_ratios.sh`, which is easy to
forget — e1/e2 silently produced nothing until that map was updated.

## 7. Artifacts and where things live

| what | where |
|---|---|
| **Figures** (4 charts: depth placement, composition at both scales, rho sweep, mechanism decomposition) | https://claude.ai/code/artifact/19f6f7f1-5244-444a-9b6e-62067c30160e |
| rho at init across weight inits (ViT) | `i100_playground/measure_init_rho.py` |
| **rho at init for nanochat** (reports rho = 0; `--ckpt/--step` for a trained model) | `nanochat/measure_init_rho.py` |
| IN-100 arm dispatcher (kdyck4 + `_k4` + `i100-playground-kdyck4` default together) | `i100_playground/run_train_i100.sh` |
| IN-1k arms | `vitbase_runs/run_train_ftb*.sh`, launched via `sbatch --requeue`, not the chain launchers (§5.7) |

**Init methods added to `main.py` during this study**, all no-ops unless explicitly selected:
`downscale_mixed_match_delta_norms`, `clip_outlier_weights`, `match_target_block_norms`,
`quantile_match_target_blocks` (+ `--quantile_source empirical|parametric`). Orthogonal flags:
`--target_ratio_absolute/scale/flatten`, `--clip_outlier_blocks`, `--outlier_clip_frac`,
`--weight_init`.

**Operational conventions now in force**, each earned the hard way:

- seeds pinned to seed-0's `SLURM_ID` so a run's seeds share one output directory (ViT-B paths
  are job-keyed; IN-100 paths are arm-keyed and need no pinning)
- `--requeue` rather than chain launchers — the chains do not handle `PREEMPTED` (§5.7)
- unique rendezvous port per job — `--standalone` collides on `localhost:29400` (§5.12)
- in-job retry loop on every ViT-B script (§5.7)
- **checkpoints must be pruned periodically** — `--save_ckpt_freq` is still 1, and unbounded
  per-epoch checkpoints hit the 10 TB workspace quota once already (§5.11). Keep
  `checkpoint-best`, any final, and the **newest two** epochs per run.

