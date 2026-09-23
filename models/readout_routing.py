"""Readout-only residual routing: a weighted mix over the 2L+1 sources feeds the classifier.

Follow-up to models/residual_routing.py (docs/residual_routing.md). There, every branch AND the
readout learned its own mix and routing cost ~1.9 points of top-1; the router went local and its
readout row soft-suppressed the early sources. This module isolates that one row: every block
connection stays VANILLA (the ordinary residual stream, Block.forward runs untouched) and only
the readout reads

    mixed = sum_{j=0}^{2L} w_j * h_j

where h_0 is the tensor entering block 0 (patch_embed -> _pos_embed -> norm_pre) and h_k
(k >= 1) is the k-th branch output in execution order (attn_0, mlp_0, ..., mlp_{L-1}) INCLUDING
ls/drop_path -- exactly what the vanilla Block ADDED to the stream, and exactly history[k] of the
full router. The mix replaces "the stream after the last block" and goes through the existing
norm -> pool -> fc_norm -> head.

The parametrization is LINEAR (a_j = w_j, no softplus): the fixed init [0]*17 + [1]*8 must be
exactly representable, 0 must be an ordinary point that a trainable weight can leave (softplus
cannot represent 0 and Adam's eps freezes near-zero coefficients; the router's mask gives exact
zeros with a deliberately zero gradient), and all-ones must make the model the vanilla one bit
for bit. Negative weights are allowed. With every w_j == 1 the accumulation below runs in the
vanilla stream's own fp32 summation order, so the FORWARD of 'learn_ones' at init is bit-identical
to the baseline built from the same seed (the mixer draws no random numbers), in fp32 and under
fp16 autocast. The backward is the same maths but not the same graph -- every branch output now
has a direct edge to the readout next to its path through the stream, so autograd accumulates
dL/dh_j in a different order: fp32 gradients agree to ~4e-7 relative (fp64: 5e-16), not bit for bit.

Modes ("last k blocks" = blocks L-k .. L-1 = sources 2(L-k)+1 .. 2L, i.e. 17..24 for k=4 and
19..24 for k=3 at L=12; x_b denotes the stream after block b, x_{-1} the embedding):

    fixed_last4      w = [0]*(2L-7) + [1]*8, a buffer -- can the head live on x_{L-1} - x_{L-5}?
    learn_last4      same init, nn.Parameter           -- does it put the embedding / earlier back?
    learn_ones       w = all ones, nn.Parameter        -- CONTROL: does a free readout row alone hurt?
    fixed_last3      w = [0]*(2L-5) + [1]*6, a buffer  -- the harder cut x_{L-1} - x_{L-4} alone
    fixed_emb_last3  fixed_last3 + the embedding (source 0), a buffer
                     -- readout input emb + (x_{L-1} - x_{L-4}): the input tokens skip to the head
    fixed_b0_last3   fixed_last3 + sources 0..2, a buffer
                     -- readout input x_0 + (x_{L-1} - x_{L-4}): the stream after block 0 skips to it

The weight is 1-D, so optim_factory.get_parameter_groups puts it in the no_decay group at the
backbone LR (no decay toward init, by decision). Gradient SIGN consistency per weight is counted
at every optimizer step (engine.train_one_epoch calls record_grad_sign) so that "moved by a
consistent signal" and "random-walked under Adam" are distinguishable in the log.
"""

import os

import torch
import torch.nn as nn

from models.residual_routing import source_labels

# mode -> (head sources that read 1, number of last blocks that read 1, trainable);
# None = every source reads 1
_SPECS = {
    "fixed_last4": ((), 4, False),
    "learn_last4": ((), 4, True),
    "learn_ones": (None, None, True),
    "fixed_last3": ((), 3, False),
    "fixed_emb_last3": ((0,), 3, False),
    "fixed_b0_last3": ((0, 1, 2), 3, False),
}
READOUT_MODES = ("off",) + tuple(_SPECS)


def _mix(w, sources):
    """sum_j w[j] * sources[j], accumulated in the stream's dtype (at least fp32) in source order.

    Under fp16 autocast the branch outputs are fp16 and the stream fp32; the vanilla residual add
    upcasts each h, so `1.0 * h.to(fp32)` summed in this order IS the vanilla stream bit for bit.
    (A 0-dim fp32 weight times an fp16 tensor would stay fp16 -- hence the explicit cast.)
    Elementwise mul is not on autocast's cast lists, so the products stay in the acc dtype.
    """
    acc = torch.promote_types(sources[0].dtype, torch.float32)
    mixed = w[0] * sources[0].to(acc)
    for j in range(1, len(sources)):
        mixed = mixed + w[j] * sources[j].to(acc)
    return mixed.to(sources[0].dtype)


# Eager, the 25-term sum is 75 unfused memory-bound kernels over B*N*D elements (+11.5% per
# fwd+bwd micro-step on an H200 at batch 128, measured); inductor fuses it into one kernel that
# reads every source once (+1.9%, and no fp32 copies: -1.7 GB), with the same summation order, so
# the compiled path is still bit-identical to vanilla at w == 1 (the gate checks exactly that).
# CUDA only; CPU (the tests) stays eager. READOUT_MIXER_COMPILE=0 switches it off.
_COMPILE_MIXER = os.environ.get("READOUT_MIXER_COMPILE", "1") != "0"
_mix_compiled = torch.compile(_mix)   # lazy: costs nothing until called (~5 s, once per shape)


def first_late_source(mode, depth):
    """Index of the first source of the mode's late-block window (0 for learn_ones)."""
    last = _SPECS[mode][1]
    return 0 if last is None else 2 * (depth - last) + 1


def readout_init(mode, depth):
    """(initial weight vector of length 2*depth+1, trainable) for a mode; 'off' is refused."""
    assert mode in _SPECS, f"readout_routing mode {mode!r}"
    head, last, trainable = _SPECS[mode]
    n = 2 * depth + 1
    if head is None:
        return torch.ones(n), trainable
    # the head sources must end before the late window, i.e. not overlap it
    assert depth >= last and all(j < first_late_source(mode, depth) for j in head), \
        f"{mode} reads the last {last} blocks, but depth is {depth}"
    w = torch.zeros(n)
    w[first_late_source(mode, depth):] = 1.0    # blocks depth-last .. depth-1
    w[list(head)] = 1.0                         # embedding / block 0 skip to the head
    return w, trainable


class ReadoutMixer(nn.Module):
    """weight[j] scales source j; forward(sources) accumulates in fp32 in source order."""

    def __init__(self, depth, mode):
        super().__init__()
        w, trainable = readout_init(mode, depth)
        self.mode = str(mode)
        self.depth = int(depth)
        self.n_sources = 2 * self.depth + 1
        self.trainable = bool(trainable)
        # first "late" source for summary(); learn_ones has no window and keeps the last-4 split
        self.first_late = (first_late_source(self.mode, self.depth) if self.mode != "learn_ones"
                           else 2 * (self.depth - min(4, self.depth)) + 1)
        if self.trainable:
            self.weight = nn.Parameter(w)
        else:
            self.register_buffer("weight", w)
        # gradient-sign counters: plain attributes, NOT buffers -- DDP would broadcast buffers on
        # every forward and checkpoints would carry them. Reset per epoch by the trainer.
        self._grad_pos = None
        self._grad_neg = None
        self._grad_steps = 0
        self._grad_nonfinite = 0

    def labels(self):
        return source_labels(self.depth)

    def forward(self, sources):
        assert len(sources) == self.n_sources, (len(sources), self.n_sources)
        fn = _mix_compiled if (_COMPILE_MIXER and sources[0].is_cuda) else _mix
        return fn(self.weight, list(sources))

    # ----- gradient sign consistency (one call per optimizer step, after unscale) -----
    @torch.no_grad()
    def record_grad_sign(self):
        if not self.trainable:
            return
        g = self.weight.grad
        if g is None:
            return
        if self._grad_pos is None:
            self._grad_pos = torch.zeros_like(g, dtype=torch.int64)
            self._grad_neg = torch.zeros_like(g, dtype=torch.int64)
        if not torch.isfinite(g).all():      # a GradScaler-skipped step; its sign means nothing
            self._grad_nonfinite += 1
            return
        self._grad_pos += (g > 0)
        self._grad_neg += (g < 0)
        self._grad_steps += 1

    def reset_grad_stats(self):
        self._grad_pos = None
        self._grad_neg = None
        self._grad_steps = 0
        self._grad_nonfinite = 0

    @torch.no_grad()
    def grad_stats(self):
        """pos_frac[j] = fraction of steps with a positive gradient on w_j; sign_consistency[j] =
        fraction of steps with the majority sign (0.5 = coin flip, 1.0 = every step agrees)."""
        n = self._grad_steps
        if n == 0 or self._grad_pos is None:        # fixed arm, or no step yet: null, not NaN
            return dict(n_steps=0, n_nonfinite=self._grad_nonfinite, pos_frac=None,
                        sign_consistency=None)
        pos = self._grad_pos.float() / n
        neg = self._grad_neg.float() / n
        return dict(n_steps=int(n), n_nonfinite=int(self._grad_nonfinite),
                    pos_frac=pos.tolist(), sign_consistency=torch.maximum(pos, neg).tolist())

    @torch.no_grad()
    def summary(self):
        """Scalars for log.txt / wandb; the full row goes to readout.jsonl."""
        w = self.weight.detach().float()
        early, late = w[:self.first_late], w[self.first_late:]
        late_mean = float(late.mean())
        early_mean = float(early.mean())
        # None (JSON null) where the ratio is undefined: json.dumps would write Infinity / NaN,
        # which is not JSON and which strict readers and wandb reject
        return {
            "mean": float(w.mean()),
            "min": float(w.min()),
            "max": float(w.max()),
            "emb": float(w[0]),
            "early_mean": early_mean,
            "late_mean": late_mean,
            "late_early_ratio": late_mean / early_mean if early_mean != 0 else None,
            "n_neg": int((w < 0).sum()),
            "n_zero": int((w == 0).sum()),
        }

    def extra_repr(self):
        return f"mode={self.mode}, n_sources={self.n_sources}, trainable={self.trainable}"
