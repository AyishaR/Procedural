"""Static residual routing: independently weighted residual inputs for the timm-style ViT.

Ported verbatim (everything down to _mix_buf) from the layer-scale repo,
arch/residual_routing.py; spec and design note in that repo under
analysis/notes/residual_routing_handoff.md and analysis/notes/residual_routing.md.

The residual stream is replaced by an explicit per-token history: history[0] is the tensor
entering the first block (patch_embed -> _pos_embed -> norm_pre, i.e. cls token, position
embedding and pos_drop applied) and history[k] (k >= 1) is the k-th branch's contribution --
the attn / mlp branch output INCLUDING ls1/ls2 (LayerScale) and drop_path1/drop_path2, i.e.
exactly what the vanilla Block would have ADDED to the stream. Every consumer (each branch
in execution order, and the final readout) reads its own weighted sum of every earlier entry,

    mixed_i = sum_{j < i} a[i, j] * history[j]        i = 1 .. M+1,   M = 2 * n_layer

and the mixes REPLACE the residual additions (nothing is added back). With every a[i, j] == 1
this is the vanilla pre-norm transformer term for term and in the same summation order, so a
routed model at init is bit-identical to the baseline built from the same seed (the router
draws no random numbers).

Indexing conventions (shared by models/vision_transformer.py, main.py's logging and the
analysis scripts):
  consumers  i = 1..M+1, 1-based, execution order: i <= M is branch i = layer (i-1)//2,
             attention when i is odd, MLP when i is even; i = M+1 is the readout
             (norm -> pool -> fc_norm -> head).
  sources    j = 0..M, 0-based: j = 0 is the embedding stream, j >= 1 the output of consumer j.
  edge (i, j) is valid iff j < i.  theta[i-1] has shape (i,); entry j is the logit of (i, j).
  coefficient_matrix() is (M+1, M+1), lower-triangular INCLUDING the diagonal: row r is
  consumer r+1, column c is source c. Column 0 = the embedding, the diagonal C[r, r] = the
  immediately preceding branch, the last row = the readout. The edge mask has the same layout.

Parameterisation (static mode): a = softplus(theta) with theta = log(expm1(1)), so every
coefficient starts at exactly 1.0 (bit-exact in fp32); optionally a = bound * sigmoid(theta)
for a finite bound > 1, initialised so that a == 1 up to fp32 rounding of the sigmoid (exact
for most bounds, 1 ulp off for some, e.g. 8.0 -- so only the softplus mode is guaranteed
bit-identical to the baseline at init). Coefficients are nonnegative and
never normalised. Exact zeros come from the explicit edge mask, which multiplies the activated
coefficient, so a masked theta stays in the autograd graph with an exactly-zero gradient (DDP
never sees an unused parameter).

Thetas are stored per consumer as 1-D parameters (nn.ParameterList): optim_factory.py's
get_parameter_groups puts any len(shape) == 1 parameter in the "no_decay" group at the same
LR, so the thetas land there with zero edits to the optimizer.

Two implementations of the same math (VisionTransformer routing_impl):
  "loop" (default) -- ordered scalar x tensor accumulation through autograd: bit-identical to the
      vanilla model at init (same summation order), but the per-edge autograd traffic is far off
      the bandwidth bound for deep models (+55% per micro-step at 124M, +250-340% at 350M).
  "buf" -- ResidualHistory keeps the branch outputs in one contiguous buffer; each mix is a fused
      weighted row-sum (Triton kernel: branch-dtype reads -- fp16 under this repo's AMP --
      with fp32 math and fp32 coefficients), the
      backward stores dL/dmixed_i and computes grad_theta by fused row-dots, and the gradient of
      every history entry is DEFERRED: consumer i's node is the last one (in backward order) to
      read h_{i-1}, so _Write_j's backward returns the complete sum_{i>j} a_ij dL/dmixed_i in one
      pass. Same parameters, checkpoints and results as "loop" up to fp32 summation order.
      Under this repo's fp16 autocast "buf" is also the better-conditioned path: the mix
      and its backward are accumulated in fp32 and cast once at the end, whereas "loop"
      performs a genuine fp16 multiply per edge (a[j] * history[j]).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

THETA_ONE = math.log(math.expm1(1.0))  # softplus(THETA_ONE) == 1.0, bit-exact in fp32


def consumer_labels(n_layer):
    """Row labels of coefficient_matrix(): 'attn_0', 'mlp_0', ..., 'mlp_{L-1}', 'readout'."""
    labels = []
    for i in range(1, 2 * n_layer + 1):
        labels.append(f"{'attn' if i % 2 == 1 else 'mlp'}_{(i - 1) // 2}")
    return labels + ["readout"]


def source_labels(n_layer):
    """Column labels of coefficient_matrix(): 'emb', 'attn_0', 'mlp_0', ..., 'mlp_{L-1}'."""
    return ["emb"] + consumer_labels(n_layer)[:-1]


class ResidualRouter(nn.Module):
    """One trainable nonnegative scalar per valid (consumer, source) edge; see the module doc."""

    def __init__(self, n_layer, bound=0.0):
        super().__init__()
        assert n_layer >= 1
        assert bound == 0.0 or bound > 1.0, f"routing bound must be 0 (softplus) or > 1: {bound}"
        self.n_layer = int(n_layer)
        self.M = 2 * self.n_layer
        self.n_consumers = self.M + 1
        self.bound = float(bound)
        theta0 = THETA_ONE if self.bound == 0.0 else math.log(1.0 / (self.bound - 1.0))
        # consumer i (1-based) owns theta[i-1] of shape (i,): one logit per earlier source
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.full((i,), theta0)) for i in range(1, self.n_consumers + 1)]
        )
        n = self.n_consumers
        self.register_buffer("mask", torch.tril(torch.ones(n, n)))

    # -- parameterisation -------------------------------------------------------------------
    def _activate(self, theta):
        if self.bound == 0.0:
            return F.softplus(theta)
        return self.bound * torch.sigmoid(theta)

    def theta_from_coefficient(self, a):
        """Inverse of _activate (elementwise); a must be > 0 (and < bound when bounded).
        Zeros are expressed with the edge mask, not with theta."""
        a = torch.as_tensor(a, dtype=torch.float32)
        assert bool((a > 0).all()), "coefficients must be > 0 (use set_edge_mask for zeros)"
        if self.bound == 0.0:
            return a + torch.log(-torch.expm1(-a))  # stable inverse softplus: log(exp(a) - 1)
        assert bool((a < self.bound).all()), f"coefficients must be < bound {self.bound}"
        return torch.log(a / (self.bound - a))

    def coefficients(self, i):
        """(i,) coefficients of consumer i (1-based) over sources 0..i-1, mask applied."""
        return self._activate(self.theta[i - 1]) * self.mask[i - 1, :i]

    # -- forward ----------------------------------------------------------------------------
    def forward(self, i, history, state=None):
        """Mixed input of consumer i: sum_j a[i, j] * history[j] over the i entries so far.

        state None ("loop"): ordered scalar-times-tensor accumulation, starting from history[0]:
        under autocast history[0] is fp32 (embeddings), so the accumulator stays fp32 like the
        vanilla residual stream, while each 0-d fp32 coefficient times a bf16 branch output stays
        bf16 (no fp32 temporaries). Deliberately NOT torch.stack + einsum: that would retain an
        O(M^2 * B*T*d) stacked copy for backward; this loop retains nothing beyond history.
        state a ResidualHistory ("buf"): the fused-buffer path (module docstring); history then
        holds the _Write tokens and only carries the autograd dependency edges.
        """
        assert len(history) == i, f"consumer {i} expects {i} history entries, got {len(history)}"
        a = self.coefficients(i)
        if state is not None:
            return _mix_buf(a, state, i, history)
        mixed = a[0] * history[0]
        for j in range(1, i):
            mixed = mixed + a[j] * history[j]
        return mixed

    # -- inspection / ablation --------------------------------------------------------------
    @torch.no_grad()
    def coefficient_matrix(self):
        """(M+1, M+1) fp32 lower-triangular (incl. diagonal) matrix, mask applied, detached."""
        n = self.n_consumers
        C = torch.zeros(n, n, dtype=torch.float32, device=self.mask.device)
        for i in range(1, n + 1):
            C[i - 1, :i] = self.coefficients(i).float()
        return C

    @torch.no_grad()
    def set_coefficient_matrix(self, C):
        """Set every theta so that coefficient_matrix() reproduces the valid entries of C
        (positive values only; the mask is left untouched -- masked edges stay zero)."""
        C = torch.as_tensor(C, dtype=torch.float32)
        n = self.n_consumers
        assert C.shape == (n, n), f"expected ({n}, {n}), got {tuple(C.shape)}"
        for i in range(1, n + 1):
            self.theta[i - 1].copy_(self.theta_from_coefficient(C[i - 1, :i]))

    @torch.no_grad()
    def set_edge_mask(self, i, j, value):
        """Multiplicative mask on edge (consumer i, source j); 0.0 = exact-zero ablation."""
        assert 1 <= i <= self.n_consumers and 0 <= j < i, f"invalid edge ({i}, {j})"
        self.mask[i - 1, j] = float(value)

    @torch.no_grad()
    def summary(self):
        """Scalar statistics over the valid, unmasked edges (for wandb): mean / min / max of the
        coefficients, the embedding column, the immediate-predecessor diagonal, the readout row,
        and the number of edges counted."""
        C = self.coefficient_matrix()
        n = self.n_consumers
        valid = torch.tril(torch.ones(n, n, dtype=torch.bool, device=C.device)) & (self.mask > 0)
        vals = C[valid]
        col0, diag, last = valid[:, 0], valid.diagonal(), valid[-1]
        return {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "emb_col_mean": float(C[:, 0][col0].mean()),
            "prev_diag_mean": float(C.diagonal()[diag].mean()),
            "readout_row_mean": float(C[-1][last].mean()),
            "n_edges": int(valid.sum()),
        }

    def extra_repr(self):
        kind = "softplus" if self.bound == 0.0 else f"{self.bound:g}*sigmoid"
        return f"n_layer={self.n_layer}, consumers={self.n_consumers}, {kind}"


# ------------------------------------------------------------------------------------------
# "buf" implementation: contiguous history buffer, fused row-sums, deferred per-source grads
# ------------------------------------------------------------------------------------------
def _wsum_eager(X, a):
    """(rows, N) x (rows,) -> (N,) fp32: sum_r a[r] * X[r] with fp32 math (CPU / reference)."""
    return (X.float() * a.float()[:, None]).sum(0)


def _rowdots_eager(X, g):
    """(rows, N), (N,) -> (rows,) fp32: per-row dot products with fp32 math (CPU / reference)."""
    return (X.float() * g.float()[None, :]).sum(1)


try:  # CUDA: hand-written Triton kernels (one read of the rows, fp32 math, no dynamo involved)
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _wsum_kernel(x_ptr, a_ptr, out_ptr, R, N, stride_r, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < N
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for r in range(R):
            c = tl.load(a_ptr + r)
            x = tl.load(x_ptr + r.to(tl.int64) * stride_r + offs, mask=mask, other=0.0)
            acc += c * x.to(tl.float32)
        tl.store(out_ptr + offs, acc, mask=mask)

    @triton.jit
    def _rowdots_kernel(x_ptr, g_ptr, out_ptr, N, stride_r, n_chunks, BLOCK: tl.constexpr):
        r = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1).to(tl.int64)
        offs = c * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < N
        x = tl.load(x_ptr + r * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + r * n_chunks + c, tl.sum(x * g, axis=0))


_WSUM_BLOCK = 1024
_ROWDOTS_BLOCK = 4096


def _wsum(X, a):
    """sum_r a[r] * X[r] over the rows of a (rows, N) view with unit inner stride -> (N,) fp32."""
    if not (X.is_cuda and _HAS_TRITON):
        return _wsum_eager(X, a)
    R, N = X.shape
    assert X.stride(1) == 1
    out = torch.empty(N, dtype=torch.float32, device=X.device)
    if R == 0:
        return out.zero_()
    a = a.detach().float().contiguous()
    grid = (triton.cdiv(N, _WSUM_BLOCK),)
    _wsum_kernel[grid](X, a, out, R, N, X.stride(0), BLOCK=_WSUM_BLOCK)
    return out


def _rowdots(X, g):
    """Per-row dot products of a (rows, N) view with g (N,) -> (rows,) fp32 (deterministic:
    fixed-size partial sums, then a torch reduction)."""
    if not (X.is_cuda and _HAS_TRITON):
        return _rowdots_eager(X, g)
    R, N = X.shape
    assert X.stride(1) == 1
    if R == 0:
        return torch.empty(0, dtype=torch.float32, device=X.device)
    g = g.detach().float().contiguous()
    n_chunks = triton.cdiv(N, _ROWDOTS_BLOCK)
    partial = torch.empty(R, n_chunks, dtype=torch.float32, device=X.device)
    _rowdots_kernel[(R, n_chunks)](X, g, partial, N, X.stride(0), n_chunks, BLOCK=_ROWDOTS_BLOCK)
    return partial.sum(1)


class ResidualHistory:
    """Per-forward scratch of the "buf" path (never part of the module / checkpoint).

    h0    the fp32 stream entering block 0 (kept by reference, no copy)
    buf   (n, numel) rows 1..n-1 = branch outputs in their own dtype (row 0 unused)
    G     (n, numel) fp32, row n-i = dL/dmixed_i, written during backward (REVERSED so that
          the rows a source j needs, consumers i > j, form the prefix G[:n-j]: every helper
          call then sees storage offset 0 / a fixed offset and dynamo compiles it once)
    A     (n, n) fp32 copy of the coefficient rows used in this forward
    """

    def __init__(self, n, x0):
        self.n = n
        self.shape = tuple(x0.shape)
        self.numel = x0.numel()
        self.h0 = x0.detach()
        self.buf = None
        self.G = torch.zeros(n, self.numel, dtype=torch.float32, device=x0.device)
        self.A = torch.zeros(n, n, dtype=torch.float32, device=x0.device)
        self.done = [False] * (n + 1)  # done[i]: consumer i stored dL/dmixed_i in this backward
        self.h0_needs_grad = False  # set by _Write_0: does the embedding stream get a gradient?

    def check_backward(self):
        """One complete backward per forward: no double backward (the detached scratch cannot
        carry higher-order dependencies) and no second / partial backward after the scratch was
        released (routing_impl="loop" supports all of those)."""
        if torch.is_grad_enabled():
            self.fail(
                "routing_impl='buf' does not support double backward (create_graph=True); "
                "use routing_impl='loop'"
            )
        if self.G is None:
            raise RuntimeError(
                "routing_impl='buf' supports exactly one backward per forward (its scratch is "
                "released at the end of the pass); use routing_impl='loop' for retain_graph"
            )

    def release(self):
        """Drop the scratch as soon as the last node of the backward has used it, so the memory
        does not live on with the autograd graph (train.py keeps the previous step's outputs
        referenced while the next forward runs). The last node is _Write_0 (h0 is read by every
        consumer) when the embedding stream needs a gradient, else _Mix_1 (the last consumer in
        backward order). A backward that never reaches either (e.g. autograd.grad w.r.t. the
        thetas only) keeps the scratch alive with the graph."""
        self.G = self.buf = self.A = self.h0 = None

    def fail(self, msg):
        self.release()
        raise RuntimeError(msg)

    def write(self, j, z):
        if self.buf is None:
            self.buf = torch.empty(self.n, self.numel, dtype=z.dtype, device=z.device)
        self.buf[j].copy_(z.reshape(-1))


class _Write(torch.autograd.Function):
    """Registers history entry j (identity in forward). Its backward returns the COMPLETE
    gradient of h_j, deferred until every consumer i > j has stored G[i-1] -- autograd runs all
    consumers of the token before this node."""

    @staticmethod
    def forward(ctx, z, state, j):
        if j == 0:
            state.h0 = z.detach()
            state.h0_needs_grad = bool(z.requires_grad)
        else:
            state.write(j, z)
        ctx.state, ctx.j, ctx.dtype = state, j, z.dtype
        ctx.set_materialize_grads(False)
        return z.view_as(z)

    @staticmethod
    def backward(ctx, gz):
        st, j = ctx.state, ctx.j
        st.check_backward()
        missing = [i for i in range(j + 1, st.n + 1) if not st.done[i]]
        if missing:  # a backward rooted at an intermediate would read stale / unwritten rows
            st.fail(
                f"routing_impl='buf': backward reached source {j} before its consumers "
                f"{missing} ran -- partial backward is unsupported, use routing_impl='loop'"
            )
        k = st.n - j  # consumers j+1..n live in the prefix G[:k], consumer i at row n-i
        coef = st.A[j:, j].flip(0)  # prefix row r is consumer n-r -> coefficient A[n-r-1, j]
        g = _wsum(st.G[:k], coef).view(st.shape).to(ctx.dtype)  # sum_{i>j} a_ij dL/dmixed_i
        if gz is not None:
            g = g + gz
        if j == 0:  # h0 is read by every consumer: this is the last node of the pass
            st.release()
        return g, None, None


class _Mix(torch.autograd.Function):
    """mixed_i = a[0] h0 + sum_{1<=j<i} a[j] buf[j]. Backward stores G[i-1] = dL/dmixed_i and
    returns grad_a (row dots); the token inputs only carry dependency edges (their gradients
    come from _Write)."""

    @staticmethod
    def forward(ctx, a, state, i, *tokens):
        state.A[i - 1, :i] = a.detach()
        out = a[0] * state.h0
        if i > 1:
            out = out + _wsum(state.buf[1:i], a[1:i]).view(state.shape)
        ctx.state, ctx.i = state, i
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, g):
        st, i = ctx.state, ctx.i
        st.check_backward()
        if st.done[i]:  # a second backward pass over a graph whose first pass never released
            st.fail(
                f"routing_impl='buf': consumer {i} ran its backward twice -- exactly one "
                "backward per forward is supported, use routing_impl='loop'"
            )
        gflat = g.reshape(-1).float()
        st.G[st.n - i].copy_(gflat)
        st.done[i] = True
        grad_a = torch.empty(i, dtype=torch.float32, device=g.device)
        grad_a[0] = torch.dot(gflat, st.h0.reshape(-1).float())
        if i > 1:
            grad_a[1:] = _rowdots(st.buf[1:i], gflat)
        if i == 1 and not st.h0_needs_grad:  # no _Write_0 node will run: last user of the scratch
            st.release()
        return (grad_a, None, None) + (None,) * i


def _mix_buf(a, state, i, history):
    return _Mix.apply(a, state, i, *history)


# --------------------------------------------------------------------------------------------
# Model-specific: the timm-style VisionTransformer of models/vision_transformer.py.
# Everything above this line is architecture-agnostic and copied verbatim from the layer-scale
# repo; only these last ~60 lines were rewritten for the ViT (the GPT versions read
# model.transformer.h and blk.ln_1 / blk.ln_2).
# --------------------------------------------------------------------------------------------
def _attn_branch(norm1, attn, ls1, drop_path1, x):
    return drop_path1(ls1(attn(norm1(x))))


def _mlp_branch(norm2, mlp, ls2, drop_path2, x):
    return drop_path2(ls2(mlp(norm2(x))))


# ONE compiled graph per branch kind, shared by every block: the modules are arguments, and
# dynamo guards on their structure, not their identity. This repo never calls torch.compile
# (the ViT trains eagerly), so the compiled variants stay switched off; flip _COMPILE_BRANCHES
# if that ever changes. torch.compile is lazy, so defining them costs nothing until called.
_COMPILE_BRANCHES = False
_attn_branch_c = torch.compile(_attn_branch)
_mlp_branch_c = torch.compile(_mlp_branch)


def _branch_modules(blk):
    """Every module the two branches run through, in execution order."""
    return (blk.norm1, blk.attn, blk.ls1, blk.drop_path1,
            blk.norm2, blk.mlp, blk.ls2, blk.drop_path2)


def has_hooks(model):
    """True if any module inside a branch carries forward (pre-)hooks -- the measurement tools
    do, training never does.

    The walk is RECURSIVE, unlike the nanoGPT version: utils.HookCollector registers on
    block.mlp.act and block.mlp.fc2, i.e. DESCENDANTS of a branch module rather than the branch
    module itself. The hook it also puts on the Block is irrelevant here -- Block.forward never
    runs under routing, which is exactly why utils.RoutedHookCollector exists.
    """
    return any(
        m._forward_hooks or m._forward_pre_hooks
        for blk in model.blocks
        for top in _branch_modules(blk)
        for m in top.modules()
    )


def _branches(model, compile_blocks):
    """[(attn_fn, mlp_fn)] per block: x -> drop_path1(ls1(attn(norm1(x)))) and the mlp twin.

    LayerScale and DropPath stay INSIDE the branch, so history[k] is exactly what the vanilla
    Block would have added to the stream. Block.forward is deliberately not used. Runs eagerly
    on CPU (tests), whenever forward hooks are registered (dynamo traces hooks into the graph
    and may capture their closures), and -- for now -- always, since this repo never compiles.
    """
    compile_blocks = compile_blocks and _COMPILE_BRANCHES and not has_hooks(model)
    fa, fm = (_attn_branch_c, _mlp_branch_c) if compile_blocks else (_attn_branch, _mlp_branch)
    return [
        (
            (lambda x, b=blk: fa(b.norm1, b.attn, b.ls1, b.drop_path1, x)),
            (lambda x, b=blk: fm(b.norm2, b.mlp, b.ls2, b.drop_path2, x)),
        )
        for blk in model.blocks
    ]


# The whole routed traversal is hidden from dynamo (buffer side effects in the Functions, a
# growing history list, per-consumer shapes): a torch.compile'd forward would graph-break once
# here and resume at model.norm.
@torch.compiler.disable
def routed_forward_buf(model, x):
    """VisionTransformer._forward_routed for routing_impl == "buf": the same traversal as the
    loop path.

    ``x`` is the tensor entering block 0 (patch_embed -> _pos_embed -> norm_pre); the return
    value is the readout consumer's mix, BEFORE model.norm.
    """
    state = ResidualHistory(model.router.n_consumers, x)
    history = [_Write.apply(x, state, 0)]
    i = 1
    for attn, mlp in _branches(model, compile_blocks=x.is_cuda):
        history.append(_Write.apply(attn(model.router(i, history, state)), state, i))
        i += 1
        history.append(_Write.apply(mlp(model.router(i, history, state)), state, i))
        i += 1
    return model.router(i, history, state)  # readout (consumer M+1) over all M+1 entries
