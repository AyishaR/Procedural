"""Model-side primitives shared by the dynamics stages.

fp32 residual-stream states with the complete token sequence, the CLS-only readout through the checkpoint's own
final norm + head (= the raw logit lens), per-branch residual gains (bypass / attenuation / gain gradients),
per-image metrics, exact chunked full-batch gradients and grouped gradient norms.  Depth indexing follows
probes: depth 0 = patch+pos embedding, depth d = output of block d-1; the after-attention state of block L is
the historical `layer_sub = L - 0.5` row.
"""
import contextlib
import math

import torch
import torch.nn.functional as F

SMOOTHING = 0.1          # the runs' label smoothing (timm Mixup: q_y = 1 - s + s/C, others s/C)


def _f(t):
    """fp32 for the metric arithmetic; float64 inputs stay float64 (tests, reference computations)."""
    return t if t.dtype == torch.float64 else t.float()


@contextlib.contextmanager
def precision(mode="fp32"):
    """'fp32' (matmul precision 'highest', TF32 off, autocast off), 'tf32' (fp32 tensors, TF32 matmuls/convs,
    autocast off) or 'bf16' (autocast) for the enclosed forward.  Uses torch.set_float32_matmul_precision (never
    the legacy allow_tf32 flag, which PyTorch refuses to mix with it) and restores the previous state."""
    assert mode in ("fp32", "tf32", "bf16"), mode
    old_mm, old_cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    torch.backends.cudnn.allow_tf32 = mode == "tf32"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if mode == "bf16":
            with torch.autocast(dev, dtype=torch.bfloat16):
                yield
        else:
            with torch.autocast(dev, enabled=False):
                yield
    finally:
        torch.set_float32_matmul_precision(old_mm)
        torch.backends.cudnn.allow_tf32 = old_cudnn


# ------------------------------------------------------------------------------------------------- forward pieces
def embed(m, img):
    """Depth-0 state: patch embedding + cls token + position embedding (+ norm_pre, Identity for vit_base)."""
    return m.norm_pre(m._pos_embed(m.patch_embed(img)))


def attn_branch(blk, x):
    return blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))


def mlp_branch(blk, x):
    return blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))


def block_forward(blk, x, g_attn=1.0, g_mlp=1.0):
    """The block's own forward with a gain on each residual branch (python float or tensor broadcastable to x).

    g = 1 reproduces `blk(x)` bit for bit (multiplication by 1.0 is exact), g = 0 bypasses the branch, and a
    tensor gain with requires_grad gives dL/dg, the exact local sensitivity of the loss to the branch."""
    x = x + g_attn * attn_branch(blk, x)
    return x + g_mlp * mlp_branch(blk, x)


def block_update(blk, x):
    """(F_attn, F_mlp, F): the attention branch output on x, the MLP branch output on x + F_attn, and their sum
    F = F_attn + F_mlp, i.e. the block's residual update blk(x) - x computed without the cancelling x."""
    a = attn_branch(blk, x)
    mm = mlp_branch(blk, x + a)
    return a, mm, a + mm


@torch.no_grad()
def states(m, img, mid=False):
    """[h_0 (embedding), h_1, ..., h_L] full token sequences ([B, T, D]); with mid=True also the L after-attention
    states.  Each h_{l+1} equals m.blocks[l](h_l) bit for bit."""
    x = embed(m, img)
    hs, mids = [x], []
    for blk in m.blocks:
        a = x + attn_branch(blk, x)
        x = a + mlp_branch(blk, a)
        mids.append(a)
        hs.append(x)
    return (hs, mids) if mid else hs


def suffix(m, h, from_block):
    """Run blocks[from_block:] on the state entering block `from_block` -> final residual stream."""
    for l in range(from_block, len(m.blocks)):
        h = m.blocks[l](h)
    return h


def head_on(m, h):
    """CLS-only readout through the checkpoint's own final norm and head -> logits [B, C] (raw logit lens)."""
    z = h[:, 0] if h.dim() == 3 else h
    return m.head(m.fc_norm(m.norm(z)))


def forward_with_gains(m, img, g):
    """Full forward with per-sample branch gains g [B, L, 2] (attention, MLP) -> final residual stream."""
    x = embed(m, img)
    for l, blk in enumerate(m.blocks):
        x = block_forward(blk, x, g[:, l, 0][:, None, None], g[:, l, 1][:, None, None])
    return x


# ------------------------------------------------------------------------------------------------- k-bias
def k_bias_norms(m):
    """{block index: L2 norm of the (softmax-invariant) key bias} of a model in memory."""
    out = {}
    for l, blk in enumerate(m.blocks):
        b = blk.attn.qkv.bias
        D = b.shape[0] // 3
        out[l] = float(b[D:2 * D].detach().float().norm())
    return out


# ------------------------------------------------------------------------------------------------- metrics
def per_image(logits, y, smoothing=SMOOTHING):
    """Per-image readout metrics from logits [B, C] (any dtype) and labels [B] -> dict of [B] tensors:
    pred, correct, top5, ce (hard label), ce_smooth (label-smoothed target), margin (logit_y - best other),
    maxp (confidence), p_y."""
    logits = _f(logits)
    logp = F.log_softmax(logits, -1)
    lp_y = logp.gather(1, y[:, None])[:, 0]
    ce = -lp_y
    ce_s = (1 - smoothing) * ce + smoothing * (-logp.mean(-1))
    top = logits.topk(min(5, logits.shape[1]), -1).indices
    pred = logits.argmax(-1)                              # argmax (first maximum), the pipeline's tie convention
    other = logits.scatter(1, y[:, None], float("-inf")).max(-1).values
    margin = logits.gather(1, y[:, None])[:, 0] - other
    return dict(pred=pred, correct=pred == y, top5=(top == y[:, None]).any(-1), ce=ce, ce_smooth=ce_s,
                margin=margin, maxp=logp.max(-1).values.exp(), p_y=lp_y.exp())


def loss_fn(logits, y, objective="smooth", smoothing=SMOOTHING):
    """Mean loss over the batch: 'smooth' = CE against the label-smoothed target (the training criterion without
    mixup/cutmix), 'hard' = plain CE."""
    logp = F.log_softmax(_f(logits), -1)
    ce = -logp.gather(1, y[:, None])[:, 0]
    if objective == "hard":
        return ce.mean()
    assert objective == "smooth", objective
    return ((1 - smoothing) * ce + smoothing * (-logp.mean(-1))).mean()


# ------------------------------------------------------------------------------------------------- gradients
def accumulate_grads(m, x, y, objective="smooth", chunk=256, forward=None, on_logits=None):
    """Exact gradient of the mean loss over all N examples, accumulated over chunks (each chunk's mean loss is
    scaled by n_chunk / N).  Leaves .grad on the parameters; returns (loss, [per-chunk gradient norms]), the
    norms being those of each chunk's own mean-loss gradient (computed separately, then added to the total).
    `on_logits(logits, yb)` is called with every chunk's detached logits (per-image statistics)."""
    forward = forward or (lambda xb: m(xb))
    N = x.shape[0]
    acc = {n: None for n, p in m.named_parameters() if p.requires_grad}
    total, chunk_norms = 0.0, []
    for s in range(0, N, chunk):
        m.zero_grad(set_to_none=True)
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        logits = forward(xb)
        if on_logits is not None:
            on_logits(logits.detach(), yb)
        loss = loss_fn(logits, yb, objective)
        loss.backward()
        w = xb.shape[0] / N
        total += float(loss.detach()) * w
        sq = 0.0
        for n, p in m.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            sq += float(g.double().pow(2).sum())
            acc[n] = g * w if acc[n] is None else acc[n] + g * w
        chunk_norms.append(math.sqrt(sq))
    m.zero_grad(set_to_none=True)
    for n, p in m.named_parameters():
        if acc.get(n) is not None:
            p.grad = acc[n]
    return total, chunk_norms


def grad_norms(m):
    """Grouped L2 norms of the current .grad: total, per block (blocks.l), q/k/v split of every qkv tensor,
    per tensor, and 'other' (embedding, cls/pos, final norm, head)."""
    per_tensor, sq_block, sq_other, qkv = {}, {}, 0.0, {}
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().double()
        per_tensor[n] = float(g.norm())
        if n.startswith("blocks."):
            l = int(n.split(".")[1])
            sq_block[l] = sq_block.get(l, 0.0) + float(g.pow(2).sum())
            if ".attn.qkv." in n:
                for part, gg in zip("qkv", g.chunk(3, 0)):
                    qkv[n.replace("qkv", part)] = float(gg.norm())
        else:
            sq_other += float(g.pow(2).sum())
    per_block = {l: math.sqrt(v) for l, v in sorted(sq_block.items())}
    total = math.sqrt(sum(sq_block.values()) + sq_other)
    return dict(total=total, per_block=per_block, other=math.sqrt(sq_other), per_tensor=per_tensor, qkv_split=qkv)


def param_norms(m, exclude_k_bias=True):
    """Per-block parameter L2 norm (the k-bias rows of qkv.bias excluded by default) and the 'other' group."""
    sq_block, sq_other = {}, 0.0
    for n, p in m.named_parameters():
        v = p.detach().double()
        if exclude_k_bias and n.endswith("attn.qkv.bias"):
            D = v.shape[0] // 3
            v = torch.cat([v[:D], v[2 * D:]])
        if n.startswith("blocks."):
            l = int(n.split(".")[1])
            sq_block[l] = sq_block.get(l, 0.0) + float(v.pow(2).sum())
        else:
            sq_other += float(v.pow(2).sum())
    return dict(per_block={l: math.sqrt(v) for l, v in sorted(sq_block.items())}, other=math.sqrt(sq_other))


def hist_style_grad_norms(m):
    """The training loop's own grouping (utils.NativeScalerWithGradNormCount.get_layer_grad_norms with the DDP
    'module.' prefix and q/k/v splits) so keys match grad_norms_IMNET_BASE_*.json, plus a per-block level.

    Two quirks of the training code, reproduced for parity: (1) the expanded q/k/v slices are added to the
    buckets *next to* the whole qkv tensor, so the 'total' counts every qkv gradient twice
    (total^2 = sum(others^2) + 2 |g_qkv|^2); (2) its level-N lambdas close over the loop variable, so only the
    last level is ever applied and no `module.blocks.N` bucket exists in the historical files -- the per-block
    dict returned here (same double-counting convention) is our own aggregation.  Use `grad_norms` for plain norms."""
    import sys
    from probes import registry as preg
    sys.path.insert(0, str(preg.ROOT))
    import utils
    named = [("module." + n, p) for n, p in m.named_parameters()]
    total, groups = utils.NativeScalerWithGradNormCount.get_layer_grad_norms(named, norm_type=2.0, group_levels=[None, 6])
    per_block = {}
    for n, p in m.named_parameters():
        if p.grad is None or not n.startswith("blocks."):
            continue
        l = int(n.split(".")[1])
        g2 = float(p.grad.detach().float().pow(2).sum())
        per_block[l] = per_block.get(l, 0.0) + (2 * g2 if ".attn.qkv." in n else g2)
    return float(total), {k: float(v) for k, v in groups.items()}, {l: math.sqrt(v) for l, v in sorted(per_block.items())}
