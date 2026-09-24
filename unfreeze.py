"""Progressive layer unfreezing for the timm-style ViT (main.py --unfreeze_arm <tag>).

Port of layer-scale's nanoGPT/unfreeze.py (docs/unfreeze_vit_handoff.md). The pure schedule
functions and UnfreezePlan's hooks are copied verbatim; only param_chunk (which tensor belongs to
which block), the decay rule (optim_factory.get_parameter_groups') and the tag's default ramp are
ViT / recipe specific.

The `depth` transformer blocks are split into `chunks` contiguous chunks of depth // chunks blocks.
Chunk 0 (in unfreeze order: the FRONT chunk for direction 'front', the LAST chunk for 'back')
trains from step 0; chunk k is frozen until optimizer step u_k and unfrozen from then on.
Everything that is not a block (patch_embed, cls_token / reg_token, pos_embed, norm_pre, norm /
fc_norm, head) is never chunked: it trains from step 0 in both directions.

    lin:  u_k = round(k / (C - 1) * E * T)            (evenly spaced, the last at E*T)
    geo:  u_0 = 0, u_k = round(E * T / 2**(C-1-k))     (each stage waits twice as long)
    one:  u_k = 0 for k < C-1, u_{C-1} = round(E * T)  (ONLY the last chunk in unfreeze order is
          frozen - 'front' freezes the back chunk, 'back' the front chunk; C = 2 equals lin)

T = the run's total optimizer steps (len(lr_schedule_values) = epochs x steps per epoch),
E = the fraction of the run at which the LAST chunk unfreezes. unfreeze_iters() is the single
source of truth for u_k (Python's banker's round; every consumer calls it, none re-derives).
Everything is a pure function of the global step `it`, so a resumed run continues the schedule
without extra checkpoint state.

Mechanism ("cold moments"): every parameter keeps requires_grad=True and sits in the optimizer
(DDP's reducer registers its hooks once, so the parameter set must stay static); after the
backward of the last micro-step (DDP's all-reduce has written the reduced grads back) and BEFORE
GradScaler.unscale_ / the clip / optimizer.step, mask_grads() sets p.grad = None for every
parameter of a still-frozen chunk (NativeScalerWithGradNormCount's pre_unscale hook). AdamW
collects only params whose grad is not None, and creates their state lazily in that same branch,
so a frozen parameter receives no update, no decoupled weight decay and no Adam state: it stays
bit-identical to the init, and a newly unfrozen chunk starts at Adam step 1. The scaler's inf
check, the grad-norm logging and any clip see the trainable grads only. As belt and braces the
optimizer holds one param-group pair (decay / no-decay) per chunk plus one for the
always-trainable tensors, and apply_lr() sets a frozen chunk's lr to 0. ("Warm moments" - lr = 0
without masking - is a different intervention and is not implemented.)

Per-chunk ramp: a chunk with u_k > 0 gets lr * min(1, (it - u_k + 1) / (W + 1)) for it >= u_k
(the shape of a linear warmup); chunks alive from step 0 get the recipe's own warmup, so E = 0 is
identical to the vanilla run for any W. W defaults to one epoch's worth of steps.

Arm tags: u<C><f|b>-<lin|geo|one><E>[-w<W>], e.g. u4f-lin0.2 (W = one epoch) or u4b-one0.2-w100.
"""

import re

DIRECTIONS = ("front", "back")
SCHEDS = ("lin", "geo", "one")
ARM_RE = re.compile(r"^u(\d+)([fb])-(lin|geo|one)(\d+(?:\.\d+)?)(?:-w(\d+))?$")
_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_ALWAYS = ("patch_embed.", "cls_token", "reg_token", "pos_embed", "norm_pre.", "norm.",
           "fc_norm.", "head.")
_PREFIXES = ("module.", "_orig_mod.")  # the DDP and torch.compile wrappers


# --------------------------------------------------------------------------------------
# pure schedule functions
# --------------------------------------------------------------------------------------
def partition(n_layer, chunks, direction):
    """Block indices per chunk IN UNFREEZE ORDER: [[0..L/C-1], ...] for 'front', reversed for
    'back' (the last L/C blocks first)."""
    assert direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}: {direction!r}"
    assert chunks >= 2, f"need >= 2 chunks (1 = vanilla): {chunks}"
    assert n_layer % chunks == 0, f"{chunks} chunks do not divide {n_layer} blocks"
    per = n_layer // chunks
    parts = [list(range(k * per, (k + 1) * per)) for k in range(chunks)]
    return parts[::-1] if direction == "back" else parts


def unfreeze_iters(sched, chunks, end, max_iters):
    """(u_0 = 0, u_1, ..., u_{C-1}): the step at which chunk k (unfreeze order) unfreezes."""
    assert sched in SCHEDS, f"sched must be one of {SCHEDS}: {sched!r}"
    assert chunks >= 2, chunks
    assert 0.0 <= end <= 1.0, f"unfreeze end must be in [0, 1]: {end}"
    assert max_iters > 0, max_iters
    last = end * max_iters
    if sched == "lin":
        us = [round(k / (chunks - 1) * last) for k in range(chunks)]
    elif sched == "geo":
        us = [0] + [round(last / 2 ** (chunks - 1 - k)) for k in range(1, chunks)]
    else:  # one: only the last chunk (in unfreeze order) is ever frozen
        us = [0] * (chunks - 1) + [round(last)]
    assert us[0] == 0 and all(us[i] <= us[i + 1] for i in range(len(us) - 1)), us
    assert us[-1] <= max_iters, us
    return tuple(int(u) for u in us)


def lr_mult(it, u, warmup):
    """Multiplier on the global lr for a chunk that unfreezes at u: 0 while frozen, a linear ramp
    over `warmup` steps after an unfreeze at u > 0 (the global warmup's shape), else 1."""
    if it < u:
        return 0.0
    if u == 0 or warmup <= 0:
        return 1.0
    return min(1.0, (it - u + 1) / (warmup + 1))


def strip_prefix(name):
    """Drop any stack of DDP / torch.compile wrapper prefixes."""
    changed = True
    while changed:
        changed = False
        for pre in _PREFIXES:
            if name.startswith(pre):
                name, changed = name[len(pre):], True
    return name


def param_chunk(name):
    """Block index of a block parameter, None for the always-trainable tensors (patch_embed,
    cls/reg token, pos_embed, norm_pre, norm / fc_norm, head); KeyError for anything else
    (router thetas, a readout mixer, a distillation head, an EMA copy, ...), so an unexpected
    parameter is refused rather than silently frozen or silently trained."""
    n = strip_prefix(name)
    if n.startswith(_ALWAYS):
        return None
    m = _BLOCK.match(n)
    if m is None:
        raise KeyError(f"no unfreeze chunk for parameter {name!r}")
    return int(m.group(1))


# --------------------------------------------------------------------------------------
# arm tags
# --------------------------------------------------------------------------------------
def parse_arm(tag):
    """'u4f-lin0.2' -> dict(chunks=4, direction='front', sched='lin', end=0.2, warmup=None);
    warmup None = one epoch's worth of steps, resolved when the plan is built."""
    m = ARM_RE.match(tag)
    if m is None:
        raise ValueError(f"not an unfreeze arm tag (u<C><f|b>-<lin|geo|one><E>[-w<W>]): {tag!r}")
    spec = dict(
        chunks=int(m.group(1)),
        direction="front" if m.group(2) == "f" else "back",
        sched=m.group(3),
        end=float(m.group(4)),
        warmup=int(m.group(5)) if m.group(5) is not None else None,
    )
    if not (spec["chunks"] >= 2 and 0.0 <= spec["end"] <= 1.0):
        raise ValueError(f"unfreeze arm needs >= 2 chunks and E in [0, 1]: {tag!r}")
    return spec


def arm_tag(spec):
    """Canonical tag of a spec (inverse of parse_arm up to the E spelling: 0.20 -> 0.2)."""
    tag = f"u{spec['chunks']}{spec['direction'][0]}-{spec['sched']}{spec['end']:g}"
    if spec["warmup"] is not None:
        tag += f"-w{spec['warmup']}"
    return tag


def train_flags(spec):
    """The main.py flags of a spec."""
    return ["--unfreeze_arm", arm_tag(spec)]


# --------------------------------------------------------------------------------------
# the plan bound to a model's parameters
# --------------------------------------------------------------------------------------
class UnfreezePlan:
    """The chunk partition + schedule bound to a model's parameters (build it from
    model_without_ddp: the parameter objects are shared with the DDP wrapper, the names clean)."""

    def __init__(self, named_params, n_layer, chunks, direction, sched, end, warmup, max_iters):
        self.n_layer, self.chunks, self.direction = n_layer, chunks, direction
        self.sched, self.end, self.warmup, self.max_iters = sched, end, warmup, max_iters
        self.layers = partition(n_layer, chunks, direction)
        self.iters = unfreeze_iters(sched, chunks, end, max_iters)
        layer_to_chunk = {layer: k for k, ls in enumerate(self.layers) for layer in ls}
        self.chunk_params = [[] for _ in range(chunks)]
        self.always_params = []
        self.chunk_names = [[] for _ in range(chunks)]
        self.always_names = []
        for name, p in named_params:
            layer = param_chunk(name)
            if layer is None:
                self.always_params.append(p)
                self.always_names.append(strip_prefix(name))
            else:
                assert layer in layer_to_chunk, f"{name}: block {layer} outside depth {n_layer}"
                self.chunk_params[layer_to_chunk[layer]].append(p)
                self.chunk_names[layer_to_chunk[layer]].append(strip_prefix(name))
        assert all(self.chunk_params), "every chunk needs at least one parameter"
        self.numel = [sum(p.numel() for p in ps) for ps in self.chunk_params]
        self.always_numel = sum(p.numel() for p in self.always_params)
        self.total = sum(self.numel) + self.always_numel

    @classmethod
    def from_model(cls, model, chunks, direction, sched, end, warmup, max_iters):
        return cls(list(model.named_parameters()), len(model.blocks), chunks, direction, sched,
                   end, warmup, max_iters)

    @classmethod
    def from_arm(cls, model, tag, max_iters, steps_per_epoch):
        """The plan of an arm tag; a tag without -w<W> ramps over one epoch's worth of steps."""
        spec = parse_arm(tag)
        warmup = steps_per_epoch if spec["warmup"] is None else spec["warmup"]
        return cls.from_model(model, spec["chunks"], spec["direction"], spec["sched"],
                              spec["end"], warmup, max_iters)

    # -- optimizer -----------------------------------------------------------------------
    def param_groups(self, weight_decay, skip_list=()):
        """2C + 2 groups: per chunk {decay, no_decay}, then the always-trainable pair. The decay
        rule is optim_factory.get_parameter_groups' (1-d tensors, biases and skip_list names -
        the model's no_weight_decay() - are not decayed). Extra keys: `chunk` (None = always),
        `group_name` (engine.py logs lr/<group_name>) and lr_scale = 1."""
        groups = []
        sets = [(k, ps, ns) for k, (ps, ns) in enumerate(zip(self.chunk_params, self.chunk_names))]
        sets.append((None, self.always_params, self.always_names))
        for k, ps, ns in sets:
            tag = "always" if k is None else f"chunk{k}"
            no_decay = [len(p.shape) == 1 or n.endswith(".bias") or n in skip_list
                        for p, n in zip(ps, ns)]
            groups.append({"params": [p for p, nd in zip(ps, no_decay) if not nd],
                           "weight_decay": weight_decay, "lr_scale": 1.0, "chunk": k,
                           "group_name": f"{tag}/decay"})
            groups.append({"params": [p for p, nd in zip(ps, no_decay) if nd],
                           "weight_decay": 0.0, "lr_scale": 1.0, "chunk": k,
                           "group_name": f"{tag}/no_decay"})
        return groups

    # -- per-step hooks ------------------------------------------------------------------
    def mult(self, k, it):
        return lr_mult(it, self.iters[k], self.warmup)

    def apply_lr(self, optimizer, it, base_lr):
        """group lr = base_lr x lr_scale x the chunk's multiplier (1 for the always groups)."""
        for g in optimizer.param_groups:
            k = g["chunk"]
            m = 1.0 if k is None else self.mult(k, it)
            g["lr"] = base_lr * g.get("lr_scale", 1) * m

    def frozen(self, it):
        return [k for k, u in enumerate(self.iters) if it < u]

    def mask_grads(self, it):
        """Drop the grads of every still-frozen chunk (call after the all-reduce, before
        unscale_ / the clip / the step): AdamW then skips these params entirely."""
        for k in self.frozen(it):
            for p in self.chunk_params[k]:
                p.grad = None

    def is_event(self, it):
        """True at a step where a chunk unfreezes (never at 0: chunk 0 is alive from the start)."""
        return it > 0 and it in self.iters

    def trainable_layers(self, it):
        return sorted(
            layer for k, ls in enumerate(self.layers) if it >= self.iters[k] for layer in ls
        )

    def summary(self, it):
        alive = [k for k in range(self.chunks) if it >= self.iters[k]]
        pending = [u for u in self.iters if u > it]
        return {
            "trainable_layers": len(self.trainable_layers(it)),
            "trainable_param_frac": (self.always_numel + sum(self.numel[k] for k in alive))
            / self.total,
            "n_unfrozen": len(alive),
            "mults": [self.mult(k, it) for k in range(self.chunks)],
            "next_unfreeze": min(pending) if pending else None,
        }

    def wandb_scalars(self, it, base_lr):
        s = self.summary(it)
        out = {
            "unfreeze/trainable_layers": s["trainable_layers"],
            "unfreeze/trainable_param_frac": s["trainable_param_frac"],
            "unfreeze/n_unfrozen": s["n_unfrozen"],
        }
        for k, m in enumerate(s["mults"]):
            out[f"unfreeze/lr_chunk{k}"] = base_lr * m
        return out

    def describe(self, it=None):
        head = (
            f"unfreeze: {self.chunks} chunks ({self.direction}, {self.sched}, end {self.end:g} "
            f"of {self.max_iters}, ramp {self.warmup}): "
            + ", ".join(
                f"chunk{k}=B{ls[0]}-{ls[-1]}@{u}"
                for k, (ls, u) in enumerate(zip(self.layers, self.iters))
            )
        )
        if it is None:
            return head
        s = self.summary(it)
        return (
            f"step {it}: unfreeze event -> {s['n_unfrozen']}/{self.chunks} chunks trainable "
            f"({s['trainable_layers']}/{self.n_layer} blocks, {100 * s['trainable_param_frac']:.1f}% "
            f"of params); next at {s['next_unfreeze']}"
        )
