"""Per-row learning-rate scales inside one tensor (the fused attn.qkv.weight), as an exact post-step correction to AdamW.

Why: a learning-rate scale is a property of a parameter group, i.e. of a whole tensor. q, k and v live in one fused tensor, so
"slow steps on q and k, normal steps on v" cannot be expressed with --lr_scale_json alone (docs/proc_init_recipe.md, sections
12-13). Scaling the gradient of some rows would not help either: Adam normalises every coordinate by its own second moment.

How: AdamW's update is linear in the learning rate per coordinate,

    p <- p - lr * wd * p                      (decoupled decay)
    p <- p - lr * m_hat / (sqrt(v_hat) + eps)  (adaptive step)

so after the stock step has run with the plain lr, the rows of a masked tensor receive the extra update
-lr * (lambda_row - 1) * m_hat / (sqrt(v_hat) + eps), computed from the moment estimates the optimiser has just stored. The
result is exactly the update of a parameter group with lr * lambda and wd / lambda (the --lr_scale_json semantics: relative
step scaled by lambda, relative decay unchanged), row by row. Weight decay is never touched.

Skipped steps: the correction is applied only when the optimiser's own step counter of the tensor advanced during
optimizer.step(). A step the AMP grad scaler skips never calls optimizer.step() on the ordinary path; native fused AdamW is
called with the overflow flag and skips inside the kernel without advancing the counter -- both leave the tensor untouched.

Specification: in the lr-scale JSON a tensor may map to {"rows": [[start, end, lambda], ...]} instead of a number; rows not
listed keep lambda = 1. The tensor then belongs to an ordinary parameter group (lr_scale 1). Only torch.optim.AdamW without
amsgrad is supported; anything else raises. Lambdas must be finite and positive, row boundaries integers.

Provenance: the optimiser carries the full lr specification (scalar scales and row masks) as `_lr_scale_spec`; utils.save_model
writes it into every checkpoint and utils.auto_load_model refuses to resume with a different one.

Off by default: nothing here runs unless --lr_scale_json names a tensor with a "rows" entry.
Verified by plots/verify/test_row_lr_mask.py against a model with the rows split into separate tensors.
"""
import math
import torch


def _finite_positive(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: lambda must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{what}: lambda must be finite and positive, got {value!r}")
    return v


def split_lr_scale_spec(spec):
    """{name: number | {"rows": [[start, end, lambda], ...]}} -> (scalar part {name: lambda}, row part {name: [(start, end, lambda)]}).
    Values are validated: finite positive lambdas, integer row boundaries, no other keys in a row entry."""
    if not isinstance(spec, dict):
        raise ValueError(f"lr scale specification must be a JSON object, got {type(spec).__name__}")
    scalar, rows = {}, {}
    for name, value in spec.items():
        if isinstance(value, dict):
            ranges = value.get("rows")
            if set(value) != {"rows"} or not isinstance(ranges, list) or not ranges or \
                    any(not isinstance(r, (list, tuple)) or len(r) != 3 for r in ranges):
                raise ValueError(f"row lr mask for {name}: expected {{'rows': [[start, end, lambda], ...]}}, got {value}")
            out = []
            for a, b, l in ranges:
                for x in (a, b):
                    if isinstance(x, bool) or not isinstance(x, (int, float)) or not float(x).is_integer():
                        raise ValueError(f"row lr mask for {name}: row boundaries must be integers, got {a!r}, {b!r}")
                out.append((int(a), int(b), _finite_positive(l, f"row lr mask for {name}")))
            rows[name] = out
        else:
            scalar[name] = _finite_positive(value, f"lr scale for {name}")
    return scalar, rows


def row_mask_vector(param, ranges, name=""):
    """lambda per row (dim 0) of `param`; rows outside every range keep 1; ranges must lie inside the tensor and not overlap."""
    n = param.shape[0]
    lam = torch.ones(n, dtype=torch.float64)
    covered = torch.zeros(n, dtype=torch.bool)
    for start, end, value in ranges:
        if not (0 <= start < end <= n):
            raise ValueError(f"row lr mask for {name}: range [{start}, {end}) outside the tensor's {n} rows")
        if covered[start:end].any():
            raise ValueError(f"row lr mask for {name}: overlapping ranges")
        _finite_positive(value, f"row lr mask for {name}")
        lam[start:end] = value; covered[start:end] = True
    return lam


def _step_value(state):
    """The optimiser's step counter of one parameter as a float, or None before the first applied step."""
    if not state or "step" not in state:
        return None
    step = state["step"]
    return float(step.item()) if torch.is_tensor(step) else float(step)


def install_row_lr_masks(optimizer, model, rows, verbose=True):
    """Attach the post-step correction for the tensors named in `rows` ({name: [(start, end, lambda)]}). Returns the pair of
    hook handles (None when `rows` is empty). Raises if the optimizer is not a plain AdamW, if amsgrad is on, if a masked tensor
    sits in a group with its own lr_scale, or if a name is unknown."""
    if not rows:
        return None
    if type(optimizer) is not torch.optim.AdamW and not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError(f"row lr masks need torch.optim.AdamW, got {type(optimizer).__name__}")
    params = dict(model.named_parameters())
    masked = []
    for name, ranges in rows.items():
        if name not in params:
            raise KeyError(f"row lr mask: no parameter named {name!r}")
        p = params[name]
        if not p.requires_grad:
            raise ValueError(f"row lr mask: {name!r} is frozen (requires_grad False)")
        group = _group_of(optimizer, p)
        if group is None:
            raise KeyError(f"row lr mask: {name!r} is not in any optimizer parameter group")
        if group.get("amsgrad", False):
            raise ValueError("row lr masks are not implemented for amsgrad")
        if abs(group.get("lr_scale", 1.0) - 1.0) > 0:
            raise ValueError(f"row lr mask: {name!r} sits in a group with lr_scale {group['lr_scale']}; give it either a group scale or a row mask")
        lam = row_mask_vector(p, ranges, name).to(device=p.device, dtype=p.dtype)
        masked.append((name, p, lam))
        if verbose:
            print(f"[row-lr] {name:32s} " + ", ".join(f"rows [{a},{b}) lr x{l:.4f}" for a, b, l in ranges) + " (wd unchanged)", flush=True)
    optimizer._row_lr_masks = masked                      # kept for inspection / tests
    optimizer._row_lr_spec = {name: [[int(a), int(b), float(l)] for a, b, l in ranges] for name, ranges in rows.items()}
    steps_before = {}

    def _remember(opt, args, kwargs):
        for name, p, lam in masked:
            steps_before[name] = _step_value(opt.state.get(p))

    def _correct(opt, args, kwargs):
        for name, p, lam in masked:
            state = opt.state.get(p)
            after = _step_value(state)
            if after is None or after == steps_before.get(name) or "exp_avg" not in state:
                continue                                   # no applied update for this tensor: no grad, or a skipped (overflow) step
            group = _group_of(opt, p)
            beta1, beta2 = group["betas"]
            lr, eps = float(group["lr"]), float(group["eps"])
            bias_correction1 = 1.0 - beta1 ** after
            bias_correction2 = 1.0 - beta2 ** after
            denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            adaptive = state["exp_avg"] / denom * (lr / bias_correction1)          # what the stock step subtracted, per coordinate
            shape = (-1,) + (1,) * (p.dim() - 1)
            with torch.no_grad():
                p.sub_((lam - 1.0).view(shape) * adaptive)

    return optimizer.register_step_pre_hook(_remember), optimizer.register_step_post_hook(_correct)


def lr_spec_of(optimizer):
    """The learning-rate specification the optimiser was built with ({"scalar": {name: lambda}, "rows": {name: [[a, b, lambda], ...]}}),
    or None when no --lr_scale_json was given. Written into checkpoints by utils.save_model."""
    scalar = getattr(optimizer, "_lr_scale_spec", None)
    rows = getattr(optimizer, "_row_lr_spec", None)
    if scalar is None and rows is None:
        return None
    return {"scalar": {k: float(v) for k, v in (scalar or {}).items()}, "rows": {k: [list(map(float, r)) for r in v] for k, v in (rows or {}).items()}}


def assert_same_lr_spec(saved, current, where=""):
    """Refuse to resume when the checkpoint's lr specification and the optimiser's differ (a mutable JSON path is not provenance)."""
    def _norm(s):
        if s is None:
            return None
        return {"scalar": {k: round(float(v), 12) for k, v in s.get("scalar", {}).items()},
                "rows": {k: [[round(float(x), 12) for x in r] for r in v] for k, v in s.get("rows", {}).items()}}
    if _norm(saved) != _norm(current):
        raise RuntimeError(f"lr specification mismatch on resume{(' from ' + where) if where else ''}: checkpoint {saved} vs optimizer {current}")


def _group_of(optimizer, p):
    for group in optimizer.param_groups:
        for q in group["params"]:
            if q is p:
                return group
    return None
