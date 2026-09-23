"""Layer-wise learning-rate multipliers that fade in over a window of epochs (--custom_lr_layer).

At every optimisation step   lr(group) = global schedule(step) * multiplier(group, epoch) * lr_scale(group),
with   multiplier = 1 before `transition_start`, a cosine ramp from 1 to the group's target between `transition_start` and
`transition_end`, and the target afterwards. The ramp is evaluated per EPOCH, so the multiplier is constant within an epoch
while the global schedule changes every step.

Groups: patch_embed, pos_embed, cls_token, one per block, the final norm, the head, each split into a "decay" and a
"no_decay" part by the rule of optim_factory.get_parameter_groups (1-D tensors, biases and the model's no_weight_decay()
names are not decayed), so switching the option on does not change the weight decay. Every group remembers its own key
("lr_key"), so the per-step update does not depend on the order of optimizer.param_groups.
"""
import math


def cosine_interp(start, end, factor):
    factor = min(max(factor, 0.0), 1.0)
    smooth = 0.5 * (1.0 - math.cos(math.pi * factor))
    return start + smooth * (end - start)


def get_transition_factor(epoch, transition_start=90, transition_end=110):
    if epoch < transition_start:
        return 0.0
    if epoch >= transition_end:
        return 1.0
    return (epoch - transition_start) / float(transition_end - transition_start)


def get_layer_lr_multiplier(block_idx, epoch, custom_block_targets, transition_start=90, transition_end=110):
    factor = get_transition_factor(epoch, transition_start, transition_end)
    return cosine_interp(1.0, custom_block_targets[block_idx], factor)


def get_non_block_lr_multiplier(param_name, epoch, custom_non_block_targets, transition_start=90, transition_end=110):
    factor = get_transition_factor(epoch, transition_start, transition_end)
    return cosine_interp(1.0, custom_non_block_targets.get(param_name, 1.0), factor)


def lr_multiplier(lr_key, epoch, custom_block_targets, custom_non_block_targets, transition_start=90, transition_end=110):
    """Multiplier of one parameter group; `lr_key` is ("block", index) or ("non_block", name)."""
    kind, which = lr_key
    if kind == "block":
        return get_layer_lr_multiplier(which, epoch, custom_block_targets, transition_start, transition_end)
    return get_non_block_lr_multiplier(which, epoch, custom_non_block_targets, transition_start, transition_end)


NON_BLOCK_OWNERS = ("patch_embed", "pos_embed", "cls_token", "norm", "head")


def build_vit_param_groups(model, base_lr, epoch, custom_block_targets, custom_non_block_targets,
                           transition_start=90, transition_end=110, weight_decay=0.0, skip_list=()):
    num_blocks = len(model.blocks)
    if len(custom_block_targets) != num_blocks:
        raise ValueError(f"--custom_block_targets_scale needs one target per block: expected {num_blocks}, got {len(custom_block_targets)}")
    unknown = set(custom_non_block_targets) - set(NON_BLOCK_OWNERS)
    if unknown:
        raise ValueError(f"custom_non_block_targets has keys without a parameter group: {sorted(unknown)} (known: {NON_BLOCK_OWNERS})")

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        owner = name.split(".")[0]
        if owner == "blocks":
            lr_key, label = ("block", int(name.split(".")[1])), f"block_{int(name.split('.')[1])}"
        elif owner in NON_BLOCK_OWNERS:
            lr_key, label = ("non_block", owner), owner
        else:       # a parameter without a group would silently never be trained
            raise ValueError(f"--custom_lr_layer: parameter {name!r} belongs to no learning-rate group")
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list      # optim_factory.get_parameter_groups' rule
        group_name = f"{label}_{'no_decay' if no_decay else 'decay'}"
        if group_name not in groups:
            multiplier = lr_multiplier(lr_key, epoch, custom_block_targets, custom_non_block_targets, transition_start, transition_end)
            groups[group_name] = {"params": [], "lr": base_lr * multiplier, "weight_decay": 0.0 if no_decay else weight_decay,
                                  "group_name": group_name, "lr_key": lr_key, "lr_mult": multiplier}
            if lr_key[0] == "block":
                groups[group_name]["block_idx"] = lr_key[1]
        groups[group_name]["params"].append(param)
    return list(groups.values())


def apply_custom_lr_to_optimizer(optimizer, base_lr, epoch, custom_block_targets, custom_non_block_targets,
                                 transition_start=90, transition_end=110, model=None):
    """Set every group's learning rate for this step. Groups are identified by the "lr_key" they were created with, not by
    their position. `model` is accepted for backward compatibility and not used."""
    for group in optimizer.param_groups:
        if "lr_key" not in group:
            raise ValueError("--custom_lr_layer: the optimizer has a parameter group that build_vit_param_groups did not create "
                             f"(keys {sorted(k for k in group if k != 'params')}); the layer-wise multipliers cannot be applied to it")
        multiplier = lr_multiplier(tuple(group["lr_key"]), epoch, custom_block_targets, custom_non_block_targets, transition_start, transition_end)
        group["lr"] = base_lr * multiplier * group.get("lr_scale", 1.0)
        group["lr_mult"] = multiplier
