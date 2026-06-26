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


def get_layer_lr_multiplier(
    block_idx,
    epoch,
    custom_block_targets,
    transition_start=90,
    transition_end=110,
    num_blocks=12,
):
    factor = get_transition_factor(epoch, transition_start, transition_end)
    target = custom_block_targets[block_idx]

    return cosine_interp(1.0, target, factor)


def get_non_block_lr_multiplier(
    param_name,
    epoch,
    custom_non_block_targets,
    transition_start=90,
    transition_end=110,
):
    factor = get_transition_factor(epoch, transition_start, transition_end)
    # print(f"Transition factor for parameter '{param_name}' at epoch {epoch}: {factor:.4f}")
    target = custom_non_block_targets.get(param_name, 1.0)

    return cosine_interp(1.0, target, factor)


def build_vit_param_groups(
    model,
    base_lr,
    epoch,
    custom_block_targets,
    custom_non_block_targets,
    transition_start=90,
    transition_end=110
):
    num_blocks = len(model.blocks)
    param_groups = []

    patch_mult = get_non_block_lr_multiplier(
        "patch_embed",
        epoch,
        custom_non_block_targets,
        transition_start,
        transition_end,
    )
    param_groups.append(
        {
            "params": list(model.patch_embed.parameters()),
            "lr": base_lr * patch_mult,
            "group_name": "patch_embed",
            "lr_mult": patch_mult,
        }
    )

    embed_params = []
    if hasattr(model, "pos_embed") and model.pos_embed is not None:
        embed_params.append(model.pos_embed)
    if hasattr(model, "cls_token") and model.cls_token is not None:
        embed_params.append(model.cls_token)

    if embed_params:
        embed_mult = get_non_block_lr_multiplier(
            "embeddings",
            epoch,
            custom_non_block_targets,
            transition_start,
            transition_end
        )
        param_groups.append(
            {
                "params": embed_params,
                "lr": base_lr * embed_mult,
                "group_name": "embeddings",
                "lr_mult": embed_mult,
            }
        )

    for block_idx, block in enumerate(model.blocks):
        block_mult = get_layer_lr_multiplier(
            block_idx=block_idx,
            epoch=epoch,
            custom_block_targets=custom_block_targets,
            transition_start=transition_start,
            transition_end=transition_end
        )
        param_groups.append(
            {
                "params": list(block.parameters()),
                "lr": base_lr * block_mult,
                "group_name": f"block_{block_idx}",
                "block_idx": block_idx,
                "lr_mult": block_mult,
            }
        )

    if hasattr(model, "norm") and model.norm is not None:
        norm_mult = get_non_block_lr_multiplier(
            "norm",
            epoch,
            custom_non_block_targets,
            transition_start,
            transition_end
        )
        param_groups.append(
            {
                "params": list(model.norm.parameters()),
                "lr": base_lr * norm_mult,
                "group_name": "norm",
                "lr_mult": norm_mult,
            }
        )

    if hasattr(model, "head") and model.head is not None:
        head_mult = get_non_block_lr_multiplier(
            "head",
            epoch,
            custom_non_block_targets,
            transition_start,
            transition_end
        )
        param_groups.append(
            {
                "params": list(model.head.parameters()),
                "lr": base_lr * head_mult,
                "group_name": "head",
                "lr_mult": head_mult,
            }
        )

    return param_groups


def apply_custom_lr_to_optimizer(
    optimizer,
    model,
    base_lr,
    epoch,
    custom_block_targets,
    custom_non_block_targets,
    transition_start=90,
    transition_end=110
):
    new_groups = build_vit_param_groups(
        model=model,
        base_lr=base_lr,
        epoch=epoch,
        custom_block_targets=custom_block_targets,
        custom_non_block_targets=custom_non_block_targets,
        transition_start=transition_start,
        transition_end=transition_end
    )

    if len(new_groups) != len(optimizer.param_groups):
        raise ValueError(
            f"Param group count mismatch: expected {len(new_groups)}, got {len(optimizer.param_groups)}"
        )

    for opt_group, new_group in zip(optimizer.param_groups, new_groups):
        opt_group["lr"] = new_group["lr"]
        opt_group["lr_mult"] = new_group.get("lr_mult", 1.0)
        opt_group["group_name"] = new_group.get("group_name", None)
        if "block_idx" in new_group:
            opt_group["block_idx"] = new_group["block_idx"]


def get_epoch_base_lr(base_lr_schedule, epoch):
    if epoch < 0 or epoch >= len(base_lr_schedule):
        raise IndexError(
            f"Epoch {epoch} is outside base_lr_schedule of length {len(base_lr_schedule)}"
        )
    return base_lr_schedule[epoch]
