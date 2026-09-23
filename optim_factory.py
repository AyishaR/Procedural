import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
# from timm.optim.nadam import Nadam
#from timm.optim.novograd import NovoGrad
#from timm.optim.nvnovograd import NvNovoGrad
# from timm.optim.radam import RAdam
# from timm.optim.rmsprop_tf import RMSpropTF
# from timm.optim.sgdp import SGDP
from custom_lr import *
from row_lr_mask import split_lr_scale_spec, install_row_lr_masks

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_convnext(var_name):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three 
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            layer_id = 3 + block_id // 3 
        elif stage_id == 3:
            layer_id = 12
        return layer_id
    else:
        return num_max_layer + 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())



def build_step_matched_param_groups(model, weight_decay, ckpt_path, blocks, skip_list=()):
    """Parameter groups whose per-group learning-rate scale makes a RANDOM-init tensor take the
    same RELATIVE AdamW step as the same tensor of a reference checkpoint would at the base lr.

    Adam's per-element step is ~lr regardless of the weight scale, so the relative change of a
    tensor per step is ~lr / rms(W). A tensor whose checkpoint version has k times the rms of
    the random init therefore moves k times slower (in relative terms) when trained from the
    checkpoint. Setting lr_scale = rms(W_random) / rms(W_ckpt) reproduces that slowdown (or
    speedup, for proc's fc1/fc2 whose rms is below random's) WITHOUT changing the forward pass
    at init. wd_scale = 1 / lr_scale keeps lr*wd, i.e. the relative decay per step, unchanged.
    The fused attn.qkv.weight is one parameter, so q, k and v share the pooled multiplier.
    Blocks outside `blocks`, all 1-D params and biases keep lr_scale 1 (docs 0d.9, arm ftblrm).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("state", "model", "module", "model_state_dict", "state_dict"):
        if isinstance(ck, dict) and key in ck:
            ck = ck[key]; break
    matched = {"attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"}
    groups, table = {}, []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list
        lr_scale = 1.0
        parts = name.split(".")
        if parts[0] == "blocks" and parts[1].isdigit() and int(parts[1]) in blocks:
            tname = ".".join(parts[2:])
            if tname in matched and name in ck:
                r_rms = param.data.float().pow(2).mean().sqrt().item()
                c_rms = ck[name].float().pow(2).mean().sqrt().item()
                lr_scale = r_rms / c_rms
                table.append((name, r_rms, c_rms, lr_scale))
        key = ("no_decay" if no_decay else "decay", round(lr_scale, 6))
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay,
                           "lr_scale": lr_scale, "wd_scale": (1.0 / lr_scale) if (not no_decay and lr_scale > 0) else 1.0}
        groups[key]["params"].append(param)
    print("[lr-match] per-tensor lr_scale = rms(random) / rms(ckpt):", flush=True)
    for name, r, c, m in table:
        print(f"[lr-match]   {name:32s} rms {r:.5f} / {c:.5f} -> lr x{m:.3f}, wd x{1/m:.3f}", flush=True)
    print(f"[lr-match] {len(table)} tensors matched, {len(groups)} param groups", flush=True)
    return list(groups.values())


def build_lr_scaled_param_groups(model, weight_decay, spec, skip_list=()):
    """Parameter groups with per-tensor learning-rate scales given explicitly: `spec` is
    {parameter name: lr_scale} (or a path to such a JSON). Like build_step_matched_param_groups,
    wd_scale = 1 / lr_scale keeps the relative decay per step unchanged; every other tensor keeps
    lr_scale 1. First used by ftbanal (2026-09): ftbana's forward pass (effective scales in the raw
    weights, gains 1) with the relative Adam steps of ftbanag, whose q/k/v/fc1 weights are 1/rms(gamma)
    larger, i.e. lr_scale = rms(gamma_b) on attn.qkv.weight and mlp.fc1.weight of blocks 0-8 (docs 0d.11);
    then by the late-lever 2x2 (ftbrhopl, ftbrhosl). Every name in `spec` must be a trainable parameter
    and every value a finite positive number; groups are keyed by the exact value, and each group
    carries "decay" (True/False) so the schedule in engine.py never uses the current coefficient as
    the eligibility flag."""
    import json, os, math
    if isinstance(spec, str):
        spec = json.load(open(spec)) if os.path.exists(spec) else json.loads(spec)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    unknown = sorted(k for k in spec if k not in trainable)
    if unknown:
        raise KeyError(f"--lr_scale_json names {len(unknown)} tensor(s) that are not trainable parameters: {unknown[:5]}")
    for k, v in spec.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError(f"--lr_scale_json: lr scale of {k} must be a finite positive number, got {v!r}")
    groups, table = {}, []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list
        lr_scale = float(spec.get(name, 1.0))
        if name in spec:
            table.append((name, lr_scale))
        key = ("no_decay" if no_decay else "decay", lr_scale)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay, "decay": not no_decay,
                           "lr_scale": lr_scale, "wd_scale": (1.0 / lr_scale) if not no_decay else 1.0}
        groups[key]["params"].append(param)
    print("[lr-scale] explicit per-tensor lr_scale:", flush=True)
    for name, m in table:
        print(f"[lr-scale]   {name:32s} lr x{m:.4f}, wd x{1/m:.3f}", flush=True)
    print(f"[lr-scale] {len(table)} tensors scaled, {len(groups)} param groups", flush=True)
    return list(groups.values())


def release_factor(epoch_position, start, end):
    """Learning-rate factor of the blocks under --release_blocks: 0 before `start` (in epochs, fractional), linear to 1 at `end`,
    1 afterwards. With lr = 0 AdamW changes nothing (the decoupled decay is lr * wd * w), so the blocks are frozen bit for bit while
    their Adam moments keep following the gradient; the release therefore starts with warm moments."""
    if epoch_position < start: return 0.0
    if epoch_position >= end or end <= start: return 1.0
    return (epoch_position - start) / float(end - start)


def build_block_release_param_groups(model, weight_decay, blocks, skip_list=()):
    """Parameter groups for a delayed start of chosen blocks (docs/early_lever_mechanism_plan.md, converse of C1): the tensors of
    `blocks` go into groups flagged "release", whose learning rate engine.train_one_epoch multiplies by release_factor; everything
    else is grouped as get_parameter_groups does (decay / no_decay, lr_scale 1, wd_scale 1). Weight decay is NOT compensated: the
    released blocks simply train with a smaller step, and do not move at all while the factor is 0."""
    blocks = sorted(int(b) for b in blocks); prefixes = tuple(f"blocks.{b}." for b in blocks)
    assert blocks and all(0 <= b < len(model.blocks) for b in blocks), blocks
    groups, n_released = {}, 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = len(param.shape) == 1 or name.endswith(".bias") or name in skip_list
        released = name.startswith(prefixes); n_released += int(released)
        key = ("no_decay" if no_decay else "decay", released)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay, "decay": not no_decay, "lr_scale": 1.0, "wd_scale": 1.0, "release": released}
        groups[key]["params"].append(param)
    print(f"[release] blocks {blocks}: {n_released} tensors in {sum(1 for k in groups if k[1])} released groups, {len(groups)} param groups in total", flush=True)
    return list(groups.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, start_lr=None, custom_block_targets=None, custom_non_block_targets=None, custom_lr_transition_start=90, custom_lr_transition_end=110):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]

    if getattr(args, "lr_match_ckpt", "") and getattr(args, "lr_scale_json", ""):
        raise ValueError("--lr_scale_json and --lr_match_ckpt both build the parameter groups; give one of them")
    if getattr(args, "release_blocks", "") and (getattr(args, "lr_match_ckpt", "") or getattr(args, "lr_scale_json", "") or args.custom_lr_layer):
        raise ValueError("--release_blocks builds its own parameter groups and cannot be combined with --lr_match_ckpt / --lr_scale_json / --custom_lr_layer")
    if getattr(args, "lr_match_ckpt", ""):
        blocks = [int(x) for x in str(args.lr_match_blocks).split(",") if x.strip() != ""]
        parameters = build_step_matched_param_groups(model, args.weight_decay, args.lr_match_ckpt, blocks,
                                                     skip_list=skip if filter_bias_and_bn else ())
        weight_decay = 0.

    row_lr_masks, _scalar = {}, None
    if getattr(args, "lr_scale_json", ""):
        import os as _os
        _spec = args.lr_scale_json
        _spec = json.load(open(_spec)) if _os.path.exists(_spec) else json.loads(_spec)
        _scalar, row_lr_masks = split_lr_scale_spec(_spec)      # per-tensor scales here, per-row masks after the optimizer exists
        _trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        _unknown = sorted(k for k in row_lr_masks if k not in _trainable)
        if _unknown:
            raise KeyError(f"--lr_scale_json row masks name tensors that are not trainable parameters: {_unknown}")
        parameters = build_lr_scaled_param_groups(model, args.weight_decay, _scalar,
                                                  skip_list=skip if filter_bias_and_bn else ())
        weight_decay = 0.

    if getattr(args, "release_blocks", ""):
        parameters = build_block_release_param_groups(model, args.weight_decay, [int(b) for b in str(args.release_blocks).split(",") if b.strip() != ""],
                                                      skip_list=skip if filter_bias_and_bn else ())
        weight_decay = 0.

    if args.custom_lr_layer:
        if getattr(args, "lr_match_ckpt", "") or getattr(args, "lr_scale_json", ""):
            raise ValueError("--custom_lr_layer builds its own parameter groups and cannot be combined with --lr_match_ckpt / --lr_scale_json")
        no_decay_names = (skip_list if skip_list is not None else model.no_weight_decay() if hasattr(model, "no_weight_decay") else ()) if filter_bias_and_bn else ()
        parameters = build_vit_param_groups(
            model=model,
            base_lr=start_lr if start_lr is not None else args.lr,
            epoch=0,
            custom_block_targets=custom_block_targets,
            custom_non_block_targets=custom_non_block_targets,
            transition_start=custom_lr_transition_start,
            transition_end=custom_lr_transition_end,
            weight_decay=args.weight_decay,          # carried per group, as get_parameter_groups does (the optimizer default is 0 here)
            skip_list=no_decay_names
        )
        weight_decay = 0.

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    # elif opt_lower == 'nadam':
    #     optimizer = Nadam(parameters, **opt_args)
    # elif opt_lower == 'radam':
    #     optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    # elif opt_lower == 'sgdp':
    #     optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'rmsproptf':
    #     optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    if _scalar is not None:
        optimizer._lr_scale_spec = dict(_scalar)                # provenance: written into checkpoints, validated on resume
    if row_lr_masks:
        install_row_lr_masks(optimizer, model, row_lr_masks)    # exact post-step correction, see row_lr_mask.py

    return optimizer
