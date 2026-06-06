from utils import *
import torch
import pandas as pd
import math
from typing import Iterable, Optional
import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from pprint import pprint
import torch.nn.functional as F
import utils
import json
import os
import sys
from matplotlib import pyplot as plt
from collections import defaultdict
from cka_utils import linear_cka, gram_cka
from models.vitp import VitProcedural
from kdyck.kdyck_dataset import KDyckDataset, mask_kdyck_dataset
from procedural_data.repeat_dataset import RepeatDataset, mask_repeat_dataset
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from scipy.stats import ks_2samp, ttest_ind
from metrics.ddp_multiclassECE import DDPMulticlassECE

from matplotlib import colors
from scipy.stats import gaussian_kde, skew, kurtosis, ks_2samp, wasserstein_distance

def cosine(a, b, eps=1e-8):
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def select_tokens(x, token_mode="patch"):
    # x: [B, N, D]
    if token_mode == "cls":
        return x[:, :1, :]
    elif token_mode == "patch":
        return x[:, 1:, :]
    elif token_mode == "all":
        return x
    else:
        raise ValueError(f"Unsupported token_mode: {token_mode}")


def attention_residual_loss(
    rin,
    rout,
    targets,
    weights
): #TODO
    loss = rin.new_tensor(0.0)
    logs = {}
    current_values = {}

    for token_mode in ["cls", "patch"]:
        tag = "" if token_mode=="patch" else f"{token_mode}_"
        rin_part = select_tokens(rin, token_mode)
        rout_part = select_tokens(rout, token_mode)

        delta = rout_part - rin_part
        rin_norm = rin_part.norm(dim=-1)
        rout_norm = rout_part.norm(dim=-1)
        norm_ratio = rout_norm / (rin_norm + 1e-8)
        cosine_rin_rout = F.cosine_similarity(rin_part, rout_part, dim=-1)

        current_values.update({
            f"norm_ratio_{tag}mean": norm_ratio.mean(),
            f"norm_ratio_{tag}std": norm_ratio.std(),
            f"cosine_rin_rout_{tag}mean": cosine_rin_rout.mean(),
            f"cosine_rin_rout_{tag}std": cosine_rin_rout.std(),
        }) 

    for target_name, target_value in targets.items():
        tgt = torch.as_tensor(target_value, device=rout.device, dtype=rout.dtype)
        if target_name == "cka_rin_rout": 
            continue
        try:
            existing_value = current_values[target_name]
        except KeyError:
            raise KeyError(f"Target value '{target_name}' not found in current values. Available keys: {list(current_values.keys())}")
        logs[target_name+"_loss"] = weights[target_name] * F.mse_loss(existing_value, tgt)
        logs[target_name] = existing_value
        loss += logs[target_name+"_loss"]

    if "cka_rin_rout" in targets:
        rin = select_tokens(rin, "all")
        rout = select_tokens(rout, "all")
        rin_flat = rin.view(rin.shape[0], -1)
        rout_flat = rout.view(rout.shape[0], -1)
        cka_rin_rout = gram_cka(rin_flat, rout_flat)
        current_values["cka_rin_rout"] = cka_rin_rout
        tgt = torch.as_tensor(targets["cka_rin_rout"], device=rout.device, dtype=rout.dtype)
        logs["cka_rin_rout_loss"] = weights["cka_rin_rout"] * F.mse_loss(cka_rin_rout, tgt)
        loss += logs["cka_rin_rout_loss"]
        logs["cka_rin_rout"] = cka_rin_rout.item()

    print("Current values for loss computation:")
    pprint(current_values)
    logs["loss"] = loss.item()
    return loss, logs

def custom_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    value_targets, value_weights,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    grad_metric_logger = utils.MetricLogger(delimiter=" ")

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # if use_amp:
        #     with torch.cuda.amp.autocast():
        #         output = model(samples)
        #         loss = criterion(output, targets)
        # else: # full precision
        #     output = model(samples)
        #     loss = criterion(output, targets)
        with utils.HookCollectorTrain(model) as acts:
            output = model(samples)

        loss, loss_logs = attention_residual_loss(
            rin=acts[11]['inp'],
            rout=acts[11]['attn'],
            targets=value_targets,
            weights=value_weights
        )

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm, parameter_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.named_parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            parameter_norm = None
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        param_norms_logging = {}
        if parameter_norm is not None:
            for param_name, param_grad in parameter_norm.items():
                param_norms_logging[f'Rank-0 Batch Wise/param_norm/{param_name}'] = param_grad
                grad_metric_logger.meters[param_name].update(param_grad, n=samples.shape[0])

        if wandb_logger:
            for key, value in loss_logs.items():
                wandb_logger._wandb.log({f'Custom train/{key}': value}, commit=False)
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                if grad_norm is not None:
                    wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
                if parameter_norm is not None:
                    wandb_logger._wandb.log(param_norms_logging, commit=False)
                        
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it, 'Rank-0 Batch Wise/epoch': epoch})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    grad_metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},{k: meter.global_avg for k, meter in grad_metric_logger.meters.items()}

