# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
from custom_lr import *

from optim_factory import *

def train_one_epoch(model: torch.nn.Module, model_without_ddp, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,
                    custom_lr_layer=False,
                    custom_lr_transition_start=None,
                    custom_lr_transition_end=None,
                    custom_block_targets=None, custom_non_block_targets=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

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
                if custom_lr_layer:
                    apply_custom_lr_to_optimizer(
                        optimizer=optimizer,
                        model=model_without_ddp,
                        base_lr=lr_schedule_values[it],
                        epoch=epoch,
                        custom_block_targets=custom_block_targets,
                        custom_non_block_targets=custom_non_block_targets,
                        transition_start=custom_lr_transition_start,
                        transition_end=custom_lr_transition_end
                    )
                else:
                    for i, param_group in enumerate(optimizer.param_groups):
                        if lr_schedule_values is not None:
                            param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1)
                        if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                            param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

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
        param_group_lrs = {}
        for gi, group in enumerate(optimizer.param_groups):
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            # print(group)
            param_group_lrs[group.get("group_name", f"group{gi}")] = group["lr"]

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
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            for pname, lr_val in param_group_lrs.items():
                wandb_logger._wandb.log({f'lr/{pname}': lr_val}, commit=False)
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

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_distance_matrix(length, device, patch_size=16):
    """
    length: number of patches along one axis (e.g. 14 for 224/16).
    patch_size: patch edge in pixels (e.g. 16). Only a scale factor.
    Returns D \in R^{P x P}, P = length * length.
    """
    # Grid coordinates in patch index space
    ys, xs = torch.meshgrid(
        torch.arange(length, device=device),
        torch.arange(length, device=device),
        indexing="ij",
    )  # (length, length)

    coords = torch.stack([ys, xs], dim=-1).float().view(-1, 2)  # (P, 2), P = length*length

    # Euclidean distance in index space, scaled by patch size -> pixels
    dist = torch.cdist(coords, coords, p=2) * float(patch_size)  # (P, P)
    return dist  # (P, P)

def compute_detailed_metrics(acts, i, images, metric_logger, num_heads, prefix=""):
    attn_map = acts[i]['attn_map']
    attn_norm = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum(dim=-1)
    entropy_mean = entropy.mean(dim=(0,2))
    for hi in range(num_heads):
        metric_logger.meters[f'{prefix}attn_entropy_head{hi}_layer{i}'].update(entropy_mean[hi].item(), n=images.shape[0])
    metric_logger.meters[f'{prefix}attn_entropy_layer{i}'].update(entropy_mean.mean().item(), n=images.shape[0])

    cls_attn = attn_map[:, :, 0, 1:]
    cls_entropy = -(cls_attn * torch.log(cls_attn + 1e-8)).sum(dim=-1)
    cls_entropy_mean = cls_entropy.mean(dim=(0))
    for hi in range(num_heads):
        metric_logger.meters[f'{prefix}cls_attn_entropy_head{hi}_layer{i}'].update(cls_entropy_mean[hi].item(), n=images.shape[0])
    metric_logger.meters[f'{prefix}cls_attn_entropy_layer{i}'].update(cls_entropy_mean.mean().item(), n=images.shape[0])

    attn_pp = attn_map[..., 1:, 1:]
    P = attn_pp.shape[-1]
    length = int(P ** 0.5)

    D = compute_distance_matrix(length, attn_pp.device)
    D = D.view(1, 1, P, P)
    mad_per_token = (attn_pp * D).sum(dim=-1)
    mad_per_head = mad_per_token.mean(dim=(0, 2))
    for hi in range(num_heads):
        metric_logger.meters[f'{prefix}attn_mad_head{hi}_layer{i}'].update(mad_per_head[hi].item(), n=images.shape[0])
    metric_logger.meters[f'{prefix}attn_mad_layer{i}'].update(mad_per_head.mean().item(), n=images.shape[0])

@torch.no_grad()
def model_analyse(
    model, 
    data_loader, 
    device, 
    epoch, 
    args, 
    prefix="", 
    shuffled_block_order=None, 
    parameter_norm=None, 
    wandb_logger=None
):
    criterion = torch.nn.CrossEntropyLoss()

    stats = []
    # cka_features = defaultdict(list)

    metric_logger = utils.MetricLogger(delimiter="  ")
    detailed_metrics_logger = utils.MetricLogger(delimiter="  ")
    header = 'attention_analyse:'

    ece_metric = DDPMulticlassECE(n_bins=15, device=device)
    ece_metric.reset()

    layers_to_analyse = range(len(model.blocks))
    # layers_to_analyse = [1]

    patch_count = 14  # 224/16
    num_heads = model.blocks[0].attn.num_heads
    loss = None
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                with utils.HookCollector(model) as acts:
                    with torch.no_grad():
                        output = model(images)
                        if criterion is not None:
                            loss = criterion(output, target)
        else:
            with utils.HookCollector(model) as acts:
                with torch.no_grad():
                    output = model(images)
                    if criterion is not None:
                        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        if loss is not None:
            metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        layer_wise_blk_logits, layer_wise_attn_logits = {}, {}

        with torch.no_grad():
            for i in layers_to_analyse:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x = model.norm(acts[i]['attn'])
                    x = model.fc_norm(x)
                    x = model.head(x) 
                    x = F.softmax(x[:, 0, :].squeeze(), dim=-1)
                # layer_wise_attn_logits[i] = x.argmax(dim=-1)
                
                attn_pred = accuracy(x, target)
                detailed_metrics_logger.meters[f'{prefix}attn_acc_layer{i}'].update(attn_pred[0].item(), n=images.shape[0])

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x = model.norm(acts[i]['blk'])
                    x = model.fc_norm(x)
                    x = model.head(x) 
                    x = F.softmax(x[:, 0, :].squeeze(), dim=-1)
                # layer_wise_blk_logits[i] = x.argmax(dim=-1)

                ece_metric.update(x, target)

                blk_pred = accuracy(x, target)
                detailed_metrics_logger.meters[f'{prefix}blk_acc_layer{i}'].update(blk_pred[0].item(), n=images.shape[0])
                detailed_metrics_logger.meters[f'{prefix}blk_act_norm_layer{i}'].update(acts[i]['blk_act_norm'], n=images.shape[0])
                detailed_metrics_logger.meters[f'{prefix}blk_act_rms_layer{i}'].update(acts[i]['blk_act_rms'], n=images.shape[0])

                # cka_features[i].append(acts[i]['blk'].cpu())

                if args.detailed_metrics:
                    compute_detailed_metrics(acts, i, images, detailed_metrics_logger, num_heads, prefix=prefix)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    try:
        print('* loss {losses.global_avg:.3f}'
            .format(losses=metric_logger.loss))
    except AttributeError:
        pass

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    detailed_metrics_logger.synchronize_between_processes()

    ece = ece_metric.compute()
    # if wandb_logger:
    #     wandb_logger.log_epoch_metrics({"epoch": epoch, "test_ece": ece})
    if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
        print("ECE:", ece)
        detailed_metrics_logger.meters["ece"].update(ece, n=1)

    blk_metric_keys = ["cls_attn_entropy", "attn_entropy", "attn_mad", "blk_acc", "blk_act_norm", "blk_act_rms"]
    attn_metric_keys = ["attn_acc"]

    for layer in layers_to_analyse:
        stats_dict = {
            "epoch": epoch if prefix == "" else -1,
            "layer": layer,
        }
        for head in range(num_heads):
            head_stats_dict = stats_dict.copy()
            head_stats_dict["head"] = head
            for key in blk_metric_keys:
                metric_name = f'{prefix}{key}_head{head}_layer{layer}'
                # print(metric_name)
                if metric_name in detailed_metrics_logger.meters:
                    head_stats_dict.update({
                        f'{prefix}{key}': detailed_metrics_logger.meters[metric_name].global_avg,
                    })
                else:
                    # print(f"Metric Warning: {metric_name} not found in detailed_metrics_logger.meters")
                    pass
            # if args.gpu == 0: pprint(head_stats_dict)
            stats.append(head_stats_dict)

        attn_stats_dict = stats_dict.copy()
        for key in attn_metric_keys:
            metric_name = f'{prefix}{key}_layer{layer}'
            # print(metric_name)
            if metric_name in detailed_metrics_logger.meters:
                if key == "attn_acc":
                    attn_stats_dict["layer_sub"] = attn_stats_dict["layer"]-0.5
                    key="acc"
                attn_stats_dict.update({
                    f'{prefix}{key}': detailed_metrics_logger.meters[metric_name].global_avg,
                })
            else:
                # print(f"Metric Warning: {metric_name} not found in detailed_metrics_logger.meters")
                pass
        if attn_stats_dict.keys() > stats_dict.keys():
            stats.append(attn_stats_dict)
            # if args.gpu == 0: pprint(attn_stats_dict)
        if wandb_logger:
            layer = attn_stats_dict.get("layer")
            layer_sub = attn_stats_dict.get("layer_sub", None)
            epoch = attn_stats_dict.get("epoch")

            epoch_wise = {f"Epoch-wise/{k}_layer{layer}": v for k, v in attn_stats_dict.items() if k not in ['cka_feature', 'epoch', 'layer', 'layer_sub']}
            epoch_wise["Epoch-wise/epoch"] = epoch
            wandb_logger._wandb.log(epoch_wise)
            layer_wise = {f"Layer-wise/{k}_epoch{epoch}": v for k, v in attn_stats_dict.items() if k not in ['cka_feature', 'epoch', 'layer', 'layer_sub']}
            layer_wise["Layer-wise/layer"] = layer
            layer_wise["Layer-wise/layer_sub"] = layer_sub if layer_sub is not None else layer
            wandb_logger._wandb.log(layer_wise)
            
        blk_stats_dict = stats_dict.copy()
        for key in blk_metric_keys:
            metric_name = f'{prefix}{key}_layer{layer}'
            # print(metric_name)
            if metric_name in detailed_metrics_logger.meters:
                if key == "blk_acc":
                    blk_stats_dict["layer_sub"] = blk_stats_dict["layer"]
                    key="acc"
                blk_stats_dict.update({
                    f'{prefix}{key}': detailed_metrics_logger.meters[metric_name].global_avg,
                })
            else:
                # print(f"Metric Warning: {metric_name} not found in detailed_metrics_logger.meters")
                pass
        blk_stats_dict["grad_norm"] = parameter_norm[f'module.blocks.{layer}'] if parameter_norm is not None else None
        # if epoch in [49, 99, 149, 199, 249, 299]: blk_stats_dict["cka_feature"] = torch.cat(cka_features[layer], dim=0).cpu().tolist() if layer in cka_features else None
        # print("cka_feature shape:", blk_stats_dict["cka_feature"].shape if blk_stats_dict["cka_feature"] is not None else None)
        if blk_stats_dict.keys() > stats_dict.keys():   # only append if at least one metric was found
            stats.append(blk_stats_dict)
            # if args.gpu == 0: pprint(blk_stats_dict)
        if wandb_logger:
            layer = blk_stats_dict["layer"]
            epoch = blk_stats_dict["epoch"]
            layer_sub = blk_stats_dict.get("layer_sub", None)

            epoch_wise = {f"Epoch-wise/{k}_layer{layer}": v for k, v in blk_stats_dict.items() if k not in ['cka_feature', 'epoch', 'layer', 'layer_sub']}
            epoch_wise["Epoch-wise/epoch"] = epoch
            wandb_logger._wandb.log(epoch_wise)
            layer_wise = {f"Layer-wise/{k}_epoch{epoch}": v for k, v in blk_stats_dict.items() if k not in ['cka_feature', 'epoch', 'layer', 'layer_sub']}
            layer_wise["Layer-wise/layer"] = layer
            layer_wise["Layer-wise/layer_sub"] = layer_sub if layer_sub is not None else layer
            wandb_logger._wandb.log(layer_wise)
    
    # stats = {k: meter.global_avg for k, meter in detailed_metrics_logger.meters.items()}
    if args.gpu == 0:
        print("Averaged stats:", detailed_metrics_logger)
        # print("Layer-wise attention and block prediction accuracies:", stats)

        if args.accuracy_json:
            update_accuracy_json(
                args=args,
                stats=stats,
                model_stats = {
                    "acc1": metric_logger.meters["acc1"].global_avg,
                    "acc5": metric_logger.meters["acc5"].global_avg,
                    "ece": detailed_metrics_logger.meters["ece"].global_avg,
                    "epoch": epoch
                },
                epoch=epoch,
                shuffled_block_order=shuffled_block_order,
                device=device
            )

    return test_stats, stats

@torch.no_grad()
def attention_analyse_final(data_loader, device, args, classes=None, wandb_logger=None):

    shuffled_block_order = None

    model_rand = utils.build_model(args).to(device)
    if args.distributed:
        print("Using distributed data parallel with GPU %d" % args.gpu)
        model_rand = torch.nn.parallel.DistributedDataParallel(model_rand, device_ids=[args.gpu], find_unused_parameters=False)
        model_rand_without_ddp = model_rand.module

    model_analyse(
        model=model_rand_without_ddp,
        data_loader=data_loader,
        device=device,
        epoch=None,
        args=args,
        prefix="rand_",
        wandb_logger=wandb_logger,
    )

    if args.initialize:
        print(f"Loading pr-tuned model from: {args.initialize}")
        if args.analyse_only:
            pr_args = utils.parse_args_for_blocks(args)
        else:
            pr_args = args
        print(f"PR args for loading model: {pr_args}")
        model_pr = utils.build_model(pr_args)
        model_pr.load_state_dict(model_rand_without_ddp.state_dict())
        model_pr, model_pr_without_ddp,  shuffled_block_order = utils.pr_load_model(args.initialize, pr_args, device, model=model_pr)
        model_analyse(
            model=model_pr_without_ddp,
            data_loader=data_loader,
            device=device,
            epoch=None,
            args=args,
            prefix="pr_",
            shuffled_block_order=shuffled_block_order,
            wandb_logger=wandb_logger
        )

    if args.output_dir:
        if args.output_dir.endswith(".pth"):
            ft_path = args.output_dir
        else:
            ft_path = args.output_dir+f"/checkpoint-{args.epochs-1}.pth"
        print(f"Loading fine-tuned model from: {ft_path}")
        model = utils.build_model(args)
        model.load_state_dict(model_rand_without_ddp.state_dict())
        model, model_without_ddp = utils.ft_load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=model)
        model_analyse(
            model=model_without_ddp,
            data_loader=data_loader,
            device=device,
            epoch=None,
            args=args,
            prefix="",
            shuffled_block_order=shuffled_block_order,
            wandb_logger=wandb_logger
        )
    
def update_accuracy_json(args, stats, model_stats, epoch, shuffled_block_order, device):
    if args.analyse_only:    # loading form JSON file
        notes = f"{args.pr_notes}"
    else:
        notes = f"{args.pr_notes} f{str(args.freeze_blocks)} s{str(args.skip_load_blocks)} d{str(args.delete_blocks)} r{str(args.random_blocks)} sba{str(args.skip_load_block_attributes)} fba{str(args.freeze_block_attributes)}"
        if args.shuffle_load:
            notes += f" shuffle hb{str(args.hold_back_blocks)}"
        if args.skip_norm:
            notes += f" skip_norm"
        if args.custom_pr_load:
            notes += f" pr[{str(args.custom_pr_load)}]"
    print(f"Notes for JSON entry: {notes}")
    try:
        with open(args.accuracy_json, "r") as f:
            path_map = json.load(f)
    except FileNotFoundError:
        path_map = []

    block_index = None
    for si in range(len(path_map)):
        if path_map[si]["procedural_data"] == args.procedural_data and \
        path_map[si]["procedural_order"] == args.procedural_order and \
        path_map[si]["notes"] == notes and \
        path_map[si].get("model", args.model) == args.model and \
        path_map[si].get("dataset", args.data_set) == args.data_set:
            block_index = si
    if block_index is None:
        path_map.append({
            "procedural_data": args.procedural_data,
            "procedural_order": args.procedural_order,
            "notes": notes,
            "model": args.model,
            "dataset": args.data_set,
            "pr": [
                {
                    "path": args.initialize,
                    "seed": args.pr_seed,

                }
            ],
            "ft": []
        })
        block_index = len(path_map)-1

    ft_index = None
    if 'ft' not in path_map[block_index]:
        path_map[block_index]["ft"] = []
    for fi in range(len(path_map[block_index]["ft"])):
        if path_map[block_index]['ft'][fi]["path"] == args.output_dir and \
        path_map[block_index]['ft'][fi]["seed"] == args.seed:
            ft_index = fi
            break
    if ft_index is None:
        path_map[block_index]['ft'].append({"path": args.output_dir, "seed": args.seed, "stats": []})
        if shuffled_block_order is not None:
            path_map[block_index]['ft'][-1]["shuffled_block_order"] = shuffled_block_order
        ft_index = len(path_map[block_index]['ft'])-1

    path_map[block_index]['ft'][ft_index].update(
        model_stats
    )
    path_map[block_index]['ft'][ft_index].get("stats", []).extend(stats)

    # if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
    with open(args.accuracy_json, "w") as f:
        json.dump(path_map, f, indent=4)
        print(f"Updated {args.accuracy_json} with new layer accuracies.")

@torch.no_grad()
def cka_final(data_loader, device, args, classes=None, wandb_logger=None):

    shuffled_block_order = None

    # pre-ft image
    random_model = utils.build_model(args).to(device)
    if args.distributed:
        print("Using distributed data parallel with GPU %d" % args.gpu)
        random_model = torch.nn.parallel.DistributedDataParallel(random_model, device_ids=[args.gpu], find_unused_parameters=False)
        random_model_without_ddp = random_model.module
    else:
        random_model_without_ddp = random_model

    if args.initialize:
        print(f"Loading pr-tuned model from: {args.initialize}")
        if args.analyse_only:
            pr_args = utils.parse_args_for_blocks(args)
        else:
            pr_args = args
        print(f"PR args for loading model: {pr_args}")
        pre_ft = utils.build_model(pr_args)
        pre_ft.load_state_dict(random_model_without_ddp.state_dict())
        pre_ft, pre_ft_without_ddp, shuffled_block_order = utils.pr_load_model(args.initialize, pr_args, device, model=pre_ft)
    else:
        pre_ft_without_ddp = random_model_without_ddp

    # procedural model
    pr_model = VitProcedural(args).cuda()
    state_for_pr = pre_ft_without_ddp.state_dict()
    del state_for_pr["head.weight"]
    del state_for_pr["head.bias"]
    del state_for_pr["pos_embed"]
    pr_model.model.load_state_dict(state_for_pr, strict=False)
    if args.distributed:
        pr_model = torch.nn.parallel.DistributedDataParallel(pr_model, device_ids=[args.gpu], find_unused_parameters=False)
        pr_model_without_ddp = pr_model.module
    else:
        pr_model_without_ddp = pr_model
    
    # post-ft image
    if args.output_dir:
        if args.output_dir.endswith(".pth"):
            ft_path = args.output_dir
        else:
            ft_path = args.output_dir+f"/checkpoint-{args.epochs-1}.pth"
        print(f"Loading fine-tuned model from: {ft_path}")
        model = utils.build_model(args)
        model.load_state_dict(pre_ft_without_ddp.state_dict())
        model, model_without_ddp = utils.ft_load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=model)
    
    if args.model == "vit_small":
        kdyck_embeddings_path = "kdyck/kdyck_orthogonal_embeddings_vits.pt"
    else:
        raise NotImplementedError(f"Model {args.model} not supported for kdyck embedding loading")

    if "kdyck" in args.procedural_data:
        pr_mask_function = mask_kdyck_dataset
        dataset = KDyckDataset(args)
    else:
        pr_mask_function = mask_repeat_dataset
        dataset = RepeatDataset(args)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True, seed=args.seed,
    )
    pr_loader = DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=int((args.batch_size)*1.5)
    )
    print("*"*20, "CKA - random vs pre-ft", "*"*20)
    cka_calculate(
        model_A=random_model_without_ddp,
        model_B=pre_ft_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args
    )
    print("*"*20, "CKA - pre-ft vs post-ft", "*"*20)
    cka_calculate(
        model_A=pre_ft_without_ddp,
        model_B=model_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args,
        procedural_data_loader=pr_loader,
        pr_model=pr_model_without_ddp,
        mask_function=pr_mask_function
    )

@torch.no_grad()
def cka_compare(data_loader, device, args, classes=None, wandb_logger=None):
    print("Running cka compare")

    shuffled_block_order = None

    # random_model image
    random_model = utils.build_model(args).to(device)
    if args.distributed:
        print("Using distributed data parallel with GPU %d" % args.gpu)
        random_model = torch.nn.parallel.DistributedDataParallel(random_model, device_ids=[args.gpu], find_unused_parameters=False)
        random_model_without_ddp = random_model.module
    else:
        random_model_without_ddp = random_model

    # model A
    if args.output_dir:
        if args.output_dir.endswith(".pth"):
            ft_path = args.output_dir
        else:
            ft_path = args.output_dir+f"/checkpoint-{args.epochs-1}.pth"
        print(f"Loading fine-tuned model from: {ft_path}")
        model = utils.build_model(args)
        model.load_state_dict(random_model_without_ddp.state_dict())
        model, model_without_ddp = utils.ft_load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=model)

    print("*"*20, "CKA - model A vs model A", "*"*20)
    cka_calculate_self(
        model_A=model_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args
    )
    
    # model B
    if args.output_dir_B != "":
        if args.output_dir_B.endswith(".pth"):
            ft_path = args.output_dir_B
        else:
            ft_path = args.output_dir_B+f"/checkpoint-{args.epochs-1}.pth"
        print(f"Loading fine-tuned model from: {ft_path}")
        model_B = utils.build_model(args)
        model_B.load_state_dict(random_model_without_ddp.state_dict())
        model_B, model_B_without_ddp = utils.ft_load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=model_B)
    else:
        return   # only comparing model A with random, skipping model B

    if args.model == "vit_small":
        kdyck_embeddings_path = "kdyck/kdyck_orthogonal_embeddings_vits.pt"
    else:
        raise NotImplementedError(f"Model {args.model} not supported for kdyck embedding loading")

    if "kdyck" in args.procedural_data:
        pr_mask_function = mask_kdyck_dataset
        dataset = KDyckDataset(args)
    else:
        pr_mask_function = mask_repeat_dataset
        dataset = RepeatDataset(args)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True, seed=args.seed,
    )
    pr_loader = DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=int((args.batch_size)*1.5)
    )

    print("*"*20, "CKA - random model vs model A", "*"*20)
    cka_calculate(
        model_A=random_model_without_ddp,
        model_B=model_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args
    )
    print("*"*20, "CKA - random model vs model B", "*"*20)
    cka_calculate(
        model_A=random_model_without_ddp,
        model_B=model_B_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args
    )
    print("*"*20, "CKA - model A vs model B", "*"*20)
    cka_calculate(
        model_A=model_without_ddp,
        model_B=model_B_without_ddp,
        data_loader=data_loader,
        device=device,
        args=args
    )

def cka_calculate_self(model_A, data_loader, device, args, mask_function=None):
    # Procedural data always compared against model A
    print("Calculating self-CKA for model A")

    # Image dataset forward pass
    metric_logger = utils.MetricLogger(delimiter="  ")
    cka_feats_A = {str(lid): [] for lid in np.arange(-1, len(model_A.blocks)-0.5, 0.5)}
    print("Initialized CKA feature storage for layers:", list(cka_feats_A.keys()))

    for batch in metric_logger.log_every(data_loader, 10, "cka-only: "):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.use_amp:
            with torch.amp.autocast('cuda'):
                with utils.HookCollector(model_A) as acts_A:
                    with torch.no_grad():
                        output = model_A(images)

        with torch.no_grad():
            for i in np.arange(-1, len(model_A.blocks), 1.0):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if i == -1.0:
                        cka_feats_A[str(i)].append(acts_A[0]["inp"])
                    else:
                        cka_feats_A[str(i)].append(acts_A[int(i)]['blk'])
                        cka_feats_A[str(i-0.5)].append(acts_A[int(i)]['attn'])

    cka_feats_A = {str(lid): torch.cat(cka_feats_A[str(lid)], dim=0) for lid, _ in cka_feats_A.items()}
    print(cka_feats_A.keys())
    
    N = cka_feats_A["0.0"].shape[0]
    if utils.get_rank()==0:
        for l in np.arange(-1, len(model_A.blocks)-1, 0.5):
            print("\n---------- Layer", l, l+0.5, "----------")
            cka_layer_A_p = cka_feats_A[str(l)][:N, 0, :].reshape(N, -1)
            cka_layer_A = cka_feats_A[str(l+0.5)][:N, 0, :].reshape(N, -1)

            cka_gram_value = gram_cka(cka_layer_A_p, cka_layer_A)
            print(f"CLS : {cka_gram_value:.4f}")

            cka_layer_A_p = cka_feats_A[str(l)][:N, 1:, :].reshape(N, -1)
            cka_layer_A = cka_feats_A[str(l+0.5)][:N, 1:, :].reshape(N, -1)

            cka_gram_value = gram_cka(cka_layer_A_p, cka_layer_A)
            print(f"Feat: {cka_gram_value:.4f}")

    # print(f"GPU {args.gpu} done with CKA computations, entering barrier")
    dist.barrier()
    # print(f"GPU {args.gpu} passed barrier, ending attention_analyse_final")


def cka_calculate(model_A, model_B, data_loader, device, args, procedural_data_loader=None, pr_model=None, mask_function=None):
    # Procedural data always compared against model A

    # Image dataset forward pass
    metric_logger = utils.MetricLogger(delimiter="  ")
    cka_feats_A = {str(lid): [] for lid in np.arange(-1, len(model_A.blocks)-0.5, 0.5)}
    cka_feats_B = {str(lid): [] for lid in np.arange(-1, len(model_B.blocks)-0.5, 0.5)}
    print("Initialized CKA feature storage for layers:", list(cka_feats_A.keys()))
    cka_feats_pr = {str(lid): [] for lid in np.arange(-1, len(pr_model.model.blocks)-0.5, 0.5)} if pr_model is not None else None

    pr_dataset_iter = None
    if procedural_data_loader:
        pr_dataset_iter = iter(procedural_data_loader)
        assert pr_model is not None, "pr_model must be provided if procedural_data_loader is given"

    for batch in metric_logger.log_every(data_loader, 10, "cka-only: "):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # procedural
        if pr_dataset_iter is not None:
            try:
                pr_target = next(pr_dataset_iter)
            except StopIteration:
                pr_dataset_iter = iter(procedural_data_loader)
                pr_target = next(pr_dataset_iter)
            pr_input = mask_function(pr_target).cuda()

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    with utils.HookCollector(pr_model.model) as acts_pr:
                        pr_output = pr_model(pr_input)
            for i in np.arange(-1, len(pr_model.model.blocks), 1.0): 
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        if i == -1.0:
                            cka_feats_pr[str(i)].append(acts_pr[0]["inp"])
                        else:
                            cka_feats_pr[str(i)].append(acts_pr[int(i)]['blk'])
                            cka_feats_pr[str(i-0.5)].append(acts_pr[int(i)]['attn'])

        if args.use_amp:
            with torch.amp.autocast('cuda'):
                with utils.HookCollector(model_A) as acts_A:
                    with torch.no_grad():
                        output = model_A(images)
                with utils.HookCollector(model_B) as acts_B:
                    with torch.no_grad():
                        output = model_B(images)

        with torch.no_grad():
            for i in np.arange(-1, len(model_B.blocks), 1.0):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if i == -1.0:
                        cka_feats_B[str(i)].append(acts_B[0]["inp"])
                        cka_feats_A[str(i)].append(acts_A[0]["inp"])
                    else:
                        cka_feats_B[str(i)].append(acts_B[int(i)]['blk'])
                        cka_feats_A[str(i)].append(acts_A[int(i)]['blk'])
                        cka_feats_B[str(i-0.5)].append(acts_B[int(i)]['attn'])
                        cka_feats_A[str(i-0.5)].append(acts_A[int(i)]['attn'])

    cka_feats_B = {str(lid): torch.cat(cka_feats_B[str(lid)], dim=0) for lid, _ in cka_feats_B.items()}
    cka_feats_A = {str(lid): torch.cat(cka_feats_A[str(lid)], dim=0) for lid, _ in cka_feats_A.items()}
    if cka_feats_pr is not None: cka_feats_pr = {str(lid): torch.cat(cka_feats_pr[str(lid)], dim=0) for lid, _ in cka_feats_pr.items()}
    
    N = cka_feats_B["0.0"].shape[0]
    if utils.get_rank()==0:
        for l in np.arange(-1, len(model_B.blocks)-0.5, 0.5):
            print("\n---------- Layer", l, "----------")
            cka_layer_B = cka_feats_B[str(l)][:N, 0, :].reshape(N, -1)
            cka_layer_A = cka_feats_A[str(l)][:N, 0, :].reshape(N, -1)

            cka_gram_value = gram_cka(cka_layer_B, cka_layer_A)
            print(f"CLS : {cka_gram_value:.4f}")

            cka_layer_B = cka_feats_B[str(l)][:N, 1:, :].reshape(N, -1)
            cka_layer_A = cka_feats_A[str(l)][:N, 1:, :].reshape(N, -1)

            cka_gram_value = gram_cka(cka_layer_B, cka_layer_A)
            print(f"Feat: {cka_gram_value:.4f}")

            if cka_feats_pr is not None:
                cka_layer_pr = cka_feats_pr[str(l)][:N, :, :].reshape(N, -1)
                cka_gram_value_pr = gram_cka(cka_layer_pr, cka_layer_A)
                print(f"Procedural vs Image CKA: {cka_gram_value_pr:.4f}")

    # print(f"GPU {args.gpu} done with CKA computations, entering barrier")
    dist.barrier()
    # print(f"GPU {args.gpu} passed barrier, ending attention_analyse_final")

def gather_tensor(cka_feat, rank, world_size):
    """
    Gather tensors from all ranks onto rank 0.
    cka_feat: local [n_local, d] on current rank
    returns: [n_total, d] on rank 0, None on others
    """
    # First communicate sizes (each rank may have different n_local)
    local_size = torch.tensor([cka_feat.size(0)], device=cka_feat.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=cka_feat.device)
                for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]

    # Pad to max size so all_gather can work with uniform shape
    max_size = max(all_sizes)
    d = cka_feat.size(1)
    padded = torch.zeros(max_size, d, device=cka_feat.device, dtype=cka_feat.dtype)
    padded[:cka_feat.size(0)] = cka_feat

    gathered = [torch.zeros(max_size, d, device=cka_feat.device, dtype=cka_feat.dtype)
                for _ in range(world_size)]
    if rank == 0:
        dist.gather(padded, gather_list=gathered, dst=0)

        # Trim padding from each rank's contribution
        trimmed = [gathered[str(i)][:all_sizes[str(i)]] for i in range(world_size)]
        return torch.cat(trimmed, dim=0)  # [n_total, d]
    return None
    
@torch.no_grad()
def attention_visualise(data_loader, model, device, args=None):
    # criterion = torch.nn.CrossEntropyLoss()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'attention_visualise:'

    layers_to_analyse = range(len(model.blocks))
    # layers_to_analyse = [1]
    patch_size = 14  # 224/16
    num_heads = model.blocks[0].attn.num_heads

    targets = []

    # attn_map_l11_per_head = {i: [] for i in range(-1, num_heads)} if args.kde_l11 else None
    attn_map_l11_per_head_scaled = {i: [] for i in range(-1, num_heads)} if args.kde_l11 else None
    object_attn_map_l11_per_head_scaled = {i: [] for i in range(-1, num_heads)} if args.kde_l11 else None
    non_object_attn_map_l11_per_head_scaled = {i: [] for i in range(-1, num_heads)} if args.kde_l11 else None

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        images = batch[0]
        target = batch[1]
        masks = batch[2]
        patched_masks = batch[3]
        im_names = batch[4]
        print(f"Image shape: {images[0].shape}, Masks shape: {masks[0].shape}, Patched Masks shape: {patched_masks[0].shape}")

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with utils.HookCollector(model) as acts:
            with torch.no_grad():
                _ = model(images)

        layer_wise_blk_logits = {}
        layer_wise_attn_logits = {}
        with torch.no_grad():
            targets.extend(target)

            for i in layers_to_analyse:
                model.eval()
                x = model.norm(acts[i]['blk'])
                x = model.fc_norm(x)
                x = model.head(x) 
                x = F.softmax(x[:, 0, :].squeeze())
                layer_wise_blk_logits[i] = x.argmax(dim=-1)
                # layer_accuracy[i]['blk_pred'].extend(x)

                x = model.norm(acts[i]['attn'])
                x = model.fc_norm(x)
                x = model.head(x) 
                x = F.softmax(x[:, 0, :].squeeze())
                layer_wise_attn_logits[i] = x.argmax(dim=-1)
                # layer_accuracy[i]['attn_pred'].extend(x)

        for si in range(images.shape[0]):
            if args.kde_l11:
                object_mask = patched_masks[si][0].cpu().numpy()
                for hi in range(-1, num_heads):
                    for i, layer_idx in enumerate([11], 1):
                        
                        pi = i

                        attn_map = acts[layer_idx]['attn_map']

                        if hi==-1:
                            cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                        else:
                            cls_attn = attn_map[:, :, 0, 1:][si][hi].cpu().numpy()  # Avg heads [196] → [14,14]

                        cls_attn = cls_attn.reshape(patch_size, patch_size)
                        
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                        attn_map_l11_per_head_scaled[hi].extend(cls_attn.flatten())
                        object_attn_map_l11_per_head_scaled[hi].extend(cls_attn.flatten()[object_mask.flatten()==1])
                        non_object_attn_map_l11_per_head_scaled[hi].extend(cls_attn.flatten()[object_mask.flatten()==0])
    
            elif args.per_head:
                object_mask = patched_masks[si][0].cpu().numpy()
                original_object_mask = masks[si][0].cpu().numpy()

                fig, axs = plt.subplots(num_heads+1, len(layers_to_analyse)+1, figsize=(5 * len(layers_to_analyse) + 1, 10*num_heads))

                axs[0, 0].imshow(utils.denormalize(images[si].cpu().squeeze()).permute(1, 2, 0), alpha=0.8); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                axs[0,0].imshow(original_object_mask, cmap='Reds', alpha=0.8)
                
                
                for hi in range(-1, num_heads):
                    for i, layer_idx in enumerate(layers_to_analyse, 1):
                        
                        pi = i
                        # print()
                        attn_map = acts[layer_idx]['attn_map']

                        if hi==-1:
                            cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                        else:
                            cls_attn = attn_map[:, :, 0, 1:][si][hi].cpu().numpy()  # Avg heads [196] → [14,14]

                        cls_attn = cls_attn.reshape(patch_size, patch_size)
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                        # print(f"Layer {layer_idx} head {hi} attention map shape:", cls_attn.shape, flush=True)
                        
                        # # Resize overlay
                        # cls_resized = F.interpolate(
                        #     torch.tensor(cls_attn[None, None, :, :]), size=(224, 224), mode='bilinear', align_corners=False
                        # ).squeeze().numpy()
                        
                        # Plots
                        im = axs[hi+1, layer_idx+1].imshow(cls_attn, cmap='viridis')
                        _ = axs[hi+1, layer_idx+1].imshow(object_mask, cmap='Reds', alpha=0.4)

                        # add text to plot with colour, not title
                        if hi==-1:
                            # Logit lens
                            pred_blk_label = layer_wise_blk_logits[layer_idx][si].item()
                            pred_attn_label = layer_wise_attn_logits[layer_idx][si].item()
                            axs[hi+1, layer_idx+1].set_title(f'Layer {layer_idx}')
                        axs[hi+1, layer_idx+1].axis('off')

                    if hi!=-1:
                        axs[hi+1, 0].set_axis_off()
                        axs[hi+1, 0].set_visible(False)
                    plt.tight_layout()
                    # plt.colorbar(im, ax=axs[hi+1, :], fraction=0.02, pad=0.01)

            else:
                plt_column = len(layers_to_analyse)//2 + 1
                fig, axs = plt.subplots(2, plt_column, figsize=(5 * (len(layers_to_analyse)//2 + 1), 10))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).numpy().permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                
                for i, layer_idx in enumerate(layers_to_analyse, 1):
                    
                    pi = i
                    # print()
                    attn_map = acts[layer_idx]['attn_map']
                    # print(f"Layer {layer_idx} attention map shape:", attn_map.shape)

                    cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                    cls_attn = cls_attn.reshape(patch_size, patch_size)
                    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                    
                    # # Resize overlay
                    # cls_resized = F.interpolate(
                    #     torch.tensor(cls_attn[None, None, :, :]), size=(224, 224), mode='bilinear', align_corners=False
                    # ).squeeze().numpy()
                    
                    # Plots
                    im = axs[pi//plt_column, pi%plt_column].imshow(cls_attn, cmap='viridis')

                    # Logit lens
                    axs[pi//plt_column, pi%plt_column].set_title(f'Layer {layer_idx}')
                    axs[pi//plt_column, pi%plt_column].axis('off')

                axs[1, plt_column-1].set_axis_off()
                axs[1, plt_column-1].set_visible(False) 
                plt.tight_layout()
            if args.visualise_output_path and args.per_head:
                os.makedirs(args.visualise_output_path, exist_ok=True)
            # if False:
                plt.savefig(args.visualise_output_path+f"/sample_{si}_{target[si]}_{im_names[si]}.png", dpi=300, bbox_inches='tight')
                break
                # print(f"Saved attention visualization for sample {si}")
            # else:
            #     plt.show()
        sys.stdout.flush()

    if args.kde_l11:
        print("|Head\t\t|KS Statistic\t\t|KS p-value\t\t|T-test Statistic\t\t|T-test p-value\t\t|Cohen's d\t\t|")
        for hi in range(-1, num_heads):
            ks_stat, p_value = ks_2samp(object_attn_map_l11_per_head_scaled[hi], non_object_attn_map_l11_per_head_scaled[hi])
            # print(f"KS test for head {hi}: statistic={ks_stat:.4f}, p-value={p_value:.4f}")

            t_stat, p_t = ttest_ind(object_attn_map_l11_per_head_scaled[hi], non_object_attn_map_l11_per_head_scaled[hi], equal_var=False)
            # print(f"T-test for head {hi}: statistic={t_stat:.4f}, p-value={p_t:.4f}")

            d = cohens_d(object_attn_map_l11_per_head_scaled[hi], non_object_attn_map_l11_per_head_scaled[hi])
            # print(f"Cohen's d for head {hi}: d={d:.4f}")

            print(f"|{hi}\t\t|{ks_stat:.4f}\t\t|{p_value:.4f}\t\t|{t_stat:.4f}\t\t|{p_t:.4f}\t\t|{d:.4f}\t\t|")

            # attn_map_l11_per_head[hi] = np.stack(attn_map_l11_per_head[hi], axis=0).flatten()
            # attn_map_l11_per_head_scaled[hi] = np.stack(attn_map_l11_per_head_scaled[hi], axis=0).flatten()
            # object_attn_map_l11_per_head_scaled[hi] = np.stack(object_attn_map_l11_per_head_scaled[hi], axis=0).flatten()
        # single_kde([attn_map_l11_per_head[hi] for hi in range(-1, num_heads)], [f'Head avg', *[f'Head {hi}' for hi in range(num_heads)]], f'Attention Map Distribution', args.visualise_output_path+f"/kde_pr_attn.png")
        os.makedirs(args.visualise_output_path, exist_ok=True)
        single_kde([np.array(attn_map_l11_per_head_scaled[hi]) for hi in range(-1, num_heads)],  [f'Head avg', *[f'Head {hi}' for hi in range(num_heads)]], f'Attention Map Distribution - scaled', args.visualise_output_path+f"/kde_pr_attn_scaled.png")
        single_kde([np.array(object_attn_map_l11_per_head_scaled[hi]) for hi in range(-1, num_heads)],  [f'Head avg', *[f'Head {hi}' for hi in range(num_heads)]], f'Object Attention Map Distribution - scaled', args.visualise_output_path+f"/kde_pr_attn_object_scaled.png")
        single_kde([np.array(non_object_attn_map_l11_per_head_scaled[hi]) for hi in range(-1, num_heads)],  [f'Head avg', *[f'Head {hi}' for hi in range(num_heads)]], f'Object Attention Map Distribution - scaled', args.visualise_output_path+f"/kde_pr_attn_non_object_scaled.png")
    sys.stdout.flush()

def single_kde(values, labels, title, save_path):
    plt.figure(figsize=(12, 6))
    for value, label in zip(values, labels):
        # print(f"Value shape for label {label}:", value.shape, flush=True)
        kde = gaussian_kde(value)
        x_range = np.linspace(value.min(), value.max(), 100)
        plt.plot(x_range, kde(x_range), label=label)
    plt.legend()
    # plt.xlim(-3, 3)
    plt.ylim(0, 8)
    plt.title(f"KDE {title}")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

def cohens_d(x1, x2):
    return (np.mean(x1) - np.mean(x2)) / np.sqrt((np.var(x1) + np.var(x2)) / 2)

@torch.no_grad()
def attention_residual_analysis(data_loader, model, device, args=None):
    header = 'attention_residual_analysis:'

    # layers_to_analyse = range(len(model.blocks))
    layers_to_analyse = [11]
    patch_size = 14  # 224/16
    num_heads = model.blocks[0].attn.num_heads

    all_attn_in = {i:[] for i in layers_to_analyse}
    all_attn_out = {i:[] for i in layers_to_analyse}
    all_attn_res_out = {i:[] for i in layers_to_analyse}
    all_attn_delta = {i:[] for i in layers_to_analyse}

    model.eval()
    for batch in data_loader:
        images = batch[0]
        target = batch[1]
        masks = batch[2]
        patched_masks = batch[3]
        im_names = batch[4]
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with utils.HookCollector(model) as acts:
            with torch.no_grad():
                _ = model(images)

        with torch.no_grad():
            for i in layers_to_analyse:
                model.eval()
                attn_in = acts[i]['inp']
                attn_out = acts[i]['attn_out']
                attn_res_out = acts[i]['attn']
                attn_delta = attn_res_out - attn_in

                all_attn_in[i].append(attn_in.cpu())
                all_attn_out[i].append(attn_out.cpu())
                all_attn_res_out[i].append(attn_res_out.cpu())
                all_attn_delta[i].append(attn_delta.cpu())

    stats = {i:{} for i in layers_to_analyse}
    stats_aggregated = {}
    for i in layers_to_analyse:
        all_attn_in[i] = torch.cat(all_attn_in[i], dim=0)
        all_attn_out[i] = torch.cat(all_attn_out[i], dim=0)
        all_attn_res_out[i] = torch.cat(all_attn_res_out[i], dim=0)
        all_attn_delta[i] = torch.cat(all_attn_delta[i], dim=0)

        rin_norm = torch.norm(all_attn_in[i], dim=-1)
        rout_norm = torch.norm(all_attn_res_out[i], dim=-1)
        norm_ratio = rout_norm / (rin_norm + 1e-8)
        delta_norm = torch.norm(all_attn_delta[i], dim=-1)
        atnn_out_norm = torch.norm(all_attn_out[i], dim=-1)

        rin_unit = all_attn_in[i] / (rin_norm.unsqueeze(-1) + 1e-8)
        rout_unit = all_attn_res_out[i] / (rout_norm.unsqueeze(-1) + 1e-8)
        cosine_rin_rout = F.cosine_similarity(rin_unit, rout_unit, dim=-1)
        # save all_attn_in and all_attn_res_out to disk for layer i
        all_attn_in[i] = all_attn_in[i].cpu()
        all_attn_res_out[i] = all_attn_res_out[i].cpu()
        os.makedirs(args.attention_residual_stats_path, exist_ok=True)
        torch.save(all_attn_in[i], os.path.join(args.attention_residual_stats_path, f"attn_in_layer_{i}.pt"))
        print(f"Saved attn_in for layer {i} with shape {all_attn_in[i].shape} to {os.path.join(args.attention_residual_stats_path, f'attn_in_layer_{i}.pt')}")
        torch.save(all_attn_res_out[i], os.path.join(args.attention_residual_stats_path, f"attn_res_out_layer_{i}.pt"))
        print(f"Saved attn_res_out for layer {i} with shape {all_attn_res_out[i].shape} to {os.path.join(args.attention_residual_stats_path, f'attn_res_out_layer_{i}.pt')}")
        
        cka_rin_rout = gram_cka(all_attn_in[i].reshape(all_attn_in[i].shape[0], -1), all_attn_res_out[i].reshape(all_attn_res_out[i].shape[0], -1))

        stats[i] = {
            "rin_norm": rin_norm.tolist(),
            "rout_norm": rout_norm.tolist(),
            "norm_ratio": norm_ratio.tolist(),
            "delta_norm": delta_norm.tolist(),
            "atnn_out_norm": atnn_out_norm.tolist(),
            "cosine_rin_rout": cosine_rin_rout.tolist(),
            "cka_rin_rout": cka_rin_rout.item()
        }
    
        stats_aggregated[i] = {
            "rin_norm_mean": rin_norm.mean().item(),
            "rin_norm_std": rin_norm.std().item(),
            "rout_norm_mean": rout_norm.mean().item(),
            "rout_norm_std": rout_norm.std().item(),
            "norm_ratio_mean": norm_ratio.mean().item(),
            "norm_ratio_std": norm_ratio.std().item(),
            "delta_norm_mean": delta_norm.mean().item(),
            "delta_norm_std": delta_norm.std().item(),
            "attn_out_norm_mean": atnn_out_norm.mean().item(),
            "attn_out_norm_std": atnn_out_norm.std().item(),
            "cosine_rin_rout_mean": cosine_rin_rout.mean().item(),
            "cosine_rin_rout_std": cosine_rin_rout.std().item(),
            "cka_rin_rout": cka_rin_rout.item()
        }
    
    stats.update({"aggregated": stats_aggregated})
    pprint(stats_aggregated)
    sys.stdout.flush()

    json.dump(stats, open(args.attention_residual_stats_path+"/attn_res_stats.json", "w"), indent=4)
