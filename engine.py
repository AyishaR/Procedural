# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from pprint import pprint
import torch.nn.functional as F
import utils
import json
from matplotlib import pyplot as plt
from collections import defaultdict

from metrics.ddp_multiclassECE import DDPMulticlassECE

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
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
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
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
