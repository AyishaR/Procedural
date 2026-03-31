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

from metrics.ddp_multiclassECE import DDPMulticlassECE
from metrics.ddp_multiclassBrier import DDPMulticlassBrier

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
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
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

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

@torch.no_grad()
def attention_analyse(data_loader, device, args=None, classes=None, wandb_logger=None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'attention_analyse:'

    # metrics
    ece_metric = DDPMulticlassECE(n_bins=15, device=device)
    ece_metric.reset()
    brier_metric = DDPMulticlassBrier(device=device)
    brier_metric.reset()

    title_colour = {1: 'green', 0: 'red'}  # Correct: green, Incorrect: red
    plotted=0

    if args.output_dir=="":
        ft_path = args.initialize
        pr_path = ""
    else:
        ft_path = args.output_dir+f"/checkpoint-299.pth"
        pr_path = args.initialize

    if args.visualise and args.per_stage:
        model_rand = utils.load_model("", args, device)
        if pr_path:
            print(f"Loading pr-tuned model from: {pr_path}")
            pr_args = utils.parse_args_for_blocks(args)
            print(f"PR args for loading model: {pr_args}")
            model_pr = utils.build_model(pr_args)
            model_pr.load_state_dict(model_rand.state_dict())
            model_pr = utils.pr_load_model(pr_path, pr_args, device, model=model_pr)
        else:
            model_pr = None
        print(f"Loading fine-tuned model from: {ft_path}")
        model = utils.build_model(args)
        model.load_state_dict(model_rand.state_dict())
        model = utils.load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=model)
    else:
        model_rand = None
        model_pr = None
        print(f"Loading fine-tuned model from: {ft_path}")
        model = utils.load_model(ft_path, args, device, delete_blocks=args.delete_blocks, model=utils.build_model(args))

    layers_to_analyse = range(len(model.blocks))
    # layers_to_analyse = [1]
    patch_count = 14  # 224/16
    num_heads = model.blocks[0].attn.num_heads

    # switch to evaluation mode
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
                        acc1, acc5 = accuracy(output, target, topk=(1, 5))
                        metric_logger.meters[f'final_acc'].update(acc1.item(), n=images.shape[0])
                        print(f"Batch final accuracy: {acc1.item():.3f}%")

                if model_rand is not None:
                    with utils.HookCollector(model_rand) as rand_acts:
                        with torch.no_grad():
                            _ = model_rand(images)
                else:
                    rand_acts = None
                if model_pr is not None:
                    with utils.HookCollector(model_pr) as pr_acts:
                        with torch.no_grad():
                            _ = model_pr(images)
                else:
                    pr_acts = None

        layer_wise_blk_logits, layer_wise_attn_logits = {}, {}
        rand_blk_logits, rand_attn_logits = {}, {}
        pr_blk_logits, pr_attn_logits = {}, {}

        with torch.no_grad():

            for i in layers_to_analyse:
                model.eval()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x = model.norm(acts[i]['attn'])
                    x = model.fc_norm(x)
                    x = model.head(x) 
                    x = F.softmax(x[:, 0, :].squeeze(), dim=-1)
                layer_wise_attn_logits[i] = x.argmax(dim=-1)
                
                attn_pred = accuracy(x, target)
                metric_logger.meters[f'attn_{i}'].update(attn_pred[0].item(), n=images.shape[0])

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x = model.norm(acts[i]['blk'])
                    x = model.fc_norm(x)
                    x = model.head(x) 
                    x = F.softmax(x[:, 0, :].squeeze(), dim=-1)
                layer_wise_blk_logits[i] = x.argmax(dim=-1)

                if args.detailed_metrics:
                    attn_map = acts[i]['attn_map']
                    attn_norm = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
                    entropy = -(attn_norm * torch.log(attn_norm + 1e-8)).sum(dim=-1)  # [batch, heads, tokens]
                    entropy_mean = entropy.mean(dim=(0,2))
                    for hi in range(num_heads):
                        metric_logger.meters[f'attn_entropy_head{hi}_layer{i}'].update(entropy_mean[hi].item(), n=images.shape[0])
                    metric_logger.meters[f'attn_entropy_layer{i}'].update(entropy_mean.mean().item(), n=images.shape[0])

                    cls_attn = attn_map[:, :, 0, 1:]
                    cls_entropy = -(cls_attn * torch.log(cls_attn + 1e-8)).sum(dim=-1)  # [batch, heads, tokens]
                    cls_entropy_mean = cls_entropy.mean(dim=(0))
                    for hi in range(num_heads):
                        metric_logger.meters[f'cls_attn_entropy_head{hi}_layer{i}'].update(cls_entropy_mean[hi].item(), n=images.shape[0])
                    metric_logger.meters[f'cls_attn_entropy_layer{i}'].update(cls_entropy_mean.mean().item(), n=images.shape[0])

                    attn_pp = attn_map[..., 1:, 1:]
                    P = attn_pp.shape[-1]
                    length = int(P ** 0.5)

                    D = compute_distance_matrix(length, attn_pp.device)
                    D = D.view(1, 1, P, P)
                    mad_per_token = (attn_pp * D).sum(dim=-1)
                    mad_per_head = mad_per_token.mean(dim=(0, 2))
                    for hi in range(num_heads):
                        metric_logger.meters[f'attn_mad_head{hi}_layer{i}'].update(mad_per_head[hi].item(), n=images.shape[0])
                    metric_logger.meters[f'attn_mad_layer{i}'].update(mad_per_head.mean().item(), n=images.shape[0])

                    cls_attn_pp = attn_map[:, :, 0, 1:]
                    print(f"Layer {i} cls_attn_pp shape:", cls_attn_pp.shape)
                    print(f"Layer {i} distance matrix D shape:", D[0, :, 0, :].shape)
                    cls_mad_per_token = (cls_attn_pp * D[0, :, 0, :]).sum(dim=-1)
                    cls_mad_per_head = cls_mad_per_token.mean(dim=(0))
                    for hi in range(num_heads):
                        metric_logger.meters[f'cls_attn_mad_head{hi}_layer{i}'].update(cls_mad_per_head[hi].item(), n=images.shape[0])
                    metric_logger.meters[f'cls_attn_mad_layer{i}'].update(cls_mad_per_head.mean().item(), n=images.shape[0])

                ece_metric.update(x, target)
                brier_metric.update(x, target)

                blk_pred = accuracy(x, target)
                metric_logger.meters[f'blk_{i}'].update(blk_pred[0].item(), n=images.shape[0])

                if rand_acts is not None:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        x_rand = model.norm(rand_acts[i]['attn'])
                        x_rand = model.fc_norm(x_rand)
                        x_rand = model.head(x_rand) 
                        x_rand = F.softmax(x_rand[:, 0, :].squeeze(), dim=-1)
                        rand_attn_logits[i] = x_rand.argmax(dim=-1)
                        if args.per_stage_metrics:
                            rand_attn_pred = accuracy(x_rand, target)
                            metric_logger.meters[f'rand_attn_{i}'].update(rand_attn_pred[0].item(), n=int(1.5 * args.batch_size))

                        x_rand_blk = model.norm(rand_acts[i]['blk'])
                        x_rand_blk = model.fc_norm(x_rand_blk)
                        x_rand_blk = model.head(x_rand_blk) 
                        x_rand_blk = F.softmax(x_rand_blk[:, 0, :].squeeze(), dim=-1)
                        rand_blk_logits[i] = x_rand_blk.argmax(dim=-1)
                        if args.per_stage_metrics:
                            rand_blk_pred = accuracy(x_rand_blk, target)
                            metric_logger.meters[f'rand_blk_{i}'].update(rand_blk_pred[0].item(), n=int(1.5 * args.batch_size))
                if pr_acts is not None:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        x_pr = model.norm(pr_acts[i]['attn'])
                        x_pr = model.fc_norm(x_pr)
                        x_pr = model.head(x_pr) 
                        x_pr = F.softmax(x_pr[:, 0, :].squeeze(), dim=-1)
                        pr_attn_logits[i] = x_pr.argmax(dim=-1)
                        if args.per_stage_metrics:
                            pr_attn_pred = accuracy(x_pr, target)
                            metric_logger.meters[f'pr_attn_{i}'].update(pr_attn_pred[0].item(), n=int(1.5 * args.batch_size))

                        x_pr_blk = model.norm(pr_acts[i]['blk'])
                        x_pr_blk = model.fc_norm(x_pr_blk)
                        x_pr_blk = model.head(x_pr_blk) 
                        x_pr_blk = F.softmax(x_pr_blk[:, 0, :].squeeze(), dim=-1)
                        pr_blk_logits[i] = x_pr_blk.argmax(dim=-1)
                        if args.per_stage_metrics:
                            pr_blk_pred = accuracy(x_pr_blk, target)
                            metric_logger.meters[f'pr_blk_{i}'].update(pr_blk_pred[0].item(), n=int(1.5 * args.batch_size))

        if not args.visualise: continue

        for si in range(args.plot_count):
            if plotted >= args.plot_count: break
            if si >= len(target): break
            plotted+=1

            images = images.cpu()
            target = target.cpu()

            if args.per_head:
                fig, axs = plt.subplots(num_heads+1, len(layers_to_analyse)+1, figsize=(5 * len(layers_to_analyse) + 1, 10*num_heads))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                class_label = target[si].item()
                if classes is not None: class_label = f"Class: {classes[target[si].item()]}"
                axs[0, 0].set_title(class_label)
                
                # rollout = torch.eye(1 + patch_count**2)[None, None, :, 1:].to(images.device)  # [1,1,197,197]
                for hi in range(-1, num_heads):
                    for i, layer_idx in enumerate(layers_to_analyse, 1):
                        
                        pi = i
                        # print()
                        attn_map = acts[layer_idx]['attn_map']

                        if hi==-1:
                            cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                        else:
                            cls_attn = attn_map[:, :, 0, 1:][si][hi].cpu().numpy()  # Avg heads [196] → [14,14]

                        cls_attn = cls_attn.reshape(patch_count, patch_count)
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                        
                        # Resize overlay
                        cls_resized = F.interpolate(
                            torch.tensor(cls_attn[None, None, :, :]), size=(224, 224), mode='bilinear', align_corners=False
                        ).squeeze().numpy()
                        
                        # Plots
                        im = axs[hi+1, layer_idx+1].imshow(cls_attn, cmap='viridis')

                        
                        # add text to plot with colour, not title
                        if hi==-1:
                            # Logit lens
                            pred_blk_label = layer_wise_blk_logits[layer_idx][si].item()
                            if classes is not None: pred_blk_label = classes[pred_blk_label]
                            pred_attn_label = layer_wise_attn_logits[layer_idx][si].item()
                            if classes is not None: pred_attn_label = classes[pred_attn_label]
                            axs[hi+1, layer_idx+1].text(0.5, 16, f'{pred_blk_label}', {'color': title_colour[int(layer_wise_blk_logits[layer_idx][si].item() == target[si].item())]})
                            # axs[hi+1, layer_idx+1].text(0.5, 18, f'Attn Pred: {pred_attn_label}', {'color': title_colour[int(layer_wise_attn_logits[layer_idx][si].item() == target[si].item())]})
                            axs[hi+1, layer_idx+1].set_title(f'Layer {layer_idx}')
                        axs[hi+1, layer_idx+1].axis('off')

                    if hi!=-1:
                        axs[hi+1, 0].set_axis_off()
                        axs[hi+1, 0].set_visible(False)
                    plt.tight_layout()
            elif args.per_stage:
                plt_column = len(layers_to_analyse)//2 + 1
                fig, axs = plt.subplots(2*3, plt_column, figsize=(5 * (len(layers_to_analyse)//2 + 1), 10*3))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                class_label = target[si].item()
                if classes is not None: class_label = f"Class: {classes[target[si].item()]}"
                axs[0, 0].set_title(f"Fine-tuned\n{class_label}")
                
                # rollout = torch.eye(1 + patch_count**2)[None, None, :, 1:].to(images.device)  # [1,1,197,197]
                
                for model_idx, (model_name, model_acts, model_blk_logits, model_attn_logits) in enumerate(zip(["Fine-tuned", "Procedural", "Random init"], [acts, pr_acts, rand_acts], [layer_wise_blk_logits, pr_blk_logits, rand_blk_logits], [layer_wise_attn_logits, pr_attn_logits, rand_attn_logits])):
                    if model_acts is None: continue
                    for i, layer_idx in enumerate(layers_to_analyse, 1):
                        
                        pi = i
                        print(i, layer_idx, "plt idx:", (model_idx*2)+(pi//plt_column), pi%plt_column)
                        attn_map = model_acts[layer_idx]['attn_map']
                        # print(f"Layer {layer_idx} attention map shape:", attn_map.shape)

                        cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                        cls_attn = cls_attn.reshape(patch_count, patch_count)
                        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                        
                        # Resize overlay
                        cls_resized = F.interpolate(
                            torch.tensor(cls_attn[None, None, :, :]), size=(224, 224), mode='bilinear', align_corners=False
                        ).squeeze().numpy()
                        
                        # Plots
                        im = axs[(model_idx*2)+(pi//plt_column), pi%plt_column].imshow(cls_attn, cmap='viridis')

                        # Logit lens
                        # if model_idx==0:
                        pred_blk_label = model_blk_logits[layer_idx][si].item()
                        if classes is not None: pred_blk_label = classes[pred_blk_label]
                        pred_attn_label = model_attn_logits[layer_idx][si].item()
                        if classes is not None: pred_attn_label = classes[pred_attn_label]
                        # add text to plot with colour, not title
                        axs[(model_idx*2)+(pi//plt_column), pi%plt_column].text(0.5, 14, f'{pred_blk_label}', {'color': title_colour[int(model_blk_logits[layer_idx][si].item() == target[si].item())]})
                        # axs[pi//plt_column, pi%plt_column].text(0.5, 18, f'Attn Pred: {pred_attn_label}', {'color': title_colour[int(layer_wise_attn_logits[layer_idx][si].item() == target[si].item())]})
                        axs[(model_idx*2)+(pi//plt_column), pi%plt_column].set_title(f'Layer {layer_idx}')
                        axs[(model_idx*2)+(pi//plt_column), pi%plt_column].axis('off')

                    axs[(model_idx*2)+1, plt_column-1].set_axis_off()
                    axs[(model_idx*2)+1, plt_column-1].set_visible(False) 
                    if model_idx!=0:
                        axs[(model_idx*2), 0].set_axis_off()
                        # axs[(model_idx*2), 0].set_visible(False)
                        # axs[(model_idx*2), 0].set_title(model_name)
                        axs[(model_idx*2), 0].text(0.3, 0.5, f"{model_name}", ha='left', va='top', transform=axs[(model_idx*2), 0].transAxes)
                plt.tight_layout()
            else:
                plt_column = len(layers_to_analyse)//2 + 1
                fig, axs = plt.subplots(2, plt_column, figsize=(5 * (len(layers_to_analyse)//2 + 1), 10))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                class_label = target[si].item()
                if classes is not None: class_label = f"Class: {classes[target[si].item()]}"
                axs[0, 0].set_title(class_label)
                
                # rollout = torch.eye(1 + patch_count**2)[None, None, :, 1:].to(images.device)  # [1,1,197,197]
                
                for i, layer_idx in enumerate(layers_to_analyse, 1):
                    
                    pi = i
                    # print()
                    attn_map = acts[layer_idx]['attn_map']
                    # print(f"Layer {layer_idx} attention map shape:", attn_map.shape)

                    cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                    cls_attn = cls_attn.reshape(patch_count, patch_count)
                    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
                    
                    # Resize overlay
                    cls_resized = F.interpolate(
                        torch.tensor(cls_attn[None, None, :, :]), size=(224, 224), mode='bilinear', align_corners=False
                    ).squeeze().numpy()
                    
                    # Plots
                    im = axs[pi//plt_column, pi%plt_column].imshow(cls_attn, cmap='viridis')

                    # Logit lens
                    pred_blk_label = layer_wise_blk_logits[layer_idx][si].item()
                    if classes is not None: pred_blk_label = classes[pred_blk_label]
                    pred_attn_label = layer_wise_attn_logits[layer_idx][si].item()
                    if classes is not None: pred_attn_label = classes[pred_attn_label]
                    # add text to plot with colour, not title
                    axs[pi//plt_column, pi%plt_column].text(0.5, 16, f'{pred_blk_label}', {'color': title_colour[int(layer_wise_blk_logits[layer_idx][si].item() == target[si].item())]})
                    # axs[pi//plt_column, pi%plt_column].text(0.5, 18, f'Attn Pred: {pred_attn_label}', {'color': title_colour[int(layer_wise_attn_logits[layer_idx][si].item() == target[si].item())]})
                    axs[pi//plt_column, pi%plt_column].set_title(f'Layer {layer_idx}')
                    axs[pi//plt_column, pi%plt_column].axis('off')

                axs[1, plt_column-1].set_axis_off()
                axs[1, plt_column-1].set_visible(False) 
                plt.tight_layout()

            if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
                if args.visualise_output_path:
                # if False:
                    plt.savefig(args.visualise_output_path+f"/sample_{si}_s{args.seed}.png", dpi=300, bbox_inches='tight')
                    print(f"Saved attention visualization for sample {si}")
                else:
                    plt.show()
        
    metric_logger.synchronize_between_processes()

    ece = ece_metric.compute()
    if wandb_logger:
        wandb_logger.log_epoch_metrics({"epoch": -1, "test_ece": ece})
        pass
    if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
        print("ECE:", ece)
        metric_logger.meters["ece"].update(ece, n=1)

    
    brier = brier_metric.compute()
    if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
        print("Brier:", brier)
        metric_logger.meters["brier"].update(brier, n=1)

    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Layer-wise attention and block prediction accuracies:", stats)

    if args.accuracy_json:
        if args.attention_analyse:    # loading form JSON file
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
        found = False
        for si in range(len(path_map)):
            if path_map[si]["procedural_data"] == args.procedural_data and \
            path_map[si]["procedural_order"] == args.procedural_order and \
            path_map[si]["notes"] == notes:
                for fi in range(len(path_map[si]["ft"])):
                    if path_map[si]['ft'][fi]["path"] == ft_path:
                        path_map[si]['ft'][fi]["layers_accuracy"] = stats
                        found = True
                        break
                if not found:
                    path_map[si]['ft'].append({"path": ft_path, "layers_accuracy": stats, "seed": args.seed})
                    found = True
                break
        if not found:
            print(f"No existing entry found in {args.accuracy_json}. Adding new entry.")
            path_map.append({
                "procedural_data": args.procedural_data,
                "procedural_order": args.procedural_order,
                "notes": notes,
                "pr": [
                    {
                        "path": pr_path,
                        "seed": args.pr_seed
                    }
                ],
                "ft": [{"path": ft_path, "layers_accuracy": stats, "seed": args.seed}]
            })

        if device == torch.device('cpu') or (device.type == 'cuda' and torch.distributed.get_rank() == 0):
            with open(args.accuracy_json, "w") as f:
                json.dump(path_map, f, indent=4)
                pprint(path_map)
                print(f"Updated {args.accuracy_json} with new layer accuracies.")

    return stats
