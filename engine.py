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

@torch.no_grad()
def attention_analyse(data_loader, model, device, args=None, classes=None, per_head=True):
    # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'attention_analyse:'

    layers_to_analyse = range(len(model.blocks))
    # layers_to_analyse = [1]
    patch_size = 14  # 224/16
    num_heads = model.blocks[0].attn.num_heads

    title_colour = {1: 'green', 0: 'red'}  # Correct: green, Incorrect: red

    layer_accuracy = {layer_idx: {'blk_pred': [], 'attn_pred': []} for layer_idx in layers_to_analyse}
    targets = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

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
                x = model.norm(acts[i]['resid'])
                x = model.fc_norm(x)
                x = model.head(x) 
                x = F.softmax(x[:, 0, :].squeeze())
                layer_wise_blk_logits[i] = x.argmax(dim=-1)
                layer_accuracy[i]['blk_pred'].extend(x)

                x = model.norm(acts[i]['attn'])
                x = model.fc_norm(x)
                x = model.head(x) 
                x = F.softmax(x[:, 0, :].squeeze())
                layer_wise_attn_logits[i] = x.argmax(dim=-1)
                layer_accuracy[i]['attn_pred'].extend(x)

                # print(targets, torch.stack(layer_accuracy[i]['blk_pred']), torch.stack(layer_accuracy[i]['attn_pred']))
                # print(len(targets), torch.stack(layer_accuracy[i]['blk_pred']).shape, torch.stack(layer_accuracy[i]['attn_pred']).shape)
                # input()
        
        if not args.visualise: continue

        for si in range(2):
            if per_head:
                fig, axs = plt.subplots(num_heads+1, len(layers_to_analyse)+1, figsize=(5 * len(layers_to_analyse) + 1, 10*num_heads))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                class_label = target[si].item()
                if classes is not None: class_label = f"Class: {classes[target[si].item()]}"
                axs[0, 0].set_title(class_label)
                
                # rollout = torch.eye(1 + patch_size**2)[None, None, :, 1:].to(images.device)  # [1,1,197,197]
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
                            axs[hi+1, layer_idx+1].text(0.5, 16, f'Blk Pred: {pred_blk_label}', {'color': title_colour[int(layer_wise_blk_logits[layer_idx][si].item() == target[si].item())]})
                            # axs[hi+1, layer_idx+1].text(0.5, 18, f'Attn Pred: {pred_attn_label}', {'color': title_colour[int(layer_wise_attn_logits[layer_idx][si].item() == target[si].item())]})
                            axs[hi+1, layer_idx+1].set_title(f'Layer {layer_idx}')
                        axs[hi+1, layer_idx+1].axis('off')

                    if hi!=-1:
                        axs[hi+1, 0].set_axis_off()
                        axs[hi+1, 0].set_visible(False)
                    plt.tight_layout()

            else:
                plt_column = len(layers_to_analyse)//2 + 1
                fig, axs = plt.subplots(2, plt_column, figsize=(5 * (len(layers_to_analyse)//2 + 1), 10))

                axs[0, 0].imshow(utils.denormalize(images[si].squeeze()).permute(1, 2, 0)); axs[0, 0].set_title('Input'); axs[0, 0].axis('off')
                class_label = target[si].item()
                if classes is not None: class_label = f"Class: {classes[target[si].item()]}"
                axs[0, 0].set_title(class_label)
                
                # rollout = torch.eye(1 + patch_size**2)[None, None, :, 1:].to(images.device)  # [1,1,197,197]
                
                for i, layer_idx in enumerate(layers_to_analyse, 1):
                    
                    pi = i
                    # print()
                    attn_map = acts[layer_idx]['attn_map']
                    # print(f"Layer {layer_idx} attention map shape:", attn_map.shape)

                    cls_attn = attn_map[:, :, 0, 1:][si].mean(0).cpu().numpy()  # Avg heads [196] → [14,14]
                    cls_attn = cls_attn.reshape(patch_size, patch_size)
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
            if args.visualise_output_path:
            # if False:
                plt.savefig(args.visualise_output_path+f"/sample_{si}.png", dpi=300, bbox_inches='tight')
                # print(f"Saved attention visualization for sample {si}")
            else:
                plt.show()
        
    targets = torch.stack(targets)
    accs = []
    for i in layers_to_analyse:
        blk_acc = accuracy(torch.stack(layer_accuracy[i]['blk_pred']), targets)
        accs.append(round(blk_acc[0].item(), 2))
        # attn_acc = accuracy(torch.stack(layer_accuracy[i]['attn_pred']), targets)
        # print(f"Layer {i} block-based prediction accuracy: {round(blk_acc[0].item(), 2)}%")
        # print(f"Layer {i} attn-based prediction accuracy: {attn_acc.item()}%")
    
    if args.attention_analyse:
        ft_path = args.initialize
        pr_path = ""
    else:
        ft_path = args.output_dir+f"/checkpoint-{args.epochs-1}.pth"
        pr_path = args.initialize
    if args.accuracy_json:
        with open(args.accuracy_json, "r") as f:
            path_map = json.load(f)
        found = False
        for si in range(len(path_map)):
            if path_map[si]["procedural_data"] == args.procedural_data and \
            path_map[si]["procedural_order"] == args.procedural_order and \
            path_map[si]["notes"] == args.pr_notes:
                for fi in range(len(path_map[si]["ft"])):
                    if path_map[si]['ft'][fi]["path"] == ft_path:
                        path_map[si]['ft'][fi]["layers_accuracy"] = accs
                        found = True
                        break
                if not found:
                    path_map[si]['ft'].append({"path": ft_path, "layers_accuracy": accs, "seed": args.seed})
                    found = True
                break
        if not found:
            path_map.append({
                "procedural_data": args.procedural_data,
                "procedural_order": args.procedural_order,
                "notes": args.notes,
                "pr": [
                    {
                        "path": pr_path,
                        "seed": args.pr_seed
                    }
                ],
                "ft": [{"path": ft_path, "layers_accuracy": accs, "seed": args.seed}]
            })

        with open(args.accuracy_json, "w") as f:
            json.dump(path_map, f, indent=4)
            pprint(path_map)
            print(f"Updated {args.accuracy_json} with new layer accuracies.")

    print("layers_accuracy", accs)
    with open("layer_accuracy.txt", "a") as f:
        f.write(f"\nInstance: {args.initialize}\n{str(accs)}")

    return accs