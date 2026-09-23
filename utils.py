import os
import math
import time
from collections import defaultdict, deque, OrderedDict
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import random
import torch
from row_lr_mask import lr_spec_of, assert_same_lr_spec
from torch import inf
from timm.models import create_model
import torch.distributed as dist

from functools import partial
from collections import defaultdict
# from tensorboardX import SummaryWriter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# class TensorboardLogger(object):
#     def __init__(self, log_dir):
#         self.writer = SummaryWriter(logdir=log_dir)
#         self.step = 0

#     def set_step(self, step=None):
#         if step is not None:
#             self.step = step
#         else:
#             self.step += 1

#     def update(self, head='scalar', step=None, **kwargs):
#         for k, v in kwargs.items():
#             if v is None:
#                 continue
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

#     def flush(self):
#         self.writer.flush()


class WandbLogger(object):
    def __init__(self, args, name):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                entity=args.wandb_entity_name,
                project=args.project,
                config=args,
                name=name,
                notes=args.notes if args.notes!= "" else f"{args.procedural_data} {args.procedural_order} {args.pr_notes}".strip(),
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if "probe" in k:
                self._wandb.log({f'Probe/{k}': v}, commit=False)
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)
            elif 'rand' in k:
                self._wandb.log({f'Random/{k}': v}, commit=False)
            elif 'pr' in k:
                self._wandb.log({f'PR/{k}': v}, commit=False)
            else:
                self._wandb.log({f'PR/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')

    def update_config(self, key, value):
        self._wandb.config[key] = value


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class AuxLensLoss:
    """Auxiliary loss on the head's read-out of an intermediate block: the "lens" of engine.model_analyse,
    head(fc_norm(norm(block output)))[:, 0], during training (docs/early_lever_mechanism_plan.md, group C). Active for epoch < until.
      align     cross-entropy of the lens against the batch targets (soft targets under mixup): the class is forced to be readable at
                that block; the final norm and the head receive this gradient as well (it is the model's own lens)
      suppress  cross-entropy of the lens against the UNIFORM distribution minus log K = KL(uniform || lens) >= 0, zero iff the lens
                logits are constant across classes (a logit-VARIANCE penalty: it does not act on their ordering, and the early stack
                can satisfy it by inflating the class token along dimensions the detached final-LayerNorm gain suppresses; review
                2026-09-22); the final norm and the head enter detached, so only the blocks up to `block` (and the embeddings) move: the
                read-out is removed from the early stack, the classifier is not asked to look away. Bounded, unlike plain ascent.
    The block output is taken by a forward hook on the un-wrapped model, so it works under DDP (one backward over main + aux loss)."""
    def __init__(self, model_without_ddp, block, mode, weight, until):
        assert mode in ("align", "suppress") and 0 <= block < len(model_without_ddp.blocks) and weight >= 0 and until >= 0, (block, mode, weight, until)
        self.m, self.block, self.mode, self.weight, self.until = model_without_ddp, int(block), mode, float(weight), int(until)
        self.active, self.x = False, None
        model_without_ddp.blocks[self.block].register_forward_hook(self._keep)
    def _keep(self, module, inputs, output):
        self.x = output if (self.active and module.training) else None
    def set_epoch(self, epoch):
        self.active = epoch < self.until and self.weight > 0; self.x = None
        return self.active
    @staticmethod
    def _norm(layer, x, detach):
        if not isinstance(layer, torch.nn.LayerNorm): return layer(x)                   # Identity (token pooling)
        w, b = (layer.weight.detach(), layer.bias.detach()) if detach else (layer.weight, layer.bias)
        return torch.nn.functional.layer_norm(x, layer.normalized_shape, w, b, layer.eps)
    def __call__(self, criterion, targets):
        assert self.x is not None, "AuxLensLoss: no block output captured (call set_epoch before the forward pass)"
        detach = self.mode == "suppress"; x = self._norm(self.m.fc_norm, self._norm(self.m.norm, self.x[:, 0], detach), detach); self.x = None
        head = self.m.head
        logits = torch.nn.functional.linear(x, head.weight.detach(), head.bias.detach()) if detach else head(x)
        if self.mode == "align": return criterion(logits, targets)
        return -torch.log_softmax(logits.float(), dim=-1).mean(dim=-1).mean() - math.log(logits.shape[-1])


def dump_nonfinite_state(model, samples, targets, output, epoch, step, args):
    """Diagnostics at a non-finite training loss; never raises and uses no collectives (other ranks may not be in this branch).
    Every rank that sees the non-finite loss writes its micro-batch and output to <output_dir>/nan_dump_e<E>_it<S>_rank<R>.pt; the
    first of them (exclusive create) also writes the weights of this moment to nan_dump_e<E>_it<S>_weights.pt. At most 2 weight and
    8 batch dumps per run. The caller asserts afterwards; the pause lets the weight writer finish before torchrun's SIGTERM."""
    try:
        out_dir = getattr(args, "output_dir", None)
        if not out_dir or not os.path.isdir(out_dir): return
        have = os.listdir(out_dir); tag = f"nan_dump_e{epoch}_it{step}"
        if sum(f.startswith("nan_dump_") and "_rank" in f for f in have) < 8:
            torch.save({"samples": samples.detach().half().cpu(), "targets": targets.detach().cpu(), "output": output.detach().float().cpu(), "epoch": epoch, "step": step,
                        "amp_dtype": str(AMP_DTYPE)}, os.path.join(out_dir, f"{tag}_rank{get_rank()}.pt"))
            print(f"[nan-dump] rank {get_rank()}: micro-batch written ({tag})", flush=True)
        if sum(f.endswith("_weights.pt") and f.startswith("nan_dump_") for f in have) < 2:
            path = os.path.join(out_dir, f"{tag}_weights.pt")
            try: fd = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); writer = True
            except FileExistsError: writer = False
            if writer:
                m = model.module if hasattr(model, "module") else model
                torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, path); os.remove(path + ".lock")
                print(f"[nan-dump] rank {get_rank()}: weights written to {path}", flush=True)
        time.sleep(30)
    except Exception as error:
        print(f"[nan-dump] failed: {error!r}", flush=True)


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


# autocast dtype for training/eval under --use_amp; set from --amp_dtype in main (fp16 default, bf16 needs no loss scaling)
AMP_DTYPE = torch.float16


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    @staticmethod
    def get_layer_grad_norms(parameters, norm_type=2.0, group_levels=[None, 6]):
        group_fn = []
        for group_level in group_levels:
            if group_level is None:
                group_fn.append(lambda name: name.rsplit(".", 1)[0])  # group weight+bias together
            else:
                group_fn.append(lambda name: ".".join(name.split(".")[:group_level]))

        group_fn = list(set(group_fn))  # remove duplicates

        # total_grads = []
        # # device = parameters[0][1].grad.device


        # buckets = defaultdict(list)
        # for name, p in parameters:
        #     if p.grad is not None:
        #         grad_norm = p.grad.detach()
        #         for fn in group_fn:
        #             buckets[fn(name)].append(grad_norm)
        #         total_grads.append(grad_norm)
                
        # device = parameters[0][1].grad.device
        expanded_grads = []
        total_grads = []          # one entry per parameter tensor: what the total norm is taken over
        for name, p in parameters:
            if p.grad is not None:
                grad = p.grad.detach()
                total_grads.append(grad)
                
                # if 'qkv.weight' in name or 'qkv.bias' in name:
                if 'qkv' in name:
                    # Split the gradient into 3 equal chunks along the out_features dimension
                    expanded_grads.append((name, grad))
                    q_grad, k_grad, v_grad = torch.chunk(grad, 3, dim=0)
                    
                    # Rename and store them separately
                    base_name = name.replace('qkv', '{}')
                    expanded_grads.append((base_name.format('q'), q_grad))
                    expanded_grads.append((base_name.format('k'), k_grad))
                    expanded_grads.append((base_name.format('v'), v_grad))
                else:
                    expanded_grads.append((name, grad))
        # 2. CHANGED: Loop over our new `expanded_grads` list instead of `parameters`
        buckets = defaultdict(list)

        for name, grad_norm in expanded_grads:
            for fn in group_fn:
                buckets[fn(name)].append(grad_norm)

        layer_norms = {}
        if norm_type == inf:
            for layer, grads in buckets.items():
                layer_norms[layer] = max([g.abs().max() for g in grads])
            total_grad = max([g.abs().max() for g in total_grads])
        else:
            for layer, grads in buckets.items():
                layer_norms[layer] = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grads]), norm_type)
            total_grad = torch.norm(torch.stack([torch.norm(g, norm_type) for g in total_grads]), norm_type)
    

        return total_grad, layer_norms

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,   # should be model.named_parameters()
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        layer_grad_norms = None
        norm = None

        if update_grad:
            self._scaler.unscale_(optimizer)

            if clip_grad is not None:
                assert parameters is not None, "parameters must be provided when clip_grad is not None"
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for _, p in parameters], clip_grad
                )
            else:
                norm, layer_grad_norms = self.get_layer_grad_norms(parameters)
                # print("Layer grad norms:", layer_grad_norms)
                # layer_grad_norms_g3 = self.get_layer_grad_norms(parameters, group_level=3)
                # print("Layer grad norms with group_level=3:", layer_grad_norms_g3)
                # layer_grad_norms.update(layer_grad_norms_g3)
                # norm = get_grad_norm_([p for _, p in parameters])
                # print("Total grad norm:", norm)

            self._scaler.step(optimizer)
            self._scaler.update()

        return norm, layer_grad_norms

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'lr_scale_spec': lr_spec_of(optimizer),      # --lr_scale_json contents (scalar scales + row masks), validated on resume
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        # if to_del in [49, 99, 149, 199, 249, 299] and not args.save_for_analysis: # keep every 50th checkpoint
        #     pass
        # else:
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    backup_resume = None
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            if latest_ckpt > 0:
                backup_resume = os.path.join(output_dir, 'checkpoint-%d.pth' % (latest_ckpt - 1))
        print("Auto resume checkpoint: %s" % args.resume)
        print("Backup resume checkpoint: %s" % backup_resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            try:
                checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load checkpoint from {args.resume} with error {e}")
                if backup_resume is not None:
                    print(f"Trying backup checkpoint {backup_resume}")
                    checkpoint = torch.load(backup_resume, map_location='cpu', weights_only=False)
                else:
                    raise e
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            model_without_ddp.load_state_dict(checkpoint, strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            if 'lr_scale_spec' in checkpoint:
                assert_same_lr_spec(checkpoint['lr_scale_spec'], lr_spec_of(optimizer), args.resume)
            elif lr_spec_of(optimizer) is not None:
                print(f"WARNING: {args.resume} predates the lr-specification record; resuming with {lr_spec_of(optimizer)} unvalidated")
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint['model_ema'])
                else:
                    model_ema.ema.load_state_dict(checkpoint['model'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def reg_scheduler(base_value, final_value, epochs, niter_per_ep, early_epochs=0, early_value=None, 
           mode='linear', early_mode='regular'):
    early_schedule = np.array([])
    early_iters = early_epochs * niter_per_ep
    if early_value is None:
        early_value = final_value
    if early_epochs > 0:
        print(f"Set early value to {early_mode} {early_value}")
        if early_mode == 'regular':
            early_schedule = np.array([early_value] * early_iters)
        elif early_mode == 'linear':
            early_schedule = np.linspace(early_value, base_value, early_iters)
        elif early_mode == 'cosine':
            early_schedule = np.array(
            [base_value + 0.5 * (early_value - base_value) * (1 + math.cos(math.pi * i / early_iters)) for i in np.arange(early_iters)])
    regular_epochs = epochs - early_epochs
    iters = np.arange(regular_epochs * niter_per_ep)
    schedule = np.linspace(base_value, final_value, len(iters))
    schedule = np.concatenate((early_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def calculate_distance(args, model_without_ddp, device):
    output_dir = Path(args.output_dir)
    start_path = os.path.join(output_dir, 'checkpoint-start.pth')
    if not os.path.exists(start_path):
        return -1
    model_start = build_model(args)
    checkpoint_start = torch.load(start_path, map_location='cpu')
    model_start.load_state_dict(checkpoint_start['model'])
    model_start.to(device)
    cur = torch.tensor([]).to(device)
    start = torch.tensor([]).to(device)
    with torch.no_grad():
        for name, p in model_without_ddp.named_parameters():
            cur = torch.cat((cur, p.flatten().clone().detach()))
        for name, p in model_start.named_parameters():
            start = torch.cat((start, p.flatten().clone().detach()))
    return torch.nn.MSELoss()(start, cur).item()

def build_model(args):
    if args.model.startswith("convnext"):
        model = create_model(
            args.model, 
            pretrained=False, 
            num_classes=args.nb_classes, 
            drop_path_rate=args.drop_path,
            ls_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
            )
    else:
        model = create_model(
            args.model, 
            pretrained=False, 
            num_classes=args.nb_classes, 
            drop_path_rate=args.drop_path,
            )
    return model

SPECTRAL_INTERVENTION_MATRICES = ["attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]

def _get_nested_attr(root, dotted_path):
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj

def apply_spectral_intervention(model, intervention_type, target_k=None, args=None, matrices=SPECTRAL_INTERVENTION_MATRICES, blocks=None):
    """Apply a one-time SVD-based spectral intervention, in place, to a set of
    per-block linear weight matrices (e.g. attn.qkv.weight, attn.proj.weight,
    mlp.fc1.weight, mlp.fc2.weight) in model.blocks. If matrices is empty/falsy,
    falls back to SPECTRAL_INTERVENTION_MATRICES (all four per-block linears). If
    blocks is empty/falsy, applies to every block; otherwise only to the block
    indices listed in blocks.

    intervention_type == 'truncate_random': truncates each matrix's spectrum to the
    top target_k singular values (Experiment 1, the sufficiency test - run this on a
    randomly-initialized model, i.e. one built without loading a PR checkpoint).

    intervention_type == 'swap_spectrum': replaces each matrix's singular values with
    those of a freshly-built random reference model of the same architecture (built via
    build_model(args)), while keeping the matrix's own singular vectors (Experiment 2,
    the necessity test - run this on a PR-pretrained model).

    Both branches rescale the reconstructed matrix so its Frobenius norm matches the
    original matrix's Frobenius norm before writing it back.
    """
    assert intervention_type in ("truncate_random", "swap_spectrum"), \
        f"Unknown intervention_type: {intervention_type!r}, expected 'truncate_random' or 'swap_spectrum'"

    if not matrices:
        matrices = SPECTRAL_INTERVENTION_MATRICES

    block_indices = set(blocks) if blocks else set(range(len(model.blocks)))

    if intervention_type == "truncate_random":
        assert isinstance(target_k, int) and target_k > 0, \
            f"target_k must be a positive int for truncate_random, got {target_k!r}"
        random_model = None
    else:
        assert args is not None, "args is required for swap_spectrum, to build the random reference model"
        random_model = build_model(args)

    for i, block in enumerate(model.blocks):
        if i not in block_indices:
            continue
        for name in matrices:
            W = _get_nested_attr(block, name)
            assert W.dim() == 2, f"Expected a 2D weight matrix for blocks.{i}.{name}, got shape {tuple(W.shape)}"

            with torch.no_grad():
                if intervention_type == "truncate_random":
                    assert target_k <= min(W.shape), \
                        f"target_k={target_k} exceeds min(shape)={min(W.shape)} for blocks.{i}.{name} with shape {tuple(W.shape)}"
                    U, S, Vt = torch.linalg.svd(W.data, full_matrices=False)
                    S_trunc = S.clone()
                    S_trunc[target_k:] = 0
                    W_new = (U * S_trunc) @ Vt
                else:
                    W_rand = _get_nested_attr(random_model.blocks[i], name).data.to(device=W.device, dtype=W.dtype)
                    U_pr, _, Vt_pr = torch.linalg.svd(W.data, full_matrices=False)
                    _, S_rand, _ = torch.linalg.svd(W_rand, full_matrices=False)
                    W_new = (U_pr * S_rand) @ Vt_pr

                scale = torch.linalg.norm(W.data) / torch.linalg.norm(W_new)
                W.data.copy_(W_new * scale)

    return model

def ft_load_model(path, args, device, delete_blocks=None, model=None, keep_all=False):
    """Load a checkpoint into a model for fine-tuning or evaluation. With keep_all=False (the fine-tuning default) the head,
    class token, position embedding and patch projection are dropped whenever the file name contains "pr" or
    args.initialize_as_pr is set. Post-training evaluation of a trained checkpoint must pass keep_all=True: the trained
    model IS the state to evaluate, and dropping those keys silently evaluates trained blocks with a random head."""
    if model is None:
        model = build_model(args)
    for block in model.blocks:
        block.attn.fused_attn = False
    if delete_blocks is not None:
        for i in delete_blocks:
            print(f"Deleting block {i} from model")
            del model.blocks[i]
    if path:
        print("Loading model from %s" % path)
        if path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        print("Load initialization from %s" % path)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        print("All keys in checkpoint_model", checkpoint_model.keys())
        if keep_all:
            print("Keeping every key of the checkpoint (evaluation load)")
        elif "pr" in path.split("/")[-1] or args.initialize_as_pr:
            for k in ['head.weight', 'head.bias', 'cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        else:
            for k in ['head.weight', 'head.bias']:
                print(f"Checking key {k} in pretrained checkpoint for finetuning", checkpoint_model[k].shape, state_dict[k].shape)
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)
    if args.distributed:
        print("Using distributed data parallel with GPU %d" % args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    return model, model_without_ddp

def pr_load_model(path, args, device, model=None):
    random.seed(args.seed)
    if model is None:
        model = build_model(args)
    new_block_order = None
    block_attributes = ["norm1.weight", "norm1.bias", "attn.qkv.bias", "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.bias", "mlp.fc2.bias", "attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
    if path:
        if path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        print("Load initialization from %s" % path)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        print("All keys in checkpoint_model", checkpoint_model.keys())
        if "pr" in path.split("/")[-1]:
            for k in ['head.weight', 'head.bias', 'cls_token', 'pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        else:
            for k in ['head.weight', 'head.bias']:
                print(f"Checking key {k} in pretrained checkpoint for finetuning", checkpoint_model[k].shape, state_dict[k].shape)
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        # customise loading/copying weights
        if args.custom_pr_load == "L3 - keep mid,end, repeat start":
            for bi in range(len(model.blocks)-1, 0, -1):
                if bi==11: ri=2
                elif bi==10: ri=1
                else: ri=0
                for k in block_attributes:        
                    checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
                print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
        elif args.custom_pr_load == "L3 - keep start,end, repeat mid":
            for bi in range(len(model.blocks)-1, 1, -1):
                if bi==11: ri=2
                else: ri=1
                for k in block_attributes:        
                    checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
                print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
        elif args.custom_pr_load == "L3 - keep start,mid, repeat end":
            for bi in range(len(model.blocks)-1, 2, -1):
                for k in block_attributes:        
                    checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.2.{k}"]
                print(f"Loading weights for block {bi} from block 2 in pretrained checkpoint")
        elif args.custom_pr_load == "L3 - keep end, repeat mid 8-10":
            for bi in range(len(model.blocks)-1, 7, -1):
                if bi==11: ri=2
                else: ri=1
                for k in block_attributes:        
                    checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
                print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in [0,1,2]:
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")
        elif args.custom_pr_load == "L3 - repeat mid 8-11":
            for bi in range(len(model.blocks)-1, 7, -1):
                ri=1
                for k in block_attributes:        
                    checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
                print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in [0,1,2]:
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")
        elif args.custom_pr_load == "L3 - end 11":
            bi=11
            ri=2
            for k in block_attributes:        
                checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
            print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in [0,1,2]:
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")
        elif args.custom_pr_load == "L12 - end 10":
            bi=11
            ri=10
            for k in block_attributes:        
                checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
            print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in range(11):
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")
        elif args.custom_pr_load == "L12 - end 9":
            bi=11
            ri=9
            for k in block_attributes:        
                checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
            print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in range(11):
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")
        elif args.custom_pr_load == "L12 - end 0":
            bi=11
            ri=0
            for k in block_attributes:        
                checkpoint_model[f"blocks.{bi}.{k}"] = checkpoint_model[f"blocks.{ri}.{k}"]
            print(f"Loading weights for block {bi} from block {ri} in pretrained checkpoint")
            for bi in range(11):
                for k in block_attributes:        
                    del checkpoint_model[f"blocks.{bi}.{k}"]
                print(f"Removing key blocks.{bi}... from pretrained checkpoint")

        for bi in args.skip_load_blocks:
            for k in args.skip_load_block_attributes:
                if f"blocks.{bi}.{k}" in checkpoint_model:
                    print(f"Removing key blocks.{bi}.{k} from pretrained checkpoint")
                    del checkpoint_model[f"blocks.{bi}.{k}"]

        for bi in args.random_blocks:
            for k in block_attributes:
                if f"blocks.{bi}.{k}" in checkpoint_model:
                    print(f"Removing key blocks.{bi}.{k} from pretrained checkpoint")
                    del checkpoint_model[f"blocks.{bi}.{k}"]

        if args.skip_norm:
            for k in ["norm.weight", "norm.bias"]:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        print(f"Loading state dict with {len(checkpoint_model)} keys from pretrained checkpoint after custom modifications")
        print("Keys in checkpoint_model after custom modifications", checkpoint_model.keys())

        for l_no, l_segments in args.skip_attn_segments.items():
            qkv_weight_default = state_dict.get(f"blocks.{l_no}.attn.qkv.weight", None)
            qkv_bias_default = state_dict.get(f"blocks.{l_no}.attn.qkv.bias", None)
            total_dim = qkv_weight_default.shape[0] if qkv_weight_default is not None else qkv_bias_default.shape[0]
            embed_dim = total_dim // 3
            for segment in l_segments:
                if segment == "q": 
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.weight"][:embed_dim, :] = qkv_weight_default[:embed_dim, :]
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.bias"][:embed_dim] = qkv_bias_default[:embed_dim]
                    print(f"Skipping query weights for block {l_no}. Replacing with default initialization")
                elif segment == "k":
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.weight"][embed_dim:2*embed_dim, :] = qkv_weight_default[embed_dim:2*embed_dim, :]
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.bias"][embed_dim:2*embed_dim] = qkv_bias_default[embed_dim:2*embed_dim]
                    print(f"Skipping key weights for block {l_no}. Replacing with default initialization")
                elif segment == "v":
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.weight"][2*embed_dim:3*embed_dim, :] = qkv_weight_default[2*embed_dim:3*embed_dim, :]
                    checkpoint_model[f"blocks.{l_no}.attn.qkv.bias"][2*embed_dim:3*embed_dim] = qkv_bias_default[2*embed_dim:3*embed_dim]
                    print(f"Skipping value weights for block {l_no}. Replacing with default initialization")
                
        load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        if args.shuffle_load and "pr" in args.initialize.split("/")[-1]:
            if type(args.hold_back_blocks) == str:
                if args.hold_back_blocks=="":
                    args.hold_back_blocks = []
                elif args.hold_back_blocks == "all":
                    args.hold_back_blocks = list(range(len(model.blocks)))
                else:
                    args.hold_back_blocks = [int(x) for x in args.hold_back_blocks.split(",")]

            shuffle_blocks = list(range(len(model.blocks)))
            for bi in args.hold_back_blocks:
                shuffle_blocks.remove(bi)
            
            while True:
                new_block_order = random.sample(shuffle_blocks, len(shuffle_blocks))
                if any(shuffle_blocks[i] == new_block_order[i] for i in range(len(shuffle_blocks))):
                    pass
                else:
                    break
            for bi in args.hold_back_blocks:
                new_block_order.insert(bi, bi)

            shuffled_block_order = ",".join([str(i) for i in new_block_order])
            print(f"Shuffling blocks {shuffle_blocks} to new order {new_block_order}, while holding back blocks {args.hold_back_blocks}")

            forward_map = {i:new_block_order[i] for i in range(len(new_block_order))}
            reverse_map = {new_block_order[i]:i for i in range(len(new_block_order))}

            current_state = model.state_dict()
            shuffled_state = {}
            for k, v in current_state.items():
                if k.startswith("blocks."):
                    parts = k.split(".")
                    try:
                        old_idx = int(parts[1])
                    except ValueError:
                        shuffled_state[k] = v
                        continue

                    if old_idx in reverse_map:
                        parts[1] = str(reverse_map[old_idx])  # remap index
                        new_k = ".".join(parts)
                    else:
                        new_k = k
                    shuffled_state[new_k] = v
                else:
                    shuffled_state[k] = v
            load_state_dict(model, shuffled_state, prefix=args.model_prefix)

    for i in args.freeze_blocks:
        if len(args.freeze_block_attributes) > 0:
            for name, p in model.blocks[i].named_parameters():
                if name in args.freeze_block_attributes:
                    p.requires_grad = False
                    print(f"Freezing param {name} in block {i}")
        else:
            print(f"Freezing all params in block {i}")
            for p in model.blocks[i].parameters():
                p.requires_grad = False

    try:
        for pname, p in model.named_parameters():
            if pname in args.train_param_list:
                pass
                print(f"-- Training {pname}")
            else:
                p.requires_grad = False
                print(f"-- Freezing {pname}")
    except AttributeError:
        pass
            
    for i in args.delete_blocks:
        print(f"Deleting block {i}")
        del model.blocks[i]

    model.to(device)
    if args.distributed:
        print("Using distributed data parallel with GPU %d" % args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    return model, model_without_ddp, new_block_order

def parse_args_for_blocks(args):
    args.freeze_blocks = []
    if "f[" in args.pr_notes:
        freeze_str = args.pr_notes.split("f[")[1].split("]")[0]
        if freeze_str.strip() != "":
            args.freeze_blocks = [int(x.strip()) for x in freeze_str.split(",")]
    args.skip_load_blocks = []
    if "s[" in args.pr_notes:
        skip_str = args.pr_notes.split("s[")[1].split("]")[0]
        if skip_str.strip() != "":
            args.skip_load_blocks = [int(x.strip()) for x in skip_str.split(",")]
    args.skip_load_block_attributes = []
    if "sba[" in args.pr_notes:
        sba_str = args.pr_notes.split("sba[")[1].split("]")[0]
        if sba_str.strip() != "":
            args.skip_load_block_attributes = [x.strip() for x in sba_str.split(",")]
    args.freeze_block_attributes = []
    if "fba[" in args.pr_notes:
        fba_str = args.pr_notes.split("fba[")[1].split("]")[0]
        if fba_str.strip() != "":
            args.freeze_block_attributes = [x.strip() for x in fba_str.split(",")]
    args.random_blocks = []
    if "r[" in args.pr_notes:
        try:
            random_str = args.pr_notes.split("r[")[1].split("]")[0]
            if random_str.strip() != "":
                args.random_blocks = [int(x.strip()) for x in random_str.split(",")]
        except ValueError:
            if " r[" in args.pr_notes:
                random_str = args.pr_notes.split(" r[")[1].split("]")[0]
                if random_str.strip() != "":
                    args.random_blocks = [int(x.strip()) for x in random_str.split(",")]
    args.delete_blocks = []
    if "d[" in args.pr_notes:
        delete_str = args.pr_notes.split("d[")[1].split("]")[0]
        if delete_str.strip() != "":
            args.delete_blocks = [int(x.strip()) for x in delete_str.split(",")]
    if "sh" in args.pr_notes:
        args.shuffle_load = True
    args.hold_back_blocks = []
    if "hb[" in args.pr_notes:
        hb_str = args.pr_notes.split("hb[")[1].split("]")[0]
        if hb_str.strip() != "":
            args.hold_back_blocks = [int(x.strip()) for x in hb_str.split(",")]
    args.custom_pr_load = ""
    if "pr[" in args.pr_notes:
        pr_str = args.pr_notes.split("pr[")[1].split("]")[0]
        args.custom_pr_load = pr_str.strip()
    return args

class HookCollector:
    def __init__(self, model):
        self.model = model
        try:
            self.model_without_ddp = model.module
        except AttributeError:
            self.model_without_ddp = model
        self.handles = []
        # Cache: {layer: {'resid': tensor, 'attn': tensor}}
        self.acts = defaultdict(dict)

    def __enter__(self):
        def make_block_hook(idx):
            def hook_block(mod, inp, out):
                self.acts[idx]['inp'] = inp[0].detach()
                x = out.detach()
                self.acts[idx]['blk'] = x
                
                flat = x.reshape(x.shape[0], -1)
                blk_act_norm_per_sample = flat.norm(dim=1)
                # print(f"Block {idx} activation norm per sample: {self.acts[idx]['blk_act_norm_per_sample']}")
                self.acts[idx]['blk_act_norm'] = blk_act_norm_per_sample.mean().item()
                blk_act_rms_per_sample = torch.sqrt((flat ** 2).mean(dim=-1))
                self.acts[idx]['blk_act_rms'] = blk_act_rms_per_sample.mean().item()

                step_wise = mod.norm1(inp[0])
                self.acts[idx]['ln1'] = step_wise.detach()
                step_wise = mod.attn(step_wise)
                self.acts[idx]['qkvp1'] = step_wise.detach()
                step_wise = mod.ls1(step_wise)
                self.acts[idx]['ls1'] = step_wise.detach()
                attn_out = mod.drop_path1(step_wise)
                self.acts[idx]['attn_out'] = attn_out.detach()
                self.acts[idx]['attn'] = inp[0] + attn_out.detach()

                
            def hook_attn_map(mod, inp, out):
                B, N, C = inp[0].shape
                qkv = mod.qkv(inp[0]).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * mod.scale
                attn = attn.softmax(dim=-1)
                self.acts[idx]['attn_map'] = attn  # [B, heads, N, N]

            def hook_mlp_fc1_act(mod, inp, out):
                self.acts[idx]['mlp_fc1_act'] = out.detach()  # GELU(fc1(x)), before fc2

            def hook_mlp_fc2(mod, inp, out):
                self.acts[idx]['mlp_fc2'] = out.detach()  # fc2(...), before ls2/drop_path2/residual

            return hook_block, hook_attn_map, hook_mlp_fc1_act, hook_mlp_fc2

        for i, block in enumerate(self.model_without_ddp.blocks):
            block_hook, attn_map_hook, mlp_fc1_act_hook, mlp_fc2_hook = make_block_hook(i)
            h1 = block.register_forward_hook(block_hook)
            h2 = block.attn.register_forward_hook(attn_map_hook)
            h3 = block.mlp.act.register_forward_hook(mlp_fc1_act_hook)
            h4 = block.mlp.fc2.register_forward_hook(mlp_fc2_hook)
            self.handles.extend([h1, h2, h3, h4])

        return self.acts

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()

class HookCollectorTrain:
    def __init__(self, model):
        self.model = model
        try:
            self.model_without_ddp = model.module
        except AttributeError:
            self.model_without_ddp = model
        self.handles = []
        # Cache: {layer: {'resid': tensor, 'attn': tensor}}
        self.acts = defaultdict(dict)

    def __enter__(self):
        def make_block_hook(idx):
            def hook_block(mod, inp, out):
                self.acts[idx]['inp'] = inp[0]
                x = out
                self.acts[idx]['blk'] = x
                
                flat = x.reshape(x.shape[0], -1)
                blk_act_norm_per_sample = flat.norm(dim=1)
                # print(f"Block {idx} activation norm per sample: {self.acts[idx]['blk_act_norm_per_sample']}")
                self.acts[idx]['blk_act_norm'] = blk_act_norm_per_sample.mean().item()
                blk_act_rms_per_sample = torch.sqrt((flat ** 2).mean(dim=-1))
                self.acts[idx]['blk_act_rms'] = blk_act_rms_per_sample.mean().item()

                attn_out = mod.drop_path1(mod.ls1(mod.attn(mod.norm1(inp[0]))))
                self.acts[idx]['attn_out'] = attn_out
                self.acts[idx]['attn'] = inp[0] + attn_out
                
            def hook_attn_map(mod, inp, out):
                B, N, C = inp[0].shape
                qkv = mod.qkv(inp[0]).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * mod.scale
                attn = attn.softmax(dim=-1)
                self.acts[idx]['attn_map'] = attn  # [B, heads, N, N]
                

            return hook_block, hook_attn_map

        for i, block in enumerate(self.model_without_ddp.blocks):
            block_hook, attn_map_hook = make_block_hook(i)
            h1 = block.register_forward_hook(block_hook)
            h2 = block.attn.register_forward_hook(attn_map_hook)
            self.handles.extend([h1, h2])

        return self.acts

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denorm [B,C,H,W] or [C,H,W] -> [0,1] for imshow.
    Works on uint8 or float32 inputs."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Reverse: x_norm * std + mean
    deno = tensor * std.to(tensor.device) + mean.to(tensor.device)
    return torch.clamp(deno, 0, 1)

def apply_layer_11_scale(model, init_value):
    model.blocks[11].ls1.gamma.data.fill_(init_value)
    model.blocks[11].ls2.gamma.data.fill_(init_value)

def scale_layer_weights(model, layers, scale_factor, init_method_bias_scaling=False):
    depth = len(model.blocks)
    for block_idx in layers:
        block = model.blocks[block_idx]
        # if scale_type == "scale_weights_attn_blk_only":
        block.norm1.weight.data *= scale_factor.get("norm1", 1.0)
        scale_qk = scale_factor.get("qk", 1.0)
        scale_v = scale_factor.get("v", 1.0)
        if scale_qk==scale_v:
            block.attn.qkv.weight.data *= scale_qk
        else:
            total_dim = block.attn.qkv.weight.data.shape[0]
            embed_dim = total_dim // 3
            block.attn.qkv.weight.data[:embed_dim, :] *= scale_qk
            block.attn.qkv.weight.data[embed_dim:2*embed_dim, :] *= scale_qk
            block.attn.qkv.weight.data[2*embed_dim:3*embed_dim, :] *= scale_v
        block.attn.proj.weight.data *= scale_factor.get("proj", 1.0)
        
        block.norm2.weight.data *= scale_factor.get("norm2", 1.0)
        block.mlp.fc1.weight.data *= scale_factor.get("fc1", 1.0)
        block.mlp.fc2.weight.data *= scale_factor.get("fc2", 1.0)

        if init_method_bias_scaling:
            block.norm1.bias.data *= scale_factor.get("norm1", 1.0)
            scale_qk = scale_factor.get("qk", 1.0)
            scale_v = scale_factor.get("v", 1.0)
            if scale_qk==scale_v:
                block.attn.qkv.bias.data *= scale_qk
            else:
                total_dim = block.attn.qkv.bias.data.shape[0]
                embed_dim = total_dim // 3
                block.attn.qkv.bias.data[:embed_dim] *= scale_qk
                block.attn.qkv.bias.data[embed_dim:2*embed_dim] *= scale_qk
                block.attn.qkv.bias.data[2*embed_dim:3*embed_dim] *= scale_v
            block.attn.proj.bias.data *= scale_factor.get("proj", 1.0)
            
            block.norm2.bias.data *= scale_factor.get("norm2", 1.0)
            block.mlp.fc1.bias.data *= scale_factor.get("fc1", 1.0)
            block.mlp.fc2.bias.data *= scale_factor.get("fc2", 1.0)

def shuffle_weights(model, weight_shuffle_dict):
    for block_idx, shuffle_info in weight_shuffle_dict.items():
        print(f"Shuffling weights for block {block_idx} with shuffle_info: {shuffle_info}")
        block = model.blocks[block_idx]
        for weight_name in shuffle_info:
            print(f"Shuffling weights for block {block_idx}, weight {weight_name}")
            # Slice names for the fused attn.qkv. "attn.qk.weight" shuffles rows [0:2e] as ONE
            # pool, which does NOT preserve ||W_q|| and ||W_k|| separately -- proc has 55.11 and
            # 61.90 and the pooled shuffle returns 58.64/58.62, 6.4% off (docs 0c.1). That is the
            # only known difference between ftb4e3fix (79.49) and ftbqmlnvo (79.93), so
            # "attn.q.weight" / "attn.k.weight" shuffle the q and k slices as SEPARATE pools and
            # keep both norms exact. Added 2026-09-04 for arm ftbqks.
            if weight_name in ["attn.qk.weight", "attn.qk.bias", "attn.v.weight", "attn.v.bias",
                               "attn.q.weight", "attn.k.weight"]:
                fused_name = "attn.qkv.bias" if weight_name.endswith(".bias") else "attn.qkv.weight"
                weight_tensor = resolve_param_path(block, fused_name)
                if weight_tensor is not None:
                    total_dim = weight_tensor.data.shape[0]
                    embed_dim = total_dim // 3
                    spans = {"attn.qk.weight": (0, 2 * embed_dim), "attn.qk.bias": (0, 2 * embed_dim),
                             "attn.q.weight": (0, embed_dim),
                             "attn.k.weight": (embed_dim, 2 * embed_dim),
                             "attn.v.weight": (2 * embed_dim, 3 * embed_dim), "attn.v.bias": (2 * embed_dim, 3 * embed_dim)}
                    if weight_name in spans:
                        lo, hi = spans[weight_name]
                        sl = weight_tensor.data[lo:hi]
                        flat_weights = sl.reshape(-1)
                        shuffled_weights = flat_weights[torch.randperm(flat_weights.size(0))]
                        weight_tensor.data[lo:hi].copy_(shuffled_weights.view(sl.shape))
                else:
                    print(f"WARNING: block {block_idx} has no {fused_name}; {weight_name} not shuffled")
            else:
                weight_tensor = resolve_param_path(block, weight_name)
                if weight_tensor is not None:
                    original_shape = weight_tensor.data.shape
                    flat_weights = weight_tensor.data.view(-1)
                    shuffled_weights = flat_weights[torch.randperm(flat_weights.size(0))]
                    weight_tensor.data.copy_(shuffled_weights.view(original_shape))

def resolve_param_path(obj, path):
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

def patched_last_block_forward(self, x):
    y = self.norm1(x)
    attn_out = self.attn(y)
    x = (x * self.attn_res_scale) + (self.drop_path1(self.ls1(attn_out)) * self.attn_out_scale)
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


_PROFILE_SLICES = ("q", "k", "v", "proj", "fc1", "fc2")
# keys of a profile specification that are not weight slices (read below, by calibrate_joint_statistics or by main.py)
_PROFILE_KEYS_THAT_ARE_NOT_SLICES = ("extra", "ln", "fc1_bias", "q_sink", "qk_entropy", "fc1_gate", "common_write", "write_ratio",
                                     "realise", "gain_fold")
_NORM_IN_FRONT_OF = {"q": "1", "k": "1", "v": "1", "fc1": "2"}    # norm1 feeds q, k, v and norm2 feeds fc1; proj and fc2 have none


def _root_mean_square(tensor):
    return float(tensor.detach().float().pow(2).mean().sqrt())


def _weight_slices(block):
    """The six weight slices of one block, as views: q, k, v are the three row groups of the fused attn.qkv.weight."""
    fused_qkv = block.attn.qkv.weight
    width = fused_qkv.shape[1]
    return {"q": fused_qkv[:width], "k": fused_qkv[width:2 * width], "v": fused_qkv[2 * width:],
            "proj": block.attn.proj.weight, "fc1": block.mlp.fc1.weight, "fc2": block.mlp.fc2.weight}


def _declared_scale(slice_specification, block_index, ramp_blocks):
    """The number the specification gives one slice in one block: its own per-block value, or "b0" for block 0 and a linear
    ramp from "start" (first of `ramp_blocks`) to "end" (last) for the others."""
    if "per_block" in slice_specification:            # explicit per-block scales, e.g. a checkpoint's exact profile (ftbanak)
        return float(slice_specification["per_block"][str(block_index)])
    if block_index == 0:
        return float(slice_specification["b0"])
    position = ramp_blocks.index(block_index)
    fraction = position / (len(ramp_blocks) - 1) if len(ramp_blocks) > 1 else 0.0
    return float(slice_specification["start"]) + fraction * (float(slice_specification["end"]) - float(slice_specification["start"]))


def _write_layernorm_vectors(block, block_index, layernorm, moments_per_block, checkpoint_state, seed, width):
    """Write new gains and / or biases into norm1 and norm2 of one block, as the specification's "ln" entry asks.

    Returns {"1": rms of the gain written into norm1, "2": ... norm2}; 1.0 where no gain was written or "compensate" is false,
    i.e. where the multipliers of the slices behind that norm are not to be divided by it.
    The generator depends on (seed, block) only, so every rank and every call draws the same vectors. Per norm the draws are
    always: a permutation (used by the "permute" source only, but drawn in every mode), then the gain, then the bias."""
    gain_rms = {"1": 1.0, "2": 1.0}
    generator = torch.Generator().manual_seed(1000 + 10 * int(seed) + block_index)
    parametric = layernorm.get("source", "permute") == "parametric"
    for norm_index, norm in (("1", block.norm1), ("2", block.norm2)):
        permutation = torch.randperm(width, generator=generator)
        # inline {"gain_mean", "gain_std", "bias_mean", "bias_std"}; without them the checkpoint's vectors are read
        moments = moments_per_block[str(block_index)][f"norm{norm_index}"] if moments_per_block else None
        if layernorm.get("gain"):
            checkpoint_gain = None if moments else checkpoint_state[f"blocks.{block_index}.norm{norm_index}.weight"].float()
            if parametric or moments:   # Gaussian with the checkpoint vector's mean and std (2 numbers), not its values
                override = layernorm.get("gain_stats")   # optional {"mean": m, "std": s}: no checkpoint statistic at all
                if override:
                    mean, std = float(override["mean"]), float(override["std"])
                elif moments:
                    mean, std = moments["gain_mean"], moments["gain_std"]
                else:
                    mean, std = checkpoint_gain.mean(), checkpoint_gain.std()
                gain = torch.randn(width, generator=generator) * std + mean
            else:
                gain = checkpoint_gain[permutation]
            norm.weight.copy_(gain.to(norm.weight.dtype))
            # "compensate" (default true): divide the input-side multipliers by rms(gamma) so the
            # effective scales equal the spec; false leaves the weights exactly as in ftbana
            gain_rms[norm_index] = _root_mean_square(gain) if layernorm.get("compensate", True) else 1.0
        if layernorm.get("bias"):
            if moments:
                bias = torch.randn(width, generator=generator) * moments["bias_std"] + moments["bias_mean"]
            else:
                checkpoint_bias = checkpoint_state[f"blocks.{block_index}.norm{norm_index}.bias"].float()
                bias = (torch.randn(width, generator=generator) * checkpoint_bias.std() + checkpoint_bias.mean()) if parametric \
                    else checkpoint_bias[permutation]
            norm.bias.copy_(bias.to(norm.bias.dtype))
    return gain_rms


def apply_analytic_profile(model, spec, blocks, timm_std=0.02, seed=0):
    """Checkpoint-free early-block init: keep timm's random matrices, write new LayerNorm vectors, and multiply every weight
    slice (q, k, v, proj, fc1, fc2) of `blocks` by ONE scalar so that it has the rms the specification declares, relative to
    timm's trunc-normal std. The joint statistics (sink, gate, ...) are not installed here but by calibrate_joint_statistics.

    `spec` is a dict, a path to a JSON file, or a JSON string (extract_profile.py writes it):

      slice entries  {"q": ..., "k": ..., "v": ..., "proj": ..., "fc1": ..., "fc2": ...}; a slice that is not named keeps timm's
                     initialisation. An entry is {"b0": m0, "start": s, "end": e}: block 0 (if listed) gets m0, the remaining
                     listed blocks ramp linearly from `s` (first) to `e` (last); or {"per_block": {"0": m, "1": m, ...}}.
                     The numbers are *effective* scales: rms(W diag(gamma)) / 0.02 for q, k, v (gamma = norm1's gain) and fc1
                     (norm2's), rms(W) / 0.02 for proj and fc2, read off the proc checkpoint (docs/proc_init_recipe.md section 3,
                     docs 0d.11). Without "ln" the LayerNorm gains stay 1 and biases 0, so the multipliers are these numbers.
      "ln"           {"ckpt": path, "gain": bool, "bias": bool, "source": "permute"|"parametric"} copies the checkpoint's
                     LayerNorm gains and/or biases of the listed blocks, each vector permuted across channels ("permute",
                     default) or replaced by a Gaussian sample with that vector's mean and std ("parametric", ftbanap: the
                     checkpoint contributes two numbers per vector), with a fixed generator (seed-dependent, identical on every
                     rank). "stats": {block: {"norm1": {"gain_mean", "gain_std", "bias_mean", "bias_std"}, "norm2": {...}}}
                     gives those moments inline; the checkpoint file is then not opened. "gain_stats": {"mean", "std"} overrides
                     the gain moments. With "gain", the q/k/v multipliers are divided by rms(gamma1) and fc1 by rms(gamma2) of
                     that block, so the *effective* scales stay those of the spec (ftbanag), unless "compensate" is false;
                     "bias" leaves the multipliers alone (ftbanab).
      "extra"        {block: {slice: multiplier}} applies fixed multipliers to further blocks outside `blocks` (used by ftbanaf
                     to flatten blocks 9-11, and by the late-lever arms ftbrhop / ftbrhoplv with `blocks` empty).
      "fc1_bias", "q_sink"   the data-free gate and sink of the earlier ksd arms, see the comments at the end of the function.

    How a number of the spec is turned into weights ("realise"):
      absent / "multiplier"  (every specification before 2026-09-17): W <- (e / rms(gamma)) * W_timm. The declared scale is met
                 only as far as timm's own rms is 0.02 and the sampled gain is uncorrelated with the columns of W: within 0.2%.
      "exact"    (written by extract_profile.py from 2026-09-17 on): after the LayerNorm vectors have been sampled, the slice is
                 rescaled so that the quantity the spec declares is met to float precision: rms(W diag(gamma)) = e * 0.02 when
                 the spec says "gain_fold": "exact", rms(W) * rms(gamma) = e * 0.02 when it says "product"; rms(W) = e * 0.02
                 for proj and fc2, which have no LayerNorm in front. Still one scalar per slice, so the weights stay timm's
                 up to that scalar.
    Deterministic, so it is safe on every rank before or after the DDP broadcast.
    Returns {block: {slice: (multiplier, raw rms / 0.02, effective rms / 0.02)}} for logging / verification (two entries per
    slice, without the effective one, for the blocks of "extra").
    """
    import json as _json
    if isinstance(spec, str):
        spec = _json.load(open(spec)) if os.path.exists(spec) else _json.loads(spec)
    realise = spec.get("realise", "multiplier")
    if realise not in ("multiplier", "exact"):
        raise ValueError(f"analytic profile: unknown 'realise' value {spec.get('realise')!r}")
    gain_fold = spec.get("gain_fold", "exact")
    slice_specifications = {name: entry for name, entry in spec.items() if name not in _PROFILE_KEYS_THAT_ARE_NOT_SLICES}

    blocks = sorted(int(block) for block in blocks)
    ramp_blocks = [block for block in blocks if block != 0]
    width = model.blocks[0].attn.qkv.weight.shape[1]

    layernorm = spec.get("ln")
    layernorm_moments = (layernorm or {}).get("stats")
    checkpoint_state = None
    if layernorm and not layernorm_moments:       # otherwise the statistics are read from the checkpoint at init time
        checkpoint = torch.load(layernorm["ckpt"], map_location="cpu", weights_only=False)
        checkpoint_state = checkpoint.get("state", checkpoint.get("model", checkpoint))

    applied = {}
    with torch.no_grad():
        for block_index in blocks:
            block = model.blocks[block_index]

            # 1. LayerNorm vectors first: the scales below are declared behind them
            gain_rms = {"1": 1.0, "2": 1.0}
            if layernorm:
                gain_rms = _write_layernorm_vectors(block, block_index, layernorm, layernorm_moments, checkpoint_state, seed, width)
            weights = _weight_slices(block)
            gains = {name: getattr(block, f"norm{norm_index}").weight for name, norm_index in _NORM_IN_FRONT_OF.items()}

            # 2. one multiplier per named slice
            multipliers = {}
            for name, slice_specification in slice_specifications.items():
                gain_rms_in_front = gain_rms[_NORM_IN_FRONT_OF[name]] if name in _NORM_IN_FRONT_OF else 1.0
                # legacy realisation: W <- (e / rms(gamma)) * W_timm, which keeps rms(gamma) * rms(W) at e * timm_std
                multiplier = _declared_scale(slice_specification, block_index, ramp_blocks) / gain_rms_in_front
                if realise == "exact":
                    # the multiplier that meets the declared quantity exactly, measured on the actual tensors. `declared` is
                    # the spec's number again; it is recovered from the legacy multiplier (not read a second time) so that
                    # the weights stay bit-identical to every initialisation verified so far.
                    declared = multiplier * gain_rms_in_front
                    weight = weights[name]
                    if name in gains and gain_rms_in_front != 1.0:   # a written gain enters the declared scale (not with "compensate": false, nor without "ln")
                        gamma = gains[name].detach().float()
                        if gain_fold == "exact":
                            current = _root_mean_square(weight.detach().float() * gamma[None, :])
                        else:
                            current = _root_mean_square(weight) * _root_mean_square(gamma)
                    else:
                        current = _root_mean_square(weight)
                    multiplier = declared * timm_std / current
                multipliers[name] = multiplier

            # 3. apply them in place (views of the fused qkv for q, k, v)
            for name in _PROFILE_SLICES:
                if name in multipliers:
                    weights[name].mul_(multipliers[name])

            # for the log: (multiplier applied, raw rms / timm std, effective rms(W diag gamma) / timm std -- the quantity the
            # spec controls for q, k, v, fc1; equals the raw value for proj and fc2, which have no LayerNorm in front)
            applied[block_index] = {}
            for name, weight in weights.items():
                raw = _root_mean_square(weight) / timm_std
                effective = _root_mean_square(weight.float() * gains[name].detach().float()[None, :]) / timm_std if name in gains else raw
                applied[block_index][name] = (round(multipliers.get(name, 1.0), 3), round(raw, 3), round(effective, 3))

        # "extra": {block: {slice: multiplier}} -- fixed multipliers on timm's weights, no LayerNorm vectors, no exact realisation
        for block_key, fixed_multipliers in spec.get("extra", {}).items():
            block_index = int(block_key)
            weights = _weight_slices(model.blocks[block_index])
            for name in _PROFILE_SLICES:
                if name in fixed_multipliers:
                    weights[name].mul_(float(fixed_multipliers[name]))
            applied[block_index] = {name: (round(float(fixed_multipliers.get(name, 1.0)), 3), round(_root_mean_square(weight) / timm_std, 3))
                                    for name, weight in weights.items()}

        # "fc1_bias": {block: value} -- one constant per block written into the (zero) fc1 bias, shifting every MLP
        # pre-activation by the same amount. Both procedural prefixes have their fc1 pre-activations shifted to a mean
        # of -2 to -3 rms through an alignment of the fc1 rows with the normalised stream (GELU mostly off); the shift is
        # the checkpoint-free stand-in for that alignment (ftbanakb, docs 0d.11 "Generality test").
        for block_key, value in spec.get("fc1_bias", {}).items():
            block_index = int(block_key)
            model.blocks[block_index].mlp.fc1.bias.fill_(float(value))
            applied.setdefault(block_index, {})["fc1_bias"] = (round(float(value), 3), round(float(value), 3))

        # "q_sink": {block: B} -- attention sink: the q part of the (zero) qkv bias of every head is set to a random unit
        # direction (seeded) of norm B. logits_ij += (b_h . k_j) / sqrt(d), the same key ranking for every query, so all
        # queries of a head read one key: a common-mode attention write, as in both procedural prefixes at init (the
        # most-attended key receives 30-80% of the mass). One number per block (ftbanaks, docs 0d.11 "Generality test").
        for block_key, sink_norm in spec.get("q_sink", {}).items():
            block_index = int(block_key)
            attention = model.blocks[block_index].attn
            num_heads, head_dim = attention.num_heads, width // attention.num_heads
            directions = sink_directions(num_heads, head_dim, seed, block_index)
            attention.qkv.bias[:width].copy_((directions * float(sink_norm)).reshape(-1).to(attention.qkv.bias.dtype))
            realised_norm = float(attention.qkv.bias[:width].reshape(num_heads, head_dim).norm(dim=1).mean())
            applied.setdefault(block_index, {})["q_sink"] = (round(float(sink_norm), 3), round(realised_norm, 3))
    return applied


def sink_directions(num_heads, head_dim, seed, block):
    """Per-head random unit directions for the attention-sink q bias, fixed by (seed, block) so that a dump through
    main.py and an offline calibration agree bit for bit."""
    gen = torch.Generator().manual_seed(2000 + 10 * seed + block)
    d = torch.randn(num_heads, head_dim, generator=gen)
    return d / d.norm(dim=1, keepdim=True)


# ----------------------------------------------------------------------------- joint statistics: rank-one components
#
# Per-tensor second moments cannot express how a weight matrix is aligned with the residual stream. In both procedural
# prefixes two such alignments dominate the function of blocks 1..8 (docs/i100_late_block_scaling.md 0d.11):
#
#   * attention sink: W_q maps the direction every token shares onto one query, so logit_ij depends on the key only
#     (attention entropy 0.4 to 1.0 nats on kdyck against 5.2 for any random q/k pair; the top singular component holds
#     22% of W_q's energy against 0.5% for a random matrix);
#   * MLP gate: the average row of fc1 (22 to 30% of fc1's energy on kdyck, 0.03% for a random matrix) points against the
#     stream's common direction, shifting every pre-activation by -2 to -2.7 so the GELU is off.
#
# Both are reproduced here by ONE rank-one component per tensor, placed along the initialised model's OWN stream
# direction and sized by bisection on training images until a functional target read off the checkpoint prefix is met:
#
#   "qk_entropy":      {"entropy": {block: nats}}                W_q <- s_q W_q + alpha P c1^T ,  W_k <- s_k W_k + alpha P r^T
#   "fc1_gate":     {"active_units": {block: fraction}}       W_fc1 <- s W_fc1 - beta (1/sqrt(n)) 1 c2^T
#                   (or {"pre_activation_mean": {block: value}}, the target of the first reconstruction arms)
#   "common_write": {"token_cosine": {block: value}}          W_fc2 <- s W_fc2 + beta u (1/sqrt(n)) 1^T
#
#   The third one is block 0's job in the prefixes: its MLP writes one vector shared by all tokens (91% of the write's energy on
#   kdyck; the mean column of fc2 holds 2.6% of its energy against 0.03% if random), which makes the tokens nearly parallel
#   (cosine 0.91) and is the shared direction the sink and the gate of the later blocks read. The target is the token cosine of
#   the block's OUTPUT, not the write ratio: a rank-one write is purer than the checkpoint's, so matching the ratio (28.8) would
#   drive the cosine to 0.997 and leave the sink no token-specific content to resolve keys with. u is a seeded random unit vector.
#
#   c1, c2   unit direction of the token- and image-mean of norm1's / norm2's output at that block (measured)
#   P        per-head random unit vectors stacked (sink_directions);  r  a seeded random unit direction for the keys
#   1        the all-ones vector over the n hidden units: every unit gets the same shift, like the checkpoint's mean row
#   s, s_q, s_k   with "renormalize" (default) chosen so the EFFECTIVE scale rms(W diag(gamma)) of each tensor is
#            unchanged, i.e. the numbers the specification declares stay exact and only the joint statistic is added.
#            (c1, c2 are means of gamma * x_hat + b and hence correlated with gamma, so keeping the RAW rms instead
#            would shift the effective scale by up to 5%; the raw rms moves by about that much, reported.)
#
#   "write_ratio":  {"attention": {block: ratio}, "mlp": {block: ratio}, "tensors": ["v", "proj", "fc2"] or ["proj", "fc2"]}
#
#   A fourth, scalar component for the OUTPUT side. proj and fc2 write into the un-normalised residual stream, so their weight
#   scale does not carry from one network to another (the same kdyck tail weights write 0.26 of their own stream and 1.45 of a
#   random prefix's); what carries is the write ratio mean_tokens ||sublayer output|| / ||sublayer input stream||. The listed
#   tensors are multiplied by one scalar per sublayer until the ratio measured on the calibration images equals the target
#   read off the checkpoint prefix: attention factor f on proj alone, or sqrt(f) on each of v and proj when "v" is listed (the
#   split upscale_random_match_delta_norms uses for the late lever); MLP factor on fc2. The write is linear in these tensors
#   while their biases are zero, so one step is exact; otherwise the step is repeated. No direction is added: the weights stay
#   timm's up to a scalar. When a block carries both an MLP write ratio and a common write (block 0 of the write-matched
#   reconstruction arms), fc2 has two targets and two free numbers and is solved jointly (_install_common_write_at_ratio):
#   W_fc2 <- c (W_fc2 + m ||W_fc2||_F u (1/sqrt(n)) 1^T), with c fixed by the write ratio for every m (the write is linear in
#   c) and m found by bisection on the token cosine of the block's output. fc2's scale is then an outcome, not an input.
#   The rank-one part can only ADD shared content: if the write-matched random fc2 already makes the tokens more parallel
#   than the target (ksd: 0.73 against 0.64), m = 0 is kept and the report says so ("exceeded_without_component").
#
# Blocks are processed in depth order; within a block: sink, attention write ratio, gate (norm2 sees the attention output),
# common write (fc2 reads fc1's activations), MLP write ratio. Data enters through c1, c2, alpha, beta only: label-free, TRAINING images, evaluation transform.
# Not rank-safe by itself: main.py runs it on rank 0 and broadcasts the calibrated weights.

def calibration_images(samples, loader, transform, n, seed):
    """`n` images of an ImageFolder-style sample list, chosen by a seeded permutation, under `transform` (no labels used)."""
    gen = torch.Generator().manual_seed(4000 + int(seed))
    index = torch.randperm(len(samples), generator=gen)[:n].tolist()
    return torch.stack([transform(loader(samples[i][0])) for i in index])


def _attention_rows(block, stream, qkv_weight):
    """Attention probabilities (B, H, N, N), logits and queries of `block` for input `stream`, using `qkv_weight` in place
    of the block's own (so a candidate can be evaluated without touching the parameters)."""
    attn = block.attn
    y = block.norm1(stream)
    B, N, C = y.shape
    H = attn.num_heads
    qkv = torch.nn.functional.linear(y, qkv_weight, attn.qkv.bias).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    logits = (q @ k.transpose(-2, -1)) * attn.scale
    return logits.softmax(dim=-1), logits, q


@torch.no_grad()
def block_input_stream(model, images):
    """The residual stream entering block 0 for `images` (patch embedding, position embedding, class token), obtained by
    stopping the model's own forward pass at the first block."""
    captured = {}

    class _Stop(Exception):
        pass

    def _grab(module, inputs):
        captured["stream"] = inputs[0]
        raise _Stop

    handle = model.blocks[0].register_forward_pre_hook(_grab)
    try:
        model(images)
    except _Stop:
        pass
    handle.remove()
    return captured["stream"].float()


@torch.no_grad()
def fc1_input(block, stream):
    """norm2 of the stream after the block's attention sub-layer, following the block's own forward: layer scale (ls1),
    stochastic depth (drop_path1) and, for a block patched by patched_last_block_forward, the residual / attention-output
    scales. All of these are the identity for the evaluation-mode ViT-B used so far, so the value is unchanged there;
    spelling them out keeps the calibration target meaning "what fc1 really reads" under any other configuration."""
    attention_out = block.attn(block.norm1(stream))
    for name in ("ls1", "drop_path1"):
        if hasattr(block, name):
            attention_out = getattr(block, name)(attention_out)
    return block.norm2(stream * getattr(block, "attn_res_scale", 1.0) + attention_out * getattr(block, "attn_out_scale", 1.0))


@torch.no_grad()
def sublayer_write_ratios(block, stream):
    """(attention write ratio, MLP write ratio, block output) of `block` for input `stream`: mean over images and tokens of
    ||sublayer output|| / ||stream the sublayer adds to||, following the block's own forward (layer scale and stochastic
    depth, the identity for the evaluation-mode ViT-B used so far)."""
    attention_out = block.attn(block.norm1(stream))
    for name in ("ls1", "drop_path1"):
        if hasattr(block, name):
            attention_out = getattr(block, name)(attention_out)
    after_attention = stream * getattr(block, "attn_res_scale", 1.0) + attention_out * getattr(block, "attn_out_scale", 1.0)
    mlp_out = block.mlp(block.norm2(after_attention))
    for name in ("ls2", "drop_path2"):
        if hasattr(block, name):
            mlp_out = getattr(block, name)(mlp_out)
    return (float((attention_out.norm(dim=-1) / stream.norm(dim=-1)).mean()),
            float((mlp_out.norm(dim=-1) / after_attention.norm(dim=-1)).mean()), after_attention + mlp_out)


@torch.no_grad()
def joint_statistics_per_block(model, images, blocks):
    """{block: {"entropy", "sink_share", "pre_activation_mean", "active_units", "token_cosine", "attention_write",
    "mlp_write"}} on `images`: the functional quantities the components target (mean attention entropy in nats, share of
    attention mass on the most-attended key, mean fc1 pre-activation, fraction of positive fc1 pre-activations, mean cosine
    between the patch tokens of the block's OUTPUT, and the two write ratios of sublayer_write_ratios)."""
    if not blocks:
        return {}
    was_training = model.training
    model.eval()
    stream, result = block_input_stream(model, images), {}
    for index, block in enumerate(model.blocks):
        if index > max(blocks):
            break
        if index in blocks:
            probabilities, _, _ = _attention_rows(block, stream, block.attn.qkv.weight)
            pre_activation = block.mlp.fc1(fc1_input(block, stream))
            output = block(stream)
            attention_write, mlp_write, _ = sublayer_write_ratios(block, stream)
            result[index] = {"attention_write": attention_write, "mlp_write": mlp_write,
                             "entropy": float(-(probabilities * (probabilities + 1e-12).log()).sum(-1).mean()),
                             "sink_share": float(probabilities.mean(2).max(-1).values.mean()),
                             "pre_activation_mean": float(pre_activation.mean()),
                             "active_units": float((pre_activation > 0).float().mean()),
                             "token_cosine": _token_cosine(output)}
        stream = block(stream)
    if was_training:
        model.train()
    return result


def _token_cosine(stream):
    """Mean cosine between the patch tokens of one image (class token excluded), averaged over images."""
    tokens = torch.nn.functional.normalize(stream[:, 1:].float(), dim=-1)
    return float((tokens @ tokens.transpose(1, 2)).mean())


def _rescale_to_keep_norm(base, component, strength):
    """The factor s > 0 by which `base` must shrink so that adding `strength * component` leaves its Frobenius norm unchanged,

        || s * base + strength * component ||_F  =  || base ||_F ,

    or None when the strength is too large for that. Expanding the square gives a quadratic in s,

        ||base||^2 s^2  +  2 strength <base, component> s  +  strength^2 ||component||^2 - ||base||^2  =  0 ,

    of which the larger root is returned (the other one is negative: it would flip the sign of `base`). When the component is
    orthogonal to `base` this is Pythagoras, s = sqrt(1 - strength^2 ||component||^2 / ||base||^2): the component takes that share
    of the squared norm and `base` keeps the rest. For strength < ||base|| / ||component|| a positive s always exists, so callers
    that cap the strength below that bound never receive None. To keep a weighted norm, e.g. the gain-folded ||W diag(gamma)||_F,
    pass both tensors already multiplied by the weights: the equation is linear in them."""
    base_squared_norm = float(base.pow(2).sum())
    overlap = float((base * component).sum())                       # <base, component>
    component_squared_norm = float(component.pow(2).sum())
    discriminant = (strength * overlap) ** 2 - base_squared_norm * (strength ** 2 * component_squared_norm - base_squared_norm)
    if discriminant < 0:        # no real root: the component alone already exceeds the norm of `base`
        return None
    shrink = (-strength * overlap + discriminant ** 0.5) / base_squared_norm
    return shrink if shrink > 0 else None


def _bisect(value_at, target, hi, decreasing=True, steps=40, tolerance=1e-4):
    """Smallest strength in [0, hi] at which `value_at` reaches `target` (value_at monotone in the strength).
    Returns (strength, reachable)."""
    below = (lambda v: v <= target) if decreasing else (lambda v: v >= target)
    if not below(value_at(hi)):
        return hi, False
    if below(value_at(0.0)):          # already past the target without any component: no positive strength matches it
        return 0.0, False
    lo = 0.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if below(value_at(mid)):
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            break
    return hi, True


def _mean_attention_entropy(probabilities):
    """Entropy (nats) of each attention row, averaged over images, heads and queries."""
    return float(-(probabilities * (probabilities + 1e-12).log()).sum(-1).mean())


def _install_rank_one_sink(block, stream, target_entropy, renormalize, seed, index, gain_fold="exact"):
    """Attention sink of one block: one rank-one component in W_q and one in W_k (see the section comment), sized by bisection
    until the block's mean attention entropy on `stream` (the block's input) equals `target_entropy`.

        W_q <- s_q W_q + strength * P c^T       c   unit direction of the mean of norm1's output: what every token shares
        W_k <- s_k W_k + strength * P r^T       r   a seeded random unit direction
                                                P   per-head random unit vectors, stacked (sink_directions)

    c is common to all tokens, so every query receives the same extra vector along P; r is not, so the keys receive a
    token-dependent amount of it. All queries of a head therefore rank the keys alike: a sink, and the entropy falls as the
    strength grows. With `renormalize`, s_q and s_k shrink the existing weights so that the scale the specification declares
    is unchanged; `gain_fold` names that scale: "exact" = rms(W diag(gamma)), "product" = rms(W) * rms(gamma), which for the
    fixed sampled gamma means keeping the raw norm of W. Without it s_q = s_k = 1.
    Writes the result into block.attn.qkv.weight (also when the target is not reachable: see "reachable" / "matched") and
    returns its report."""
    fused_qkv = block.attn.qkv.weight.data
    width, num_heads = fused_qkv.shape[1], block.attn.num_heads
    query_rows, key_rows = slice(0, width), slice(width, 2 * width)
    own_query, own_key = fused_qkv[query_rows].clone().float(), fused_qkv[key_rows].clone().float()     # before the component

    common_direction = torch.nn.functional.normalize(block.norm1(stream).mean(dim=(0, 1)), dim=0)
    key_direction = torch.randn(width, generator=torch.Generator().manual_seed(3000 + 10 * int(seed) + index)).to(fused_qkv.device)
    key_direction = torch.nn.functional.normalize(key_direction, dim=0)
    head_directions = sink_directions(num_heads, width // num_heads, int(seed), index).reshape(width, 1).to(fused_qkv.device)
    query_component, key_component = head_directions * common_direction[None, :], head_directions * key_direction[None, :]

    gain = block.norm1.weight.detach().float()[None, :]
    # column weights of the norm that the renormalisation keeps: the folded W diag(gamma), or the raw W
    kept_norm = gain if gain_fold == "exact" else torch.ones_like(gain)

    def with_component(strength):
        """(fused qkv weight carrying the component at `strength`, s_q, s_k)"""
        shrink_query = _rescale_to_keep_norm(own_query * kept_norm, query_component * kept_norm, strength) if renormalize else 1.0
        shrink_key = _rescale_to_keep_norm(own_key * kept_norm, key_component * kept_norm, strength) if renormalize else 1.0
        candidate = fused_qkv.clone().float()
        candidate[query_rows] = shrink_query * own_query + strength * query_component
        candidate[key_rows] = shrink_key * own_key + strength * key_component
        return candidate, shrink_query, shrink_key

    def entropy_at(strength):
        probabilities, _, _ = _attention_rows(block, stream, with_component(strength)[0])
        return _mean_attention_entropy(probabilities)

    if renormalize:     # the rank-one part cannot exceed the tensor's own norm (in the kept norm); stay 2% below that
        max_strength = 0.98 * min(float((own_query * kept_norm).norm() / (query_component * kept_norm).norm()),
                                  float((own_key * kept_norm).norm() / (key_component * kept_norm).norm()))
    else:
        max_strength = 64.0
    strength, reachable = _bisect(entropy_at, target_entropy, max_strength, decreasing=True)

    candidate, shrink_query, shrink_key = with_component(strength)
    probabilities, logits, queries = _attention_rows(block, stream, candidate)
    fused_qkv.copy_(candidate.to(fused_qkv.dtype))

    entropy = _mean_attention_entropy(probabilities)
    # share of the queries' energy that is the same for every token of an image: (B, H, N, d) -> (B, N, H * d)
    queries = queries.transpose(1, 2).reshape(queries.shape[0], queries.shape[2], -1)
    common_query = queries.mean(1, keepdim=True)
    new_query, new_key = candidate[query_rows], candidate[key_rows]
    return {"alpha": strength, "s_q": shrink_query, "s_k": shrink_key, "target": target_entropy, "reachable": bool(reachable),
            "matched": bool(reachable) and abs(entropy - target_entropy) <= 0.02,    # the statistic itself, not the bracket
            "entropy": entropy,
            "sink_share": float(probabilities.mean(2).max(-1).values.mean()),       # attention mass on the most-attended key
            "common_query": float((common_query.pow(2).sum(-1) / queries.pow(2).sum(-1).mean(1, keepdim=True)).mean()),
            "row_logit_std": float(logits.std(-1).mean()), "max_abs_logit": float(logits.abs().max()),
            "rms_q_ratio": _root_mean_square(new_query) / _root_mean_square(own_query),
            "rms_k_ratio": _root_mean_square(new_key) / _root_mean_square(own_key),
            "effective_q_ratio": _root_mean_square(new_query * gain) / _root_mean_square(own_query * gain),
            "effective_k_ratio": _root_mean_square(new_key * gain) / _root_mean_square(own_key * gain)}


def _install_fc1_gate(block, stream, target, renormalize, statistic="pre_activation_mean", gain_fold="exact"):
    """MLP gate of one block: a rank-one mean-row component in fc1 (see the section comment), sized by bisection until the
    chosen statistic of the fc1 pre-activations on `stream` equals `target`.

        W_fc1 <- s W_fc1 - strength * (1/sqrt(n)) 1 c^T      c   unit direction of the mean of what fc1 reads (norm2's output)
                                                             1   the all-ones vector over the n hidden units

    c is common to all tokens, so every hidden unit of every token is shifted down by the same amount and the GELU closes;
    the component has unit Frobenius norm. `stream` is the block's input and the block's attention must already be final,
    because fc1 reads norm2(stream + attention output). `statistic` names what `target` is: "active_units" = the fraction of
    positive fc1 pre-activations (how much of the MLP is switched on), or "pre_activation_mean". Both fall monotonically as
    the component grows; the report carries both, and the GELU output rms, whichever is the target. `renormalize` and
    `gain_fold` as in _install_rank_one_sink: s shrinks the existing weight so that the declared scale is unchanged.
    Writes the result into block.mlp.fc1.weight (also when the target is not reachable: see "reachable" / "matched") and
    returns its report."""
    if statistic not in ("active_units", "pre_activation_mean"):
        raise ValueError(f"fc1_gate: unknown target statistic {statistic!r}")
    fc1 = block.mlp.fc1
    weight = fc1.weight.data
    own_weight = weight.clone().float()                                 # before the component
    hidden_units = weight.shape[0]
    bias = fc1.bias.detach().float()

    fc1_reads = fc1_input(block, stream)                                # norm2 of the stream after the attention sub-layer
    common_direction = torch.nn.functional.normalize(fc1_reads.mean(dim=(0, 1)), dim=0)
    # unit Frobenius norm; every hidden unit shifted alike, against the common direction
    gate_component = -torch.ones(hidden_units, 1, device=weight.device) / hidden_units ** 0.5 * common_direction[None, :]

    gain = block.norm2.weight.detach().float()[None, :]
    # column weights of the norm that the renormalisation keeps: the folded W diag(gamma), or the raw W
    kept_norm = gain if gain_fold == "exact" else torch.ones_like(gain)

    def with_component(strength):
        """(fc1 weight carrying the component at `strength`, s)"""
        shrink = _rescale_to_keep_norm(own_weight * kept_norm, gate_component * kept_norm, strength) if renormalize else 1.0
        return shrink * own_weight + strength * gate_component, shrink

    def statistic_at(strength):
        pre_activation = torch.nn.functional.linear(fc1_reads, with_component(strength)[0], bias)
        return float((pre_activation > 0).float().mean()) if statistic == "active_units" else float(pre_activation.mean())

    if renormalize:     # the rank-one part cannot exceed the tensor's own norm (in the kept norm); stay 2% below that
        max_strength = 0.98 * float((own_weight * kept_norm).norm() / (gate_component * kept_norm).norm())
    else:
        max_strength = 64.0
    strength, reachable = _bisect(statistic_at, target, max_strength, decreasing=True, tolerance=1e-5)

    candidate, shrink = with_component(strength)
    pre_activation = torch.nn.functional.linear(fc1_reads, candidate, bias)
    folded = candidate * gain
    weight.copy_(candidate.to(weight.dtype))

    active_units, pre_activation_mean = float((pre_activation > 0).float().mean()), float(pre_activation.mean())
    if statistic == "active_units":     # the statistic itself, not the bracket; a fraction of a finite sample: relative with a floor
        matched = bool(reachable) and abs(active_units - target) <= max(2e-5, 0.03 * target)
    else:
        matched = bool(reachable) and abs(pre_activation_mean - target) <= 0.02
    return {"beta": strength, "s": shrink, "target": target, "statistic": statistic, "reachable": bool(reachable), "matched": matched,
            "pre_activation_mean": pre_activation_mean, "pre_activation_std": float(pre_activation.std()),
            "active_units": active_units, "gelu_rms": _root_mean_square(block.mlp.act(pre_activation)),
            # share of the folded matrix's energy in its mean row (random: 1 / n)
            "mean_row_energy_share": float(hidden_units * folded.mean(0).pow(2).sum() / folded.pow(2).sum()),
            "rms_ratio": _root_mean_square(candidate) / _root_mean_square(own_weight),
            "effective_ratio": _root_mean_square(folded) / _root_mean_square(own_weight * gain)}


def _install_common_write(block, stream, target_cosine, renormalize, seed, index):
    """Rank-one mean-column component of one block's fc2 (see the section comment); `stream` is the block's input, the
    block's attention and fc1 must already be final. Returns its report."""
    fc2 = block.mlp.fc2
    W = fc2.weight.data
    base = W.clone().float()
    n = W.shape[1]
    hidden = block.mlp.act(block.mlp.fc1(fc1_input(block, stream)))
    u = torch.randn(W.shape[0], generator=torch.Generator().manual_seed(5000 + 10 * int(seed) + index)).to(W.device)
    u = torch.nn.functional.normalize(u, dim=0)
    delta = u[:, None] * torch.ones(1, n, device=W.device) / n ** 0.5     # unit Frobenius norm; every hidden unit writes u alike
    bias = fc2.bias.detach().float()

    def candidate(beta):                                                  # fc2 has no LayerNorm in front: effective = raw scale
        s = _rescale_to_keep_norm(base, delta, beta) if renormalize else 1.0
        return s * base + beta * delta, s

    def cosine_at(beta):                 # the block's own forward with the candidate installed: exactly what training will see
        W.copy_(candidate(beta)[0].to(W.dtype))
        return _token_cosine(block(stream))

    hi = 0.98 * float(base.norm() / delta.norm()) if renormalize else 256.0
    beta, reachable = _bisect(cosine_at, target_cosine, hi, decreasing=False)
    Wc, s = candidate(beta)
    W.copy_(Wc.to(W.dtype))
    output = block(stream)
    write = torch.nn.functional.linear(hidden, Wc, bias)
    patches = write[:, 1:]
    rms = lambda t: float(t.pow(2).mean().sqrt())
    cosine = _token_cosine(output)
    return {"beta": beta, "s": s, "target": target_cosine, "reachable": bool(reachable),
            "matched": bool(reachable) and abs(cosine - target_cosine) <= 0.005, "token_cosine": cosine,
            "mlp_write_ratio": float((write.norm(dim=-1) / (output - write).norm(dim=-1)).mean()),
            "common_share_of_write": float(patches.mean(1, keepdim=True).pow(2).sum() * patches.shape[1] / patches.pow(2).sum()),
            "mean_column_energy_share": float(n * Wc.mean(1).pow(2).sum() / Wc.pow(2).sum()), "rms_ratio": rms(Wc) / rms(base)}


def _install_common_write_at_ratio(block, stream, target_cosine, target_ratio, seed, index):
    """fc2 of one block with BOTH targets (see the section comment): token cosine of the block's output and MLP write ratio.
    `stream` is the block's input; attention and fc1 must already be final. Returns its report."""
    fc2 = block.mlp.fc2
    W = fc2.weight.data
    base = W.clone().float()
    n = W.shape[1]
    u = torch.randn(W.shape[0], generator=torch.Generator().manual_seed(5000 + 10 * int(seed) + index)).to(W.device)
    u = torch.nn.functional.normalize(u, dim=0)
    delta = u[:, None] * torch.ones(1, n, device=W.device) / n ** 0.5     # unit Frobenius norm, as in _install_common_write
    base_norm = float(base.norm())

    def install(m):                       # direction first, then the scale that meets the write ratio (linear while the bias is 0)
        W.copy_((base + m * base_norm * delta).to(W.dtype))
        scale = 1.0
        for _ in range(4):
            ratio = sublayer_write_ratios(block, stream)[1]
            if abs(ratio / target_ratio - 1.0) < 1e-6:
                break
            W.mul_(target_ratio / ratio); scale *= target_ratio / ratio
        return scale

    def cosine_at(m):
        install(m)
        return _token_cosine(block(stream))

    cosine_without = cosine_at(0.0)
    if cosine_without >= target_cosine:    # a rank-one common component can only raise the cosine
        m, reachable, exceeded = 0.0, False, True
    else:
        hi = 1.0
        while cosine_at(hi) < target_cosine and hi < 4096.0:
            hi *= 2.0
        m, reachable = _bisect(cosine_at, target_cosine, hi, decreasing=False, tolerance=1e-5)
        exceeded = False
    c = install(m)
    attention_write, mlp_write, output = sublayer_write_ratios(block, stream)
    hidden = block.mlp.act(block.mlp.fc1(fc1_input(block, stream)))
    patches = torch.nn.functional.linear(hidden, W.float(), fc2.bias.detach().float())[:, 1:]
    rms = lambda t: float(t.pow(2).mean().sqrt())
    cosine = _token_cosine(output)
    matched = (bool(exceeded) or (bool(reachable) and abs(cosine - target_cosine) <= 0.005)) and abs(mlp_write / target_ratio - 1.0) < 1e-3
    return {"m": m, "c": c, "beta": c * m * base_norm, "target": target_cosine, "target_ratio": target_ratio,
            "reachable": bool(reachable), "exceeded_without_component": bool(exceeded), "token_cosine_without_component": cosine_without,
            "matched": matched, "token_cosine": cosine, "mlp_write_ratio": mlp_write,
            "common_share_of_write": float(patches.mean(1, keepdim=True).pow(2).sum() * patches.shape[1] / patches.pow(2).sum()),
            "mean_column_energy_share": float(n * W.float().mean(1).pow(2).sum() / W.float().pow(2).sum()), "rms_ratio": rms(W.float()) / rms(base)}


def _match_write_ratio(block, stream, sublayer, target, tensors, steps=6, tolerance=1e-5):
    """Multiply the listed write-side tensors of one sublayer ("attention": v rows of qkv and / or proj; "mlp": fc2) by one
    scalar so that the sublayer's write ratio on `stream` equals `target` (see the section comment). Biases of the scaled
    tensors are scaled along. Returns its report."""
    attn, D = block.attn, block.attn.qkv.weight.shape[1]
    if sublayer == "attention":
        parts = [(attn.qkv.weight.data[2 * D:], None if attn.qkv.bias is None else attn.qkv.bias.data[2 * D:])] if "v" in tensors else []
        parts += [(attn.proj.weight.data, None if attn.proj.bias is None else attn.proj.bias.data)] if "proj" in tensors else []
    else:
        parts = [(block.mlp.fc2.weight.data, None if block.mlp.fc2.bias is None else block.mlp.fc2.bias.data)]
    measure = lambda: sublayer_write_ratios(block, stream)[0 if sublayer == "attention" else 1]
    before, factor = measure(), 1.0
    for _ in range(steps):
        current = measure()
        if abs(current / target - 1.0) < tolerance:
            break
        step = (target / current) ** (1.0 / len(parts))          # the write is multilinear in the listed tensors
        for weight, bias in parts:
            weight.mul_(step)
            if bias is not None:
                bias.mul_(step)
        factor *= step
    reached = measure()
    return {"target": target, "before": before, "write_ratio": reached, "factor_per_tensor": factor,
            "tensors": [name for name in (("v", "proj") if sublayer == "attention" else ("fc2",)) if name in tensors],
            "reachable": bool(abs(reached / target - 1.0) < 1e-3), "matched": bool(abs(reached / target - 1.0) < 1e-3)}


@torch.no_grad()
def calibrate_joint_statistics(model, spec, images, seed=0):
    """Install the components requested by `spec` ("qk_entropy", "fc1_gate", "common_write", "write_ratio"; see the section
    comment). `images`: (n, 3, H, W) on the model's device. Modifies attn.qkv.weight / attn.proj.weight / mlp.fc1.weight /
    mlp.fc2.weight of the listed blocks in place. Returns {block: {component: report}}; the write ratio reports under
    "write_ratio_attention" and "write_ratio_mlp", a block's fc2 with both a common write and a write ratio under
    "common_write_at_ratio"."""
    if "qk_sink" in spec:
        raise ValueError("specification key 'qk_sink' was renamed to 'qk_entropy' on 2026-09-17 (its target is the attention entropy); "
                         "rename the key, the numbers are unchanged")
    sink, gate, common = spec.get("qk_entropy") or {}, spec.get("fc1_gate") or {}, spec.get("common_write") or {}
    write = spec.get("write_ratio") or {}
    # the scale the specification declares, which the rank-one components must leave unchanged; specifications written before
    # the key existed all carry joint statistics from --gain_fold exact extractions
    gain_fold = spec.get("gain_fold", "exact")
    if gain_fold not in ("exact", "product"):
        raise ValueError(f"joint statistics: unknown gain_fold {gain_fold!r}")
    sink_targets = {int(b): float(v) for b, v in sink.get("entropy", {}).items()}
    if "active_units" in gate and "pre_activation_mean" in gate:
        raise ValueError("fc1_gate: give either 'active_units' or 'pre_activation_mean' targets, not both (one number per block)")
    gate_statistic = "active_units" if "active_units" in gate else "pre_activation_mean"
    gate_targets = {int(b): float(v) for b, v in gate.get(gate_statistic, {}).items()}
    common_targets = {int(b): float(v) for b, v in common.get("token_cosine", {}).items()}
    write_tensors = list(write.get("tensors", []))
    attention_write_targets = {int(b): float(v) for b, v in write.get("attention", {}).items()} if {"v", "proj"} & set(write_tensors) else {}
    mlp_write_targets = {int(b): float(v) for b, v in write.get("mlp", {}).items()} if "fc2" in write_tensors else {}
    if set(write_tensors) - {"v", "proj", "fc2"}:
        raise ValueError(f"write_ratio: tensors must be among v, proj, fc2, got {write_tensors}")
    if not sink_targets and not gate_targets and not common_targets and not attention_write_targets and not mlp_write_targets:
        return {}
    was_training = model.training
    model.eval()
    stream, report = block_input_stream(model, images), {}
    last = max(list(sink_targets) + list(gate_targets) + list(common_targets) + list(attention_write_targets) + list(mlp_write_targets))
    for index, block in enumerate(model.blocks):
        if index > last:
            break
        if index in sink_targets:
            report.setdefault(index, {})["qk_entropy"] = _install_rank_one_sink(
                block, stream, sink_targets[index], bool(sink.get("renormalize", True)), seed, index, gain_fold)
        if index in attention_write_targets:
            report.setdefault(index, {})["write_ratio_attention"] = _match_write_ratio(
                block, stream, "attention", attention_write_targets[index], write_tensors)
        if index in gate_targets:
            report.setdefault(index, {})["fc1_gate"] = _install_fc1_gate(
                block, stream, gate_targets[index], bool(gate.get("renormalize", True)), gate_statistic, gain_fold)
        if index in common_targets and index in mlp_write_targets:          # fc2 with two targets: solved jointly
            report.setdefault(index, {})["common_write_at_ratio"] = _install_common_write_at_ratio(
                block, stream, common_targets[index], mlp_write_targets[index], seed, index)
        elif index in common_targets:
            report.setdefault(index, {})["common_write"] = _install_common_write(
                block, stream, common_targets[index], bool(common.get("renormalize", True)), seed, index)
        elif index in mlp_write_targets:
            report.setdefault(index, {})["write_ratio_mlp"] = _match_write_ratio(
                block, stream, "mlp", mlp_write_targets[index], write_tensors)
        stream = block(stream)
    if was_training:
        model.train()
    return report
