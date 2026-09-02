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
        # if not is_dist_avail_and_initialized():
        #     return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
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


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

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
        for name, p in parameters:
            if p.grad is not None:
                grad = p.grad.detach()
                
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
        total_grads = []
      
        for name, grad_norm in expanded_grads:
            for fn in group_fn:
                buckets[fn(name)].append(grad_norm)
            total_grads.append(grad_norm)

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

def ft_load_model(path, args, device, delete_blocks=None, model=None):
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
        if "pr" in path.split("/")[-1] or args.initialize_as_pr:
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
            if weight_name in ["attn.qk.weight", "attn.qk.bias", "attn.v.weight", "attn.v.bias"]:
                weight_tensor = resolve_param_path(block, "attn.qkv.weight")
                if weight_tensor is not None:
                    total_dim = weight_tensor.data.shape[0]
                    embed_dim = total_dim // 3
                    if weight_name == "attn.qk.weight":
                        qk_weights = weight_tensor.data[:2*embed_dim, :]
                        flat_weights = qk_weights.view(-1)
                        shuffled_weights = flat_weights[torch.randperm(flat_weights.size(0))]
                        weight_tensor.data[:2*embed_dim, :].copy_(shuffled_weights.view(qk_weights.shape))
                    elif weight_name == "attn.v.weight":
                        v_weights = weight_tensor.data[2*embed_dim:3*embed_dim, :]
                        flat_weights = v_weights.view(-1)
                        shuffled_weights = flat_weights[torch.randperm(flat_weights.size(0))]
                        weight_tensor.data[2*embed_dim:3*embed_dim, :].copy_(shuffled_weights.view(v_weights.shape))
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
