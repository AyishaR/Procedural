import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
import pickle
import torch.distributed as dist
import random
from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import *

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from linear_probe_utils import *
import models.convnext
import models.vision_transformer

# from hessian_calculate import *

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--total_batch_size', default=64, type=int,
                        help='Effective batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    parser.add_argument('--slurm_id', type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--freeze_blocks', default="", type=str,
                        help='Comma separated list of layer indices to freeze, e.g. "0,1,2" to freeze the first 3 layers; supports "all" to freeze all layers and "" to not freeze any layers (default: "")')
    parser.add_argument('--delete_blocks', default="", type=str,
                        help='Comma separated list of layer indices to delete, e.g. "0,1,2" to delete the first 3 layers; supports "all" to delete all layers and "" to not delete any layers (default: "")')
    parser.add_argument('--skip_load_blocks', default="", type=str,
                        help='Comma separated list of layer indices to skip loading, e.g. "0,1,2" to freeze the first 3 layers; supports "all" to freeze all layers and "" to not freeze any layers (default: "")')
    parser.add_argument('--skip_load_block_attributes', default="", type=str,
                        help='Comma separated list of block attributes to skip loading for the --skip_load_blocks blocks, e.g. "norm1.weight,norm1.weight')
    parser.add_argument('--freeze_block_attributes', default="", type=str,
                        help='Comma separated list of block attributes to freeze for the --freeze_blocks blocks, e.g. "norm1.weight,norm1.weight')
    parser.add_argument('--shuffle_load', type=str2bool, default=False, help='Whether to shuffle the order of blocks when loading from pretrained checkpoint, which is used to test the importance of block ordering in procedural pretraining')
    parser.add_argument('--skip_norm', type=str2bool, default=False, help='Whether to skip loading train norm from pretrained checkpoint, which is used to test the importance of train norm in procedural pretraining')
    parser.add_argument('--hold_back_blocks', default='', type=str, help='Comma separated list of layer indices to hold back from training, e.g. "0,1,2" to hold back the first 3 layers; supports "all" to hold back all layers and "" to not hold back any layers (default: "")')
    parser.add_argument('--custom_pr_load', default='', type=str, help='Custom config to control how weights are loaded from the pretrained checkpoint for procedural pretraining. Options: "L3 - keep start,end, repeat mid"')
    parser.add_argument('--random_blocks', default="", type=str,
                        help='Comma separated list of layer indices to keep random, e.g. "0,1,2" to keep the first 3 layers random; supports "all" to keep all layers random and "" to not keep any layers random (default: "")')
    # parser.add_argument('--checkpoint_num', default=-1, type=int, help='Checkpoint number to load for evaluation or resuming training, default is 299 to load the last checkpoint saved at the end of training')
    # add argument probe_mode default all
    parser.add_argument('--probe_mode', type=str, default='flatten_patches', help='defines input to the linear probe')
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--custom_lr_layer', type=bool, default=False,
                        help='Whether to use custom layer-wise learning rates, which is used for testing whether the block-wise learning rate decay contributes to the performance of procedural pretraining')
    parser.add_argument('--custom_lr_transition_start', type=int, default=0,
                        help='Epoch to start transition to custom layer-wise learning rates, used for testing whether the block-wise learning rate decay contributes to the performance of procedural pretraining')
    parser.add_argument('--custom_lr_transition_end', type=int, default=0,
                        help='Epoch to end transition to custom layer-wise learning rates, used for testing whether the block-wise learning rate decay contributes to the performance of procedural pretraining')
    parser.add_argument('--custom_block_targets_scale', type=str, default="",
                        help='Comma separated list of target scales for custom block-wise learning rates, e.g. "0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0" to set the target learning rate for the 12 blocks in convnext_small to gradually increase from 0.1x to 1x the base learning rate; supports "all" to set the same target scale for all blocks and "" to not apply custom block-wise learning rates (default: "0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0")')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--image_flip', type=str2bool, default=False,
                        help='Whether to flip the image HxW to WxH, which is used to test vertical procedural pretraining')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    parser.add_argument('--accuracy_json', type=str, default='accuracy.json')

    # * Mixup params
    
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--initialize', default='',
                        help='initialize from a model file')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module|model_state_dict|state|model_state', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/dawooda/code/procedural/data/imagenet100', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET',
                        type=str, help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_dir_B', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # Procedural init
    parser.add_argument("--procedural_data", type=str, default="kdyck",
                        help="Type of procedural data to generate (e.g., kdyck, kdyck_truncated)")
    parser.add_argument("--procedural_order", type=str, default="standard",
                        help="Type of re-ordering (if any) to apply to the  procedural data (e.g., standard, spiral)")
    parser.add_argument("--pr_notes", type=str, default="",
                        help="Additional notes about the procedural pretraining instance, to be saved in the output json and used for labelling visualisation plots")
    parser.add_argument('--pr_seed', default=42, type=int)

    parser.add_argument('--calculate_hessian', type=str2bool, default=False,
                        help='Whether to calculate hessian during training for analysis purposes, which is set to False by default to save time')

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--skip_attn_segments', type=str, default="",
                        help='format - <layer_number>[<segment1>,<segment2>];, e.g. "0[q,k,v];2[v]" to skip attention segments when visualising attention maps')

    # Procedural
    parser.add_argument("--k", type=int, default=64,
                        help="K")
    parser.add_argument("--num_blocks", type=int, default=12,
                        help="Number of transformer blocks in the model")
    parser.add_argument("--p_open", type=float, default=0.6,
                        help="Probability of opening a new bracket vs closing an existing one during generation")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="Maximum depth of nested brackets during generation")
    # parser.add_argument("--procedural_data", type=str, default="kdyck_truncated",
    #                     help="Type of procedural data to generate (e.g., kdyck, kdyck_truncated)")
    parser.add_argument("--seq_length", type=int, default=196,
                        help="Length of input sequence (e.g., 196 for 14x14 patches)")
    parser.add_argument("--mask_token", type=int, default=128,
                        help="Mask token value (e.g., 128 for k=64)")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Ratio of closing brackets to mask")
    parser.add_argument('--freeze_patch_embeddings', type=str2bool, default=True)
    parser.add_argument('--freeze_pos_embeddings', type=str2bool, default=True)
    parser.add_argument('--skip_pos_embeddings', type=str2bool, default=False)
    parser.add_argument("--embeddings_path", type=str, default="kdyck/kdyck_orthogonal_embeddings_vits.pt",
                        help="Path to fixed orthogonal embeddings for the patch tokens")
    # parser.add_argument("--procedural_order", type=str, default="standard",
    #                     help="Type of re-ordering (if any) to apply to the  procedural data (e.g., standard, spiral)")
    parser.add_argument("--shuffle", type=str, default="input",
                        help="Which part of the data to shuffle (e.g., input, pos) for different procedural orders")


    parser.add_argument('--analyse_only', type=str2bool, default=False,
                        help='Perform attention_analyse only on test set data')
    parser.add_argument('--cka_only', type=str2bool, default=False,
                        help='Perform cka only on test set data')
    parser.add_argument('--cka_compare', type=str2bool, default=False,
                        help='Perform cka only on test set data')
    parser.add_argument('--detailed_metrics', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--stage_wise_metrics', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--pr_visualise', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--ft_visualise', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--kde_l11', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--segmentation_overlay', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--segmentation_path', type=str, default="",
                        help='Perform visualisation only on test set data')
    parser.add_argument('--per_head', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--per_stage', type=str2bool, default=False,
                        help='Perform visualisation only on test set data')
    parser.add_argument('--visualise_output_path', default='', type=str,
                        help='path to save visualisation outputs, empty for no saving')
    parser.add_argument('--plot_count', default=10, type=int,
                        help='Number of samples to visualise')

    # residuals analysis
    parser.add_argument('--attention_residual_analysis', type=str2bool, default=False,
                        help="Stats on attention_residuals")
    parser.add_argument('--attention_residual_stats_path', type=str, default="",
                        help='Path to store attention residual stats')

    parser.add_argument('--custom_init_qk_scale', default=1, type=float,
                        help="Parameters to control custom init")
    parser.add_argument('--custom_init_qk_noise', default=0, type=float,
                        help="Parameters to control custom init")
    parser.add_argument('--custom_init_vp_scale', default=0.05, type=float,
                        help="Parameters to control custom init")
    parser.add_argument('--custom_init_type', default="", type=str,
                        help="Parameters to control custom init")
    parser.add_argument('--custom_init_blocks', default="", type=str,
                        help='Comma separated list of layer indices to apply custom init, e.g. "0,1,2" to apply custom init to the first 3 layers; supports "all" to apply custom init to all layers and "" to not apply custom init to any layers (default: "")')
    parser.add_argument('--save_for_analysis', default=True, type=bool,
                        help='Whether to save model checkpoints and training data for further analysis, which will be used for the paper but is set to False by default to save storage space and speed up training')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--wandb_entity_name', default='ayisharyhanadawood-universit-t-freiburg', type=str,
                        help="The name of the W&B entity where you're sending the new run.")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--notes', default='', type=str,
                        help="The notes for the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    gpu_id = int(os.environ["LOCAL_RANK"])

    block_attributes = ["norm1.weight", "norm1.bias", "attn.qkv.bias", "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.bias", "mlp.fc2.bias", "attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(args.seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    classes = dataset_train.classes if hasattr(dataset_train, 'classes') else None
    print("Number of classes = %d" % args.nb_classes)
    print("Number of training samples:", len(dataset_train))
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)
        print("Number of validation samples:", len(dataset_val))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.enable_wandb:
        wandb_logger = utils.WandbLogger(args, f"GPU {gpu_id}")
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup is activated!")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = utils.build_model(args)

    if args.freeze_blocks == "":
        args.freeze_blocks = []
    elif args.freeze_blocks == "all":
        args.freeze_blocks = list(range(len(model.blocks)))
    else:
        args.freeze_blocks = [int(x) for x in args.freeze_blocks.split(",")]

    if args.skip_load_blocks == "":
        args.skip_load_blocks = []
    elif args.skip_load_blocks == "all":
        args.skip_load_blocks = list(range(len(model.blocks)))
    else:
        args.skip_load_blocks = [int(x) for x in args.skip_load_blocks.split(",")]

    if args.freeze_block_attributes == "":
        args.freeze_block_attributes = []
    else:
        args.freeze_block_attributes = [x for x in args.freeze_block_attributes.split(",")]

    if args.skip_load_block_attributes == "":
        args.skip_load_block_attributes = []
    else:
        args.skip_load_block_attributes = [x for x in args.skip_load_block_attributes.split(",")]

    if args.random_blocks == "":
        args.random_blocks = []
    elif args.random_blocks == "all":
        args.random_blocks = list(range(len(model.blocks)))
    else:
        args.random_blocks = [int(x) for x in args.random_blocks.split(",")]

    if args.delete_blocks == "":
        args.delete_blocks = []
    elif args.delete_blocks == "all":
        args.delete_blocks = list(range(len(model.blocks)))
    else:
        args.delete_blocks = [int(x) for x in args.delete_blocks.split(",")]
    args.delete_blocks.sort(reverse=True) # sort in reverse order to avoid messing up block indices when deleting

    if args.skip_attn_segments == "":
        args.skip_attn_segments = {}
    else:
        segments = args.skip_attn_segments.split(";")
        args.skip_attn_segments = {}
        for segment in segments:
            layer_num = int(segment.split("[")[0])
            segment_names = segment.split("[")[1].split("]")[0].split(",")
            args.skip_attn_segments[layer_num] = segment_names

    for block in model.blocks:
        block.attn.fused_attn = False

    shuffled_block_order = None

    # if args.checkpoint_num != -1:
    #     args.initialize = args.output_dir + f"/checkpoint-{args.checkpoint_num}-model.pth" # default to loading the last checkpoint saved at the end of training if --initialize is not specified and --eval is True

    model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
        path = args.initialize,
        args = args,
        device = device,
        model = model
    )
    
    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model_without_ddp.parameters() if not p.requires_grad)
    print(f"Parameters - Total: {total_params:,} Trainable: {trainable:,}, Non-trainable: {non_trainable:,}")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model_without_ddp,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    num_training_steps_per_epoch = len(dataset_train) // args.total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % args.total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.pr_visualise:
        attention_visualise(
            data_loader = data_loader_val,
            model = model_without_ddp,
            device = device,
            args = args
        )
        return

    if args.attention_residual_analysis:
        os.makedirs(args.attention_residual_stats_path, exist_ok=True)
        attention_residual_analysis(
            data_loader = data_loader_val,
            model = model_without_ddp,
            device = device,
            args = args
        )
        return

    if args.cka_only:
        cka_final(data_loader_val, device, args)
        return

    if args.cka_compare:
        cka_compare(data_loader_val, device, args)
        return


    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    vit_small_probe_dims = {
        "cls": 384,
        "mean_patch": 384,
        "max_patch": 384,
        "mean_all": 384,
        "cls_mean_patch": 384,
        "concat_cls_mean_patch": 768,
        "identity": 384,
        # for 224x224 with 16x16 patches: 14x14 = 196 patch tokens
        "flatten_patches": 196 * 384,  # 75264
    }
    model_without_ddp.eval() # set model to eval mode to avoid any randomness from dropout or batchnorm, which is important for accurate evaluation of the probes and analysis of training dynamics

    master_epoch = get_master_epoch(args.initialize)
    wandb_logger.update_config("master_epoch", master_epoch)

    print(f"Running on model epoch {master_epoch}")


    probe_trainer =  LinearProbeTrainer(
        num_probes=args.num_blocks*2,
        input_dim=vit_small_probe_dims[args.probe_mode],
        num_classes=args.nb_classes,
        probe_mode=args.probe_mode,
        device=device
    )
    load_probe_trainer(probe_trainer, args.output_dir, master_epoch)

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(probe_trainer.epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        epoch_stats = []
        json_stats = {
            "epoch": epoch
        }

        running_loss = 0.0
        running_acc = torch.zeros(probe_trainer.num_probes)
        running_probe_losses = torch.zeros(probe_trainer.num_probes)

        for images, targets in data_loader_train:
            images = images.to(device)
            targets = targets.to(device)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    with utils.HookCollector(model) as acts:
                        with torch.no_grad():
                            output = model(images)
            else:
                with utils.HookCollector(model) as acts:
                    with torch.no_grad():
                        output = model(images)
            
            probe_features = []
            for bi in range(args.num_blocks):
                probe_features.append(select_probe_feature(acts[bi]["attn"], mode=args.probe_mode))
                probe_features.append(select_probe_feature(acts[bi]["blk"], mode=args.probe_mode))
                        
            step_out = probe_trainer.training_step(probe_features, targets)

            torch.cuda.synchronize()

            running_loss += step_out["loss"]
            running_acc += torch.tensor(step_out["probe_accuracies"], dtype=torch.float64).cpu()
            running_probe_losses += torch.tensor(step_out["probe_losses"], dtype=torch.float64).cpu()

        json_stats.update({
            "train_loss": running_loss / len(data_loader_train),
            "train_running_probe_acc": (running_acc / len(data_loader_train)).tolist(),
            "train_running_probe_losses": (running_probe_losses / len(data_loader_train)).tolist(),
        })
        for pi in range(probe_trainer.num_probes):
            epoch_stats.append({
                "epoch": epoch,
                "probe_loss": running_probe_losses[pi].item() / len(data_loader_train),
                "probe_acc": running_acc[pi].item() / len(data_loader_train),
                "probe_num": pi,
                "probe_name": get_probe_label(pi),
            })

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                save_probe_trainer(probe_trainer, args.output_dir, epoch, master_epoch)
            
        if data_loader_val is not None and not args.disable_eval:
            eval_results = probe_trainer.evaluate(model_without_ddp, data_loader_val, device)

            for pi in range(probe_trainer.num_probes):
                epoch_stats.append({
                    "epoch": epoch,
                    "eval_probe_acc": eval_results["probe_accuracies"][pi],
                    "eval_probe_loss": eval_results["probe_losses"][pi],
                    "probe_num": pi,
                    "probe_name": get_probe_label(pi),
                })
            json_stats.update(eval_results)
        
        append_epoch_metrics_to_json(args.accuracy_json, json_stats)

        if wandb_logger:
            for epoch_stat in epoch_stats:
                probe_num = epoch_stat.get("probe_num")
                probe_name = epoch_stat.get("probe_name")
                epoch = epoch_stat.get("epoch")

                epoch_wise = {f"Epoch-wise/{k}_probe_{probe_name}": v for k, v in epoch_stat.items() if k not in ['probe_name', 'epoch', 'probe_num']}
                epoch_wise["Epoch-wise/epoch"] = epoch
                wandb_logger._wandb.log(epoch_wise)
                layer_wise = {f"Probe-wise/{k}_epoch{epoch:02d}": v for k, v in epoch_stat.items() if k not in ['probe_name', 'epoch', 'probe_num']}
                layer_wise["Probe-wise/probe_num"] = probe_num
                layer_wise["Probe-wise/probe_name"] = probe_name
                wandb_logger._wandb.log(layer_wise)


        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(json_stats) + "\n")

        if args.model == "vit_base" and (epoch+1)!=args.epochs:
            return

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir and not args.output_dir.endswith(".pth"):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

    if dist.is_initialized():
        dist.destroy_process_group()
