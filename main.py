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
import types
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

    parser.add_argument('--accuracy_json', type=str, default='')
    parser.add_argument('--grad_norms_json', type=str, default='')

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
    parser.add_argument('--initialize_as_pr', type=str2bool, default=True)
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

    parser.add_argument('--layer_11_scale_ln', type=float, default=1.0)
    # parser.add_argument('--layer_11_scale_attn', type=float, default=1.0)
    parser.add_argument('--layer_11_scale_attn_qk', type=float, default=1.0)
    parser.add_argument('--layer_11_scale_attn_v', type=float, default=1.0)
    parser.add_argument('--layer_11_scale_attn_proj', type=float, default=1.0)
    parser.add_argument('--layer_11_target_qkvp_ln1_norm_ratio', type=float, default=-1.0)
    parser.add_argument('--layer_11_scale_method', type=str, default="")
    parser.add_argument('--init_method', type=str, default="default", help='Initialization method for the model weights. Options: "default", "match_L11_activations"')
    parser.add_argument('--init_method_scaled_blocks', type=str, default="", help='Comma separated list of layer indices to apply scaled initialization, e.g. "0,1,2" to apply scaled initialization to the first 3 layers; supports "all" to apply scaled initialization to all layers and "" to not apply scaled initialization to any layers (default: "")')
    parser.add_argument('--init_method_scaled_attributes', type=str, default="", help='Comma separated list of layer indices to apply scaled initialization, e.g. "0,1,2" to apply scaled initialization to the first 3 layers; supports "all" to apply scaled initialization to all layers and "" to not apply scaled initialization to any layers (default: "")')
    parser.add_argument('--init_method_copied_blocks', type=str, default="", help='format - <layer_number>[<segment1>,<segment2>];, e.g. "0[attn.qkv.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias];2[attn.qkv.weight,attn.qkv.bias]" to shuffle weights in the specified segments of the specified layers')
    parser.add_argument('--simultaneous_init_scaling', type=bool, default=False, help='Scale all blocks simultaneously during initialization, instead of scaling each block individually')
    parser.add_argument('--init_method_bias_scaling', type=bool, default=False, help='Scale bias too')
    parser.add_argument('--post_hoc_act_norm_track', type=bool, default=False, help='Rerun old models to track act norms before training starts')

    parser.add_argument('--weight_shuffle', type=str, default="",
                        help='format - <layer_number>[<segment1>,<segment2>];, e.g. "0[attn.qkv.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias];2[attn.qkv.weight,attn.qkv.bias]" to shuffle weights in the specified segments of the specified layers')
    parser.add_argument('--target_model_weight_shuffle', type=str, default="",
                        help='format - <layer_number>[<segment1>,<segment2>];, e.g. "0[attn.qkv.weight,attn.qkv.bias,attn.proj.weight,attn.proj.bias];2[attn.qkv.weight,attn.qkv.bias]" to shuffle weights in the specified segments of the specified layers')

    parser.add_argument('--attention_residual_scaling', type=str, default="",
                        help='format - <layer_number>[<scale_ratio>];, e.g. "0[0.5];2[0.5]" to scale residuals in the specified layers by the specified ratios')

    parser.add_argument('--attention_out_scaling', type=str, default="",
                        help='format - <layer_number>[<scale_ratio>];, e.g. "0[0.5];2[0.5]" to scale residuals in the specified layers by the specified ratios')

    parser.add_argument('--learning_rate_scaling', type=bool, default=False)
    parser.add_argument('--learning_rate_scaling_params', type=str, default="",
                        help='format = <param_name>[<scale_ratio>];, e.g. "blocks.11.attn.qkv.weight[0.5];blocks.11.attn.qkv.bias[0.5]" to scale learning rates for the specified parameters by the specified ratios')
    

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

    # if global_rank == 0 and args.log_dir is not None:
    if args.log_dir is not None:
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
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

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
            if "[" in segment and "]" in segment:
                layer_num = int(segment.split("[")[0])
                segment_names = segment.split("[")[1].split("]")[0].split(",")
                args.init_method_copied_blocks[layer_num] = segment_names
            else:
                layer_num = int(segment)
                args.init_method_copied_blocks[layer_num] = [
                    'norm1.weight',
                    # 'norm1.bias',
                    'attn.qkv.weight',
                    # 'attn.qkv.bias',
                    'attn.proj.weight',
                    # 'attn.proj.bias',
                    'norm2.weight',
                    # 'norm2.bias',
                    'mlp.fc1.weight',
                    'mlp.fc2.weight',
                    # 'mlp.fc1.bias',
                    # 'mlp.fc2.bias'
            ]

    if args.weight_shuffle == "":
        args.weight_shuffle = {}
    else:
        segments = args.weight_shuffle.split(";")
        args.weight_shuffle = {}
        for segment in segments:
            layer_num = int(segment.split("[")[0])
            segment_names = segment.split("[")[1].split("]")[0].split(",")
            args.weight_shuffle[layer_num] = segment_names

    if args.target_model_weight_shuffle == "":
        args.target_model_weight_shuffle = {}
    else:
        segments = args.target_model_weight_shuffle.split(";")
        args.target_model_weight_shuffle = {}
        for segment in segments:
            layer_num = int(segment.split("[")[0])
            segment_names = segment.split("[")[1].split("]")[0].split(",")
            args.target_model_weight_shuffle[layer_num] = segment_names

    if args.init_method_copied_blocks == "":
        args.init_method_copied_blocks = {}
    else:
        segments = args.init_method_copied_blocks.split(";")
        args.init_method_copied_blocks = {}
        for segment in segments:
            if "[" in segment and "]" in segment:
                layer_num = int(segment.split("[")[0])
                segment_names = segment.split("[")[1].split("]")[0].split(",")
                args.init_method_copied_blocks[layer_num] = segment_names
            else:
                layer_num = int(segment)
                args.init_method_copied_blocks[layer_num] = [
                    'norm1.weight',
                    'norm1.bias',
                    'attn.qkv.weight',
                    'attn.qkv.bias',
                    'attn.proj.weight',
                    'attn.proj.bias',
                    'norm2.weight',
                    'norm2.bias',
                    'mlp.fc1.weight',
                    'mlp.fc2.weight',
                    'mlp.fc1.bias',
                    'mlp.fc2.bias'
            ]

    if args.attention_residual_scaling == "":
        args.attention_residual_scaling = {}
    else:
        segments = args.attention_residual_scaling.split(";")
        args.attention_residual_scaling = {}
        for segment in segments:
            layer_num = int(segment.split("[")[0])
            scale_ratio = float(segment.split("[")[1].split("]")[0])
            args.attention_residual_scaling[layer_num] = scale_ratio

    if args.attention_out_scaling == "":
        args.attention_out_scaling = {}
    else:
        segments = args.attention_out_scaling.split(";")
        args.attention_out_scaling = {}
        for segment in segments:
            layer_num = int(segment.split("[")[0])
            scale_ratio = float(segment.split("[")[1].split("]")[0])
            args.attention_out_scaling[layer_num] = scale_ratio

    attn_scaling_elements_count, mlp_scaling_elements_count = 0, 0
    if args.init_method_scaled_blocks == "":
        args.init_method_scaled_blocks = []
    else:
        args.init_method_scaled_blocks = [int(x) for x in args.init_method_scaled_blocks.split(",")]
        if args.init_method_scaled_attributes == "":
            args.init_method_scaled_attributes = ["v", "proj","fc2"]
        else:
            args.init_method_scaled_attributes =args.init_method_scaled_attributes.split(",")
        attn_scaling_elements = set(["norm1", "qk", "v", "proj"]).intersection(set(args.init_method_scaled_attributes))
        attn_scaling_elements = set(["qkv" if x in ["qk", "v"] else x for x in attn_scaling_elements])
        attn_scaling_elements_count = len(attn_scaling_elements)

        mlp_scaling_elements = set(["norm2", "fc1", "fc2"]).intersection(set(args.init_method_scaled_attributes))
        mlp_scaling_elements_count = len(mlp_scaling_elements)

        print(f"Attention scaling elements: {attn_scaling_elements} (count: {attn_scaling_elements_count})")
        print(f"MLP scaling elements: {mlp_scaling_elements} (count: {mlp_scaling_elements_count})")

    if args.learning_rate_scaling_params == "":
        args.learning_rate_scaling_params = {}
    else:
        segments = args.learning_rate_scaling_params.split(";")
        args.learning_rate_scaling_params = {}
        for segment in segments:
            param_name = segment.split("[")[0]
            scale_ratio = float(segment.split("[")[1].split("]")[0])
            args.learning_rate_scaling_params[param_name] = scale_ratio

    for block in model.blocks:
        block.attn.fused_attn = False

    shuffled_block_order = None

    if args.init_method == "default":
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model
        )
    # elif args.init_method in ["upscale_random_match_L11_attn_norms", "upscale_random_match_L11_attn_delta_norms"]:
    elif "upscale_random" in args.init_method:
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = "",
            args = args,
            device = device,
            model = model
        )
        model_temp = utils.build_model(args)
        model_temp.load_state_dict(model_without_ddp.state_dict())
        _, target_model_without_ddp, _ = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model_temp
        )
    elif args.init_method == "match_target_block_norms":
        # Random model in which every tensor of args.init_method_scaled_blocks is rescaled
        # to the norm of the corresponding tensor in the checkpoint. Directions stay random,
        # so this isolates the per-block weight-norm profile from the learned structure.
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = "",
            args = args,
            device = device,
            model = model
        )
        model_temp = utils.build_model(args)
        model_temp.load_state_dict(model_without_ddp.state_dict())
        _, target_model_without_ddp, _ = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model_temp
        )
        with torch.no_grad():
            target_params = dict(target_model_without_ddp.named_parameters())
            for name, p in model_without_ddp.named_parameters():
                block_idx = None
                if name.startswith("blocks."):
                    try:
                        block_idx = int(name.split(".")[1])
                    except (IndexError, ValueError):
                        block_idx = None
                if block_idx is None or block_idx not in args.init_method_scaled_blocks:
                    continue
                target_p = target_params.get(name)
                if target_p is None:
                    continue
                cur_norm = p.data.norm().item()
                target_norm = target_p.data.norm().item()
                # zero-init tensors (timm inits all Linear biases to 0) cannot be rescaled
                if cur_norm <= 0 or target_norm <= 0:
                    print(f"Skipping norm match for {name} (|W|={cur_norm:.4f}, target={target_norm:.4f})", flush=True)
                    continue
                p.data.mul_(target_norm / cur_norm)
                print(f"Norm match {name}: {cur_norm:.4f} -> {target_norm:.4f} (x{target_norm/cur_norm:.4f})", flush=True)
    elif "downscale_mixed" in args.init_method:
        # target: random init, then the checkpoint loaded into every block except
        # args.random_blocks, so target and model differ only in those blocks
        target_model_without_ddp = utils.build_model(args)
        target_model_without_ddp.load_state_dict(model.state_dict())
        _, target_model_without_ddp, _ = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = target_model_without_ddp
        )
        # model: the checkpoint in all blocks
        random_blocks = args.random_blocks
        args.random_blocks = []
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model
        )
        args.random_blocks = random_blocks
    # elif args.init_method in ["downscale_pr_match_L11_attn_norms", "downscale_pr_match_L11_attn_delta_norms", "downscale_pr_match_attn_norms", "downscale_pr_match_attn_delta_norms"]:
    elif "downscale_pr" in args.init_method:
        target_model_without_ddp = utils.build_model(args)
        target_model_without_ddp.load_state_dict(model.state_dict())
        _, target_model_without_ddp, _ = utils.pr_load_model(
            path = "",
            args = args,
            device = device,
            model = target_model_without_ddp
        )
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model
        )
    # elif args.init_method in ["upscale_pr_match_L11_attn_norms", "upscale_pr_match_L11_attn_delta_norms", "upscale_pr_match_attn_norms", "upscale_pr_match_attn_delta_norms"]:
    elif "upscale_pr" in args.init_method:
        target_model_without_ddp = utils.build_model(args)
        target_model_without_ddp.load_state_dict(model.state_dict())
        _, target_model_without_ddp, _ = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = target_model_without_ddp
        )
        model, model_without_ddp, shuffled_block_order = utils.pr_load_model(
            path = args.initialize,
            args = args,
            device = device,
            model = model
        )
    elif args.init_method == "match_target_qkvp_ln1_norm_ratio":
        if args.layer_11_target_qkvp_ln1_norm_ratio < 0:
            raise ValueError("layer_11_target_qkvp_ln1_norm_ratio must be set to a positive float value when using match_target_qkvp_ln1_norm_ratio init method")
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

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    custom_block_targets = args.custom_block_targets_scale.split(",") if args.custom_block_targets_scale != "" else []
    custom_block_targets = [float(x) for x in custom_block_targets] if custom_block_targets else []
    custom_non_block_targets = {
        "patch_embed": 1.0,
        "cls_token": 1.0,
        "pos_embed": 1.0,
        "norm": 1.0,
        "head": 1.0
    }
    
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None,
        start_lr = lr_schedule_values[0],
        custom_block_targets = custom_block_targets,
        custom_non_block_targets = custom_non_block_targets,
        custom_lr_transition_start = args.custom_lr_transition_start,
        custom_lr_transition_end = args.custom_lr_transition_end
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.calculate_hessian:
        total_params = sum(p.numel() for p in model_without_ddp.parameters())
        trainable = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in model_without_ddp.parameters() if not p.requires_grad)
        print(f"bf: Parameters - Total: {total_params:,} Trainable: {trainable:,}, Non-trainable: {non_trainable:,}")

        if args.start_epoch==0:
            eigenvalues_1, eigenvectors_1, meta_1, layer_hessian_values = compute_hessian_at_initialization(
                model_without_ddp, criterion, data_loader_train, num_eigenvalues=10, device=device
            )
            print(f"Hessian top eigenvalues at initialization: {eigenvalues_1}")
            layer_hessian_values["overall"] = eigenvalues_1

            trace_1 = compute_hessian_trace(model_without_ddp, criterion, data_loader_train, device=device)
            print(f"Hessian trace at initialization: {trace_1}")
            layer_hessian_values["trace"] = trace_1

            # save in destination folder
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                hessian_stats_path = os.path.join(args.output_dir, "hessian_stats_initialization.json")
                with open(hessian_stats_path, "w") as f:
                    json.dump(layer_hessian_values, f, indent=4)
        return

    if args.analyse_only:
        print(f"Attention analyse only mode")
        stats = attention_analyse_final(data_loader_val, device, args=args, classes=classes)
        return stats

    if not args.post_hoc_act_norm_track:
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    else:
        print(f"Post-hoc act norm tracking mode, start_epoch={args.start_epoch}")

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    if args.start_epoch==0 and shuffled_block_order is not None:
        wandb_logger.update_config("block_order", shuffled_block_order)

    if args.start_epoch==0 and args.custom_lr_layer:
        for bi, blrs in enumerate(custom_block_targets):
            wandb_logger.update_config(f"block_{bi}_scale", blrs)
        for npname, nplrs in custom_non_block_targets.items():
            wandb_logger.update_config(f"{npname}_scale", nplrs)

    if len(args.attention_residual_scaling)>0 or len(args.attention_out_scaling)>0:
        for layer_num in range(len(model_without_ddp.blocks)):
            block_item = model_without_ddp.blocks[layer_num]
            block_item.attn_res_scale = args.attention_residual_scaling.get(layer_num, 1.0)
            block_item.attn_out_scale = args.attention_out_scaling.get(layer_num, 1.0)
            block_item.forward = types.MethodType(utils.patched_last_block_forward, block_item)

    if args.start_epoch==0:
        if len(args.weight_shuffle) > 0:
            utils.shuffle_weights(model_without_ddp, args.weight_shuffle)

        if len(args.target_model_weight_shuffle) > 0:
            utils.shuffle_weights(target_model_without_ddp, args.target_model_weight_shuffle)

        if args.layer_11_scale_method != "":
            scale_weights = {
                "norm1": args.layer_11_scale_ln,
                "qk": args.layer_11_scale_attn_qk,
                "v": args.layer_11_scale_attn_v,
                "proj": args.layer_11_scale_attn_proj
            }
            utils.scale_layer_weights(model_without_ddp, [11], scale_weights, args.init_method_bias_scaling)

        n = len(dataset_train)
        k = 5000
        print(f"Using {k} samples from the training dataset for attention residual analysis", flush=True)

        indices = list(range(n))
        random.shuffle(indices)
        subset_indices = indices[:k]

        dataset_ref_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=torch.utils.data.SubsetRandomSampler(subset_indices),
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # loop through init_method_copied_blocks and copy the weights from the target model to the current model for the specified segments
        if args.init_method_copied_blocks:
            edited_weights = {}
            target_model_weights = target_model_without_ddp.state_dict()
            for block_idx, segments in args.init_method_copied_blocks.items():
                for segment in segments:
                    print(f"Copying weights for layer {block_idx} segment {segment} from target model to current model", flush=True)
                    edited_weights[f"blocks.{block_idx}.{segment}"] = target_model_weights[f"blocks.{block_idx}.{segment}"]
            utils.load_state_dict(model_without_ddp, edited_weights)
            
        if args.init_method in ["downscale_pr_match_attn_delta_norms", "upscale_random_match_attn_delta_norms", "upscale_pr_match_attn_delta_norms", "downscale_mixed_match_attn_delta_norms"]:
            if utils.is_main_process():
                target_stats = attention_residual_analysis(
                    data_loader = dataset_ref_loader,
                    model = target_model_without_ddp,
                    device = device,
                    args = args,
                    save=True,
                    filename="target_res_stats",
                    layers_to_analyse = args.init_method_scaled_blocks,
                    detailed=False
                )
            else:
                target_stats = {}
            if args.simultaneous_init_scaling:
                target_stats = broadcast_dict([target_stats], src=0)[0]

                if utils.is_main_process():
                    current_stats = attention_residual_analysis(
                        data_loader = dataset_ref_loader,
                        model = model_without_ddp,
                        device = device,
                        args = args,
                        save=False,
                        layers_to_analyse = args.init_method_scaled_blocks,
                        detailed=False
                    )
                else:
                    current_stats = {}
                current_stats = broadcast_dict([current_stats], src=0)[0]

                for block_idx in args.init_method_scaled_blocks:
                    scale_sq = target_stats[block_idx]["norm_ratio_attn_delta_in_mean"]/current_stats[block_idx]["norm_ratio_attn_delta_in_mean"]
                    print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                    # scale = math.sqrt(scale_sq)
                    scale = scale_sq ** (1/attn_scaling_elements_count) if attn_scaling_elements_count > 0 else 1.0
                    print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                    scale_weights = {
                        "norm1": scale if "norm1" in args.init_method_scaled_attributes else 1.0,
                        "qk": scale if "qk" in args.init_method_scaled_attributes else 1.0,
                        "v": scale if "v" in args.init_method_scaled_attributes else 1.0,
                        "proj": scale if "proj" in args.init_method_scaled_attributes else 1.0
                    }
                    for s, sw in scale_weights.items():
                        wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                    wandb_logger.update_config(f"current_layer_{block_idx}_attn_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                    wandb_logger.update_config(f"target_layer_{block_idx}_attn_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                    utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling)

            else:    
                for block_idx in args.init_method_scaled_blocks:
                    if utils.is_main_process():
                        current_stats = attention_residual_analysis(
                            data_loader = dataset_ref_loader,
                            model = model_without_ddp,
                            device = device,
                            args = args,
                            save=False,
                            layers_to_analyse = [block_idx],
                            detailed=False
                        )
                        # current_stats = {}
                        scale_sq = target_stats[block_idx]["norm_ratio_attn_delta_in_mean"]/current_stats[block_idx]["norm_ratio_attn_delta_in_mean"]
                        print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                        # scale = math.sqrt(scale_sq)
                        scale = scale_sq ** (1/attn_scaling_elements_count) if attn_scaling_elements_count > 0 else 1.0
                        print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                        scale_weights = {
                            "norm1": scale if "norm1" in args.init_method_scaled_attributes else 1.0,
                            "qk": scale if "qk" in args.init_method_scaled_attributes else 1.0,
                            "v": scale if "v" in args.init_method_scaled_attributes else 1.0,
                            "proj": scale if "proj" in args.init_method_scaled_attributes else 1.0
                        }
                        for s, sw in scale_weights.items():
                            wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                        wandb_logger.update_config(f"current_layer_{block_idx}_attn_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                        wandb_logger.update_config(f"target_layer_{block_idx}_attn_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_attn_delta_in_mean"])

                    else:
                        scale_weights = {}
                    torch.distributed.barrier()
                    print(utils.get_rank(), f"Torch barrier: Scale weights for layer {block_idx}: {scale_weights}", flush=True)

                    scale_weights = broadcast_dict([scale_weights], src=0)[0]
                    utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling) 
        
        if args.init_method in ["downscale_pr_match_mlp_delta_norms", "upscale_random_match_mlp_delta_norms", "upscale_pr_match_mlp_delta_norms", "downscale_mixed_match_mlp_delta_norms"]:
            if utils.is_main_process():
                target_stats = attention_residual_analysis(
                    data_loader = dataset_ref_loader,
                    model = target_model_without_ddp,
                    device = device,
                    args = args,
                    save=True,
                    filename="target_res_stats",
                    layers_to_analyse = args.init_method_scaled_blocks,
                    detailed=False

                )
            else:
                target_stats = {}
            if args.simultaneous_init_scaling:
                target_stats = broadcast_dict([target_stats], src=0)[0]

                if utils.is_main_process():
                    current_stats = attention_residual_analysis(
                        data_loader = dataset_ref_loader,
                        model = model_without_ddp,
                        device = device,
                        args = args,
                        save=False,
                        layers_to_analyse = args.init_method_scaled_blocks,
                        detailed=False
                    )
                else:
                    current_stats = {}
                current_stats = broadcast_dict([current_stats], src=0)[0]

                for block_idx in args.init_method_scaled_blocks:
                    scale_sq = target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]/current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]
                    print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                    # scale = math.sqrt(scale_sq)
                    scale = scale_sq ** (1/mlp_scaling_elements_count) if mlp_scaling_elements_count > 0 else 1.0
                    print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                    scale_weights = {
                        "norm2": scale if "norm2" in args.init_method_scaled_attributes else 1.0,
                        "fc1": scale if "fc1" in args.init_method_scaled_attributes else 1.0,
                        "fc2": scale if "fc2" in args.init_method_scaled_attributes else 1.0
                    }

                    for s, sw in scale_weights.items():
                        wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                    wandb_logger.update_config(f"current_layer_{block_idx}_mlp_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                    wandb_logger.update_config(f"target_layer_{block_idx}_mlp_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                    utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling)

            else:    
                for block_idx in args.init_method_scaled_blocks:
                    if utils.is_main_process():
                        current_stats = attention_residual_analysis(
                            data_loader = dataset_ref_loader,
                            model = model_without_ddp,
                            device = device,
                            args = args,
                            save=False,
                            layers_to_analyse = [block_idx],
                            detailed=False
                        )
                        # current_stats = {}
                        scale_sq = target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]/current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]
                        print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                        # scale = math.sqrt(scale_sq)
                        scale = scale_sq ** (1/mlp_scaling_elements_count) if mlp_scaling_elements_count > 0 else 1.0
                        print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                        scale_weights = {
                            "norm2": scale if "norm2" in args.init_method_scaled_attributes else 1.0,
                            "fc1": scale if "fc1" in args.init_method_scaled_attributes else 1.0,
                            "fc2": scale if "fc2" in args.init_method_scaled_attributes else 1.0
                        }

                        for s, sw in scale_weights.items():
                            wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                        wandb_logger.update_config(f"current_layer_{block_idx}_mlp_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                        wandb_logger.update_config(f"target_layer_{block_idx}_mlp_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])

                    else:
                        scale_weights = {}
                    torch.distributed.barrier()
                    print(utils.get_rank(), f"Torch barrier: Scale weights for layer {block_idx}: {scale_weights}", flush=True)

                    scale_weights = broadcast_dict([scale_weights], src=0)[0]
                    utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling) 
    
        if args.init_method in ["downscale_pr_match_delta_norms", "upscale_random_match_delta_norms", "upscale_pr_match_delta_norms", "downscale_mixed_match_delta_norms"]:
            if utils.is_main_process():
                target_stats = attention_residual_analysis(
                    data_loader = dataset_ref_loader,
                    model = target_model_without_ddp,
                    device = device,
                    args = args,
                    save=True,
                    filename="target_res_stats",
                    layers_to_analyse = args.init_method_scaled_blocks,
                    detailed=False
                )
            else:
                target_stats = {}
            if args.simultaneous_init_scaling:
                target_stats = broadcast_dict([target_stats], src=0)[0]

                if utils.is_main_process():
                    current_stats = attention_residual_analysis(
                        data_loader = dataset_ref_loader,
                        model = model_without_ddp,
                        device = device,
                        args = args,
                        save=False,
                        layers_to_analyse = args.init_method_scaled_blocks,
                        detailed=False
                    )
                else:
                    current_stats = {}
                current_stats = broadcast_dict([current_stats], src=0)[0]

                for block_idx in args.init_method_scaled_blocks:
                    for sub_idx in ["attn", "mlp"]:
                        if sub_idx== "attn": # attention
                            scale_sq = target_stats[block_idx]["norm_ratio_attn_delta_in_mean"]/current_stats[block_idx]["norm_ratio_attn_delta_in_mean"]
                            print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                            # scale = math.sqrt(scale_sq)
                            scale = scale_sq ** (1/attn_scaling_elements_count) if attn_scaling_elements_count > 0 else 1.0
                            print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                            scale_weights = {
                                "norm1": scale if "norm1" in args.init_method_scaled_attributes else 1.0,
                                "qk": scale if "qk" in args.init_method_scaled_attributes else 1.0,
                                "v": scale if "v" in args.init_method_scaled_attributes else 1.0,
                                "proj": scale if "proj" in args.init_method_scaled_attributes else 1.0
                            }
                            for s, sw in scale_weights.items():
                                wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                            wandb_logger.update_config(f"current_layer_{block_idx}_attn_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                            wandb_logger.update_config(f"target_layer_{block_idx}_attn_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_attn_delta_in_mean"])

                        elif sub_idx == "mlp": # mlp
                            scale_sq = target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]/current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]
                            print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                            # scale = math.sqrt(scale_sq)
                            scale = scale_sq ** (1/mlp_scaling_elements_count) if mlp_scaling_elements_count > 0 else 1.0
                            print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                            scale_weights = {
                                "norm2": scale if "norm2" in args.init_method_scaled_attributes else 1.0,
                                "fc1": scale if "fc1" in args.init_method_scaled_attributes else 1.0,
                                "fc2": scale if "fc2" in args.init_method_scaled_attributes else 1.0
                            }
                            for s, sw in scale_weights.items():
                                wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                            wandb_logger.update_config(f"current_layer_{block_idx}_mlp_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                            wandb_logger.update_config(f"target_layer_{block_idx}_mlp_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])

                        utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling)

            else:    
                for block_idx in args.init_method_scaled_blocks:
                    for sub_idx in ["attn", "mlp"]:
                        if utils.is_main_process():
                            current_stats = attention_residual_analysis(
                                data_loader = dataset_ref_loader,
                                model = model_without_ddp,
                                device = device,
                                args = args,
                                save=False,
                                layers_to_analyse = [block_idx],
                                detailed=False
                            )
                            # current_stats = {}
                            if sub_idx== "attn": # attention
                                scale_sq = target_stats[block_idx]["norm_ratio_attn_delta_in_mean"]/current_stats[block_idx]["norm_ratio_attn_delta_in_mean"]
                                print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                                # scale = math.sqrt(scale_sq)
                                scale = scale_sq ** (1/attn_scaling_elements_count) if attn_scaling_elements_count > 0 else 1.0
                                print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)
                                scale_weights = {
                                    "norm1": scale if "norm1" in args.init_method_scaled_attributes else 1.0,
                                    "qk": scale if "qk" in args.init_method_scaled_attributes else 1.0,
                                    "v": scale if "v" in args.init_method_scaled_attributes else 1.0,
                                    "proj": scale if "proj" in args.init_method_scaled_attributes else 1.0
                                }
                                for s, sw in scale_weights.items():
                                    wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                                wandb_logger.update_config(f"current_layer_{block_idx}_attn_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                                wandb_logger.update_config(f"target_layer_{block_idx}_attn_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                            elif sub_idx == "mlp": # mlp
                                scale_sq = target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]/current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"]
                                print(f"Scale squared for layer {block_idx} delta norm ratio: {scale_sq}", flush=True)
                                # scale = math.sqrt(scale_sq)
                                scale = scale_sq ** (1/mlp_scaling_elements_count) if mlp_scaling_elements_count > 0 else 1.0
                                print(f"Scale for layer {block_idx} delta norm ratio: {scale}", flush=True)

                                scale_weights = {
                                    "norm2": scale if "norm2" in args.init_method_scaled_attributes else 1.0,
                                    "fc1": scale if "fc1" in args.init_method_scaled_attributes else 1.0,
                                    "fc2": scale if "fc2" in args.init_method_scaled_attributes else 1.0
                                }
                                for s, sw in scale_weights.items():
                                    wandb_logger.update_config(f"calculated_layer_{block_idx}_scale_{s}", sw)
                                wandb_logger.update_config(f"current_layer_{block_idx}_mlp_delta_in_norm_ratio", current_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                                wandb_logger.update_config(f"target_layer_{block_idx}_mlp_delta_in_norm_ratio", target_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])

                        else:
                            scale_weights = {}

                        torch.distributed.barrier()
                        print(utils.get_rank(), f"Torch barrier: Scale weights for layer {block_idx}: {scale_weights}", flush=True)

                        scale_weights = broadcast_dict([scale_weights], src=0)[0]
                        utils.scale_layer_weights(model_without_ddp, [block_idx], scale_weights, args.init_method_bias_scaling) 

        print("Completed initialization and scaling of weights based on attention residual analysis")
        # vit_base skips this during training for speed; post-hoc analysis jobs still need it
        if args.model != "vit_base" or args.post_hoc_act_norm_track:
            updated_stats = attention_residual_analysis(
                data_loader = dataset_ref_loader,
                model = model_without_ddp,
                device = device,
                args = args,
                layers_to_analyse = None,
                filename="final_res_stats",
                detailed=False
            )
            for block_idx in range(len(model_without_ddp.blocks)):
                wandb_logger.update_config(f"final_layer_{block_idx}_attn_delta_in_norm_ratio", updated_stats[block_idx]["norm_ratio_attn_delta_in_mean"])
                wandb_logger.update_config(f"final_layer_{block_idx}_mlp_delta_in_norm_ratio", updated_stats[block_idx]["norm_ratio_mlp_delta_in_mean"])
                # wandb_logger.update_config(f"final_attn_delta_in_norm_ratio", updated_stats[11]["norm_ratio_attn_delta_in_mean"])
        print("Custom model analysis before training")
        model_analyse(
            model=model_without_ddp,
            data_loader=data_loader_val,
            device=device,
            epoch=None,
            args=args,
            prefix="custom_",
            wandb_logger=wandb_logger,
        )

    if args.post_hoc_act_norm_track: return
    # return
    
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()

        train_stats, parameter_norm = train_one_epoch(
            model, model_without_ddp, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp,
            custom_lr_layer=args.custom_lr_layer,
            custom_lr_transition_start=args.custom_lr_transition_start,
            custom_lr_transition_end=args.custom_lr_transition_start,
            custom_block_targets=custom_block_targets,
            custom_non_block_targets=custom_non_block_targets, args=args
        )
        if utils.is_main_process():
            parameter_norm["epoch"] = epoch
            parameter_norm["seed"] = args.seed
            if args.grad_norms_json != "":
                current_grad_norms_list = []
                if os.path.exists(args.grad_norms_json):
                    with open(args.grad_norms_json, 'r') as f:
                        current_grad_norms_list = json.load(f)
                current_grad_norms_list.append(parameter_norm)
                with open(args.grad_norms_json, 'w') as f:
                    json.dump(current_grad_norms_list, f, indent=4)

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            if args.save_for_analysis:
                checkpoint_path = Path(args.output_dir) / ('checkpoint-%s-model.pth' % str(epoch))
                torch.save({k: v.half() for k, v in model_without_ddp.state_dict().items()}, checkpoint_path)
            
        if data_loader_val is not None and \
        ((args.model == "vit_base" and (epoch+1)%5 == 0) or (args.model != "vit_base")):
            if (epoch+1)%10 == 0 or epoch < 20:
                test_stats, stats = model_analyse(
                    model=model_without_ddp, 
                    data_loader=data_loader_val, 
                    device=device, 
                    epoch=epoch, 
                    args=args, 
                    prefix="", 
                    shuffled_block_order=None,
                    parameter_norm=parameter_norm,
                    wandb_logger=wandb_logger
                )
            else:
                test_stats = evaluate(data_loader_val, model_without_ddp, device, use_amp=args.use_amp)
                # train_stats_temp = evaluate(data_loader_train, model_without_ddp, device, use_amp=args.use_amp)
            if args.model != "vit_base" and (epoch+1)%20 == 0:
                train_stats_temp = evaluate(data_loader_train, model_without_ddp, device, use_amp=args.use_amp)
            else:
                train_stats_temp = None

            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            if train_stats_temp is not None:
                log_stats_train = {
                    f'train_eval_{k}': v for k, v in train_stats_temp.items()
                }
                log_stats.update(log_stats_train)

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            print("Logging metrics to wandb")
            wandb_logger.log_epoch_metrics(log_stats)

        # if args.model == "vit_base" and (epoch+1)!=args.epochs and (epoch+1)%2 == 0:
        #     return
        # if args.learning_rate_scaling: return 

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # model_analyse(
    #     model=model_without_ddp,
    #     data_loader=data_loader_val, 
    #     device=device, 
    #     epoch=None, 
    #     args=args, 
    #     prefix="", 
    #     shuffled_block_order=None
    # )
    attention_analyse_final(
        data_loader=data_loader_val, 
        device=device, 
        args=args, 
        classes=classes,
        wandb_logger=wandb_logger
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir and not args.output_dir.endswith(".pth"):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.accuracy_json and not args.accuracy_json.endswith(".json"):
        Path(args.accuracy_json).parent.mkdir(parents=True, exist_ok=True)
    elif args.accuracy_json and args.accuracy_json.endswith(".json"):
        Path(args.accuracy_json).parent.mkdir(parents=True, exist_ok=True)
    if args.grad_norms_json == "":
        existing_filename = args.accuracy_json.split("/")[-1]
        args.grad_norms_json = args.accuracy_json.replace(existing_filename, "gn_"+existing_filename)
    main(args)

    if dist.is_initialized():
        dist.destroy_process_group()
