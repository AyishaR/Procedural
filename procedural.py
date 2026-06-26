import argparse
import random
import glob
import logging
import math
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
import wandb
from datetime import datetime
from pprint import pprint
from torch.distributed import (
    ReduceOp,
    barrier,
    destroy_process_group,
    init_process_group,
    reduce,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from models.vitp import VitProcedural
from kdyck.kdyck_generation import *
from kdyck.utils import *
from kdyck.kdyck_dataset import *
from procedural_data.repeat_dataset import *
import utils

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"

def ddp_setup():
    """
    Initialize DDP and set device id
    """
    rank = int(os.environ["LOCAL_RANK"])
    print(f"\n[Rank {rank}]: CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all GPUs visible)')}")
    print(f"\n[Rank {rank}]: CUDA available: {torch.cuda.is_available()}")
    print(f"\n[Rank {rank}]: Visible device count: {torch.cuda.device_count()}")

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank)
    print(f"\n[Rank {rank}]: DDP setup complete. Process group initialized with backend 'nccl'.")

def set_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)  # Per-process offset for data shuffling
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class Trainer:
    def __init__(
        self,
        args
    ):
        """
        Initialize class attributes

        1. Fix number of GPUs to be used for training. Assign GPU ID for self and the feature bank.
        2. Initialise all necessary training objects.
        3. Fix start epoch. Load from state(s) from checkpoint if necessary.
        4. Create DDP instance of the model.
        5. Initialize model wrapper for validation.
        """
        self.args = args
        
        self.n_gpus = torch.cuda.device_count()
        self.gpu_id = int(os.environ["LOCAL_RANK"])

        if self.gpu_id == 0:
            pprint(args)

        self.run_stats = wandb.init(
            entity=self.args.wandb_entity_name,
            project=self.args.wandb_project_name,
            config=self.args,
            name=f"GPU {self.gpu_id}",
            notes=self.args.wandb_notes
        )

        self.start_step = 0

        self.initialize_training_objects()    

        self.vitp_model = DDP(self.vitp_model.cuda(), device_ids=[self.gpu_id], find_unused_parameters=True)
        print(f"[GPU{self.gpu_id}]: DDP model initialized")
        assert hasattr(self.vitp_model, 'module'), "DDP model does not have 'module' attribute. Check DDP initialization." 

    def initialize_training_objects(self):
        """
        Initialize all training objects.

        Feature bank, Dataset and its loaders, Model, Optimizer and Loss.
        """
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # targets, masked = load_dataset(self.args.test_dataset)

        # val_set = torch.utils.data.TensorDataset(
        #     torch.tensor(masked, dtype=torch.long),
        #     torch.tensor(targets, dtype=torch.long)
        # )

        self.vitp_model = VitProcedural(self.args).cuda()
        for i in range(len(self.vitp_model.model.blocks)-self.args.num_blocks):
            del self.vitp_model.model.blocks[-1]

        if self.args.initialize:
            state_dict = torch.load(self.args.initialize, weights_only=False)["state"]
            self.model.load_state_dict(state_dict)
            print(f"Model initialized from {self.args.initialize}")

        total_params = sum(p.numel() for p in self.vitp_model.model.parameters())
        trainable = sum(p.numel() for p in self.vitp_model.model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.vitp_model.model.parameters() if not p.requires_grad)
        if self.gpu_id == 0:
            print(f"[GPU{self.gpu_id}]: Model initialized with {len(self.vitp_model.model.blocks)} blocks")
            print(f"Parameters - Total: {total_params:,} Trainable: {trainable:,}, Non-trainable: {non_trainable:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.vitp_model.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.000001, 
            end_factor=1.0, 
            total_iters=self.args.warmup_steps)
        anneal_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.args.training_steps-self.args.warmup_steps, 
            eta_min=self.args.eta_min_ratio*self.args.lr
            )
        self.lr_scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, anneal_scheduler], milestones=[self.args.warmup_steps])

        # Loss
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none').cuda()

        if self.args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(self.args.output_dir, '*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('_')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                ckpt_path = os.path.join(self.args.output_dir, f'pr_{self.args.slurm_id}_{latest_ckpt}.pth')
                print("Auto resume checkpoint: %s" % ckpt_path)

                checkpoint = torch.load(ckpt_path, weights_only=False)
                print("Checkpoint loaded. Keys in checkpoint:", checkpoint.keys())
                self.vitp_model.model.load_state_dict(checkpoint["state"], strict=False)

                if 'optimizer' in checkpoint and 'epoch' in checkpoint and 'lr_scheduler' in checkpoint:
                    self.start_step = checkpoint['epoch'] + 1
                    print(f"Starting from step {self.start_step}")
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    print("With optim & sched!")
            else:
                print("No checkpoint found for auto-resume. Starting from scratch.")

        if "kdyck" in self.args.procedural_data:
            self.dataset = KDyckDataset(self.args)
            self.mask_function = mask_kdyck_dataset
        elif "repeat" in self.args.procedural_data:
            self.dataset = RepeatDataset(self.args)
            self.mask_function = mask_repeat_dataset
        else:
            raise ValueError(f"Unknown procedural data type: {self.args.procedural_data}")
        self.sampler_train = torch.utils.data.DistributedSampler(
            self.dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True, seed=self.args.seed,
        )
        self.loader = DataLoader(
            self.dataset,
            sampler=self.sampler_train,
            batch_size=self.args.batch_size
        )
        self.dataset_iter = iter(self.loader)

        print(f"[GPU{self.gpu_id}]: All training objects initialized")

    def generate_kdyck_batch(self):
        try:
            targets = next(self.dataset_iter)
        except StopIteration:
            self.dataset_iter = iter(self.loader)
            targets = next(self.dataset_iter)
        # inputs = mask_kdyck_dataset(targets, mask_token=self.args.mask_token, mask_prob=self.args.mask_ratio, close_brack_start_token=self.args.k)
        inputs = self.mask_function(targets, mask_token=self.args.mask_token, mask_prob=self.args.mask_ratio)

        inputs = inputs.cuda()
        targets = targets.cuda()

        if self.args.procedural_order == "standard":
            pass
        elif self.args.procedural_order == "spiral":
            inputs = spiral_unravel(inputs)
            targets = spiral_unravel(targets)
        elif self.args.procedural_order == "vertical":
            inputs = vertical_unravel(inputs)
            targets = vertical_unravel(targets)
        elif self.args.procedural_order == "row_alternate":
            inputs = row_alternate(inputs)
            targets = row_alternate(targets)
        elif self.args.procedural_order == "row_alternate_half":
            inputs = row_alternate_half(inputs)
            targets = row_alternate_half(targets)
        elif self.args.procedural_order == "row_alternate_quarter":
            inputs = row_alternate_quarter(inputs)
            targets = row_alternate_quarter(targets)
        elif self.args.procedural_order == "waterfall":
            inputs = waterfall(inputs)
            targets = waterfall(targets)
        elif self.args.procedural_order == "waterfall_half":
            inputs = waterfall_half(inputs)
            targets = waterfall_half(targets)
        else:
            raise ValueError(f"Unknown procedural order type: {self.args.procedural_order}")

        return targets, inputs
        
    def training(self):
        """
        Perform training according to the configuration in args.
        """
        print("Start Training!")

        print(f"[GPU{self.gpu_id}]: Training from Epoch {self.start_step} to {self.args.training_steps}")

        self.vitp_model.module.set_train()

        for epoch in range(self.start_step, self.args.training_steps):
            self.loader.sampler.set_epoch(epoch)
            for batch_step in range(self.args.update_freq):
                with self.vitp_model.join():
                    targets, inputs = self.generate_kdyck_batch()

                    masked_count = (inputs == self.args.mask_token).sum().item()

                    outputs = self.vitp_model(inputs)

                    mask = inputs == self.args.mask_token

                    logits = outputs.argmax(dim=-1)
                    acc_mask = targets == logits
                    masked_acc = acc_mask[mask].sum()/mask.sum()
                    non_masked_acc = acc_mask[~mask].sum()/(~mask).sum()
                    acc = acc_mask.sum()/acc_mask.numel()

                    loss_per_pos = self.ce_loss(outputs.transpose(1,2), targets)

                    masked_loss = loss_per_pos*mask
                    
                    loss = masked_loss.sum() / mask.sum()
                    
                    loss /= self.args.update_freq
                    loss.backward()
                    if (batch_step + 1) % self.args.update_freq == 0:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                    
                    print(f"[GPU{self.gpu_id}]: Step {epoch} - loss={loss}; Acc={acc}; Masked Acc={masked_acc}; Non-masked Acc={non_masked_acc}")

                    log_dict = {
                                "train_epoch": epoch+1,
                                "train_loss": loss.item(),
                                "gpu_id": self.gpu_id,
                                "masked_acc": masked_acc,
                                "non_masked_acc": non_masked_acc,
                                "acc": acc,
                                "lr": self.optimizer.param_groups[0]['lr'],
                                "masked_token_count": masked_count
                            }
                    self.run_stats.log(
                        log_dict
                    )

                    print(f"[GPU{self.gpu_id}]: Epoch {epoch} completed.")

                    if (epoch + 1) % self.args.save_every == 0 and self.gpu_id == 0:
                        torch.save(
                            {
                                "state": self.vitp_model.module.model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),  
                                "lr_scheduler": self.lr_scheduler.state_dict(),
                                "epoch": epoch, 
                            },
                            os.path.join(self.args.output_dir, f"pr_{self.args.slurm_id}_{epoch}.pth")
                        )
                        print(f"[GPU{self.gpu_id}]: Epoch {epoch} | Training snapshot saved")


                    print(f"[GPU{self.gpu_id}]: Finished epoch {epoch}. Before synchronization/training.")
                torch.cuda.synchronize(device=self.gpu_id)
                print(f"[GPU{self.gpu_id}]: Finished epoch {epoch}. After synchronization/training.")

                print(f"[GPU{self.gpu_id}]: Reached training barrier at Epoch {epoch}")
                torch.distributed.barrier(device_ids=[self.gpu_id])
                print(f"[GPU{self.gpu_id}]: Crossed training barrier at Epoch {epoch}")

        # self.vitp_model.module.set_eval()
        # self.validate(epoch+1)
        # self.vitp_model.module.set_train()

        if self.gpu_id == 0:
            torch.save(
                {
                    "state": self.vitp_model.module.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),  
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": epoch+1, 
                },
                os.path.join(self.args.output_dir, f"pr_{self.args.slurm_id}_final.pth")
            )

        print(f"[GPU{self.gpu_id}]: Reached training barrier at Epoch {epoch}")
        torch.distributed.barrier(device_ids=[self.gpu_id])
        print(f"[GPU{self.gpu_id}]: Crossed training barrier at Epoch {epoch}")


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="Training Procedural Warm-up")

    parser = argparse.ArgumentParser()

    # Top-level training config
    parser.add_argument('--slurm_id', type=int)

    parser.add_argument("--model", type=str, default="vit_tiny_patch16_224",
                        help="Model name or config (e.g., vit_tiny_patch16_224)")

    parser.add_argument("--output_dir", type=str, default="./procedural_ckpts")
    parser.add_argument('--initialize', default='',
                        help='initialize from a model file')

    parser.add_argument('--auto_resume', type=str2bool, default=True)

    # Procedural
    parser.add_argument("--k", type=int, default=64,
                        help="K")
    parser.add_argument("--num_blocks", type=int, default=12,
                        help="Number of transformer blocks in the model")
    parser.add_argument("--p_open", type=float, default=0.6,
                        help="Probability of opening a new bracket vs closing an existing one during generation")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="Maximum depth of nested brackets during generation")
    parser.add_argument("--procedural_data", type=str, default="kdyck",
                        help="Type of procedural data to generate (e.g., kdyck, kdyck_truncated)")
    parser.add_argument("--seq_length", type=int, default=196,
                        help="Length of input sequence (e.g., 196 for 14x14 patches)")
    parser.add_argument("--mask_token", type=int, default=128,
                        help="Mask token value (e.g., 128 for k=64)")
    parser.add_argument("--mask_ratio", type=float, default=0.5,
                        help="Ratio of closing brackets to mask")
    parser.add_argument('--freeze_patch_embeddings', type=str2bool, default=True)
    parser.add_argument('--freeze_pos_embeddings', type=str2bool, default=True)
    parser.add_argument('--skip_pos_embeddings', type=str2bool, default=False)
    parser.add_argument("--embeddings_path", type=str, default="kdyck/kdyck_orthogonal_embeddings_vitt.pt",
                        help="Path to fixed orthogonal embeddings for the patch tokens")
    parser.add_argument("--procedural_order", type=str, default="standard",
                        help="Type of re-ordering (if any) to apply to the  procedural data (e.g., standard, spiral)")
    parser.add_argument("--shuffle", type=str, default="input",
                        help="Which part of the data to shuffle (e.g., input, pos) for different procedural orders")

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument('--total_batch_size', default=64, type=int,
                        help='Effective batch size')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")

    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate (default: 2e-3)')
    # parser.add_argument('--layer_decay', type=float, default=1.0)
    # parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--eta_min_ratio', type=float, default=0.0,
                        help='eta_min ratio for cosine schedulers (default: 0.0)')
    parser.add_argument("--training_steps", type=int, default=15000,
                        help="Number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")

    # Val 
    # parser.add_argument("--test_dataset", type=str, default="kdyck/kdyck_dataset_test.npz",
    #                     help="Path to test dataset")
    # parser.add_argument("--val_batch_size", type=int, default=128,
    #                     help="Batch size")

    parser.add_argument('--wandb_entity_name', default='ayisharyhanadawood-universit-t-freiburg', type=str,
                        help="The name of the W&B entity where you're sending the new run.")
    parser.add_argument('--wandb_project_name', default='procedural_training', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_notes', default='procedural_training', type=str,
                        help="The notes for the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', action="store_true",
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed value")

    args = parser.parse_args()
    pprint(args)
    
    set_seed(args.seed, int(os.environ["LOCAL_RANK"]))

    try:
        ddp_setup()
        
        trainer = Trainer(args)
        trainer.training()
        trainer.run_stats.finish()

    finally:
        destroy_process_group()
    
