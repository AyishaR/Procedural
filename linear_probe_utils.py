import torch
import torch.nn as nn
from typing import List, Dict, Optional
from utils import *
from pathlib import Path

class LinearProbeTrainer(nn.Module):
    def __init__(
        self,
        num_probes: int,
        input_dim: int,
        num_classes: int,
        probe_mode="cls",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.num_probes = num_probes
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device if device is not None else torch.device("cpu")
        self.probe_mode = probe_mode

        self.probes = nn.ModuleList([
            nn.Linear(input_dim, num_classes) for _ in range(num_probes)
        ])

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.probes.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.epoch = 0
        self.to(self.device)

    def forward(self, probe_features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(probe_features) != self.num_probes:
            raise ValueError(
                f"Expected {self.num_probes} probe inputs, got {len(probe_features)}"
            )

        logits = []
        for i, x in enumerate(probe_features):
            if x.dim() != 2:
                raise ValueError(
                    f"Probe input at index {i} must have shape [B, D], got {tuple(x.shape)}"
                )
            if x.size(1) != self.input_dim:
                raise ValueError(
                    f"Probe input at index {i} has feature dim {x.size(1)}, expected {self.input_dim}"
                )
            x = x.to(self.device, non_blocking=True)
            logits.append(self.probes[i](x))
        return logits

    def training_step(
        self,
        probe_features: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, object]:
        targets = targets.to(self.device, non_blocking=True)

        self.train()
        self.optimizer.zero_grad()

        logits_per_probe = self.forward(probe_features)

        losses = []
        correct = []
        batch_size = targets.size(0)

        for logits in logits_per_probe:
            loss = self.criterion(logits, targets)
            losses.append(loss)
            preds = logits.argmax(dim=1)
            correct.append((preds == targets).sum().item())

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.detach().item(),
            "probe_losses": [loss.detach().item() for loss in losses],
            "probe_accuracies": [c / batch_size for c in correct],
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader,
        # hook_collector,
        device: Optional[torch.device] = None,
    ) -> Dict[str, object]:
        eval_device = device if device is not None else self.device

        self.eval()
        model.eval()

        total_samples = 0
        total_probe_loss = torch.zeros(self.num_probes, dtype=torch.float64)
        total_probe_correct = torch.zeros(self.num_probes, dtype=torch.float64)

        for images, targets in test_loader:
            images = images.to(eval_device, non_blocking=True)
            targets = targets.to(eval_device, non_blocking=True)

            with HookCollector(model) as acts:
                with torch.no_grad():
                    output = model(images)

            probe_features = []
            for bi in range(self.num_probes // 2):
                probe_features.append(select_probe_feature(acts[bi]["attn"], mode=self.probe_mode))
                probe_features.append(select_probe_feature(acts[bi]["blk"], mode=self.probe_mode))
            
            logits_per_probe = self.forward(probe_features)

            batch_size = targets.size(0)
            total_samples += batch_size

            for i, logits in enumerate(logits_per_probe):
                loss = self.criterion(logits, targets)
                preds = logits.argmax(dim=1)

                total_probe_loss[i] += loss.item() * batch_size
                total_probe_correct[i] += (preds == targets).sum().item()

        avg_probe_loss = (total_probe_loss / total_samples).tolist()
        avg_probe_acc = (total_probe_correct / total_samples).tolist()

        return {
            "probe_losses": avg_probe_loss,
            "probe_accuracies": avg_probe_acc,
            "mean_loss": float(sum(avg_probe_loss) / self.num_probes),
            "mean_accuracy": float(sum(avg_probe_acc) / self.num_probes),
            "num_samples": total_samples,
        }

def get_probe_label(probe_index: int) -> str:
    probe_name = f"probe_layer{probe_index//2}"
    if probe_index % 2 == 0:
        probe_name += "_attn"
    else:
        probe_name += "_mlp"
    return probe_name

def load_probe_trainer(probe_trainer, probe_directory, master_epoch):
    import glob
    all_checkpoints = glob.glob(os.path.join(probe_directory, f'probe_E{master_epoch:03d}_checkpoint-*.pth'))
    latest_ckpt = -1
    resume_path, backup_resume_path = None, None
    for ckpt in all_checkpoints:
        t = ckpt.split('-')[-1].split('.')[0]
        if t.isdigit():
            latest_ckpt = max(int(t), latest_ckpt)
    if latest_ckpt >= 0:
        resume_path = os.path.join(probe_directory, f'probe_E{master_epoch:03d}_checkpoint-{latest_ckpt:02d}.pth')
        if latest_ckpt > 0:
            backup_resume_path = os.path.join(probe_directory, f'probe_E{master_epoch:03d}_checkpoint-{(latest_ckpt-1):02d}.pth')
    print("Auto resume checkpoint: %s" % resume_path)
    print("Backup resume checkpoint: %s" % backup_resume_path)

    if resume_path is not None and os.path.exists(resume_path):
        try:
            checkpoint_content = torch.load(resume_path, weights_only=False)
            print(f"Successfully loaded probe checkpoint from {resume_path}")
        except Exception as e:
            print(f"Failed to load probe checkpoint from {resume_path}: {e}")
            if backup_resume_path is not None and os.path.exists(backup_resume_path):
                try:
                    checkpoint_content = torch.load(backup_resume_path, weights_only=False)
                    print(f"Successfully loaded backup probe checkpoint from {backup_resume_path}")
                except Exception as e2:
                    raise e2
            else:
                raise e
        if "model_state_dict" in checkpoint_content:
            probe_trainer.load_state_dict(checkpoint_content["model_state_dict"])
        if "optimizer" in checkpoint_content:
            probe_trainer.optimizer.load_state_dict(checkpoint_content["optimizer"])
        if "epoch" in checkpoint_content:
            probe_trainer.epoch = checkpoint_content["epoch"] + 1
        assert master_epoch == checkpoint_content.get("master_epoch", master_epoch), f"Master epoch mismatch: expected {master_epoch}, got {checkpoint_content.get('master_epoch')}"
    else:
        print("No valid checkpoint found to resume.")

def save_probe_trainer(probe_trainer, probe_directory, epoch, master_epoch):
    os.makedirs(probe_directory, exist_ok=True)
    filename = f'probe_E{master_epoch:03d}_checkpoint-{epoch:02d}.pth'
    checkpoint_path = os.path.join(probe_directory, filename)

    to_save = {
        "model_state_dict": probe_trainer.state_dict(),
        "optimizer": probe_trainer.optimizer.state_dict(),
        "epoch": epoch,
        "master_epoch": master_epoch,
        "args": probe_trainer.args if hasattr(probe_trainer, 'args') else None,
    }
    save_on_master(to_save, checkpoint_path)
    print(f"Saved probe checkpoint to {checkpoint_path}")

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - 1
        # if to_del in [49, 99, 149, 199, 249, 299] and not args.save_for_analysis: # keep every 50th checkpoint
        #     pass
        # else:
        old_ckpt = Path(probe_directory) / (f'probe_E{master_epoch:03d}_checkpoint-{to_del:02d}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

def select_probe_feature(x: torch.Tensor, mode: str = "cls") -> torch.Tensor:
    """
    Convert a ViT hook output into a [B, D] probe feature.

    Expected inputs:
    - [B, D]       : already pooled / vector feature
    - [B, T, D]    : token sequence, where token 0 is CLS for CLS-based ViTs

    Modes:
    - "cls"
    - "mean_patch"
    - "max_patch"
    - "mean_all"
    - "cls_mean_patch"
    - "concat_cls_mean_patch"
    - "flatten_patches"
    - "identity"
    """

    if x.dim() == 2:
        if mode in {"identity", "cls", "mean_patch", "max_patch", "mean_all", "cls_mean_patch"}:
            return x
        elif mode == "concat_cls_mean_patch":
            return torch.cat([x, x], dim=-1)
        elif mode == "flatten_patches":
            return x
        else:
            raise ValueError(f"Unsupported mode '{mode}' for 2D input {tuple(x.shape)}")

    if x.dim() != 3:
        raise ValueError(f"Expected input of shape [B, D] or [B, T, D], got {tuple(x.shape)}")

    cls_token = x[:, 0, :]
    patch_tokens = x[:, 1:, :]

    if mode == "cls":
        return cls_token

    if mode == "sample_wise_cls":
        return cls_token

    if mode == "mean_patch":
        if patch_tokens.size(1) == 0:
            raise ValueError("No patch tokens available for mean_patch")
        return patch_tokens.mean(dim=1)

    if mode == "max_patch":
        if patch_tokens.size(1) == 0:
            raise ValueError("No patch tokens available for max_patch")
        return patch_tokens.max(dim=1).values

    if mode == "mean_all":
        return x.mean(dim=1)

    if mode == "cls_mean_patch":
        if patch_tokens.size(1) == 0:
            return cls_token
        return 0.5 * (cls_token + patch_tokens.mean(dim=1))

    if mode == "concat_cls_mean_patch":
        if patch_tokens.size(1) == 0:
            return torch.cat([cls_token, cls_token], dim=-1)
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

    if mode == "flatten_patches":
        if patch_tokens.size(1) == 0:
            raise ValueError("No patch tokens available for flatten_patches")
        return patch_tokens.flatten(start_dim=1)

    if mode == "identity":
        return x

    raise ValueError(f"Unknown feature selection mode: {mode}")

import os
import json


def append_epoch_metrics_to_json(json_path, epoch_metrics):
    """
    json_path: path to metrics json
    epoch_metrics: dict for current epoch, e.g.
        {
            "epoch": 3,
            "probe/train/loss": 0.52,
            "probe/train/mean_acc": 0.81,
            "probe/val/mean_acc": 0.79
        }

    Stored format:
        {
            "epoch": [0, 1, 2, 3],
            "probe/train/loss": [...],
            "probe/train/mean_acc": [...],
            ...
        }
    """

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(epoch_metrics)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data

def get_master_epoch(path):
    if "pr" in path or path=='':
        return 0
    me = path.split('/')[-1].split('.')[0].split('-')[-2]
    print(f"Extracted master epoch string: '{me}' from path: '{path}'")
    me = int(me) if me.isdigit() else None
    return me