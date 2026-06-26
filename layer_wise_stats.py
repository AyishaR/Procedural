# probe_stats.py

import os, glob
import torch
import torch.distributed as dist
from typing import List, Dict
from pathlib import Path

def _to_class_indices(targets: torch.Tensor) -> torch.Tensor:
    if targets.dim() == 2:
        return targets.argmax(dim=1)
    if targets.dim() == 1:
        return targets
    raise ValueError(f"Unsupported target shape: {tuple(targets.shape)}")


class StreamingProbeStats:
    def __init__(self, feature_dim: int, num_classes: int, device: str = "cpu", probe_mode: str = "cls"):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = torch.device(device)

        self.n_total = torch.tensor(0, dtype=torch.long, device=self.device)
        self.sum_x = torch.zeros(feature_dim, dtype=torch.float64, device=self.device)
        self.sum_xxT = torch.zeros(feature_dim, feature_dim, dtype=torch.float64, device=self.device)

        self.class_counts = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        self.class_sums = torch.zeros(num_classes, feature_dim, dtype=torch.float64, device=self.device)
        self.class_sum_xxT = torch.zeros(num_classes, feature_dim, feature_dim, dtype=torch.float64, device=self.device)

        self.probe_mode = probe_mode

    @torch.no_grad()
    def update(self, features: torch.Tensor, targets: torch.Tensor):
        x = features.detach().to(self.device, dtype=torch.float64)
        if self.probe_mode == "sample_wise_cls": x = x.T
        # print(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        y = _to_class_indices(targets).detach().to(self.device, dtype=torch.long)

        if x.dim() != 2:
            raise ValueError(f"features must be [B, D], got {tuple(x.shape)}")
        if x.size(1) != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {x.size(1)}")

        bsz = x.size(0)
        self.n_total += bsz
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += x.T @ x

        unique_classes = torch.unique(y)
        for cls in unique_classes:
            cls_int = int(cls.item())
            mask = (y == cls)
            x_c = x[mask]

            self.class_counts[cls_int] += x_c.size(0)
            self.class_sums[cls_int] += x_c.sum(dim=0)
            self.class_sum_xxT[cls_int] += x_c.T @ x_c

    @torch.no_grad()
    def update_sample_wise(self, features: torch.Tensor, targets: torch.Tensor):
        x = features.detach().to(self.device, dtype=torch.float64)
        if self.probe_mode == "sample_wise_cls":
            x = x.T
        x = x.contiguous()

        y = _to_class_indices(targets).detach().to(self.device, dtype=torch.long)
        y = y.contiguous()

        print(f"x shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
        print(f"y shape: {y.shape}, dtype: {y.dtype}, device: {y.device}")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        gather_list_x = [torch.empty_like(x) for _ in range(world_size)]
        gather_list_y = [torch.empty_like(y) for _ in range(world_size)]

        dist.all_gather(gather_list_x, x)
        dist.all_gather(gather_list_y, y)
        print(f"Rank {rank} - Completed all_gather. Gathered x shapes: {[g.shape for g in gather_list_x]}, Gathered y shapes: {[g.shape for g in gather_list_y]}")

        if rank != 0:
            print(f"Rank {rank} - Not the main process, skipping update after all_gather")
            return

        x = torch.cat(gather_list_x, dim=1).contiguous()
        y = torch.cat(gather_list_y, dim=0).contiguous()
        print(f"After all_gather - x shape: {x.shape}, y shape: {y.shape}")
        print(y)
        torch.save({"x": x.cpu(), "y": y.cpu()}, f"debug_rank{rank}_gathered.pt")

        bsz = x.size(0)
        self.n_total += bsz
        self.sum_x += x.sum(dim=0)
        self.sum_xxT += x.T.contiguous() @ x
        print(f"After updating global sums - n_total: {self.n_total.item()}, sum_x norm: {self.sum_x.norm().item()}, sum_xxT norm: {self.sum_xxT.norm().item()}")

        # unique_classes = torch.unique(y)
        # print(f"Unique classes in this batch: {unique_classes.cpu().numpy()}")
        # for cls_ in unique_classes:
        #     print(f"Processing class {cls_.item()} with {int((y == cls_).sum().item())} samples")
        #     cls_int = int(cls_.item())
        #     mask = (y == cls_)
        #     print("Before contiguous")
        #     x_c = x.T[mask]
        #     print(f"Class {cls_int.item()} - x_c shape: {x_c.shape}, sum: {x_c.sum(dim=0)}, xxT norm: {(x_c.T.contiguous() @ x_c).norm().item()}")

        #     self.class_counts[cls_int] += x_c.size(0)
        #     self.class_sums[cls_int] += x_c.sum(dim=0)
        #     self.class_sum_xxT[cls_int] += x_c.T @ x_c
        #     print(f"After updating class {cls_.item()} stats - count: {self.class_counts[cls_int].item()}, sum norm: {self.class_sums[cls_int].norm().item()}, sum_xxT norm: {self.class_sum_xxT[cls_int].norm().item()}")


    @torch.no_grad()
    def sync_ddp(self):
        if not dist.is_available() or not dist.is_initialized():
            return

        dist.all_reduce(self.n_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sum_xxT, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.class_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.class_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.class_sum_xxT, op=dist.ReduceOp.SUM)

    def to_device(self, device: str):
        self.device = torch.device(device)
        self.n_total = self.n_total.to(self.device)
        self.sum_x = self.sum_x.to(self.device)
        self.sum_xxT = self.sum_xxT.to(self.device)
        self.class_counts = self.class_counts.to(self.device)
        self.class_sums = self.class_sums.to(self.device)
        self.class_sum_xxT = self.class_sum_xxT.to(self.device)

    @torch.no_grad()
    def finalize(self, eps: float = 1e-3, unbiased_cov: bool = False) -> Dict[str, float]:
        n = int(self.n_total.item())
        if n < 2:
            raise ValueError("Need at least 2 samples to finalize statistics")

        mu = self.sum_x / n
        centered_second_moment = self.sum_xxT - n * torch.outer(mu, mu)
        cov = centered_second_moment / ((n - 1) if unbiased_cov else n)
        # print(f"Covariance matrix for feature dim {self.feature_dim}:\n{cov}")

        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = torch.clamp(eigvals, min=0.0)
        eigvals = torch.flip(eigvals, dims=[0])
        # print(f"Eigenvalues for feature dim {self.feature_dim}:\n{eigvals}")

        max_eig = eigvals[0]
        effective_rank = 0 if max_eig <= 0 else int((eigvals > eps * max_eig).sum().item())

        sb_trace = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        sw_trace = torch.tensor(0.0, dtype=torch.float64, device=self.device)

        if self.probe_mode != "sample_wise_cls":    
            valid_classes = torch.nonzero(self.class_counts > 0, as_tuple=False).flatten()
    
            for cls in valid_classes:
                n_c = int(self.class_counts[cls].item())
                mu_c = self.class_sums[cls] / n_c

                diff_between = mu_c - mu
                sb_trace += n_c * torch.dot(diff_between, diff_between)

                sw_trace += torch.trace(self.class_sum_xxT[cls]) - n_c * torch.dot(mu_c, mu_c)

        return {
            "effective_rank": float(effective_rank),
            "inter_class_variance": float((sb_trace / n).item()),
            "intra_class_variance": float((sw_trace / n).item()),
            "num_samples": n,
        }

    def state_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "n_total": self.n_total.cpu(),
            "sum_x": self.sum_x.cpu(),
            "sum_xxT": self.sum_xxT.cpu(),
            "class_counts": self.class_counts.cpu(),
            "class_sums": self.class_sums.cpu(),
            "class_sum_xxT": self.class_sum_xxT.cpu(),
        }

    def load_state_dict(self, state):
        self.n_total = state["n_total"].to(self.device)
        self.sum_x = state["sum_x"].to(self.device)
        self.sum_xxT = state["sum_xxT"].to(self.device)
        self.class_counts = state["class_counts"].to(self.device)
        self.class_sums = state["class_sums"].to(self.device)
        self.class_sum_xxT = state["class_sum_xxT"].to(self.device)


class StreamingAllProbeStats:
    def __init__(self, feature_dims: List[int], num_classes: int, device: str = "cpu", probe_mode: str = "cls", probe_names: List[int] = None):
        self.feature_dims = feature_dims
        self.num_probes = len(feature_dims)
        self.num_classes = num_classes
        self.device = device
        self.stats = [
            StreamingProbeStats(d, num_classes=num_classes, device=device, probe_mode=probe_mode)
            for d in feature_dims
        ]
        self.probe_mode = probe_mode
        self.epoch = 0
        self.batch_idx = 0
        self.probe_names = probe_names

    @torch.no_grad()
    def update(self, probe_features: List[torch.Tensor], targets: torch.Tensor):
        if len(probe_features) != self.num_probes:
            raise ValueError(f"Expected {self.num_probes} probe features, got {len(probe_features)}")
        for stat, feats in zip(self.stats, probe_features):
            if self.probe_mode == "sample_wise_cls":
                stat.update_sample_wise(feats, targets)
            else:
                stat.update(feats, targets)

    @torch.no_grad()
    def sync_ddp(self):
        for stat in self.stats:
            stat.sync_ddp()

    @torch.no_grad()
    def finalize(self, eps: float = 1e-3, unbiased_cov: bool = False) -> Dict[str, Dict[str, float]]:
        return {
            f"probe_{self.probe_names[i]}": stat.finalize(eps=eps, unbiased_cov=unbiased_cov)
            for i, stat in enumerate(self.stats)
        }

    def to_device(self, device: str):
        self.device = device
        for stat in self.stats:
            stat.to_device(device)

    def state_dict(self):
        return {
            "feature_dims": self.feature_dims,
            "num_classes": self.num_classes,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "stats": [s.state_dict() for s in self.stats],
        }

    def load_state_dict(self, state):
        self.epoch = state.get("epoch", 0)
        self.batch_idx = state.get("batch_idx", 0)
        for s, sd in zip(self.stats, state["stats"]):
            s.load_state_dict(sd)


def save_streaming_stats_checkpoint(stats_obj, output_dir, master_epoch, epoch, batch_idx, rank=0):
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"probe_stats_resume_E{master_epoch:03d}_e{epoch:03d}_b{batch_idx:03d}.pth")
    stats_obj.epoch = epoch
    stats_obj.batch_idx = batch_idx
    torch.save(stats_obj.state_dict(), ckpt_path)
    # to_del = epoch - 2
        # if to_del in [49, 99, 149, 199, 249, 299] and not args.save_for_analysis: # keep every 50th checkpoint
        #     pass
        # else:
    all_checkpoints = glob.glob(os.path.join(output_dir, f"probe_stats_resume_E{master_epoch:03d}_e{epoch:03d}_b*.pth"))
    all_checkpoints = sorted(all_checkpoints, key=os.path.getctime)
    print(f"Found {len(all_checkpoints)} probe stats checkpoints for master epoch {master_epoch} after saving new checkpoint")
    print(f"All checkpoints: {all_checkpoints}")
    for ckpt in all_checkpoints[:-1]:
        old_ckpt = Path(output_dir) / ckpt
        if os.path.exists(old_ckpt):
            print(f"Removing old checkpoint {old_ckpt}")
            os.remove(old_ckpt)


def load_streaming_stats_checkpoint(stats_obj, output_dir, master_epoch):
    all_checkpoints = glob.glob(os.path.join(output_dir, f"probe_stats_resume_E{master_epoch:03d}_*.pth"))
    print(f"Load: Found {len(all_checkpoints)} probe stats checkpoints in {output_dir}")
    print(f"Load: All checkpoints: {all_checkpoints}")
    latest_ckpt = max(all_checkpoints, key=os.path.getctime) if all_checkpoints else None
    print(f"Load: Loading probe stats checkpoint from {latest_ckpt}")
    ckpt_path = os.path.join(output_dir, latest_ckpt) if latest_ckpt else None
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return 0, 0
    print(f"Loading probe stats checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats_obj.load_state_dict(state)
    return stats_obj.epoch, stats_obj.batch_idx
