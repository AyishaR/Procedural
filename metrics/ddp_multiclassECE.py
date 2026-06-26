import torch
import torch.distributed as dist
import torch.nn.functional as F

class DDPMulticlassECE:
    def __init__(self, n_bins=15, device=None):
        self.n_bins = n_bins
        self.device = device if device is not None else torch.device('cpu')

        # Per-bin accumulators (local, then reduced across ranks in compute())
        self.bin_count = torch.zeros(n_bins, device=self.device)
        self.bin_conf_sum = torch.zeros(n_bins, device=self.device)
        self.bin_acc_sum = torch.zeros(n_bins, device=self.device)

    @torch.no_grad()
    def update(self, probs, labels):
        """
        probs: (N_local, C)
        labels: (N_local,)
        """
        probs = probs.to(self.device)
        labels = labels.to(self.device)

        # probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)           # (N_local,)
        acc = (pred == labels).float()          # (N_local,)

        # Bin indices for each example based on confidence
        # confidences in [0,1] -> bins [0, n_bins-1]
        bin_ids = torch.clamp(
            (conf * self.n_bins).long(),
            min=0,
            max=self.n_bins - 1,
        )

        # Accumulate per-bin counts, confidence sums, accuracy sums
        for b in range(self.n_bins):
            mask = (bin_ids == b)
            if mask.any():
                self.bin_count[b] += mask.sum()
                self.bin_conf_sum[b] += conf[mask].sum()
                self.bin_acc_sum[b] += acc[mask].sum()

    @torch.no_grad()
    def compute(self):
        """
        Synchronize across ranks and return global ECE (scalar float).
        """
        if not dist.is_available() or not dist.is_initialized():
            # Single-process: just compute locally
            bin_count = self.bin_count
            bin_conf_sum = self.bin_conf_sum
            bin_acc_sum = self.bin_acc_sum
        else:
            # Clone and all_reduce to get global sums
            bin_count = self.bin_count.clone()
            bin_conf_sum = self.bin_conf_sum.clone()
            bin_acc_sum = self.bin_acc_sum.clone()

            dist.all_reduce(bin_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_conf_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_acc_sum, op=dist.ReduceOp.SUM)

        N = bin_count.sum()
        if N == 0:
            return 0.0

        # Compute global ECE from global bin stats
        ece = 0.0
        for b in range(self.n_bins):
            count_b = bin_count[b].item()
            if count_b > 0:
                avg_conf = (bin_conf_sum[b] / count_b).item()
                avg_acc = (bin_acc_sum[b] / count_b).item()
                ece += (count_b / N.item()) * abs(avg_conf - avg_acc)

        return float(ece)

    @torch.no_grad()
    def reset(self):
        self.bin_count.zero_()
        self.bin_conf_sum.zero_()
        self.bin_acc_sum.zero_()
