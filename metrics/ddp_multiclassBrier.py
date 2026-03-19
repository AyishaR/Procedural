import torch
import torch.distributed as dist
import torch.nn.functional as F

class DDPMulticlassBrier:
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.sse = torch.tensor(0.0, device=self.device)  # sum of squared errors
        self.count = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def update(self, logits, labels):
        """
        logits: (N_local, C)
        labels: (N_local,)
        """
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        probs = F.softmax(logits, dim=1)  # (N_local, C)
        N, C = probs.shape

        one_hot = torch.zeros_like(probs)
        one_hot[torch.arange(N, device=self.device), labels] = 1.0

        sq_err = (probs - one_hot) ** 2         # (N_local, C)
        self.sse += sq_err.sum()               # scalar
        self.count += float(N)

    @torch.no_grad()
    def compute(self):
        """
        Synchronize across ranks and return global multiclass Brier score.
        """
        if not dist.is_available() or not dist.is_initialized():
            sse = self.sse
            count = self.count
        else:
            sse = self.sse.clone()
            count = self.count.clone()
            dist.all_reduce(sse, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        if count.item() == 0:
            return 0.0

        return float(sse / count)

    @torch.no_grad()
    def reset(self):
        self.sse.zero_()
        self.count.zero_()
