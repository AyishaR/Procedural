import math
import torch

def spiral_unravel(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]
    rows, cols = n, n
    rows, cols = n, n
    r, c = n//2, n//2-1
    directions = torch.tensor([[0,1], [-1,0], [0,-1], [1,0] ])  # right, up, left, down
    dir_idx = 0
    visited = torch.zeros((n, n), dtype=torch.bool)
    order = []
    steps = 1
    leg_count = 0

    while len(order) < n*n:
        for _ in range(steps):
            if 0 <= r < rows and 0 <= c < cols and not visited[r, c]:
                idx = r * cols + c
                order.append(idx)
                visited[r, c] = True
            # Move always
            dr, dc = directions[dir_idx]
            r += dr.item()
            c += dc.item()
        dir_idx = (dir_idx + 1) % 4
        leg_count += 1
        if leg_count % 2 == 0:
            steps += 1
            leg_count = 0

    order = torch.tensor(order).expand(b, -1).cuda() if t.is_cuda else torch.tensor(order).expand(b, -1)

    return torch.gather(t, 1, order)