import math
import torch

def spiral_unravel(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]
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

def vertical_unravel(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]

    indices = torch.arange(n*n).reshape(n, n)
    rotated_indices = indices.t()
    order = rotated_indices.flatten()
    print(t.shape, order.shape)
    
    order = order.expand(b, -1).cuda() if t.is_cuda else order.expand(b, -1)

    return torch.gather(t, 1, order)

def row_alternate(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]

    indices = torch.arange(n*n).reshape(n, n)
    
    even_rows = indices[::2].flatten()  # Even rows: 0, 2, ...
    odd_rows = indices[1::2].flatten()  # Odd rows: 1, 3, ...
    new_indices = torch.cat((even_rows, odd_rows), dim=0)
    
    order = new_indices.flatten()
    
    order = order.expand(b, -1).cuda() if t.is_cuda else order.expand(b, -1)

    return torch.gather(t, 1, order)

def row_alternate_half(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]

    indices = torch.arange(n*n).reshape(n*2, n//2)
    
    even_rows = indices[::2].flatten()  # Even rows: 0, 2, ...
    odd_rows = indices[1::2].flatten()  # Odd rows: 1, 3, ...
    new_indices = torch.cat((even_rows, odd_rows), dim=0)
    
    order = new_indices.flatten()
    
    order = order.expand(b, -1).cuda() if t.is_cuda else order.expand(b, -1)

    return torch.gather(t, 1, order)

def waterfall(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]

    indices = torch.arange(n*n).reshape(n, n)
    
    indices[1::2] = indices[1::2].flip(1)
    order = indices.flatten()
    
    order = order.expand(b, -1).cuda() if t.is_cuda else order.expand(b, -1)

    return torch.gather(t, 1, order)

def waterfall_half(t):
    n = math.isqrt(t.shape[1])
    b = t.shape[0]

    indices = torch.arange(n*n).reshape(n*2, n//2)
    
    indices[1::2] = indices[1::2].flip(1)
    order = indices.flatten()
    
    order = order.expand(b, -1).cuda() if t.is_cuda else order.expand(b, -1)

    return torch.gather(t, 1, order)