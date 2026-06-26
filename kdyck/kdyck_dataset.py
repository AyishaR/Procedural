import numpy as np
import torch, random

class KDyckDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.k = self.args.k
        self.min_depth = 1
        self.seq_length = self.args.seq_length
        self.p_open = self.args.p_open
        self.max_depth = self.args.max_depth

    def __len__(self):
        return 4000000
    
    def generate_kdyck(self):
        """
        Generate random k-Dyck word (perfectly balanced brackets).
        k: number of bracket types (64)
        length: total symbols (196 for 14x14 flattened)
        """
        
        stack = []
        dyck = np.zeros(self.seq_length, dtype=np.int16)  # 0-127 vocab

        for i in range(self.seq_length):
            if len(stack) == self.seq_length-i:
                while stack:
                    top_type = stack.pop()
                    dyck[i] = self.k + top_type
                    i+=1
                break
            if len(stack) < self.min_depth or (np.random.rand() < self.p_open and len(stack)<self.max_depth): # p(open) given in appendix
                bracket_type = np.random.randint(0, self.k)
                dyck[i] = bracket_type  # Open symbol
                stack.append(bracket_type)
            else:
                top_type = stack[-1]
                dyck[i] = self.k + top_type  # Close symbol (64-127)
                stack.pop()

        return torch.tensor(dyck.astype(np.int64))

    def generate_kdyck_shuffled(self):
        """
        Generate random k-Dyck word (perfectly balanced brackets).
        k: number of bracket types (64)
        length: total symbols (196 for 14x14 flattened)
        """
        
        stack = []
        dyck = np.zeros(self.seq_length, dtype=np.int16)  # 0-127 vocab

        for i in range(self.seq_length):
            if len(stack) == self.seq_length-i:
                while stack:
                    top_type = stack.pop()
                    dyck[i] = self.k + top_type
                    i+=1
                break
            if len(stack) < self.min_depth or (np.random.rand() < self.p_open and len(stack)<self.max_depth): # p(open) given in appendix
                bracket_type = np.random.randint(0, self.k)
                dyck[i] = bracket_type  # Open symbol
                stack.append(bracket_type)
            else:
                top_type = stack.pop(random.randint(0, len(stack)-1))  # Randomly pop from stack
                dyck[i] = self.k + top_type  # Close symbol (64-127)

        return torch.tensor(dyck.astype(np.int64))

    def generate_kdyck_truncated(self):
        result = []
        stack = []

        # Initialize with minimum depth
        for _ in range(self.min_depth):
            opening_symbol = np.random.randint(0, self.k)
            result.append(opening_symbol)
            stack.append(opening_symbol)

        while len(result) < self.seq_length:
            if (len(stack) < self.max_depth and random.random() < self.p_open) or len(stack)<self.min_depth: 
                # if len(result) >= self.max_depth - 1:
                #     closing_symbol = stack.pop() + offset
                #     result.append(closing_symbol)
                #     continue
                opening_symbol = np.random.randint(0, self.k)
                result.append(opening_symbol)
                stack.append(opening_symbol)
            else: 
                closing_symbol = stack.pop() + self.k
                result.append(closing_symbol)

        # result = result[:self.max_depth]
        return torch.tensor(np.array(result), dtype=torch.int64)

    def generate_kdyck_truncated_shuffled(self):
        result = []
        stack = []

        # Initialize with minimum depth
        for _ in range(self.min_depth):
            opening_symbol = np.random.randint(0, self.k)
            result.append(opening_symbol)
            stack.append(opening_symbol)

        while len(result) < self.seq_length:
            if (len(stack) < self.max_depth and random.random() < self.p_open) or len(stack)<self.min_depth: 
                # if len(result) >= self.max_depth - 1:
                #     closing_symbol = stack.pop() + offset
                #     result.append(closing_symbol)
                #     continue
                opening_symbol = np.random.randint(0, self.k)
                result.append(opening_symbol)
                stack.append(opening_symbol)
            else: 
                closing_symbol = stack.pop(random.randint(0, len(stack)-1)) + self.k
                result.append(closing_symbol)

        # result = result[:self.max_depth]
        return torch.tensor(np.array(result), dtype=torch.int64)

    def __getitem__(self, _idx):
        if self.args.procedural_data == "kdyck_truncated":
            kdyck_seq = self.generate_kdyck_truncated()
        elif self.args.procedural_data == "kdyck_truncated_shuffled":
            kdyck_seq = self.generate_kdyck_truncated_shuffled()
        elif self.args.procedural_data == "kdyck_shuffled": 
            kdyck_seq = self.generate_kdyck_shuffled()
        elif self.args.procedural_data == "kdyck":
            kdyck_seq = self.generate_kdyck()
        return kdyck_seq

def mask_kdyck_dataset(kdyck_seqs, mask_token=128, mask_prob=0.5, close_brack_start_token=64):
    prob_mask = torch.rand(kdyck_seqs.shape) < mask_prob 
    is_close_bracket = kdyck_seqs >= close_brack_start_token
    mask = prob_mask & is_close_bracket
    masked_seqs = kdyck_seqs.clone()
    masked_seqs[mask] = mask_token
    return masked_seqs

def load_dataset(file_path):
    """
    Load k-Dyck dataset from .npz file.
    Returns tuple of (kdyck_seqs, masked_seqs)
    """
    data = np.load(file_path)
    kdyck_seqs = data['kdyck_seqs']
    masked_seqs = data['masked_seqs']
    return kdyck_seqs, masked_seqs  

if __name__ == "__main__":
    # Generate
    dyck_seq = generate_k_dyck(k=64, length=8)
    print(dyck_seq)
    masked_seq = mask_kdyck(dyck_seq, mask_token=128, mask_prob=0.5)
    print(masked_seq)

    # Verify perfect balance per bracket type
    open_counts = np.bincount(dyck_seq[dyck_seq < 64], minlength=64)
    close_counts = np.bincount(dyck_seq[dyck_seq >= 64] - 64, minlength=64)

    assert np.all(open_counts == close_counts)

    generate_dataset(length=10000, save_path='kdyck_dataset.npz', k=64, seq_length=196, mask_token=128, mask_prob=0.5)
    # s, m = load_dataset('kdyck_dataset.npz')
    # print("Loaded dataset shapes:", s.shape, m.shape)
    # print("Sample k-Dyck sequence:", s[0])
    # print("Sample masked sequence:", m[0])
    # print("Number of masked tokens in sample:", np.sum(m[0]==128))