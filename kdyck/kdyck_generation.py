import numpy as np
import torch, random

def generate_k_dyck(k=64, length=196, p_open=0.6, max_depth=4):
    """
    Generate random k-Dyck word (perfectly balanced brackets).
    k: number of bracket types (64)
    length: total symbols (196 for 14x14 flattened)
    """
    if length is None:
        length = 14 * 14  # 196
    
    # Ensure even length
    if length % 2 != 0:
        length += 1
    
    # Stack tracks bracket_type
    stack = []
    dyck = np.zeros(length, dtype=np.int16)  # 0-127 vocab

    for i in range(length):
        if len(stack) == length-i:
            while stack:
                top_type = stack.pop()
                dyck[i] = k + top_type
                i+=1
            break
        if len(stack) == 0 or (np.random.rand() < p_open and len(stack)<max_depth): # p(open) given in appendix
            bracket_type = np.random.randint(0, k)
            dyck[i] = bracket_type  # Open symbol
            stack.append(bracket_type)
        else:
            top_type = stack[-1]
            dyck[i] = k + top_type  # Close symbol (64-127)
            stack.pop()
        # print("Dyck so far:", dyck[:i+1])

    return torch.tensor(dyck.astype(np.int64))

def mask_kdyck(kdyck_seq, mask_token=128, mask_prob=0.5, close_brack_start_token=64):
    """
    Mask k-Dyck sequence tokens with given probability.
    """
    masked_seq = kdyck_seq.clone()
    for i in range(len(kdyck_seq)):
        if masked_seq[i]>=close_brack_start_token and np.random.rand() < mask_prob:
            masked_seq[i] = mask_token
    return masked_seq

def generate_dataset(length=1000, save_path='', k=64, seq_length=196, mask_token=128, mask_prob=0.5, p_open=0.6, max_depth=4):
    """
    Generate dataset of k-Dyck sequences and masked sequences.
    length: number of sequences
    save_path: path to save .npz file
    k: number of bracket types
    seq_length: length of each sequence
    mask_token: token used for masking
    mask_prob: probability of masking a closing bracket
    """
    kdyck_seqs = []
    masked_seqs = []
    for _ in range(length):
        dyck_seq = generate_k_dyck(
            k=k, 
            length=seq_length, 
            p_open=p_open,
            max_depth=max_depth
        )
        masked_seq = mask_kdyck(
            dyck_seq, 
            mask_token=mask_token, 
            mask_prob=mask_prob, 
            close_brack_start_token=k
        )
        kdyck_seqs.append(dyck_seq.numpy())
        masked_seqs.append(masked_seq.numpy())

    kdyck_seqs = torch.tensor(np.array(kdyck_seqs), dtype=torch.int64)
    masked_seqs = torch.tensor(np.array(masked_seqs), dtype=torch.int64)
    
    if save_path:
        np.savez_compressed(save_path, kdyck_seqs=np.array(kdyck_seqs), masked_seqs=np.array(masked_seqs))
        print(f"Dataset saved to {save_path}")

    return kdyck_seqs, masked_seqs

def generate_dataset_truncated(length=1000, k=64, min_depth=1, max_depth=4, seq_length=510, offset=None, mask_token=128, mask_prob=0.5, save_path="", p_open=0.6):
    """Generates a Dyck sequence with specified number of symbols and depth constraints.

    Args:
        k: The number of distinct symbol pairs (k in k-Dyck).
        min_depth: Minimum required depth of nested brackets.
        max_depth: Maximum allowed depth of nested brackets.
        seq_length: The maximum length of the generated sequence.

    Returns:
        A list representing the Dyck sequence, or None if generation fails.
    """
    kdyck_seqs = []
    masked_seqs = []
    while len(kdyck_seqs) < length:
        result = []
        stack = []

        if min_depth < 1:
            raise ValueError("min_depth must be at least 1.")

        if offset is None:
            offset = k

        # Initialize with minimum depth
        for _ in range(min_depth):
            opening_symbol = np.random.randint(0, k)
            result.append(opening_symbol)
            stack.append(opening_symbol)

        while len(result) < seq_length:
            if (
                len(stack) < max_depth and random.random() < p_open
            ) or len(stack)==0:  # Try to open if under max depth
                if len(result) >= seq_length - 1:
                    closing_symbol = stack.pop() + offset
                    result.append(closing_symbol)
                    continue
                opening_symbol = np.random.randint(0, k)
                result.append(opening_symbol)
                stack.append(opening_symbol)
            else:  # Close existing bracket
                closing_symbol = stack.pop() + offset
                result.append(closing_symbol)
                # if not stack:
                #     break

        result = result[:seq_length]  # Truncate to desired length
        kdyck_seqs.append(np.array(result))
        masked_seq = mask_kdyck(torch.tensor(kdyck_seqs[-1]), mask_token=mask_token, mask_prob=mask_prob, close_brack_start_token=k)
        masked_seqs.append(masked_seq.numpy())
    
    kdyck_seqs = torch.tensor(np.array(kdyck_seqs), dtype=torch.int64)
    masked_seqs = torch.tensor(np.array(masked_seqs), dtype=torch.int64)
    if save_path:
        np.savez_compressed(save_path, kdyck_seqs=np.array(kdyck_seqs), masked_seqs=np.array(masked_seqs))
        print(f"Dataset saved to {save_path}")

    return kdyck_seqs, masked_seqs
    

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