from kdyck.kdyck_dataset import KDyckDataset
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def center_gram(G):
    n = G.size(0)
    eye = torch.eye(n, device=G.device, dtype=G.dtype)
    ones = torch.ones((n, n), device=G.device, dtype=G.dtype) / n
    H = eye - ones
    return H @ G @ H

def linear_cka(X, Y, eps=1e-12):
    """
    X: [n_samples, d1]
    Y: [n_samples, d2]
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XT_Y = X.T @ Y
    hsic = (XT_Y ** 2).sum()

    norm_x = torch.norm(X.T @ X, p="fro")
    norm_y = torch.norm(Y.T @ Y, p="fro")

    return hsic / (norm_x * norm_y + eps)

def gram_cka(X, Y, eps=1e-12):
    """
    Kernel CKA using Gram matrices.
    X: [n_samples, d1]
    Y: [n_samples, d2]
    """
    K = X @ X.T
    L = Y @ Y.T
    Kc = center_gram(K)
    Lc = center_gram(L)

    hsic = (Kc * Lc).sum()
    norm = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return hsic / (norm + eps)

if __name__ == "__main__":
    k_data = KDyckDataset(k=64, num_samples=1000, max_length=196)
    # get 20 samples
    N = 20
    kset1 = k_data[:N]
    kset2 = k_data[N:2*N]

    embeddings = torch.load("kdyck/kdyck_orthogonal_embeddings_vits.pt")
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)

    # get the embeddings for the two sets
    kset1_embeddings = embedding_layer(kset1).reshape(N, -1)
    kset2_embeddings = embedding_layer(kset2).reshape(N, -1)

    cka_linear = linear_cka(kset1_embeddings, kset2_embeddings)
    print(f"Linear CKA: {cka_linear.item():.4f}")
    cka_gram = gram_cka(kset1_embeddings, kset2_embeddings)
    print(f"Gram CKA: {cka_gram.item():.4f}")
