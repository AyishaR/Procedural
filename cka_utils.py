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

def center_gram_unbiased(G):
    """
    Unbiased (debiased) Gram-matrix centering via the U-statistic formulation of
    Szekely & Rizzo (2014, "Partial distance correlation with methods for
    dissimilarity selection", section 2.5.1) -- the same formulation Kornblith et
    al. 2019's reference CKA implementation uses to compute the Song et al. 2007
    unbiased HSIC_1 estimator. Unlike directly evaluating HSIC_1's U-statistic
    formula (tr(K~L~) + ... - ...), which subtracts terms of similar magnitude and
    can go slightly negative under floating-point cancellation for near-degenerate
    Gram matrices (e.g. the near-uniform attention maps typical of a ViT's first
    and last blocks), this centering makes the self-similarity term
    sum(center_gram_unbiased(K) ** 2) a sum of squares -- non-negative by
    construction, so gram_cka_unbiased's sqrt() below can never see a negative
    input. G: [n, n] Gram matrix (not pre-centered).
    """
    n = G.size(0)
    G = G.clone()
    G.fill_diagonal_(0)
    means = G.sum(dim=0) / (n - 2)
    means = means - means.sum() / (2 * (n - 1))
    G = G - means[:, None] - means[None, :]
    G.fill_diagonal_(0)
    return G

def gram_cka_unbiased(X, Y, eps=1e-12):
    """
    Unbiased (debiased) kernel CKA using Gram matrices, via center_gram_unbiased
    above -- numerically-stable equivalent of plugging Song et al. 2007's unbiased
    HSIC_1 estimator into gram_cka's biased sum(K_c * L_c) formula. Computed in
    float64 (like Kornblith et al.'s reference implementation) to further limit
    cancellation error, then cast back to X's dtype.
    X: [n_samples, d1]
    Y: [n_samples, d2]
    """
    orig_dtype = X.dtype
    K = (X.double() @ X.double().T)
    L = (Y.double() @ Y.double().T)
    Kc = center_gram_unbiased(K)
    Lc = center_gram_unbiased(L)

    hsic = (Kc * Lc).sum()
    norm = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return (hsic / (norm + eps)).to(orig_dtype)

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
