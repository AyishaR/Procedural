import torch
import torch.nn as nn
import numpy as np
from kdyck_generation import generate_k_dyck

def generate_orthogonal_embeddings(vocab_size=128, embed_dim=768, device='cuda'):
    """
    Generate fixed orthogonal embeddings for k=64 Dyck tokens (128 symbols).
    Ensures perfect orthogonality: E^T * E = I (identity matrix).
    """
    rand_matrix = torch.randn(embed_dim, vocab_size, device=device)
    q, r = torch.linalg.qr(rand_matrix)
    
    # print sizes of all matrices
    print("Random matrix shape:", rand_matrix.shape)
    print("Q matrix shape:", q.shape)
    print("R matrix shape:", r.shape)

    # Ensure proper orthogonality (fix reflection component)
    # q = q @ torch.diag(torch.sign(r.diag())) - TODO: Q. reflection component?
    
    return q

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embeddings = generate_orthogonal_embeddings(128, 192, device=device).T  
print("Embeddings shape:", embeddings.shape) 
print(embeddings)

# Verify orthogonality
gram_matrix = embeddings @ embeddings.T
ortho_error = torch.norm(gram_matrix - torch.eye(128, device=device))
print(f"Orthogonality error: {ortho_error:.2e}") 

# Usage
embeddings_layer = nn.Embedding.from_pretrained(ortho_embeddings, freeze=True)
sample_dyck = generate_k_dyck(k=64, length=16).unsqueeze(0).to(device)
embedded = embeddings_layer(sample_dyck) 

print("Embedded output shape:", embedded.shape) 
print(embedded)