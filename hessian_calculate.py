import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
# from scipy.sparse import 

# ============================================================================
# MAIN FUNCTION: Compute Hessian eigenvalues for initialization
# ============================================================================

def compute_hessian_at_initialization(model, loss_fn, dataloader, 
                                       num_eigenvalues=10, device='cuda',
                                       use_subset=True, subset_size=150):
    """
    Memory-efficient version for Hessian computation.
    """
    # model.eval()
    model = model.to(device)
    
    # Get all parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    params_non = [p for p in model.parameters() if not p.requires_grad]
    param_shapes = [p.shape for p in params]
    
    print(f"Number of parameter tensors: {len(params)}")
    print(f"Number of non-trainable parameter tensors: {len(params_non)}")
    print(f"Total parameters: {sum(p.numel() for p in params):,}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Parameters - Total: {total_params:,} Trainable: {trainable:,}, Non-trainable: {non_trainable:,}")

    
    # Use subset of data to save memory
    if use_subset:
        from torch.utils.data import DataLoader, Subset
        dataset = dataloader.dataset
        indices = list(range(min(subset_size, len(dataset))))
        subset = Subset(dataset, indices)
        subset_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    else:
        subset_loader = dataloader
    
    def get_params_vector():
        return torch.cat([p.data.flatten() for p in params])
    
    # Memory-efficient loss computation
    def total_loss():
        total_loss = torch.tensor(0.0, device=device) 
        for inputs, targets in subset_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # with torch.no_grad():
            outputs = model(inputs)
            
            num_classes = outputs.shape[1]  # 100
            targets_onehot = torch.zeros_like(outputs)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
            batch_loss = loss_fn(outputs, targets_onehot)
            # return batch_loss
            
            total_loss = total_loss + batch_loss
            
            # Clear intermediate activations
            del inputs, targets, outputs, batch_loss
            torch.cuda.empty_cache()
        
        return total_loss
    
    def hessian_vector_product(v):
        # First gradient
        grad_loss = grad(total_loss(), params, create_graph=True, retain_graph=True)
        grad_flat = torch.cat([g.flatten() for g in grad_loss])
        
        # Dot product
        dot_product = (grad_flat * v).sum()
        
        # Second gradient
        hvp = grad(dot_product, params, retain_graph=False, create_graph=False)
        hvp_flat = torch.cat([h.flatten() for h in hvp])
        
        # Clear memory
        del grad_loss, grad_flat, dot_product, hvp
        torch.cuda.empty_cache()
        
        return hvp_flat
    
    v0 = get_params_vector()
    num_params = v0.numel()
    print(f"Total parameters: {num_params:,}")
    
    def matvec(v_numpy):
        v_tensor = torch.tensor(v_numpy, dtype=torch.float32, device=device)
        result = hessian_vector_product(v_tensor)
        return result.detach().cpu().numpy()
    
    linear_op = LinearOperator(
        shape=(num_params, num_params),
        matvec=matvec,
        dtype=np.float32
    )
    
    print(f"\nComputing top {num_eigenvalues} eigenvalues...")
    print(f"This may take a while and use significant memory...")
    
    try:
        eigenvalues, eigenvectors = eigsh(
            linear_op,
            k=num_eigenvalues,
            which='LM',
            v0=v0.detach().cpu().numpy(),
            maxiter=2000,
            tol=1e-5,
            ncv=min(50, num_params - 1)  # Reduced ncv for memory
        )
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM! Try running on CPU instead.")
        print(f"Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Lanczos failed ({e}), falling back to power iteration...")
        eigenvalues, eigenvectors = power_iteration(
            model, loss_fn, subset_loader, num_iterations=200, device=device
        )
    
    metadata = {
        'num_params': num_params,
        'num_eigenvalues_computed': len(eigenvalues) if eigenvalues is not None else 0,
        'model_type': 'initialization'
    }

    layer_hessian_values = compute_layer_wise_hessian_norms(model, subset_loader, device=device)
    print("\nLayer-wise Hessian norms:")
    for layer_name, hessian_norm in layer_hessian_values.items():
        print(f"  {layer_name}: {hessian_norm:.4f}")
    
    return eigenvalues, eigenvectors, metadata, layer_hessian_values

def power_iteration(model, loss_fn, dataloader, num_iterations=100, device='cuda'):
    """
    Fallback: Simple power iteration for largest eigenvalue.
    More robust but only gives top eigenvalue.
    """
    # model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    
    def get_params_vector():
        return torch.cat([p.flatten() for p in params])
    
    def total_loss():
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss = total_loss + loss_fn(outputs, targets)
        return total_loss
    
    # Initialize random unit vector
    v = torch.randn_like(get_params_vector())
    v = v / v.norm()
    
    eigenvalue = 0.0
    for i in range(num_iterations):
        h_v = hessian_vector_product(model, params, total_loss, v)
        
        # Rayleigh quotient: λ = v^T H v / v^T v
        new_eigenvalue = (v * h_v).sum().item()
        
        # Normalize
        norm = h_v.norm()
        if norm > 1e-10:
            v = h_v / norm
        
        if i % 20 == 0:
            print(f"  Power iteration {i}: λ = {new_eigenvalue:.4f}")
        
        eigenvalue = new_eigenvalue
    
    return np.array([eigenvalue]), np.array([v.detach().cpu().numpy()])


def hessian_vector_product(model, params, loss_fn, v):
    """Helper: compute H @ v"""
    grad_loss = grad(loss_fn(), params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grad_loss])
    dot_product = (grad_flat * v).sum()
    hvp = grad(dot_product, params, retain_graph=True)
    return torch.cat([h.flatten() for h in hvp])


def compute_hessian_trace(model, loss_fn, dataloader, num_samples=50, device='cuda', use_subset=True, subset_size=150):
    """
    Estimate Hessian trace using Hutchinson's method.
    Trace = sum of all eigenvalues = total curvature.
    """
    # model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    
    def get_params_vector():
        return torch.cat([p.flatten() for p in params])

    if use_subset:
        from torch.utils.data import DataLoader, Subset
        dataset = dataloader.dataset
        indices = list(range(min(subset_size, len(dataset))))
        subset = Subset(dataset, indices)
        subset_loader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=False)
    else:
        subset_loader = dataloader
    
    def total_loss():
        total_loss = torch.tensor(0.0, device=device) 
        for inputs, targets in subset_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # with torch.no_grad():
            outputs = model(inputs)
            
            num_classes = outputs.shape[1]  # 100
            targets_onehot = torch.zeros_like(outputs)
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
            batch_loss = loss_fn(outputs, targets_onehot)
            # return batch_loss
            
            total_loss = total_loss + batch_loss
            
            # Clear intermediate activations
            del inputs, targets, outputs, batch_loss
            torch.cuda.empty_cache()
        
        return total_loss

    trace_estimate = 0.0
    for i in range(num_samples):
        v = torch.randn_like(get_params_vector())
        v = v / v.norm()
        
        h_v = hessian_vector_product(model, params, total_loss, v)
        trace_estimate += (v * h_v).sum().item()
    
    trace_estimate /= num_samples
    return trace_estimate


# ============================================================================
# USAGE: Compare Model 1 vs Model 3 at Initialization
# ============================================================================

def compare_initialization_curvature(model1, model3, dataloader, 
                                      loss_fn, device='cuda'):
    """
    Compare curvature of Model 1 (random init) vs Model 3 (modified last layer)
    BEFORE fine-tuning on Data B starts.
    
    THIS DIRECTLY TESTS YOUR HYPOTHESIS.
    """
    print("=" * 70)
    print("COMPARING CURVATURE AT INITIALIZATION")
    print("=" * 70)
    print("\nModel 1: Random initialization")
    print("Model 3: Same as Model 1 but with modified last-layer weights")
    print("\nComputing Hessian eigenvalues BEFORE fine-tuning on Data B...")
    print("=" * 70)
    
    # Compute eigenvalues for Model 1
    print("\n[Model 1] Computing Hessian eigenvalues at initialization...")
    eigenvalues_1, eigenvectors_1, meta_1 = compute_hessian_at_initialization(
        model1, loss_fn, dataloader, num_eigenvalues=10, device=device
    )
    trace_1 = compute_hessian_trace(model1, loss_fn, dataloader, device=device)
    
    # Compute eigenvalues for Model 3
    print("\n[Model 3] Computing Hessian eigenvalues at initialization...")
    eigenvalues_3, eigenvectors_3, meta_3 = compute_hessian_at_initialization(
        model3, loss_fn, dataloader, num_eigenvalues=10, device=device
    )
    trace_3 = compute_hessian_trace(model3, loss_fn, dataloader, device=device)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Curvature at Initialization (Epoch 0)")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Model 1':<15} {'Model 3':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Top eigenvalue (λ_max)':<30} {eigenvalues_1[0]:<15.4f} {eigenvalues_3[0]:<15.4f} {eigenvalues_3[0] - eigenvalues_1[0]:<15.4f}")
    print(f"{'2nd eigenvalue':<30} {eigenvalues_1[1] if len(eigenvalues_1) > 1 else 0:<15.4f} {eigenvalues_3[1] if len(eigenvalues_3) > 1 else 0:<15.4f} {(eigenvalues_3[1] - eigenvalues_1[1]) if len(eigenvalues_1) > 1 else 0:<15.4f}")
    print(f"{'Hessian Trace (total curvature)':<30} {trace_1:<15.2f} {trace_3:<15.2f} {trace_3 - trace_1:<15.2f}")
    print(f"{'Condition number':<30} {eigenvalues_1[0]/eigenvalues_1[-1]:<15.2f} {eigenvalues_3[0]/eigenvalues_3[-1]:<15.2f} {(eigenvalues_3[0]/eigenvalues_3[-1]) - (eigenvalues_1[0]/eigenvalues_1[-1]):<15.2f}")
    
    print("\nTop 10 Eigenvalues (descending):")
    print(f"{'Rank':<6} {'Model 1':<15} {'Model 3':<15} {'Difference':<15}")
    print("-" * 50)
    for i in range(min(10, len(eigenvalues_1))):
        diff = eigenvalues_3[i] - eigenvalues_1[i]
        print(f"{i+1:<6} {eigenvalues_1[i]:<15.4f} {eigenvalues_3[i]:<15.4f} {diff:<15.4f}")
    
    # Interpret results
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    lambda_max_diff = eigenvalues_3[0] - eigenvalues_1[0]
    trace_diff = trace_3 - trace_1
    
    if lambda_max_diff < 0:
        print(f"\n✅ Model 3 has SMALLER λ_max ({lambda_max_diff:+.4f})")
        print("   → Model 3 starts in a FLATTER region")
        print("   → Supports hypothesis: better initialization curvature")
    elif lambda_max_diff > 0:
        print(f"\n❌ Model 3 has LARGER λ_max ({lambda_max_diff:+.4f})")
        print("   → Model 3 starts in a SHARPER region")
        print("   → Hypothesis REJECTED at initialization")
    else:
        print(f"\n⚠️  Similar initial curvature ({lambda_max_diff:+.4f})")
    
    if trace_diff < 0:
        print(f"\n✅ Model 3 has SMALLER trace ({trace_diff:+.2f})")
        print("   → Model 3 has LOWER total curvature")
        print("   → Wider basin of attraction")
    
    return {
        'model1_eigenvalues': eigenvalues_1,
        'model3_eigenvalues': eigenvalues_3,
        'model1_trace': trace_1,
        'model3_trace': trace_3,
        'lambda_max_diff': lambda_max_diff,
        'trace_diff': trace_diff
    }

def compute_layer_wise_hessian_norms(model, data_loader, batch_size=32, device='cuda', num_iterations=30):
    """
    Compute Hessian norm for each layer separately.
    
    FIXED: Recompute gradient inside power iteration loop to avoid retain_graph issues.
    """
    model.eval()
    model = model.to(device)
    
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(data_loader))
    x, y = batch[0].to(device), batch[1].to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    layer_hessians = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        num_params = param.numel()
        
        # Initialize random vector for this layer
        v = torch.randn(num_params, device=device)
        v = v / torch.norm(v)
        
        # Power iteration - recompute gradient each iteration
        for iteration in range(num_iterations):
            # Forward pass and loss (fresh computation each time)
            output = model(x)
            loss = criterion(output, y)
            
            # Compute gradient for THIS layer (fresh graph each iteration)
            grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)
            g_flat = grad[0].flatten()
            
            # Power iteration step
            directional = torch.dot(v, g_flat)
            hvp = torch.autograd.grad(directional, param, create_graph=False, retain_graph=False)
            hvp_flat = hvp[0].flatten()
            
            hvp_norm = torch.norm(hvp_flat)
            if hvp_norm > 1e-8:
                v = hvp_flat / hvp_norm
            else:
                break
        
        # Compute final eigenvalue (fresh computation)
        output = model(x)
        loss = criterion(output, y)
        grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)
        g_flat = grad[0].flatten()
        directional = torch.dot(v, g_flat)
        hvp_final = torch.autograd.grad(directional, param, retain_graph=False)
        hvp_flat_final = hvp_final[0].flatten()
        
        eigenvalue = torch.dot(v, hvp_flat_final).item()
        layer_hessians[name] = abs(eigenvalue)
    
    return layer_hessians
# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load your model architecture
    # model1 = YourModelClass()  # Random initialization
    # model3 = YourModelClass()  # Same, but modify last layer
    
    # Example: Modify last layer weights for Model 3
    # def modify_last_layer(model, modification_func):
    #     # Get the last linear layer
    #     last_layer_name = list(model.named_parameters())[-1][0]
    #     last_layer = dict(model.named_parameters())[last_layer_name]
    #     last_layer.data = modification_func(last_layer.data)
    #     return model
    
    # Example usage:
    # model1 = YourModelClass()  # Random init
    # model3 = YourModelClass()  # Random init
    # model3 = modify_last_layer(model3, lambda w: w + 0.1 * torch.randn_like(w))
    
    # Your data loader for Data B
    # dataloader = your_data_loader  # Data B, training set
    
    # Your loss function
    # loss_fn = nn.CrossEntropyLoss()
    
    # Run the comparison
    # results = compare_initialization_curvature(
    #     model1, model3, dataloader, loss_fn, device=device
    # )
    
    print("\nReplace the example variables with your actual model and data.")
    print("Then run: results = compare_initialization_curvature(model1, model3, dataloader, loss_fn)")
