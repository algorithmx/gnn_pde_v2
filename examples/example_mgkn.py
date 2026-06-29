"""
Multipole Graph Neural Operator (MGKN) for Parametric PDEs

This implementation recreates the MGKN model from:
Li et al., "Multipole Graph Neural Operator for Parametric PDEs", NeurIPS 2020

Key innovations:
- Multi-level hierarchical graph structure
- V-cycle algorithm for message passing
- Kernel networks for integral operators
- Linear complexity with mesh-invariant generalization

Uses gnn_pde_v2 framework components where possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
import math

# Framework imports
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.models import MGKN


# =============================================================================
# Data Generation: Darcy Flow Equation
# =============================================================================

def generate_random_coefficient(
    resolution: int, 
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate random permeability coefficient a(x) for Darcy flow.
    
    Uses random fields - simplified version for demonstration.
    
    Args:
        resolution: Grid resolution (s x s)
        seed: Random seed for reproducibility
        
    Returns:
        Coefficient field [resolution, resolution]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Simple approach: random smooth field using low-rank components
    n_components = 5
    
    coeffs = torch.randn(n_components, device='cpu')
    
    field = torch.zeros(resolution, resolution)
    for k in range(n_components):
        # Create spatial pattern
        freq_x = (k + 1) * 2 * math.pi
        freq_y = (k + 1) * 3 * math.pi
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        pattern = torch.sin(freq_x * xx + coeffs[k]) * torch.cos(freq_y * yy + coeffs[k])
        field += pattern * (1.0 / (k + 1))
    
    # Make positive
    field = (field - field.min() + 0.5)
    
    return field


def solve_darcy_finite_difference(
    a: torch.Tensor, 
    f: Optional[torch.Tensor] = None,
    n_iter: int = 100,
) -> torch.Tensor:
    """
    Solve steady-state Darcy flow - simplified version using scipy.
    
    -∇·(a(x)∇u(x)) = f(x),  x ∈ (0,1)²
    u(x) = 0,  x ∈ ∂(0,1)²
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    
    # Convert to numpy
    a_np = a.cpu().numpy() if a.device.type == 'cuda' else a.numpy()
    if f is None:
        f_np = np.ones_like(a_np)
    else:
        f_np = f.cpu().numpy() if f.device.type == 'cuda' else f.numpy()
    
    resolution = a_np.shape[0]
    n = resolution * resolution
    h = 1.0 / (resolution - 1)
    
    # Build sparse matrix
    diagonals = []
    offsets = []
    
    # Create finite difference stencil
    # For each interior point: -a*(u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/h^2 - a*(u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/h^2
    
    main_diag = np.zeros(n)
    x_minus = np.zeros(n)
    x_plus = np.zeros(n)
    y_minus = np.zeros(n)
    y_plus = np.zeros(n)
    rhs = f_np.flatten()
    
    for i in range(resolution):
        for j in range(resolution):
            idx = i * resolution + j
            
            if i == 0 or i == resolution - 1 or j == 0 or j == resolution - 1:
                # Dirichlet boundary: u = 0
                main_diag[idx] = 1.0
                rhs[idx] = 0.0
            else:
                a_center = a_np[i, j]
                coeff = a_center / (h * h)
                
                main_diag[idx] = 4.0 * coeff
                
                # West neighbor
                if j > 0:
                    x_minus[idx] = -coeff
                
                # East neighbor  
                if j < resolution - 1:
                    x_plus[idx] = -coeff
                    
                # South neighbor
                if i > 0:
                    y_minus[idx] = -coeff
                    
                # North neighbor
                if i < resolution - 1:
                    y_plus[idx] = -coeff
    
    # Assemble sparse matrix
    data = [main_diag, x_minus, x_plus, y_minus, y_plus]
    offsets = [0, -1, 1, -resolution, resolution]
    A = sp.diags(data, offsets, shape=(n, n), format='csc')
    
    # Solve
    u_np = spla.spsolve(A, rhs)
    
    return torch.from_numpy(u_np.reshape(resolution, resolution)).float().to(a.device)


def generate_darcy_data(
    n_samples: int = 100,
    resolution: int = 41,
    seed: int = 42
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate training data for Darcy flow.
    
    Args:
        n_samples: Number of samples to generate
        resolution: Grid resolution
        seed: Random seed
        
    Returns:
        List of dicts with 'input' and 'output' tensors
    """
    torch.manual_seed(seed)
    
    data = []
    for i in range(n_samples):
        # Generate random coefficient
        a = generate_random_coefficient(resolution, seed=seed + i)
        
        # Solve PDE
        u = solve_darcy_finite_difference(a)
        
        data.append({
            'input': a,  # [resolution, resolution]
            'output': u,  # [resolution, resolution]
            'resolution': resolution,
        })
        
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{n_samples} samples")
    
    return data


def create_graphs_from_grid(
    a: torch.Tensor,
    u: torch.Tensor,
    k_neighbors: int = 8,
) -> GraphsTuple:
    """
    Create GraphsTuple from grid data for MGKN.
    
    Edge features: [x_i, y_i, x_j, y_j, dx, dy] = 6 dims
    Node features: [a(x), x, y] = 3 dims
    
    Args:
        a: Input coefficient [resolution, resolution]
        u: Output solution [resolution, resolution]
        k_neighbors: Number of nearest neighbors for edges
        
    Returns:
        GraphsTuple with node/edge features
    """
    resolution = a.shape[0]
    device = a.device
    
    # Flatten grid to points
    a_flat = a.flatten()  # [n]
    u_flat = u.flatten()  # [n]
    
    # Create coordinates
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [n, 2]
    
    n = coords.shape[0]
    
    # Build k-NN graph using simple distance-based approach
    senders = []
    receivers = []
    edge_attrs = []
    
    # Compute edges using vectorized approach
    dist = torch.cdist(coords, coords)  # [n, n]
    
    # Get k+1 nearest (including self)
    _, indices = torch.topk(dist, k=k_neighbors + 1, largest=False)
    
    # Create edges (excluding self-loops)
    for i in range(n):
        neighbors = indices[i, 1:k_neighbors+1]  # Skip self
        for j in neighbors:
            senders.append(i)
            receivers.append(j)
            
            # Edge attribute: [x_i, y_i, x_j, y_j, dx, dy]
            diff = coords[i] - coords[j]
            edge_attrs.append(torch.cat([
                coords[i],      # position i
                coords[j],      # position j
                diff           # relative position
            ]))
    
    senders = torch.tensor(senders, dtype=torch.long, device=device)
    receivers = torch.tensor(receivers, dtype=torch.long, device=device)
    edges = torch.stack(edge_attrs, dim=0)  # [n_edges, 6]
    
    # Node features: [a(x), x, y]
    nodes = torch.cat([
        a_flat.unsqueeze(1),
        coords
    ], dim=1)  # [n, 3]
    
    return GraphsTuple.from_flat(
        nodes=nodes,
        n_node=torch.tensor([nodes.shape[0]], device=nodes.device),
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_edge=torch.tensor([edges.shape[0]], device=edges.device),
        positions=coords,
    )


# =============================================================================
# MGKN Model
# =============================================================================

class MultipoleGraphNeuralOperator(MGKN, name='mgkn_example', namespace='example'):
    """Example MGKN subclass adding grid-based predict() for mesh-invariance tests."""

    k_neighbors = 8

    def predict(self, a, resolution=41):
        device = next(self.parameters()).device
        a_tensor = (a.float() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)).to(device)
        graph = create_graphs_from_grid(a_tensor, torch.zeros_like(a_tensor), k_neighbors=self.k_neighbors)
        return self.forward(graph).view(resolution, resolution)


# =============================================================================
# Training Utilities
# =============================================================================

def train_mgkn(
    model: MultipoleGraphNeuralOperator,
    train_data: List[Dict[str, torch.Tensor]],
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> Dict[str, List[float]]:
    """
    Train MGKN model.
    
    Args:
        model: MGKN model
        train_data: List of {'input': a, 'output': u}
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {'loss': []}
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for sample in train_data:
            a = sample['input'].to(device)
            u = sample['output'].to(device)
            resolution = sample['resolution']
            
            # Create graph
            graph = create_graphs_from_grid(a, u, k_neighbors=model.k_neighbors)
            graph = graph.to(device)
            
            # Forward
            output = model(graph)  # [n, 1]
            
            # Get target
            target = u.flatten().unsqueeze(1)  # [n, 1]
            
            # Loss
            loss = F.mse_loss(output, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        history['loss'].append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return history


def evaluate_mesh_invariance(
    model: MultipoleGraphNeuralOperator,
    train_res: int,
    test_res: int,
    n_samples: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> float:
    """
    Test mesh invariance: train on coarse, test on fine.
    
    Args:
        model: MGKN model
        train_res: Training resolution
        test_res: Test resolution (different from train)
        n_samples: Number of test samples
        device: Device
        
    Returns:
        Average relative L2 error
    """
    model.eval()
    torch.manual_seed(123)
    
    errors = []
    
    with torch.no_grad():
        for i in range(n_samples):
            # Generate test sample
            a = generate_random_coefficient(test_res, seed=1000 + i).to(device)
            u = solve_darcy_finite_difference(a)
            
            # Predict (should work at any resolution)
            pred = model.predict(a, resolution=test_res)
            
            # Compute relative L2 error (ensure same device)
            pred = pred.to(u.device)
            error = torch.sqrt(((pred - u) ** 2).sum()) / torch.sqrt((u ** 2).sum())
            errors.append(error.item())
    
    return np.mean(errors)


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """
    Demonstrate MGKN model usage.
    """
    print("=" * 60)
    print("Multipole Graph Neural Operator (MGKN)")
    print("Reference: Li et al., NeurIPS 2020")
    print("=" * 60)
    
    # Configuration
    resolution = 21  # Small for fast training
    latent_dim = 32
    n_levels = 3
    nodes_per_level = [100, 50, 25]
    
    print(f"\nConfiguration:")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Levels: {n_levels}")
    print(f"  Nodes per level: {nodes_per_level}")
    
    # Create model
    model = MultipoleGraphNeuralOperator(
        node_in_dim=3,  # [a(x), x, y]
        out_dim=1,
        latent_dim=latent_dim,
        n_levels=n_levels,
        nodes_per_level=nodes_per_level,
        n_message_passing=2,
        hidden_dim=128,
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate small training set
    print("\nGenerating training data...")
    train_data = generate_darcy_data(n_samples=20, resolution=resolution, seed=42)
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining on {device}...")
    
    history = train_mgkn(
        model, 
        train_data, 
        n_epochs=50, 
        lr=1e-3,
        device=device,
    )
    
    print(f"\nFinal training loss: {history['loss'][-1]:.6f}")
    
    # Test mesh invariance
    print("\n" + "-" * 60)
    print("Testing mesh invariance (train on 41, test on 61)...")
    
    test_error = evaluate_mesh_invariance(
        model,
        train_res=resolution,
        test_res=61,
        n_samples=5,
        device=device,
    )
    
    print(f"Test relative L2 error: {test_error:.4f}")
    
    # Quick inference demo
    print("\n" + "-" * 60)
    print("Quick inference demo...")
    
    model.eval()
    with torch.no_grad():
        a_test = generate_random_coefficient(resolution, seed=999)
        pred = model.predict(a_test, resolution=resolution)
        print(f"Input shape: {a_test.shape}")
        print(f"Output shape: {pred.shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    model, history = example_usage()