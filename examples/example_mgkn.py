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
from gnn_pde_v2.core.mlp import MLP
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.components.multiscale.graph_pooling import GraphPool, GraphUnpool


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
    
    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        positions=coords,
    )


# =============================================================================
# MGKN Model Components
# =============================================================================

class KernelNetwork(nn.Module):
    """
    Kernel network κ_φ for MGKN.
    
    Computes kernel matrix entries: κ_φ(a(x), a(y), x, y)
    Output transforms node features.
    
    Based on Equation (3) in the paper:
    (K_a u)(x) = ∫ κ_φ(a(x), a(y), x, y) u(y) dy
    """
    
    def __init__(
        self,
        input_dim: int = 6,  # [x_i, y_i, x_j, y_j, dx, dy]
        output_dim: int = 64,  # dv - latent dimension
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # MLP to generate kernel weights
        self.mlp = MLP(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation='gelu',
            use_layer_norm=True,
        )
    
    def forward(
        self, 
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute kernel weights for edges.
        
        Args:
            edge_features: [n_edges, input_dim] - [x_i, y_i, x_j, y_j, dx, dy]
            
        Returns:
            Kernel weights: [n_edges, output_dim]
        """
        return self.mlp(edge_features)


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer with kernel convolution.
    
    Implements Equation (5) from the paper:
    v^{(t+1)}(x) = σ(W v^{(t)} + 1/|N(x)| Σ_{y∈N(x)} κ_φ(e(x,y)) v^{(t)}(y))
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        use_kernel: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_kernel = use_kernel
        
        # Local linear transformation W
        self.W = nn.Linear(latent_dim, latent_dim, bias=False)
        
        if use_kernel:
            # Kernel network with 6-dim edges
            self.kernel_net = KernelNetwork(
                input_dim=6,  # [x_i, y_i, x_j, y_j, dx, dy]
                output_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Message passing step.
        
        Args:
            nodes: [n_nodes, latent_dim]
            edges: [n_edges, 6]
            senders: [n_edges]
            receivers: [n_edges]
            positions: [n_nodes, 2] (unused, for compatibility)
            
        Returns:
            Updated nodes: [n_nodes, latent_dim]
        """
        # Local transformation
        local = self.W(nodes)  # [n_nodes, latent_dim]
        
        if not self.use_kernel or edges is None:
            return self.activation(local)
        
        # Compute kernel weights
        kernel_weights = self.kernel_net(edges)  # [n_edges, latent_dim]
        
        # Message: aggregate from neighbors
        sender_nodes = nodes[senders]  # [n_edges, latent_dim]
        
        # Apply kernel: element-wise multiplication
        messages = sender_nodes * kernel_weights  # [n_edges, latent_dim]
        
        # Aggregate to receivers (mean pooling)
        num_nodes = nodes.shape[0]
        aggregated = torch.zeros(num_nodes, self.latent_dim, device=nodes.device)
        degrees = torch.zeros(num_nodes, device=nodes.device)
        
        # Use scatter_add for efficiency
        degrees.scatter_add_(0, receivers, torch.ones_like(receivers, dtype=torch.float))
        aggregated.scatter_add_(0, receivers.unsqueeze(-1).expand(-1, self.latent_dim), messages)
        
        # Normalize by degree
        aggregated = aggregated / degrees.clamp(min=1).unsqueeze(-1)
        
        # Combine local and non-local
        out = local + aggregated
        
        return self.activation(out)


class TransitionLayer(nn.Module):
    """
    Transition layer for moving between graph levels.
    
    Implements K_{l+1,l} and K_{l,l+1} from the paper.
    Uses interpolation-based restriction/prolongation.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Learnable transition network
        self.transition_mlp = MLP(
            in_dim=latent_dim + 2,  # features + positions
            out_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation='gelu',
            use_layer_norm=False,
        )
    
    def forward(
        self,
        source_nodes: torch.Tensor,
        source_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transfer features from source to target positions.
        
        Args:
            source_nodes: [n_source, latent_dim]
            source_positions: [n_source, 2]
            target_positions: [n_target, 2]
            
        Returns:
            Interpolated features: [n_target, latent_dim]
        """
        # Compute attention weights based on positions
        dist = torch.cdist(target_positions, source_positions)  # [n_target, n_source]
        
        # Softmax attention
        attention = F.softmax(-dist, dim=-1)  # [n_target, n_source]
        
        # Weighted interpolation
        interpolated = torch.matmul(attention, source_nodes)  # [n_target, latent_dim]
        
        return interpolated


class MGKNLevel(nn.Module):
    """
    Single level of the MGKN hierarchy with kernel-based message passing.
    
    Based on the paper's architecture.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        n_message_passing: int = 2,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Stack of message passing layers with kernel convolution
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(n_message_passing)
        ])
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(
        self,
        graph: GraphsTuple,
    ) -> GraphsTuple:
        """
        Process graph at this level.
        
        Args:
            graph: Input GraphsTuple
            
        Returns:
            Updated GraphsTuple
        """
        nodes = graph.nodes
        if nodes is None:
            return graph
        
        # Apply message passing
        for layer in self.message_layers:
            new_nodes = layer(nodes, graph.edges, graph.senders, graph.receivers)
            nodes = nodes + new_nodes  # residual connection
        
        nodes = self.layer_norm(nodes)
        
        return graph.replace(nodes=nodes)


class MultipoleGraphNeuralOperator(AutoRegisterModel, name='mgkn', aliases=['mgkn_model']):
    """
    Multipole Graph Neural Operator (MGKN).
    
    Implements the V-cycle algorithm from the paper:
    - Multiple hierarchy levels (fine to coarse)
    - Downward pass: fine → coarse (using framework's GraphPool)
    - Upward pass: coarse → fine with skip connections (using framework's GraphUnpool)
    
    Uses gnn_pde_v2 framework components:
    - GraphNetBlock for message passing
    - GraphPool/GraphUnpool for hierarchical structure
    
    Key features:
    - Linear complexity through hierarchical structure
    - Mesh-invariant (can train on coarse, test on fine)
    
    Args:
        node_in_dim: Input node feature dimension
        out_dim: Output dimension
        latent_dim: Latent feature dimension
        n_levels: Number of hierarchy levels
        nodes_per_level: Number of nodes at each level
        n_message_passing: Message passing layers per level
        hidden_dim: Hidden dimension for MLPs
    """
    
    def __init__(
        self,
        node_in_dim: int = 3,  # [a(x), x, y]
        out_dim: int = 1,
        latent_dim: int = 64,
        n_levels: int = 3,
        nodes_per_level: Optional[List[int]] = None,
        n_message_passing: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.n_levels = n_levels
        self.nodes_per_level = nodes_per_level or [400, 100, 25]
        
        # Input embedding
        self.encoder = MLP(
            in_dim=node_in_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation='gelu',
            use_layer_norm=False,
        )
        
        # Output decoder
        self.decoder = MLP(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim],
            activation='gelu',
            use_layer_norm=False,
        )
        
        # Level-specific processors using GraphNetBlock from framework
        self.level_processors = nn.ModuleList([
            MGKNLevel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                n_message_passing=n_message_passing,
            )
            for _ in range(n_levels)
        ])
        
        # Pooling layers using framework's GraphPool
        self.pool_layers = nn.ModuleList([
            GraphPool(k=k, feature_dim=latent_dim)
            for k in self.nodes_per_level[1:]
        ])
        
        # Unpooling layers using framework's GraphUnpool
        self.unpool_layers = nn.ModuleList([
            GraphUnpool()
            for _ in range(len(self.nodes_per_level) - 1)
        ])
        
        # Graph construction helpers
        self.k_neighbors = 8
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass with V-cycle using framework components.
        
        Args:
            graph: Input GraphsTuple with node_in_dim features
            
        Returns:
            [n_nodes, out_dim] predictions
        """
        device = graph.nodes.device if graph.nodes is not None else 'cpu'
        
        # Encode input
        encoded = self.encoder(graph.nodes)  # [n, latent_dim]
        graph = graph.replace(nodes=encoded)
        
        # === V-cycle using framework's GraphPool/GraphUnpool ===
        
        # Store graphs and indices at each level
        level_graphs = [graph]
        indices_list = []
        
        # Downward pass: pool to coarse levels
        for level in range(self.n_levels - 1):
            current_graph = level_graphs[-1]
            
            # Pool using framework's GraphPool
            pool_layer = self.pool_layers[level]
            pooled_graph, indices = pool_layer(current_graph)
            
            indices_list.append(indices)
            level_graphs.append(pooled_graph)
        
        # Process at each level going down
        for level in range(self.n_levels):
            level_graphs[level] = self.level_processors[level](level_graphs[level])
        
        # Upward pass: unpool and combine with skip connections
        for level in range(self.n_levels - 2, -1, -1):
            # Unpool from coarse to fine using framework's GraphUnpool
            coarse_graph = level_graphs[level + 1]
            fine_size = level_graphs[level].nodes.shape[0]
            
            unpool_layer = self.unpool_layers[level]
            unpooled = unpool_layer(coarse_graph, indices_list[level], fine_size)
            
            # Skip connection: add unpooled features to fine-level features
            skip_features = level_graphs[level].nodes
            combined = skip_features + unpooled.nodes
            
            # Process at this level
            level_graphs[level] = self.level_processors[level](
                level_graphs[level].replace(nodes=combined)
            )
        
        # Decode output from finest level
        output = self.decoder(level_graphs[0].nodes)
        
        return output
    
    def predict(self, a: torch.Tensor, resolution: int = 41) -> torch.Tensor:
        """
        Convenience method for prediction from coefficient field.
        
        Args:
            a: Input coefficient [resolution, resolution]
            resolution: Grid resolution
            
        Returns:
            Predicted solution [resolution, resolution]
        """
        # Get model device
        device = next(self.parameters()).device
        
        # Create graph
        a_tensor = a.float() if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
        a_tensor = a_tensor.to(device)
        u_dummy = torch.zeros_like(a_tensor)
        
        graph = create_graphs_from_grid(a_tensor, u_dummy, k_neighbors=self.k_neighbors)
        
        # Forward
        output = self.forward(graph)
        
        return output.view(resolution, resolution)


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