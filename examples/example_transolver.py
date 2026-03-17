"""
Example: Transolver (Physics-Attention Mechanism)

This example recreates the Transolver model from:
https://github.com/thuml/Transolver

Original Work Reference:
------------------------
Wu, H., Hu, T., Luo, H., Wang, J., & Long, M. (2024).
"Transolver: A Fast Transformer Solver for PDEs on General Geometries."
International Conference on Machine Learning (ICML 2024).
Paper: https://arxiv.org/abs/2402.02366

Key Innovation:
---------------
Transolver introduces "Physics-Attention" which reduces attention complexity
from O(N²) to O(N×G + G²) where G << N (number of physics tokens).

This implementation uses the gnn_pde_v2 framework components:
- PhysicsTokenAttention: Framework's implementation with paper-faithful defaults
- MLP: Framework's MLP component
- AutoRegisterModel: Framework's model registration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

# Import framework components
from gnn_pde_v2.core import AutoRegisterModel, MLP
from gnn_pde_v2.components.attention import PhysicsTokenAttention


class TransolverBlock(nn.Module):
    """
    Transolver transformer block using framework's PhysicsTokenAttention.

    Uses paper-faithful defaults:
    - use_slice_normalization=True: Critical for proper token aggregation
    - use_learnable_tokens=False: Paper doesn't use learnable token bias
    - qkv_mode='direct': Uses per-head Q/K/V linears as in paper
    - use_orthogonal_init=True: Orthogonal init for slice projection
    - temperature=0.5: Paper's default temperature
    - temperature_mode='learnable_scalar': Paper's learnable temperature

    Parity notes vs upstream `external_research/Transolver/.../models/Transolver.py`:
    - Upstream defines both `mlp` and `mlp_new`, but only `mlp` is used in the forward pass.
      We intentionally implement just the used path.
    - Feed-forward is a 2-linear mapping with an activation in between; we keep `use_layer_norm=False`
      to avoid introducing extra normalization.
    """
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act: str = 'gelu',
        mlp_ratio: float = 4.0,
        last_layer: bool = False,
        out_dim: int = 1,
        slice_num: int = 32,
    ):
        super().__init__()
        
        self.last_layer = last_layer
        
        self.ln_1 = nn.LayerNorm(hidden_dim)
        
        # Use framework's PhysicsTokenAttention with paper-faithful defaults
        self.attn = PhysicsTokenAttention(
            dim=hidden_dim,
            n_tokens=slice_num,
            n_heads=num_heads,
            dropout=dropout,
            temperature=0.5,               # Paper default
            temperature_mode='learnable_scalar',  # Matches paper's learnable temp
            # Paper-fidelity options:
            use_slice_normalization=True,  # Critical from paper
            use_learnable_tokens=False,     # Paper doesn't use this
            qkv_mode='direct',              # Paper-faithful QKV
            use_orthogonal_init=True,       # Recommended in paper
        )
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network using framework's MLP
        self.mlp = MLP(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dims=[int(hidden_dim * mlp_ratio)],
            activation=act,
            use_layer_norm=False,
        )
        
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fx: [B, N, hidden_dim]
        Returns:
            [B, N, out_dim] if last_layer else [B, N, hidden_dim]
        """
        # Attention with residual
        fx = self.attn(self.ln_1(fx)) + fx
        
        # MLP with residual using framework component
        fx = self.mlp(self.ln_2(fx)) + fx
        
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(AutoRegisterModel, name='transolver', namespace='example'):
    """
    Transolver implementation using gnn_pde_v2 framework components.
    
    Precise equivalent of Transolver model.
    Original implementation: Airfoil-Design-AirfRANS/models/Transolver.py
    
    Architecture:
        Input x [B, N, fun_dim], positions [B, N, space_dim]
            ↓
        Unified Position Encoding: distances to ref×ref grid
            ↓
        Preprocessor: MLP(x, pos, distances) using framework's MLP
            ↓
        Transolver Blocks × n_layers:
            - Physics-Attention: Slice → Attend(G tokens) → Deslice
              (using framework's PhysicsTokenAttention with paper defaults)
            - Feed-Forward: Framework's MLP on each point
            - LayerNorm + Residual
            ↓
        Output [B, N, out_dim] (last layer has direct projection)
    
    Key innovation: Physics-Attention reduces complexity from O(N²) to O(N×G + G²)
    """
    
    def __init__(
        self,
        space_dim: int = 2,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = 'gelu',
        mlp_ratio: float = 1.0,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = True,
    ):
        super().__init__()
        
        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.ref = ref
        self.unified_pos = unified_pos
        
        # Activation mapping
        ACTIVATION = {
            'gelu': nn.GELU,
            'relu': nn.ReLU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
        }
        self.act = ACTIVATION.get(act, nn.GELU)()
        
        # Preprocessor MLP using framework component
        if unified_pos:
            preproc_input = fun_dim + space_dim + ref * ref
        else:
            preproc_input = fun_dim + space_dim
        
        self.preprocess = MLP(
            in_dim=preproc_input,
            out_dim=n_hidden,
            hidden_dims=[n_hidden * 2],  # 2-layer MLP
            activation=act,
            use_layer_norm=False,
        )
        
        # Learnable placeholder (from original implementation)
        self.placeholder = nn.Parameter(
            (1.0 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        
        # Transolver blocks using framework's PhysicsTokenAttention
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim,
                slice_num=slice_num,
            )
            for i in range(n_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def get_grid(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute unified position encoding.
        
        For each point, computes distances to all reference grid points.
        This provides translation-invariant position encoding.
        """
        batchsize = pos.shape[0]
        device = pos.device
        
        # Create reference grid (uniformly spaced in [-2, 4] × [-1.5, 1.5])
        gridx = torch.tensor(np.linspace(-2, 4, self.ref), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        
        gridy = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref ** 2, 2)
        
        # Compute Euclidean distances from each node to each reference point
        diff = pos[:, :, None, :] - grid_ref[:, None, :, :]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        
        return distances
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, N, fun_dim] - Input function values
            pos: [B, N, space_dim] - Node positions
            
        Returns:
            [B, N, out_dim] - Output predictions
        """
        B, N, _ = x.shape
        
        # Unified position encoding: distances to reference grid
        if self.unified_pos:
            new_pos = self.get_grid(pos)
            x = torch.cat((x, new_pos), dim=-1)
        
        # Concatenate with spatial coordinates
        x = torch.cat((x, pos), dim=-1)
        
        # Preprocess: Project to hidden dimension using framework MLP
        fx = self.preprocess(x.reshape(-1, x.shape[-1])).reshape(B, N, self.n_hidden)
        
        # Add learnable placeholder (bias term)
        fx = fx + self.placeholder[None, None, :]
        
        # Transolver blocks
        for block in self.blocks:
            fx = block(fx)
        
        return fx
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'transolver',
            'space_dim': self.space_dim,
            'n_hidden': self.n_hidden,
            'n_layers': len(self.blocks),
            'ref': self.ref,
            'unified_pos': self.unified_pos,
        }


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the Transolver equivalent.
    
    This configuration is suitable for airfoil design problems
    (AirfRANS dataset) as described in the paper.
    """
    print("=" * 60)
    print("Transolver Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Create model with paper defaults
    model = Transolver(
        space_dim=2,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act='gelu',
        mlp_ratio=1.0,
        fun_dim=1,
        out_dim=1,
        slice_num=32,
        ref=8,
        unified_pos=True,
    )
    
    # Example input
    batch_size = 2
    num_points = 1000
    
    # Input function values
    x = torch.randn(batch_size, num_points, 1)
    
    # Node positions (can be unstructured)
    pos = torch.randn(batch_size, num_points, 2)
    
    # Forward pass
    output = model(x, pos)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden dimension: {model.n_hidden}")
    print(f"  Number of layers: {len(model.blocks)}")
    print(f"  Reference grid: {model.ref}×{model.ref}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input shape: {x.shape}")
    print(f"  Position shape: {pos.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Show framework integration
    first_block = model.blocks[0]
    print(f"\nFramework Integration:")
    print(f"  Using PhysicsTokenAttention: {type(first_block.attn).__name__}")
    print(f"  Paper-faithful settings:")
    print(f"    - use_slice_normalization: {first_block.attn.use_slice_normalization}")
    print(f"    - use_learnable_tokens: {first_block.attn.use_learnable_tokens}")
    print(f"    - qkv_mode: {first_block.attn.qkv_mode}")
    print(f"    - temperature_mode: {first_block.attn.temperature_mode}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    return model, x, pos, output


if __name__ == "__main__":
    model, x, pos, output = example_usage()
