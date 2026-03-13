"""
Example: NeuralOperator FNO (Fourier Neural Operator)

This example recreates the FNO model from the neuraloperator repository:
https://github.com/neuraloperator/neuraloperator

Original Work Reference:
------------------------
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021).
"Fourier Neural Operator for Parametric Partial Differential Equations."
International Conference on Learning Representations (ICLR 2021).
Paper: https://arxiv.org/abs/2010.08895

Key Innovation:
---------------
FNO learns operators between function spaces using Fourier transformations.
Instead of convolution in physical space, it performs convolution in Fourier space.

This implementation uses the gnn_pde_v2 framework components while maintaining
exact equivalence to the original neuraloperator FNO architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union

# Import framework components
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.components import FNOProcessor, SpectralConv
from gnn_pde_v2.core import MLP


class NeuralOperatorFNO(AutoRegisterModel, name='neuraloperator_fno', namespace='example'):
    """
    Precise equivalent of neuraloperator's FNO model using gnn_pde_v2 framework.
    
    Original implementation: neuralop/models/fno.py
    
    Architecture:
        Input [B, C_in, *spatial]
            ↓
        Positional Embedding: Concatenate coordinate grid
            ↓
        Lifting: Conv1×1 encoder (framework's lifting in FNOProcessor)
            ↓
        FNO Blocks × n_layers (framework's FNOBlock):
            - Spectral Conv: FFT → Mode multiplication → IFFT
            - ChannelMLP: 1×1 conv MLP
            - GELU activation
            ↓
        Projection: Conv1×1 decoder (framework's projection in FNOProcessor)
            ↓
        Output [B, C_out, *spatial]
    """
    
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        lifting_channel_ratio: float = 2.0,
        projection_channel_ratio: float = 2.0,
        use_channel_mlp: bool = True,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_dropout: float = 0.0,
    ):
        """
        Initialize FNO model using framework components.
        
        Args:
            n_modes: Number of Fourier modes kept per dimension, e.g., (16, 16)
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Width of the latent representation (default: 64)
            n_layers: Number of Fourier layers (default: 4)
            lifting_channel_ratio: Hidden dim in lifting = ratio * hidden_channels
            projection_channel_ratio: Hidden dim in projection = ratio * hidden_channels
            use_channel_mlp: Whether to use ChannelMLP in FNO blocks
            channel_mlp_expansion: Expansion factor for ChannelMLP
            channel_mlp_dropout: Dropout rate for ChannelMLP
        """
        super().__init__()
        
        self.n_modes = n_modes
        self.n_dim = len(n_modes)
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        
        # Calculate lifting and projection channels (ratios from neuraloperator)
        self.lifting_channels = int(lifting_channel_ratio * hidden_channels)
        self.projection_channels = int(projection_channel_ratio * hidden_channels)
        
        # Positional embedding adds n_dim channels for grid coordinates
        # This is handled in forward() by concatenating the grid
        
        # Lifting layer: ChannelMLP-style 2-layer pointwise projection
        lifting_in_channels = in_channels + self.n_dim  # +n_dim for positional encoding
        if self.n_dim not in {1, 2, 3}:
            raise ValueError(f"n_dim must be 1, 2, or 3, got {self.n_dim}")
        conv_factory = self._make_pointwise_factory(self.n_dim)
        self.lifting = MLP(
            in_dim=lifting_in_channels,
            out_dim=hidden_channels,
            hidden_dims=[self.lifting_channels],
            activation='gelu',
            dropout=0.0,
            norm=None,
            linear_factory=conv_factory,
            use_layer_norm=False,
        )
        
        # FNO Blocks: Core processing layers using framework's SpectralConv
        # Each block consists of: SpectralConv + ChannelMLP + Residual
        self.fno_blocks = nn.ModuleList([
            FNOBlockFramework(
                hidden_channels=hidden_channels,
                n_modes=n_modes,
                n_dim=self.n_dim,
                use_channel_mlp=use_channel_mlp,
                channel_mlp_expansion=channel_mlp_expansion,
                channel_mlp_dropout=channel_mlp_dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Projection layer: ChannelMLP-style 2-layer pointwise projection
        self.projection = MLP(
            in_dim=hidden_channels,
            out_dim=out_channels,
            hidden_dims=[self.projection_channels],
            activation='gelu',
            dropout=0.0,
            norm=None,
            linear_factory=conv_factory,
            use_layer_norm=False,
        )
    
    def _get_grid(self, shape, device):
        """
        Generate coordinate grid in [0, 1]^d.
        
        For each spatial dimension i, creates a grid:
            grid_i[j] = j / (size_i - 1) for j in [0, size_i)
        """
        grids = []
        for i, s in enumerate(shape):
            grid = torch.linspace(0, 1, s, device=device)
            # Reshape to broadcast: [1, 1, ..., s, ..., 1]
            view_shape = [1] * (self.n_dim + 2)
            view_shape[i + 2] = s
            grid = grid.view(*view_shape)
            # Broadcast to full shape [1, 1, *shape]
            broadcast_shape = [1, 1] + list(shape)
            grid = grid.expand(*broadcast_shape)
            grids.append(grid)
        # Concatenate along channel dim: [1, n_dim, *shape]
        return torch.cat(grids, dim=1)

    @staticmethod
    def _make_pointwise_factory(n_dim: int):
        conv = nn.Conv1d if n_dim == 1 else nn.Conv2d if n_dim == 2 else nn.Conv3d
        return lambda a, b: conv(a, b, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching neuraloperator's FNO.
        
        Args:
            x: [B, in_channels, *spatial_dims] - Input function discretization
            
        Returns:
            [B, out_channels, *spatial_dims] - Output function discretization
        """
        B = x.shape[0]
        spatial_shape = x.shape[2:]
        
        # 1. Positional embedding: concatenate grid coordinates
        grid = self._get_grid(spatial_shape, x.device)
        grid = grid.expand(B, -1, *([-1] * self.n_dim))  # [B, n_dim, *spatial]
        x = torch.cat([x, grid], dim=1)  # [B, in_channels + n_dim, *spatial]
        
        # 2. Lifting: Project to hidden dimension
        x = self.lifting(x)  # [B, hidden_channels, *spatial]
        
        # 3. FNO Blocks: Fourier domain processing with residual inside each block
        for block in self.fno_blocks:
            x = block(x)
        
        # 4. Projection: Map to output channels
        x = self.projection(x)  # [B, out_channels, *spatial]
        
        return x
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'neuraloperator_fno',
            'n_modes': self.n_modes,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'hidden_channels': self.hidden_channels,
            'n_layers': self.n_layers,
        }


# Backward-compatible alias expected by tests/examples.
FNO = NeuralOperatorFNO


class FNOBlockFramework(nn.Module):
    """
    FNO Block using framework's SpectralConv.
    
    Matches the essential neuraloperator FNO block behavior:
    - spectral branch + linear 1x1 skip branch
    - optional channel MLP after branch summation
    - residual connection handled inside the block
    """
    
    def __init__(
        self,
        hidden_channels: int,
        n_modes: Tuple[int, ...],
        n_dim: int,
        use_channel_mlp: bool = True,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.n_dim = n_dim
        conv = nn.Conv1d if n_dim == 1 else nn.Conv2d if n_dim == 2 else nn.Conv3d
        
        # Spectral convolution using framework component
        self.spectral_conv = SpectralConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=list(n_modes),
            separable=False,
        )
        self.linear_skip = conv(hidden_channels, hidden_channels, 1)
        
        # Channel MLP: Local operator via 1×1 convolutions
        if use_channel_mlp:
            mlp_hidden = int(hidden_channels * channel_mlp_expansion)
            conv_factory = NeuralOperatorFNO._make_pointwise_factory(n_dim)
            self.channel_mlp = MLP(
                in_dim=hidden_channels,
                out_dim=hidden_channels,
                hidden_dims=[mlp_hidden],
                activation='gelu',
                dropout=channel_mlp_dropout,
                norm=None,
                linear_factory=conv_factory,
                use_layer_norm=False,
            )
        else:
            self.channel_mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, *spatial]
        Returns:
            [B, C, *spatial]
        """
        residual = x
        x1 = self.spectral_conv(x) + self.linear_skip(x)
        if self.channel_mlp is not None:
            x1 = self.channel_mlp(x1)
        return residual + x1


# ============================================================================
# Alternative: Simple FNO using framework's FNOProcessor
# ============================================================================

class SimpleFNO(AutoRegisterModel, name='simple_fno', namespace='example'):
    """
    Simplified FNO using framework's FNOProcessor directly.
    
    This version uses the framework's built-in FNOProcessor which handles
    lifting, FNO blocks, and projection in a single component.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_layers: int = 4,
        n_dim: int = 2,
    ):
        super().__init__()
        
        self.processor = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the neuraloperator FNO equivalent.
    
    This example shows the default configuration used in the neuraloperator
    library for 2D problems like Darcy flow.
    """
    print("=" * 60)
    print("NeuralOperator FNO Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Create model matching neuraloperator's FNO defaults
    # For Darcy flow (2D elliptic PDE):
    #   - n_modes=(16, 16): Keep 16 modes per dimension
    #   - hidden_channels=64: Latent dimension
    #   - n_layers=4: Number of Fourier layers
    model = NeuralOperatorFNO(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        n_layers=4,
        lifting_channel_ratio=2.0,
        projection_channel_ratio=2.0,
        use_channel_mlp=True,
        channel_mlp_expansion=0.5,
    )
    
    # Example input: 2D grid
    batch_size = 2
    spatial_size = 64
    x = torch.randn(batch_size, 1, spatial_size, spatial_size)
    
    # Forward pass
    y = model(x)
    
    print(f"\nModel Configuration:")
    print(f"  Fourier modes: {model.n_modes}")
    print(f"  Hidden channels: {model.hidden_channels}")
    print(f"  Number of layers: {model.n_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    # Also demonstrate SimpleFNO
    print("\n" + "-" * 60)
    print("Alternative: SimpleFNO using framework's FNOProcessor")
    print("-" * 60)
    
    simple_model = SimpleFNO(
        in_channels=1,
        out_channels=1,
        width=64,
        modes=[16, 16],
        n_layers=4,
        n_dim=2,
    )
    
    y_simple = simple_model(x)
    print(f"SimpleFNO output shape: {y_simple.shape}")
    print(f"SimpleFNO parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    return model, x, y


if __name__ == "__main__":
    model, x, y = example_usage()
