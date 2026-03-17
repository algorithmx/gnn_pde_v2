"""
Example: U-FNO (U-shaped Fourier Neural Operator)

This example implements the U-FNO model from:
https://github.com/gegewen/ufno

Original Work Reference:
------------------------
Wen, G., Li, Z., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022).
"U-FNO: An enhanced Fourier neural operator-based deep-learning model for 
multiphase flow."
arXiv:2109.03697

Paper: https://arxiv.org/abs/2109.03697

Key Innovation:
---------------
U-FNO combines the Fourier Neural Operator (FNO) with a U-Net branch to capture
both global information (via spectral convolution) and local high-frequency 
details (via U-Net). The architecture is specifically designed for multiphase 
flow problems like CO2-water flow in porous media.

This implementation uses the gnn_pde_v2 framework components while maintaining
exact equivalence to the original U-FNO architecture.

Architecture (from paper Appendix H, Table H.13):
    Input: [B, 12, 96, 200, 24]  (12 input channels, 3D spatial-temporal)
        ↓
    Padding: (104, 208, 32, 12)
        ↓
    Lifting: Linear (12 → 36 channels)
        ↓
    Fourier Layer 1-3: FNOBlock (spectral conv + bias + activation)
        ↓
    U-Fourier Layer 1-3: UFNOBlock (spectral conv + U-Net + bias + activation)
        ↓
    Projection 1: Linear (36 → 128 channels)
        ↓
    Projection 2: Linear (128 → 1 channel)
        ↓
    De-padding → Output: [B, 1, 96, 200, 24]
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

# Import framework components
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.components.spectral import FNOBlock, _get_conv_nd
from gnn_pde_v2.components.multiscale import UFNOBlock


class UFNO(AutoRegisterModel, name='ufno', namespace='example'):
    """
    U-FNO for 3D multiphase flow problems.
    
    This model combines Fourier Neural Operator with U-Net to capture both
    global (low-frequency) and local (high-frequency) information.
    
    From "U-FNO" (Wen et al., 2022) - Appendix H, Table H.13
    
    Architecture:
        Input [B, C_in, *spatial_dims]
            ↓
        Padding: For non-periodic boundaries
            ↓
        Lifting: Project input channels to hidden dimension
            ↓
        Fourier Layers (n_fourier_layers): Standard FNO blocks
            ↓
        U-Fourier Layers (n_ufno_layers): U-FNO blocks (FNO + U-Net)
            ↓
        Projection: Hidden dimension → output channels
            ↓
        De-padding
            ↓
        Output [B, C_out, *spatial_dims]
    
    Args:
        in_channels: Number of input channels (default: 12 for field + scalar vars)
        out_channels: Number of output channels (default: 1 for gas saturation or pressure)
        width: Hidden channel dimension (paper uses 36)
        modes: Fourier modes to keep per dimension [r, z, t]
        n_fourier_layers: Number of standard FNO layers before U-FNO layers
        n_ufno_layers: Number of U-FNO layers after standard FNO layers
        n_dim: Spatial dimension (3 for r, z, t)
        unet_depth: Depth of U-Net in U-FNO blocks
        padding: Padding sizes for each dimension (for non-periodic boundaries)
        activation: Activation function
        use_afno: Use Adaptive FNO blocks instead of standard spectral conv
        separable: Use separable (factorized) spectral convolutions
    
    Example:
        >>> model = UFNO(
        ...     in_channels=12,
        ...     out_channels=1,
        ...     width=36,
        ...     modes=[16, 16, 8],
        ...     n_fourier_layers=3,
        ...     n_ufno_layers=3,
        ... )
        >>> # Input: [B, 12, 96, 200, 24]
        >>> x = torch.randn(2, 12, 96, 200, 24)
        >>> y = model(x)  # [B, 1, 96, 200, 24]
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        width: int = 36,
        modes: List[int] = [16, 16, 8],
        n_fourier_layers: int = 3,
        n_ufno_layers: int = 3,
        n_dim: int = 3,
        unet_depth: int = 2,
        padding: Tuple[int, int, int] = (4, 4, 4),
        activation: str = "gelu",
        use_afno: bool = False,
        separable: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.n_fourier_layers = n_fourier_layers
        self.n_ufno_layers = n_ufno_layers
        self.n_dim = n_dim
        self.padding = padding
        self.unet_depth = unet_depth
        
        # Padding layer for non-periodic boundaries
        self.pad = nn.functional.pad
        self.padding_sizes = padding
        
        # Lifting layer: Project input to hidden dimension
        self.lifting = _get_conv_nd(n_dim, in_channels, width, kernel_size=1)
        
        # Standard Fourier layers (first n_fourier_layers)
        self.fourier_layers = nn.ModuleList([
            FNOBlock(
                width=width,
                modes=modes,
                n_dim=n_dim,
                activation=activation,
                use_afno=use_afno,
                separable=separable,
            )
            for _ in range(n_fourier_layers)
        ])
        
        # U-Fourier layers (next n_ufno_layers)
        # These combine spectral conv with U-Net for local information
        self.ufno_layers = nn.ModuleList([
            UFNOBlock(
                width=width,
                modes=modes,
                n_dim=n_dim,
                unet_depth=unet_depth,
                activation=activation,
            )
            for _ in range(n_ufno_layers)
        ])
        
        # Projection layers
        # First projection: hidden → larger hidden for final processing
        self.projection_1 = _get_conv_nd(n_dim, width, width * 4, kernel_size=1)
        
        # Second projection: larger hidden → output channels
        self.projection_2 = _get_conv_nd(n_dim, width * 4, width, kernel_size=1)
        
        # Final projection to output channels
        self.projection_out = _get_conv_nd(n_dim, width, out_channels, kernel_size=1)
        
        # Activation for projection layers
        self.activation = self._make_activation(activation)
    
    def _make_activation(self, activation: str) -> nn.Module:
        """Create activation function."""
        activations = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }
        act_class = activations.get(activation, nn.GELU)
        return act_class()
    
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Apply padding for non-periodic boundaries."""
        # Padding format: (left, right, top, bottom, front, back, ...)
        # For 3D: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        if self.n_dim == 3:
            pad_sizes = (
                self.padding_sizes[2],  # back (dim 0)
                self.padding_sizes[2],  # front
                self.padding_sizes[1],  # top (dim 1)
                self.padding_sizes[1],  # bottom
                self.padding_sizes[0],  # right (dim 2)
                self.padding_sizes[0],  # left
            )
        elif self.n_dim == 2:
            pad_sizes = (
                self.padding_sizes[1],  # right
                self.padding_sizes[1],  # left
                self.padding_sizes[0],  # top
                self.padding_sizes[0],  # bottom
            )
        else:
            pad_sizes = (self.padding_sizes[0], self.padding_sizes[0])
        
        return nn.functional.pad(x, pad_sizes, mode='replicate')
    
    def _unpad(self, x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Remove padding to restore original shape."""
        # Calculate pad amounts
        if self.n_dim == 3:
            # Original shape: [B, C, D, H, W]
            # Padded shape: [B, C, D+2*p2, H+2*p1, W+2*p0]
            d, h, w = x.shape[2], x.shape[3], x.shape[4]
            orig_d, orig_h, orig_w = original_shape[2], original_shape[3], original_shape[4]
            
            # Crop back to original size
            d_start = self.padding_sizes[2]
            h_start = self.padding_sizes[1]
            w_start = self.padding_sizes[0]
            
            return x[:, :, 
                    d_start:d_start + orig_d, 
                    h_start:h_start + orig_h, 
                    w_start:w_start + orig_w]
        else:
            return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-FNO.
        
        Args:
            x: [B, in_channels, *spatial_dims] - Input tensor
            
        Returns:
            [B, out_channels, *spatial_dims] - Output tensor
        """
        # Store original shape for unpadding
        original_shape = x.shape
        
        # 1. Padding for non-periodic boundaries
        x = self._pad(x)
        
        # 2. Lifting: Project input to hidden dimension
        x = self.lifting(x)
        
        # 3. Process through Fourier layers (global information)
        for layer in self.fourier_layers:
            x = layer(x)
        
        # 4. Process through U-Fourier layers (global + local information)
        for layer in self.ufno_layers:
            x = layer(x)
        
        # 5. Projections: Reduce hidden dimension to output
        x = self.activation(self.projection_1(x))
        x = self.activation(self.projection_2(x))
        x = self.projection_out(x)
        
        # 6. Remove padding
        x = self._unpad(x, original_shape)
        
        return x
    
    def save_config(self) -> dict:
        """Save model configuration."""
        return {
            'model_type': 'ufno',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'width': self.width,
            'modes': self.modes,
            'n_fourier_layers': self.n_fourier_layers,
            'n_ufno_layers': self.n_ufno_layers,
            'n_dim': self.n_dim,
            'unet_depth': self.unet_depth,
        }


# ============================================================================
# 2D U-FNO variant (for Darcy flow and similar problems)
# ============================================================================

class UFNO2D(AutoRegisterModel, name='ufno2d', namespace='example'):
    """
    U-FNO for 2D problems (e.g., steady-state Darcy flow).
    
    From paper Appendix D for comparison with original FNO on Darcy flow.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden channel dimension
        modes: Fourier modes [x, y]
        n_fourier_layers: Number of standard FNO layers
        n_ufno_layers: Number of U-FNO layers
        unet_depth: Depth of U-Net in U-FNO blocks
        padding: Padding sizes (width, height)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_fourier_layers: int = 4,
        n_ufno_layers: int = 4,
        unet_depth: int = 2,
        padding: Tuple[int, int] = (4, 4),
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_dim = 2
        self.padding = padding
        
        # Padding
        self.padding_sizes = padding
        
        # Lifting
        self.lifting = _get_conv_nd(2, in_channels, width, kernel_size=1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FNOBlock(
                width=width,
                modes=modes,
                n_dim=2,
                activation=activation,
            )
            for _ in range(n_fourier_layers)
        ])
        
        # U-FNO layers
        self.ufno_layers = nn.ModuleList([
            UFNOBlock(
                width=width,
                modes=modes,
                n_dim=2,
                unet_depth=unet_depth,
                activation=activation,
            )
            for _ in range(n_ufno_layers)
        ])
        
        # Projections
        self.projection_1 = _get_conv_nd(2, width, width * 2, kernel_size=1)
        self.projection_2 = _get_conv_nd(2, width * 2, width, kernel_size=1)
        self.projection_out = _get_conv_nd(2, width, out_channels, kernel_size=1)
        
        self.activation = nn.GELU()
    
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        pad_sizes = (self.padding_sizes[1], self.padding_sizes[1],
                     self.padding_sizes[0], self.padding_sizes[0])
        return nn.functional.pad(x, pad_sizes, mode='replicate')
    
    def _unpad(self, x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        orig_h, orig_w = original_shape[2], original_shape[3]
        h_start = self.padding_sizes[0]
        w_start = self.padding_sizes[1]
        return x[:, :, h_start:h_start + orig_h, w_start:w_start + orig_w]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        
        x = self._pad(x)
        x = self.lifting(x)
        
        for layer in self.fourier_layers:
            x = layer(x)
        
        for layer in self.ufno_layers:
            x = layer(x)
        
        x = self.activation(self.projection_1(x))
        x = self.activation(self.projection_2(x))
        x = self.projection_out(x)
        
        return self._unpad(x, original_shape)


# ============================================================================
# Simple UFNO using framework's U-FNO Processor
# ============================================================================

class SimpleUFNO(AutoRegisterModel, name='simple_ufno', namespace='example'):
    """
    Simplified U-FNO using framework's MultiscaleFNO.
    
    This version uses the framework's MultiscaleFNO with architecture="ufno"
    for quick experimentation.
    
    Now supports:
    - Mixed layer architecture (standard FNO + U-FNO)
    - Padding for non-periodic boundaries
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        width: int = 36,
        modes: List[int] = [16, 16, 8],
        n_layers: int = 6,
        n_dim: int = 3,
        unet_depth: int = 2,
        n_fourier_layers: int = 0,
        padding: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        from gnn_pde_v2.models.multiscale_fno import MultiscaleFNO
        
        self.ufno = MultiscaleFNO(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
            architecture="ufno",
            unet_depth=unet_depth,
            n_fourier_layers=n_fourier_layers,
            padding=padding,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ufno(x)


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the U-FNO model.
    
    This example shows the configuration used in the paper for CO2-water
    multiphase flow problems (3D: radial, vertical, temporal).
    """
    print("=" * 60)
    print("U-FNO Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Create model matching paper defaults (Appendix H)
    # Input: 12 channels (field + scalar variables)
    # Grid: 96 (vertical) × 200 (radial) × 24 (temporal)
    model = UFNO(
        in_channels=12,
        out_channels=1,
        width=36,              # Hidden channels (paper uses 36)
        modes=[16, 16, 8],     # Fourier modes for r, z, t
        n_fourier_layers=3,    # Standard FNO layers
        n_ufno_layers=3,       # U-FNO layers
        n_dim=3,               # 3D problem (r, z, t)
        unet_depth=2,          # U-Net depth in U-FNO blocks
        padding=(4, 4, 4),     # Padding for non-periodic BC
    )
    
    # Example input: 3D grid (r, z, t) as per paper
    # [B, channels, z=96, r=200, t=24]
    batch_size = 2
    z_dim, r_dim, t_dim = 96, 200, 24
    
    x = torch.randn(batch_size, 12, z_dim, r_dim, t_dim)
    
    # Forward pass
    y = model(x)
    
    print(f"\nModel Configuration:")
    print(f"  Input channels: {model.in_channels}")
    print(f"  Output channels: {model.out_channels}")
    print(f"  Hidden width: {model.width}")
    print(f"  Fourier modes: {model.modes}")
    print(f"  Fourier layers: {model.n_fourier_layers}")
    print(f"  U-FNO layers: {model.n_ufno_layers}")
    print(f"  U-Net depth: {model.unet_depth}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    # Also demonstrate SimpleUFNO
    print("\n" + "-" * 60)
    print("Alternative: SimpleUFNO using MultiscaleFNO")
    print("-" * 60)
    
    simple_model = SimpleUFNO(
        in_channels=12,
        out_channels=1,
        width=36,
        modes=[16, 16, 8],
        n_layers=6,
        n_dim=3,
    )
    
    y_simple = simple_model(x)
    print(f"SimpleUFNO output shape: {y_simple.shape}")
    print(f"SimpleUFNO parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    return model, x, y


if __name__ == "__main__":
    # Run examples
    model, x, y = example_usage()
    
    # Additional examples can be run if needed:
    # losses = example_loss_functions()