"""
Convenience models for Fourier Neural Operators.
"""

from typing import List
import torch
import torch.nn as nn
from ..core.registry import MODEL_REGISTRY
from ..components.spectral import FNOProcessor


@MODEL_REGISTRY.register('fno', aliases=['fourier_no', 'fno2d'])
class FNO(nn.Module):
    """
    Fourier Neural Operator for regular grids.
    
    Direct use of FNOProcessor with simple API. Learns solution operators
    for PDEs in the Fourier space.
    
    Args:
        in_channels: Number of input channels (e.g., initial condition + parameters)
        out_channels: Number of output channels (e.g., solution at next timestep)
        width: Width of the FNO (hidden dimension in Fourier space)
        modes: Number of Fourier modes per dimension (e.g., [16, 16] for 2D)
        n_layers: Number of spectral convolution layers
        n_dim: Spatial dimension (1, 2, or 3)
        
    Example:
        >>> model = FNO(in_channels=1, out_channels=1, width=64, modes=[16, 16])
        >>> x = torch.randn(1, 1, 64, 64)  # [B, C, H, W]
        >>> y = model(x)  # [B, C, H, W]
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
        
        self.fno = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
        )
        
        self.n_dim = n_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, *spatial_dims]
            
        Returns:
            [B, out_channels, *spatial_dims]
        """
        return self.fno(x)


@MODEL_REGISTRY.register('tfno', aliases=['tensorized_fno'])
class TFNO(nn.Module):
    """
    Tensorized Fourier Neural Operator (TFNO).

    Uses separable (factorized) spectral convolutions for improved memory
    efficiency. Instead of storing a full weight tensor of shape
    [C, C, *modes], stores one tensor per dimension, reducing memory from
    O(C^2 * prod(modes)) to O(n_dim * C^2 * max(modes)).

    This is particularly beneficial for high-dimensional problems (2D, 3D)
    where the full spectral weight tensor becomes prohibitively large.

    Reference: Li et al. "Fourier Neural Operator for Parametric Partial
    Differential Equations" (2021) - see Section 3.3 on tensor decomposition.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Width of the TFNO (hidden dimension)
        modes: Number of Fourier modes per dimension
        n_layers: Number of spectral convolution layers
        n_dim: Spatial dimension (1, 2, or 3)
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

        # Use separable convolutions for factorized weights
        self.fno = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
            separable=True,  # Key difference from FNO: factorized weights
        )

        self.n_dim = n_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, *spatial_dims]
            
        Returns:
            [B, out_channels, *spatial_dims]
        """
        return self.fno(x)


@MODEL_REGISTRY.register('afno', aliases=['adaptive_fno'])
class AFNO(nn.Module):
    """
    Adaptive Fourier Neural Operator.
    
    Uses block-diagonal weights and soft-thresholding for improved
    performance on high-resolution inputs.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Width of the AFNO (hidden dimension)
        modes: Number of Fourier modes per dimension
        n_layers: Number of spectral convolution layers
        n_dim: Spatial dimension (1, 2, or 3)
        num_blocks: Number of blocks for block-diagonal weight matrix
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_layers: int = 4,
        n_dim: int = 2,
        num_blocks: int = 8,
    ):
        super().__init__()
        
        self.fno = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
            use_afno=True,
            num_blocks=num_blocks,
        )
        
        self.n_dim = n_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, *spatial_dims]
            
        Returns:
            [B, out_channels, *spatial_dims]
        """
        return self.fno(x)
