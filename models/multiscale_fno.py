"""
Multiscale FNO models.

Includes:
- MultiscaleFNO: FNO with multi-band or U-FNO blocks
- HierarchicalFNO: FNO with hierarchical processing
"""

from typing import List, Optional
import torch
import torch.nn as nn

from ..components.spectral import _get_conv_nd
from ..components.multiscale import (
    MultiResolutionFNOBlock,
    UFNOBlock,
    HierarchicalFNOBlock,
)


class MultiscaleFNO(nn.Module):
    """Multiscale FNO with enhanced spectral blocks.
    
    Supports multiple architectures:
    - "standard": Regular FNO blocks
    - "multiband": MultiResolutionFNOBlock with parallel frequency bands
    - "ufno": U-FNO blocks with local U-Net processing
    - "hierarchical": HierarchicalFNOBlock with coarse-to-fine processing
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden channel dimension
        modes: Fourier modes to keep
        n_layers: Number of FNO layers
        n_dim: Spatial dimension (1, 2, or 3)
        architecture: "standard", "multiband", "ufno", or "hierarchical"
        unet_depth: Depth of U-Net in U-FNO (if used)
        activation: Activation function
    
    Example:
        >>> model = MultiscaleFNO(
        ...     in_channels=1,
        ...     out_channels=1,
        ...     width=64,
        ...     modes=[16, 16],
        ...     n_layers=4,
        ...     architecture="ufno",
        ... )
        >>> output = model(input)  # [B, 1, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_layers: int = 4,
        n_dim: int = 2,
        architecture: str = "ufno",
        unet_depth: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_dim = n_dim
        self.architecture = architecture
        
        # Lifting layer
        self.lifting = _get_conv_nd(n_dim, in_channels, width, kernel_size=1)
        
        # FNO blocks based on architecture
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if architecture == "multiband":
                block = MultiResolutionFNOBlock(
                    width=width,
                    modes_list=[[m//2 for m in modes], modes, [m*2 for m in modes]],
                    n_dim=n_dim,
                    activation=activation,
                )
            elif architecture == "ufno":
                block = UFNOBlock(
                    width=width,
                    modes=modes,
                    n_dim=n_dim,
                    unet_depth=unet_depth,
                    activation=activation,
                )
            elif architecture == "hierarchical":
                block = HierarchicalFNOBlock(
                    width=width,
                    modes=modes,
                    n_levels=3,
                    n_dim=n_dim,
                    activation=activation,
                )
            else:  # standard
                from ..components.spectral import FNOBlock
                block = FNOBlock(
                    width=width,
                    modes=modes,
                    n_dim=n_dim,
                    activation=activation,
                )
            self.blocks.append(block)
        
        # Projection layer
        self.projection = _get_conv_nd(n_dim, width, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [B, in_channels, *spatial_dims]
        
        Returns:
            Output [B, out_channels, *spatial_dims]
        """
        # Lift
        x = self.lifting(x)
        
        # Process
        for block in self.blocks:
            x = block(x)
        
        # Project
        x = self.projection(x)
        
        return x
    
    def forward_super_resolution(
        self,
        x: torch.Tensor,
        target_resolution: List[int],
    ) -> torch.Tensor:
        """Forward with super-resolution output.
        
        Args:
            x: Input at training resolution
            target_resolution: Target output resolution
        
        Returns:
            Output at target resolution
        """
        # Process at input resolution
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        
        # Interpolate to target resolution before projection
        x = torch.nn.functional.interpolate(
            x,
            size=target_resolution,
            mode='bilinear' if self.n_dim == 2 else 'trilinear',
            align_corners=False,
        )
        
        # Project at target resolution
        x = self.projection(x)
        
        return x


__all__ = ["MultiscaleFNO"]
