"""
Multiscale FNO models.

Includes:
- MultiscaleFNO: FNO with multi-band or U-FNO blocks
- HierarchicalFNO: FNO with hierarchical processing
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from ..components.spectral import _get_conv_nd, FNOBlock
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
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        n_fourier_layers: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_dim = n_dim
        self.architecture = architecture
        self.padding = padding
        self.padding_sizes = padding
        self.n_fourier_layers = n_fourier_layers
        
        # Lifting layer
        self.lifting = _get_conv_nd(n_dim, in_channels, width, kernel_size=1)
        
        # FNO blocks based on architecture
        # If n_fourier_layers > 0, first n_fourier_layers use standard FNO, rest use specified architecture
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            # Determine which block type to use
            use_standard_fno = (architecture == "ufno" and 
                                n_fourier_layers > 0 and 
                                i < n_fourier_layers)
            
            if use_standard_fno:
                block = FNOBlock(
                    width=width,
                    modes=modes,
                    n_dim=n_dim,
                    activation=activation,
                )
            elif architecture == "multiband":
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
                block = FNOBlock(
                    width=width,
                    modes=modes,
                    n_dim=n_dim,
                    activation=activation,
                )
            self.blocks.append(block)
        
        # Projection layer
        self.projection = _get_conv_nd(n_dim, width, out_channels, kernel_size=1)
    
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Apply padding for non-periodic boundaries."""
        if self.padding_sizes is None:
            return x
        
        # Convert single int or short tuple to full pad format
        if isinstance(self.padding_sizes, int):
            # Single int: same padding on all sides for all dims
            pad_sizes = tuple([self.padding_sizes] * (self.n_dim * 2))
        elif len(self.padding_sizes) == self.n_dim:
            # Tuple of length n_dim: (d0, d1, d2) -> expand to (d0, d0, d1, d1, d2, d2)
            pad_sizes = tuple(v for p in self.padding_sizes for v in (p, p))
        else:
            # Already in full format
            pad_sizes = self.padding_sizes
        
        # Pad format for nn.functional.pad: (dimN_left, dimN_right, ..., dim0_left, dim0_right)
        # For 3D: (d2_left, d2_right, d1_left, d1_right, d0_left, d0_right)
        # Reverse to match expected order
        pad_sizes = pad_sizes[::-1]
        
        return nn.functional.pad(x, pad_sizes, mode='replicate')
    
    def _unpad(self, x: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Remove padding to restore original shape."""
        if self.padding_sizes is None:
            return x
        
        # Extract pad values - support same formats as _pad
        if isinstance(self.padding_sizes, int):
            pad_values = [self.padding_sizes] * self.n_dim
        elif len(self.padding_sizes) == self.n_dim:
            pad_values = list(self.padding_sizes)
        else:
            pad_values = self.padding_sizes[::2]
        
        # Unpad based on n_dim
        # padding convention: padding[i] corresponds to dimension i in spatial dims
        # For shape [B, C, D, H, W] (3D), padding[0]=D_pad, padding[1]=H_pad, padding[2]=W_pad
        if self.n_dim == 1:
            p = pad_values[0] if pad_values else 0
            return x[..., p:-p] if p > 0 else x
        elif self.n_dim == 2:
            p0 = pad_values[0] if pad_values else 0  # H padding
            p1 = pad_values[1] if len(pad_values) > 1 else 0  # W padding
            return x[..., p0:-p0, p1:-p1] if p0 > 0 or p1 > 0 else x
        else:  # 3D
            p0 = pad_values[0] if pad_values else 0  # D padding
            p1 = pad_values[1] if len(pad_values) > 1 else 0  # H padding
            p2 = pad_values[2] if len(pad_values) > 2 else 0  # W padding
            return x[..., p0:-p0, p1:-p1, p2:-p2] if p0 > 0 or p1 > 0 or p2 > 0 else x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [B, in_channels, *spatial_dims]
        
        Returns:
            Output [B, out_channels, *spatial_dims]
        """
        # Store original shape for unpadding
        original_shape = x.shape
        
        # Pad for non-periodic boundaries
        x = self._pad(x)
        
        # Lift
        x = self.lifting(x)
        
        # Process
        for block in self.blocks:
            x = block(x)
        
        # Project
        x = self.projection(x)
        
        # Unpad
        x = self._unpad(x, original_shape)
        
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
