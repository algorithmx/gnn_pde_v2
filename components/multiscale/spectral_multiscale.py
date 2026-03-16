"""
Multi-resolution spectral components for FNO.

Includes:
- MultiResolutionFNOBlock: Different FNO blocks for different frequency bands
- UFNOBlock: U-FNO combining spectral and local processing
- Super-resolution utilities

Based on:
- "Fourier Neural Operator" (Li et al., ICLR 2021)
- "U-FNO" (Wen et al., 2022)
"""

from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spectral import FNOBlock, SpectralConv, _get_conv_nd


class MultiResolutionFNOBlock(nn.Module):
    """Multi-resolution FNO block with parallel frequency bands.
    
    Processes different frequency bands with different FNO blocks,
    then combines the results.
    
    Args:
        width: Hidden channel dimension
        modes_list: List of mode configurations for each band
        n_dim: Spatial dimension
        band_weights: Optional weights for combining bands
    
    Example:
        >>> block = MultiResolutionFNOBlock(
        ...     width=64,
        ...     modes_list=[[8, 8], [16, 16], [32, 32]],
        ...     n_dim=2,
        ... )
    """
    
    def __init__(
        self,
        width: int,
        modes_list: List[List[int]],
        n_dim: int = 2,
        activation: str = "gelu",
        band_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.width = width
        self.n_dim = n_dim
        self.n_bands = len(modes_list)
        
        # Create FNO block for each frequency band
        self.band_blocks = nn.ModuleList([
            FNOBlock(
                width=width,
                modes=modes,
                n_dim=n_dim,
                activation=activation,
            )
            for modes in modes_list
        ])
        
        # Learnable band combination weights
        if band_weights is None:
            self.band_weights = nn.Parameter(torch.ones(self.n_bands) / self.n_bands)
        else:
            self.register_buffer('band_weights', band_weights)
        
        # Output projection
        self.output_conv = _get_conv_nd(n_dim, width, width, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through multiple frequency bands.
        
        Args:
            x: [B, C, *spatial_dims]
        
        Returns:
            Combined output [B, C, *spatial_dims]
        """
        # Process each band
        band_outputs = []
        for block in self.band_blocks:
            band_outputs.append(block(x))
        
        # Weighted combination
        weights = F.softmax(self.band_weights, dim=0)
        output = sum(w * out for w, out in zip(weights, band_outputs))
        
        # Final projection
        output = self.output_conv(output)
        
        return output


class MiniUNet(nn.Module):
    """Mini U-Net for local processing in U-FNO.
    
    Simple encoder-decoder for capturing local high-frequency details.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        hidden_channels: Hidden channels
        depth: Number of encoder/decoder levels
        n_dim: Spatial dimension
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        depth: int = 2,
        n_dim: int = 2,
    ):
        super().__init__()
        self.n_dim = n_dim
        
        # Encoder
        self.encoder_convs = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(depth):
            out_ch = hidden_channels * (2 ** i)
            self.encoder_convs.append(
                nn.Sequential(
                    _get_conv_nd(n_dim, in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    _get_conv_nd(n_dim, out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                )
            )
            self.encoder_pools.append(
                nn.MaxPool2d(2) if n_dim == 2 else nn.MaxPool3d(2)
            )
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = hidden_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            _get_conv_nd(n_dim, in_ch, bottleneck_ch, kernel_size=3, padding=1),
            nn.GELU(),
            _get_conv_nd(n_dim, bottleneck_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Decoder
        self.decoder_ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            out_ch = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            self.decoder_ups.append(
                nn.Upsample(scale_factor=2, mode='bilinear' if n_dim == 2 else 'trilinear',
                           align_corners=False)
            )
            self.decoder_convs.append(
                nn.Sequential(
                    _get_conv_nd(n_dim, in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    _get_conv_nd(n_dim, out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                )
            )
            in_ch = out_ch
        
        # Output
        self.output_conv = _get_conv_nd(n_dim, hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through mini U-Net.
        
        Args:
            x: [B, C, *spatial_dims]
        
        Returns:
            Output [B, C_out, *spatial_dims]
        """
        # Encoder
        encoder_features = []
        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            x = conv(x)
            encoder_features.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for up, conv in zip(self.decoder_ups, self.decoder_convs):
            x = up(x)
            # Skip connection
            if encoder_features:
                skip = encoder_features.pop()
                # Handle size mismatch
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear' if self.n_dim == 2 else 'trilinear',
                                     align_corners=False)
                x = x + skip
            x = conv(x)
        
        # Output
        x = self.output_conv(x)
        return x


class UFNOBlock(nn.Module):
    """U-FNO block: Spectral conv + U-Net + skip connection.
    
    From "U-FNO: An Enhanced Fourier Neural Operator" (Wen et al., 2022).
    
    Combines:
    - K: Spectral convolution (global information)
    - U: Mini U-Net (local high-frequency details)
    - W: Pointwise linear (bias term)
    
    Args:
        width: Hidden channel dimension
        modes: Fourier modes to keep
        n_dim: Spatial dimension
        unet_depth: Depth of mini U-Net
        activation: Activation function
    
    Example:
        >>> block = UFNOBlock(width=64, modes=[16, 16], n_dim=2)
        >>> output = block(input)  # [B, 64, H, W]
    """
    
    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int = 2,
        unet_depth: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.width = width
        self.n_dim = n_dim
        
        # Spectral branch (global)
        self.spectral_conv = SpectralConv(width, width, modes)
        
        # U-Net branch (local)
        self.unet = MiniUNet(
            in_channels=width,
            out_channels=width,
            hidden_channels=width // 2,
            depth=unet_depth,
            n_dim=n_dim,
        )
        
        # Pointwise bias
        self.bias = _get_conv_nd(n_dim, width, width, kernel_size=1)
        
        # Activation
        self.activation = self._make_activation(activation)
    
    def _make_activation(self, activation: str) -> nn.Module:
        """Create activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        if activation in activations:
            return activations[activation]
        return nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through U-FNO block.
        
        Args:
            x: [B, C, *spatial_dims]
        
        Returns:
            Output [B, C, *spatial_dims]
        """
        # Spectral path (global)
        x1 = self.spectral_conv(x)
        
        # U-Net path (local)
        x2 = self.unet(x)
        
        # Bias path
        x3 = self.bias(x)
        
        # Combine
        return self.activation(x1 + x2 + x3)


class HierarchicalFNOBlock(nn.Module):
    """Hierarchical FNO block with coarse-to-fine processing.
    
    Processes at multiple resolutions, useful for capturing features
    at different scales.
    
    Args:
        width: Channel dimension
        modes: Base modes
        n_levels: Number of resolution levels
        n_dim: Spatial dimension
    """
    
    def __init__(
        self,
        width: int,
        modes: List[int],
        n_levels: int = 3,
        n_dim: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.width = width
        self.n_levels = n_levels
        self.n_dim = n_dim
        
        # FNO blocks at different resolutions
        self.fno_blocks = nn.ModuleList()
        for level in range(n_levels):
            # Reduce modes for coarser levels
            level_modes = [max(4, m // (2 ** level)) for m in modes]
            self.fno_blocks.append(
                FNOBlock(
                    width=width,
                    modes=level_modes,
                    n_dim=n_dim,
                    activation=activation,
                )
            )
        
        # Output combination
        self.output_conv = _get_conv_nd(n_dim, width, width, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through hierarchical levels.
        
        Args:
            x: [B, C, *spatial_dims]
        
        Returns:
            Output [B, C, *spatial_dims]
        """
        original_size = x.shape[2:]
        
        # Process at each level
        level_outputs = []
        for level, fno in enumerate(self.fno_blocks):
            # Downsample for coarser levels
            if level > 0:
                size = tuple(s // (2 ** level) for s in original_size)
                x_level = F.interpolate(x, size=size, mode='bilinear' if self.n_dim == 2 else 'trilinear',
                                       align_corners=False)
            else:
                x_level = x
            
            # Process
            out = fno(x_level)
            
            # Upsample back
            if level > 0:
                out = F.interpolate(out, size=original_size, mode='bilinear' if self.n_dim == 2 else 'trilinear',
                                   align_corners=False)
            
            level_outputs.append(out)
        
        # Combine all levels
        output = sum(level_outputs) / len(level_outputs)
        output = self.output_conv(output)
        
        return output


def super_resolution_interpolation(
    x: torch.Tensor,
    target_size: List[int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Interpolate tensor to target resolution.
    
    Used for zero-shot super-resolution in FNO.
    
    Args:
        x: Input tensor [B, C, *spatial_dims]
        target_size: Target spatial size
        mode: Interpolation mode
    
    Returns:
        Interpolated tensor
    """
    return F.interpolate(x, size=target_size, mode=mode, align_corners=False)


def super_resolution_fourier(
    x_ft: torch.Tensor,
    target_modes: List[int],
) -> torch.Tensor:
    """Super-resolution by padding Fourier modes.
    
    For FNO: can train at low resolution and evaluate at high resolution
    by padding with zeros in Fourier space.
    
    Args:
        x_ft: Input in Fourier space [B, C, *ft_dims]
        target_modes: Target number of modes
    
    Returns:
        Padded Fourier tensor
    """
    # Pad with zeros to target size
    # This is a simplified version - full implementation would handle
    # complex padding properly
    pad_dims = []
    for current, target in zip(x_ft.shape[2:], target_modes):
        pad = target - current
        if pad > 0:
            # Pad on the right (high frequencies)
            pad_dims.extend([0, pad])
        else:
            pad_dims.extend([0, 0])
    
    if any(p > 0 for p in pad_dims):
        x_ft_padded = F.pad(x_ft, pad_dims)
        return x_ft_padded
    return x_ft


__all__ = [
    "MultiResolutionFNOBlock",
    "MiniUNet",
    "UFNOBlock",
    "HierarchicalFNOBlock",
    "super_resolution_interpolation",
    "super_resolution_fourier",
]
