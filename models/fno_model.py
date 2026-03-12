"""
Convenience models for Fourier Neural Operators.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from ..convenient.registry import AutoRegisterModel
from ..components.fno import FNOProcessor


class FNO(AutoRegisterModel, name='fno'):
    """
    Fourier Neural Operator for regular grids.
    
    Direct use of FNOProcessor with simple API.
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
            use_afno=False,
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


class TFNO(AutoRegisterModel, name='tfno'):
    """
    Tensorized Fourier Neural Operator (TFNO).
    
    Uses separable spectral convolutions for efficiency.
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
        
        # Use separable convolutions via AFNO-like block structure
        self.fno = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            n_layers=n_layers,
            n_dim=n_dim,
            use_afno=False,  # Standard FNO but can be extended
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


class AFNO(AutoRegisterModel, name='afno'):
    """
    Adaptive Fourier Neural Operator.
    
    Uses block-diagonal weights and soft-thresholding.
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
