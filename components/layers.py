"""
Minimal residual connection for GNN-PDE v2.

Single implementation - no variants, no factory functions.
For more complex needs, compose directly with PyTorch.
"""

from typing import Optional
import torch.nn as nn


class Residual(nn.Module):
    """
    Simple residual connection wrapper.
    
    output = x + module(norm(x)) if norm else x + module(x)
    
    Args:
        module: The module to wrap
        norm: Optional normalization layer (e.g., nn.LayerNorm)
        
    Example:
        >>> block = Residual(nn.Sequential(
        ...     nn.Linear(128, 128),
        ...     nn.GELU(),
        ...     nn.Linear(128, 128),
        ... ))
        >>> output = block(input)  # input + module(input)
        
        >>> # With pre-normalization (Transformer-style)
        >>> block = Residual(
        ...     module=nn.MultiheadAttention(128, 8),
        ...     norm=nn.LayerNorm(128),
        ... )
    """
    
    def __init__(self, module: nn.Module, norm: Optional[nn.Module] = None):
        super().__init__()
        self.module = module
        self.norm = norm
    
    def forward(self, x):
        """Apply module with residual connection."""
        residual = x
        if self.norm is not None:
            x = self.norm(x)
        out = self.module(x)
        return residual + out
