"""
Residual connection utilities for GNN-PDE v2.

Provides standardized residual patterns used across different architectures.
"""

from typing import Optional, Callable, Literal
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Wrapper that adds residual connections to any module.
    
    Supports different residual types:
    - 'add': Simple addition (x + f(x))
    - 'scaled': Scaled addition with learnable or fixed scale
    - 'none': No residual (pass-through for compatibility)
    
    Args:
        module: The module to wrap
        residual_type: Type of residual connection
        scale: Optional fixed scale for residual branch
        learnable_scale: Whether to learn the residual scale
        
    Examples:
        >>> # Simple residual
        >>> block = ResidualBlock(nn.Linear(64, 64), residual_type='add')
        >>> 
        >>> # Learnable gated residual
        >>> block = ResidualBlock(
        ...     nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)),
        ...     residual_type='scaled',
        ...     learnable_scale=True
        ... )
    """
    
    def __init__(
        self,
        module: nn.Module,
        residual_type: Literal['add', 'scaled', 'none'] = 'add',
        scale: Optional[float] = None,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.module = module
        self.residual_type = residual_type
        
        if residual_type == 'scaled':
            if learnable_scale:
                self.scale = nn.Parameter(torch.tensor(1.0 if scale is None else scale))
            else:
                self.register_buffer('scale', torch.tensor(1.0 if scale is None else scale))
        elif residual_type == 'add':
            self.scale = 1.0
        else:
            self.scale = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        out = self.module(x)
        
        if self.residual_type == 'none':
            return out
        
        # Check dimension compatibility
        if x.shape != out.shape:
            raise ValueError(
                f"Residual shapes don't match: input {x.shape}, output {out.shape}. "
                "Consider using a projection layer or disable residual."
            )
        
        return x + self.scale * out
    
    def __repr__(self) -> str:
        return f"ResidualBlock(module={self.module}, type={self.residual_type})"


class GatedResidual(nn.Module):
    """
    Gated residual connection with learnable gate.
    
    Similar to Highway Networks or Gated Residual Networks.
    The gate controls how much of the residual branch to add.
    
    Formula: output = (1 - gate) * x + gate * f(x)
    
    Args:
        module: The module to wrap
        gate_activation: Activation for gate ('sigmoid', 'tanh', etc.)
        gate_bias: Initial bias for gate (positive = more residual, negative = less)
        
    Examples:
        >>> block = GatedResidual(
        ...     nn.Sequential(nn.Linear(64, 64), nn.ReLU()),
        ...     gate_activation='sigmoid',
        ...     gate_bias=2.0  # Start with more residual
        ... )
    """
    
    def __init__(
        self,
        module: nn.Module,
        gate_activation: Literal['sigmoid', 'tanh', 'hard_sigmoid'] = 'sigmoid',
        gate_bias: float = 0.0,
    ):
        super().__init__()
        self.module = module
        
        # Determine input dimension
        self.in_features = None
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.in_features = m.in_features
                break
        
        if self.in_features is None:
            raise ValueError("Could not determine input dimension from module")
        
        # Gate projection
        self.gate_proj = nn.Linear(self.in_features, self.in_features)
        nn.init.constant_(self.gate_proj.bias, gate_bias)
        
        # Gate activation
        if gate_activation == 'sigmoid':
            self.gate_act = nn.Sigmoid()
        elif gate_activation == 'tanh':
            self.gate_act = nn.Tanh()
        elif gate_activation == 'hard_sigmoid':
            self.gate_act = nn.Hardsigmoid()
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with gated residual."""
        out = self.module(x)
        
        if x.shape != out.shape:
            raise ValueError(
                f"Residual shapes don't match: input {x.shape}, output {out.shape}"
            )
        
        gate = self.gate_act(self.gate_proj(x))
        return (1 - gate) * x + gate * out
    
    def __repr__(self) -> str:
        return f"GatedResidual(module={self.module})"


class PreNormResidual(nn.Module):
    """
    Pre-normalization residual block (Transformer-style).
    
    Applies LayerNorm before the module, then adds residual:
        output = x + module(norm(x))
    
    This is the modern Transformer architecture (Pre-LN) which is more
    stable for deep networks compared to Post-LN.
    
    Args:
        module: The module to wrap
        dim: Dimension for LayerNorm
        eps: Epsilon for LayerNorm
        
    Examples:
        >>> # Standard Transformer block
        >>> block = PreNormResidual(
        ...     nn.Sequential(
        ...         nn.MultiheadAttention(embed_dim=256, num_heads=8),
        ...         nn.Linear(256, 256)
        ...     ),
        ...     dim=256
        ... )
    """
    
    def __init__(
        self,
        module: nn.Module,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-normalization residual."""
        out = self.module(self.norm(x))
        
        if x.shape != out.shape:
            raise ValueError(
                f"Residual shapes don't match: input {x.shape}, output {out.shape}"
            )
        
        return x + out
    
    def __repr__(self) -> str:
        return f"PreNormResidual(dim={self.norm.normalized_shape[0]})"


class ResidualSequence(nn.Module):
    """
    Sequence of residual blocks with consistent interface.
    
    Useful for building deep networks with the same residual pattern.
    
    Args:
        blocks: List of modules to wrap
        residual_type: Type of residual for all blocks
        **residual_kwargs: Additional arguments for residual blocks
        
    Examples:
        >>> processor = ResidualSequence([
        ...     GraphNetBlock(node_dim=128, edge_dim=128),
        ...     GraphNetBlock(node_dim=128, edge_dim=128),
        ...     GraphNetBlock(node_dim=128, edge_dim=128),
        ... ], residual_type='add')
    """
    
    def __init__(
        self,
        blocks: list,
        residual_type: Literal['add', 'scaled', 'none'] = 'add',
        **residual_kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(block, residual_type=residual_type, **residual_kwargs)
            for block in blocks
        ])
    
    def forward(self, x):
        """Forward through all residual blocks."""
        for block in self.blocks:
            x = block(x)
        return x
    
    def __len__(self) -> int:
        return len(self.blocks)
    
    def __getitem__(self, idx):
        return self.blocks[idx]


class SkipConnection(nn.Module):
    """
    Flexible skip connection with optional projection.
    
    Useful when input and output dimensions differ.
    
    Args:
        module: The module to wrap
        projection: Optional projection layer for matching dimensions
        aggregation: How to combine ('add', 'concat', 'none')
        
    Examples:
        >>> # Dimension change with projection
        >>> block = SkipConnection(
        ...     nn.Sequential(nn.Linear(64, 128), nn.ReLU()),
        ...     projection=nn.Linear(64, 128),
        ...     aggregation='add'
        ... )
    """
    
    def __init__(
        self,
        module: nn.Module,
        projection: Optional[nn.Module] = None,
        aggregation: Literal['add', 'concat', 'none'] = 'add',
    ):
        super().__init__()
        self.module = module
        self.projection = projection
        self.aggregation = aggregation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with skip connection."""
        out = self.module(x)
        
        if self.aggregation == 'none':
            return out
        
        # Project input if needed
        residual = x if self.projection is None else self.projection(x)
        
        if self.aggregation == 'add':
            if residual.shape != out.shape:
                raise ValueError(
                    f"Cannot add: residual {residual.shape}, output {out.shape}"
                )
            return residual + out
        elif self.aggregation == 'concat':
            return torch.cat([residual, out], dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def __repr__(self) -> str:
        return f"SkipConnection(aggregation={self.aggregation})"


def make_residual(
    module: nn.Module,
    residual_type: Literal['add', 'scaled', 'gated', 'prenorm', 'none'] = 'add',
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate residual wrapper.
    
    Args:
        module: Module to wrap
        residual_type: Type of residual wrapper
        **kwargs: Additional arguments for specific residual types
        
    Returns:
        Wrapped module with residual connections
        
    Examples:
        >>> block = make_residual(nn.Linear(64, 64), 'add')
        >>> block = make_residual(nn.Linear(64, 64), 'gated', gate_bias=2.0)
        >>> block = make_residual(attention_module, 'prenorm', dim=256)
    """
    if residual_type == 'add':
        return ResidualBlock(module, residual_type='add', **kwargs)
    elif residual_type == 'scaled':
        return ResidualBlock(module, residual_type='scaled', **kwargs)
    elif residual_type == 'gated':
        return GatedResidual(module, **kwargs)
    elif residual_type == 'prenorm':
        return PreNormResidual(module, **kwargs)
    elif residual_type == 'none':
        return module
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")


__all__ = [
    'ResidualBlock',
    'GatedResidual',
    'PreNormResidual',
    'ResidualSequence',
    'SkipConnection',
    'make_residual',
]
