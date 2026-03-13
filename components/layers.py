"""
Residual connection utilities for GNN-PDE v2.

Provides standardized residual patterns used across different architectures.
The primary interface is `Residual` for simple cases, `GatedResidual` for
learnable gating, and `make_residual` factory for string-based selection.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn


class Residual(nn.Module):
    """
    Simple residual connection wrapper.

    Formula: output = x + scale * module(norm(x) if norm else x)

    This is the primary interface for residual connections.

    Args:
        module: The module to wrap
        norm: Optional normalization layer (e.g., nn.LayerNorm)
        scale: Optional scale factor (float or learnable)
        learnable_scale: Whether scale is learnable (only if scale is float)

    Example:
        >>> # Simple residual
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

        >>> # With learnable scale
        >>> block = Residual(nn.Linear(64, 64), scale=0.5, learnable_scale=True)
    """

    def __init__(
        self,
        module: nn.Module,
        norm: Optional[nn.Module] = None,
        scale: Optional[float] = None,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.module = module
        self.norm = norm

        if scale is not None:
            if learnable_scale:
                self.scale = nn.Parameter(torch.tensor(scale))
            else:
                self.register_buffer('scale', torch.tensor(scale))
        else:
            self.scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply module with residual connection."""
        residual = x
        if self.norm is not None:
            x = self.norm(x)
        out = self.module(x)

        if x.shape != out.shape:
            raise ValueError(
                f"Residual shapes don't match: input {x.shape}, output {out.shape}. "
                "Consider using a projection layer or disable residual."
            )

        if self.scale is not None:
            return residual + self.scale * out
        return residual + out


class GatedResidual(nn.Module):
    """
    Gated residual connection with learnable gate.

    Similar to Highway Networks or Gated Residual Networks.
    The gate controls how much of the residual branch to add.

    Formula: output = (1 - gate) * x + gate * f(x)

    Args:
        module: The module to wrap
        gate_activation: Activation for gate ('sigmoid', 'tanh', 'hard_sigmoid')
        gate_bias: Initial bias for gate (positive = more residual, negative = less)

    Example:
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


def make_residual(
    module: nn.Module,
    residual_type: Literal['add', 'scaled', 'gated', 'prenorm', 'none'] = 'add',
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate residual wrapper.

    Args:
        module: Module to wrap
        residual_type: Type of residual wrapper:
            - 'add': Simple addition (x + f(x))
            - 'scaled': Scaled addition with optional learnable scale
            - 'gated': Learnable gated residual
            - 'prenorm': Pre-normalization residual (Transformer-style)
            - 'none': No residual (return module as-is)
        **kwargs: Additional arguments:
            - For 'scaled': scale (float), learnable_scale (bool)
            - For 'gated': gate_activation, gate_bias
            - For 'prenorm': dim (int), eps (float)

    Returns:
        Wrapped module with residual connections

    Example:
        >>> block = make_residual(nn.Linear(64, 64), 'add')
        >>> block = make_residual(nn.Linear(64, 64), 'gated', gate_bias=2.0)
        >>> block = make_residual(attention_module, 'prenorm', dim=256)
    """
    if residual_type == 'add':
        return Residual(module)
    elif residual_type == 'scaled':
        scale = kwargs.pop('scale', 1.0)
        learnable_scale = kwargs.pop('learnable_scale', False)
        return Residual(module, scale=scale, learnable_scale=learnable_scale)
    elif residual_type == 'gated':
        return GatedResidual(module, **kwargs)
    elif residual_type == 'prenorm':
        dim = kwargs.pop('dim')
        eps = kwargs.pop('eps', 1e-5)
        return Residual(module, norm=nn.LayerNorm(dim, eps=eps))
    elif residual_type == 'none':
        return module
    else:
        raise ValueError(f"Unknown residual type: {residual_type}")


__all__ = [
    'Residual',
    'GatedResidual',
    'make_residual',
]
