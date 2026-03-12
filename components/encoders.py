"""
Encoders for the GNN-PDE framework.

Single MLP class with functional initialization.
No wrapper classes - use MLP directly.
"""

from typing import List, Optional, Callable
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    """
    Flexible MLP with functional weight initialization.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh')
        dropout: Dropout probability
        use_layer_norm: Whether to use LayerNorm
        weight_init: Weight initialization function (default: xavier_uniform_)
        bias_init: Bias initialization function (default: constant 0)
        
    Example:
        >>> # Standard usage
        >>> mlp = MLP(64, 64, hidden_dims=[128, 128])
        >>> 
        >>> # With custom initialization
        >>> mlp = MLP(64, 64, [128], weight_init=init.kaiming_normal_)
        >>> 
        >>> # For single layer (no hidden)
        >>> linear = MLP(64, 64, hidden_dims=[])
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()
        
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'silu':
            act_fn = nn.SiLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Don't add activation/norm/dropout to last layer
            if i < len(dims) - 2:
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        self.weight_init = weight_init
        self.bias_init = bias_init
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.weight_init(m.weight)
            if m.bias is not None:
                self.bias_init(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Convenience functions for common encoder patterns

def make_mlp_encoder(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    **mlp_kwargs
) -> MLP:
    """
    Create an MLP encoder with standard architecture.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers (including output, excluding input)
        **mlp_kwargs: Additional arguments for MLP
        
    Returns:
        MLP encoder
        
    Example:
        >>> encoder = make_mlp_encoder(10, 128, hidden_dim=128, num_layers=2)
        >>> # Creates: Linear(10, 128) -> LayerNorm -> GELU -> Linear(128, 128)
    """
    if num_layers == 1:
        hidden_dims = []
    else:
        hidden_dims = [hidden_dim] * (num_layers - 1)
    
    return MLP(in_dim, out_dim, hidden_dims, **mlp_kwargs)
