"""
Encoders for the GNN-PDE framework.

Canonical MLP building blocks and graph encoder wrappers.
"""

from dataclasses import replace
from typing import List, Optional, Callable
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.graph import GraphsTuple


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


class MLPEncoder(nn.Module):
    """
    Simple graph encoder with separate node and optional edge MLPs.
    """

    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        edge_in_dim: Optional[int] = None,
        edge_out_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()

        hidden_dims = [128, 128] if hidden_dims is None else hidden_dims

        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=node_out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )

        if edge_in_dim is not None and edge_out_dim is not None:
            self.edge_encoder = MLP(
                in_dim=edge_in_dim,
                out_dim=edge_out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                weight_init=weight_init,
                bias_init=bias_init,
            )
        else:
            self.edge_encoder = None

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if self.edge_encoder is not None and graph.edges is not None else None
        return replace(graph, nodes=nodes, edges=edges)


class MLPMeshEncoder(nn.Module):
    """
    MeshGraphNets-style encoder with separate node, edge, and optional global MLPs.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_in_dim: Optional[int],
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()

        hidden_dims = [128] if hidden_dims is None else hidden_dims

        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.edge_encoder = MLP(
            in_dim=edge_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.global_encoder = (
            MLP(
                in_dim=global_in_dim,
                out_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                weight_init=weight_init,
                bias_init=bias_init,
            )
            if global_in_dim is not None
            else None
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if graph.edges is not None else None
        globals_ = self.global_encoder(graph.globals) if self.global_encoder is not None and graph.globals is not None else None
        return replace(graph, nodes=nodes, edges=edges, globals=globals_)


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
