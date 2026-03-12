"""
MLP-based encoders for node and edge features.
"""

from typing import Optional, List, Union
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..initializers import Initializer, get_initializer


class MLP(nn.Module):
    """
    Flexible MLP with configurable layers, activation, and normalization.
    
    Supports custom weight initialization schemes for compatibility with
    various deep learning libraries (DeepXDE, NeuralOperator, etc.).
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh')
        dropout: Dropout probability
        use_layer_norm: Whether to use LayerNorm
        initializer: Weight initializer ('glorot_uniform', 'he_normal', etc.)
        bias_init: Bias initializer (default: 'constant_0')
        
    Examples:
        >>> # Standard MLP
        >>> mlp = MLP(64, 64, hidden_dims=[128, 128])
        >>> 
        >>> # With custom initialization (DeepXDE-style)
        >>> mlp = MLP(64, 64, hidden_dims=[128], initializer='glorot_uniform')
        >>> 
        >>> # With He initialization (good for ReLU)
        >>> mlp = MLP(64, 64, hidden_dims=[128], initializer='he_normal')
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        initializer: Union[str, Initializer] = 'glorot_uniform',
        bias_init: Union[str, Initializer] = 'constant_0',
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
        
        # Store initializers
        self.weight_initializer = get_initializer(initializer)
        self.bias_initializer = get_initializer(bias_init)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.weight_initializer(m.weight)
            if m.bias is not None:
                self.bias_initializer(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for node and edge features.
    
    Applies separate MLPs to nodes and edges (if present).
    """
    
    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        edge_in_dim: Optional[int] = None,
        edge_out_dim: Optional[int] = None,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=node_out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        if edge_in_dim is not None and edge_out_dim is not None:
            self.edge_encoder = MLP(
                in_dim=edge_in_dim,
                out_dim=edge_out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.edge_encoder = None
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Encode node and edge features."""
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if self.edge_encoder and graph.edges is not None else None
        
        return graph.replace(nodes=nodes, edges=edges)


class MLPMeshEncoder(nn.Module):
    """
    MeshGraphNets-style encoder with separate MLPs for nodes, edges, and globals.
    """
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_in_dim: Optional[int],
        latent_dim: int,
        hidden_dims: List[int] = [128],
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        self.edge_encoder = MLP(
            in_dim=edge_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        if global_in_dim is not None:
            self.global_encoder = MLP(
                in_dim=global_in_dim,
                out_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.global_encoder = None
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Encode all feature types."""
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if graph.edges is not None else None
        globals_ = self.global_encoder(graph.globals) if self.global_encoder and graph.globals is not None else None
        
        return graph.replace(nodes=nodes, edges=edges, globals=globals_)
