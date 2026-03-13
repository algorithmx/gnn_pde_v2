"""
Encoders for the GNN-PDE framework.

Graph encoder wrappers using the canonical MLP from core.mlp.
"""

from dataclasses import replace
from typing import List, Optional, Callable
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.graph import GraphsTuple
from ..core.mlp import MLP


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
        edges = self.edge_encoder(graph.edges) if self.edge_encoder is not None and graph.edges is not None else None
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
