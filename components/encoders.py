
from dataclasses import replace
from typing import Optional
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..core.mlp import MLP


class MeshEncoder(nn.Module):
    """
    Encoder for mesh-based graphs using MLPs for node/edge/global features.

    Encodes input features into a common latent space suitable for processing
    by GraphNet processors. Each feature type (nodes, edges, globals) has its
    own MLP encoder.

    Args:
        node_in_dim: Input dimension for node features
        edge_in_dim: Input dimension for edge features
        global_in_dim: Optional input dimension for global features. If None,
            no global encoder is created.
        latent_dim: Output dimension for all encoded features
        hidden_dim: Hidden dimension for internal MLP layers
        activation: Activation function name ('relu', 'gelu', 'silu', 'tanh')

    Example:
        >>> encoder = MeshEncoder(
        ...     node_in_dim=11,
        ...     edge_in_dim=3,
        ...     global_in_dim=None,
        ...     latent_dim=128
        ... )
        >>> encoded = encoder(graph)
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_in_dim: Optional[int],
        latent_dim: int,
        hidden_dim: int = 128,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, latent_dim, [hidden_dim], activation=activation)
        self.edge_encoder = MLP(edge_in_dim, latent_dim, [hidden_dim], activation=activation)
        self.global_encoder = (
            MLP(global_in_dim, latent_dim, [hidden_dim], activation=activation)
            if global_in_dim is not None else None
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Encode input graph features to latent space.

        Args:
            graph: Input GraphsTuple with nodes, edges, and optionally globals

        Returns:
            GraphsTuple with encoded features in latent space

        Note:
            If graph.nodes or graph.edges is None, they remain None in output.
            If global_in_dim was None during construction, globals are not encoded.
        """
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if graph.edges is not None else None
        globals_ = self.global_encoder(graph.globals) if self.global_encoder is not None and graph.globals is not None else None
        return replace(graph, nodes=nodes, edges=edges, globals=globals_)
