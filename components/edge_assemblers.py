"""Composable edge feature assemblers for EdgeConvBlock.

Each assembler defines how to construct edge features from graph structure
(sender/receiver nodes and edge attributes).

Example::

    from gnn_pde_v2.components import (
        EdgeConvBlock,
        NodeDifferenceAssembler,
        ConcatWithEdgesAssembler,
    )
    from gnn_pde_v2.core import MLP

    # Default DGCNN-style with node difference
    block = EdgeConvBlock(
        latent_dim=128,
        edge_assembler=NodeDifferenceAssembler(128),
    )

    # With edge attributes
    block = EdgeConvBlock(
        latent_dim=128,
        edge_assembler=ConcatWithEdgesAssembler(128, edge_dim=3),
        edge_transform=MLP(259, 128, [128], 'relu'),
    )
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from ..core.graph import GraphsTuple

__all__ = [
    "EdgeFeatureAssembler",
    "NodeDifferenceAssembler",
    "ConcatAssembler",
    "DifferenceOnlyAssembler",
    "ConcatWithEdgesAssembler",
]


class EdgeFeatureAssembler(nn.Module, ABC):
    """Abstract base for edge feature assembly strategies.

    Implementations define how to construct per-edge feature vectors from
    graph structure (sender/receiver nodes, edge attributes).

    Used by :class:`~gnn_pde_v2.components.EdgeConvBlock` to support
    flexible edge feature construction without modifying the block itself.

    Subclasses must implement:
    - :attr:`out_dim`: Property returning the output dimension
    - :meth:`forward`: Method taking a GraphsTuple and returning edge features

    Example::

        class MyAssembler(EdgeFeatureAssembler):
            def __init__(self, latent_dim: int):
                super().__init__()
                self._out_dim = latent_dim * 2

            @property
            def out_dim(self) -> int:
                return self._out_dim

            def forward(self, graph: GraphsTuple) -> Tensor:
                v_i = graph.nodes[graph.receivers]
                v_j = graph.nodes[graph.senders]
                return torch.cat([v_i, v_j], dim=-1)
    """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Output dimension of assembled edge features.

        Returns:
            The dimension of the tensor returned by :meth:`forward`.
        """
        ...

    @abstractmethod
    def forward(self, graph: GraphsTuple) -> Tensor:
        """Assemble edge features from graph.

        Args:
            graph: Input graph with nodes, edges, senders, receivers.
                The graph must have non-None senders and receivers.

        Returns:
            Assembled edge features of shape ``[E, out_dim]``, where ``E``
            is the number of edges in the graph.
        """
        ...


class NodeDifferenceAssembler(EdgeFeatureAssembler):
    """Assemble ``[v_i; v_j - v_i]`` — DGCNN default.

    Concatenates receiver features with sender-receiver difference.
    This is the default mode for EdgeConvBlock and matches the
    original DGCNN paper.

    Output dimension: ``2 * latent_dim``

    Args:
        latent_dim: Node feature dimension.

    Example::

        assembler = NodeDifferenceAssembler(latent_dim=128)
        # For latent_dim=128, out_dim=256
        # Features: [v_i; v_j - v_i]
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim

    @property
    def out_dim(self) -> int:
        return 2 * self.latent_dim

    def forward(self, graph: GraphsTuple) -> Tensor:
        v_i = graph.nodes[graph.receivers]  # [E, H]
        v_j = graph.nodes[graph.senders]    # [E, H]
        return torch.cat([v_i, v_j - v_i], dim=-1)


class ConcatAssembler(EdgeFeatureAssembler):
    """Assemble ``[v_i; v_j]`` — simple concatenation.

    Concatenates receiver and sender features directly.

    Output dimension: ``2 * latent_dim``

    Args:
        latent_dim: Node feature dimension.

    Example::

        assembler = ConcatAssembler(latent_dim=128)
        # For latent_dim=128, out_dim=256
        # Features: [v_i; v_j]
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim

    @property
    def out_dim(self) -> int:
        return 2 * self.latent_dim

    def forward(self, graph: GraphsTuple) -> Tensor:
        v_i = graph.nodes[graph.receivers]
        v_j = graph.nodes[graph.senders]
        return torch.cat([v_i, v_j], dim=-1)


class DifferenceOnlyAssembler(EdgeFeatureAssembler):
    """Assemble ``v_j - v_i`` — difference only.

    Returns only the sender-receiver difference, without receiver features.
    Most compact representation but may lose information.

    Output dimension: ``latent_dim``

    Args:
        latent_dim: Node feature dimension.

    Example::

        assembler = DifferenceOnlyAssembler(latent_dim=128)
        # For latent_dim=128, out_dim=128
        # Features: v_j - v_i
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim

    @property
    def out_dim(self) -> int:
        return self.latent_dim

    def forward(self, graph: GraphsTuple) -> Tensor:
        v_i = graph.nodes[graph.receivers]
        v_j = graph.nodes[graph.senders]
        return v_j - v_i


class ConcatWithEdgesAssembler(EdgeFeatureAssembler):
    """Assemble ``[v_i; v_j - v_i; e_ij]`` — include edge attributes.

    Concatenates receiver features, sender-receiver difference, and
    original edge attributes. Use this when edge features contain
    important information (e.g., distance, direction, edge type).

    Output dimension: ``2 * latent_dim + edge_dim``

    Args:
        latent_dim: Node feature dimension.
        edge_dim: Edge attribute dimension.

    Example::

        assembler = ConcatWithEdgesAssembler(latent_dim=128, edge_dim=3)
        # For latent_dim=128, edge_dim=3, out_dim=259
        # Features: [v_i; v_j - v_i; e_ij]
    """

    def __init__(self, latent_dim: int, edge_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}")
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim

    @property
    def out_dim(self) -> int:
        return 2 * self.latent_dim + self.edge_dim

    def forward(self, graph: GraphsTuple) -> Tensor:
        v_i = graph.nodes[graph.receivers]
        v_j = graph.nodes[graph.senders]
        return torch.cat([v_i, v_j - v_i, graph.edges], dim=-1)
