"""
MLP-based decoders for output generation.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..core.mlp import MLP


class MLPDecoder(nn.Module):
    """
    Simple MLP decoder that operates on node features.

    Outputs predictions at each node position.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()

        self.mlp = MLP(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode node features to output.
        
        Args:
            graph: Processed GraphsTuple
            query_positions: Ignored for this decoder (outputs at nodes)
            
        Returns:
            [N, out_dim] - Output at each node
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for MLPDecoder")
        
        return self.mlp(graph.nodes)


class IndependentMLPDecoder(nn.Module):
    """
    Decoder with separate MLPs for each output component.

    Useful for multi-task settings or when outputs have different scales.

    Outputs are concatenated along the feature dimension to satisfy the
    :class:`~gnn_pde_v2.core.protocols.Decoder` protocol (returns Tensor).

    Args:
        latent_dim: Input feature dimension
        out_dims: List of output dimensions for each component. The total
            output dimension will be ``sum(out_dims)``.
        hidden_dims: Hidden layer dimensions for each MLP
        activation: Activation function name

    Note:
        Use :meth:`forward_separate` if you need individual component outputs
        as a list instead of concatenated tensor.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dims: List[int],
        hidden_dims: List[int] = [128],
        activation: str = 'gelu',
    ):
        super().__init__()

        self.out_dims = out_dims
        self.total_out_dim = sum(out_dims)
        self.decoders = nn.ModuleList([
            MLP(
                in_dim=latent_dim,
                out_dim=dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )
            for dim in out_dims
        ])

    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode to concatenated output tensor.

        Args:
            graph: Processed GraphsTuple
            query_positions: Ignored for this decoder (outputs at nodes)

        Returns:
            [N, sum(out_dims)] - Concatenated output at each node
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for IndependentMLPDecoder")

        outputs = [decoder(graph.nodes) for decoder in self.decoders]
        return torch.cat(outputs, dim=-1)

    def forward_separate(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Decode to separate output tensors (one per component).

        Use this method when you need individual component outputs rather than
        a concatenated tensor.

        Args:
            graph: Processed GraphsTuple
            query_positions: Ignored for this decoder

        Returns:
            List of [N, out_dim_i] tensors, one per output component
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for IndependentMLPDecoder")

        return [decoder(graph.nodes) for decoder in self.decoders]
