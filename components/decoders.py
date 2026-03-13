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
        node_dim: int,
        out_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'gelu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.mlp = MLP(
            in_dim=node_dim,
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
    """
    
    def __init__(
        self,
        node_dim: int,
        out_dims: List[int],
        hidden_dims: List[int] = [128],
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.out_dims = out_dims
        self.decoders = nn.ModuleList([
            MLP(
                in_dim=node_dim,
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
    ) -> List[torch.Tensor]:
        """
        Decode to multiple outputs.
        
        Returns:
            List of [N, out_dim_i] tensors
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for IndependentMLPDecoder")
        
        return [decoder(graph.nodes) for decoder in self.decoders]
