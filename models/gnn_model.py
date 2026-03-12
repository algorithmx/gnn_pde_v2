"""
Convenience models for Graph Neural Networks.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..convenient.registry import AutoRegisterModel
from ..components.encoders import MLP, make_mlp_encoder
from ..components.processors import GraphNetProcessor
from ..components.decoders import MLPDecoder
from .encode_process_decode import EncodeProcessDecode


class _MeshEncoder(nn.Module):
    """Simple mesh encoder using separate MLPs for nodes, edges, globals."""
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_in_dim: Optional[int],
        latent_dim: int,
        hidden_dims: List[int],
        activation: str = 'gelu',
    ):
        super().__init__()
        self.node_encoder = make_mlp_encoder(node_in_dim, latent_dim, hidden_dims[0] if hidden_dims else latent_dim, len(hidden_dims) + 1, activation=activation)
        self.edge_encoder = make_mlp_encoder(edge_in_dim, latent_dim, hidden_dims[0] if hidden_dims else latent_dim, len(hidden_dims) + 1, activation=activation)
        self.global_encoder = make_mlp_encoder(global_in_dim, latent_dim, hidden_dims[0] if hidden_dims else latent_dim, len(hidden_dims) + 1, activation=activation) if global_in_dim else None
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if graph.edges is not None else None
        globals_ = self.global_encoder(graph.globals) if self.global_encoder and graph.globals is not None else None
        from dataclasses import replace
        return replace(graph, nodes=nodes, edges=edges, globals=globals_)


class GraphNet(AutoRegisterModel, name='graphnet'):
    """
    Standard Graph Neural Network.
    
    Simple encoder-processor-decoder with configurable components.
    """
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        latent_dim: int = 128,
        n_layers: int = 4,
        hidden_dim: int = 128,
        global_in_dim: Optional[int] = None,
        activation: str = 'gelu',
        residual: bool = True,
    ):
        super().__init__()
        
        # Encoder
        encoder = _MeshEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=global_in_dim,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
        
        # Processor
        processor = GraphNetProcessor(
            node_dim=latent_dim,
            edge_dim=latent_dim,
            global_dim=latent_dim if global_in_dim else None,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=residual,
        )
        
        # Decoder
        decoder = MLPDecoder(
            node_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
        
        self.epd = EncodeProcessDecode(encoder, processor, decoder)
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        return self.epd(graph)


class MeshGraphNet(AutoRegisterModel, name='meshgraphnet'):
    """
    MeshGraphNets-style model for mesh-based simulations.
    
    Pre-configured with MeshGraphNets hyperparameters.
    """
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        latent_dim: int = 128,
        n_message_passing: int = 15,
        hidden_dim: int = 128,
        activation: str = 'silu',  # MeshGraphNets uses silu
    ):
        super().__init__()
        
        # Encoder
        encoder = _MeshEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=None,
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
        
        # Processor (MeshGraphNets uses many layers)
        processor = GraphNetProcessor(
            node_dim=latent_dim,
            edge_dim=latent_dim,
            global_dim=None,
            n_layers=n_message_passing,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=True,
        )
        
        # Decoder
        decoder = MLPDecoder(
            node_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
        
        self.epd = EncodeProcessDecode(encoder, processor, decoder)
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        return self.epd(graph)
