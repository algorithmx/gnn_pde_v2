"""
Convenience models for Graph Neural Networks.
"""

from typing import Optional
import torch
from ..core.graph import GraphsTuple
from ..convenient.registry import AutoRegisterModel
from ..components.encoders import MLPMeshEncoder
from ..components.processors import GraphNetProcessor
from ..components.decoders import MLPDecoder
from .encode_process_decode import EncodeProcessDecode


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
        encoder = MLPMeshEncoder(
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
            global_dim=latent_dim if global_in_dim is not None else None,
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
        n_layers: int = 15,
        hidden_dim: int = 128,
        activation: str = 'silu',  # MeshGraphNets uses silu
    ):
        super().__init__()
        
        # Encoder
        encoder = MLPMeshEncoder(
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
            n_layers=n_layers,
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
