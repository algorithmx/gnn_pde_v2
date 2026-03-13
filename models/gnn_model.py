"""
Convenience models for Graph Neural Networks.
"""

from dataclasses import replace
from typing import Optional
import torch
from ..core.graph import GraphsTuple
from ..core.mlp import MLP
from ..core.registry import AutoRegisterModel
from ..components.encoders import MeshEncoder
from ..components.processors import GraphNetProcessor
from ..components.decoders import MLPDecoder
from .encode_process_decode import EncodeProcessDecode


class GraphNet(AutoRegisterModel, name='graphnet'):
    """
    Standard Graph Neural Network.
    
    Simple encoder-processor-decoder with configurable components.
    Suitable for general graph-based learning tasks.
    
    Args:
        node_in_dim: Input dimension for node features
        edge_in_dim: Input dimension for edge features
        out_dim: Output dimension for predictions
        latent_dim: Latent dimension for internal representations
        n_layers: Number of processor layers
        hidden_dim: Hidden dimension for MLPs
        global_in_dim: Optional input dimension for global features
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh')
        residual: Whether to use residual connections in processor
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
        encoder = MeshEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=global_in_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        
        # Processor
        processor = GraphNetProcessor(
            latent_dim=latent_dim,
            global_latent_dim=latent_dim if global_in_dim is not None else None,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=residual,
        )

        # Decoder
        decoder = MLPDecoder(
            latent_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
        
        self.epd = EncodeProcessDecode(encoder, processor, decoder)
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass through the GraphNet.
        
        Args:
            graph: Input GraphsTuple with node and edge features
            
        Returns:
            [N, out_dim] - Output predictions at each node
        """
        return self.epd(graph)


class MeshGraphNet(AutoRegisterModel, name='meshgraphnet'):
    """
    MeshGraphNets-style model for mesh-based simulations.
    
    Pre-configured with MeshGraphNets hyperparameters from DeepMind's
    "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., 2021).
    
    Args:
        node_in_dim: Input dimension for node features (e.g., position, velocity)
        edge_in_dim: Input dimension for edge features (e.g., relative displacement)
        out_dim: Output dimension for predictions (e.g., next-step velocity)
        latent_dim: Latent dimension for internal representations (default: 128)
        n_layers: Number of processor layers (default: 15, as per paper)
        hidden_dim: Hidden dimension for MLPs (default: 128)
        activation: Activation function (default: 'silu' as per paper)
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
        encoder = MeshEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            global_in_dim=None,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        
        # Processor (MeshGraphNets uses many layers)
        processor = GraphNetProcessor(
            latent_dim=latent_dim,
            global_latent_dim=None,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            residual=True,
        )

        # Decoder
        decoder = MLPDecoder(
            latent_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
        
        self.epd = EncodeProcessDecode(encoder, processor, decoder)
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass through the MeshGraphNet.
        
        Args:
            graph: Input GraphsTuple with node and edge features
            
        Returns:
            [N, out_dim] - Output predictions at each node
        """
        return self.epd(graph)
