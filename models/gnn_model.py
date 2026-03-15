"""
Convenience models for Graph Neural Networks.
"""

from dataclasses import replace
from typing import Optional
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..core.mlp import MLP
from ..core.registry import AutoRegisterModel
from ..components.processors import GraphNetProcessor, GlobalGraphNetProcessor
from ..components.decoders import MLPDecoder
from .encode_process_decode import EncodeProcessDecode


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


class GraphNet(AutoRegisterModel, name='graphnet', aliases=['gnn', 'graph_net']):
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
        if global_in_dim is not None:
            processor = GlobalGraphNetProcessor(
                latent_dim=latent_dim,
                global_latent_dim=latent_dim,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                activation=activation,
                residual=residual,
            )
        else:
            processor = GraphNetProcessor(
                latent_dim=latent_dim,
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


class MeshGraphNet(AutoRegisterModel, name='meshgraphnet', aliases=['mgn', 'mesh_graph_net']):
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
        
        # Processor (MeshGraphNets uses many layers; no global state)
        processor = GraphNetProcessor(
            latent_dim=latent_dim,
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
