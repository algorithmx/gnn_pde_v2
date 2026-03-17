"""
Graph U-Net processor with encoder-decoder architecture.

Based on "Graph U-Nets" (Gao & Ji, ICML 2019).
"""

from typing import List, Optional, Callable
import torch
import torch.nn as nn

from ...core.graph import GraphsTuple
from ...core.mlp import MLP
from ..processors import GraphNetBlock
from .graph_pooling import GraphPool, GraphUnpool


class GraphUNetBlock(nn.Module):
    """Single block in Graph U-Net.
    
    Consists of GraphNet processing followed by optional pooling.
    
    Args:
        latent_dim: Feature dimension
        hidden_dim: MLP hidden dimension
        n_layers: Number of GraphNet layers in this block
        activation: Activation function
        pool_ratio: If < 1.0, pool to this ratio of nodes
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        activation: str = "gelu",
        pool_ratio: float = 1.0,
        aggregate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pool_ratio = pool_ratio
        
        # Graph processing layers
        self.layers = nn.ModuleList([
            GraphNetBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                aggregate_fn=aggregate_fn,
            )
            for _ in range(n_layers)
        ])
        
        # Pooling (if needed)
        self.pool = None
        if pool_ratio < 1.0:
            # Will be created during forward when we know k
            self.pool = None  # Created dynamically
    
    def forward(
        self,
        graph: GraphsTuple,
        target_k: Optional[int] = None
    ) -> tuple[GraphsTuple, Optional[torch.Tensor]]:
        """Process and optionally pool graph.
        
        Returns:
            processed_graph: Processed graph
            indices: Indices if pooled, None otherwise
        """
        # Process through GraphNet layers
        x = graph
        for layer in self.layers:
            x = layer(x)
        
        # Pool if needed
        if self.pool_ratio < 1.0 and target_k is not None:
            device = x.nodes.device if x.nodes is not None else torch.device('cpu')
            pool = GraphPool(k=target_k, feature_dim=self.latent_dim).to(device)
            x, indices = pool(x)
            return x, indices
        
        return x, None


class GraphUNetProcessor(nn.Module):
    """Graph U-Net processor with encoder-decoder architecture.
    
    Architecture:
        Input
          ↓
    [Encoder 1] ───────────┐
          ↓                │
    [Encoder 2] ───────────┤ Skip
          ↓                │ Connections
    [Encoder 3] ───────────┤
          ↓                │
      [Bottleneck]         │
          ↓                │
    [Decoder 3] ←──────────┘
          ↓
    [Decoder 2]
          ↓
    [Decoder 1]
          ↓
        Output
    
    Args:
        latent_dim: Feature dimension
        n_levels: Number of encoder-decoder levels
        nodes_per_level: List of node counts per level
        hidden_dim: MLP hidden dimension
        n_layers_per_level: GraphNet layers per level
        skip_connection: "add" or "concat"
        activation: Activation function
    
    Reference:
        Gao & Ji, "Graph U-Nets", ICML 2019
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_levels: int = 3,
        nodes_per_level: Optional[List[int]] = None,
        hidden_dim: int = 128,
        n_layers_per_level: int = 2,
        skip_connection: str = "add",
        activation: str = "gelu",
        aggregate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_levels = n_levels
        self.nodes_per_level = nodes_per_level
        self.skip_connection = skip_connection
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(n_levels):
            pool_ratio = 0.5 if i < n_levels - 1 else 1.0
            self.encoders.append(
                GraphUNetBlock(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers_per_level,
                    activation=activation,
                    pool_ratio=pool_ratio,
                    aggregate_fn=aggregate_fn,
                )
            )
        
        # Decoder blocks (no pooling)
        self.decoders = nn.ModuleList()
        decoder_input_dim = latent_dim * 2 if skip_connection == "concat" else latent_dim
        for i in range(n_levels):
            self.decoders.append(
                GraphUNetBlock(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers_per_level,
                    activation=activation,
                    pool_ratio=1.0,  # No pooling
                    aggregate_fn=aggregate_fn,
                )
            )
            
            # Projection for skip connections if concat
            if skip_connection == "concat":
                self.skip_projections = nn.ModuleList([
                    nn.Linear(latent_dim * 2, latent_dim)
                    for _ in range(n_levels)
                ])
        
        self.unpool = GraphUnpool()
        
        # Store encoder outputs for skip connections
        self._encoder_outputs = []
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Process through U-Net architecture.
        
        Args:
            graph: Input graph
        
        Returns:
            Processed graph
        """
        x = graph
        # encoder_outputs stores pre-pool features at each level (for skip connections)
        encoder_outputs = []
        indices_list = []
        # original_sizes[j] = node count before the j-th pooling operation
        original_sizes = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoders):
            # Determine target k for pooling
            target_k = None
            if i < self.n_levels - 1:
                num_nodes = x.nodes.shape[0] if x.nodes is not None else 0
                if self.nodes_per_level is not None and i < len(self.nodes_per_level):
                    target_k = self.nodes_per_level[i]
                else:
                    target_k = int(num_nodes * 0.5)
            
            # Process only (pass target_k=None so encoder skips its internal pool)
            x_processed, _ = encoder(x, target_k=None)
            
            # Save pre-pool features for skip connections
            encoder_outputs.append(x_processed)
            
            # Pool separately, recording the pre-pool node count
            if target_k is not None:
                original_sizes.append(x_processed.nodes.shape[0])
                device = x_processed.nodes.device
                pool = GraphPool(k=target_k, feature_dim=self.latent_dim).to(device)
                x, indices = pool(x_processed)
                indices_list.append(indices)
            else:
                x = x_processed
        
        # Decoder path
        for i in range(self.n_levels - 1, -1, -1):
            # Unpool using the stored pre-pool size so indices are in-bounds
            if i < len(indices_list):
                original_size = original_sizes[i]
                x = self.unpool(x, indices_list[i], original_size)
            
            # Skip connection with pre-pool encoder features (sizes now match)
            if i < len(encoder_outputs):
                skip = encoder_outputs[i]
                if self.skip_connection == "concat" and x.nodes is not None and skip.nodes is not None:
                    # Concatenate and project
                    combined = torch.cat([x.nodes, skip.nodes], dim=-1)
                    x = x.replace(nodes=self.skip_projections[i](combined))
                elif self.skip_connection == "add" and x.nodes is not None and skip.nodes is not None:
                    # Add skip connection
                    x = x.replace(nodes=x.nodes + skip.nodes)
            
            # Decode
            x, _ = self.decoders[i](x, target_k=None)
        
        return x


__all__ = [
    "GraphUNetBlock",
    "GraphUNetProcessor",
]
