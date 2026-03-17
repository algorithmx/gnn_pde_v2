"""
Multipole Graph Neural Operator (MGKN) processor.

Based on "Multipole Graph Neural Operator for Parametric PDEs" (Li et al., NeurIPS 2020).
"""

from typing import List, Optional, Callable
import torch
import torch.nn as nn

from ...core.graph import GraphsTuple
from ...core.mlp import MLP
from ..processors import GraphNetBlock
from .graph_pooling import GraphPool, GraphUnpool


class MGKNProcessor(nn.Module):
    """Multipole Graph Neural Operator processor with V-cycle.
    
    Implements the V-cycle algorithm from MGKN paper:
    - Downward pass: Fine → Coarse
    - Upward pass: Coarse → Fine with skip connections
    
    Args:
        latent_dim: Feature dimension
        n_levels: Number of hierarchy levels
        nodes_per_level: Node counts per level
        hidden_dim: MLP hidden dimension
        n_layers_per_level: Layers per level
    
    Reference:
        Li et al., "Multipole Graph Neural Operator", NeurIPS 2020
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_levels: int = 3,
        nodes_per_level: Optional[List[int]] = None,
        hidden_dim: int = 128,
        n_layers_per_level: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_levels = n_levels
        self.nodes_per_level = nodes_per_level or [400, 100, 25]
        
        # Level-specific processors
        self.level_processors = nn.ModuleList([
            GraphNetBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                activation="gelu",
            )
            for _ in range(n_levels)
        ])
        
        self.pool = GraphPool
        self.unpool = GraphUnpool()
    
    def _ensure_edges(self, graph: GraphsTuple) -> GraphsTuple:
        """Add self-loops with zero features if graph has no edges.

        Required because GraphNetBlock needs edge tensors to compute messages.
        After aggressive pooling, all edges may have been filtered out.
        """
        if graph.senders is not None and graph.edges is not None:
            return graph
        device = graph.nodes.device if graph.nodes is not None else torch.device('cpu')
        n = graph.nodes.shape[0] if graph.nodes is not None else 0
        self_idx = torch.arange(n, device=device)
        return graph.with_topology(
            senders=self_idx,
            receivers=self_idx,
            edges=torch.zeros(n, self.latent_dim, device=device,
                              dtype=graph.nodes.dtype if graph.nodes is not None else torch.float32),
            n_edge=torch.tensor([n], device=device),
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """V-cycle processing.

        Args:
            graph: Input graph

        Returns:
            Processed graph
        """
        x = graph
        # graphs[j+1] stores the post-processed graph just BEFORE the j-th pool,
        # used for correct unpool sizes, edge restoration, and skip connections.
        graphs = [x]
        indices_list = []

        # Downward pass
        for i in range(self.n_levels - 1):
            # Process at current level
            x = self.level_processors[i](x)
            graphs.append(x)  # save pre-pool features for this level

            # Pool to next level
            num_nodes = x.nodes.shape[0] if x.nodes is not None else 0
            if i < len(self.nodes_per_level):
                target_k = self.nodes_per_level[i]
            else:
                target_k = num_nodes // 2

            if target_k < num_nodes:
                pool = GraphPool(k=target_k, feature_dim=self.latent_dim)
                pool = pool.to(x.nodes.device if x.nodes is not None else 'cpu')
                x, indices = pool(x)
                indices_list.append(indices)

            # Ensure edges exist (pooling may eliminate all edges)
            x = self._ensure_edges(x)

        # Coarsest level
        x = self.level_processors[-1](x)

        # Upward pass
        for i in range(self.n_levels - 2, -1, -1):
            # Unpool using the pre-pool size stored in graphs[i+1]
            if i < len(indices_list):
                ref = graphs[i + 1]  # graph saved just before pool i
                original_size = ref.nodes.shape[0] if ref.nodes is not None else x.nodes.shape[0]
                x = self.unpool(x, indices_list[i], original_size)
                # Restore graph topology from pre-pool graph
                x = x.with_topology(ref)

            # Skip connection: add pre-pool features at the same resolution
            ref = graphs[i + 1] if i + 1 < len(graphs) else graphs[-1]
            if x.nodes is not None and ref.nodes is not None:
                x = x.replace(nodes=x.nodes + ref.nodes)

            # Process
            x = self.level_processors[i](x)

        return x


__all__ = ["MGKNProcessor"]
