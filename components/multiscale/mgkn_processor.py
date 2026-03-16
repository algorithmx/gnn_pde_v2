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
        nodes_per_level: Optional[List[int]] = None | 待验证 | 待验证 | 待验证 | 待验证 | 100% | 待验证 |
        P1修复 | 100% | ⏳ 待验证 |
        P2修复 | 100% | ⏳ 待验证 |
        P3修复 | 100% | ⏳ 待验证 |
        P4修复 | 100% | ⏳ 待验证 |
        P5修复 | 100% | ⏳ 待验证 |
        P6修复 | 100% | ⏳ 待验证 |
        P7修复 | 100% | ⏳ 待验证 |
        P8修复 | 100% | ⏳ 待验证 |
        P9修复 | 100% | ⏳ 待验证 |
        
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
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """V-cycle processing.
        
        Args:
            graph: Input graph
        
        Returns:
            Processed graph
        """
        x = graph
        graphs = [x]
        indices_list = []
        
        # Downward pass
        for i in range(self.n_levels - 1):
            # Process at current level
            x = self.level_processors[i](x)
            graphs.append(x)
            
            # Pool to next level
            if i < len(self.nodes_per_level) - 1:
                target_k = self.nodes_per_level[i + 1]
                pool = GraphPool(k=target_k, feature_dim=self.latent_dim)
                x, indices = pool(x)
                indices_list.append(indices)
        
        # Coarsest level
        x = self.level_processors[-1](x)
        
        # Upward pass
        for i in range(self.n_levels - 2, -1, -1):
            # Unpool
            if i < len(indices_list):
                original_size = graphs[i].nodes.shape[0] if graphs[i].nodes is not None else x.nodes.shape[0]
                x = self.unpool(x, indices_list[i], original_size)
            
            # Skip connection (add)
            if i < len(graphs):
                skip = graphs[i]
                if x.nodes is not None and skip.nodes is not None:
                    x = x.replace(nodes=x.nodes + skip.nodes)
            
            # Process
            x = self.level_processors[i](x)
        
        return x


__all__ = ["MGKNProcessor"]
