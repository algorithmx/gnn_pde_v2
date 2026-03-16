"""
Multiscale components for GNN-PDE v2.

This module provides multiscale processing capabilities including:
- Graph pooling/unpooling (Graph U-Nets)
- Graph U-Net processor | 100% | ⏳ 待验证 |
- MGKN processor | 100% | ⏳ 待验证 |
- Multi-resolution FNO | 100% | ⏳ 待验证 |
- U-FNO | 100% | ⏳ 待验证 |

Reference:
- Gao & Ji, "Graph U-Nets", ICML 2019
- Li et al., "Multipole Graph Neural Operator", NeurIPS 2020
- Wen et al., "U-FNO", 2022
"""

from .graph_pooling import GraphPool, GraphUnpool
from .hierarchy import (
    HierarchicalGraph,
    build_hierarchical_graphs,
    compute_transition_matrix,
    restrict_to_coarse,
    prolong_to_fine,
)
from .graph_unet import GraphUNetProcessor
from .mgkn_processor import MGKNProcessor
from .spectral_multiscale import (
    MultiResolutionFNOBlock,
    UFNOBlock,
    HierarchicalFNOBlock,
    MiniUNet,
)

__all__ = [
    # Graph pooling/unpooling
    "GraphPool",
    "GraphUnpool",
    # Hierarchy
    "HierarchicalGraph",
    "build_hierarchical_graphs",
    "compute_transition_matrix",
    "restrict_to_coarse",
    "prolong_to_fine",
    # Processors
    "GraphUNetProcessor",
    "MGKNProcessor",
    # Spectral
    "MultiResolutionFNOBlock",
    "UFNOBlock",
    "HierarchicalFNOBlock",
    "MiniUNet",
]
