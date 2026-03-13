"""
GNN-PDE v2: Unified PDE-GNN Framework

A clean implementation of the Encode-Process-Decode architecture
with modular, composable components.

Usage:
    # Lean core (recommended for research)
    from gnn_pde_v2 import GraphsTuple, BaseModel
    from gnn_pde_v2.core import MLP
    from gnn_pde_v2.components import GraphNetBlock, Residual

    # Convenient API (optional, for quick experimentation)
    from gnn_pde_v2.convenient import (
        AutoRegisterModel, ModelConfig, ConfigBuilder, Model
    )

Version: 2.1.0
"""

__version__ = "2.1.0"

# Core exports (always available)
from .core.graph import GraphsTuple, batch_graphs, unbatch_graphs
from .core.base import BaseModel
from .core.functional import (
    scatter_sum,
    scatter_mean,
    scatter_max,
    aggregate_edges,
    broadcast_nodes_to_edges,
)

__all__ = [
    # Core data structures
    "GraphsTuple",
    "batch_graphs",
    "unbatch_graphs",
    "BaseModel",
    # Core functional
    "scatter_sum",
    "scatter_mean",
    "scatter_max",
    "aggregate_edges",
    "broadcast_nodes_to_edges",
]
