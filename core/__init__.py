"""Core components for the GNN-PDE framework (minimal)."""

from .graph import GraphsTuple, batch_graphs, unbatch_graphs
from .base import BaseModel
from .functional import (
    scatter_sum,
    scatter_mean,
    scatter_max,
    scatter_min,
    scatter_softmax,
    aggregate_edges,
    broadcast_nodes_to_edges,
)
from .mlp import MLP, SinActivation
from .registry import AutoRegisterModel

__all__ = [
    "GraphsTuple",
    "batch_graphs",
    "unbatch_graphs",
    "BaseModel",
    "scatter_sum",
    "scatter_mean",
    "scatter_max",
    "scatter_min",
    "scatter_softmax",
    "aggregate_edges",
    "broadcast_nodes_to_edges",
    "MLP",
    "SinActivation",
    "AutoRegisterModel",
]
