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
from .aggregation import (
    Aggregation,
    Sum,
    Mean,
    Max,
    Min,
    get_aggregation,
)
from .mlp import MLP, SinActivation
from .registry import AutoRegisterModel, MODEL_REGISTRY
from .protocols import (
    Modulation,
    ConditioningProtocol,
    GraphEncoder,
    GraphProcessor,
    NodeDecoder,
    QueryDecoder,
    Decoder,
    GraphModel,
    PositionEncoder,
    GridProcessor,
    GridModel,
)

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
    # Aggregation
    "Aggregation",
    "Sum",
    "Mean",
    "Max",
    "Min",
    "get_aggregation",
    "MLP",
    "SinActivation",
    "AutoRegisterModel",
    "MODEL_REGISTRY",
    # Protocols
    "Modulation",
    "ConditioningProtocol",
    "GraphEncoder",
    "GraphProcessor",
    "NodeDecoder",
    "QueryDecoder",
    "Decoder",
    "GraphModel",
    "PositionEncoder",
    "GridProcessor",
    "GridModel",
]
