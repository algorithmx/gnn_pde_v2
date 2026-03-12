"""
Backward compatibility for aggregation.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.core.functional for essential operations, or
gnn_pde_v2.convenient.aggregation for full API.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.utils.aggregation is deprecated. "
    "Use gnn_pde_v2.core.functional for essential operations (scatter_sum, scatter_mean), "
    "or gnn_pde_v2.convenient.aggregation for full API. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new locations
from ..core.functional import (
    scatter_sum,
    scatter_mean,
    scatter_max,
    aggregate_edges,
    broadcast_nodes_to_edges,
)

try:
    from ..convenient.aggregation import (
        scatter_min,
        scatter_softmax,
        segment_sum,
        segment_mean,
        segment_max,
        segment_min,
    )
except ImportError:
    scatter_min = scatter_softmax = None
    segment_sum = segment_mean = segment_max = segment_min = None

__all__ = [
    'scatter_sum',
    'scatter_mean',
    'scatter_max',
    'scatter_min',
    'scatter_softmax',
    'aggregate_edges',
    'broadcast_nodes_to_edges',
    'segment_sum',
    'segment_mean',
    'segment_max',
    'segment_min',
]
