"""
Backward compatibility for processors.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.components.processors instead.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.processors is deprecated. "
    "Use gnn_pde_v2.components.processors instead. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
from ..components.processors import GraphNetBlock, GraphNetProcessor
from ..components.transformer import TransformerBlock, TransformerProcessor
from ..components.fno import FNOBlock, FNOProcessor

__all__ = [
    "GraphNetBlock",
    "GraphNetProcessor",
    "TransformerBlock",
    "TransformerProcessor",
    "FNOBlock",
    "FNOProcessor",
]
