"""
Backward compatibility for layers.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.components.layers instead.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.layers is deprecated. "
    "Use gnn_pde_v2.components.layers instead. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
from ..components.layers import Residual

# Provide aliases for old names
ResidualBlock = Residual

__all__ = [
    "Residual",
    "ResidualBlock",
]
