"""
Backward compatibility for encoders.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.components.encoders instead.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.encoders is deprecated. "
    "Use gnn_pde_v2.components.encoders instead. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
from ..components.encoders import MLP, make_mlp_encoder

try:
    from ..components.encoders import MLPEncoder, MLPMeshEncoder
except ImportError:
    MLPEncoder = MLPMeshEncoder = None

__all__ = [
    "MLP",
    "make_mlp_encoder",
    "MLPEncoder",
    "MLPMeshEncoder",
]
