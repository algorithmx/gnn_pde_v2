"""
Backward compatibility for decoders.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.components.decoders instead.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.decoders is deprecated. "
    "Use gnn_pde_v2.components.decoders instead. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
from ..components.decoders import MLPDecoder, IndependentMLPDecoder
from ..components.probe import ProbeDecoder

__all__ = [
    "MLPDecoder",
    "IndependentMLPDecoder",
    "ProbeDecoder",
]
