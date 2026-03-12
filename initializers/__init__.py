"""
Backward compatibility for initializers.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.convenient.initializers instead, or use torch.nn.init directly.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.initializers is deprecated. "
    "Use gnn_pde_v2.convenient.initializers for string-based initialization, "
    "or torch.nn.init directly for functional initialization. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
try:
    from ..convenient.initializers import (
        get_initializer,
        initialize_module,
    )
except ImportError:
    get_initializer = None
    initialize_module = None

__all__ = [
    "get_initializer",
    "initialize_module",
]
