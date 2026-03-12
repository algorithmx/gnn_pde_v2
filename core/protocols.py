"""
Backward compatibility shim for protocol definitions.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Python duck-typing makes explicit protocols unnecessary.
Simply implement the required methods directly.
"""

import warnings
from typing import Protocol, Optional, Tuple, runtime_checkable
from torch import Tensor

warnings.warn(
    "gnn_pde_v2.core.protocols is deprecated. "
    "Use Python duck-typing instead of explicit protocols. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)


@runtime_checkable
class EncoderProtocol(Protocol):
    """DEPRECATED: Just implement forward(graph) -> graph."""
    pass


@runtime_checkable
class MessagePassingOutput(Protocol):
    """DEPRECATED: Use tuple or dataclass instead."""
    pass


@runtime_checkable
class MessagePassingBlock(Protocol):
    """DEPRECATED: Just implement forward(nodes, edges, edge_index)."""
    pass


@runtime_checkable
class ProcessorProtocol(Protocol):
    """DEPRECATED: Just implement forward(graph) -> graph."""
    pass


@runtime_checkable
class DecoderProtocol(Protocol):
    """DEPRECATED: Just implement forward(graph, query_positions)."""
    pass
