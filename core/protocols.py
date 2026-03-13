"""
Structural protocols for the GNN-PDE framework.

These protocols define the interfaces that components must implement.
Using ``typing.Protocol`` allows structural (duck-type) checking without
requiring inheritance, making the system open for extension.

All graph-world protocols work with :class:`~gnn_pde_v2.core.GraphsTuple`.
All grid-world protocols work with plain :class:`torch.Tensor`.

Protocols are ``runtime_checkable``, so ``isinstance(obj, GraphProcessor)``
works at runtime in addition to static type checking::

    from gnn_pde_v2.core.protocols import GraphProcessor
    assert isinstance(my_processor, GraphProcessor)  # True if forward matches

Conditioning primitives (``Modulation``, ``ConditioningProtocol``) are also
defined here and re-exported through ``components.transformer`` for
backwards compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from .graph import GraphsTuple


# ---------------------------------------------------------------------------
# Conditioning primitives
# (originally in components/transformer.py, centralised here)
# ---------------------------------------------------------------------------

@dataclass
class Modulation:
    """Container for transformer modulation parameters.

    All fields are optional; absent fields mean no modulation is applied
    for that axis.

    Args:
        shift: Additive bias term  ``[..., dim]``
        scale: Multiplicative scale term ``[..., dim]``
        gate: Post-residual gate term ``[..., dim]``
        cross_kv: Pre-computed key/value tensors for cross-attention
    """
    shift: Tensor | None = None
    scale: Tensor | None = None
    gate: Tensor | None = None
    cross_kv: Tensor | None = None


class ConditioningProtocol(nn.Module, ABC):
    """Abstract base class for conditioning mechanisms.

    All conditioning modules should inherit from this class and implement
    :meth:`forward`, which maps an arbitrary condition (scalar, tensor,
    or embedding) to a :class:`Modulation`.

    Example::

        class MyConditioning(ConditioningProtocol):
            def __init__(self, cond_dim: int, out_dim: int):
                super().__init__()
                self.proj = nn.Linear(cond_dim, out_dim * 2)

            def forward(self, condition: Tensor) -> Modulation:
                shift, scale = self.proj(condition).chunk(2, dim=-1)
                return Modulation(shift=shift, scale=scale)
    """

    @abstractmethod
    def forward(self, condition: Any) -> Modulation:
        """Convert condition to modulation parameters.

        Args:
            condition: Conditioning input (type depends on implementation)

        Returns:
            Modulation with optional shift/scale/gate/cross_kv fields
        """
        ...


# ---------------------------------------------------------------------------
# Graph-world protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphEncoder(Protocol):
    """Protocol for modules that encode a raw graph into a latent graph.

    Satisfied by any module whose ``forward`` maps
    ``GraphsTuple → GraphsTuple``.

    Example::

        from gnn_pde_v2.core.protocols import GraphEncoder
        from gnn_pde_v2.components import MeshEncoder

        enc: GraphEncoder = MeshEncoder(node_in_dim=11, edge_in_dim=3,
                                        global_in_dim=None, latent_dim=128)
    """

    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...


@runtime_checkable
class GraphProcessor(Protocol):
    """Protocol for modules that evolve a latent graph representation.

    Any module whose ``forward`` maps ``GraphsTuple → GraphsTuple`` satisfies
    this protocol, including :class:`~gnn_pde_v2.components.GraphNetProcessor`,
    :class:`~gnn_pde_v2.components.TransformerProcessor`, and custom blocks,
    without any inheritance required.

    Example::

        from gnn_pde_v2.core.protocols import GraphProcessor
        from gnn_pde_v2.components import GraphNetProcessor

        proc: GraphProcessor = GraphNetProcessor(latent_dim=128, n_layers=6)
        assert isinstance(proc, GraphProcessor)  # True at runtime
    """

    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...


@runtime_checkable
class Decoder(Protocol):
    """Protocol for modules that decode a processed graph into a tensor.

    ``query_positions`` is optional:

    - Decoders that output at fixed node positions (e.g.
      :class:`~gnn_pde_v2.components.MLPDecoder`) may ignore it.
    - Probe-based decoders (e.g.
      :class:`~gnn_pde_v2.components.ProbeDecoder`) require it and raise
      ``ValueError`` when it is ``None``.

    All implementations **must** accept ``query_positions`` as an optional
    keyword argument so that generic pipelines (e.g.
    :class:`~gnn_pde_v2.models.EncodeProcessDecode`) can call them uniformly.

    Example::

        from gnn_pde_v2.core.protocols import Decoder
        from gnn_pde_v2.components import MLPDecoder, ProbeDecoder

        assert isinstance(MLPDecoder(128, 3), Decoder)
        assert isinstance(ProbeDecoder(128, out_dim=3), Decoder)
    """

    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[Tensor] = None,
    ) -> Tensor: ...


@runtime_checkable
class GraphModel(Protocol):
    """Protocol for end-to-end models that map a graph to a tensor.

    Satisfied by :class:`~gnn_pde_v2.models.GraphNet`,
    :class:`~gnn_pde_v2.models.MeshGraphNet`, and any custom model whose
    ``forward`` accepts a :class:`~gnn_pde_v2.core.GraphsTuple`.
    """

    def forward(self, graph: GraphsTuple) -> Tensor: ...


# ---------------------------------------------------------------------------
# Grid-world protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class PositionEncoder(Protocol):
    """Protocol for modules that encode raw coordinates into feature vectors.

    Satisfied by :class:`~gnn_pde_v2.components.FourierFeatureEncoder` and
    any linear projection that maps ``[..., spatial_dim] → [..., feat_dim]``.
    """

    def forward(self, x: Tensor) -> Tensor: ...


@runtime_checkable
class GridProcessor(Protocol):
    """Protocol for modules that process regular grid/tensor representations.

    Satisfied by spectral layers
    (:class:`~gnn_pde_v2.components.FNOBlock`,
    :class:`~gnn_pde_v2.components.AFNOBlock`), the full
    :class:`~gnn_pde_v2.models.FNO` model, and any ``Tensor → Tensor``
    transformation.
    """

    def forward(self, x: Tensor) -> Tensor: ...


@runtime_checkable
class GridModel(Protocol):
    """Protocol for end-to-end grid-to-grid models.

    Satisfied by :class:`~gnn_pde_v2.models.FNO`,
    :class:`~gnn_pde_v2.models.TFNO`, :class:`~gnn_pde_v2.models.FNOPlus`,
    and any ``Tensor → Tensor`` model.
    """

    def forward(self, x: Tensor) -> Tensor: ...


__all__ = [
    # Conditioning
    "Modulation",
    "ConditioningProtocol",
    # Graph-world
    "GraphEncoder",
    "GraphProcessor",
    "Decoder",
    "GraphModel",
    # Grid-world
    "PositionEncoder",
    "GridProcessor",
    "GridModel",
]
