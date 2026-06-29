"""
Structural protocols for the GNN-PDE framework.

These protocols define the interfaces that components implement.  Using
``typing.Protocol`` allows structural (duck-type) checking by static type
checkers (mypy / pyright) without requiring inheritance, making the system open
for extension.

All graph-world protocols work with :class:`~gnn_pde_v2.core.GraphsTuple`.

Two tiers of protocol live here:

**Stage protocols** (``GraphEncoder``, ``GraphProcessor``, ``NodeDecoder``,
``QueryDecoder``, ``GraphModel``) describe the *role* a module plays in the
:class:`~gnn_pde_v2.models.EncodeProcessDecode` pipeline.  They are **plain
``Protocol`` classes and deliberately NOT ``runtime_checkable``**: because
``@runtime_checkable`` only verifies that a method *name* exists (it cannot
inspect signatures, return types, or distinguish ``GraphsTuple`` from
``Tensor``), every ``nn.Module`` would satisfy every single-method stage
protocol — even ``nn.ReLU()``.  Such "runtime checks" carried no enforceable
contract and were removed (see ``docs/remediation-plan-issue4-structural-protocols.md``).
Use these protocols as **static-typing hints only**.  Code that needs to branch
on a component's role at runtime must use an explicit discriminator — for
example :class:`~gnn_pde_v2.models.EncodeProcessDecode` dispatches on the
decoder's ``is_query_decoder`` class attribute, not on ``isinstance``.

**Component contracts** (``EdgeMessageProcessor``, ``NodeUpdateStrategy``,
``EdgeFeatureAssembler``) declare data/property members in addition to
``forward`` and remain ``@runtime_checkable``.  Their real runtime enforcement,
however, comes from the construction-time validators in
:mod:`gnn_pde_v2.components.processor_validators` (which check ``nn.Module``
membership and ``latent_dim`` type/value), not from ``isinstance`` alone.

Conditioning primitives (``Modulation``, ``ConditioningProtocol``) are
**nominal** ABCs, not structural protocols.  They live in
:mod:`gnn_pde_v2.core.conditioning` and are merely re-exported here for
backwards compatibility.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from torch import Tensor

from .graph import GraphsTuple

# ---------------------------------------------------------------------------
# Conditioning primitives — re-exported for backwards compatibility.
# These are *nominal* (inheritance-based) types and live in
# ``gnn_pde_v2.core.conditioning`` so that this module can remain a pure
# *structural* protocol surface.  Import them from their canonical home in new
# code: ``from gnn_pde_v2.core.conditioning import Modulation, ...``.
# ---------------------------------------------------------------------------
from .conditioning import CondT, ConditioningProtocol, Modulation  # noqa: F401


# ---------------------------------------------------------------------------
# Graph-world stage protocols.
#
# These are plain ``Protocol`` classes (NOT ``runtime_checkable``).  They serve
# as static-typing hints for :class:`~gnn_pde_v2.models.EncodeProcessDecode`.
# Runtime dispatch on a component's role must use explicit discriminators
# (e.g. the ``is_query_decoder`` class attribute), never ``isinstance``.
# ---------------------------------------------------------------------------

class GraphEncoder(Protocol):
    """Protocol for modules that encode a raw graph into a latent graph.

    Satisfied by any module whose ``forward`` maps
    ``GraphsTuple → GraphsTuple``.  Static-typing hint only; not
    ``runtime_checkable``.
    """

    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...


class GraphProcessor(Protocol):
    """Protocol for modules that evolve a latent graph representation.

    Any module whose ``forward`` maps ``GraphsTuple → GraphsTuple`` satisfies
    this protocol at the type-checker level, including
    :class:`~gnn_pde_v2.components.GraphNetProcessor`,
    :class:`~gnn_pde_v2.components.TransformerProcessor`, and custom blocks,
    without any inheritance required.  Static-typing hint only; not
    ``runtime_checkable``.
    """

    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...


@runtime_checkable
class EdgeMessageProcessor(Protocol):
    """Protocol for edge-conditioned message transforms.

    Implementations are expected to be used as fixed submodules inside
    :class:`~gnn_pde_v2.components.EdgeConditionedConvBlock`. They receive
    sender node features and the per-edge weights produced by an external
    network, then return transformed messages of shape ``[E, latent_dim]``.

    Notes:
        - The protocol is intentionally minimal for compile-friendliness.
        - Implementations should also be ``nn.Module`` instances in practice,
          so they participate correctly in module registration and
          ``torch.compile()`` specialization.
        - Shape correctness is verified at block construction time via a
          full pipeline check; processors do not need to perform their own
          runtime shape assertions.
    """

    latent_dim: int

    @property
    def weight_out_dim(self) -> int: ...

    def forward(self, src_x: Tensor, edge_weights: Tensor) -> Tensor: ...


@runtime_checkable
class NodeUpdateStrategy(Protocol):
    """Protocol for composable node-update strategies in message passing.

    Implementations transform a node's current features and its aggregated
    incoming messages into updated node features.  They are injected into
    :class:`~gnn_pde_v2.components.processors.MessagePassingBase` subclasses
    so that the node-update rule can be swapped without touching the rest of
    the message-passing logic.

    Notes:
        - Implementations should also be ``nn.Module`` instances so they
          participate in parameter registration and ``torch.compile()``.
        - The protocol deliberately takes only ``nodes`` and ``aggregated``
          (not the full graph) to stay compile-friendly and simple.
    """

    latent_dim: int

    def forward(self, nodes: Tensor, aggregated: Tensor) -> Tensor: ...


@runtime_checkable
class EdgeFeatureAssembler(Protocol):
    """Protocol for edge feature assembly strategies.

    Implementations define how to construct per-edge feature vectors from
    graph structure for use in :class:`~gnn_pde_v2.components.EdgeConvBlock`.

    Satisfied by
    :class:`~gnn_pde_v2.components.NodeDifferenceAssembler`,
    :class:`~gnn_pde_v2.components.ConcatAssembler`, and other assembler
    classes in :mod:`~gnn_pde_v2.components.edge_assemblers`.

    The protocol requires:
    - An :attr:`out_dim` property returning the output feature dimension
    - A ``forward(graph: GraphsTuple) -> Tensor`` method that assembles edge features
    """

    @property
    def out_dim(self) -> int: ...

    def forward(self, graph: GraphsTuple) -> Tensor: ...


class NodeDecoder(Protocol):
    """Protocol for decoders that output at fixed node positions.

    These decoders operate on the graph's existing nodes and do not require
    arbitrary query positions. Examples include MLPDecoder and IndependentMLPDecoder.

    Implementations set ``is_query_decoder = False`` (the default) so that
    :class:`~gnn_pde_v2.models.EncodeProcessDecode` never forwards
    ``query_positions`` to them.  Static-typing hint only; not
    ``runtime_checkable`` — runtime dispatch uses the ``is_query_decoder``
    class attribute.
    """

    def forward(self, graph: GraphsTuple) -> Tensor: ...


class QueryDecoder(Protocol):
    """Protocol for decoders that output at arbitrary query positions.

    These decoders require explicit ``query_positions`` to determine where in
    space to make predictions.  The canonical implementation is
    :class:`~gnn_pde_v2.components.ProbeDecoder`, whose ``forward`` also accepts
    optional ``query_features`` and ``n_query`` arguments; they are part of the
    real contract and are therefore declared here as optional parameters.

    Implementations must set the class attribute ``is_query_decoder = True`` so
    that :class:`~gnn_pde_v2.models.EncodeProcessDecode` forwards
    ``query_positions`` to them.  Static-typing hint only; not
    ``runtime_checkable`` — runtime dispatch uses the ``is_query_decoder``
    class attribute.
    """

    is_query_decoder: bool

    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Tensor,
        query_features: Optional[Tensor] = ...,
        n_query: Optional[Tensor] = ...,
    ) -> Tensor: ...


class GraphModel(Protocol):
    """Protocol for end-to-end models that map a graph to a tensor.

    Satisfied by :class:`~gnn_pde_v2.models.GraphNet`,
    :class:`~gnn_pde_v2.models.MeshGraphNet`, and any custom model whose
    ``forward`` accepts a :class:`~gnn_pde_v2.core.GraphsTuple`.
    Static-typing hint only; not ``runtime_checkable``.
    """

    def forward(self, graph: GraphsTuple) -> Tensor: ...


__all__ = [
    # Conditioning (re-exported from gnn_pde_v2.core.conditioning for compat)
    "CondT",
    "Modulation",
    "ConditioningProtocol",
    # Graph-world stage protocols (static-typing hints, NOT runtime_checkable)
    "GraphEncoder",
    "GraphProcessor",
    "NodeDecoder",
    "QueryDecoder",
    "GraphModel",
    # Component contracts (runtime_checkable; enforced by validators)
    "EdgeMessageProcessor",
    "NodeUpdateStrategy",
    "EdgeFeatureAssembler",
]
