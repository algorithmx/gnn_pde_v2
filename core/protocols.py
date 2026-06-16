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

.. warning:: Runtime-checking limitations

    ``@runtime_checkable`` only verifies that the named **methods exist**;
    it does **not** inspect signatures, return types, or the types of
    declared data/property members (see the CPython docs for
    ``typing.runtime_checkable``).  Because every ``nn.Module`` defines a
    ``forward`` method, **every** ``nn.Module`` satisfies every single-method
    protocol in this file.  Concretely:

    - ``GraphEncoder``, ``GraphProcessor``, ``NodeDecoder`` and ``GraphModel``
      are **indistinguishable** at runtime — ``isinstance`` cannot tell a
      decoder apart from an encoder.  The same applies to the grid-world
      ``PositionEncoder`` / ``GridProcessor`` / ``GridModel`` trio (even
      ``nn.ReLU()`` satisfies all three).
    - The "stage" a module occupies in a pipeline (encode vs. process vs.
      decode) is a *positional* fact about where it is wired, not a
      *structural* property these protocols can carry.

    Therefore these protocols are useful as **documentation and static-typing
    hints**, but must **not** be used as runtime discriminators.  Code that
    needs to branch on a component's role must use an explicit discriminator
    (e.g. ``EncodeProcessDecode`` dispatches on the decoder's
    ``is_query_decoder`` class attribute, not on ``isinstance``).  For data /
    property members (``EdgeMessageProcessor.weight_out_dim``,
    ``NodeUpdateStrategy.latent_dim``) the real contract is enforced by the
    validators in :mod:`gnn_pde_v2.components.processor_validators`, which the
    consuming blocks call at construction time.

Conditioning primitives (``Modulation``, ``ConditioningProtocol``) are
**nominal** ABCs, not structural protocols.  They now live in
:mod:`gnn_pde_v2.core.conditioning` and are merely re-exported here for
backwards compatibility (see ``docs/protocol_issues_2026_06.md`` §5).
"""

from __future__ import annotations

from typing import Optional, Protocol, Union, runtime_checkable

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
# Graph-world protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphEncoder(Protocol):
    """Protocol for modules that encode a raw graph into a latent graph.

    Satisfied by any module whose ``forward`` maps
    ``GraphsTuple → GraphsTuple``.

    .. warning::

        This protocol is structurally identical to :class:`GraphProcessor`,
        :class:`NodeDecoder` and :class:`GraphModel` at runtime — see the
        module docstring.  Use it as a static-typing hint only, never as an
        ``isinstance`` discriminator.
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

    Example::

        from gnn_pde_v2.core.protocols import EdgeFeatureAssembler
        from gnn_pde_v2.components import NodeDifferenceAssembler

        assembler: EdgeFeatureAssembler = NodeDifferenceAssembler(128)
        assert isinstance(assembler, EdgeFeatureAssembler)  # True at runtime
    """

    @property
    def out_dim(self) -> int: ...

    def forward(self, graph: GraphsTuple) -> Tensor: ...


@runtime_checkable
class NodeDecoder(Protocol):
    """Protocol for decoders that output at fixed node positions.

    These decoders operate on the graph's existing nodes and do not require
    arbitrary query positions. Examples include MLPDecoder and IndependentMLPDecoder.

    Implementations set ``is_query_decoder = False`` (the default) so that
    :class:`~gnn_pde_v2.models.EncodeProcessDecode` never forwards
    ``query_positions`` to them.  Note that ``@runtime_checkable`` cannot
    distinguish this protocol from :class:`QueryDecoder` or even
    :class:`GraphEncoder` at runtime (see the module docstring); it is a
    static-typing hint only.

    Example::

        from gnn_pde_v2.components import MLPDecoder

        assert MLPDecoder(128, 3).is_query_decoder is False
    """

    def forward(self, graph: GraphsTuple) -> Tensor: ...


@runtime_checkable
class QueryDecoder(Protocol):
    """Protocol for decoders that output at arbitrary query positions.

    These decoders require explicit ``query_positions`` to determine where in
    space to make predictions.  The canonical implementation is
    :class:`~gnn_pde_v2.components.ProbeDecoder`, whose ``forward`` also accepts
    optional ``query_features`` and ``n_query`` arguments; they are part of the
    real contract and are therefore declared here as optional parameters.

    Implementations must set the class attribute ``is_query_decoder = True`` so
    that :class:`~gnn_pde_v2.models.EncodeProcessDecode` forwards
    ``query_positions`` to them (``@runtime_checkable`` cannot distinguish this
    protocol from :class:`NodeDecoder` — see the module docstring).

    Example::

        from gnn_pde_v2.core.protocols import QueryDecoder
        from gnn_pde_v2.components import ProbeDecoder

        assert ProbeDecoder(128, out_dim=3).is_query_decoder is True
    """

    is_query_decoder: bool

    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Tensor,
        query_features: Optional[Tensor] = ...,
        n_query: Optional[Tensor] = ...,
    ) -> Tensor: ...


# Backwards-compatible type alias.
#
# .. deprecated::
#     ``Decoder`` is a ``Union`` of two ``@runtime_checkable`` protocols, which
#     means ``isinstance(x, Decoder)`` degrades to "has a ``forward`` method"
#     and is therefore useless as a runtime discriminator (see
#     ``docs/protocol_issues_2026_06.md`` §4).  It is retained only so that
#     existing imports keep working.  New code should annotate decoders with
#     ``NodeDecoder`` or ``QueryDecoder`` explicitly and dispatch on the
#     ``is_query_decoder`` class attribute, the way ``EncodeProcessDecode``
#     does.
Decoder = Union[NodeDecoder, QueryDecoder]


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
#
# .. warning::
#     As with the graph-world protocols, these three are structurally
#     identical at runtime: every ``Tensor -> Tensor`` ``nn.Module`` (even
#     ``nn.ReLU()``) satisfies all of ``PositionEncoder``, ``GridProcessor``
#     and ``GridModel``.  Treat them as static-typing / documentation hints,
#     not as ``isinstance`` discriminators.
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
    :class:`~gnn_pde_v2.models.TFNO`, :class:`~gnn_pde_v2.models.AFNO`,
    and any ``Tensor → Tensor`` model.
    """

    def forward(self, x: Tensor) -> Tensor: ...


__all__ = [
    # Conditioning (re-exported from gnn_pde_v2.core.conditioning for compat)
    "CondT",
    "Modulation",
    "ConditioningProtocol",
    # Graph-world
    "GraphEncoder",
    "GraphProcessor",
    "EdgeMessageProcessor",
    "NodeUpdateStrategy",
    "EdgeFeatureAssembler",
    "NodeDecoder",
    "QueryDecoder",
    "Decoder",  # Backwards-compatible alias (deprecated)
    "GraphModel",
    # Grid-world
    "PositionEncoder",
    "GridProcessor",
    "GridModel",
]
