"""
Structural typing surface for the GNN-PDE framework.

This module exposes lightweight type aliases for the graph stages and re-exports
the conditioning primitives.  The component contracts themselves are ABCs that
live in :mod:`gnn_pde_v2.components`.

All graph-world hints work with :class:`~gnn_pde_v2.core.GraphsTuple`.

Two tiers of type surface live here:

**Stage hints** (``GraphEncoder``, ``GraphProcessor``, ``NodeDecoder``,
``QueryDecoder``) describe the *role* a module plays in the
:class:`~gnn_pde_v2.models.EncodeProcessDecode` pipeline.  They are plain
``Callable`` type aliases used purely as readable annotations for that one
class — they carry no runtime contract.  Earlier ``Protocol`` versions could
not distinguish the stages (every ``nn.Module`` matched a single-method
protocol), so they were demoted to aliases.  Code that needs to branch on a
component's role at runtime must use an explicit discriminator — for example
:class:`~gnn_pde_v2.models.EncodeProcessDecode` dispatches on the decoder's
``is_query_decoder`` class attribute, not on ``isinstance``.

**Component contracts** (``EdgeMessageProcessor``, ``NodeUpdateStrategy``,
``EdgeFeatureAssembler``) are *nominal* abstract base classes, not structural
protocols.  They live next to their concrete implementations in
:mod:`gnn_pde_v2.components` (``edge_processors``, ``node_updaters``,
``edge_assemblers``) — there is exactly one mechanism (the ABC) per concept.
Their runtime enforcement comes from inheritance (instantiation fails unless
``forward``/``weight_out_dim`` are implemented) plus the construction-time
validators in :mod:`gnn_pde_v2.components.processor_validators`.

Conditioning primitives (``Modulation``, ``ConditioningProtocol``) are
**nominal** ABCs, not structural protocols.  They live in
:mod:`gnn_pde_v2.core.conditioning` and are merely re-exported here for
backwards compatibility.
"""

from __future__ import annotations

from typing import Callable

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
# Graph-world stage hints.
#
# Plain ``Callable`` type aliases used only as annotations for
# :class:`~gnn_pde_v2.models.EncodeProcessDecode`.  They are NOT runtime
# checkable and carry no enforceable contract; role dispatch uses explicit
# discriminators (e.g. the ``is_query_decoder`` class attribute).
# ---------------------------------------------------------------------------

#: Encodes a raw graph into a latent graph (``GraphsTuple → GraphsTuple``).
GraphEncoder = Callable[[GraphsTuple], GraphsTuple]

#: Evolves a latent graph representation (``GraphsTuple → GraphsTuple``).
GraphProcessor = Callable[[GraphsTuple], GraphsTuple]

#: Decodes node-level predictions (``GraphsTuple → Tensor``).
NodeDecoder = Callable[[GraphsTuple], Tensor]

#: Decodes at arbitrary query positions (``(graph, query_positions, ...) → Tensor``).
QueryDecoder = Callable[..., Tensor]


__all__ = [
    # Conditioning (re-exported from gnn_pde_v2.core.conditioning for compat)
    "CondT",
    "Modulation",
    "ConditioningProtocol",
    # Graph-world stage hints (Callable aliases, NOT runtime contracts)
    "GraphEncoder",
    "GraphProcessor",
    "NodeDecoder",
    "QueryDecoder",
]
