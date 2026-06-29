"""Protocol conformance and dispatch-correctness tests.

These tests lock in the behaviour described in
``docs/protocol_issues_2026_06.md``:

* ``EncodeProcessDecode`` dispatches on the ``is_query_decoder`` discriminator,
  **not** on ``isinstance(decoder, QueryDecoder)`` (§1, §4).
* Decoder classes declare the correct discriminator value (§1).
* The conditioning primitives are importable from their canonical home and the
  backwards-compatible re-export points to the *same* objects (§5).
* The construction-time validators for node-update strategies and edge-message
  processors actually reject malformed components (§6).
"""

import collections.abc
import inspect

import pytest
import torch
import torch.nn as nn

from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import (
    MLPDecoder,
    IndependentMLPDecoder,
    ProbeDecoder,
    GraphNetBlock,
    ConcatMLPNodeUpdater,
)
from gnn_pde_v2.models import EncodeProcessDecode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _IdentityGraphModule(nn.Module):
    """Passes a GraphsTuple through unchanged (stand-in encoder/processor)."""

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        return graph


class _StrictNodeDecoder(nn.Module):
    """A node decoder whose forward accepts *only* the graph.

    This is the exact case the broken ``isinstance(decoder, QueryDecoder)``
    dispatch used to mishandle: it would pass ``query_positions`` positionally
    and crash. With discriminator-based dispatch it must work.
    """

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        return graph.nodes


class _StubQueryDecoder(nn.Module):
    is_query_decoder = True

    def __init__(self):
        super().__init__()
        self.received_query = False

    def forward(self, graph: GraphsTuple, query_positions: torch.Tensor) -> torch.Tensor:
        self.received_query = query_positions is not None
        return query_positions


def _make_graph(n_node: int = 5, dim: int = 16) -> GraphsTuple:
    return GraphsTuple.from_flat(
        nodes=torch.randn(n_node, dim),
        edges=torch.randn(6, dim),
        receivers=torch.tensor([1, 2, 3, 0, 1, 2]),
        senders=torch.tensor([0, 1, 2, 3, 0, 1]),
        n_node=torch.tensor([n_node]),
        n_edge=torch.tensor([6]),
        positions=torch.randn(n_node, 2),
    )


# ---------------------------------------------------------------------------
# Discriminator flags (§1)
# ---------------------------------------------------------------------------

class TestDecoderDiscriminator:
    def test_node_decoders_are_not_query_decoders(self):
        assert MLPDecoder(16, 3).is_query_decoder is False
        assert IndependentMLPDecoder(16, [1, 2]).is_query_decoder is False

    def test_probe_decoder_is_query_decoder(self):
        decoder = ProbeDecoder(
            latent_dim=16,
            processor=GraphNetBlock(latent_dim=16),
            out_dim=3,
            k_nearest=3,
        )
        assert decoder.is_query_decoder is True


# ---------------------------------------------------------------------------
# EPD dispatch correctness (§1, §4)
# ---------------------------------------------------------------------------

class TestEPDDispatch:
    def test_node_decoder_never_receives_query_positions(self):
        """A strict node decoder must work even when query_positions is given."""
        epd = EncodeProcessDecode(
            encoder=_IdentityGraphModule(),
            processor=_IdentityGraphModule(),
            decoder=_StrictNodeDecoder(),
        )
        graph = _make_graph()
        # query_positions supplied, but the node decoder must NOT receive it.
        out = epd(graph, query_positions=torch.randn(4, 2))
        assert out.shape == graph.nodes.shape

    def test_mlp_decoder_runs_through_epd(self):
        epd = EncodeProcessDecode(
            encoder=_IdentityGraphModule(),
            processor=_IdentityGraphModule(),
            decoder=MLPDecoder(16, 3),
        )
        graph = _make_graph()
        out = epd(graph)
        assert out.shape == (5, 3)

    def test_query_decoder_receives_query_positions(self):
        stub = _StubQueryDecoder()
        epd = EncodeProcessDecode(
            encoder=_IdentityGraphModule(),
            processor=_IdentityGraphModule(),
            decoder=stub,
        )
        graph = _make_graph()
        query = torch.randn(4, 2)
        out = epd(graph, query_positions=query)
        assert stub.received_query is True
        assert torch.equal(out, query)

    def test_dispatch_uses_flag_not_isinstance(self):
        """The dispatch must read the discriminator attribute, not a protocol."""
        src = inspect.getsource(EncodeProcessDecode.forward)
        assert "is_query_decoder" in src
        assert "isinstance(self.decoder, QueryDecoder)" not in src


# ---------------------------------------------------------------------------
# Conditioning relocation (§5)
# ---------------------------------------------------------------------------

class TestConditioningRelocation:
    def test_canonical_and_compat_imports_are_identical(self):
        from gnn_pde_v2.core.conditioning import (
            ConditioningProtocol as CanonCond,
            Modulation as CanonMod,
        )
        from gnn_pde_v2.core.protocols import (
            ConditioningProtocol as CompatCond,
            Modulation as CompatMod,
        )
        assert CanonCond is CompatCond
        assert CanonMod is CompatMod

    def test_node_update_strategy_no_longer_a_protocol(self):
        """Issue #3: NodeUpdateStrategy is now an ABC in components, not a protocol."""
        from gnn_pde_v2.core import protocols
        assert "NodeUpdateStrategy" not in protocols.__all__
        assert not hasattr(protocols, "NodeUpdateStrategy")


# ---------------------------------------------------------------------------
# Construction-time validators (§6)
# ---------------------------------------------------------------------------

class TestNodeUpdaterValidation:
    def test_valid_node_updater_accepted(self):
        block = GraphNetBlock(latent_dim=16, node_updater=ConcatMLPNodeUpdater(16))
        assert block.node_updater.latent_dim == 16

    def test_mismatched_latent_dim_rejected(self):
        with pytest.raises(ValueError, match="latent_dim"):
            GraphNetBlock(latent_dim=16, node_updater=ConcatMLPNodeUpdater(32))

    def test_non_module_node_updater_rejected(self):
        from gnn_pde_v2.components.processor_validators import (
            validate_node_update_strategy,
        )

        class _FakeUpdater:
            latent_dim = 16

            def forward(self, nodes, aggregated):
                return nodes

        with pytest.raises(TypeError, match="nn.Module"):
            validate_node_update_strategy(_FakeUpdater(), 16)

    def test_non_int_latent_dim_rejected(self):
        from gnn_pde_v2.components.processor_validators import (
            validate_node_update_strategy,
        )

        class _BadUpdater(nn.Module):
            latent_dim = "banana"

            def forward(self, nodes, aggregated):
                return nodes

        with pytest.raises(ValueError, match="positive int"):
            validate_node_update_strategy(_BadUpdater(), 16)


# ---------------------------------------------------------------------------
# Issue #4: structural stage protocols are NOT runtime_checkable.
# They carried no enforceable contract (signatures are unchecked) and were
# decorative. Issue #3: component-contract types (EdgeMessageProcessor,
# NodeUpdateStrategy, EdgeFeatureAssembler) are now plain ABCs in
# gnn_pde_v2.components — a single mechanism per concept. The real graph-stage
# dispatch uses is_query_decoder.
# ---------------------------------------------------------------------------

class TestProtocolsNotRuntimeCheckable:
    """Lock in the issue #4 remediation so it cannot silently regress."""

    @pytest.mark.parametrize(
        "name",
        [
            "GraphEncoder",
            "GraphProcessor",
            "NodeDecoder",
            "QueryDecoder",
        ],
    )
    def test_graph_stage_hints_are_callable_aliases(self, name):
        """Issue #3 option 2: stage hints are plain Callable aliases, not protocols."""
        import typing
        from gnn_pde_v2.core import protocols

        alias = getattr(protocols, name)
        assert typing.get_origin(alias) is collections.abc.Callable
        assert not hasattr(alias, "_is_protocol")

    def test_graph_model_removed(self):
        from gnn_pde_v2.core import protocols

        assert not hasattr(protocols, "GraphModel")
        assert "GraphModel" not in protocols.__all__

    @pytest.mark.parametrize(
        "name",
        ["EdgeMessageProcessor", "NodeUpdateStrategy", "EdgeFeatureAssembler"],
    )
    def test_component_contracts_are_single_abc(self, name):
        """Issue #3: each contract is a single ABC in components, not a protocol."""
        import abc
        from gnn_pde_v2 import components
        from gnn_pde_v2.core import protocols

        contract = getattr(components, name)
        assert isinstance(contract, abc.ABCMeta), f"{name} should be an ABC"
        assert not hasattr(contract, "_is_protocol"), f"{name} must not be a Protocol"
        assert name in components.__all__
        # The colliding protocol must no longer exist in core.protocols.
        assert not hasattr(protocols, name), f"{name} protocol must be removed"

    def test_decoder_union_removed(self):
        from gnn_pde_v2.core import protocols

        assert not hasattr(protocols, "Decoder")

    def test_grid_trio_removed(self):
        from gnn_pde_v2.core import protocols

        for name in ("PositionEncoder", "GridProcessor", "GridModel"):
            assert not hasattr(protocols, name), f"{name} should be deleted"

    def test_removed_names_absent_from_core_all(self):
        from gnn_pde_v2 import core

        for name in ("Decoder", "PositionEncoder", "GridProcessor", "GridModel"):
            assert name not in core.__all__, f"{name} should not be exported"

    def test_removed_names_absent_from_components_all(self):
        from gnn_pde_v2 import components

        for name in ("Decoder", "PositionEncoder", "GridProcessor", "GridModel"):
            assert name not in components.__all__, f"{name} should not be exported"
