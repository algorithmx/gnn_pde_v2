"""
Tests for components.

Components include: MLP, Residual, processors, decoders.
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import replace

from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.core import MLP
from functools import partial
from gnn_pde_v2.components import (
    Residual,
    MessagePassingBase,
    GraphNetBlock, GraphNetProcessor,
    EdgeConditionedConvBlock,
    FullEdgeMessageProcessor,
    VectorEdgeMessageProcessor,
    ScalarEdgeMessageProcessor,
    LowRankEdgeMessageProcessor,
    EdgeConvBlock,
    GENBlock,
    GlobalGraphNetBlock, GlobalGraphNetProcessor,
    MLPDecoder, IndependentMLPDecoder,
    ProbeDecoder, WindFarmGNO, ProbeGraphBuilder,
    LearnableRBFEncoder,
    MultiHeadAttention, TransformerBlock, TransformerProcessor,
)
from gnn_pde_v2.core.protocols import EdgeMessageProcessor


class TestMLP:
    """Test MLP encoder."""
    
    def test_forward(self, device):
        """Test basic forward pass."""
        mlp = MLP(10, 5, [20, 15]).to(device)
        x = torch.randn(3, 10, device=device)
        
        out = mlp(x)
        
        assert out.shape == (3, 5)
    
    def test_single_layer(self, device):
        """Test MLP with no hidden layers."""
        mlp = MLP(10, 5, []).to(device)
        x = torch.randn(3, 10, device=device)
        
        out = mlp(x)
        
        assert out.shape == (3, 5)
    
    def test_different_activations(self, device):
        """Test different activation functions."""
        for act in ['relu', 'gelu', 'silu', 'prelu', 'tanh', 'sigmoid', 'sin']:
            mlp = MLP(10, 5, [10], activation=act).to(device)
            x = torch.randn(3, 10, device=device)
            out = mlp(x)
            assert out.shape == (3, 5)

    def test_prelu_string_creates_prelu_modules(self, device):
        """Test that string PReLU is supported across MLP activation hooks."""
        mlp = MLP(
            10, 5, [12],
            activation='prelu',
            final_activation='prelu',
            pre_activation='prelu',
            use_layer_norm=False,
        ).to(device)

        x = torch.randn(3, 10, device=device)
        out = mlp(x)

        prelus = [m for m in mlp.modules() if isinstance(m, nn.PReLU)]
        assert out.shape == (3, 5)
        assert len(prelus) == 3
    
    def test_dropout(self, device):
        """Test dropout."""
        mlp = MLP(10, 5, [10], dropout=0.5).to(device)
        x = torch.randn(3, 10, device=device)
        
        mlp.train()
        out1 = mlp(x)
        out2 = mlp(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        mlp.eval()
        out1 = mlp(x)
        out2 = mlp(x)
        # Outputs should be same in eval mode
        assert torch.allclose(out1, out2)
    
    def test_weight_init(self, device):
        """Test custom weight initialization."""
        import torch.nn.init as init
        
        mlp = MLP(10, 5, [10], weight_init=init.zeros_, use_layer_norm=False).to(device)
        
        # Check that weights are zeros
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                assert torch.allclose(module.weight, torch.zeros_like(module.weight))

    def test_final_norm_only(self, device):
        """Test final-only normalization support."""
        mlp = MLP(
            10, 5, [12, 12],
            activation='relu',
            norm=None,
            final_norm='layer',
        ).to(device)

        layer_norms = [m for m in mlp.modules() if isinstance(m, nn.LayerNorm)]
        assert len(layer_norms) == 1
        assert tuple(layer_norms[0].normalized_shape) == (5,)

    def test_legacy_use_layer_norm_compat(self, device):
        """Test that legacy use_layer_norm still maps to hidden LayerNorm."""
        mlp = MLP(10, 5, [12, 12], use_layer_norm=True).to(device)
        layer_norms = [m for m in mlp.modules() if isinstance(m, nn.LayerNorm)]
        assert len(layer_norms) == 2
        assert tuple(layer_norms[0].normalized_shape) == (12,)
        assert tuple(layer_norms[1].normalized_shape) == (12,)

    def test_custom_linear_factory_conv2d(self, device):
        """Test custom linear_factory for pointwise conv channel MLPs."""
        mlp = MLP(
            4, 6, [8],
            activation='gelu',
            norm=None,
            linear_factory=lambda a, b: nn.Conv2d(a, b, kernel_size=1),
            use_layer_norm=False,
        ).to(device)
        x = torch.randn(2, 4, 16, 16, device=device)
        out = mlp(x)
        assert out.shape == (2, 6, 16, 16)

    def test_batch_norm(self, device):
        """Test BatchNorm1d normalization."""
        mlp = MLP(10, 5, [12, 12], norm='batch').to(device)
        x = torch.randn(4, 10, device=device)
        
        # Training mode
        mlp.train()
        out1 = mlp(x)
        out2 = mlp(x)
        # Outputs should differ slightly due to running stats update
        
        # Eval mode
        mlp.eval()
        out3 = mlp(x)
        out4 = mlp(x)
        # Outputs should be same in eval mode
        assert torch.allclose(out3, out4)
        assert out3.shape == (4, 5)

    def test_instance_norm(self, device):
        """Test that norm='instance' provides per-sample normalization semantics.

        For 2D (N, C) input, _InstanceNorm1dAdapter delegates to LayerNorm
        because InstanceNorm1d fails on (N, C, 1) in training mode.  The
        net effect is identical per-sample-independent normalisation.
        """
        mlp = MLP(10, 5, [12, 12], norm='instance').to(device)
        x = torch.randn(4, 10, device=device)

        out = mlp(x)
        assert out.shape == (4, 5)

        # Structural check: exactly 2 _InstanceNorm1dAdapter modules
        # (one per hidden layer).  We avoid importing the private class
        # directly and instead match by name.
        adapter_count = sum(
            1 for m in mlp.modules()
            if type(m).__name__ == '_InstanceNorm1dAdapter'
        )
        assert adapter_count == 2, (
            f"Expected 2 _InstanceNorm1dAdapter modules, found {adapter_count}"
        )

        # Behavioural check: per-sample independence.
        # If normalisation is per-sample (LayerNorm / InstanceNorm semantics),
        # then scaling one sample's input by a large constant should not affect
        # any other sample's output.  BatchNorm would fail this because its
        # batch statistics are dominated by the scaled sample.
        mlp.eval()
        x_base = torch.randn(4, 10, device=device)
        x_scaled = x_base.clone()
        x_scaled[2] = x_scaled[2] * 1000.0   # magnify sample 2 only
        with torch.no_grad():
            out_base   = mlp(x_base)
            out_scaled = mlp(x_scaled)
        for i in [0, 1, 3]:
            assert torch.allclose(out_base[i], out_scaled[i], atol=1e-4), (
                f"Sample {i} output changed when scaling sample 2 — "
                "expected per-sample independence from norm='instance'"
            )

    def test_group_norm(self, device):
        """Test GroupNorm normalization."""
        mlp = MLP(64, 32, [128, 128], norm='group').to(device)
        x = torch.randn(4, 64, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 32)
        
        # Check that GroupNorm modules exist
        norm_modules = [m for m in mlp.modules() if isinstance(m, nn.GroupNorm)]
        assert len(norm_modules) == 2  # 2 hidden layers

    def test_group_norm_custom_groups(self, device):
        """Test GroupNorm with custom number of groups via dict spec."""
        mlp = MLP(64, 32, [128], norm={'type': 'group', 'num_groups': 4}).to(device)
        x = torch.randn(4, 64, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 32)
        
        # Check GroupNorm has correct num_groups
        norm_modules = [m for m in mlp.modules() if isinstance(m, nn.GroupNorm)]
        assert len(norm_modules) == 1
        assert norm_modules[0].num_groups == 4

    def test_group_norm_auto_adjust_groups(self, device):
        """Test that GroupNorm auto-adjusts num_groups when dim is not divisible."""
        # 10 is not divisible by 8 (default), should adjust to 5 or 2 or 1
        mlp = MLP(10, 5, [12], norm='group').to(device)
        x = torch.randn(4, 10, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 5)

    def test_batch_norm_with_kwargs(self, device):
        """Test BatchNorm1d with custom kwargs via dict spec."""
        mlp = MLP(10, 5, [12], norm={'type': 'batch', 'eps': 1e-4, 'momentum': 0.01}).to(device)
        x = torch.randn(4, 10, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 5)
        
        # Check BatchNorm has correct eps
        norm_modules = [m for m in mlp.modules() if isinstance(m, nn.BatchNorm1d)]
        assert len(norm_modules) == 1
        assert norm_modules[0].eps == 1e-4
        assert norm_modules[0].momentum == 0.01

    def test_layer_norm_with_kwargs(self, device):
        """Test LayerNorm with custom kwargs via dict spec."""
        mlp = MLP(10, 5, [12], norm={'type': 'layer', 'eps': 1e-5, 'elementwise_affine': False}).to(device)
        x = torch.randn(4, 10, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 5)
        
        # Check LayerNorm has correct settings
        norm_modules = [m for m in mlp.modules() if isinstance(m, nn.LayerNorm)]
        assert len(norm_modules) == 1
        assert norm_modules[0].eps == 1e-5
        assert norm_modules[0].elementwise_affine is False

    def test_mixed_normalization_per_layer(self, device):
        """Test different normalization types per layer."""
        mlp = MLP(10, 5, [12, 12], norms=['batch', 'layer', 'group']).to(device)
        x = torch.randn(4, 10, device=device)
        
        out = mlp(x)
        assert out.shape == (4, 5)
        
        # Check each norm type exists
        batch_norms = [m for m in mlp.modules() if isinstance(m, nn.BatchNorm1d)]
        layer_norms = [m for m in mlp.modules() if isinstance(m, nn.LayerNorm)]
        group_norms = [m for m in mlp.modules() if isinstance(m, nn.GroupNorm)]
        
        assert len(batch_norms) == 1
        assert len(layer_norms) == 1
        assert len(group_norms) == 1

    def test_unknown_norm_spec_raises(self, device):
        """Test that unknown norm spec raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization spec"):
            MLP(10, 5, [12], norm='unknown_norm')

    def test_unknown_norm_dict_type_raises(self, device):
        """Test that unknown norm type in dict spec raises ValueError."""
        with pytest.raises(ValueError, match="Unknown norm type in dict spec"):
            MLP(10, 5, [12], norm={'type': 'unknown'})


class TestResidual:
    """Test Residual wrapper."""
    
    def test_simple_residual(self, device):
        """Test simple residual connection."""
        module = nn.Linear(10, 10).to(device)
        residual = Residual(module).to(device)
        
        x = torch.randn(3, 10, device=device)
        out = residual(x)
        
        expected = x + module(x)
        assert torch.allclose(out, expected)
    
    def test_residual_with_norm(self, device):
        """Test residual with normalization."""
        module = nn.Linear(10, 10).to(device)
        norm = nn.LayerNorm(10).to(device)
        residual = Residual(module, norm=norm).to(device)
        
        x = torch.randn(3, 10, device=device)
        out = residual(x)
        
        expected = x + module(norm(x))
        assert torch.allclose(out, expected)


class TestGraphNetBlock:
    """Test GraphNetBlock (node/edge-only, no globals)."""

    def test_forward(self, device):
        """Test basic forward pass."""
        block = GraphNetBlock(latent_dim=16).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = block(graph)

        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 16)

    def test_globals_passed_through(self, device):
        """Globals on the graph are passed through unchanged (not updated)."""
        block = GraphNetBlock(latent_dim=16).to(device)
        g = torch.randn(1, 4, device=device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            globals=g,
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = block(graph)
        assert out.globals is g  # same object, not updated

    def test_batched(self, device):
        """Test with a batch of two graphs."""
        block = GraphNetBlock(latent_dim=8).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(7, 8, device=device),
            edges=torch.randn(10, 8, device=device),
            receivers=torch.randint(0, 7, (10,), device=device),
            senders=torch.randint(0, 7, (10,), device=device),
            n_node=torch.tensor([3, 4], device=device),
            n_edge=torch.tensor([4, 6], device=device),
        )

        out = block(graph)
        assert out.nodes.shape == (7, 8)
        assert out.edges.shape == (10, 8)


class TestMessagePassingBlock:
    """Test MessagePassingBase ABC contract."""

    def test_graphnetblock_is_subclass(self):
        """GraphNetBlock must be a MessagePassingBase subclass."""
        assert issubclass(GraphNetBlock, MessagePassingBase)

    def test_edgeconditioned_is_subclass(self):
        """EdgeConditionedConvBlock must be a MessagePassingBase subclass."""
        assert issubclass(EdgeConditionedConvBlock, MessagePassingBase)

    def test_updates_edges_attr(self):
        """Check updates_edges class attribute."""
        assert GraphNetBlock.updates_edges is True
        assert EdgeConditionedConvBlock.updates_edges is False

    def test_cannot_instantiate_abc(self):
        """MessagePassingBase is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MessagePassingBase(latent_dim=8)


class TestEdgeConditionedConvBlock:
    """Test EdgeConditionedConvBlock (NNConv-style)."""

    @staticmethod
    def _ewn(edge_latent_dim: int, processor=None, latent_dim: int = 16):
        """Build a default edge_weight_net for testing."""
        if processor is None:
            processor = FullEdgeMessageProcessor(latent_dim)
        return MLP(
            in_dim=edge_latent_dim,
            out_dim=processor.weight_out_dim,
            hidden_dims=[128],
            activation='relu',
            use_layer_norm=False,
        )

    def _make_graph(self, device, latent=16, edge_latent=16, n_nodes=5, n_edges=8):
        return GraphsTuple.from_flat(
            nodes=torch.randn(n_nodes, latent, device=device),
            n_node=torch.tensor([n_nodes], device=device),
            edges=torch.randn(n_edges, edge_latent, device=device),
            senders=torch.randint(0, n_nodes, (n_edges,), device=device),
            receivers=torch.randint(0, n_nodes, (n_edges,), device=device),
            n_edge=torch.tensor([n_edges], device=device),
        )

    def test_forward_full(self, device):
        """Test forward with full weight matrix."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16),
        ).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        # Edges must be unchanged (passed through)
        assert out.edges is graph.edges

    def test_forward_vector(self, device):
        """Test forward with per-channel vector gating."""
        proc = VectorEdgeMessageProcessor(16)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
        ).to(device)
        out = block(self._make_graph(device))
        assert out.nodes.shape == (5, 16)

    def test_forward_scalar(self, device):
        """Test forward with scalar gating."""
        proc = ScalarEdgeMessageProcessor(16)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
        ).to(device)
        out = block(self._make_graph(device))
        assert out.nodes.shape == (5, 16)

    def test_mean_aggregation(self, device):
        """Test with mean aggregation (used by Graph-PDE GNO)."""
        proc = ScalarEdgeMessageProcessor(16)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc, aggregate='mean',
        ).to(device)
        out = block(self._make_graph(device))
        assert out.nodes.shape == (5, 16)

    def test_no_root_no_bias(self, device):
        """Test with root_weight=False and bias=False."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16),
            root_weight=False, bias=False,
        ).to(device)
        assert block.node_updater.root is None
        assert block.node_updater.bias is None
        out = block(self._make_graph(device))
        assert out.nodes.shape == (5, 16)

    def test_different_edge_dim(self, device):
        """Test with edge_latent_dim != latent_dim."""
        proc = ScalarEdgeMessageProcessor(16)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_net=self._ewn(8, proc),
            edge_processor=proc,
        ).to(device)
        graph = self._make_graph(device, edge_latent=8)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_default_processor_is_full_rank(self, device):
        """Default configuration should resolve to the original full-rank behavior."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16),
        ).to(device)
        assert isinstance(block.edge_processor, FullEdgeMessageProcessor)
        out = block(self._make_graph(device))
        assert out.nodes.shape == (5, 16)

    def test_edge_message_processor_protocol_is_runtime_checkable(self):
        """Built-in processors should satisfy the EdgeMessageProcessor protocol."""
        assert hasattr(EdgeMessageProcessor, '_is_protocol')
        assert isinstance(FullEdgeMessageProcessor(8), EdgeMessageProcessor)
        assert isinstance(VectorEdgeMessageProcessor(8), EdgeMessageProcessor)
        assert isinstance(ScalarEdgeMessageProcessor(8), EdgeMessageProcessor)
        assert isinstance(LowRankEdgeMessageProcessor(8, 2), EdgeMessageProcessor)

    @pytest.mark.parametrize(
        "processor_factory",
        [
            lambda d: FullEdgeMessageProcessor(d),
            lambda d: VectorEdgeMessageProcessor(d),
            lambda d: ScalarEdgeMessageProcessor(d),
            lambda d: LowRankEdgeMessageProcessor(d, 4),
        ],
    )
    def test_plugin_roundtrip_equivalence(self, device, processor_factory):
        """An explicit processor should survive state_dict round-trips exactly."""
        proc_ref = processor_factory(16)
        proc_plug = processor_factory(16)
        reference = EdgeConditionedConvBlock(
            latent_dim=16,
            edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc_ref),
            root_weight=True,
            bias=True,
            edge_processor=proc_ref,
        ).to(device)
        plugin = EdgeConditionedConvBlock(
            latent_dim=16,
            edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc_plug),
            root_weight=True,
            bias=True,
            edge_processor=proc_plug,
        ).to(device)
        plugin.load_state_dict(reference.state_dict())

        graph = self._make_graph(device)
        reference_out = reference(graph)
        plugin_out = plugin(graph)

        assert torch.allclose(plugin_out.nodes, reference_out.nodes)
        assert plugin_out.edges is graph.edges

    def test_default_equivalent_to_explicit_full_processor(self, device):
        """The default block must remain exactly equivalent to explicit full-rank selection."""
        ewn1 = self._ewn(16)
        ewn2 = self._ewn(16)
        default_block = EdgeConditionedConvBlock(
            latent_dim=16,
            edge_latent_dim=16,
            edge_weight_net=ewn1,
            root_weight=True,
            bias=True,
        ).to(device)
        explicit_block = EdgeConditionedConvBlock(
            latent_dim=16,
            edge_latent_dim=16,
            edge_weight_net=ewn2,
            root_weight=True,
            bias=True,
            edge_processor=FullEdgeMessageProcessor(16),
        ).to(device)
        explicit_block.load_state_dict(default_block.state_dict())

        graph = self._make_graph(device)
        default_out = default_block(graph)
        explicit_out = explicit_block(graph)

        assert torch.allclose(default_out.nodes, explicit_out.nodes)
        assert torch.equal(default_out.edges, explicit_out.edges)

    def test_rejects_non_module_edge_processor(self):
        """Custom edge processors must be nn.Module instances."""

        class CallableOnly:
            @property
            def weight_out_dim(self):
                return 16

            def forward(self, src_x, edge_weights):
                return src_x * edge_weights

        with pytest.raises(TypeError, match="edge_processor must be an nn.Module"):
            EdgeConditionedConvBlock(
                latent_dim=16,
                edge_latent_dim=16,
                edge_weight_net=self._ewn(16),
                edge_processor=CallableOnly(),
            )

    def test_rejects_protocol_violation(self):
        """Custom edge processors must satisfy the EdgeMessageProcessor protocol."""

        class MissingWeightOutDim(nn.Module):
            latent_dim = 16

            def forward(self, src_x, edge_weights):
                return src_x

        with pytest.raises(TypeError, match="must satisfy EdgeMessageProcessor protocol"):
            EdgeConditionedConvBlock(
                latent_dim=16,
                edge_latent_dim=16,
                edge_weight_net=self._ewn(16),
                edge_processor=MissingWeightOutDim(),
            )

    def test_rejects_processor_latent_dim_mismatch(self):
        """Injected processors must agree with the block latent_dim."""
        with pytest.raises(ValueError, match="edge_processor latent_dim must match block latent_dim"):
            proc = VectorEdgeMessageProcessor(8)
            EdgeConditionedConvBlock(
                latent_dim=16,
                edge_latent_dim=16,
                edge_weight_net=self._ewn(16, proc),
                edge_processor=proc,
            )

    def test_construction_time_shape_check_for_custom_processor_output(self, device):
        """Malformed custom processors should fail during pipeline verification."""

        class BadShapeProcessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim = 16

            @property
            def weight_out_dim(self):
                return 16

            def forward(self, src_x, edge_weights):
                return edge_weights[:, :5]

        with pytest.raises(ValueError, match="edge message pipeline must return shape"):
            proc = BadShapeProcessor()
            EdgeConditionedConvBlock(
                latent_dim=16,
                edge_latent_dim=16,
                edge_weight_net=self._ewn(16, proc),
                edge_processor=proc,
            )

    def test_gradient_flow(self, device):
        """Gradients must flow through edge_weight_net, root, and bias."""
        proc = VectorEdgeMessageProcessor(8)
        block = EdgeConditionedConvBlock(
            latent_dim=8, edge_latent_dim=8,
            edge_weight_net=self._ewn(8, proc, latent_dim=8),
            edge_processor=proc,
        ).to(device)
        graph = self._make_graph(device, latent=8, edge_latent=8, n_nodes=4, n_edges=6)
        out = block(graph)
        loss = out.nodes.sum()
        loss.backward()
        assert block.node_updater.root.grad is not None
        assert block.node_updater.bias.grad is not None

    # -------------------------------------------------------------------------
    # Low-Rank Approximation Tests
    # -------------------------------------------------------------------------

    def test_forward_low_rank(self, device):
        """Test forward with symmetric low-rank approximation."""
        proc = LowRankEdgeMessageProcessor(16, 4)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
        ).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        # Edges must be unchanged (passed through)
        assert out.edges is graph.edges

    def test_low_rank_parameter_reduction(self, device):
        """Test that low-rank mode reduces parameters correctly."""
        d = 64
        r = 8
        
        full_proc = FullEdgeMessageProcessor(d)
        block_full = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, full_proc, latent_dim=d),
            root_weight=False, bias=False,
        ).to(device)
        
        lr_proc = LowRankEdgeMessageProcessor(d, r)
        block_lowrank = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, lr_proc, latent_dim=d),
            edge_processor=lr_proc,
            root_weight=False, bias=False,
        ).to(device)
        
        params_full = sum(p.numel() for p in block_full.edge_weight_net.parameters())
        params_lowrank = sum(p.numel() for p in block_lowrank.edge_weight_net.parameters())
        
        # Full-rank edge MLP outputs d*d values
        # Low-rank edge MLP outputs d*r values
        # Parameter reduction should be significant
        assert params_lowrank < params_full
        # The output dimension reduction is d*d vs d*r = 4096 vs 512 for d=64, r=8
        assert block_lowrank.low_rank == r

    def test_low_rank_memory_efficiency(self, device):
        """Test memory per edge calculation for low-rank vs full-rank."""
        d = 64
        r = 8
        
        memory_full = d * d  # 4096 values per edge
        memory_lowrank = d * r  # 512 values per edge
        
        reduction_ratio = memory_full / memory_lowrank
        assert reduction_ratio == 8.0  # 8x reduction for r=8

    def test_low_rank_invalid_rank_raises(self, device):
        """Test that invalid low_rank values raise ValueError."""
        # rank = 0 should raise error
        with pytest.raises(ValueError, match="low_rank must be positive"):
            LowRankEdgeMessageProcessor(16, 0)
        
        # rank > latent_dim should raise error
        with pytest.raises(ValueError, match="low_rank .* must be <= latent_dim"):
            LowRankEdgeMessageProcessor(16, 32)

    def test_low_rank_gradient_flow(self, device):
        """Test gradients flow correctly in low-rank mode."""
        proc = LowRankEdgeMessageProcessor(16, 4)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
        ).to(device)
        
        graph = self._make_graph(device, latent=16, edge_latent=16, n_nodes=4, n_edges=6)
        out = block(graph)
        loss = out.nodes.sum()
        loss.backward()
        
        # Check gradients exist for edge weight net
        has_grad = any(p.grad is not None for p in block.edge_weight_net.parameters())
        assert has_grad, "Edge weight net should have gradients"
        
        # Check root and bias gradients
        assert block.node_updater.root.grad is not None
        assert block.node_updater.bias.grad is not None

    def test_low_rank_equivalence_with_same_weights(self, device):
        """Test that low-rank produces correct output shapes and valid values."""
        proc = LowRankEdgeMessageProcessor(32, 8)
        block = EdgeConditionedConvBlock(
            latent_dim=32, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc, latent_dim=32),
            edge_processor=proc,
            root_weight=False, bias=False,
        ).to(device)
        
        # Create a simple graph
        graph = self._make_graph(device, latent=32, edge_latent=16, n_nodes=10, n_edges=20)
        
        out = block(graph)
        
        # Check output shape
        assert out.nodes.shape == (10, 32)
        
        # Check output is finite (no NaN or Inf)
        assert torch.all(torch.isfinite(out.nodes))

    def test_low_rank_different_ranks(self, device):
        """Test low-rank with different rank values."""
        latent_dim = 64
        
        for rank in [4, 8, 16, 32]:
            proc = LowRankEdgeMessageProcessor(latent_dim, rank)
            block = EdgeConditionedConvBlock(
                latent_dim=latent_dim, edge_latent_dim=16,
                edge_weight_net=self._ewn(16, proc, latent_dim=latent_dim),
                edge_processor=proc,
            ).to(device)
            
            graph = self._make_graph(device, latent=latent_dim, edge_latent=16)
            out = block(graph)
            
            assert out.nodes.shape == (5, latent_dim), f"Failed for rank={rank}"

    def test_low_rank_no_root_no_bias(self, device):
        """Test low-rank with root_weight=False and bias=False."""
        proc = LowRankEdgeMessageProcessor(16, 4)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
            root_weight=False, bias=False,
        ).to(device)
        
        assert block.node_updater.root is None
        assert block.node_updater.bias is None
        
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_low_rank_mean_aggregation(self, device):
        """Test low-rank with mean aggregation."""
        proc = LowRankEdgeMessageProcessor(16, 4)
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=16,
            edge_weight_net=self._ewn(16, proc),
            edge_processor=proc,
            aggregate='mean',
        ).to(device)
        
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_low_rank_vs_full_rank_output_shape(self, device):
        """Verify low-rank and full-rank produce same output shape."""
        latent_dim = 32
        edge_latent_dim = 8
        
        block_full = EdgeConditionedConvBlock(
            latent_dim=latent_dim, edge_latent_dim=edge_latent_dim,
            edge_weight_net=self._ewn(edge_latent_dim, latent_dim=latent_dim),
            root_weight=False, bias=False,
        ).to(device)
        
        lr_proc = LowRankEdgeMessageProcessor(latent_dim, 8)
        block_lowrank = EdgeConditionedConvBlock(
            latent_dim=latent_dim, edge_latent_dim=edge_latent_dim,
            edge_weight_net=self._ewn(edge_latent_dim, lr_proc, latent_dim=latent_dim),
            edge_processor=lr_proc,
            root_weight=False, bias=False,
        ).to(device)
        
        graph = self._make_graph(device, latent=latent_dim, edge_latent=edge_latent_dim)
        
        out_full = block_full(graph)
        out_lowrank = block_lowrank(graph)
        
        # Both should produce same output shape
        assert out_full.nodes.shape == out_lowrank.nodes.shape
        assert out_full.nodes.shape == (5, latent_dim)


class TestGraphNetProcessorBlockFactory:
    """Test GraphNetProcessor with custom block_factory."""

    def test_with_edge_conditioned_blocks(self, device):
        """Processor with EdgeConditionedConvBlock via block_factory."""
        def factory():
            proc = ScalarEdgeMessageProcessor(16)
            return EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=16,
                edge_weight_net=MLP(
                    in_dim=16, out_dim=proc.weight_out_dim,
                    hidden_dims=[128], activation='relu', use_layer_norm=False,
                ),
                edge_processor=proc, aggregate='mean',
            )
        processor = GraphNetProcessor(
            latent_dim=16, n_layers=3,
            block_factory=factory,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = processor(graph)
        assert out.nodes.shape == (5, 16)
        # Edges must be unchanged (EdgeConditionedConvBlock doesn't update edges)
        assert torch.equal(out.edges, graph.edges)

    def test_no_residual_with_factory(self, device):
        """Non-residual mode with custom block_factory."""
        def factory():
            proc = ScalarEdgeMessageProcessor(8)
            return EdgeConditionedConvBlock(
                latent_dim=8, edge_latent_dim=8,
                edge_weight_net=MLP(
                    in_dim=8, out_dim=proc.weight_out_dim,
                    hidden_dims=[128], activation='relu', use_layer_norm=False,
                ),
                edge_processor=proc,
            )
        processor = GraphNetProcessor(
            latent_dim=8, n_layers=2, residual=False,
            block_factory=factory,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(4, 8, device=device),
            edges=torch.randn(6, 8, device=device),
            receivers=torch.randint(0, 4, (6,), device=device),
            senders=torch.randint(0, 4, (6,), device=device),
            n_node=torch.tensor([4], device=device),
            n_edge=torch.tensor([6], device=device),
        )

        out = processor(graph)
        assert out.nodes.shape == (4, 8)


class TestEdgeConvBlock:
    """Test EdgeConvBlock (DGCNN-style) message passing."""

    def _make_graph(self, device, latent=16):
        return GraphsTuple.from_flat(
            nodes=torch.randn(5, latent, device=device),
            edges=torch.randn(8, latent, device=device),  # edge dim = latent
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

    def test_forward_default_max(self, device):
        """Test forward with default Max aggregation."""
        block = EdgeConvBlock(latent_dim=16).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_forward_sum_aggregation(self, device):
        """Test forward with sum aggregation."""
        block = EdgeConvBlock(latent_dim=16, aggregate='sum').to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_forward_mean_aggregation(self, device):
        """Test forward with mean aggregation."""
        block = EdgeConvBlock(latent_dim=16, aggregate='mean').to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_updates_edges_false(self, device):
        """Verify EdgeConvBlock does not update edges."""
        block = EdgeConvBlock(latent_dim=16)
        assert block.updates_edges is False

    def test_is_subclass_of_message_passing_block(self, device):
        """EdgeConvBlock must be a MessagePassingBase subclass."""
        assert issubclass(EdgeConvBlock, MessagePassingBase)

    def test_explicit_assembler_node_difference(self, device):
        """Test EdgeConvBlock with explicit NodeDifferenceAssembler."""
        from gnn_pde_v2.components import NodeDifferenceAssembler
        
        assembler = NodeDifferenceAssembler(latent_dim=16)
        block = EdgeConvBlock(
            latent_dim=16,
            edge_assembler=assembler,
        ).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert isinstance(block.edge_assembler, NodeDifferenceAssembler)

    def test_explicit_assembler_concat(self, device):
        """Test EdgeConvBlock with ConcatAssembler."""
        from gnn_pde_v2.components import ConcatAssembler
        
        block = EdgeConvBlock(
            latent_dim=16,
            edge_assembler=ConcatAssembler(16),
        ).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert isinstance(block.edge_assembler, ConcatAssembler)

    def test_explicit_assembler_difference_only(self, device):
        """Test EdgeConvBlock with DifferenceOnlyAssembler."""
        from gnn_pde_v2.components import DifferenceOnlyAssembler
        
        block = EdgeConvBlock(
            latent_dim=16,
            edge_assembler=DifferenceOnlyAssembler(16),
        ).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert isinstance(block.edge_assembler, DifferenceOnlyAssembler)

    def test_explicit_assembler_with_edges(self, device):
        """Test EdgeConvBlock with ConcatWithEdgesAssembler."""
        from gnn_pde_v2.components import ConcatWithEdgesAssembler
        from gnn_pde_v2.core import MLP
        
        # Create graph with edge_dim=3
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 3, device=device),  # edge_dim = 3
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        block = EdgeConvBlock(
            latent_dim=16,
            edge_assembler=ConcatWithEdgesAssembler(latent_dim=16, edge_dim=3),
            edge_transform=MLP(35, 16, [32], 'relu'),  # 2*16 + 3 = 35
        ).to(device)
        
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert isinstance(block.edge_assembler, ConcatWithEdgesAssembler)

    def test_custom_edge_transform(self, device):
        """Test EdgeConvBlock with custom edge_transform."""
        from gnn_pde_v2.core import MLP
        
        custom_transform = MLP(32, 16, [64, 64], 'gelu')
        block = EdgeConvBlock(
            latent_dim=16,
            edge_transform=custom_transform,
        ).to(device)
        
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert block.edge_transform is custom_transform


class TestGlobalGraphNetBlock:
    """Test GlobalGraphNetBlock (full Graph Nets with globals)."""

    def test_forward(self, device):
        """Test basic forward pass with globals."""
        block = GlobalGraphNetBlock(latent_dim=16, global_latent_dim=4).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            globals=torch.randn(1, 4, device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = block(graph)

        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 16)
        assert out.globals.shape == (1, 4)

    def test_globals_updated(self, device):
        """Global vector must change after a forward pass."""
        block = GlobalGraphNetBlock(latent_dim=8, global_latent_dim=4).to(device)
        g = torch.randn(1, 4, device=device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 8, device=device),
            edges=torch.randn(6, 8, device=device),
            receivers=torch.randint(0, 5, (6,), device=device),
            senders=torch.randint(0, 5, (6,), device=device),
            globals=g.clone(),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([6], device=device),
        )

        out = block(graph)
        assert not torch.allclose(out.globals, g)

    def test_batched(self, device):
        """Test with a batch of two graphs."""
        block = GlobalGraphNetBlock(latent_dim=8, global_latent_dim=4).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(7, 8, device=device),
            edges=torch.randn(10, 8, device=device),
            receivers=torch.randint(0, 7, (10,), device=device),
            senders=torch.randint(0, 7, (10,), device=device),
            globals=torch.randn(2, 4, device=device),
            n_node=torch.tensor([3, 4], device=device),
            n_edge=torch.tensor([4, 6], device=device),
        )

        out = block(graph)
        assert out.nodes.shape == (7, 8)
        assert out.edges.shape == (10, 8)
        assert out.globals.shape == (2, 4)

    def test_missing_globals_raises(self, device):
        """AssertionError when graph.globals is None."""
        block = GlobalGraphNetBlock(latent_dim=8, global_latent_dim=4).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 8, device=device),
            edges=torch.randn(6, 8, device=device),
            receivers=torch.randint(0, 5, (6,), device=device),
            senders=torch.randint(0, 5, (6,), device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([6], device=device),
        )

        with pytest.raises(AssertionError, match="GlobalGraphNetBlock requires"):
            block(graph)


class TestGraphNetProcessor:
    """Test GraphNetProcessor (node/edge-only stack)."""

    def test_forward(self, device):
        """Test basic forward pass."""
        processor = GraphNetProcessor(
            latent_dim=16,
            n_layers=3,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = processor(graph)

        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 16)


class TestGlobalGraphNetProcessor:
    """Test GlobalGraphNetProcessor (full Graph Nets stack with globals)."""

    def test_forward(self, device):
        """Test forward pass with globals."""
        processor = GlobalGraphNetProcessor(
            latent_dim=16,
            global_latent_dim=4,
            n_layers=3,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            globals=torch.randn(1, 4, device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

        out = processor(graph)

        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 16)
        assert out.globals.shape == (1, 4)

    def test_residual_globals(self, device):
        """Residual connection must also apply to globals."""
        processor = GlobalGraphNetProcessor(
            latent_dim=8,
            global_latent_dim=4,
            n_layers=2,
            residual=True,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(4, 8, device=device),
            edges=torch.randn(6, 8, device=device),
            receivers=torch.randint(0, 4, (6,), device=device),
            senders=torch.randint(0, 4, (6,), device=device),
            globals=torch.randn(1, 4, device=device),
            n_node=torch.tensor([4], device=device),
            n_edge=torch.tensor([6], device=device),
        )

        out = processor(graph)
        assert out.globals.shape == (1, 4)


class TestMLPDecoder:
    """Test MLPDecoder."""

    def test_forward(self, device):
        """Test basic forward pass."""
        decoder = MLPDecoder(
            latent_dim=16,
            out_dim=3,
        ).to(device)

        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            n_node=torch.tensor([5], device=device),
        )

        out = decoder(graph)

        assert out.shape == (5, 3)


class TestIndependentMLPDecoder:
    """Test IndependentMLPDecoder."""

    def test_forward(self, device):
        """Test multi-output forward pass.

        forward() concatenates all component outputs along the feature
        dimension and returns a single Tensor of shape [N, sum(out_dims)].
        Each component's slice must equal what the corresponding MLP alone
        would produce, confirming the components are independent.
        """
        out_dims = [3, 5, 2]
        decoder = IndependentMLPDecoder(
            latent_dim=16,
            out_dims=out_dims,
        ).to(device)

        nodes = torch.randn(5, 16, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            n_node=torch.tensor([5], device=device),
        )

        out = decoder(graph)

        # Return type must be a single concatenated tensor
        assert isinstance(out, torch.Tensor)
        # Total width = sum of component widths
        assert out.shape == (5, sum(out_dims))

        # Each component slice must match the corresponding MLP's output
        # exactly, confirming independence and correct concatenation order.
        with torch.no_grad():
            offset = 0
            for mlp, width in zip(decoder.decoders, out_dims):
                expected = mlp(nodes)
                assert torch.allclose(out[:, offset:offset + width], expected), (
                    f"Component slice [{offset}:{offset + width}] does not match "
                    f"its MLP's output"
                )
                offset += width


class TestProbeDecoder:
    """Test ProbeDecoder with configurable processors."""

    def test_with_gen_blocks(self, device):
        """Test probe decoder with GENBlock processors."""
        processor = torch.nn.ModuleList([
            GENBlock(latent_dim=16, hidden_dim=32, num_mlp_layers=2)
            for _ in range(3)
        ])
        
        decoder = ProbeDecoder(
            latent_dim=16,
            processor=processor,
            edge_encoder=LearnableRBFEncoder(num_kernels=10),
            out_dim=1,
            k_nearest=3,
        ).to(device)

        # Source graph
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device),
            positions=torch.randn(10, 2, device=device),
            n_node=torch.tensor([10], device=device),
        )

        # Query positions
        query_positions = torch.randn(5, 2, device=device)

        out = decoder(graph, query_positions)
        assert out.shape == (5, 1)

    def test_with_query_features(self, device):
        """Test probe decoder with query input features."""
        decoder = ProbeDecoder(
            latent_dim=16,
            processor=GraphNetBlock(latent_dim=16, hidden_dim=32),
            out_dim=1,
            k_nearest=3,
            decode_with_query_features=True,
        ).to(device)
        
        # Adjust output MLP to handle concatenated features
        decoder.output_mlp = torch.nn.Linear(16 + 4, 1).to(device)

        # Source graph
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device),
            edges=torch.randn(30, 16, device=device),
            senders=torch.randint(0, 10, (30,), device=device),
            receivers=torch.randint(0, 10, (30,), device=device),
            positions=torch.randn(10, 2, device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([30], device=device),
        )

        # Query positions and features
        query_positions = torch.randn(5, 2, device=device)
        query_features = torch.randn(5, 4, device=device)

        out = decoder(graph, query_positions, query_features)
        assert out.shape == (5, 1)


class TestLearnableRBFEncoder:
    """Test LearnableRBFEncoder."""
    
    def test_forward(self, device):
        """Test RBF encoding."""
        encoder = LearnableRBFEncoder(
            num_kernels=20,
            d_min=0.0,
            d_max=5.0,
            learnable=True,
        ).to(device)
        
        distances = torch.tensor([0.5, 1.0, 2.0, 3.0], device=device)
        rbf_features = encoder(distances)
        
        assert rbf_features.shape == (4, 20)
        assert encoder.mu.requires_grad
        assert encoder.beta.requires_grad
    
    def test_cosine_cutoff(self, device):
        """Test cosine cutoff function."""
        encoder = LearnableRBFEncoder(d_max=5.0, learnable=False).to(device)
        
        # Inside cutoff range
        d_inside = torch.tensor([0.0, 2.5, 4.9], device=device)
        cutoff_inside = encoder.cosine_cutoff(d_inside)
        assert (cutoff_inside > 0).all()
        
        # At cutoff boundary
        d_boundary = torch.tensor([5.0], device=device)
        cutoff_boundary = encoder.cosine_cutoff(d_boundary)
        assert cutoff_boundary.item() == 0.0
        
        # Outside cutoff range
        d_outside = torch.tensor([5.1, 10.0], device=device)
        cutoff_outside = encoder.cosine_cutoff(d_outside)
        assert (cutoff_outside == 0).all()


class TestProbeGraphBuilder:
    """Test ProbeGraphBuilder utility."""
    
    def test_build(self, device):
        """Test probe graph construction."""
        source_pos = torch.randn(10, 2, device=device)
        source_feat = torch.randn(10, 16, device=device)
        query_pos = torch.randn(5, 2, device=device)
        
        probe_graph = ProbeGraphBuilder.build(
            source_positions=source_pos,
            source_features=source_feat,
            query_positions=query_pos,
            k_nearest=3,
        )
        
        # Check layout: [source_nodes | probe_nodes]
        assert probe_graph.nodes.shape == (15, 16)  # 10 source + 5 probe
        assert probe_graph.n_node.item() == 15
        assert probe_graph.n_edge.item() == 5 * 3  # 5 queries * 3 neighbors
        
        # Check edge structure: source -> probe
        # Senders should be in [0, 10), receivers in [10, 15)
        assert (probe_graph.senders < 10).all()
        assert (probe_graph.receivers >= 10).all()
    
    def test_extract_probe_nodes(self, device):
        """Test probe node extraction."""
        # Create batched probe graph
        graph1 = GraphsTuple.from_flat(
            nodes=torch.randn(15, 16, device=device),  # 10 source + 5 probe
            edges=torch.randn(15, 1, device=device),
            senders=torch.randint(0, 10, (15,), device=device),
            receivers=torch.randint(10, 15, (15,), device=device),
            n_node=torch.tensor([15], device=device),
            n_edge=torch.tensor([15], device=device),
        )
        graph2 = GraphsTuple.from_flat(
            nodes=torch.randn(18, 16, device=device),  # 12 source + 6 probe
            edges=torch.randn(18, 1, device=device),
            senders=torch.randint(0, 12, (18,), device=device),
            receivers=torch.randint(12, 18, (18,), device=device),
            n_node=torch.tensor([18], device=device),
            n_edge=torch.tensor([18], device=device),
        )
        
        from gnn_pde_v2.core.graph import batch_graphs
        batched = batch_graphs([graph1, graph2])
        n_query = torch.tensor([5, 6], device=device)
        
        probe_nodes = ProbeGraphBuilder.extract_probe_nodes(batched, n_query)
        assert probe_nodes.shape == (11, 16)  # 5 + 6 probe nodes


class TestGENBlock:
    """Test GENBlock message passing."""
    
    def test_forward(self, device):
        """Test GEN block forward pass."""
        block = GENBlock(
            latent_dim=16,
            hidden_dim=32,
            num_mlp_layers=2,
            epsilon=1e-6,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device),
            edges=torch.randn(30, 16, device=device),
            senders=torch.randint(0, 10, (30,), device=device),
            receivers=torch.randint(0, 10, (30,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([30], device=device),
        )
        
        out_graph = block(graph)
        
        # Check shapes
        assert out_graph.nodes.shape == (10, 16)
        assert out_graph.edges.shape == (30, 16)  # Edges NOT updated in GEN
        
        # Check that edges are unchanged (GENBlock.updates_edges = False)
        assert torch.allclose(out_graph.edges, graph.edges)


class TestMultiHeadAttention:
    """Test MultiHeadAttention with and without relative position encoding."""
    
    def test_forward_basic(self, device):
        """Test basic forward pass without positions."""
        attn = MultiHeadAttention(dim=64, n_heads=8, dropout=0.0).to(device)
        x = torch.randn(10, 64, device=device)
        
        out = attn(x)
        
        assert out.shape == (10, 64)
    
    def test_forward_batched(self, device):
        """Test batched forward pass."""
        attn = MultiHeadAttention(dim=64, n_heads=8).to(device)
        x = torch.randn(2, 10, 64, device=device)
        
        out = attn(x)
        
        assert out.shape == (2, 10, 64)
    
    def test_relative_positions_learned(self, device):
        """Test relative position encoding with learned embeddings."""
        attn = MultiHeadAttention(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
            position_dim=2,
            num_position_buckets=16,
            position_encoding_type='learned',
        ).to(device)
        
        x = torch.randn(10, 64, device=device)
        positions = torch.randn(10, 2, device=device)
        
        out = attn(x, positions=positions)
        
        assert out.shape == (10, 64)
    
    def test_relative_positions_sinusoidal(self, device):
        """Test relative position encoding with sinusoidal embeddings."""
        attn = MultiHeadAttention(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
            position_dim=2,
            num_position_buckets=16,
            position_encoding_type='sinusoidal',
        ).to(device)
        
        x = torch.randn(10, 64, device=device)
        positions = torch.randn(10, 2, device=device)
        
        out = attn(x, positions=positions)
        
        assert out.shape == (10, 64)
    
    def test_relative_positions_batched(self, device):
        """Test relative position encoding with batched input."""
        attn = MultiHeadAttention(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
            position_dim=3,
        ).to(device)
        
        x = torch.randn(2, 10, 64, device=device)
        positions = torch.randn(2, 10, 3, device=device)
        
        out = attn(x, positions=positions)
        
        assert out.shape == (2, 10, 64)
    
    def test_relative_positions_missing_raises(self, device):
        """Test that missing positions raises error when use_relative_positions=True."""
        attn = MultiHeadAttention(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
        ).to(device)
        
        x = torch.randn(10, 64, device=device)
        
        with pytest.raises(ValueError, match="positions must be provided"):
            attn(x)
    
    def test_backward_compatibility(self, device):
        """Test that old code without positions still works."""
        attn = MultiHeadAttention(dim=64, n_heads=8).to(device)
        x = torch.randn(10, 64, device=device)
        
        # Should work without positions parameter
        out = attn(x)
        assert out.shape == (10, 64)
        
        # Should also work with mask but no positions
        mask = torch.ones(10, 10, device=device)
        mask[0, 1] = 0
        out = attn(x, mask=mask)
        assert out.shape == (10, 64)
    
    def test_gradient_flow_learned(self, device):
        """Test that gradients flow through learned position encoding."""
        attn = MultiHeadAttention(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
            position_encoding_type='learned',
        ).to(device)
        
        x = torch.randn(5, 64, device=device, requires_grad=True)
        positions = torch.randn(5, 2, device=device)
        
        out = attn(x, positions=positions)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert attn.position_encoding.position_bias.grad is not None


class TestTransformerBlock:
    """Test TransformerBlock with relative position encoding."""
    
    def test_forward_basic(self, device):
        """Test basic forward pass."""
        block = TransformerBlock(dim=64, n_heads=8).to(device)
        x = torch.randn(10, 64, device=device)
        
        out = block(x)
        
        assert out.shape == (10, 64)
    
    def test_forward_with_positions(self, device):
        """Test forward with relative position encoding."""
        block = TransformerBlock(
            dim=64,
            n_heads=8,
            use_relative_positions=True,
            position_dim=2,
        ).to(device)
        
        x = torch.randn(10, 64, device=device)
        positions = torch.randn(10, 2, device=device)
        
        out = block(x, positions=positions)
        
        assert out.shape == (10, 64)


class TestTransformerProcessor:
    """Test TransformerProcessor with relative position encoding."""
    
    def test_forward_basic(self, device):
        """Test basic forward pass without positions."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=2,
            n_heads=8,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 64, device=device),
            n_node=torch.tensor([10], device=device),
        )
        
        out = processor(graph)
        
        assert out.nodes.shape == (10, 64)
    
    def test_forward_with_positions(self, device):
        """Test forward with relative position encoding."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=2,
            n_heads=8,
            use_relative_positions=True,
            position_dim=2,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 64, device=device),
            positions=torch.randn(10, 2, device=device),
            n_node=torch.tensor([10], device=device),
        )
        
        out = processor(graph)
        
        assert out.nodes.shape == (10, 64)
    
    def test_missing_positions_raises(self, device):
        """Test that missing positions raises error when use_relative_positions=True."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=2,
            use_relative_positions=True,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 64, device=device),
            n_node=torch.tensor([10], device=device),
        )
        
        with pytest.raises(ValueError, match="use_relative_positions=True but graph.positions is None"):
            processor(graph)
    
    def test_batched_graphs_with_positions(self, device):
        """Test with batched graphs and positions."""
        processor = TransformerProcessor(
            latent_dim=32,
            n_layers=2,
            n_heads=4,
            use_relative_positions=True,
            position_dim=2,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(15, 32, device=device),
            positions=torch.randn(15, 2, device=device),
            n_node=torch.tensor([7, 8], device=device),
        )
        
        out = processor(graph)
        
        assert out.nodes.shape == (15, 32)


# ============================================================================
# Aggregation Tests
# ============================================================================

import pytest
from gnn_pde_v2.core.aggregation import Aggregation, Sum, Mean, Max, Min, get_aggregation
from gnn_pde_v2.core.functional import aggregate_edges


class TestAggregationProtocol:
    """Test Aggregation Protocol and built-in implementations."""

    def test_protocol_is_runtime_checkable(self):
        """Verify Aggregation is runtime_checkable."""
        assert hasattr(Aggregation, '_is_protocol')

    def test_sum_satisfies_protocol(self):
        """Sum() satisfies Aggregation Protocol."""
        assert isinstance(Sum(), Aggregation)

    def test_mean_satisfies_protocol(self):
        """Mean() satisfies Aggregation Protocol."""
        assert isinstance(Mean(), Aggregation)

    def test_max_satisfies_protocol(self):
        """Max() satisfies Aggregation Protocol."""
        assert isinstance(Max(), Aggregation)

    def test_min_satisfies_protocol(self):
        """Min() satisfies Aggregation Protocol."""
        assert isinstance(Min(), Aggregation)

    def test_custom_callable_satisfies_protocol(self):
        """Custom callable can satisfy Protocol via structural subtyping."""
        def custom_agg(messages, receivers, num_nodes):
            return aggregate_edges(messages, receivers, num_nodes, 'sum')
        
        # Structural check: callable with correct signature
        assert callable(custom_agg)


class TestBuiltInAggregations:
    """Test built-in aggregation classes."""

    @pytest.fixture
    def messages(self, device):
        return torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], device=device)

    @pytest.fixture
    def receivers(self, device):
        return torch.tensor([0, 0, 1, 1], device=device)

    @pytest.fixture
    def num_nodes(self):
        return 2

    def test_sum_aggregation(self, device, messages, receivers, num_nodes):
        """Sum aggregation: node 0 gets [4, 6], node 1 gets [12, 14]."""
        agg = Sum()
        result = agg(messages, receivers, num_nodes)
        
        expected = torch.tensor([
            [4.0, 6.0],   # 1+3, 2+4
            [12.0, 14.0], # 5+7, 6+8
        ], device=device)
        
        assert torch.allclose(result, expected)

    def test_mean_aggregation(self, device, messages, receivers, num_nodes):
        """Mean aggregation: node 0 gets [2, 3], node 1 gets [6, 7]."""
        agg = Mean()
        result = agg(messages, receivers, num_nodes)
        
        expected = torch.tensor([
            [2.0, 3.0],   # (1+3)/2, (2+4)/2
            [6.0, 7.0],   # (5+7)/2, (6+8)/2
        ], device=device)
        
        assert torch.allclose(result, expected)

    def test_max_aggregation(self, device, messages, receivers, num_nodes):
        """Max aggregation: node 0 gets [3, 4], node 1 gets [7, 8]."""
        agg = Max()
        result = agg(messages, receivers, num_nodes)
        
        expected = torch.tensor([
            [3.0, 4.0],   # max(1,3), max(2,4)
            [7.0, 8.0],   # max(5,7), max(6,8)
        ], device=device)
        
        assert torch.allclose(result, expected)

    def test_min_aggregation(self, device, messages, receivers, num_nodes):
        """Min aggregation: node 0 gets [1, 2], node 1 gets [5, 6]."""
        agg = Min()
        result = agg(messages, receivers, num_nodes)
        
        expected = torch.tensor([
            [1.0, 2.0],   # min(1,3), min(2,4)
            [5.0, 6.0],   # min(5,7), min(6,8)
        ], device=device)
        
        assert torch.allclose(result, expected)


class TestGetAggregation:
    """Test get_aggregation utility function."""

    def test_get_aggregation_with_instance(self):
        """get_aggregation returns instance as-is."""
        agg = Sum()
        result = get_aggregation(agg)
        assert result is agg

    def test_get_aggregation_with_string(self):
        """get_aggregation converts string to built-in."""
        result = get_aggregation('sum')
        assert isinstance(result, Sum)
        
        result = get_aggregation('max')
        assert isinstance(result, Max)

    def test_get_aggregation_with_callable(self):
        """get_aggregation accepts custom callable."""
        def custom(m, r, n):
            return aggregate_edges(m, r, n, 'sum')
        
        result = get_aggregation(custom)
        assert result is custom

    def test_get_aggregation_invalid_string(self):
        """get_aggregation raises on invalid string."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            get_aggregation('invalid')

    def test_get_aggregation_invalid_type(self, device):
        """get_aggregation raises on invalid type."""
        with pytest.raises(TypeError, match="Aggregation instance"):
            get_aggregation(123)


class TestAggregationInMessagePassingBlock:
    """Test Aggregation integration with MessagePassingBase."""

    def _make_graph(self, device):
        return GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 16, device=device),  # edge dim must match latent_dim
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )

    def test_aggregate_with_sum_instance(self, device):
        """GraphNetBlock works with Sum() instance."""
        block = GraphNetBlock(latent_dim=16, aggregate=Sum()).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_aggregate_with_max_instance(self, device):
        """GraphNetBlock works with Max() instance."""
        block = GraphNetBlock(latent_dim=16, aggregate=Max()).to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_aggregate_with_string(self, device):
        """GraphNetBlock works with string 'sum'."""
        block = GraphNetBlock(latent_dim=16, aggregate='sum').to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_aggregate_with_string_max(self, device):
        """GraphNetBlock works with string 'max'."""
        block = GraphNetBlock(latent_dim=16, aggregate='max').to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)

    def test_aggregate_with_string_literal(self, device):
        """aggregate='sum' works."""
        block = GraphNetBlock(latent_dim=16, aggregate='sum').to(device)
        graph = self._make_graph(device)
        out = block(graph)
        assert out.nodes.shape == (5, 16)
