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
from gnn_pde_v2.components import (
    make_mlp_encoder,
    Residual, ResidualBlock, GatedResidual, PreNormResidual,
    ResidualSequence, SkipConnection, make_residual,
    GraphNetBlock, GraphNetProcessor,
    MLPDecoder, IndependentMLPDecoder,
    ProbeDecoder,
)


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
        for act in ['relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'sin']:
            mlp = MLP(10, 5, [10], activation=act).to(device)
            x = torch.randn(3, 10, device=device)
            out = mlp(x)
            assert out.shape == (3, 5)
    
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
    
    def test_make_mlp_encoder(self, device):
        """Test make_mlp_encoder helper."""
        encoder = make_mlp_encoder(10, 20, hidden_dim=15, num_layers=3).to(device)
        x = torch.randn(3, 10, device=device)
        
        out = encoder(x)
        
        assert out.shape == (3, 20)


class TestResidual:
    """Test Residual wrapper (backward-compatible simple interface)."""

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


class TestResidualBlock:
    """Test ResidualBlock with different residual types."""

    def test_add_residual(self, device):
        """Test simple add residual."""
        module = nn.Linear(10, 10).to(device)
        block = ResidualBlock(module, residual_type='add').to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        expected = x + module(x)
        assert torch.allclose(out, expected)

    def test_scaled_residual(self, device):
        """Test scaled residual with fixed scale."""
        module = nn.Linear(10, 10).to(device)
        block = ResidualBlock(module, residual_type='scaled', scale=0.5).to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        expected = x + 0.5 * module(x)
        assert torch.allclose(out, expected)

    def test_scaled_residual_learnable(self, device):
        """Test scaled residual with learnable scale."""
        module = nn.Linear(10, 10).to(device)
        block = ResidualBlock(module, residual_type='scaled', learnable_scale=True).to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        # Should start with scale=1.0
        assert block.scale.item() == 1.0
        assert out.shape == x.shape

    def test_none_residual(self, device):
        """Test no residual (pass-through)."""
        module = nn.Linear(10, 10).to(device)
        block = ResidualBlock(module, residual_type='none').to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        expected = module(x)
        assert torch.allclose(out, expected)

    def test_shape_mismatch_raises(self, device):
        """Test that shape mismatch raises error."""
        module = nn.Linear(10, 20).to(device)  # Different output dim
        block = ResidualBlock(module, residual_type='add').to(device)

        x = torch.randn(3, 10, device=device)
        with pytest.raises(ValueError, match="shapes don't match"):
            block(x)


class TestGatedResidual:
    """Test GatedResidual with learnable gate."""

    def test_gated_residual(self, device):
        """Test gated residual forward pass."""
        module = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        block = GatedResidual(module, gate_activation='sigmoid').to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        assert out.shape == x.shape
        # Output should be different from both input and module output
        assert not torch.allclose(out, x)

    def test_gate_bias_initialization(self, device):
        """Test gate bias affects output."""
        module = nn.Linear(10, 10)
        # High bias = more weight on residual branch
        block_high = GatedResidual(module, gate_bias=5.0).to(device)
        # Low bias = less weight on residual branch
        block_low = GatedResidual(module, gate_bias=-5.0).to(device)

        x = torch.randn(3, 10, device=device)
        out_high = block_high(x)
        out_low = block_low(x)

        # Outputs should differ due to different gate biases
        assert not torch.allclose(out_high, out_low)


class TestPreNormResidual:
    """Test PreNormResidual (Transformer-style)."""

    def test_prenorm_residual(self, device):
        """Test pre-normalization residual."""
        module = nn.Linear(10, 10).to(device)
        block = PreNormResidual(module, dim=10).to(device)

        x = torch.randn(3, 10, device=device)
        out = block(x)

        expected = x + module(block.norm(x))
        assert torch.allclose(out, expected)

    def test_prenorm_with_attention(self, device):
        """Test with multi-head attention module (wrapped for self-attention)."""

        class SelfAttention(nn.Module):
            """Wrapper for self-attention that handles the query/key/value API."""
            def __init__(self, dim, heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

            def forward(self, x):
                out, _ = self.attn(x, x, x)
                return out

        block = PreNormResidual(SelfAttention(64, 4), dim=64).to(device)

        x = torch.randn(2, 10, 64, device=device)
        out = block(x)

        assert out.shape == x.shape


class TestResidualSequence:
    """Test ResidualSequence."""

    def test_sequence_forward(self, device):
        """Test forward through multiple residual blocks."""
        blocks = [nn.Linear(10, 10) for _ in range(3)]
        seq = ResidualSequence(blocks, residual_type='add').to(device)

        x = torch.randn(3, 10, device=device)
        out = seq(x)

        assert out.shape == x.shape
        assert len(seq) == 3

    def test_sequence_indexing(self, device):
        """Test indexing into sequence."""
        blocks = [nn.Linear(10, 10) for _ in range(3)]
        seq = ResidualSequence(blocks).to(device)

        assert isinstance(seq[0], ResidualBlock)
        assert seq[0].residual_type == 'add'


class TestSkipConnection:
    """Test SkipConnection with projection."""

    def test_skip_with_projection(self, device):
        """Test skip connection with dimension change."""
        module = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        proj = nn.Linear(64, 128)
        block = SkipConnection(module, projection=proj, aggregation='add').to(device)

        x = torch.randn(3, 64, device=device)
        out = block(x)

        assert out.shape == (3, 128)
        expected = proj(x) + module(x)
        assert torch.allclose(out, expected)

    def test_skip_concat(self, device):
        """Test skip connection with concatenation."""
        module = nn.Linear(64, 64)
        block = SkipConnection(module, aggregation='concat').to(device)

        x = torch.randn(3, 64, device=device)
        out = block(x)

        assert out.shape == (3, 128)  # 64 + 64

    def test_skip_none(self, device):
        """Test skip connection with no aggregation."""
        module = nn.Linear(64, 128)
        block = SkipConnection(module, aggregation='none').to(device)

        x = torch.randn(3, 64, device=device)
        out = block(x)

        assert out.shape == (3, 128)
        assert torch.allclose(out, module(x))


class TestMakeResidual:
    """Test make_residual factory function."""

    def test_make_add(self, device):
        """Test factory creates ResidualBlock with add."""
        module = nn.Linear(10, 10)
        block = make_residual(module, 'add')

        assert isinstance(block, ResidualBlock)
        assert block.residual_type == 'add'

    def test_make_scaled(self, device):
        """Test factory creates ResidualBlock with scaled."""
        module = nn.Linear(10, 10)
        block = make_residual(module, 'scaled', scale=0.5)

        assert isinstance(block, ResidualBlock)
        assert block.residual_type == 'scaled'

    def test_make_gated(self, device):
        """Test factory creates GatedResidual."""
        module = nn.Linear(10, 10)
        block = make_residual(module, 'gated')

        assert isinstance(block, GatedResidual)

    def test_make_prenorm(self, device):
        """Test factory creates PreNormResidual."""
        module = nn.Linear(10, 10)
        block = make_residual(module, 'prenorm', dim=10)

        assert isinstance(block, PreNormResidual)

    def test_make_none(self, device):
        """Test factory returns module unchanged."""
        module = nn.Linear(10, 10)
        block = make_residual(module, 'none')

        assert block is module  # Should return same object


class TestGraphNetBlock:
    """Test GraphNetBlock."""
    
    def test_forward(self, device):
        """Test basic forward pass."""
        block = GraphNetBlock(
            node_dim=16,
            edge_dim=8,
            global_dim=None,
        ).to(device)
        
        graph = GraphsTuple(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 8, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        out = block(graph)
        
        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 8)
    
    def test_with_globals(self, device):
        """Test with global features."""
        block = GraphNetBlock(
            node_dim=16,
            edge_dim=8,
            global_dim=4,
        ).to(device)
        
        graph = GraphsTuple(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 8, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            globals=torch.randn(1, 4, device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        out = block(graph)
        
        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 8)


class TestGraphNetProcessor:
    """Test GraphNetProcessor."""
    
    def test_forward(self, device):
        """Test basic forward pass."""
        processor = GraphNetProcessor(
            node_dim=16,
            edge_dim=8,
            n_layers=3,
        ).to(device)
        
        graph = GraphsTuple(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 8, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        out = processor(graph)
        
        assert out.nodes.shape == (5, 16)
        assert out.edges.shape == (8, 8)


class TestMLPDecoder:
    """Test MLPDecoder."""
    
    def test_forward(self, device):
        """Test basic forward pass."""
        decoder = MLPDecoder(
            node_dim=16,
            out_dim=3,
        ).to(device)
        
        graph = GraphsTuple(
            nodes=torch.randn(5, 16, device=device),
            n_node=torch.tensor([5], device=device),
        )
        
        out = decoder(graph)
        
        assert out.shape == (5, 3)


class TestIndependentMLPDecoder:
    """Test IndependentMLPDecoder."""
    
    def test_forward(self, device):
        """Test multi-output forward pass."""
        decoder = IndependentMLPDecoder(
            node_dim=16,
            out_dims=[3, 5, 2],
        ).to(device)
        
        graph = GraphsTuple(
            nodes=torch.randn(5, 16, device=device),
            n_node=torch.tensor([5], device=device),
        )
        
        out = decoder(graph)
        
        assert len(out) == 3
        assert out[0].shape == (5, 3)
        assert out[1].shape == (5, 5)
        assert out[2].shape == (5, 2)


class TestProbeDecoder:
    """Test ProbeDecoder."""
    
    def test_forward(self, device):
        """Test probe-based decoding."""
        decoder = ProbeDecoder(
            node_dim=16,
            edge_dim=8,
            out_dim=3,
            k_nearest=3,
        ).to(device)
        
        # Source graph
        graph = GraphsTuple(
            nodes=torch.randn(10, 16, device=device),
            positions=torch.randn(10, 2, device=device),
            n_node=torch.tensor([10], device=device),
        )
        
        # Query positions
        query_positions = torch.randn(5, 2, device=device)
        
        out = decoder(graph, query_positions)
        
        assert out.shape == (5, 3)
