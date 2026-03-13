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
    Residual,
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
    """Test GraphNetBlock."""

    def test_forward(self, device):
        """Test basic forward pass."""
        block = GraphNetBlock(
            latent_dim=16,
            global_latent_dim=None,
        ).to(device)

        graph = GraphsTuple(
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

    def test_with_globals(self, device):
        """Test with global features."""
        block = GraphNetBlock(
            latent_dim=16,
            global_latent_dim=4,
        ).to(device)

        graph = GraphsTuple(
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


class TestGraphNetProcessor:
    """Test GraphNetProcessor."""

    def test_forward(self, device):
        """Test basic forward pass."""
        processor = GraphNetProcessor(
            latent_dim=16,
            n_layers=3,
        ).to(device)

        graph = GraphsTuple(
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


class TestMLPDecoder:
    """Test MLPDecoder."""

    def test_forward(self, device):
        """Test basic forward pass."""
        decoder = MLPDecoder(
            latent_dim=16,
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
            latent_dim=16,
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
            latent_dim=16,
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
