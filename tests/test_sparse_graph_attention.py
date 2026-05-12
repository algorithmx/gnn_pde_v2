"""
Tests for SparseGraphAttention temperature reshape fix.
"""

import pytest
import torch

from gnn_pde_v2.components.attention import SparseGraphAttention


class TestSparseGraphAttentionTemperatureReshape:
    """Tests for SparseGraphAttention temperature reshape correctness."""

    @pytest.fixture
    def num_nodes(self):
        return 10

    @pytest.fixture
    def num_edges(self):
        return 30

    @pytest.fixture
    def dim(self):
        return 16

    @pytest.fixture
    def n_heads(self):
        return 4

    @pytest.fixture
    def x(self, num_nodes, dim):
        return torch.randn(num_nodes, dim)

    @pytest.fixture
    def senders(self, num_edges, num_nodes):
        return torch.randint(0, num_nodes, (num_edges,))

    @pytest.fixture
    def receivers(self, num_edges, num_nodes):
        return torch.randint(0, num_nodes, (num_edges,))

    def test_fixed_temperature_reshape(self, x, senders, receivers):
        """Test SparseGraphAttention with fixed temperature produces correct output shape."""
        attn = SparseGraphAttention(
            dim=16,
            n_heads=4,
            temperature_mode='fixed',
        )
        out = attn(x, senders, receivers)
        assert out.shape == x.shape

    def test_learnable_scalar_temperature_reshape(self, x, senders, receivers):
        """Test SparseGraphAttention with learnable scalar temperature."""
        attn = SparseGraphAttention(
            dim=16,
            n_heads=4,
            temperature_mode='learnable_scalar',
        )
        out = attn(x, senders, receivers)
        assert out.shape == x.shape

    def test_per_head_temperature_reshape(self, x, senders, receivers):
        """Test SparseGraphAttention with per-head temperature."""
        attn = SparseGraphAttention(
            dim=16,
            n_heads=4,
            temperature_mode='per_head',
        )
        out = attn(x, senders, receivers)
        assert out.shape == x.shape

    def test_annealed_temperature_reshape(self, x, senders, receivers):
        """Test SparseGraphAttention with annealed temperature."""
        attn = SparseGraphAttention(
            dim=16,
            n_heads=4,
            temperature_mode='annealed',
        )
        out = attn(x, senders, receivers)
        assert out.shape == x.shape

    def test_adaptive_temperature_raises_error(self, x, senders, receivers):
        """Test SparseGraphAttention raises error for adaptive temperature."""
        attn = SparseGraphAttention(
            dim=16,
            n_heads=4,
            temperature_mode='adaptive',
        )
        with pytest.raises(ValueError, match="does not support.*adaptive"):
            attn(x, senders, receivers)

    def test_output_shape_preserved(self, x, senders, receivers):
        """Test output shape matches input shape across all supported modes."""
        for mode in ['fixed', 'learnable_scalar', 'per_head', 'annealed']:
            attn = SparseGraphAttention(
                dim=16,
                n_heads=4,
                temperature_mode=mode,
            )
            out = attn(x, senders, receivers)
            assert out.shape == x.shape, f"Shape mismatch for mode={mode}"
