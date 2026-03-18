"""
Comprehensive tests for low-rank edge-conditioned convolution.

Tests the symmetric low-rank approximation W_e ≈ U_e · U_e^T which provides
memory-efficient message passing for large latent dimensions.

Reference: Original implementation in /home/dabajabaza/Documents/Workspace/MoM/Projects/train/gnn_solver/nn_customized.py
"""

import pytest
import torch
import torch.nn as nn
from functools import partial

from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import EdgeConditionedConvBlock, GraphNetProcessor


# ========== Test Tolerance Constants ==========
TOLERANCE_STRICT = 1e-8
TOLERANCE_GRADIENT = 1e-6
TOLERANCE_NUMERICAL = 1e-5


# ========== Fixtures ==========

@pytest.fixture
def sample_graph(device, n_nodes=50, n_edges=200, latent_dim=64, edge_dim=7):
    """Create a sample graph for testing."""
    torch.manual_seed(42)
    return GraphsTuple.from_flat(
        nodes=torch.randn(n_nodes, latent_dim, device=device),
        edges=torch.randn(n_edges, edge_dim, device=device),
        senders=torch.randint(0, n_nodes, (n_edges,), device=device),
        receivers=torch.randint(0, n_nodes, (n_edges,), device=device),
        n_node=torch.tensor([n_nodes], device=device),
        n_edge=torch.tensor([n_edges], device=device),
    )


# ========== Correctness Tests ==========

class TestLowRankCorrectness:
    """Test correctness of low-rank message passing."""

    def test_output_shape(self, device):
        """Test that low-rank produces correct output shape."""
        block = EdgeConditionedConvBlock(
            latent_dim=64, edge_latent_dim=7,
            edge_weight_type='low_rank', low_rank=8,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(50, 64, device=device),
            edges=torch.randn(100, 7, device=device),
            senders=torch.randint(0, 50, (100,), device=device),
            receivers=torch.randint(0, 50, (100,), device=device),
            n_node=torch.tensor([50], device=device),
            n_edge=torch.tensor([100], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (50, 64)

    def test_finite_outputs(self, device):
        """Test that outputs are finite (no NaN or Inf)."""
        block = EdgeConditionedConvBlock(
            latent_dim=32, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(20, 32, device=device),
            edges=torch.randn(50, 8, device=device),
            senders=torch.randint(0, 20, (50,), device=device),
            receivers=torch.randint(0, 20, (50,), device=device),
            n_node=torch.tensor([20], device=device),
            n_edge=torch.tensor([50], device=device),
        )
        
        out = block(graph)
        assert torch.all(torch.isfinite(out.nodes))

    def test_edgeless_graph(self, device):
        """Test handling of graphs with no edges."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
            root_weight=False, bias=False,
        ).to(device)
        
        # Graph with no edges
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            n_node=torch.tensor([5], device=device),
        )
        
        out = block(graph)
        # With no edges and no root/bias, output should be zeros
        assert out.nodes.shape == (5, 16)
        assert torch.allclose(out.nodes, torch.zeros_like(out.nodes))

    def test_isolated_nodes(self, device):
        """Test handling of isolated nodes (no incoming edges)."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
            aggregate='mean',
        ).to(device)
        
        # Node 4 and 5 have no incoming edges
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(6, 16, device=device),
            edges=torch.randn(4, 8, device=device),
            senders=torch.tensor([0, 1, 2, 3], device=device),
            receivers=torch.tensor([1, 2, 3, 0], device=device),
            n_node=torch.tensor([6], device=device),
            n_edge=torch.tensor([4], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (6, 16)
        # All outputs should be finite (mean aggregation handles isolated nodes)
        assert torch.all(torch.isfinite(out.nodes))

    def test_self_loops(self, device):
        """Test handling of self-loops."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(5, 8, device=device),
            senders=torch.tensor([0, 1, 2, 3, 4], device=device),
            receivers=torch.tensor([0, 1, 2, 3, 4], device=device),  # Self-loops
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([5], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (5, 16)
        assert torch.all(torch.isfinite(out.nodes))


# ========== Memory Efficiency Tests ==========

class TestLowRankMemoryEfficiency:
    """Test memory efficiency of low-rank approximation."""

    def test_memory_reduction_calculation(self, device):
        """Verify memory reduction ratio calculation."""
        d = 64
        r = 8
        
        memory_full = d * d      # 4096
        memory_lowrank = d * r   # 512
        reduction = memory_full / memory_lowrank
        
        assert reduction == 8.0

    def test_parameter_count_reduction(self, device):
        """Test actual parameter count reduction."""
        d = 64
        r = 8
        hidden_dim = 32
        edge_dim = 7
        
        block_full = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=edge_dim, hidden_dim=hidden_dim,
            edge_weight_type='full', root_weight=False, bias=False,
        ).to(device)
        
        block_lowrank = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=edge_dim, hidden_dim=hidden_dim,
            edge_weight_type='low_rank', low_rank=r,
            root_weight=False, bias=False,
        ).to(device)
        
        # Count parameters in edge weight net only
        params_full = sum(p.numel() for p in block_full.edge_weight_net.parameters())
        params_lowrank = sum(p.numel() for p in block_lowrank.edge_weight_net.parameters())
        
        # Low-rank should have fewer parameters in edge weight net
        # due to smaller output dimension
        assert params_lowrank < params_full

    @pytest.mark.parametrize("d,r,expected_reduction", [
        (64, 8, 8.0),
        (64, 16, 4.0),
        (128, 16, 8.0),
        (128, 32, 4.0),
        (32, 4, 8.0),
    ])
    def test_various_rank_reductions(self, device, d, r, expected_reduction):
        """Test memory reduction for various (d, r) combinations."""
        memory_full = d * d
        memory_lowrank = d * r
        actual_reduction = memory_full / memory_lowrank
        assert actual_reduction == expected_reduction


# ========== Gradient Tests ==========

class TestLowRankGradients:
    """Test gradient computation in low-rank mode."""

    def test_gradient_flow(self, device):
        """Test that gradients flow through all parameters."""
        block = EdgeConditionedConvBlock(
            latent_dim=32, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=8,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 32, device=device, requires_grad=True),
            edges=torch.randn(20, 8, device=device),
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        loss = out.nodes.sum()
        loss.backward()
        
        # Check edge weight net gradients
        edge_net_has_grad = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in block.edge_weight_net.parameters()
        )
        assert edge_net_has_grad, "Edge weight net should have non-zero gradients"
        
        # Check root weight gradients
        assert block.root.grad is not None
        
        # Check bias gradients
        assert block.bias.grad is not None

    def test_gradient_numerical_stability(self, device):
        """Test gradient stability with different loss functions."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(10, 8, device=device),
            senders=torch.randint(0, 5, (10,), device=device),
            receivers=torch.randint(0, 5, (10,), device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([10], device=device),
        )
        
        loss_fns = [
            lambda x: x.sum(),
            lambda x: x.mean(),
            lambda x: (x ** 2).sum(),
            lambda x: torch.abs(x).sum(),
        ]
        
        for loss_fn in loss_fns:
            block.zero_grad()
            out = block(graph)
            loss = loss_fn(out.nodes)
            loss.backward()
            
            # Check all gradients are finite
            for p in block.parameters():
                if p.grad is not None:
                    assert torch.all(torch.isfinite(p.grad)), "Gradients should be finite"


# ========== Configuration Tests ==========

class TestLowRankConfiguration:
    """Test various low-rank configurations."""

    @pytest.mark.parametrize("latent_dim,low_rank", [
        (16, 4),
        (32, 8),
        (64, 8),
        (64, 16),
        (128, 16),
        (128, 32),
    ])
    def test_valid_configurations(self, device, latent_dim, low_rank):
        """Test various valid (latent_dim, low_rank) combinations."""
        block = EdgeConditionedConvBlock(
            latent_dim=latent_dim, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=low_rank,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, latent_dim, device=device),
            edges=torch.randn(20, 8, device=device),
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (10, latent_dim)

    def test_invalid_rank_zero_raises(self, device):
        """Test that rank=0 raises ValueError."""
        with pytest.raises(ValueError, match="low_rank must be positive"):
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='low_rank', low_rank=0,
            )

    def test_invalid_rank_negative_raises(self, device):
        """Test that negative rank raises ValueError."""
        with pytest.raises(ValueError, match="low_rank must be positive"):
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='low_rank', low_rank=-1,
            )

    def test_invalid_rank_too_large_raises(self, device):
        """Test that rank > latent_dim raises ValueError."""
        with pytest.raises(ValueError, match="low_rank .* must be <= latent_dim"):
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='low_rank', low_rank=32,
            )

    def test_rank_equal_to_latent_dim(self, device):
        """Test that rank == latent_dim is valid (though not efficient)."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=16,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(10, 8, device=device),
            senders=torch.randint(0, 5, (10,), device=device),
            receivers=torch.randint(0, 5, (10,), device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([10], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (5, 16)


# ========== Aggregation Tests ==========

class TestLowRankAggregation:
    """Test low-rank with different aggregation methods."""

    @pytest.mark.parametrize("aggregate", ['sum', 'mean'])
    def test_aggregation_methods(self, device, aggregate):
        """Test low-rank with sum and mean aggregation (primary use cases)."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
            aggregate=aggregate,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device),
            edges=torch.randn(20, 8, device=device),
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        assert out.nodes.shape == (10, 16)
        assert torch.all(torch.isfinite(out.nodes))


# ========== Integration Tests ==========

class TestLowRankIntegration:
    """Test low-rank in processor stacks."""

    def test_in_graphnet_processor(self, device):
        """Test low-rank blocks in GraphNetProcessor."""
        factory = partial(
            EdgeConditionedConvBlock,
            latent_dim=32, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=8,
            aggregate='mean',
        )
        
        processor = GraphNetProcessor(
            latent_dim=32, n_layers=3,
            block_factory=factory, residual=False,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(20, 32, device=device),
            edges=torch.randn(50, 8, device=device),
            senders=torch.randint(0, 20, (50,), device=device),
            receivers=torch.randint(0, 20, (50,), device=device),
            n_node=torch.tensor([20], device=device),
            n_edge=torch.tensor([50], device=device),
        )
        
        out = processor(graph)
        assert out.nodes.shape == (20, 32)
        assert torch.all(torch.isfinite(out.nodes))

    def test_mixed_full_and_low_rank_layers(self, device):
        """Test processor with mix of full and low-rank layers."""
        # Create processor manually with mixed blocks
        blocks = nn.ModuleList([
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='full',  # First layer full
            ),
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='low_rank', low_rank=4,  # Second layer low-rank
            ),
            EdgeConditionedConvBlock(
                latent_dim=16, edge_latent_dim=8,
                edge_weight_type='low_rank', low_rank=4,  # Third layer low-rank
            ),
        ]).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device),
            edges=torch.randn(20, 8, device=device),
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        # Forward through all blocks
        out = graph
        for block in blocks:
            out = block(out)
        
        assert out.nodes.shape == (10, 16)


# ========== Numerical Precision Tests ==========

class TestLowRankNumericalPrecision:
    """Test numerical precision of low-rank computations."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_different_precisions(self, device, dtype):
        """Test low-rank with float32 and float64."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device).to(dtype)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device, dtype=dtype),
            edges=torch.randn(20, 8, device=device, dtype=dtype),
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        assert out.nodes.dtype == dtype
        assert torch.all(torch.isfinite(out.nodes))

    def test_large_input_values(self, device):
        """Test stability with large input values."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device) * 1e3,
            edges=torch.randn(20, 8, device=device) * 1e3,
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        assert torch.all(torch.isfinite(out.nodes))

    def test_small_input_values(self, device):
        """Test stability with small input values."""
        block = EdgeConditionedConvBlock(
            latent_dim=16, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=4,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(10, 16, device=device) * 1e-6,
            edges=torch.randn(20, 8, device=device) * 1e-6,
            senders=torch.randint(0, 10, (20,), device=device),
            receivers=torch.randint(0, 10, (20,), device=device),
            n_node=torch.tensor([10], device=device),
            n_edge=torch.tensor([20], device=device),
        )
        
        out = block(graph)
        assert torch.all(torch.isfinite(out.nodes))


# ========== Performance Comparison Tests ==========

class TestLowRankPerformance:
    """Compare performance characteristics of low-rank vs full-rank."""

    def test_forward_pass_time(self, device):
        """Compare forward pass time (not a strict test, just for information)."""
        import time
        
        d = 64
        r = 8
        
        block_full = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=8,
            edge_weight_type='full', root_weight=False, bias=False,
        ).to(device)
        
        block_lowrank = EdgeConditionedConvBlock(
            latent_dim=d, edge_latent_dim=8,
            edge_weight_type='low_rank', low_rank=r,
            root_weight=False, bias=False,
        ).to(device)
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(100, d, device=device),
            edges=torch.randn(500, 8, device=device),
            senders=torch.randint(0, 100, (500,), device=device),
            receivers=torch.randint(0, 100, (500,), device=device),
            n_node=torch.tensor([100], device=device),
            n_edge=torch.tensor([500], device=device),
        )
        
        # Warmup
        for _ in range(5):
            _ = block_full(graph)
            _ = block_lowrank(graph)
        
        # Time full-rank
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(10):
            _ = block_full(graph)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_full = time.perf_counter() - start
        
        # Time low-rank
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        for _ in range(10):
            _ = block_lowrank(graph)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_lowrank = time.perf_counter() - start
        
        # Just verify both run without error and produce valid outputs
        out_full = block_full(graph)
        out_lowrank = block_lowrank(graph)
        
        assert torch.all(torch.isfinite(out_full.nodes))
        assert torch.all(torch.isfinite(out_lowrank.nodes))
        
        # Print timing info (not a strict assertion)
        print(f"\nTiming (10 iterations): Full-rank={time_full:.4f}s, Low-rank={time_lowrank:.4f}s")
