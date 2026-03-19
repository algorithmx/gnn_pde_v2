"""Tests for edge feature assemblers.

These tests verify that each assembler correctly constructs edge features
from graph structure (sender/receiver nodes and edge attributes).
"""
import pytest
import torch

from gnn_pde_v2.components.edge_assemblers import (
    EdgeFeatureAssembler,
    NodeDifferenceAssembler,
    ConcatAssembler,
    DifferenceOnlyAssembler,
    ConcatWithEdgesAssembler,
)
from gnn_pde_v2.core import GraphsTuple


class TestNodeDifferenceAssembler:
    """Tests for NodeDifferenceAssembler — DGCNN default."""

    def test_out_dim(self):
        """Test out_dim property returns 2 * latent_dim."""
        assembler = NodeDifferenceAssembler(128)
        assert assembler.out_dim == 256  # 2 * 128

    def test_out_dim_different_values(self):
        """Test out_dim with different latent_dim values."""
        assert NodeDifferenceAssembler(16).out_dim == 32
        assert NodeDifferenceAssembler(64).out_dim == 128
        assert NodeDifferenceAssembler(256).out_dim == 512

    def test_forward_shape(self, device):
        """Test forward returns correct shape."""
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 3, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        assembler = NodeDifferenceAssembler(16)
        features = assembler(graph)
        
        assert features.shape == (8, 32)  # 8 edges, 32 dims (2*16)

    def test_forward_correctness(self, device):
        """Test forward computes [v_i; v_j - v_i] correctly."""
        nodes = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(2, 1, device=device),
            receivers=torch.tensor([1, 2], device=device),  # i = [1, 2]
            senders=torch.tensor([0, 1], device=device),    # j = [0, 1]
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        assembler = NodeDifferenceAssembler(2)
        features = assembler(graph)
        
        # Edge 0: i=1, j=0
        # v_i = [3, 4], v_j = [1, 2]
        # v_j - v_i = [-2, -2]
        # Result: [3, 4, -2, -2]
        expected_0 = torch.tensor([3.0, 4.0, -2.0, -2.0], device=device)
        
        # Edge 1: i=2, j=1
        # v_i = [5, 6], v_j = [3, 4]
        # v_j - v_i = [-2, -2]
        # Result: [5, 6, -2, -2]
        expected_1 = torch.tensor([5.0, 6.0, -2.0, -2.0], device=device)
        
        torch.testing.assert_close(features[0], expected_0)
        torch.testing.assert_close(features[1], expected_1)

    def test_invalid_latent_dim(self):
        """Test constructor rejects invalid latent_dim."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            NodeDifferenceAssembler(0)
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            NodeDifferenceAssembler(-1)


class TestConcatAssembler:
    """Tests for ConcatAssembler — simple concatenation."""

    def test_out_dim(self):
        """Test out_dim property returns 2 * latent_dim."""
        assembler = ConcatAssembler(128)
        assert assembler.out_dim == 256  # 2 * 128

    def test_forward_correctness(self, device):
        """Test forward computes [v_i; v_j] correctly."""
        nodes = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(2, 1, device=device),
            receivers=torch.tensor([1, 2], device=device),  # i = [1, 2]
            senders=torch.tensor([0, 1], device=device),    # j = [0, 1]
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        assembler = ConcatAssembler(2)
        features = assembler(graph)
        
        # Edge 0: i=1, j=0
        # v_i = [3, 4], v_j = [1, 2]
        # Result: [3, 4, 1, 2]
        expected_0 = torch.tensor([3.0, 4.0, 1.0, 2.0], device=device)
        
        # Edge 1: i=2, j=1
        # v_i = [5, 6], v_j = [3, 4]
        # Result: [5, 6, 3, 4]
        expected_1 = torch.tensor([5.0, 6.0, 3.0, 4.0], device=device)
        
        torch.testing.assert_close(features[0], expected_0)
        torch.testing.assert_close(features[1], expected_1)

    def test_invalid_latent_dim(self):
        """Test constructor rejects invalid latent_dim."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            ConcatAssembler(0)


class TestDifferenceOnlyAssembler:
    """Tests for DifferenceOnlyAssembler — difference only."""

    def test_out_dim(self):
        """Test out_dim property returns latent_dim."""
        assembler = DifferenceOnlyAssembler(128)
        assert assembler.out_dim == 128

    def test_forward_shape(self, device):
        """Test forward returns correct shape."""
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 3, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        assembler = DifferenceOnlyAssembler(16)
        features = assembler(graph)
        
        assert features.shape == (8, 16)  # 8 edges, 16 dims (latent_dim)

    def test_forward_correctness(self, device):
        """Test forward computes v_j - v_i correctly."""
        nodes = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(2, 1, device=device),
            receivers=torch.tensor([1, 2], device=device),
            senders=torch.tensor([0, 1], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        assembler = DifferenceOnlyAssembler(2)
        features = assembler(graph)
        
        # Edge 0: v_j - v_i = [1, 2] - [3, 4] = [-2, -2]
        expected_0 = torch.tensor([-2.0, -2.0], device=device)
        
        # Edge 1: v_j - v_i = [3, 4] - [5, 6] = [-2, -2]
        expected_1 = torch.tensor([-2.0, -2.0], device=device)
        
        torch.testing.assert_close(features[0], expected_0)
        torch.testing.assert_close(features[1], expected_1)


class TestConcatWithEdgesAssembler:
    """Tests for ConcatWithEdgesAssembler — includes edge attributes."""

    def test_out_dim(self):
        """Test out_dim property returns 2*latent_dim + edge_dim."""
        assembler = ConcatWithEdgesAssembler(latent_dim=128, edge_dim=3)
        assert assembler.out_dim == 259  # 2*128 + 3

    def test_out_dim_different_values(self):
        """Test out_dim with different values."""
        assert ConcatWithEdgesAssembler(16, 3).out_dim == 35   # 2*16 + 3
        assert ConcatWithEdgesAssembler(64, 1).out_dim == 129  # 2*64 + 1
        assert ConcatWithEdgesAssembler(256, 10).out_dim == 522  # 2*256 + 10

    def test_forward_shape(self, device):
        """Test forward returns correct shape."""
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 3, device=device),  # edge_dim = 3
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        assembler = ConcatWithEdgesAssembler(latent_dim=16, edge_dim=3)
        features = assembler(graph)
        
        assert features.shape == (8, 35)  # 8 edges, 35 dims (2*16 + 3)

    def test_forward_includes_edges(self, device):
        """Test forward correctly includes edge attributes."""
        nodes = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)
        
        edges = torch.tensor([
            [0.1, 0.2],
            [0.3, 0.4],
        ], device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=edges,
            receivers=torch.tensor([1, 2], device=device),
            senders=torch.tensor([0, 1], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        assembler = ConcatWithEdgesAssembler(latent_dim=2, edge_dim=2)
        features = assembler(graph)
        
        # Edge 0: v_i=[3,4], v_j=[1,2], v_j-v_i=[-2,-2], e=[0.1, 0.2]
        # Result: [3, 4, -2, -2, 0.1, 0.2]
        expected_0 = torch.tensor([3.0, 4.0, -2.0, -2.0, 0.1, 0.2], device=device)
        
        # Edge 1: v_i=[5,6], v_j=[3,4], v_j-v_i=[-2,-2], e=[0.3, 0.4]
        # Result: [5, 6, -2, -2, 0.3, 0.4]
        expected_1 = torch.tensor([5.0, 6.0, -2.0, -2.0, 0.3, 0.4], device=device)
        
        torch.testing.assert_close(features[0], expected_0)
        torch.testing.assert_close(features[1], expected_1)

    def test_invalid_latent_dim(self):
        """Test constructor rejects invalid latent_dim."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            ConcatWithEdgesAssembler(0, 3)

    def test_invalid_edge_dim(self):
        """Test constructor rejects invalid edge_dim."""
        with pytest.raises(ValueError, match="edge_dim must be positive"):
            ConcatWithEdgesAssembler(16, 0)


class TestEdgeFeatureAssemblerABC:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that EdgeFeatureAssembler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EdgeFeatureAssembler()

    def test_subclass_must_implement(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteAssembler(EdgeFeatureAssembler):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAssembler()


class TestAssemblerIntegration:
    """Integration tests with actual EdgeConvBlock usage."""

    def test_assembler_with_edge_conv_block(self, device):
        """Test that assemblers work correctly in EdgeConvBlock."""
        from gnn_pde_v2.components import EdgeConvBlock
        
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(5, 16, device=device),
            edges=torch.randn(8, 3, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        # Test each assembler type
        assemblers = [
            NodeDifferenceAssembler(16),
            ConcatAssembler(16),
            DifferenceOnlyAssembler(16),
        ]
        
        for assembler in assemblers:
            block = EdgeConvBlock(
                latent_dim=16,
                edge_assembler=assembler,
            ).to(device)
            
            out = block(graph)
            assert out.nodes.shape == (5, 16)
