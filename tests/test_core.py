"""
Tests for core components.

Core includes: GraphsTuple, batch_graphs, unbatch_graphs, and functional utilities.
"""

import pytest
import torch
from dataclasses import replace

from gnn_pde_v2 import GraphsTuple, batch_graphs, unbatch_graphs
from gnn_pde_v2.core import scatter_sum, scatter_mean, scatter_max
from gnn_pde_v2.core.functional import aggregate_edges, broadcast_nodes_to_edges


class TestGraphsTuple:
    """Test GraphsTuple dataclass."""
    
    def test_creation(self, device):
        """Test basic GraphsTuple creation."""
        graph = GraphsTuple(
            nodes=torch.randn(5, 10, device=device),
            edges=torch.randn(8, 4, device=device),
            receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
            senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
            n_node=torch.tensor([5], device=device),
            n_edge=torch.tensor([8], device=device),
        )
        
        assert graph.num_nodes == 5
        assert graph.num_edges == 8
        assert graph.num_graphs == 1
    
    def test_to_device(self, device):
        """Test moving to device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        graph = GraphsTuple(
            nodes=torch.randn(3, 5),
            n_node=torch.tensor([3]),
        )
        
        graph_cuda = graph.to('cuda')
        assert graph_cuda.nodes.device.type == 'cuda'
    
    def test_replace(self, device):
        """Test dataclasses.replace()."""
        graph = GraphsTuple(
            nodes=torch.randn(3, 5, device=device),
            n_node=torch.tensor([3], device=device),
        )
        
        new_nodes = torch.randn(3, 10, device=device)
        new_graph = replace(graph, nodes=new_nodes)
        
        assert new_graph.nodes.shape == (3, 10)
        assert graph.nodes.shape == (3, 5)  # Original unchanged


class TestBatching:
    """Test graph batching/unbatching."""
    
    def test_batch_single(self, device):
        """Test batching a single graph."""
        graph = GraphsTuple(
            nodes=torch.randn(3, 5, device=device),
            edges=torch.randn(4, 2, device=device),
            receivers=torch.tensor([1, 2, 0, 1], device=device),
            senders=torch.tensor([0, 1, 2, 0], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([4], device=device),
        )
        
        batched = batch_graphs([graph])
        
        assert batched.num_nodes == 3
        assert batched.num_edges == 4
        assert batched.num_graphs == 1
    
    def test_batch_multiple(self, device):
        """Test batching multiple graphs."""
        g1 = GraphsTuple(
            nodes=torch.randn(3, 5, device=device),
            edges=torch.randn(4, 2, device=device),
            receivers=torch.tensor([1, 2, 0, 1], device=device),
            senders=torch.tensor([0, 1, 2, 0], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([4], device=device),
        )
        
        g2 = GraphsTuple(
            nodes=torch.randn(2, 5, device=device),
            edges=torch.randn(2, 2, device=device),
            receivers=torch.tensor([1, 0], device=device),
            senders=torch.tensor([0, 1], device=device),
            n_node=torch.tensor([2], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        batched = batch_graphs([g1, g2])
        
        assert batched.num_nodes == 5
        assert batched.num_edges == 6
        assert batched.num_graphs == 2
        
        # Check receivers are offset correctly
        assert batched.receivers[4] == 1 + 3  # Second graph offset by 3
        assert batched.receivers[5] == 0 + 3
    
    def test_unbatch(self, device):
        """Test unbatching."""
        g1 = GraphsTuple(
            nodes=torch.randn(3, 5, device=device),
            edges=torch.randn(4, 2, device=device),
            receivers=torch.tensor([1, 2, 0, 1], device=device),
            senders=torch.tensor([0, 1, 2, 0], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([4], device=device),
        )
        
        g2 = GraphsTuple(
            nodes=torch.randn(2, 5, device=device),
            edges=torch.randn(2, 2, device=device),
            receivers=torch.tensor([1, 0], device=device),
            senders=torch.tensor([0, 1], device=device),
            n_node=torch.tensor([2], device=device),
            n_edge=torch.tensor([2], device=device),
        )
        
        batched = batch_graphs([g1, g2])
        unbatched = unbatch_graphs(batched)
        
        assert len(unbatched) == 2
        assert unbatched[0].num_nodes == 3
        assert unbatched[1].num_nodes == 2
    
    def test_batch_unbatch_roundtrip(self, device):
        """Test batch -> unbatch roundtrip."""
        g1 = GraphsTuple(
            nodes=torch.randn(3, 5, device=device),
            edges=torch.randn(4, 2, device=device),
            receivers=torch.tensor([1, 2, 0, 1], device=device),
            senders=torch.tensor([0, 1, 2, 0], device=device),
            n_node=torch.tensor([3], device=device),
            n_edge=torch.tensor([4], device=device),
        )
        
        batched = batch_graphs([g1])
        unbatched = unbatch_graphs(batched)
        
        assert len(unbatched) == 1
        assert torch.allclose(unbatched[0].nodes, g1.nodes)


class TestScatterOperations:
    """Test scatter operations."""
    
    def test_scatter_sum(self, device):
        """Test scatter_sum."""
        src = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        index = torch.tensor([0, 0, 1, 1], device=device)
        
        out = scatter_sum(src, index, dim=0, dim_size=2)
        
        expected = torch.tensor([[3.0], [7.0]], device=device)
        assert torch.allclose(out, expected)
    
    def test_scatter_mean(self, device):
        """Test scatter_mean."""
        src = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        index = torch.tensor([0, 0, 1, 1], device=device)
        
        out = scatter_mean(src, index, dim=0, dim_size=2)
        
        expected = torch.tensor([[1.5], [3.5]], device=device)
        assert torch.allclose(out, expected)
    
    def test_scatter_max(self, device):
        """Test scatter_max."""
        src = torch.tensor([[1.0], [3.0], [2.0], [4.0]], device=device)
        index = torch.tensor([0, 0, 1, 1], device=device)
        
        out = scatter_max(src, index, dim=0, dim_size=2)
        
        expected = torch.tensor([[3.0], [4.0]], device=device)
        assert torch.allclose(out, expected)
    
    def test_scatter_multi_dim(self, device):
        """Test scatter with multi-dimensional features."""
        src = torch.randn(10, 5, 3, device=device)
        index = torch.randint(0, 4, (10,), device=device)
        
        out = scatter_sum(src, index, dim=0, dim_size=4)
        
        assert out.shape == (4, 5, 3)


class TestFunctionalUtilities:
    """Test functional utilities."""
    
    def test_aggregate_edges(self, device):
        """Test aggregate_edges."""
        edge_features = torch.randn(8, 16, device=device)
        receivers = torch.tensor([0, 1, 0, 2, 1, 0, 2, 1], device=device)
        num_nodes = 3
        
        out = aggregate_edges(edge_features, receivers, num_nodes, method='sum')
        
        assert out.shape == (3, 16)
    
    def test_broadcast_nodes_to_edges(self, device):
        """Test broadcast_nodes_to_edges."""
        node_features = torch.randn(4, 8, device=device)
        senders = torch.tensor([0, 1, 2, 3], device=device)
        receivers = torch.tensor([1, 2, 3, 0], device=device)
        
        sender_feat, receiver_feat = broadcast_nodes_to_edges(
            node_features, senders, receivers
        )
        
        assert sender_feat.shape == (4, 8)
        assert receiver_feat.shape == (4, 8)
        assert torch.allclose(sender_feat[0], node_features[0])
        assert torch.allclose(receiver_feat[0], node_features[1])
