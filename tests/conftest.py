"""
Pytest fixtures and configuration.
"""

import pytest
import torch


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_pydantic: mark test as requiring pydantic"
    )
    config.addinivalue_line(
        "markers", "requires_torch_scatter: mark test as requiring torch_scatter"
    )


@pytest.fixture
def device():
    """Device fixture - returns cuda if available, else cpu."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_graph(device):
    """Create a sample graph for testing."""
    from gnn_pde_v2 import GraphsTuple
    
    return GraphsTuple(
        nodes=torch.randn(5, 10, device=device),
        edges=torch.randn(8, 4, device=device),
        receivers=torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], device=device),
        senders=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device),
        n_node=torch.tensor([5], device=device),
        n_edge=torch.tensor([8], device=device),
    )


@pytest.fixture
def batched_graphs(device):
    """Create batched graphs for testing."""
    from gnn_pde_v2 import GraphsTuple
    
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
    
    return [g1, g2]


@pytest.fixture
def sample_graph_100(device):
    """Create a sample graph with 100 nodes for pooling tests."""
    from gnn_pde_v2 import GraphsTuple
    
    num_nodes = 100
    feature_dim = 16
    
    nodes = torch.randn(num_nodes, feature_dim, device=device)
    senders = torch.randint(0, num_nodes, (200,), device=device)
    receivers = torch.randint(0, num_nodes, (200,), device=device)
    
    return GraphsTuple(
        nodes=nodes,
        edges=torch.randn(200, 4, device=device),
        receivers=receivers,
        senders=senders,
        n_node=torch.tensor([num_nodes], device=device),
        n_edge=torch.tensor([200], device=device),
    )


@pytest.fixture
def sample_grid_2d(device):
    """Create a sample 2D grid for FNO tests."""
    return torch.randn(2, 1, 32, 32, device=device)
