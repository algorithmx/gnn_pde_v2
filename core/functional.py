"""
Functional utilities for graph operations.

These are thin wrappers that use torch_scatter if available,
otherwise fall back to pure PyTorch implementations.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


def scatter_sum(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """
    Sum aggregation (scatter_add).
    
    Uses torch_scatter if available, otherwise pure PyTorch.
    
    Args:
        src: Source features [E, *feat_dims]
        index: Index tensor with destination indices [E]
        dim: Dimension to scatter along (default: 0)
        dim_size: Output size (number of destinations)
        
    Returns:
        Aggregated features [dim_size, *feat_dims]
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Try torch_scatter first (faster)
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
    except ImportError:
        pass
    
    # Fallback to pure PyTorch
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Expand index for broadcasting
    index_shape = [1] * src.dim()
    index_shape[dim] = -1
    index_expanded = index.view(index_shape).expand_as(src)
    
    out.scatter_add_(dim, index_expanded, src)
    return out


def scatter_mean(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """
    Mean aggregation.
    
    Args:
        src: Source features [E, *feat_dims]
        index: Index tensor [E]
        dim: Dimension to scatter along
        dim_size: Output size
        
    Returns:
        Mean-aggregated features
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Try torch_scatter first
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')
    except ImportError:
        pass
    
    # Fallback
    out = scatter_sum(src, index, dim, dim_size)
    
    # Count items per destination
    ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, 0, dim_size)
    
    # Reshape for broadcasting
    count_shape = [1] * out.dim()
    count_shape[dim] = dim_size
    count = count.view(count_shape)
    
    return out / count.clamp(min=1)


def scatter_max(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """
    Max aggregation.
    
    Args:
        src: Source features [E, *feat_dims]
        index: Index tensor [E]
        dim: Dimension to scatter along
        dim_size: Output size
        
    Returns:
        Max-aggregated features
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Try torch_scatter first
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    except ImportError:
        pass
    
    # Fallback
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)
    
    index_shape = [1] * src.dim()
    index_shape[dim] = -1
    index_expanded = index.view(index_shape).expand_as(src)
    
    out.scatter_reduce_(dim, index_expanded, src, reduce='amax')
    return out


def broadcast_nodes_to_edges(
    node_features: Tensor,
    senders: Tensor,
    receivers: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Broadcast node features to edges.
    
    Args:
        node_features: [N, feat_dim] - Node features
        senders: [E] - Sender node indices
        receivers: [E] - Receiver node indices
        
    Returns:
        (sender_features, receiver_features) both [E, feat_dim]
    """
    sender_features = node_features[senders]
    receiver_features = node_features[receivers]
    return sender_features, receiver_features


def aggregate_edges(
    edge_features: Tensor,
    receivers: Tensor,
    num_nodes: int,
    method: str = 'sum',
) -> Tensor:
    """
    Aggregate edge features to receiver nodes.
    
    Args:
        edge_features: [E, feat_dim] - Edge features
        receivers: [E] - Receiver node indices
        num_nodes: Number of nodes
        method: Aggregation method ('sum', 'mean', 'max')
        
    Returns:
        [num_nodes, feat_dim] - Aggregated node features
    """
    if method == 'sum':
        return scatter_sum(edge_features, receivers, dim=0, dim_size=num_nodes)
    elif method == 'mean':
        return scatter_mean(edge_features, receivers, dim=0, dim_size=num_nodes)
    elif method == 'max':
        return scatter_max(edge_features, receivers, dim=0, dim_size=num_nodes)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
