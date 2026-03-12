"""
Full aggregation API (optional convenience).

For lean usage, use gnn_pde_v2.core.functional which has the essentials.
This module provides additional operations and the full API.
"""

from typing import Optional
import torch
from torch import Tensor

from ..core.functional import scatter_sum, scatter_mean, scatter_max


def scatter_min(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> Tensor:
    """
    Min aggregation.
    
    Args:
        src: Source features [E, *feat_dims]
        index: Index tensor [E]
        dim: Dimension to scatter along
        dim_size: Output size
        
    Returns:
        Min-aggregated features
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    try:
        from torch_scatter import scatter
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce='min')
    except ImportError:
        pass
    
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device)
    
    index_shape = [1] * src.dim()
    index_shape[dim] = -1
    index_expanded = index.view(index_shape).expand_as(src)
    
    out.scatter_reduce_(dim, index_expanded, src, reduce='amin')
    return out


def scatter_softmax(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> Tensor:
    """
    Softmax aggregation (for attention-based aggregation).
    
    Computes softmax within each group defined by index.
    
    Args:
        src: Source features (typically attention scores) [E, *]
        index: Index tensor [E]
        dim: Dimension to scatter along
        dim_size: Output size
        
    Returns:
        Softmax-normalized features
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Compute max per group for numerical stability
    max_per_group = scatter_max(src, index, dim, dim_size)
    
    # Expand max for broadcasting
    index_shape = [1] * src.dim()
    index_shape[dim] = -1
    index_expanded = index.view(index_shape).expand_as(src)
    max_expanded = max_per_group.gather(dim, index_expanded)
    
    # Subtract max and exponentiate
    exp_src = torch.exp(src - max_expanded)
    
    # Sum of exponentials per group
    sum_exp = scatter_sum(exp_src, index, dim, dim_size)
    sum_exp_expanded = sum_exp.gather(dim, index_expanded)
    
    return exp_src / sum_exp_expanded.clamp(min=1e-8)


# Segment operations (alternative API)
segment_sum = scatter_sum
segment_mean = scatter_mean
segment_max = scatter_max
segment_min = scatter_min


__all__ = [
    'scatter_sum',
    'scatter_mean',
    'scatter_max',
    'scatter_min',
    'scatter_softmax',
    'segment_sum',
    'segment_mean',
    'segment_max',
    'segment_min',
]
