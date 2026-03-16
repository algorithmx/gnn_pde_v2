"""
Functional utilities for graph operations.

These are thin wrappers that use torch_scatter if available,
otherwise fall back to pure PyTorch implementations.
"""

import functools
import warnings
from typing import Literal, Optional, Tuple
import torch
from torch import Tensor

# Track whether we've warned about torch_scatter fallback
_TORCH_SCATTER_WARNING_ISSUED: bool = False

# Sentinel used to initialise output buffers for reduction ops.
_REDUCE_INIT = {
    'sum':  (0.0,      'scatter_add_',             None),
    'mean': (0.0,      'scatter_add_',             None),   # post-step: divide by count
    'max':  (float('-inf'), 'scatter_reduce_',     'amax'),
    'min':  (float('inf'),  'scatter_reduce_',     'amin'),
}


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
    reduce: Literal['sum', 'mean', 'max', 'min'] = 'sum',
) -> Tensor:
    """
    General scatter aggregation.

    Uses ``torch_scatter`` if available (fastest), otherwise falls back to
    pure PyTorch ops.  All four reductions share a single code path, so
    there is no duplication and the ``torch_scatter`` fast path is tried
    exactly once per call.

    Args:
        src: Source features ``[E, *feat_dims]``
        index: Destination indices ``[E]``
        dim: Dimension to scatter along (default: 0)
        dim_size: Output size along ``dim``; inferred from ``index`` if omitted
        reduce: One of ``'sum'``, ``'mean'``, ``'max'``, ``'min'``

    Returns:
        Aggregated tensor ``[dim_size, *feat_dims]``
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # ── fast path ────────────────────────────────────────────────────────────
    try:
        from torch_scatter import scatter as _scatter
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    except ImportError:
        global _TORCH_SCATTER_WARNING_ISSUED
        if not _TORCH_SCATTER_WARNING_ISSUED:
            _TORCH_SCATTER_WARNING_ISSUED = True
            warnings.warn(
                "torch_scatter not available; falling back to pure PyTorch scatter. "
                "Install torch_scatter for better performance: "
                "pip install torch_scatter",
                UserWarning,
                stacklevel=2,
            )

    # ── pure-PyTorch fallback ─────────────────────────────────────────────────
    init_val, op, reduce_str = _REDUCE_INIT[reduce]
    shape = list(src.shape)
    shape[dim] = dim_size

    index_shape = [1] * src.dim()
    index_shape[dim] = -1
    index_expanded = index.view(index_shape).expand_as(src)

    if op == 'scatter_add_':
        out = torch.full(shape, init_val, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index_expanded, src)
        if reduce == 'mean':
            ones = torch.ones(index.shape[0], dtype=src.dtype, device=src.device)
            count_shape = [1] * out.dim()
            count_shape[dim] = dim_size
            count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            count.scatter_add_(0, index, ones)
            out = out / count.view(count_shape).clamp(min=1)
    else:  # scatter_reduce_ (max / min)
        out = torch.full(shape, init_val, dtype=src.dtype, device=src.device)
        out.scatter_reduce_(dim, index_expanded, src, reduce=reduce_str)

    return out


# ── Named aliases (zero overhead: functools.partial is resolved at import) ────

def scatter_sum(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """Sum aggregation. See :func:`scatter`."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def scatter_mean(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """Mean aggregation. See :func:`scatter`."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_max(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """Max aggregation. See :func:`scatter`."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')


def scatter_min(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
    """Min aggregation. See :func:`scatter`."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='min')


def scatter_softmax(src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tensor:
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
    method: Literal['sum', 'mean', 'max', 'min'] = 'sum',
) -> Tensor:
    """
    Aggregate edge features to receiver nodes.

    Semantic convenience wrapper around :func:`scatter` with ``dim=0``.

    Args:
        edge_features: [E, feat_dim] - Edge features
        receivers: [E] - Receiver node indices
        num_nodes: Number of nodes
        method: Aggregation method — ``'sum'``, ``'mean'``, ``'max'``, ``'min'``

    Returns:
        [num_nodes, feat_dim] - Aggregated node features
    """
    return scatter(edge_features, receivers, dim=0, dim_size=num_nodes, reduce=method)


def broadcast_global(globals_: Tensor, counts: Tensor) -> Tensor:
    """
    Broadcast per-graph global features to every node or edge.

    Args:
        globals_: [B, global_dim] - One global vector per graph in the batch
        counts: [B] - Number of nodes (or edges) per graph

    Returns:
        [total, global_dim] - Global features repeated for each node/edge
    """
    return torch.repeat_interleave(globals_, counts, dim=0)


def aggregate_to_global(
    features: Tensor,
    counts: Tensor,
    method: Literal['sum', 'mean', 'max', 'min'] = 'mean',
) -> Tensor:
    """
    Pool per-node or per-edge features back to graph-level globals.

    Args:
        features: [total, feat_dim] - Node or edge features (flat batch)
        counts: [B] - Number of nodes (or edges) per graph
        method: Aggregation method — ``'sum'``, ``'mean'``, ``'max'``, ``'min'``

    Returns:
        [B, feat_dim] - One aggregated vector per graph
    """
    batch_index = torch.repeat_interleave(
        torch.arange(len(counts), device=features.device), counts
    )
    return scatter(features, batch_index, dim=0, dim_size=len(counts), reduce=method)


# Backward compatibility alias (broadcast_global was renamed to broadcast_global)
# TODO: Deprecate broadcast_global in favor of broadcast_global in future versions
