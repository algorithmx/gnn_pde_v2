"""
Aggregation strategies for graph message passing.

This module provides a Protocol-based abstraction for aggregation,
allowing flexible yet standardized implementations.

Design principle:
- Aggregation is constrained by the framework (single source of truth)
- Edge feature computation is left to the user (unconstrained)

Example:
    from core.aggregation import Sum, Max, Mean, Min
    
    # Use built-in aggregation
    block = GraphNetBlock(latent_dim=128, aggregate=Sum())
    
    # Use max aggregation (common in EdgeConv-style networks)
    block = GraphNetBlock(latent_dim=128, aggregate=Max())
    
    # Custom aggregation
    class AttentionAggregation:
        def __init__(self, latent_dim):
            self.attention = nn.Linear(latent_dim, 1)
        
        def __call__(self, messages, receivers, num_nodes):
            scores = torch.softmax(self.attention(messages), dim=0)
            weighted = messages * scores
            return aggregate_edges(weighted, receivers, num_nodes, 'sum')
"""

from typing import Protocol, runtime_checkable, Callable, Any
from typing_extensions import Literal

import torch
from torch import Tensor
from torch import nn

from .functional import aggregate_edges


__all__ = [
    'Aggregation',
    'Sum',
    'Mean',
    'Max',
    'Min',
    'get_aggregation',
]

# ---------------------------------------------------------------------------
# Aggregation Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Aggregation(Protocol):
    """
    Protocol for aggregation strategies in graph message passing.

    Any callable satisfying this interface can be used as the aggregation
    component in message passing blocks. The Protocol pattern ensures
    structural compatibility while allowing flexible implementations.

    Example:
        # Built-in usage
        block = GraphNetBlock(latent_dim=128, aggregate=Sum())
        
        # Custom implementation
        class CustomAgg:
            def __call__(self, messages, receivers, num_nodes):
                return aggregate_edges(messages, receivers, num_nodes, 'sum')
    """

    def __call__(
        self,
        messages: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Aggregate edge messages to receiver nodes.

        Args:
            messages: [E, H] - Edge messages from compute_messages()
            receivers: [E] - Receiver node indices for each message
            num_nodes: Total number of nodes

        Returns:
            [N, H] - Aggregated features per node
        """
        ...


# ---------------------------------------------------------------------------
# Built-in Aggregation Implementations
# ---------------------------------------------------------------------------

class Sum:
    """
    Aggregation via sum (default for most GNNs).

    Sums all incoming edge messages to each node.
    Preserves magnitude information and is the default in most GNN architectures.
    """

    def __call__(
        self,
        messages: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return aggregate_edges(messages, receivers, num_nodes, method='sum')


class Mean:
    """
    Aggregation via mean (normalizes by neighbor count).

    Computes the average of incoming edge messages.
    Useful when node degrees vary significantly.
    """

    def __call__(
        self,
        messages: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return aggregate_edges(messages, receivers, num_nodes, method='mean')


class Max:
    """
    Aggregation via max (captures strongest signal).

    Takes the maximum over incoming edge messages.
    Common in EdgeConv (DGCNN) to capture local geometric structure.
    """

    def __call__(
        self,
        messages: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return aggregate_edges(messages, receivers, num_nodes, method='max')


class Min:
    """
    Aggregation via min (captures weakest signal).

    Takes the minimum over incoming edge messages.
    Less commonly used but can capture different aspects of neighborhood.
    """

    def __call__(
        self,
        messages: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return aggregate_edges(messages, receivers, num_nodes, method='min')


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

# Mapping from string to built-in aggregation class
_AGGREGATION_MAP: dict[str, Aggregation] = {
    'sum': Sum(),
    'mean': Mean(),
    'max': Max(),
    'min': Min(),
}


def get_aggregation(
    aggregate: Aggregation | Literal['sum', 'mean', 'max', 'min'], # purged Callable
) -> Callable:
    """
    Normalize various aggregation inputs to a callable.

    Args:
        aggregate: One of:
            - Aggregation Protocol instance (Sum, Mean, Max, Min)
            - String ('sum', 'mean', 'max', 'min')

    Returns:
        A callable with signature (messages, receivers, num_nodes) -> Tensor

    Example:
        # All equivalent:
        fn = get_aggregation(Sum())
        fn = get_aggregation('sum')
        fn = get_aggregation(lambda m, r, n: aggregate_edges(m, r, n, 'sum'))
    """
    # Already a Protocol instance (runtime_checkable confirms callable)
    if isinstance(aggregate, Aggregation):
        return aggregate

    # String shortcut
    if isinstance(aggregate, str):
        if aggregate not in _AGGREGATION_MAP:
            raise ValueError(
                f"Unknown aggregation '{aggregate}'. "
                f"Choose from: {list(_AGGREGATION_MAP.keys())}"
            )
        return _AGGREGATION_MAP[aggregate]

    return aggregate

