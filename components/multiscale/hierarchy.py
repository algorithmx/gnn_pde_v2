"""
Hierarchical graph utilities for multiscale processing.

Based on "Multipole Graph Neural Operator" (Li et al., NeurIPS 2020).
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import torch
import torch.nn as nn

from ...core.graph import GraphsTuple
from .graph_pooling import GraphPool


@dataclass
class HierarchicalGraph:
    """Container for hierarchical graph representation.
    
    Stores multiple graph levels from fine to coarse.
    
    Attributes:
        graphs: List of graphs at different levels [level_0 (finest), ..., level_L (coarsest)]
        indices_list: List of indices from each pooling step
        transitions: List of transition matrices between levels
    """
    graphs: List[GraphsTuple]
    indices_list: List[torch.Tensor]
    transitions: Optional[List[torch.Tensor]] = None


def build_hierarchical_graphs(
    graph: GraphsTuple,
    levels: int,
    nodes_per_level: Optional[List[int]] = None,
    pooling_ratio: float = 0.5,
    feature_dim: int = 128,
) -> HierarchicalGraph:
    """Build multi-level graph hierarchy.
    
    Creates a hierarchy of graphs at different resolutions for MGKN-style
    processing. Each level has fewer nodes than the previous.
    
    Args:
        graph: Input graph (finest level)
        levels: Number of hierarchy levels
        nodes_per_level: Optional list of node counts per level
        pooling_ratio: Ratio of nodes to keep at each level (if nodes_per_level not given)
        feature_dim: Feature dimension for pooling
    
    Returns:
        HierarchicalGraph with levels from fine to coarse
    
    Example:
        >>> hierarchy = build_hierarchical_graphs(graph, levels=3, nodes_per_level=[400, 100, 25])
        >>> # hierarchy.graphs[0] has ~400 nodes
        >>> # hierarchy.graphs[1] has ~100 nodes
        >>> # hierarchy.graphs[2] has ~25 nodes
    
    Reference:
        Li et al., "Multipole Graph Neural Operator", NeurIPS 2020
    """
    graphs = [graph]
    indices_list = []
    
    current_graph = graph
    num_nodes = graph.nodes.shape[0] if graph.nodes is not None else 0
    device = graph.nodes.device if graph.nodes is not None else torch.device('cpu')
    
    for level in range(levels - 1):
        # Determine k for this level
        if nodes_per_level is not None and level < len(nodes_per_level):
            k = nodes_per_level[level + 1]
        else:
            k = int(num_nodes * pooling_ratio)
        
        # Create pool layer on the correct device
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        
        # Pool current graph
        pooled_graph, indices = pool(current_graph)
        
        graphs.append(pooled_graph)
        indices_list.append(indices)
        
        # Update for next iteration
        current_graph = pooled_graph
        num_nodes = pooled_graph.nodes.shape[0] if pooled_graph.nodes is not None else 0
    
    return HierarchicalGraph(
        graphs=graphs,
        indices_list=indices_list,
    )


def compute_transition_matrix(
    fine_graph: GraphsTuple,
    coarse_graph: GraphsTuple,
    method: str = "interpolation"
) -> torch.Tensor:
    """Compute transition matrix between graph levels.
    
    For MGKN, this represents the kernel K_{l, l+1} that maps from
    fine to coarse representation.
    
    Args:
        fine_graph: Fine level graph
        coarse_graph: Coarse level graph
        method: "interpolation", "attention", or "mlp"
    
    Returns:
        Transition matrix [n_fine, n_coarse]
    """
    if fine_graph.nodes is None or coarse_graph.nodes is None:
        raise ValueError("Both graphs must have nodes")
    
    n_fine = fine_graph.nodes.shape[0]
    n_coarse = coarse_graph.nodes.shape[0]
    
    if method == "interpolation":
        # Simple nearest neighbor interpolation
        # In practice, learnable transition is better
        transition = torch.zeros(n_fine, n_coarse)
        # For now, use uniform weights
        transition = torch.ones(n_fine, n_coarse) / n_coarse
        return transition
    
    elif method == "attention":
        # Attention-based transition
        # Compute similarity between fine and coarse nodes
        similarity = torch.matmul(
            fine_graph.nodes, 
            coarse_graph.nodes.t()
        )
        attention = torch.softmax(similarity, dim=-1)
        return attention
    
    else:
        # Default: uniform
        return torch.ones(n_fine, n_coarse) / n_coarse


def restrict_to_coarse(
    fine_features: torch.Tensor,
    transition: torch.Tensor
) -> torch.Tensor:
    """Restrict fine features to coarse level.
    
    Args:
        fine_features: [n_fine, feature_dim]
        transition: [n_fine, n_coarse]
    
    Returns:
        coarse_features: [n_coarse, feature_dim]
    """
    # Weighted average based on transition
    # transition[i, j] = weight from fine node i to coarse node j
    weights = transition / (transition.sum(dim=0, keepdim=True) + 1e-8)
    coarse_features = torch.matmul(weights.t(), fine_features)
    return coarse_features


def prolong_to_fine(
    coarse_features: torch.Tensor,
    transition: torch.Tensor
) -> torch.Tensor:
    """Prolong coarse features to fine level.
    
    Args:
        coarse_features: [n_coarse, feature_dim]
        transition: [n_fine, n_coarse]
    
    Returns:
        fine_features: [n_fine, feature_dim]
    """
    # Interpolate based on transition
    fine_features = torch.matmul(transition, coarse_features)
    return fine_features


__all__ = [
    "HierarchicalGraph",
    "build_hierarchical_graphs",
    "compute_transition_matrix",
    "restrict_to_coarse",
    "prolong_to_fine",
]
