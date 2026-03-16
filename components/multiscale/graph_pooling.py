"""
Graph pooling and unpooling operations.

Based on "Graph U-Nets" (Gao & Ji, ICML 2019).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.graph import GraphsTuple


class GraphPool(nn.Module):
    """Adaptive graph pooling layer (gPool).
    
    Samples a subset of nodes based on their scalar projection values
    on a trainable projection vector.
    
    Args:
        k: Number of nodes to select
        feature_dim: Dimension of node features
        use_gate: Whether to apply sigmoid gate
        connectivity_augmentation: If > 1, use graph power before pooling
    
    Reference:
        Gao & Ji, "Graph U-Nets", ICML 2019
    """
    
    def __init__(
        self,
        k: int,
        feature_dim: int,
        use_gate: bool = True,
        connectivity_augmentation: int = 1,
    ):
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        self.use_gate = use_gate
        self.connectivity_augmentation = connectivity_augmentation
        
        # Trainable projection vector
        self.proj_vector = nn.Parameter(torch.randn(feature_dim))
        nn.init.normal_(self.proj_vector, mean=0, std=0.01)
    
    def forward(
        self,
        graph: GraphsTuple
    ) -> Tuple[GraphsTuple, torch.Tensor]:
        """Pool graph to smaller size.
        
        Returns:
            pooled_graph: Graph with k nodes
            indices: Indices of selected nodes
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes")
        
        nodes = graph.nodes
        num_nodes = nodes.shape[0]
        
        # Handle case where k >= num_nodes
        if self.k >= num_nodes:
            indices = torch.arange(num_nodes, device=nodes.device)
            return graph, indices
        
        # Compute projection values
        proj_norm = torch.norm(self.proj_vector)
        if proj_norm > 0:
            y = torch.matmul(nodes, self.proj_vector) / proj_norm
        else:
            y = torch.matmul(nodes, self.proj_vector)
        
        # Select top-k nodes
        k_actual = min(self.k, num_nodes)
        values, indices = torch.topk(y, k_actual, largest=True, sorted=True)
        
        # Extract selected nodes
        selected_nodes = nodes[indices]
        
        # Apply gate
        if self.use_gate:
            gate = torch.sigmoid(values).unsqueeze(-1)
            selected_nodes = selected_nodes * gate
        
        # Build pooled edges
        pooled_edges = None
        pooled_senders = None
        pooled_receivers = None
        
        if graph.edges is not None and graph.senders is not None:
            if self.connectivity_augmentation > 1:
                adj = self._build_adjacency(num_nodes, graph.senders, graph.receivers)
                # Compute graph power
                adj_power = adj
                for _ in range(self.connectivity_augmentation - 1):
                    adj_power = torch.matmul(adj_power, adj)
                adj = (adj_power > 0).float()
                senders, receivers = torch.nonzero(adj, as_tuple=True)
            else:
                senders, receivers = graph.senders, graph.receivers
            
            # Create mask for selected nodes
            selected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=nodes.device)
            selected_mask[indices] = True
            
            # Keep edges where both endpoints are selected
            edge_mask = selected_mask[senders] & selected_mask[receivers]
            
            if edge_mask.any():
                pooled_senders = senders[edge_mask]
                pooled_receivers = receivers[edge_mask]
                pooled_edges = graph.edges[edge_mask] if graph.edges is not None else None
                
                # Remap indices
                index_map = torch.zeros(num_nodes, dtype=torch.long, device=nodes.device)
                index_map[indices] = torch.arange(k_actual, device=nodes.device)
                pooled_senders = index_map[pooled_senders]
                pooled_receivers = index_map[pooled_receivers]
        
        # Handle globals
        pooled_globals = graph.globals
        
        pooled_graph = GraphsTuple(
            nodes=selected_nodes,
            edges=pooled_edges,
            globals=pooled_globals,
            senders=pooled_senders,
            receivers=pooled_receivers,
            n_node=torch.tensor([k_actual], device=nodes.device),
            n_edge=torch.tensor([pooled_senders.shape[0] if pooled_senders is not None else 0], device=nodes.device),
        )
        
        return pooled_graph, indices
    
    def _build_adjacency(
        self,
        num_nodes: int,
        senders: torch.Tensor,
        receivers: torch.Tensor
    ) -> torch.Tensor:
        """Build dense adjacency matrix."""
        adj = torch.zeros(num_nodes, num_nodes, device=senders.device)
        adj[senders, receivers] = 1.0
        adj += torch.eye(num_nodes, device=senders.device)
        return adj


class GraphUnpool(nn.Module):
    """Graph unpooling layer (gUnpool).
    
    Restores pooled graph to original size using stored indices.
    Unselected positions are filled with zeros.
    
    Reference:
        Gao & Ji, "Graph U-Nets", ICML 2019
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pooled_graph: GraphsTuple,
        indices: torch.Tensor,
        original_num_nodes: int,
    ) -> GraphsTuple:
        """Unpool graph to original size.
        
        Args:
            pooled_graph: Pooled graph
            indices: Selected indices from pooling
            original_num_nodes: Original number of nodes
        
        Returns:
            unpooled graph
        """
        if pooled_graph.nodes is None:
            raise ValueError("Pooled graph must have nodes")
        
        feature_dim = pooled_graph.nodes.shape[-1]
        device = pooled_graph.nodes.device
        
        # Create empty feature matrix
        new_nodes = torch.zeros(
            original_num_nodes,
            feature_dim,
            device=device,
            dtype=pooled_graph.nodes.dtype
        )
        
        # Place pooled nodes at their original positions
        new_nodes[indices] = pooled_graph.nodes
        
        # Create unpooled graph (edges not restored)
        unpooled_graph = GraphsTuple(
            nodes=new_nodes,
            edges=None,
            globals=pooled_graph.globals,
            senders=None,
            receivers=None,
            n_node=torch.tensor([original_num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        return unpooled_graph


__all__ = [
    "GraphPool",
    "GraphUnpool",
]
