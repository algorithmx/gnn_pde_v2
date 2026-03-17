"""
GCN (Graph Convolutional Network) block.

Standard GCN implementation based on:
"Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)"

This module implements the spectral GCN formula:
    X' = D^(-1/2) A D^(-1/2) X W

Where:
    - A is the adjacency matrix (with self-loops)
    - D is the degree matrix
    - X is the node feature matrix
    - W is the learnable weight matrix

Also supports "Improved GCN" from the Graph U-Nets paper:
    Â = A + k*I  (self-loops with weight k, typically k=2)
"""

import torch
import torch.nn as nn
from typing import Optional

from ..core.graph import GraphsTuple


class GCNBlock(nn.Module):
    """
    Graph Convolutional Network layer.
    
    Implements both standard GCN and Improved GCN variants.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        self_loop_weight: Weight for self-loops. 
            - 1.0: Standard GCN (A + I)
            - 2.0: Improved GCN (A + 2I) from Graph U-Nets paper
            - 0.0: No self-loops
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh', or None)
        bias: Whether to add bias term
    
    Example:
        >>> gcn = GCNBlock(in_dim=128, out_dim=256, self_loop_weight=2.0)
        >>> out_graph = gcn(in_graph)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        self_loop_weight: float = 1.0,
        activation: Optional[str] = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.self_loop_weight = self_loop_weight
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Optional bias
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        # Activation function
        self.act = None
        if activation is not None:
            activation = activation.lower()
            if activation == 'relu':
                self.act = nn.ReLU()
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'silu' or activation == 'swish':
                self.act = nn.SiLU()
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()
            elif activation != 'none' and activation is not None:
                raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Apply GCN layer to graph.
        
        Args:
            graph: Input GraphsTuple with nodes, edges, senders, receivers
            
        Returns:
            GraphsTuple with updated node features
        """
        if graph.nodes is None:
            raise ValueError("Graph must have node features")
        
        nodes = graph.nodes
        num_nodes = nodes.shape[0]
        device = nodes.device
        
        # Build adjacency matrix
        # Start with zeros
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=nodes.dtype)
        
        # Add edges from senders/receivers
        if graph.senders is not None and graph.receivers is not None:
            adj[graph.senders, graph.receivers] = 1.0
        
        # Add self-loops with configurable weight
        if self.self_loop_weight != 0.0:
            adj += self.self_loop_weight * torch.eye(num_nodes, device=device, dtype=nodes.dtype)
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        
        # Compute D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-10, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Message passing: X' = A_norm @ X @ W + b
        new_nodes = adj_norm @ nodes @ self.weight
        
        if self.bias is not None:
            new_nodes = new_nodes + self.bias
        
        # Apply activation
        if self.act is not None:
            new_nodes = self.act(new_nodes)
        
        return graph.replace(nodes=new_nodes)
    
    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"self_loop_weight={self.self_loop_weight}, "
            f"activation={self.act.__class__.__name__ if self.act else None}"
        )


class GCNBlockWithEdgeFeatures(GCNBlock):
    """
    GCN block that incorporates edge features into the message passing.
    
    This is a hybrid between standard GCN and edge-conditioned convolution.
    Edge features are projected to scalar weights and used to weight the 
    adjacency matrix, then symmetric normalization is applied.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        self_loop_weight: float = 1.0,
        activation: Optional[str] = "relu",
        bias: bool = True,
    ):
        super().__init__(in_dim, out_dim, self_loop_weight, activation, bias)
        self.edge_dim = edge_dim
        
        # Edge feature to scalar weight projector
        self.edge_weight_net = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.ReLU(),  # Ensure non-negative weights
        )
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Apply GCN with edge feature weighting."""
        if graph.nodes is None:
            raise ValueError("Graph must have node features")
        
        nodes = graph.nodes
        num_nodes = nodes.shape[0]
        device = nodes.device
        
        # Build adjacency with edge weights
        adj = torch.zeros(num_nodes, num_nodes, device=device, dtype=nodes.dtype)
        
        if graph.senders is not None and graph.receivers is not None:
            # Use edge features as scalar weights if available
            if graph.edges is not None:
                # Project edge features to scalar weights
                edge_weights = self.edge_weight_net(graph.edges)  # [E, 1]
                edge_weights = edge_weights.squeeze(-1)  # [E]
                adj[graph.senders, graph.receivers] = edge_weights
            else:
                adj[graph.senders, graph.receivers] = 1.0
        
        # Add self-loops
        if self.self_loop_weight != 0.0:
            adj += self.self_loop_weight * torch.eye(num_nodes, device=device, dtype=nodes.dtype)
        
        # Symmetric normalization
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-10, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Message passing
        new_nodes = adj_norm @ nodes @ self.weight
        
        if self.bias is not None:
            new_nodes = new_nodes + self.bias
        
        if self.act is not None:
            new_nodes = self.act(new_nodes)
        
        return graph.replace(nodes=new_nodes)


__all__ = [
    "GCNBlock",
    "GCNBlockWithEdgeFeatures",
]