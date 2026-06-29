"""
Example: Graph U-Nets (Gao & Ji, ICML 2019)

Demonstrates the reusable :class:`gnn_pde_v2.models.GraphUNet`, which implements
Graph U-Nets (gPool/gUnpool + GCN encoder-decoder with skip connections) using
framework components.

Reference:
    Gao, H., & Ji, S. (2019). "Graph U-Nets." ICML 2019.
    https://arxiv.org/abs/1905.05178
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.components import GCNBlock
from gnn_pde_v2.models import GraphUNet

# Backward-compatible aliases.
GraphUNets = GraphUNet
GraphUNetsForGraphClassification = GraphUNet
ImprovedGCN = lambda in_dim, out_dim, activation=None: GCNBlock(
    in_dim=in_dim, out_dim=out_dim, self_loop_weight=2.0, activation=activation,
)

__all__ = ["GraphUNet", "GraphUNets", "ImprovedGCN", "create_synthetic_graph",
           "example_node_classification", "example_graph_classification"]


def create_synthetic_graph(num_nodes=500, num_edges=2000, feature_dim=1433, num_classes=7) -> Tuple[GraphsTuple, torch.Tensor]:
    """Create a random undirected graph and node labels for testing."""
    nodes = torch.randn(num_nodes, feature_dim)
    senders = torch.randint(0, num_nodes, (num_edges,))
    receivers = torch.randint(0, num_nodes, (num_edges,))
    mask = senders != receivers
    senders, receivers = senders[mask], receivers[mask]
    pairs = torch.unique(torch.stack([torch.cat([senders, receivers]),
                                      torch.cat([receivers, senders])], dim=1), dim=0)
    graph = GraphsTuple.from_flat(
        nodes=nodes, edges=None, senders=pairs[:, 0], receivers=pairs[:, 1],
        globals=None, n_node=torch.tensor([num_nodes]), n_edge=torch.tensor([pairs.shape[0]]),
    )
    return graph, torch.randint(0, num_classes, (num_nodes,))


def example_node_classification():
    """Cora-style node classification U-Net."""
    print("Graph U-Nets: Node Classification")
    model = GraphUNet(in_dim=1433, hidden_dim=128, out_dim=7,
                      nodes_per_level=[2000, 1000, 500, 200], skip_connection='add')
    graph, labels = create_synthetic_graph(2708, 10556, 1433, 7)
    out = model(graph)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    print(f"  out {tuple(out.shape)} | params {sum(p.numel() for p in model.parameters()):,} | loss {loss.item():.4f}")
    return model, graph, out


def example_graph_classification():
    """PROTEINS-style graph classification U-Net (global mean pool)."""
    print("Graph U-Nets: Graph Classification")
    model = GraphUNet(in_dim=50, hidden_dim=128, out_dim=2,
                      pool_ratios=[0.9, 0.7, 0.6, 0.5], global_pool='mean')
    graph, _ = create_synthetic_graph(284, 1500, 50, 2)
    out = model(graph)
    print(f"  out {tuple(out.shape)} | params {sum(p.numel() for p in model.parameters()):,}")
    return model, graph, out


if __name__ == "__main__":
    example_node_classification()
    example_graph_classification()
    print("All examples completed.")
