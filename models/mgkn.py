"""
Multipole Graph Neural Operator (MGKN).

Reusable V-cycle graph operator built from the framework's pooling components,
implementing Li et al. (NeurIPS 2020). Lets examples build an MGKN in one call.

Reference:
    Li, Z. et al. (2020). "Multipole Graph Neural Operator for Parametric PDEs."
    NeurIPS 2020.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ..core import MLP, AutoRegisterModel
from ..core.graph import GraphsTuple
from ..components.multiscale.graph_pooling import GraphPool, GraphUnpool


class KernelNetwork(nn.Module):
    """Kernel network kappa_phi mapping edge features to per-edge weights."""

    def __init__(self, input_dim: int = 6, output_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = MLP(in_dim=input_dim, out_dim=output_dim,
                       hidden_dims=[hidden_dim, hidden_dim], activation='gelu', use_layer_norm=True)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_features)


class MessagePassingLayer(nn.Module):
    """Kernel-convolution message passing: local W plus degree-normalized kernel aggregation."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, use_kernel: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_kernel = use_kernel
        self.W = nn.Linear(latent_dim, latent_dim, bias=False)
        if use_kernel:
            self.kernel_net = KernelNetwork(input_dim=6, output_dim=latent_dim, hidden_dim=hidden_dim)
        self.activation = nn.GELU()

    def forward(self, nodes, edges, senders, receivers, positions=None):
        local = self.W(nodes)
        if not self.use_kernel or edges is None:
            return self.activation(local)
        kernel_weights = self.kernel_net(edges)
        messages = nodes[senders] * kernel_weights
        num_nodes = nodes.shape[0]
        aggregated = torch.zeros(num_nodes, self.latent_dim, device=nodes.device)
        degrees = torch.zeros(num_nodes, device=nodes.device)
        degrees.scatter_add_(0, receivers, torch.ones_like(receivers, dtype=torch.float))
        aggregated.scatter_add_(0, receivers.unsqueeze(-1).expand(-1, self.latent_dim), messages)
        aggregated = aggregated / degrees.clamp(min=1).unsqueeze(-1)
        return self.activation(local + aggregated)


class MGKNLevel(nn.Module):
    """Stack of residual kernel message-passing layers with output layer norm."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_message_passing: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.message_layers = nn.ModuleList([
            MessagePassingLayer(latent_dim=latent_dim, hidden_dim=hidden_dim)
            for _ in range(n_message_passing)
        ])
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = graph.nodes
        if nodes is None:
            return graph
        for layer in self.message_layers:
            nodes = nodes + layer(nodes, graph.edges, graph.senders, graph.receivers)
        return graph.replace(nodes=self.layer_norm(nodes))


class MGKN(AutoRegisterModel, name='mgkn', aliases=['mgkn_model', 'multipole_gno']):
    """Multipole Graph Neural Operator with hierarchical V-cycle.

    Encoder -> downward pool to coarse levels -> per-level kernel message passing
    -> upward unpool with skip connections -> decoder. Linear complexity and
    mesh-invariant.

    Args:
        node_in_dim: Input node feature dimension (e.g. [a(x), x, y]).
        out_dim: Output dimension.
        latent_dim: Latent feature dimension.
        n_levels: Number of hierarchy levels.
        nodes_per_level: Node counts per level (default [400, 100, 25]).
        n_message_passing: Message-passing layers per level.
        hidden_dim: MLP hidden dimension.

    Example:
        >>> model = MGKN(node_in_dim=3, out_dim=1)
        >>> y = model(graph)  # [N, 1]
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        out_dim: int = 1,
        latent_dim: int = 64,
        n_levels: int = 3,
        nodes_per_level: Optional[List[int]] = None,
        n_message_passing: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.n_levels = n_levels
        self.nodes_per_level = nodes_per_level or [400, 100, 25]

        self.encoder = MLP(in_dim=node_in_dim, out_dim=latent_dim, hidden_dims=[hidden_dim],
                           activation='gelu', use_layer_norm=False)
        self.decoder = MLP(in_dim=latent_dim, out_dim=out_dim, hidden_dims=[hidden_dim],
                           activation='gelu', use_layer_norm=False)
        self.level_processors = nn.ModuleList([
            MGKNLevel(latent_dim=latent_dim, hidden_dim=hidden_dim, n_message_passing=n_message_passing)
            for _ in range(n_levels)
        ])
        self.pool_layers = nn.ModuleList([
            GraphPool(k=k, feature_dim=latent_dim) for k in self.nodes_per_level[1:]
        ])
        self.unpool_layers = nn.ModuleList([GraphUnpool() for _ in range(len(self.nodes_per_level) - 1)])

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        graph = graph.replace(nodes=self.encoder(graph.nodes))

        level_graphs = [graph]
        indices_list = []
        for level in range(self.n_levels - 1):
            pooled_graph, indices = self.pool_layers[level](level_graphs[-1])
            indices_list.append(indices)
            level_graphs.append(pooled_graph)

        for level in range(self.n_levels):
            level_graphs[level] = self.level_processors[level](level_graphs[level])

        for level in range(self.n_levels - 2, -1, -1):
            fine_size = level_graphs[level].nodes.shape[0]
            unpooled = self.unpool_layers[level](level_graphs[level + 1], indices_list[level], fine_size)
            combined = level_graphs[level].nodes + unpooled.nodes
            level_graphs[level] = self.level_processors[level](
                level_graphs[level].replace(nodes=combined)
            )

        return self.decoder(level_graphs[0].nodes)

    def save_config(self):
        return {
            'model_type': 'mgkn',
            'node_in_dim': self.node_in_dim,
            'out_dim': self.out_dim,
            'latent_dim': self.latent_dim,
            'n_levels': self.n_levels,
            'nodes_per_level': self.nodes_per_level,
        }
