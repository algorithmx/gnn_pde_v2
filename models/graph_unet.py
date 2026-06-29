"""
Graph U-Net: encoder-decoder GNN with graph pooling/unpooling and skip connections.

Reusable model implementing Graph U-Nets (Gao & Ji, ICML 2019) from the
framework's :class:`GraphPool`, :class:`GraphUnpool` and :class:`GCNBlock`, so
examples can build a node- or graph-classification U-Net in one call.

Reference:
    Gao, H., & Ji, S. (2019). "Graph U-Nets." ICML 2019.
    https://arxiv.org/abs/1905.05178
"""

from typing import List, Optional

import torch
import torch.nn as nn

from ..core import AutoRegisterModel
from ..core.graph import GraphsTuple
from ..components import GCNBlock
from ..components.multiscale.graph_pooling import GraphPool, GraphUnpool


def _improved_gcn(in_dim: int, out_dim: int, activation: Optional[str] = None) -> GCNBlock:
    """GCN layer with A + 2I self-loop weighting (improved GCN from the paper)."""
    return GCNBlock(in_dim=in_dim, out_dim=out_dim, self_loop_weight=2.0, activation=activation)


class GraphUNet(AutoRegisterModel, name='graph_unet', aliases=['graph_unets']):
    """Graph U-Net for node- or graph-level prediction.

    Encoder (GCN + gPool) -> bottleneck GCN -> decoder (gUnpool + GCN) with skip
    connections. Set ``global_pool`` for graph classification, leave ``None`` for
    node classification.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden width.
        out_dim: Output dimension (classes).
        nodes_per_level: Fixed node counts per pooling level (overrides ratios).
        pool_ratios: Pooling ratios per level, used when nodes_per_level is None.
        n_encoder_layers / n_decoder_layers: GCN layers per block.
        skip_connection: 'add' or 'concat'.
        connectivity_augmentation: Graph power for pooling connectivity.
        activation: GCN activation name (None = identity, as in paper).
        global_pool: None (node) or 'mean'/'sum'/'max' (graph-level output).

    Example:
        >>> model = GraphUNet(in_dim=1433, hidden_dim=128, out_dim=7,
        ...                   nodes_per_level=[2000, 1000, 500, 200])
        >>> y = model(graph)  # [N, 7]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        nodes_per_level: Optional[List[int]] = None,
        pool_ratios: Optional[List[float]] = None,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        skip_connection: str = "add",
        connectivity_augmentation: int = 2,
        activation: Optional[str] = None,
        global_pool: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.skip_connection = skip_connection
        self.connectivity_augmentation = connectivity_augmentation
        self.global_pool = global_pool

        if nodes_per_level is not None:
            self.nodes_per_level = nodes_per_level
            self.pool_ratios = None
            self.use_ratios = False
        elif pool_ratios is not None:
            self.pool_ratios = pool_ratios
            self.nodes_per_level = None
            self.use_ratios = True
        else:
            raise ValueError("Either nodes_per_level or pool_ratios must be provided")

        self.n_levels = len(self.nodes_per_level) if nodes_per_level else len(pool_ratios)

        self.embedding = _improved_gcn(in_dim, hidden_dim, activation=activation)

        self.encoder_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(self.n_levels):
            self.encoder_gcns.append(nn.ModuleList([
                _improved_gcn(hidden_dim, hidden_dim, activation=activation)
                for _ in range(n_encoder_layers)
            ]))
            if i < self.n_levels - 1:
                k = 1 if self.use_ratios else nodes_per_level[i]
                self.pools.append(GraphPool(
                    k=k, feature_dim=hidden_dim, use_gate=True,
                    connectivity_augmentation=connectivity_augmentation,
                ))

        self.bottleneck_gcn = _improved_gcn(hidden_dim, hidden_dim, activation=activation)

        self.decoder_gcns = nn.ModuleList([
            nn.ModuleList([
                _improved_gcn(hidden_dim, hidden_dim, activation=activation)
                for _ in range(n_decoder_layers)
            ])
            for _ in range(self.n_levels)
        ])
        self.unpool = GraphUnpool()

        if skip_connection == "concat":
            self.skip_projections = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(self.n_levels)
            ])

        if global_pool is not None:
            self.classifier = nn.Linear(hidden_dim, out_dim)
        else:
            self.output_gcn = _improved_gcn(hidden_dim, out_dim, activation=None)

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """Run the U-Net; returns [N, out_dim] (node) or [out_dim] (graph) tensor."""
        x = self.embedding(graph)

        encoder_outputs: List[GraphsTuple] = []
        indices_list: List[torch.Tensor] = []
        original_sizes: List[int] = []

        for i in range(self.n_levels):
            original_sizes.append(x.nodes.shape[0] if x.nodes is not None else 0)
            for gcn in self.encoder_gcns[i]:
                x = gcn(x)
            encoder_outputs.append(x)
            if i < self.n_levels - 1:
                if self.use_ratios:
                    num_nodes = x.nodes.shape[0] if x.nodes is not None else 0
                    k = max(1, int(num_nodes * self.pool_ratios[i]))
                    pool = GraphPool(k=k, feature_dim=self.hidden_dim, use_gate=True,
                                     connectivity_augmentation=self.connectivity_augmentation)
                    x, indices = pool(x)
                else:
                    x, indices = self.pools[i](x)
                indices_list.append(indices)

        x = self.bottleneck_gcn(x)

        for i in range(self.n_levels - 1, -1, -1):
            if i < len(indices_list):
                x = self.unpool(x, indices_list[i], original_sizes[i])
            skip = encoder_outputs[i]
            if x.nodes is not None and skip.nodes is not None:
                if self.skip_connection == "add":
                    x = x.replace(nodes=x.nodes + skip.nodes)
                elif self.skip_connection == "concat":
                    combined = torch.cat([x.nodes, skip.nodes], dim=-1)
                    x = x.replace(nodes=self.skip_projections[i](combined))
            for gcn in self.decoder_gcns[i]:
                x = gcn(x)

        if self.global_pool is not None:
            if x.nodes is None:
                raise ValueError("Graph must have nodes")
            if self.global_pool == "mean":
                graph_feat = x.nodes.mean(dim=0)
            elif self.global_pool == "sum":
                graph_feat = x.nodes.sum(dim=0)
            elif self.global_pool == "max":
                graph_feat = x.nodes.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown global_pool: {self.global_pool}")
            return self.classifier(graph_feat)
        return self.output_gcn(x).nodes

    def save_config(self):
        return {
            'model_type': 'graph_unet',
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'n_levels': self.n_levels,
            'skip_connection': self.skip_connection,
            'global_pool': self.global_pool,
        }
