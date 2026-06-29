"""
Graph Neural Operator (GNO): edge-conditioned graph kernel network.

Reusable encode-process-decode model implementing the Graph Kernel Network of
Li et al. (2020), built from the framework's edge-conditioned convolution. Lets
examples build a GNO in one call instead of redefining encoder/processor/decoder.

Reference:
    Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K.,
    Stuart, A., & Anandkumar, A. (2020). "Neural Operator: Graph Kernel Network
    for Partial Differential Equations." https://arxiv.org/abs/2003.03485
"""

from typing import Optional

import torch
import torch.nn as nn

from ..core import MLP, AutoRegisterModel
from ..core.graph import GraphsTuple
from ..components import EdgeConditionedConvBlock, ScalarEdgeMessageProcessor


class GraphNeuralOperator(
    AutoRegisterModel, name='gno', aliases=['graph_neural_operator', 'graph_pde_gno']
):
    """Edge-conditioned Graph Neural Operator for PDEs on irregular meshes.

    Architecture: node/edge MLP encoders -> ``num_layers`` edge-conditioned
    convolution blocks (NNConv-style, learnable edge weights) -> MLP decoder.

    Args:
        node_input_size: Input node feature dimension.
        edge_input_size: Input edge attribute dimension.
        output_size: Output dimension.
        hidden_size: Latent width for encoders, processor, decoder.
        num_layers: Number of edge-conditioned convolution blocks.
        aggregate: Neighbor aggregation method ('mean' or 'sum').
        activation: Activation name for all MLPs.

    Example:
        >>> model = GraphNeuralOperator(node_input_size=2, edge_input_size=3, output_size=1)
        >>> x = torch.randn(100, 2); ei = torch.randint(0, 100, (2, 400)); ea = torch.randn(400, 3)
        >>> y = model(x, ei, ea)  # [100, 1]
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 6,
        aggregate: str = 'mean',
        activation: str = 'relu',
    ):
        super().__init__()
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.node_encoder = MLP(
            in_dim=node_input_size, out_dim=hidden_size, hidden_dims=[hidden_size],
            activation=activation, use_layer_norm=False,
        )
        self.edge_encoder = MLP(
            in_dim=edge_input_size, out_dim=hidden_size, hidden_dims=[hidden_size],
            activation=activation, use_layer_norm=False,
        )

        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_proc = ScalarEdgeMessageProcessor(hidden_size)
            self.processor.append(EdgeConditionedConvBlock(
                latent_dim=hidden_size,
                edge_latent_dim=hidden_size,
                edge_weight_net=MLP(
                    in_dim=hidden_size, out_dim=edge_proc.weight_out_dim,
                    hidden_dims=[hidden_size], activation=activation, use_layer_norm=False,
                ),
                edge_processor=edge_proc,
                aggregate=aggregate,
                root_weight=True,
                bias=True,
            ))

        self.decoder = MLP(
            in_dim=hidden_size, out_dim=output_size, hidden_dims=[hidden_size],
            activation=activation, use_layer_norm=False,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, node_input_size] node inputs.
            edge_index: [2, E] connectivity (senders, receivers).
            edge_attr: [E, edge_input_size] edge attributes, or None.
        Returns:
            [N, output_size] predictions.
        """
        node_emb = self.node_encoder(node_features)
        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = torch.zeros(edge_index.shape[1], self.hidden_size, device=node_features.device)

        src, dst = edge_index
        graph = GraphsTuple.from_flat(
            nodes=node_emb,
            n_node=torch.tensor([node_emb.shape[0]], device=node_emb.device),
            edges=edge_emb,
            senders=src,
            receivers=dst,
            n_edge=torch.tensor([edge_emb.shape[0]], device=edge_emb.device),
        )
        for block in self.processor:
            graph = graph.replace(nodes=block(graph).nodes)
        return self.decoder(graph.nodes)

    def save_config(self):
        return {
            'model_type': 'gno',
            'node_input_size': self.node_input_size,
            'edge_input_size': self.edge_input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }
