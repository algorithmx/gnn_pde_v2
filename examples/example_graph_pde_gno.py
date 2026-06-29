"""
Example: Graph-PDE GNO (Edge-Conditioned Graph Neural Operator)

Demonstrates the reusable :class:`gnn_pde_v2.models.GraphNeuralOperator`, which
recreates the Graph Kernel Network from https://github.com/neuraloperator/graph-pde
using the framework's edge-conditioned convolution.

Reference:
    Li, Z. et al. (2020). "Neural Operator: Graph Kernel Network for Partial
    Differential Equations." https://arxiv.org/abs/2003.03485

Key Innovation:
    Edge-conditioned convolution learns edge weights from edge attributes,
    enabling PDE operators on irregular graphs.
"""

import torch
import torch.nn as nn

from gnn_pde_v2.core import AutoRegisterModel, MLP
from gnn_pde_v2.components import EdgeConditionedConvBlock
from gnn_pde_v2.models import GraphNeuralOperator

# Names expected by tests / downstream examples.
GraphPDEGNO = GraphNeuralOperator
GraphPDE_GNO = GraphNeuralOperator
GraphConvBlock = EdgeConditionedConvBlock

__all__ = ["GraphNeuralOperator", "GraphPDEGNO", "EdgeConvBlock", "example_usage"]


class EdgeConvBlock(nn.Module):
    """DGCNN-style EdgeConv block: edge features [h_i, h_j - h_i] -> MLP -> max-pool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = MLP(
            in_dim=in_channels * 2, out_dim=out_channels, hidden_dims=[out_channels],
            activation='relu', use_layer_norm=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_features = torch.cat([x[dst], x[src] - x[dst]], dim=-1)
        edge_messages = self.mlp(edge_features)
        out = torch.full((x.shape[0], edge_messages.shape[-1]), float('-inf'), device=x.device)
        out = out.index_reduce(0, dst, edge_messages, reduce='amax', include_self=False)
        return torch.nan_to_num(out, neginf=0.0)


def example_usage():
    """Build a Graph-PDE GNO and run one forward pass on a synthetic mesh."""
    print("=" * 60)
    print("Graph-PDE GNO Example using gnn_pde_v2 Framework")
    print("=" * 60)

    model = GraphNeuralOperator(node_input_size=2, edge_input_size=3, output_size=1,
                                hidden_size=128, num_layers=6)

    node_features = torch.randn(100, 2)
    edge_index = torch.randint(0, 100, (2, 400))
    edge_attr = torch.randn(400, 3)
    output = model(node_features, edge_index, edge_attr)

    print(f"Hidden {model.hidden_size} | Layers {model.num_layers} | Params {sum(p.numel() for p in model.parameters()):,}")
    print(f"nodes {tuple(node_features.shape)} edges {tuple(edge_index.shape)} -> {tuple(output.shape)}")

    edge_conv = EdgeConvBlock(2, 64)
    print(f"EdgeConv output: {tuple(edge_conv(node_features, edge_index).shape)}")
    print("Available models:", AutoRegisterModel.list_models())
    return model, output


if __name__ == "__main__":
    example_usage()
