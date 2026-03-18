"""
Example: Graph-PDE GNO (Edge-Conditioned Graph Neural Operator)

This example recreates the Graph Kernel Network model from:
https://github.com/neuraloperator/graph-pde

Original Work Reference:
------------------------
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).
"Neural Operator: Graph Kernel Network for Partial Differential Equations."
Paper: https://arxiv.org/abs/2003.03485

Key Innovation:
---------------
Graph-PDE introduces edge-conditioned convolution for learning PDEs
on irregular graphs. Unlike standard GNNs that use fixed edge weights,
this approach learns edge weights from edge attributes.

This implementation uses the gnn_pde_v2 framework components where applicable
while maintaining exact equivalence to the original Graph-PDE GNO architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import framework components
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.core import MLP
from gnn_pde_v2.components import EdgeConditionedConvBlock, ScalarEdgeMessageProcessor


class GraphPDE_GNO(AutoRegisterModel, name='graph_pde_gno', namespace='example'):
    """
    Graph-PDE GNO implementation using gnn_pde_v2 framework components.
    
    Precise equivalent of Graph-PDE GNO model.
    Original implementation: graph_neural_operator/src/models/graph_net.py
    
    Architecture:
        Input nodes [N, node_input_size], edges [E, edge_input_size]
            ↓
        Encoder:
          - Node encoder: [node_input_size] → [hidden_size] (framework's MLP)
          - Edge encoder: [edge_input_size] → [hidden_size] (framework's MLP)
            ↓
        Processor: num_layers × GraphConvBlock
          - For each block:
            a) Generate edge weights from edge features
            b) Message: aggregate weighted neighbor features
            c) Update: MLP(node, message) + residual
            ↓
        Decoder: [hidden_size] → [output_size] (framework's MLP)
            ↓
        Output [N, output_size]
    
    Key feature: Edge-conditioned convolution with learnable weight generation.
    """
    
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()
        
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ==================== Encoders using framework components ====================
        
        # Node encoder using framework's MLP
        self.node_encoder = MLP(
            in_dim=node_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],
            activation='relu',
            use_layer_norm=False,
        )
        
        # Edge encoder using framework's MLP
        self.edge_encoder = MLP(
            in_dim=edge_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],
            activation='relu',
            use_layer_norm=False,
        )
        
        # ==================== Processor ====================
        
        # Stack of edge-conditioned convolution blocks (NNConv-style)
        # Uses the framework's EdgeConditionedConvBlock (MessagePassingBase subclass)
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            edge_proc = ScalarEdgeMessageProcessor(hidden_size)
            self.processor.append(EdgeConditionedConvBlock(
                latent_dim=hidden_size,
                edge_latent_dim=hidden_size,
                edge_weight_net=MLP(
                    in_dim=hidden_size,
                    out_dim=edge_proc.weight_out_dim,
                    hidden_dims=[hidden_size],
                    activation='relu',
                    use_layer_norm=False,
                ),
                edge_processor=edge_proc,
                aggregate='mean',
                root_weight=True,
                bias=True,
            ))
        
        # ==================== Decoder using framework component ====================
        
        self.decoder = MLP(
            in_dim=hidden_size,
            out_dim=output_size,
            hidden_dims=[hidden_size],
            activation='relu',
            use_layer_norm=False,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
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
        Forward pass.
        
        Args:
            node_features: [N, node_input_size] - Node input features
            edge_index: [2, E] - Edge connectivity (source, target)
            edge_attr: [E, edge_input_size] - Edge attributes (optional)
            
        Returns:
            [N, output_size] - Predictions
        """
        # ==================== Encoding ====================
        
        # Encode node features using framework's MLP
        node_emb = self.node_encoder(node_features)
        
        # Encode edge features using framework's MLP
        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = torch.zeros(
                edge_index.shape[1],
                self.hidden_size,
                device=node_features.device,
            )
        
        # ==================== Processing ====================
        
        # Wrap raw tensors in GraphsTuple for the framework blocks
        src, dst = edge_index
        graph = GraphsTuple.from_flat(
            nodes=node_emb,
            n_node=torch.tensor([node_emb.shape[0]], device=node_emb.device),
            edges=edge_emb,
            senders=src,
            receivers=dst,
            n_edge=torch.tensor([edge_emb.shape[0]], device=edge_emb.device),
        )
        
        # Apply edge-conditioned convolution blocks
        for block in self.processor:
            graph = graph.replace(nodes=block(graph).nodes)
        
        # ==================== Decoding ====================
        
        output = self.decoder(graph.nodes)
        
        return output
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'graph_pde_gno',
            'node_input_size': self.node_input_size,
            'edge_input_size': self.edge_input_size,
            'output_size': self.output_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }


# Backward-compatible alias expected by tests/examples.
GraphPDEGNO = GraphPDE_GNO


# GraphConvBlock has been absorbed into the framework as EdgeConditionedConvBlock.
# See gnn_pde_v2.components.processors.EdgeConditionedConvBlock.
GraphConvBlock = EdgeConditionedConvBlock


# ============================================================================
# Alternative: EdgeConv-style Implementation
# ============================================================================

class EdgeConvBlock(nn.Module):
    """
    EdgeConv-style block for graph neural networks.
    
    Similar to DGCNN's EdgeConv:
    - Message: h_j - h_i (edge features as difference)
    - Combine: [h_i, h_j - h_i] as edge feature
    - Aggregate: Max pooling over neighbors
    
    This captures local geometric structure better than simple aggregation.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # MLP for edge features using framework's MLP
        self.mlp = MLP(
            in_dim=in_channels * 2,
            out_dim=out_channels,
            hidden_dims=[out_channels],
            activation='relu',
            use_layer_norm=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        EdgeConv forward.
        
        Args:
            x: [N, in_channels] - Node features
            edge_index: [2, E] - Edge connectivity
            
        Returns:
            [N, out_channels] - Updated features
        """
        src, dst = edge_index
        
        # Source and target features
        x_i = x[dst]
        x_j = x[src]
        
        # Edge features: [x_i, x_j - x_i]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        
        # Apply MLP to edge features using framework component
        edge_messages = self.mlp(edge_features)
        
        # Aggregate with max pooling
        num_nodes = x.shape[0]
        out = torch.zeros(num_nodes, edge_messages.shape[-1], device=x.device)
        
        for i in range(num_nodes):
            mask = (dst == i)
            if mask.any():
                out[i] = edge_messages[mask].max(dim=0)[0]
        
        return out


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the Graph-PDE GNO equivalent.
    
    This configuration is suitable for PDEs on irregular meshes
    with edge-conditioned message passing.
    """
    print("=" * 60)
    print("Graph-PDE GNO Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Model configuration
    model = GraphPDE_GNO(
        node_input_size=2,
        edge_input_size=3,
        output_size=1,
        hidden_size=128,
        num_layers=6,
    )
    
    # Example: Irregular mesh
    num_nodes = 100
    num_edges = 400
    
    # Node features (2D position + scalar field)
    node_features = torch.randn(num_nodes, 2)
    
    # Random edge connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Edge attributes (relative position, distance)
    edge_attr = torch.randn(num_edges, 3)
    
    # Forward pass
    output = model(node_features, edge_index, edge_attr)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Number of layers: {model.num_layers}")
    print("  Edge processor: ScalarEdgeMessageProcessor")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input nodes: {node_features.shape}")
    print(f"  Edges: {edge_index.shape}")
    print(f"  Edge attributes: {edge_attr.shape}")
    print(f"  Output: {output.shape}")
    
    # Test EdgeConv block using framework's MLP
    print("\n" + "-" * 60)
    print("EdgeConv Block (using framework's MLP)")
    print("-" * 60)
    
    edge_conv = EdgeConvBlock(in_channels=2, out_channels=64)
    edge_conv_out = edge_conv(node_features, edge_index)
    print(f"EdgeConv output: {edge_conv_out.shape}")
    print(f"EdgeConv parameters: {sum(p.numel() for p in edge_conv.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    model, output = example_usage()
