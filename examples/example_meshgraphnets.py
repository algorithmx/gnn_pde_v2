"""
Example: MeshGraphNets (DeepMind)

This example recreates the MeshGraphNets model from:
https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
PyTorch implementation: https://github.com/guochengqian/meshgraphnets_pytorch

Original Work Reference:
------------------------
Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2021).
"Learning Mesh-Based Simulation with Graph Networks."
International Conference on Machine Learning (ICML 2021).
Paper: https://arxiv.org/abs/2010.03409

Key Innovation:
---------------
MeshGraphNets applies Graph Networks to mesh-based physics simulations.
It predicts the next state of a physical system by learning on unstructured meshes.

This implementation uses the gnn_pde_v2 framework components while maintaining
exact equivalence to the original MeshGraphNets architecture.
"""

import torch
import torch.nn as nn
from typing import Optional

# Import framework components
from dataclasses import replace
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.convenient import AutoRegisterModel
from gnn_pde_v2.components import GraphNetBlock, MLP, MLPDecoder


class MeshGraphNets(AutoRegisterModel, name='meshgraphnets'):
    """
    MeshGraphNets implementation using gnn_pde_v2 framework components.
    
    Precise equivalent of MeshGraphNets from deepmind/meshgraphnets_pytorch.
    
    Default hyperparameters from the paper:
    - hidden_size: 128
    - message_passing_steps: 15
    - Encoder: 2-layer MLPs for nodes and edges
    - Processor: 15 GraphNetBlocks with residual connections
    - Decoder: 2-layer MLP
    - Activation: SiLU (Swish)
    
    Architecture:
        Graph(x, edge_attr, edge_index)
            ↓
        Encoder: MLP_v(x), MLP_e(edge_attr)
            ↓
        Processor × 15:
            For each step:
                e_ij = MLP_edge([v_i, v_j, e_ij])
                v_i = v_i + MLP_node([v_i, Σ_j e_ji])
            ↓
        Decoder: MLP_decode(v)
            ↓
        Output: acceleration [N, output_size]
    """
    
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        message_passing_steps: int = 15,
        activation: str = "silu",
    ):
        """
        Initialize MeshGraphNets model.
        
        Args:
            node_input_size: Dimension of node input features
            edge_input_size: Dimension of edge input features
            output_size: Dimension of output predictions
            hidden_size: Hidden dimension for all layers (default: 128)
            message_passing_steps: Number of message passing steps (default: 15)
            activation: Activation function - "silu" or "relu" (default: "silu")
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.message_passing_steps = message_passing_steps
        
        # Encoder: Separate node and edge MLPs using canonical components
        self.node_encoder = MLP(
            in_dim=node_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],  # 2-layer MLP: in->hidden->out
            activation=activation,
        )
        self.edge_encoder = MLP(
            in_dim=edge_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],  # 2-layer MLP: in->hidden->out
            activation=activation,
        )
        
        # Processor: Multiple message passing steps using framework GraphNetBlocks
        self.processor_blocks = nn.ModuleList([
            GraphNetBlock(
                node_dim=hidden_size,
                edge_dim=hidden_size,
                global_dim=None,
                hidden_dim=hidden_size,
                activation=activation,
            )
            for _ in range(message_passing_steps)
        ])
        
        # Decoder: MLP on node features using framework component
        self.decoder = MLPDecoder(
            node_dim=hidden_size,
            out_dim=output_size,
            hidden_dims=[hidden_size],  # 2-layer MLP
            activation=activation,
        )
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass through MeshGraphNets.
        
        Args:
            graph: GraphsTuple with:
                - nodes: [N, node_input_size] node features
                - edges: [E, edge_input_size] edge features
                - senders: [E] source node indices
                - receivers: [E] destination node indices
                
        Returns:
            [N, output_size] output predictions (typically acceleration)
        """
        # Encode: Project to latent dimension
        latent = replace(graph,
            nodes=self.node_encoder(graph.nodes),
            edges=self.edge_encoder(graph.edges),
        )
        
        # Process: Message passing with residual connections
        for block in self.processor_blocks:
            # Framework's GraphNetBlock handles the update internally
            # We apply residual connection here for MeshGraphNets-style
            processed = block(latent)
            
            # Residual connection (characteristic of MeshGraphNets)
            latent = replace(latent,
                nodes=latent.nodes + processed.nodes,
                edges=latent.edges + processed.edges,
            )
        
        # Decode: Project to output dimension
        output = self.decoder(latent)
        
        return output
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'meshgraphnets',
            'node_input_size': self.node_encoder.net[0].in_features,
            'edge_input_size': self.edge_encoder.net[0].in_features,
            'output_size': self.decoder.mlp.net[-1].out_features,
            'hidden_size': self.hidden_size,
            'message_passing_steps': self.message_passing_steps,
        }


# ============================================================================
# Wrapper for PyG Data compatibility
# ============================================================================

try:
    from torch_geometric.data import Data
    
    class MeshGraphNetsPyGWrapper(nn.Module):
        """
        Wrapper to use MeshGraphNets with PyG Data format.
        
        This adapter converts between PyG Data and GraphsTuple formats,
        allowing MeshGraphNets to be used with PyTorch Geometric datasets.
        """
        
        def __init__(self, **kwargs):
            super().__init__()
            self.model = MeshGraphNets(**kwargs)
        
        def forward(self, data: Data) -> torch.Tensor:
            """Forward with PyG Data input."""
            # Convert PyG Data to GraphsTuple
            batch_size = 1
            n_node = torch.tensor([data.x.shape[0]], dtype=torch.long, device=data.x.device)
            n_edge = torch.tensor([data.edge_attr.shape[0]], dtype=torch.long, device=data.x.device) if hasattr(data, 'edge_attr') else torch.tensor([data.edge_index.shape[1]], dtype=torch.long, device=data.x.device)
            
            graph = GraphsTuple(
                nodes=data.x,
                edges=data.edge_attr if hasattr(data, 'edge_attr') else None,
                receivers=data.edge_index[1],
                senders=data.edge_index[0],
                globals=None,
                n_node=n_node,
                n_edge=n_edge,
            )
            
            # Forward through model
            output = self.model(graph)
            
            return output
        
        def __getattr__(self, name):
            """Delegate attribute access to wrapped model."""
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)

except ImportError:
    MeshGraphNetsPyGWrapper = None


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the MeshGraphNets equivalent.
    
    This example uses hyperparameters similar to the flag (cloth) simulation
    from the MeshGraphNets paper.
    """
    print("=" * 60)
    print("MeshGraphNets Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Create model with paper defaults
    model = MeshGraphNets(
        node_input_size=11,      # e.g., position (3) + velocity (3) + node_type (5)
        edge_input_size=3,       # e.g., relative displacement (3)
        output_size=3,           # e.g., velocity acceleration (3) for 3D
        hidden_size=128,
        message_passing_steps=15,
        activation="silu",
    )
    
    # Example graph: Small mesh with 100 nodes, 400 edges
    num_nodes = 100
    num_edges = 400
    
    graph = GraphsTuple(
        nodes=torch.randn(num_nodes, 11),           # Node features
        edges=torch.randn(num_edges, 3),            # Edge features
        senders=torch.randint(0, num_nodes, (num_edges,)),
        receivers=torch.randint(0, num_nodes, (num_edges,)),
        globals=None,
        n_node=torch.tensor([num_nodes]),
        n_edge=torch.tensor([num_edges]),
    )
    
    # Forward pass
    output = model(graph)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Message passing steps: {model.message_passing_steps}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output Shapes:")
    print(f"  Input nodes: {graph.nodes.shape}")
    print(f"  Input edges: {graph.edges.shape}")
    print(f"  Output: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    return model, graph, output


if __name__ == "__main__":
    model, graph, output = example_usage()
