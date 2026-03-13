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
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.core.functional import scatter_sum
from gnn_pde_v2.core import MLP


def make_meshgraphnets_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    layer_norm_out: bool = True,
) -> MLP:
    """Create MeshGraphNets-faithful MLP using framework MLP.

    Original `build_mlp` (meshgraphnets_pytorch/model/model.py):
    - 4 Linear layers (3 hidden of same size)
    - ReLU after first 3
    - optional terminal LayerNorm
    """
    return MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=[hidden_dim, hidden_dim, hidden_dim],
        activation='relu',
        norms=[None, None, None, 'layer' if layer_norm_out else None],
    )


class MeshGraphNetsGNBlock(nn.Module):
    """MeshGraphNets-faithful processor block.

    Edge update:
      e' = MLP_e([v_s, v_r, e])
    Node update:
      v' = MLP_v([v, sum_{in edges} e'])
    Residual update is applied outside this block for both nodes and edges.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.edge_mlp = make_meshgraphnets_mlp(
            in_dim=3 * hidden_size,
            hidden_dim=hidden_size,
            out_dim=hidden_size,
            layer_norm_out=True,
        )
        self.node_mlp = make_meshgraphnets_mlp(
            in_dim=2 * hidden_size,
            hidden_dim=hidden_size,
            out_dim=hidden_size,
            layer_norm_out=True,
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        if graph.nodes is None or graph.edges is None:
            raise ValueError("MeshGraphNetsGNBlock requires both nodes and edges")

        senders = graph.senders
        receivers = graph.receivers

        v_s = graph.nodes[senders]
        v_r = graph.nodes[receivers]

        edge_in = torch.cat([v_s, v_r, graph.edges], dim=-1)
        new_edges = self.edge_mlp(edge_in)

        agg_in = scatter_sum(new_edges, receivers, dim=0, dim_size=graph.nodes.shape[0])
        node_in = torch.cat([graph.nodes, agg_in], dim=-1)
        new_nodes = self.node_mlp(node_in)

        return graph.replace(nodes=new_nodes, edges=new_edges)


class MeshGraphNets(AutoRegisterModel, name='meshgraphnets', namespace='example'):
    """MeshGraphNets implementation aligned to `meshgraphnets_pytorch`."""
    
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        hidden_size: int = 128,
        n_layers: int = 15,
        activation: str = "silu",
    ):
        """
        Initialize MeshGraphNets model.
        
        Args:
            node_input_size: Dimension of node input features
            edge_input_size: Dimension of edge input features
            output_size: Dimension of output predictions
            hidden_size: Hidden dimension for all layers (default: 128)
            n_layers: Number of message passing layers (default: 15)
            activation: Activation function - "silu" or "relu" (default: "silu")
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Encoder: faithful 4-linear MLP with terminal LayerNorm
        self.node_encoder = make_meshgraphnets_mlp(
            in_dim=node_input_size,
            hidden_dim=hidden_size,
            out_dim=hidden_size,
            layer_norm_out=True,
        )
        self.edge_encoder = make_meshgraphnets_mlp(
            in_dim=edge_input_size,
            hidden_dim=hidden_size,
            out_dim=hidden_size,
            layer_norm_out=True,
        )

        # Processor: faithful GN blocks
        self.processor_blocks = nn.ModuleList([
            MeshGraphNetsGNBlock(hidden_size=hidden_size)
            for _ in range(n_layers)
        ])

        # Decoder: faithful 4-linear MLP WITHOUT terminal LayerNorm
        self.decoder = make_meshgraphnets_mlp(
            in_dim=hidden_size,
            hidden_dim=hidden_size,
            out_dim=output_size,
            layer_norm_out=False,
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
        output = self.decoder(latent.nodes)

        return output
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'meshgraphnets',
            'node_input_size': self.node_encoder.in_features,
            'edge_input_size': self.edge_encoder.in_features,
            'output_size': self.decoder.out_features,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
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
        n_layers=15,
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
    print(f"  Layers: {model.n_layers}")
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
