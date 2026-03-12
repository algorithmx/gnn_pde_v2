"""
Minimal MeshGraphNets using core + components (lean approach).

This example shows the lean, explicit way to use the framework.
No registry, no config, no magic - just PyTorch with helpful components.
"""

import torch
import torch.nn as nn
from dataclasses import replace

# Import only what we need from core and components
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import MLP, Residual, GraphNetBlock


class MinimalMeshGraphNets(nn.Module):
    """
    MeshGraphNets with minimal dependencies.
    
    No auto-registration, no config system - just explicit PyTorch.
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
        super().__init__()
        
        self.hidden_size = hidden_size
        self.message_passing_steps = message_passing_steps
        
        # Explicit encoder MLPs
        self.node_encoder = MLP(
            in_dim=node_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],
            activation=activation,
        )
        
        self.edge_encoder = MLP(
            in_dim=edge_input_size,
            out_dim=hidden_size,
            hidden_dims=[hidden_size],
            activation=activation,
        )
        
        # Processor with explicit residual
        self.processor_blocks = nn.ModuleList([
            Residual(GraphNetBlock(
                node_dim=hidden_size,
                edge_dim=hidden_size,
                global_dim=None,
                hidden_dim=hidden_size,
                activation=activation,
            ))
            for _ in range(message_passing_steps)
        ])
        
        # Decoder
        self.decoder = MLP(
            in_dim=hidden_size,
            out_dim=output_size,
            hidden_dims=[hidden_size],
            activation=activation,
        )
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            graph: GraphsTuple with nodes, edges, senders, receivers
            
        Returns:
            [N, output_size] predictions
        """
        # Encode
        nodes = self.node_encoder(graph.nodes)
        edges = self.edge_encoder(graph.edges)
        
        # Process with explicit residual
        latent = replace(graph, nodes=nodes, edges=edges)
        
        for block in self.processor_blocks:
            # Residual wrapper handles the skip connection
            new_nodes = block(latent.nodes)
            # For edges, we do explicit residual
            processed = block.module  # Get inner GraphNetBlock
            temp_graph = replace(latent, nodes=new_nodes)
            new_graph = processed(temp_graph)
            latent = replace(
                latent,
                nodes=latent.nodes + new_graph.nodes,
                edges=latent.edges + new_graph.edges,
            )
        
        # Decode
        return self.decoder(latent.nodes)


def example_usage():
    """Demonstrate creating and using the minimal model."""
    print("=" * 60)
    print("Minimal MeshGraphNets Example (Lean Core)")
    print("=" * 60)
    
    # Create model
    model = MinimalMeshGraphNets(
        node_input_size=11,
        edge_input_size=3,
        output_size=3,
        hidden_size=128,
        message_passing_steps=15,
        activation="silu",
    )
    
    # Create sample graph
    num_nodes = 100
    num_edges = 400
    
    graph = GraphsTuple(
        nodes=torch.randn(num_nodes, 11),
        edges=torch.randn(num_edges, 3),
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
    print("Key features of this lean approach:")
    print("  - No registry, no magic")
    print("  - Explicit component composition")
    print("  - Standard PyTorch nn.Module")
    print("  - Import only what you need")
    print("=" * 60)
    
    return model, graph, output


if __name__ == "__main__":
    model, graph, output = example_usage()
