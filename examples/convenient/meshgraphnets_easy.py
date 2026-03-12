"""
MeshGraphNets using the convenient high-level API.

This example shows how to use the optional convenient features:
- Auto-registration
- Pydantic configs
- ConfigBuilder
"""

import torch
import torch.nn as nn
from dataclasses import replace

# Import from convenient (optional high-level API)
from gnn_pde_v2.convenient import AutoRegisterModel, ModelConfig, ConfigBuilder
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import MLP, GraphNetBlock


class EasyMeshGraphNets(AutoRegisterModel, name='meshgraphnets_easy'):
    """
    MeshGraphNets with auto-registration.
    
    Same implementation, but can be instantiated via ConfigBuilder.
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
        
        # Encoder MLPs
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
        
        # Processor blocks
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
        
        # Decoder
        self.decoder = MLP(
            in_dim=hidden_size,
            out_dim=output_size,
            hidden_dims=[hidden_size],
            activation=activation,
        )
    
    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """Forward pass."""
        # Encode
        nodes = self.node_encoder(graph.nodes)
        edges = self.edge_encoder(graph.edges)
        
        # Process with residual
        for block in self.processor_blocks:
            processed = block(replace(graph, nodes=nodes, edges=edges))
            nodes = nodes + processed.nodes
            edges = edges + processed.edges
        
        # Decode
        return self.decoder(nodes)


def example_with_config():
    """Example using config-based instantiation."""
    print("=" * 60)
    print("Easy MeshGraphNets Example (Convenient API)")
    print("=" * 60)
    
    # Create config
    config = ModelConfig(
        model_type='meshgraphnets_easy',
        hidden_dim=128,
        n_layers=15,
    )
    
    print(f"\nConfig:")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.n_layers}")
    
    # Instantiate via builder
    builder = ConfigBuilder(config)
    
    # Note: Since MeshGraphNetsEasy is a custom model,
    # we'd need to register it properly or instantiate directly
    print("\n" + "=" * 60)
    print("Key features of this convenient approach:")
    print("  - Auto-registration via subclassing")
    print("  - Pydantic config validation")
    print("  - ConfigBuilder for instantiation")
    print("  - Good for quick experimentation")
    print("=" * 60)
    
    # Show that it's registered
    print(f"\nRegistered models: {AutoRegisterModel.list_models()}")
    
    # Instantiate directly (ConfigBuilder needs more setup for custom models)
    model = AutoRegisterModel.create(
        'meshgraphnets_easy',
        node_input_size=11,
        edge_input_size=3,
        output_size=3,
    )
    
    print(f"\nCreated model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def example_string_initialization():
    """Example using string-based initialization."""
    from gnn_pde_v2.convenient import get_initializer
    
    print("\n" + "=" * 60)
    print("String-based Initialization Example")
    print("=" * 60)
    
    # Get initializer by string
    init_fn = get_initializer('glorot_uniform')
    print(f"\n'glorot_uniform' maps to: {init_fn.__name__}")
    
    init_fn = get_initializer('he_normal')
    print(f"'he_normal' maps to: {init_fn.__name__}")
    
    # Apply to tensor
    tensor = torch.empty(10, 10)
    init_fn(tensor)
    print(f"\nInitialized tensor with mean={tensor.mean():.4f}, std={tensor.std():.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    model = example_with_config()
    example_string_initialization()
