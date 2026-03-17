"""
Example: Wind-Farm GNO (Graph Neural Operator for Wind Farm Flow Prediction)

This example demonstrates the Wind-Farm Graph Neural Operator from:
https://github.com/jenspeterschoeler/Wind-Farm-GNO

Original Work Reference:
------------------------
Schøler, J. P., Peder Weilmann Rasmussen, F., Quick, J., & Réthoré, P.-E. (2025).
"Graph Neural Operator for windfarm wake flow."
Wind Energy Science Discussions (preprint).
Paper: https://doi.org/10.5194/wes-2025-261

The implementation is now provided by the gnn_pde_v2 framework components:
- gnn_pde_v2.components.WindFarmGNO: Paper-faithful two-stage model
- gnn_pde_v2.components.GENBlock: GEneralized aggregation Network blocks
- gnn_pde_v2.components.LearnableRBFEncoder: Learnable RBF distance encoding
- gnn_pde_v2.components.ProbeDecoder: General-purpose probe decoder

This example file demonstrates how to use these components for wind farm
flow prediction.
"""

import torch
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.components import (
    WindFarmGNO,
    LearnableRBFEncoder,
    GENBlock,
    ProbeDecoder,
)


def example_windfarm_gno():
    """
    Demonstrate paper-faithful Wind-Farm GNO.
    
    This uses the exact architecture from Schøler et al. (2025):
    - Two-stage: Turbine-to-Turbine (T2T) + Probe-to-Turbine (P2T)
    - GEN blocks with softmax aggregation
    - Learnable RBF encoding for distances
    - Paper hyperparameters: latent=128, layers=6, k=5
    """
    print("=" * 70)
    print("Wind-Farm-GNO Example using gnn_pde_v2 Framework")
    print("=" * 70)
    
    # Create model with paper defaults
    model = WindFarmGNO(
        num_turbine_features=10,      # e.g., x, y, D, CT, power, U, TI, ...
        num_edge_features=4,          # e.g., dx, dy, distance, angle
        num_probe_features=6,         # e.g., x, y, U, TI, ...
        turbine_output_dim=1,         # Effective wind speed at turbine
        probe_output_dim=1,           # Wind speed at probe location
        latent_dim=128,               # Paper default
        hidden_dim=128,               # Paper default
        num_mlp_layers=6,             # Paper default
        wt_message_passing_steps=6,   # Paper default (T2T stage)
        probe_message_passing_steps=6,# Paper default (P2T stage)
        k_neighbors=5,                # Paper default
        use_rbf=True,                 # Paper uses RBF encoding
        rbf_kwargs={
            'num_kernels': 20,
            'd_min': -1.0,
            'd_max': 1.0,
            'learnable': True,
        },
    )
    
    print(f"\nModel Configuration:")
    print(f"  Latent dimension: {model.latent_dim}")
    print(f"  k_neighbors: {model.k_neighbors}")
    print(f"  Use RBF encoding: {model.use_rbf}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create example wind farm
    num_turbines = 5
    num_probes = 30
    
    # Turbine graph (fully connected for wake interactions)
    turbine_features = torch.randn(num_turbines, 10)
    turbine_positions = torch.randn(num_turbines, 2)
    
    # Fully connected edges (excluding self-loops)
    edge_list = []
    for i in range(num_turbines):
        for j in range(num_turbines):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    num_edges = edge_index.shape[1]
    
    # Edge attributes: relative positions
    edge_attr = torch.randn(num_edges, 4)
    
    turbine_graph = GraphsTuple(
        nodes=turbine_features,
        edges=edge_attr,
        receivers=edge_index[1],
        senders=edge_index[0],
        globals=None,
        n_node=torch.tensor([num_turbines]),
        n_edge=torch.tensor([num_edges]),
        positions=turbine_positions,
    )
    
    # Probe inputs
    probe_positions = torch.randn(num_probes, 2)
    probe_features = torch.randn(num_probes, 6)
    
    # Forward pass
    output = model(
        turbine_graph=turbine_graph,
        probe_positions=probe_positions,
        probe_features=probe_features,
    )
    
    print(f"\nInput/Output Shapes:")
    print(f"  Turbine features: {turbine_features.shape}")
    print(f"  Turbine positions: {turbine_positions.shape}")
    print(f"  Probe positions: {probe_positions.shape}")
    print(f"  Probe features: {probe_features.shape}")
    print(f"  Turbine predictions: {output['turbine'].shape}")
    print(f"  Probe predictions: {output['probe'].shape}")
    
    print("\n" + "=" * 70)
    print("Paper-faithful Wind-Farm-GNO demonstration complete!")
    print("=" * 70)
    
    return model, output


def example_probe_decoder_with_gen():
    """
    Demonstrate using ProbeDecoder with GEN blocks.
    
    This shows how to compose the framework components for custom
    probe-based decoding tasks.
    """
    print("\n" + "=" * 70)
    print("ProbeDecoder with GEN Blocks Example")
    print("=" * 70)
    
    # Create a source graph
    num_nodes = 20
    source_graph = GraphsTuple(
        nodes=torch.randn(num_nodes, 128),  # Already in latent space
        edges=torch.randn(100, 1),          # Distances
        receivers=torch.randint(0, num_nodes, (100,)),
        senders=torch.randint(0, num_nodes, (100,)),
        positions=torch.randn(num_nodes, 2),
        n_node=torch.tensor([num_nodes]),
        n_edge=torch.tensor([100]),
    )
    
    # Create probe decoder with GEN blocks
    processor = torch.nn.ModuleList([
        GENBlock(latent_dim=128, hidden_dim=128, num_mlp_layers=2)
        for _ in range(3)
    ])
    
    decoder = ProbeDecoder(
        latent_dim=128,
        processor=processor,
        edge_encoder=LearnableRBFEncoder(num_kernels=20),
        out_dim=3,        # Predict 3D velocity
        hidden_dim=128,
        k_nearest=5,
    )
    
    # Query points
    query_positions = torch.randn(50, 2)  # 50 query points in 2D
    
    # Decode
    predictions = decoder(
        graph=source_graph,
        query_positions=query_positions,
    )
    
    print(f"\nProbeDecoder with GEN:")
    print(f"  Source nodes: {num_nodes}")
    print(f"  Query points: {query_positions.shape[0]}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    print("\n" + "=" * 70)
    
    return decoder, predictions


def example_learnable_rbf():
    """
    Demonstrate LearnableRBFEncoder.
    
    Shows how the RBF encoder transforms distances into high-dimensional
    features with learnable centers and widths.
    """
    print("\n" + "=" * 70)
    print("LearnableRBFEncoder Example")
    print("=" * 70)
    
    # Create encoder
    rbf = LearnableRBFEncoder(
        num_kernels=20,
        d_min=0.0,
        d_max=5.0,
        learnable=True,
    )
    
    # Encode some distances
    distances = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0])
    rbf_features = rbf(distances)
    
    print(f"\nRBF Encoding:")
    print(f"  Input distances: {distances}")
    print(f"  Output shape: {rbf_features.shape}")
    print(f"  Learnable parameters: {sum(p.numel() for p in rbf.parameters())}")
    print(f"  mu (centers): {rbf.mu.data[:5]} ...")  # First 5 centers
    print(f"  beta (widths): {rbf.beta.data[:5]} ...")  # First 5 widths
    
    print("\n" + "=" * 70)
    
    return rbf, rbf_features


if __name__ == "__main__":
    # Run all examples
    model, output = example_windfarm_gno()
    decoder, predictions = example_probe_decoder_with_gen()
    rbf, rbf_features = example_learnable_rbf()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
