"""
Example: Relative Position Encoding in MultiHeadAttention

This example demonstrates how to use relative position encoding (RPE) with
MultiHeadAttention for PDE applications. RPE allows the model to attend based
on spatial relationships between nodes.
"""

import torch
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import (
    MultiHeadAttention,
    TransformerBlock,
    TransformerProcessor,
    RelativePositionEncoding,
)


def example_basic_attention():
    """Basic MultiHeadAttention without position encoding."""
    print("=" * 60)
    print("Example 1: Basic MultiHeadAttention (no positions)")
    print("=" * 60)
    
    attn = MultiHeadAttention(dim=64, n_heads=8, dropout=0.0)
    x = torch.randn(10, 64)  # 10 nodes, 64 features
    
    out = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print()


def example_relative_position_learned():
    """MultiHeadAttention with learned relative position encoding."""
    print("=" * 60)
    print("Example 2: MultiHeadAttention with Learned RPE")
    print("=" * 60)
    
    attn = MultiHeadAttention(
        dim=64,
        n_heads=8,
        use_relative_positions=True,
        position_dim=2,  # 2D positions
        num_position_buckets=32,
        position_encoding_type='learned',
    )
    
    x = torch.randn(10, 64)  # 10 nodes, 64 features
    positions = torch.randn(10, 2)  # 10 nodes with 2D positions
    
    out = attn(x, positions=positions)
    print(f"Input shape:     {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Output shape:    {out.shape}")
    print(f"Position bias shape: {attn.position_encoding.position_bias.shape}")
    print()


def example_relative_position_sinusoidal():
    """MultiHeadAttention with sinusoidal relative position encoding."""
    print("=" * 60)
    print("Example 3: MultiHeadAttention with Sinusoidal RPE")
    print("=" * 60)
    
    attn = MultiHeadAttention(
        dim=64,
        n_heads=8,
        use_relative_positions=True,
        position_dim=3,  # 3D positions
        num_position_buckets=32,
        position_encoding_type='sinusoidal',
    )
    
    x = torch.randn(10, 64)  # 10 nodes, 64 features
    positions = torch.randn(10, 3)  # 10 nodes with 3D positions
    
    out = attn(x, positions=positions)
    print(f"Input shape:     {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Output shape:    {out.shape}")
    print()


def example_batched_with_positions():
    """Batched MultiHeadAttention with positions."""
    print("=" * 60)
    print("Example 4: Batched MultiHeadAttention with RPE")
    print("=" * 60)
    
    attn = MultiHeadAttention(
        dim=64,
        n_heads=8,
        use_relative_positions=True,
        position_dim=2,
    )
    
    x = torch.randn(2, 10, 64)  # batch=2, 10 nodes, 64 features
    positions = torch.randn(2, 10, 2)  # batch=2, 10 nodes, 2D positions
    
    out = attn(x, positions=positions)
    print(f"Input shape:     {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Output shape:    {out.shape}")
    print()


def example_transformer_processor():
    """TransformerProcessor with relative position encoding on graphs."""
    print("=" * 60)
    print("Example 5: TransformerProcessor with RPE on Graphs")
    print("=" * 60)
    
    processor = TransformerProcessor(
        latent_dim=64,
        n_layers=4,
        n_heads=8,
        use_relative_positions=True,
        position_dim=2,
    )
    
    # Create a graph with node features and positions
    graph = GraphsTuple.from_flat(
        nodes=torch.randn(20, 64),  # 20 nodes, 64 features
        positions=torch.randn(20, 2),  # 20 nodes with 2D positions
        n_node=torch.tensor([20]),
    )
    
    out = processor(graph)
    print(f"Input nodes shape:  {graph.nodes.shape}")
    print(f"Positions shape:    {graph.positions.shape}")
    print(f"Output nodes shape: {out.nodes.shape}")
    print()


def example_spatial_awareness():
    """Demonstrate that attention is spatially aware."""
    print("=" * 60)
    print("Example 6: Spatial Awareness Demonstration")
    print("=" * 60)
    
    # Create a simple 1D grid
    n = 10
    x = torch.randn(n, 64)
    positions = torch.arange(n).float().unsqueeze(1)  # [0, 1, 2, ..., 9]
    
    attn = MultiHeadAttention(
        dim=64,
        n_heads=2,
        use_relative_positions=True,
        position_dim=1,
        num_position_buckets=8,
        max_distance=10.0,
    )
    
    # Forward pass
    out = attn(x, positions=positions)
    
    # Check that position bias is computed
    position_bias = attn.position_encoding(positions)
    print(f"Position bias shape: {position_bias.shape}")
    print(f"Position bias for head 0:\n{position_bias[0]}")
    print()
    print("Note: The position bias matrix shows how attention scores are")
    print("adjusted based on spatial distance between nodes.")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("* Relative Position Encoding Examples")
    print("*" * 60)
    print()
    
    example_basic_attention()
    example_relative_position_learned()
    example_relative_position_sinusoidal()
    example_batched_with_positions()
    example_transformer_processor()
    example_spatial_awareness()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
