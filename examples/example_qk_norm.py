"""
Example: QK-Norm Attention (Query-Key Normalization)

This example demonstrates the QK-Norm attention mechanism from the paper:
"Query-Key Normalization for Transformers" (Henry et al., 2020)

Key features:
1. Applies L2 normalization along the head dimension to Q and K
2. Uses a learnable scalar parameter g instead of fixed sqrt(d) scaling
3. Converts dot products to cosine similarities scaled by g

This technique helps prevent softmax saturation and enables more diffuse
attention patterns, which is particularly useful for low-resource translation.

Reference: https://github.com/CyndxAI/QKNorm
"""

import torch
import math
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import (
    MultiHeadAttention,
    QKNormMultiHeadAttention,
)


def example_basic_qk_norm():
    """Basic QK-Norm attention without position encoding."""
    print("=" * 60)
    print("Example 1: Basic QK-Norm Attention")
    print("=" * 60)
    
    # Create QK-Norm attention module
    # Default init_g=7.0 (log2(128)) as per paper's recommendation
    attn = QKNormMultiHeadAttention(dim=64, n_heads=8, dropout=0.0)
    
    x = torch.randn(10, 64)  # 10 nodes, 64 features
    
    out = attn(x)
    print(f"Input shape:     {x.shape}")
    print(f"Output shape:   {out.shape}")
    print(f"Learnable g:    {attn.get_g_value():.4f}")
    print()


def example_qk_norm_with_custom_init():
    """QK-Norm with custom initialization based on sequence length."""
    print("=" * 60)
    print("Example 2: QK-Norm with Custom Initialization")
    print("=" * 60)
    
    # Paper's initialization formula: g0 = log2(L2 - L)
    # where L is the 97.5th percentile sequence length
    # For sequence length 100: g0 = log2(128 - 100) ≈ 4.8
    seq_len = 100
    init_g = math.log2(128 - seq_len) if seq_len < 128 else 1.0
    
    attn = QKNormMultiHeadAttention(
        dim=64, 
        n_heads=8, 
        dropout=0.0,
        init_g=init_g,
    )
    
    x = torch.randn(10, 64)
    out = attn(x)
    print(f"Sequence length:        {seq_len}")
    print(f"Computed init_g:       {init_g:.4f}")
    print(f"Learnable g (current): {attn.get_g_value():.4f}")
    print(f"Input shape:            {x.shape}")
    print(f"Output shape:           {out.shape}")
    print()


def example_comparison_with_standard_attention():
    """Compare standard attention vs QK-Norm attention."""
    print("=" * 60)
    print("Example 3: Comparison: Standard vs QK-Norm Attention")
    print("=" * 60)
    
    dim = 64
    n_heads = 8
    n_nodes = 20
    
    # Standard attention
    standard_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, dropout=0.0)
    
    # QK-Norm attention
    qk_norm_attn = QKNormMultiHeadAttention(dim=dim, n_heads=n_heads, dropout=0.0)
    
    # Same input
    x = torch.randn(n_nodes, dim)
    
    # Forward passes
    out_standard = standard_attn(x)
    out_qk_norm = qk_norm_attn(x)
    
    print(f"Input shape:            {x.shape}")
    print(f"Standard output:       {out_standard.shape}")
    print(f"QK-Norm output:        {out_qk_norm.shape}")
    print(f"Standard scale:        sqrt(d) = sqrt({dim // n_heads}) = {math.sqrt(dim // n_heads):.4f}")
    print(f"QK-Norm learnable g:   {qk_norm_attn.get_g_value():.4f}")
    
    # Compare outputs
    diff = torch.abs(out_standard - out_qk_norm).mean()
    print(f"Mean absolute diff:    {diff:.4f}")
    print()
    print("Note: QK-Norm produces different attention patterns because:")
    print("  1. Q and K are L2-normalized (cosine similarity instead of dot product)")
    print("  2. Uses learnable scaling g instead of fixed sqrt(d)")
    print()


def example_batch_with_positions():
    """QK-Norm attention with batch and positions."""
    print("=" * 60)
    print("Example 4: Batched QK-Norm with Positions")
    print("=" * 60)
    
    attn = QKNormMultiHeadAttention(
        dim=64,
        n_heads=8,
        use_relative_positions=True,
        position_dim=2,
        num_position_buckets=32,
    )
    
    # Batch of 2, 10 nodes each
    x = torch.randn(2, 10, 64)
    positions = torch.randn(2, 10, 2)
    
    out = attn(x, positions=positions)
    print(f"Input shape:      {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Output shape:     {out.shape}")
    print(f"Learnable g:     {attn.get_g_value():.4f}")
    print()


def example_attention_scores():
    """Compare attention score distributions."""
    print("=" * 60)
    print("Example 5: Attention Score Distribution Comparison")
    print("=" * 60)
    
    dim = 64
    n_heads = 4
    n_nodes = 16
    head_dim = dim // n_heads
    
    # Create simple attention modules
    standard_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, dropout=0.0)
    qk_norm_attn = QKNormMultiHeadAttention(dim=dim, n_heads=n_heads, dropout=0.0)
    
    # Get internal scores by hooking
    standard_scores = None
    qk_norm_scores = None
    
    def hook_fn(module, input, output):
        nonlocal standard_scores
        # Access would require modification - we'll compute manually
    
    x = torch.randn(n_nodes, dim)
    
    # For standard attention, scale = sqrt(d)
    # For QK-Norm, we compute manually
    qkv = standard_attn.qkv(x).reshape(n_nodes, 3, n_heads, head_dim).permute(1, 2, 0, 3)
    q_std, k_std, v_std = qkv[0], qkv[1], qkv[2]
    scores_std = (q_std @ k_std.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # QK-Norm
    qkv_qk = qk_norm_attn.qkv(x).reshape(n_nodes, 3, n_heads, head_dim).permute(1, 2, 0, 3)
    q_qk, k_qk, v_qk = qkv_qk[0], qkv_qk[1], qkv_qk[2]
    
    # L2 normalize
    q_qk_norm = torch.nn.functional.normalize(q_qk, p=2, dim=-1)
    k_qk_norm = torch.nn.functional.normalize(k_qk, p=2, dim=-1)
    g = qk_norm_attn.get_g_value()
    scores_qk = torch.matmul(q_qk_norm, k_qk_norm.transpose(-2, -1)) * g
    
    print(f"Standard attention scores:")
    print(f"  Min: {scores_std.min():.4f}, Max: {scores_std.max():.4f}")
    print(f"  Mean: {scores_std.mean():.4f}, Std: {scores_std.std():.4f}")
    print()
    print(f"QK-Norm attention scores (after softmax):")
    attn_weights = torch.softmax(scores_qk, dim=-1)
    print(f"  Min: {attn_weights.min():.4f}, Max: {attn_weights.max():.4f}")
    print(f"  Mean: {attn_weights.mean():.4f}, Std: {attn_weights.std():.4f}")
    print()
    print("Note: QK-Norm keeps attention scores in a narrower range,")
    print("preventing the 'winner-take-all' behavior of standard attention.")
    print()


def example_learnable_parameter_training():
    """Demonstrate that g is learnable and can be trained."""
    print("=" * 60)
    print("Example 6: Learnable Parameter g Training")
    print("=" * 60)
    
    # Create QK-Norm attention
    attn = QKNormMultiHeadAttention(dim=64, n_heads=8, dropout=0.0)
    
    print(f"Initial g: {attn.get_g_value():.4f}")
    
    # Simple training loop
    optimizer = torch.optim.Adam(attn.parameters(), lr=0.01)
    x = torch.randn(20, 64)
    
    for epoch in range(10):
        optimizer.zero_grad()
        out = attn(x)
        # Simple dummy loss for demonstration
        loss = out.mean()
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: g = {attn.get_g_value():.4f}")
    
    print()
    print("Note: The learnable parameter g adapts during training")
    print("to find the optimal scaling for the attention mechanism.")
    print()


def example_with_graphs():
    """Show how to use QK-Norm with GraphsTuple."""
    print("=" * 60)
    print("Example 7: QK-Norm with GraphsTuple")
    print("=" * 60)
    
    # QK-Norm can be used directly as a module on graph node features
    attn = QKNormMultiHeadAttention(
        dim=64,
        n_heads=8,
        dropout=0.0,
    )
    
    # Create a graph
    graph = GraphsTuple(
        nodes=torch.randn(20, 64),  # 20 nodes, 64 features
        n_node=torch.tensor([20]),
    )
    
    # Apply attention directly to node features
    nodes = graph.nodes
    out_nodes = attn(nodes)
    
    print(f"Input nodes shape:  {graph.nodes.shape}")
    print(f"Output nodes shape: {out_nodes.shape}")
    print(f"Graph n_node:       {graph.n_node}")
    print(f"Learnable g:        {attn.get_g_value():.4f}")
    print()


def example_fixed_g():
    """QK-Norm with non-learnable g (fixed scaling)."""
    print("=" * 60)
    print("Example 8: QK-Norm with Fixed g")
    print("=" * 60)
    
    # Fixed g (non-learnable)
    attn = QKNormMultiHeadAttention(
        dim=64,
        n_heads=8,
        dropout=0.0,
        init_g=5.0,
        learnable_g=False,  # Fixed parameter
    )
    
    x = torch.randn(10, 64)
    out = attn(x)
    
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Fixed g value: {attn.get_g_value():.4f}")
    print()
    print("Note: With learnable_g=False, g is a buffer (not a Parameter)")
    print("and won't be updated during training.")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("* QK-Norm (Query-Key Normalization) Examples")
    print("*" * 60)
    print()
    
    example_basic_qk_norm()
    example_qk_norm_with_custom_init()
    example_comparison_with_standard_attention()
    example_batch_with_positions()
    example_attention_scores()
    example_learnable_parameter_training()
    example_with_graphs()
    example_fixed_g()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    
    print("""
Reference Paper:
"Query-Key Normalization for Transformers"
Henry et al., EMNLP 2020

Key Equations:
  - L2 normalization: q̂ = q / ||q||  (along head dimension)
  - Attention: softmax(g · q̂ · k̂^T) · V
  - Initialization: g0 = log2(L2 - L) where L is 97.5th percentile seq length

Benefits:
  - Prevents softmax saturation
  - Enables more diffuse attention patterns
  - Improves low-resource translation
""")