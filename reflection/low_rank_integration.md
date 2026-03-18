# Low-Rank Approximation Integration into Framework

## Summary

Successfully integrated memory-efficient symmetric low-rank approximation into the `gnn_pde_v2` framework by extending `EdgeConditionedConvBlock` with a new `edge_weight_type='low_rank'` option.

## Mathematical Foundation

### Full-Rank Message Passing

Standard edge-conditioned convolution computes messages using full weight matrices:

```
W_e ∈ R^{d×d}  (full weight matrix per edge)
m_ij = W_e · x_j
```

Memory per edge: **d² values**

### Symmetric Low-Rank Approximation

Instead of computing full W_e, we factorize it using a symmetric decomposition:

```
W_e ≈ U_e · U_e^T  where U_e ∈ R^{d×r} and r << d

Message computation: m_ij = U_e · U_e^T · x_j

Step 1: h_e = U_e^T · x_j      (project to rank-r space)
Step 2: m_ij = U_e · h_e       (project back to d-dim)
```

Memory per edge: **d×r values**

### Memory Reduction Ratio

```
Reduction ratio = d² / (d×r) = d/r

For d=64, r=8:
- Full-rank: 64×64 = 4,096 values
- Low-rank:  64×8  =   512 values
- Reduction: 8×
```

## Framework Integration

### API Design

Added `low_rank` parameter to `EdgeConditionedConvBlock`:

```python
class EdgeConditionedConvBlock(MessagePassingBase):
    def __init__(
        self,
        latent_dim: int,
        edge_latent_dim: int,
        hidden_dim: int = 128,
        edge_weight_type: str = 'full',  # 'full', 'vector', 'scalar', 'low_rank'
        low_rank: int = 0,               # NEW: rank for approximation
        ...
    ):
```

### Usage Examples

```python
from gnn_pde_v2.components import EdgeConditionedConvBlock

# Full-rank (original behavior)
block_full = EdgeConditionedConvBlock(
    latent_dim=64,
    edge_latent_dim=7,
    edge_weight_type='full',
)

# Low-rank (memory-efficient)
block_lowrank = EdgeConditionedConvBlock(
    latent_dim=64,
    edge_latent_dim=7,
    edge_weight_type='low_rank',
    low_rank=8,  # 8× memory reduction
)
```

### Implementation Details

The low-rank message computation uses `torch.einsum` for clarity and efficiency:

```python
def compute_messages(self, graph):
    src_x = graph.nodes[graph.senders]  # [E, d]
    w = self.edge_weight_net(graph.edges)
    
    if self.edge_weight_type == 'low_rank':
        # Reshape to get U_e factors [E, d, r]
        edge_u = w.view(-1, H, self.low_rank)
        
        # Symmetric low-rank: M_e = U_e · U_e^T · x_j
        # Step 1: Project to rank-r space
        h_e = torch.einsum('ed,edr->er', src_x, edge_u)  # [E, r]
        
        # Step 2: Project back to d-dim
        msg = torch.einsum('er,edr->ed', h_e, edge_u)    # [E, d]
    
    return msg, None
```

## Integration with GNNSolver

The simplified GNNSolver example now supports low-rank mode:

```python
from examples.example_gnn_solver_simplified import GNNSolverLowRankSimplified

# Memory-efficient model
model = GNNSolverLowRankSimplified(
    in_dim=10,
    latent_dim=64,
    out_dim=1,
    kernel_width=32,
    edge_dim=7,
    rank=8,              # Low-rank approximation
    num_layers=13,
    use_batchnorm=True,
)
```

Or use the main class with `low_rank` parameter:

```python
from examples.example_gnn_solver_simplified import GNNSolverSimplified

model = GNNSolverSimplified(
    ...,
    low_rank=8,  # Enable low-rank mode
)
```

## Verification Results

### Parameter Count Comparison

| Configuration | Parameters | Reduction |
|--------------|------------|-----------|
| Full-rank (13 layers) | 1,781,573 | - |
| Low-rank (13 layers, r=8) | 244,037 | **86.3%** |

### Memory Per Edge

| Mode | Values per Edge | Reduction |
|------|-----------------|-----------|
| Full-rank (d=64) | 4,096 | - |
| Low-rank (r=8) | 512 | **8×** |
| Low-rank (r=16) | 1,024 | **4×** |

### Test Output

```
============================================================
Simplified GNNSolver using Framework Components
============================================================

--- Vanilla Version (No BatchNorm) ---
Output shape: torch.Size([100, 6])
Total parameters: 1,779,685

--- Working Version (With BatchNorm) ---
Output shape: torch.Size([100, 6])
Total parameters: 1,781,573

--- Low-Rank Version (Memory Efficient) ---
Output shape: torch.Size([100, 6])
Total parameters: 244,037

Processor parameter reduction: 87.3%
Memory per edge: 4096 -> 512 values (8.0x reduction)
```

## Benefits of Low-Rank Approximation

### 1. Memory Efficiency
- **8× reduction** in edge weight memory for d=64, r=8
- Enables training on larger graphs or with larger batch sizes

### 2. Computational Efficiency
- Fewer parameters to compute and store during forward pass
- Faster backward pass due to reduced gradient computation

### 3. Physical Interpretation
- Symmetric factorization produces positive semi-definite weight matrices
- May better match physical symmetries in Green's function kernels
- Useful for PDE problems with inherent symmetries

### 4. Flexible Trade-off
- Tune `low_rank` parameter for desired memory/accuracy trade-off
- r = d/8 to d/4 typically provides good balance

## Trade-offs and Considerations

### Model Capacity
- Low-rank approximation reduces model capacity
- May require more layers to achieve same expressiveness
- Best for problems with inherent low-rank structure

### Initialization
- Same edge MLP initialization (std=0.1) applies
- Low-rank matrices initialized via Xavier uniform

### When to Use

✅ **Recommended for:**
- Large latent dimensions (d ≥ 64)
- Memory-constrained environments
- Problems with known low-rank structure
- Inference/deployment scenarios

❌ **Not recommended for:**
- Very small latent dimensions (d < 32)
- Problems requiring full-rank transformations
- When maximum accuracy is critical

## Files Modified/Created

| File | Change |
|------|--------|
| `components/processors.py` | Added `low_rank` support to `EdgeConditionedConvBlock` |
| `examples/example_gnn_solver_simplified.py` | Updated to use framework low-rank support |
| `reflection/low_rank_integration.md` | This documentation |

## Backward Compatibility

✅ **Fully backward compatible**
- Default `low_rank=0` maintains original behavior
- Existing code continues to work without changes
- New parameter is opt-in only

## Future Enhancements

1. **Asymmetric Low-Rank**: Support U·V^T factorization (2× parameters but more expressive)
2. **Learnable Rank**: Dynamically adjust rank during training
3. **Hierarchical Low-Rank**: Different ranks for different layers
4. **Block-Diagonal Low-Rank**: Partition latent space into blocks with separate ranks

## References

- Original implementation: `/home/dabajabaza/Documents/Workspace/MoM/Projects/train/gnn_solver/nn_customized.py`
- Symmetric low-rank theory: Based on matrix factorization literature
- Use case: GNNSolver electromagnetic/physical simulation problems
