# Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries

**Paper**: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"  
**Authors**: Hang Zhou, Haixu Wu, Haonan Shangguan, Yuezhou Ma, Huikun Weng, Jianmin Wang, Mingsheng Long (Tsinghua University)  
**arXiv**: [2602.04940](https://arxiv.org/abs/2602.04940)  
**Date**: February 2026

## Overview

Transolver-3 is a highly scalable framework designed for high-fidelity physics simulations on industrial-scale geometries. It extends Transolver to handle meshes with **160+ million cells** through three key innovations:

1. **Faster Slice/Deslice**: Exploits matrix multiplication associativity to reduce O(N·D·M) operations
2. **Geometry Slice Tiling**: Partitions computation to avoid materializing full N×M slice weight matrices
3. **Physical State Caching**: Decouples physical state estimation from field prediction for inference

## Key Improvements

### 1. Memory Efficiency

| Metric | Transolver | Transolver++ | Transolver-3 |
|--------|------------|--------------|--------------|
| Single-GPU Capacity | ~100K cells | ~700K cells | **~2.9M cells** |
| Slice Weights Memory | O(N·M) | O(N·M) | **O(N·tile_size)** |
| O(N·D·M) Operations | 5 | 4 | **3** |

### 2. Performance on Industrial Benchmarks

Transolver-3 achieves state-of-the-art results on three challenging benchmarks:

| Benchmark | Mesh Size | Transolver-3 vs Transolver++ |
|-----------|-----------|------------------------------|
| NASA-CRM | 400K surface | -8.4% error |
| AhmedML | 20M volume | -14.7% error |
| DrivAerML | **160M volume** | -11.4% error |

## Implementation

### Basic Usage

```python
from gnn_pde_v2.components import PhysicsTokenAttentionV3

# Standard Transolver-3 attention
attn = PhysicsTokenAttentionV3(
    dim=256,
    n_tokens=64,
    n_heads=8,
    # Transolver-3 optimizations
    use_tiling=True,           # Enable geometry slice tiling
    tile_size=100000,          # Points per tile (paper recommends 100k)
    use_gradient_checkpointing=True,  # Trade compute for memory
    use_faster_slice=True,     # Enable operation reordering
)

# Forward pass - automatically handles tiling for large meshes
x = torch.randn(1, 1000000, 256)  # 1M points
out = attn(x)  # Works even on limited GPU memory
```

### Complete Model

```python
from gnn_pde_v2.components.transolver_v3 import TransolverV3

model = TransolverV3(
    space_dim=3,          # 3D coordinates
    input_dim=4,          # Input field channels
    output_dim=4,         # Output field channels
    hidden_dim=256,
    num_layers=8,
    num_heads=8,
    slice_num=64,
    # Transolver-3 optimizations
    use_tiling=True,
    tile_size=100000,
)
```

### Geometry Amortized Training

Train on random subsets of industrial-scale meshes:

```python
from gnn_pde_v2.components import GeometryAmortizedTraining

# Simulate 160M cell industrial mesh
full_mesh_size = 160_000_000
subset_size = 400_000  # Train on 400k cells per iteration

amortizer = GeometryAmortizedTraining(
    full_mesh_size=full_mesh_size,
    subset_size=subset_size,
    seed=42,
)

for epoch in range(num_epochs):
    for batch in dataloader:
        # Sample random subset for this iteration
        subset_indices = amortizer.get_subset_indices()
        
        x_subset = x_full[subset_indices]
        y_subset = y_full[subset_indices]
        
        loss = model(x_subset, y_subset)
        loss.backward()
        optimizer.step()
```

### Physical State Caching (Inference)

For inference on full-resolution meshes exceeding GPU memory:

```python
from gnn_pde_v2.components import PhysicalStateCache

# Create cache handler
cache = PhysicalStateCache(
    model=model,
    chunk_size=50000,  # Process 50k points at a time
)

# Build physical state cache layer-by-layer
cache_states = cache.build_cache(
    x_full,  # Can be 100M+ points
    num_layers=8,
)

# Decode predictions at specific points
predictions = cache.decode_points(
    query_indices=important_surface_points,
    cache=cache_states,
)
```

## Architecture Details

### Faster Slice/Deslice

The original Physics-Attention performs 5 operations with O(N·D·M) complexity:

```
1. in_project_x(x)     -> [B, N, D]
2. in_project_fx(x)    -> [B, N, D]
3. slice_weight_proj   -> [B, N, M]
4. slice: fx @ weights -> [B, M, D]
5. deslice: tokens @ weights.T -> [B, N, D]
```

Transolver-3 reduces this to 3 operations by exploiting associativity:

```
Optimized: Move projections into slice domain
- Eliminates intermediate tensors
- Reduces memory from 3×[B,N,D] to 1×[B,N,D]
- ~60% latency reduction in practice
```

### Geometry Slice Tiling

Instead of materializing the full N×M slice weight matrix:

```python
# Without tiling (memory: O(N·M))
slice_weights = softmax(slice_logits)  # [B, H, N, M]
tokens = einsum('bhnd,bhng->bhgd', fx, slice_weights)

# With tiling (memory: O(N·tile_size))
for tile in partition(N, tile_size):
    slice_weights_tile = softmax(slice_logits[tile])  # [B, H, tile, M]
    tokens += einsum('bhnd,bhng->bhgd', fx[tile], slice_weights_tile)
```

### Tile Size Selection

| Tile Size | Memory (GB) | Latency (ms) | Num Tiles |
|-----------|-------------|--------------|-----------|
| 800K | 22.16 | 404 | 1 |
| 200K | 13.62 | 410 | 4 |
| **100K** | **12.28** | **429** | **8** |
| 20K | 11.27 | 453 | 40 |
| 10K | 11.09 | 656 | 80 |
| 5K | 11.04 | 1307 | 160 |

**Recommendation**: Use `tile_size=100000` as the default balance between memory and speed.

## API Reference

### PhysicsTokenAttentionV3

```python
PhysicsTokenAttentionV3(
    # Standard Transolver parameters
    dim: int,
    n_tokens: int = 32,
    n_heads: int = 8,
    dropout: float = 0.0,
    temperature: float = 0.5,
    temperature_mode: str = 'learnable_scalar',
    use_gumbel_softmax: bool = False,
    min_temperature: float = 0.1,
    use_slice_normalization: bool = True,
    use_learnable_tokens: bool = False,
    qkv_mode: str = 'direct',
    use_orthogonal_init: bool = True,
    
    # Transolver-3 specific
    use_tiling: bool = True,           # Enable geometry slice tiling
    tile_size: int = 100000,          # Points per tile
    use_gradient_checkpointing: bool = True,  # Trade compute for memory
    use_faster_slice: bool = True,    # Enable operation reordering
)
```

### GeometryAmortizedTraining

```python
GeometryAmortizedTraining(
    full_mesh_size: int,    # Total mesh points (e.g., 160_000_000)
    subset_size: int,       # Points per training iteration (e.g., 400_000)
    seed: Optional[int] = None,
)

# Methods
.get_subset_indices(device=None) -> Tensor  # Random subset for iteration
.apply_subset(x, y=None) -> Tuple[Tensor, Optional[Tensor]]
```

### PhysicalStateCache

```python
PhysicalStateCache(
    model: nn.Module,
    chunk_size: int = 50000,
    device: Optional[torch.device] = None,
)

# Methods
.build_cache(x: Tensor, num_layers: int) -> List[Tensor]
.decode_points(query_indices: Tensor, cache: List[Tensor]) -> Tensor
```

## Comparison with Previous Versions

| Feature | Transolver (ICML'24) | Transolver++ | **Transolver-3** |
|---------|---------------------|--------------|------------------|
| Physics-Attention | ✓ | ✓ | ✓ (optimized) |
| Slice Tiling | ✗ | ✗ | **✓** |
| Faster Slice/Deslice | ✗ | Partial | **✓** |
| Physical State Caching | ✗ | ✗ | **✓** |
| Amortized Training | ✗ | ✗ | **✓** |
| Max Mesh (1 GPU) | ~100K | ~700K | **~2.9M** |
| Max Mesh (multi-GPU) | - | ~1M | **160M+** |

## Design Rationale

Transolver-3 follows the same philosophy as the original Transolver:

> **Learn intrinsic physical states rather than processing raw mesh points directly**

The key insight is that while industrial-scale meshes may have 100M+ cells, the underlying physical phenomena often have much lower intrinsic dimensionality. By:

1. **Projecting** N mesh points to M physics tokens (M << N)
2. **Processing** attention among M tokens (O(M²) complexity)
3. **Distributing** back to N points

Transolver-3 achieves **linear scaling** with mesh size while maintaining the model's full expressive capacity.

## Citation

If you use Transolver-3 in your research, please cite:

```bibtex
@article{zhou2026transolver3,
  title={Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries},
  author={Zhou, Hang and Wu, Haixu and Shangguan, Haonan and Ma, Yuezhou and Weng, Huikun and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2602.04940},
  year={2026}
}
```

## See Also

- [Transolver Paper (ICML 2024)](https://arxiv.org/abs/2404.14088)
- [Transolver GitHub](https://github.com/thuml/Transolver)
- Original Transolver in this framework: `PhysicsTokenAttention`
