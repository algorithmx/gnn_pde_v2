# GeoTransolver Implementation: Framework Gap Analysis

## Overview

This report documents the gaps between the `gnn_pde_v2` framework's existing components and the requirements for implementing GeoTransolver (arXiv:2512.20399). The analysis is based on comparing the reference implementation in `docs/geotransolver/impl/` against the framework's component library (`gnn_pde_v2.components`, `gnn_pde_v2.core`).

The example implementation is at `examples/example_geotransolver.py`.

---

## 1. Cross-Attention for Context Integration

### What's Missing

`PhysicsTokenAttention` implements the Transolver-style slice-attend-deslice mechanism: it projects N input points onto G learned physics tokens, applies self-attention among those G tokens, then projects back to N points. This is a **closed system** — there is no way to inject external information (geometry features, global parameters) into the attention computation.

GeoTransolver's GALE attention requires **cross-attention** between the learned physics tokens and external context vectors. Specifically:

- **Queries** come from the input features (or the intermediate per-head representations)
- **Keys and values** come from pre-computed context tokens (geometry/global embeddings projected onto physical state space)
- The cross-attention output is **blended** with the self-attention output via a learnable sigmoid-gated mixing weight

### Why It Matters

Without cross-attention, the model has no mechanism to condition its attention patterns on geometric structure or global physical parameters. The self-attention among physics tokens operates purely on the input features — it cannot "see" the geometry or operating regime. This is the core architectural difference between Transolver (no geometry awareness) and GeoTransolver (geometry-aware).

### Reference Implementation

In the reference (`docs/geotransolver/impl/gale.py`, class `GALE`):

```python
self.cross_q = linear_layer(dim_head, dim_head)
self.cross_k = linear_layer(context_dim, dim_head)
self.cross_v = linear_layer(context_dim, dim_head)
self.state_mixing = nn.Parameter(torch.tensor(0.0))
```

Cross-attention is computed after self-attention on the slice tokens, then blended:

```python
mixing_weight = torch.sigmoid(self.state_mixing)
output = (1 - mixing_weight) * self_attention + mixing_weight * cross_attention
```

### Workaround

Implemented `GeoPhysicsTokenAttention` which wraps `PhysicsTokenAttention` for self-attention and adds a parallel cross-attention path with linear projections and `scaled_dot_product_attention`. The mixing weight is learnable (initialized at 0.0 so sigmoid starts at 0.5).

---

## 2. Context Projector (Slice-Only Physics Attention)

### What's Missing

The framework's `PhysicsTokenAttention` is a complete slice → attend → deslice pipeline. There is no way to use just the **slice** half (project input features onto learned physical states) without also running the deslice (projecting back to input space).

GeoTransolver needs to project geometry and global features onto learned physical states to create **context tokens** that are reused across all transformer blocks. This is a one-way projection — the context tokens are never desliced back to point space.

### Why It Matters

The context tokens serve as a compressed representation of the geometry and global parameters. By projecting them onto the same learned physical state space as the input features, the cross-attention mechanism can directly compare "what the model is currently looking at" (self-attention tokens) with "what the geometry looks like" (context tokens). Without this, the model would need to process raw geometry features through a generic MLP, losing the alignment with the physics-aware state space.

### Reference Implementation

In the reference (`docs/geotransolver/impl/context_projector.py`, class `ContextProjector`):

```python
def forward(self, x):
    # Project onto learned latent spaces
    projected_x, feature_projection = self.project_input_onto_slices(x)
    # Project onto physical state slices
    slice_projections = self.in_project_slice(projected_x)
    # Weighted aggregation into slice tokens
    _, slice_tokens = self.compute_slices_from_projections(
        slice_projections, feature_projection
    )
    return slice_tokens  # [B, H, S, D] — no deslice
```

### Workaround

Implemented `ContextProjector` which replicates the slice-only pattern: two linear projections (one for slice weights, one for content), temperature-scaled softmax, and weighted aggregation. Uses the same per-head structure as `PhysicsTokenAttention` but omits the deslice and output projection.

---

## 3. Multi-Scale Ball Query (BQWarp)

### What's Missing

GeoTransolver uses radius-based ball queries at multiple spatial scales to extract local geometric features. For each query point, it finds all neighbors within a given radius and processes their relative features through per-scale MLPs. This is inspired by DoMINO's multi-scale neighborhood construction.

The framework provides `knn_graph` (via `torch_cluster`) for k-nearest-neighbor graph construction, but no radius-based neighbor search. The reference implementation uses NVIDIA's `BQWarp` (ball query on GPU warps) which is not available in the framework.

### Why It Matters

Multi-scale local features capture geometric detail at different resolutions — fine-grained near-wall behavior (small radii) and far-field interactions (large radii). Without radius-based queries, the model cannot efficiently extract these features. KNN would give a fixed number of neighbors regardless of spatial density, which doesn't match the ball query semantics.

### Reference Implementation

In the reference (`docs/geotransolver/impl/context_projector.py`, class `GeometricFeatureProcessor`):

```python
self.bq_warp = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

def forward(self, query_points, key_features):
    _, neighbors = self.bq_warp(query_points, key_features)
    neighbors_flat = rearrange(neighbors, "b n k c -> b n (k c)")
    return torch.nn.functional.tanh(self.mlp(neighbors_flat))
```

### Workaround

Implemented `MultiScaleBallQuery` using pairwise distance computation (`torch.cdist`) followed by masking points beyond the radius and taking top-k nearest. This approximates ball query behavior but has O(N*M) memory for the distance matrix, which is acceptable for moderate mesh sizes but would need optimization for million-scale inputs.

---

## 4. GALE Attention Module

### What's Missing

The framework provides `PhysicsTokenAttention` (Transolver-style) and `PhysicsTokenAttentionV3` (Transolver-3 with tiling), but neither supports cross-attention with external context. There is no "GALE" (Geometry-Aware Latent Embeddings) variant that combines self-attention on physics tokens with cross-attention to geometry/global context.

### Why It Matters

GALE is the central architectural innovation of GeoTransolver. It enables the model to condition its attention patterns on geometric structure at every layer, not just at the input. The persistent geometry-context projection ensures that the evolving latent states remain aligned with the underlying domain throughout the forward pass.

### Reference Implementation

In the reference (`docs/geotransolver/impl/gale.py`), GALE extends `PhysicsAttentionIrregularMesh`:

```python
class GALE(PhysicsAttentionIrregularMesh):
    def __init__(self, dim, heads, dim_head, context_dim, ...):
        super().__init__(dim, heads, dim_head, ...)
        self.cross_q = linear_layer(dim_head, dim_head)
        self.cross_k = linear_layer(context_dim, dim_head)
        self.cross_v = linear_layer(context_dim, dim_head)
        self.state_mixing = nn.Parameter(torch.tensor(0.0))
```

### Workaround

Implemented `GeoPhysicsTokenAttention` which wraps `PhysicsTokenAttention` and adds a parallel cross-attention path. This is architecturally equivalent to GALE but uses the framework's existing `PhysicsTokenAttention` for the self-attention component rather than reimplementing the full slice-attend-deslice pipeline.

---

## 5. Global Context Builder

### What's Missing

GeoTransolver's context construction involves multiple coordinated steps:

1. **Geometry tokenization**: Project geometry features onto physical state slices via `ContextProjector`
2. **Global embedding tokenization**: Project global parameters (e.g., Reynolds number) onto physical state slices
3. **Multi-scale local features**: Extract local geometric features via ball queries at multiple radii
4. **Context concatenation**: Combine all context sources into a unified context tensor

The framework has no orchestrator for this workflow. Each step requires custom code, and the interface between them (tensor shapes, concatenation dimensions) must be managed manually.

### Why It Matters

The context builder ensures that geometry, global parameters, and local features are all projected onto the same learned physical state space and combined consistently. Without a dedicated builder, the implementation is more error-prone and harder to maintain.

### Reference Implementation

In the reference (`docs/geotransolver/impl/context_projector.py`, class `GlobalContextBuilder`):

```python
class GlobalContextBuilder(nn.Module):
    def __init__(self, functional_dims, geometry_dim, global_dim, ...):
        self.local_extractors = MultiScaleFeatureExtractor(...)
        self.geometry_tokenizer = ContextProjector(...)
        self.global_tokenizer = ContextProjector(...)

    def build_context(self, local_embeddings, local_positions, geometry, global_embedding):
        context_parts = []
        # Multi-scale local features -> context tokens
        context_parts.extend(self.local_extractors.extract_context_features(...))
        # Geometry -> context tokens
        context_parts.append(self.geometry_tokenizer(geometry))
        # Global -> context tokens
        context_parts.append(self.global_tokenizer(global_embedding))
        return torch.cat(context_parts, dim=-1)
```

### Workaround

Implemented context building inline in `GeoTransolver.forward()`, calling `ContextProjector` for geometry and global features, and `MultiScaleBallQuery` for local features. The concatenation is handled directly in the forward pass rather than through a dedicated builder class.

---

## 6. ConcreteDropout (Minor Gap)

### What's Missing

The reference implementation uses `ConcreteDropout` (a learned dropout rate) instead of standard `nn.Dropout`. This is a regularization technique where the dropout probability is learned during training.

### Why It Matters

Minor — standard dropout works fine for most cases. ConcreteDropout provides slightly better regularization by adapting the dropout rate to the data, but it's not critical for correctness.

### Workaround

Used standard `nn.Dropout` throughout. The example uses `dropout=0.0` to match the reference defaults.

---

## Summary

| Component | Framework Status | Workaround |
|-----------|-----------------|------------|
| PhysicsTokenAttention | ✅ Available | Used directly for self-attention |
| Cross-attention for context | ❌ Missing | `GeoPhysicsTokenAttention` (manual Q/K/V + SDPA) |
| Context projector | ❌ Missing | `ContextProjector` (slice-only physics attention) |
| Multi-scale ball query | ❌ Missing (has knn_graph) | `MultiScaleBallQuery` (cdist + topk) |
| GALE attention | ❌ Missing | `GeoPhysicsTokenAttention` (wraps PhysicsTokenAttention) |
| Global context builder | ❌ Missing | Inline in `GeoTransolver.forward()` |
| ConcreteDropout | ❌ Missing | Standard `nn.Dropout` |

The framework provides a solid foundation for the self-attention component (PhysicsTokenAttention) but lacks the geometry-aware extensions (cross-attention, context projection, ball queries) that differentiate GeoTransolver from Transolver. These gaps are addressable through custom modules, as demonstrated in the example implementation.
