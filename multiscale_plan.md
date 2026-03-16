# Multiscale Architecture Enhancement Plan for GNN-PDE v2

## Executive Summary

Based on the code review and analysis of four key papers (FNO, Graph U-Nets, MGKN, U-FNO), this plan proposes a comprehensive multiscale enhancement for the GNN-PDE v2 framework. The framework currently lacks:
- U-Net style skip connections
- Hierarchical graph pooling/unpooling
- Multi-resolution processing

These capabilities are critical for turbulence, multi-physics, and large-domain simulations.

---

## 1. Key Insights from Papers

### 1.1 Fourier Neural Operator (FNO) - Li et al. 2021
**Key Multiscale Features:**
- Operates in Fourier space where low frequencies = large scales, high frequencies = small scales
- Mode truncation acts as a learnable bandpass filter
- Zero-shot super-resolution via padding in Fourier space
- Quasi-linear complexity O(n log n)

**Relevant for Framework:**
- Current FNO implementation exists but lacks multi-resolution FNO blocks
- Need: Multi-band FNO that processes different frequency bands separately

### 1.2 Graph U-Nets - Gao & Ji 2019
**Key Multiscale Features:**
- Graph pooling (gPool): Adaptive node selection via trainable projection
- Graph unpooling (gUnpool): Position-aware restoration using stored indices
- Skip connections between encoder-decoder levels
- Graph power augmentation (A^2) for connectivity

**Relevant for Framework:**
- No graph pooling/unpooling exists
- Need: Full encoder-decoder architecture for graphs with skip connections

### 1.3 Multipole Graph Neural Operator (MGKN) - Li et al. 2020
**Key Multiscale Features:**
- V-cycle algorithm: Downward pass (fine→coarse) + Upward pass (coarse→fine)
- Multi-level graphs with inducing points
- Kernel matrix decomposition: K = K1 + K2 + ... + KL (different ranges)
- Linear complexity O(N) via hierarchical structure

**Relevant for Framework:**
- No hierarchical graph processing exists
- Need: V-cycle processor with learnable transitions between levels

### 1.4 U-FNO - Wen et al. 2022
**Key Multiscale Features:**
- Combines FNO (global) + U-Net (local) in each layer
- U-Fourier layer: spectral conv + U-Net + skip connection
- Better handling of high-frequency information
- Superior for multi-phase flow with sharp fronts

**Relevant for Framework:**
- Current FNO lacks U-Net enhancement
- Need: Hybrid blocks combining spectral and local processing

---

## 2. Proposed Enhancements

### Phase 1: Core Multiscale Primitives

#### 2.1 Graph Pooling/Unpooling (`components/graph_pooling.py`)

```python
class GraphPool(nn.Module):
    """Adaptive graph pooling using trainable projection (Graph U-Nets style)."""
    
    def __init__(self, k: int, feature_dim: int):
        # Trainable projection vector p
        # Select top-k nodes based on scalar projection
        # Store indices for unpooling
        
class GraphUnpool(nn.Module):
    """Restore graph structure using stored indices."""
    
    def forward(self, pooled_graph, indices, original_num_nodes):
        # Place nodes back at original positions
        # Fill unselected positions with zeros
```

**Features:**
- gPool: Trainable projection, top-k selection, gate operation
- gUnpool: Index-based restoration
- Graph power augmentation option (A^2)

#### 2.2 Hierarchical Graph Builder (`utils/hierarchy.py`)

```python
def build_hierarchical_graphs(graph: GraphsTuple, levels: int, pooling_ratio: float):
    """Build multi-level graph hierarchy for MGKN-style processing."""
    # Returns list of graphs at different resolutions
    # Level 0: Finest (original)
    # Level L: Coarsest
```

---


### Phase 2: Multiscale Processors

#### 2.3 Graph U-Net Processor (`components/graph_unet.py`)

```python
class GraphUNetProcessor(nn.Module):
    """Encoder-decoder graph processor with skip connections.
    
    Architecture:
        Input
          ↓
    [Encoder Block 1] ───────┐
          ↓                   │
    [Encoder Block 2] ───────┤
          ↓                   │ Skip Connections
    [Encoder Block 3] ───────┤
          ↓                   │
      [Bottleneck]           │
          ↓                   │
    [Decoder Block 3] ───────┘
          ↓
    [Decoder Block 2]
          ↓
    [Decoder Block 1]
          ↓
        Output
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_levels: int = 3,
        pool_ratio: float = 0.5,
        hidden_dim: int = 128,
        n_layers_per_level: int = 2,
    ):
        # Encoder: GCN + gPool blocks
        # Decoder: gUnpool + GCN blocks
        # Skip connections: feature concatenation or addition
```

**Key Features:**
- Configurable depth (n_levels)
- Adjustable pooling ratio per level
- Skip connection options (add/concat)
- Pre-norm residual connections

#### 2.4 MGKN Processor (`components/mgkn_processor.py`)

```python
class MGKNProcessor(nn.Module):
    """Multipole Graph Neural Operator with V-cycle.
    
    Implements the V-cycle algorithm from MGKN paper:
    - Downward pass: Fine → Coarse via inducing points
    - Upward pass: Coarse → Fine with skip connections
    - Multi-resolution kernel decomposition
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_levels: int = 3,
        nodes_per_level: List[int] = [400, 100, 25],
        kernel_width: int = 64,
    ):
        # Level-specific kernel networks (smaller at coarser levels)
        # Transition networks: Kl,l+1 (down) and Kl+1,l (up)
        # V-cycle iteration
```

**V-Cycle Algorithm:**
```python
def v_cycle(self, representations):
    # Downward pass
    for l in range(self.n_levels - 1):
        v_down[l+1] = transition_down(v_down[l])
    
    # Coarsest level processing
    v_up[-1] = kernel_coarse(v_down[-1])
    
    # Upward pass with skip connections
    for l in range(self.n_levels - 2, -1, -1):
        v_up[l] = kernel_fine(v_down[l]) + transition_up(v_up[l+1])
    
    return v_up[0]
```

---

### Phase 3: Enhanced FNO Components

#### 2.5 Multi-Resolution FNO (`components/spectral_multiscale.py`)

```python
class MultiResolutionFNOBlock(nn.Module):
    """FNO block with multi-band processing.
    
    Different FNO blocks for different frequency bands:
    - Low frequencies (large scales): Fewer modes, wider context
    - High frequencies (small scales): More modes, local detail
    """
    
    def __init__(
        self,
        width: int,
        modes_list: List[List[int]] = [[12, 12], [24, 24], [48, 48]],
        n_dim: int = 2,
    ):
        # Multiple FNO blocks at different frequency bands
        # Band-splitting and merging
```

#### 2.6 U-FNO Block (`components/spectral_multiscale.py`)

```python
class UFNOBlock(nn.Module):
    """U-Fourier block: Spectral conv + U-Net + skip connection.
    
    From U-FNO paper - combines:
    - K: Spectral convolution (global)
    - U: U-Net convolution (local high-freq details)
    - W: Pointwise linear (bias)
    """
    
    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int = 2,
        unet_depth: int = 2,
    ):
        # Spectral branch
        self.spectral_conv = SpectralConv(width, width, modes)
        
        # U-Net branch (local processing)
        self.unet = MiniUNet(width, width, depth=unet_depth, n_dim=n_dim)
        
        # Pointwise bias
        self.bias = _get_conv_nd(n_dim, width, width, kernel_size=1)
        
    def forward(self, x):
        x1 = self.spectral_conv(x)  # Global
        x2 = self.unet(x)            # Local
        x3 = self.bias(x)            # Bias
        return activation(x1 + x2 + x3)
```

---


### Phase 4: Complete Multiscale Models

#### 2.7 Multiscale FNO Model (`models/multiscale_fno.py`)

```python
class MultiscaleFNO(nn.Module):
    """Complete multiscale FNO with U-Net enhancement.
    
    Architecture options:
    1. Multi-band FNO: Different blocks for different frequencies
    2. U-FNO: FNO + U-Net hybrid layers
    3. Hierarchical FNO: Encoder-decoder in Fourier space
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_fno_layers: int = 4,
        n_ufno_layers: int = 3,
        n_dim: int = 2,
        use_super_resolution: bool = True,
    ):
        # Lifting
        # FNO blocks (pure spectral)
        # U-FNO blocks (spectral + local)
        # Projection
        # Super-resolution capability
```

**Super-Resolution Support:**
```python
def forward(self, x, target_resolution=None):
    # If target_resolution is different from input:
    # 1. Process at input resolution
    # 2. Interpolate Fourier modes to target resolution
    # 3. Return at target resolution
```

#### 2.8 Multiscale GraphNet (`models/multiscale_graphnet.py`)

```python
class MultiscaleGraphNet(nn.Module):
    """GraphNet with hierarchical multiscale processing.
    
    Supports multiple processor types:
    - 'unet': Graph U-Net encoder-decoder
    - 'mgkn': V-cycle multipole processor
    - 'multires': Multi-resolution with skip connections
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        latent_dim: int,
        output_dim: int,
        processor_type: str = 'unet',
        n_levels: int = 3,
        n_layers_per_level: int = 2,
    ):
        # Encoder: GraphNetEncoder
        # Processor: GraphUNetProcessor or MGKNProcessor
        # Decoder: MLPDecoder or ProbeDecoder
```

---

## 3. Implementation Roadmap

### Stage 1: Foundation (Week 1-2)
1. **Graph Pooling/Unpooling** (`components/graph_pooling.py`)
   - Implement gPool and gUnpool layers
   - Add graph power augmentation
   - Unit tests with Cora/Citeseer-style graphs

2. **Hierarchical Graph Builder** (`utils/hierarchy.py`)
   - Multi-level graph construction
   - Inducing point sampling
   - Transition matrix computation

### Stage 2: Processors (Week 3-4)
3. **Graph U-Net Processor** (`components/graph_unet.py`)
   - Encoder-decoder architecture
   - Skip connections
   - Integration with existing GraphNetBlock

4. **MGKN Processor** (`components/mgkn_processor.py`)
   - V-cycle algorithm
   - Level-specific kernels
   - Linear complexity verification

### Stage 3: Spectral Enhancements (Week 5-6)
5. **Multi-Resolution FNO** (`components/spectral_multiscale.py`)
   - Multi-band FNO blocks
   - U-FNO blocks
   - Mode interpolation for super-resolution

### Stage 4: Integration (Week 7-8)
6. **Complete Models** (`models/`)
   - MultiscaleFNO
   - MultiscaleGraphNet
   - Example configs and training scripts

7. **Documentation and Examples**
   - API documentation
   - Tutorial notebooks
   - Benchmark experiments

---

## 4. Key Design Decisions

### 4.1 Pooling Strategy
| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| gPool (trainable) | Adaptive, learnable | More parameters | General graphs |
| DiffPool | Differentiable assignment | Complex, expensive | Graph classification |
| Random sampling | Fast, simple | Not adaptive | Large-scale pre-training |
| Grid coarsening | Structured, efficient | Requires regular grid | FNO hierarchies |

**Decision:** Implement gPool (Graph U-Nets) as primary method, with grid coarsening for spectral components.

### 4.2 Skip Connection Strategy
| Type | Formula | Best For |
|------|---------|----------|
| Addition | `out = F(x) + x` | Same dimension, simple |
| Concatenation | `out = [F(x), x]` | Dimension expansion |
| Gated | `out = gate * F(x) + (1-gate) * x` | Adaptive mixing |

**Decision:** Support both addition (default) and concatenation with configurable option.

### 4.3 Frequency Band Strategy
| Strategy | Implementation | Best For |
|----------|----------------|----------|
| Truncation | Fixed k_max | Standard FNO |
| Multi-band | Parallel FNO blocks | Wide range of scales |
| Hierarchical | Coarse-to-fine | Sharp discontinuities |

**Decision:** Implement multi-band FNO and U-FNO for flexibility.

---

## 5. Testing Strategy

### 5.1 Unit Tests
- Pooling/Unpooling: Preserve connectivity, correct shapes
- V-cycle: Linear complexity verification
- Super-resolution: Error consistency across resolutions

### 5.2 Integration Tests
- Graph U-Net on node classification (Cora)
- MGKN on Darcy flow
- U-FNO on Burgers equation
- Multiscale FNO on Navier-Stokes

### 5.3 Benchmarks
| Problem | Baseline | Target Improvement |
|---------|----------|-------------------|
| Darcy Flow (FNO) | 0.0098 relative L2 | 0.0060 (U-FNO level) |
| Burgers (MGKN) | 0.0364 relative L2 | Maintain with linear complexity |
| Cora (Graph U-Net) | 81.5% accuracy | 84.4% accuracy |

---

## 6. API Preview

### Graph U-Net Usage
```python
from gnn_pde_v2.components import GraphUNetProcessor

processor = GraphUNetProcessor(
    latent_dim=128,
    n_levels=3,
    pool_ratio=0.5,
    skip_connection='add',  # or 'concat'
)

output_graph = processor(input_graph)
```

### U-FNO Usage
```python
from gnn_pde_v2.components import UFNOBlock

block = UFNOBlock(
    width=64,
    modes=[16, 16],
    n_dim=2,
    unet_depth=2,
)

output = block(input_tensor)  # [B, C, H, W]
```

### MGKN Usage
```python
from gnn_pde_v2.components import MGKNProcessor

processor = MGKNProcessor(
    latent_dim=64,
    n_levels=3,
    nodes_per_level=[400, 100, 25],
)

output_graph = processor(input_graph)
```

---

## 7. Summary

This enhancement plan addresses all three multiscale deficiencies identified in the code review:

1. **U-Net style skip connections** → GraphUNetProcessor, UFNOBlock
2. **Hierarchical graph pooling/unpooling** → GraphPool, GraphUnpool, MGKNProcessor
3. **Multi-resolution processing** → MultiResolutionFNO, super-resolution support

The implementation follows the framework's existing patterns (protocols, composability, clean separation) while adding powerful new capabilities for:
- Turbulence simulation (wide range of scales)
- Multi-physics problems (different physics at different scales)
- Large-domain simulations (computational efficiency via hierarchy)

**Estimated Effort:** 8 weeks
**Priority:** High (critical for competitive PDE solver framework)
