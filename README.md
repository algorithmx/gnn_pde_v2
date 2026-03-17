# GNN-PDE v2

A clean implementation of the Encode-Process-Decode architecture for PDE-GNNs with modular components and comprehensive research reproductions.

## Overview

**Architecture layering**: core → components → models

### Key Features

- **Modular EPD Architecture**: Clean separation of Encoder, Processor, Decoder components
- **Clean API Design**: Lean, composable core API with consistent patterns
- **Component-Based**: Reusable building blocks (MLP, GraphNetBlock, FNOBlock, etc.)
- **Pluggable Conditioning**: Protocol-based conditioning system (AdaLN, FiLM, DualAdaLN)
- **Multiple Attention Mechanisms**: Standard MHA, Physics Token Attention, QK-Norm, Sparse Attention
- **Temperature Mechanisms**: Adaptive, annealed, per-head temperature scaling
- **Research Reproductions**: 10+ paper implementations with exact equivalence
- **Auto-Registration**: Models can self-register for config-based instantiation
- **Flexible Dependencies**: Graceful fallbacks when optional dependencies unavailable

### Architecture Philosophy

The framework provides two usage patterns:

1. **Lean Core API** (recommended for research): Direct component usage
2. **Model Registry** (optional): Auto-registration for config-driven workflows

## Architecture

```
gnn_pde_v2/
├── core/                    # Minimal core primitives
│   ├── base.py              # BaseModel marker class
│   ├── graph.py             # GraphsTuple, batch_graphs, unbatch_graphs
│   ├── mlp.py               # MLP, SinActivation
│   ├── functional.py        # Scatter operations (torch_scatter fallback)
│   ├── aggregation.py       # Aggregation protocol (Sum, Mean, Max, Min)
│   ├── registry.py          # AutoRegisterModel, MODEL_REGISTRY
│   └── protocols.py        # Structural protocols (GraphProcessor, etc.)
├── components/              # Reusable building blocks
│   ├── layers.py            # Residual, GatedResidual, make_residual
│   ├── encoders.py          # (legacy, use models.gnn_model.MeshEncoder)
│   ├── processors.py        # GraphNetBlock, MessagePassingBlock, GENBlock
│   ├── gcn.py               # GCNBlock, GCNBlockWithEdgeFeatures
│   ├── transformer.py       # TransformerBlock, TransformerProcessor
│   ├── attention.py         # MultiHeadAttention, PhysicsTokenAttention, etc.
│   ├── conditioning.py      # AdaLN, DualAdaLN, FiLM, ZeroConditioning
│   ├── temperature.py       # Temperature mechanisms
│   ├── spectral.py         # SpectralConv, FNOBlock, AFNOBlock, FNOProcessor
│   ├── decoders.py         # MLPDecoder, IndependentMLPDecoder
│   ├── probe.py             # ProbeDecoder, WindFarmGNO, ProbeGraphBuilder
│   ├── rbf.py              # LearnableRBFEncoder, GaussianRBFEncoder
│   └── fourier_encoder.py  # FourierFeatureEncoder
├── models/                  # Complete model implementations
│   ├── encode_process_decode.py  # Clean EPD model
│   ├── gnn_model.py             # GraphNet, MeshGraphNet, MeshEncoder
│   ├── fno_model.py             # FNO, TFNO, AFNO (lazy loaded)
│   └── multiscale_fno.py        # MultiscaleFNO (lazy loaded)
├── examples/                 # Research paper reproductions
│   ├── example_meshgraphnets.py     # MeshGraphNets (ICML 2021)
│   ├── example_deepxde.py           # DeepXDE (SIAM Review 2021)
│   ├── example_neuraloperator_fno.py # NeuralOperator FNO (ICLR 2021)
│   ├── example_transolver.py         # Transolver (ICML 2024)
│   ├── example_transolver_v3.py      # Transolver-3 - 160M+ cell (2026)
│   ├── example_unisolver.py          # Unisolver (ICML 2024)
│   ├── example_windfarm_gno.py       # WindFarm GNO (2025)
│   ├── example_graph_pde_gno.py      # Graph-PDE GNO (2020)
│   ├── example_mgkn.py               # MGKN (Mesh Graph Networks)
│   ├── example_low_width_graph_transformer.py  # Low-width transformer
│   ├── example_qk_norm.py            # QK-Norm attention
│   ├── example_relative_position_attention.py   # Relative position
│   ├── example_graph_unets.py        # Graph U-Nets
│   ├── example_graph_unets_framework.py  # Graph U-Nets framework
│   ├── example_ufno.py               # UFNO (U-shaped FNO)
│   └── training_utils.py             # Training utilities
├── utils/                   # Utility functions
│   ├── graph_utils.py      # knn_graph, radius_graph, compute_edge_features
│   └── spatial_utils.py    # grid_to_points, points_to_grid
└── tests/                   # Comprehensive test suite
    ├── test_core.py
    ├── test_components.py
    └── test_examples.py
```

## Quick Start

### Lean Core API (Recommended)

```python
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.core import MLP
from gnn_pde_v2.components import GraphNetProcessor, MLPDecoder
import torch

# Build model from components
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128])
        self.edge_encoder = MLP(in_dim=3, out_dim=128, hidden_dims=[128])
        self.processor = GraphNetProcessor(
            latent_dim=128,
            n_layers=4,
        )
        self.decoder = MLPDecoder(
            latent_dim=128,
            out_dim=2,
            hidden_dims=[128, 64],
        )
    
    def forward(self, graph):
        latent = graph.replace(
            nodes=self.node_encoder(graph.nodes),
            edges=self.edge_encoder(graph.edges),
        )
        processed = self.processor(latent)
        return self.decoder(processed)

# Create graph
graph = GraphsTuple(
    nodes=torch.randn(10, 5),
    edges=torch.randn(20, 3),
    receivers=torch.randint(0, 10, (20,)),
    senders=torch.randint(0, 10, (20,)),
    n_node=torch.tensor([10]),
    n_edge=torch.tensor([20]),
)

# Forward pass
model = MyModel()
output = model(graph)  # [10, 2]
```

## Core Layer (`gnn_pde_v2.core`)

### Data Structures
| Component | Description |
|-----------|-------------|
| `GraphsTuple` | Minimal graph representation (nodes, edges, receivers, senders, globals) |
| `batch_graphs()` | Batch multiple graphs into single GraphsTuple |
| `unbatch_graphs()` | Unbatch GraphsTuple into list of individual graphs |

### Base Classes
| Component | Description |
|-----------|-------------|
| `BaseModel` | Minimal marker class (no magic, no registry) |
| `AutoRegisterModel` | Auto-registers subclasses in MODEL_REGISTRY |
| `MODEL_REGISTRY` | Standalone registry with create(), register(), list_models() |

### MLP & Activations
| Component | Description |
|-----------|-------------|
| `MLP` | Flexible dense or pointwise-conv feedforward stack |
| `SinActivation` | SIREN-style sine activation |

### Functional Operations
| Function | Description |
|----------|-------------|
| `scatter_sum` | Sum aggregation (torch_scatter or pure PyTorch fallback) |
| `scatter_mean` | Mean aggregation |
| `scatter_max` | Max aggregation |
| `scatter_min` | Min aggregation |
| `scatter_softmax` | Softmax within groups |
| `aggregate_edges` | Aggregate edge features to receiver nodes |
| `broadcast_nodes_to_edges` | Broadcast node features to edges |
| `broadcast_global` | Broadcast per-graph globals to nodes |
| `aggregate_to_global` | Pool node features to graph-level |

### Aggregation Protocol
| Component | Description |
|-----------|-------------|
| `Aggregation` | Abstract base for aggregations |
| `Sum` | Sum aggregation |
| `Mean` | Mean aggregation |
| `Max` | Max aggregation |
| `Min` | Min aggregation |
| `get_aggregation()` | Factory function |

### Protocols
| Protocol | Description |
|----------|-------------|
| `GraphEncoder` | Protocol for encoding modules (GraphsTuple → GraphsTuple) |
| `GraphProcessor` | Protocol for processor modules (GraphsTuple → GraphsTuple) |
| `NodeDecoder` | Protocol for node decoders |
| `QueryDecoder` | Protocol for query-based decoders |
| `Decoder` | Union of NodeDecoder and QueryDecoder |
| `GridProcessor` | Protocol for grid processors (Tensor → Tensor) |
| `GridModel` | Protocol for grid-to-grid models |
| `PositionEncoder` | Protocol for position encoders |
| `ConditioningProtocol` | ABC for conditioning mechanisms |
| `Modulation` | Dataclass for modulation parameters (shift, scale, gate, cross_kv) |

## Components Layer (`gnn_pde_v2.components`)

### Residual Connections

| Component | Description |
|-----------|-------------|
| `Residual` | Simple residual: `x + f(x)` with optional norm/scale |
| `GatedResidual` | Gated residual: `(1-g)*x + g*f(x)` |
| `make_residual()` | Factory for runtime selection |

### Processors (Graph-based)

| Component | Description |
|-----------|-------------|
| `MessagePassingBlock` | Abstract base for graph message passing |
| `GraphNetBlock` | DeepMind-style node/edge update |
| `EdgeConditionedConvBlock` | Edge-conditioned convolution |
| `EdgeConvBlock` | Edge convolution (PointNet-style) |
| `GENBlock` | Graph Edges Networks block |
| `GlobalGraphNetBlock` | Full encoder-processor-decoder with globals |
| `GlobalGraphNetProcessor` | Multi-layer global GNN processor |
| `GraphNetProcessor` | Multi-layer GraphNet processor |
| `GCNBlock` | Graph Convolutional Network block |
| `GCNBlockWithEdgeFeatures` | GCN with edge features |
| `TransformerBlock` | Transformer block with optional conditioning |
| `TransformerProcessor` | Multi-layer transformer for nodes |

### Attention Mechanisms

| Component | Description |
|-----------|-------------|
| `MultiHeadAttention` | Standard multi-head self-attention |
| `PhysicsTokenAttention` | Transolver-style slice-attention-deslice (O(G²) vs O(N²)) |
| `PhysicsTokenAttentionV3` | Transolver-3 with tiling for 160M+ cell meshes |
| `QKNormMultiHeadAttention` | Attention with QK normalization |
| `SparseGraphAttention` | Sparse attention for graphs |
| `RelativePositionEncoding` | Relative position encoding |
| `TiledSliceOperation` | Tiling operation for large graphs |

### Conditioning System

| Component | Description |
|-----------|-------------|
| `ZeroConditioning` | Identity passthrough (no modulation) |
| `AdaLNConditioning` | Single-source Adaptive Layer Norm |
| `AdaLNConditioningNoGate` | AdaLN without gate |
| `DualAdaLNConditioning` | Dual AdaLN (μ + f embeddings) |
| `DualAdaLNConditioningNoGate` | Dual AdaLN without gate |
| `FiLMConditioning` | Feature-wise Linear Modulation |
| `apply_modulation()` | Apply modulation to hidden states |

### Temperature Mechanisms

| Component | Description |
|-----------|-------------|
| `TemperatureBase` | Abstract base for temperature |
| `FixedTemperature` | Fixed temperature scaling |
| `LearnableScalarTemperature` | Learnable scalar temperature |
| `PerHeadTemperature` | Per-head temperature |
| `AdaptiveTemperature` | Adaptive temperature |
| `AnnealedTemperature` | Annealed temperature |
| `create_temperature_module()` | Factory function |

### Spectral (Grid-based)

| Component | Description |
|-----------|-------------|
| `SpectralConv` | Standard spectral convolution |
| `SeparableSpectralConv` | Factorized spectral convolution |
| `SpectralConvBase` | Base class for spectral convs |
| `FNOBlock` | FNO block |
| `AFNOBlock` | Adaptive FNO block |
| `FNOMLPBlock` | FNO with MLP |
| `FNOProcessor` | Complete FNO processor |
| `make_spectral_conv()` | Factory for standard vs separable |

### Decoders

| Component | Description |
|-----------|-------------|
| `MLPDecoder` | MLP-based node decoder |
| `IndependentMLPDecoder` | Independent MLPs per output |
| `ProbeDecoder` | Query-based decoder for arbitrary positions |
| `WindFarmGNO` | Wind farm-specific GNO |
| `ProbeGraphBuilder` | Graph builder for probe decoder |

### Encoders & Feature Engineering

| Component | Description |
|-----------|-------------|
| `FourierFeatureEncoder` | Random Fourier feature lifting |
| `LearnableRBFEncoder` | Learnable RBF encoder with cosine cutoff |
| `GaussianRBFEncoder` | Fixed Gaussian RBF encoder |

## Models Layer (`gnn_pde_v2.models`)

### Lazy Loading

Models use lazy loading - some require additional dependencies:

```python
from gnn_pde_v2.models import (
    EncodeProcessDecode,  # Always available
    FNO, TFNO, AFNO,      # Lazy loaded
    GraphNet, MeshGraphNet,  # Lazy loaded
)
```

### Registered Models

| Name | Aliases | Type | Description |
|------|---------|------|-------------|
| `graphnet` | gnn, graph_net | Graph | Standard GNN |
| `meshgraphnet` | mgn, mesh_graph_net | Graph | MeshGraphNets-style |
| `fno` | fourier_no, fno2d | Grid | Fourier Neural Operator |
| `tfno` | tensorized_fno | Grid | Tensorized FNO |
| `afno` | adaptive_fno | Grid | Adaptive FNO |

### Using the Registry

```python
from gnn_pde_v2.core import MODEL_REGISTRY, AutoRegisterModel

# Create by name (for config-driven workflows)
model = MODEL_REGISTRY.create('graphnet', node_in_dim=11, edge_in_dim=3, out_dim=3)

# Or use AutoRegisterModel directly
model = AutoRegisterModel.create('fno', in_channels=1, out_channels=1, width=64)
```

### Custom Model Registration

```python
from gnn_pde_v2.core import AutoRegisterModel

class MyModel(AutoRegisterModel, name='my_model', aliases=['mymodel']):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Linear(hidden_dim, hidden_dim)

# Create by name
model = AutoRegisterModel.create('my_model', hidden_dim=256)
```

## Research Reproductions

The framework includes exact reproductions of 10+ major PDE-GNN papers:

| Paper | Model | Key Innovation |
|-------|-------|----------------|
| MeshGraphNets (ICML 2021) | `meshgraphnet` | Unstructured mesh simulation |
| DeepXDE (SIAM 2021) | `deepxde` | Physics-informed neural networks |
| NeuralOperator FNO (ICLR 2021) | `fno` | Fourier neural operators |
| Transolver (ICML 2024) | `transolver` | Physics-attention mechanism |
| Transolver-3 (2026) | `transolver_v3` | 160M+ cell industrial scale |
| Unisolver (ICML 2024) | `unisolver` | PDE-conditional transformers |
| WindFarm GNO (2025) | `windfarm_gno` | Two-stage graph operator |
| Graph-PDE GNO (2020) | `graph_pde_gno` | Edge-conditioned convolution |
| MGKN | `mgkn` | Mesh Graph Networks |
| UFNO | `ufno` | U-shaped FNO |

## Testing

```bash
# Run all tests
pytest gnn_pde_v2/tests/

# Run specific test file
pytest gnn_pde_v2/tests/test_core.py

# Run with coverage
pytest gnn_pde_v2/tests/ --cov=gnn_pde_v2

# Run example reproduction tests
pytest gnn_pde_v2/tests/test_examples.py
```

### Test Coverage

- **Core**: Graph processing, BaseModel, functional operations, aggregation
- **Components**: All encoders, processors, decoders, attention, conditioning
- **Registry**: Auto-registration for model discovery
- **Examples**: Research paper reproduction accuracy

## Installation & Dependencies

### Core Dependencies (Required)

```bash
pip install torch numpy
```

### Optional Dependencies

```bash
# For faster scatter operations (recommended for large graphs)
pip install torch-scatter

# For graph construction (knn_graph, radius_graph)
pip install torch-cluster

# For testing
pip install pytest pytest-cov
```

The framework gracefully handles missing optional dependencies:
- `torch_scatter`: Falls back to pure PyTorch implementation
- `torch_cluster`: Required for `knn_graph` and `radius_graph`

## Design Principles

1. **Composability**: Small, chainable components that work together
2. **Clean API**: Lean, composable API with consistent patterns  
3. **Component over Framework**: Focus on reusable building blocks
4. **Reproducibility**: Exact paper implementations with equivalence guarantees
5. **Graceful Degradation**: Optional dependencies with clear fallbacks
6. **Testing First**: Comprehensive test coverage for all components

---

## Global Features: Design Philosophy

This framework carries `globals` as a first-class field on `GraphsTuple`, following the [DeepMind Graph Nets](https://arxiv.org/abs/1806.01261) design. This is a deliberate choice absent from most graph ML libraries (including PyTorch Geometric) and deserves explicit explanation.

### What globals are

`globals` is a tensor of shape `[batch_size, global_feat_dim]` — one vector per graph in the batch. It lives at a level above nodes and edges and represents information that belongs to the *whole system*, not to any individual node or edge.

### Why globals matter for PDEs

In PDE solving, global features naturally encode quantities that govern the entire domain:

- **PDE parameters**: Reynolds number, viscosity, diffusion coefficient, forcing frequency — these condition the solution everywhere, not at one mesh node.
- **Boundary/initial conditions summary**: a compressed representation of the full boundary state that every node should be aware of.
- **Time step or simulation time**: when solving time-dependent PDEs, the current time `t` is a scalar that affects every node equally.
- **Geometry metadata**: domain area, characteristic length scale, or any per-simulation invariant.

Without globals, you would have to redundantly copy this information into every node's feature vector before encoding — which is what most PyG-style models do. The globals slot instead transmits it cleanly and symmetrically through the message passing stack.

### How globals flow through `GlobalGraphNetBlock`

Each message passing step performs three updates in order:

```
1. Edge update:   new_e_ij = MLP([node_i, node_j, e_ij, g])
2. Node update:   new_v_i  = MLP([v_i, Σ_j new_e_ij, g])
3. Global update: new_g    = MLP([mean(new_v), mean(new_e), g])
```

- **Steps 1 & 2**: globals are *broadcast* to every edge/node (via `broadcast_global`) and concatenated into the MLP input, so every computation is conditioned on the system-level state.
- **Step 3**: the global vector is *updated* by aggregating (via `aggregate_to_global`) the new node and edge features back up, so it can accumulate latent information about the whole graph as processing proceeds.

This creates a bidirectional information flow: globals conditioning local computation downward, and local state aggregating back upward — a complete information loop unavailable in node/edge-only frameworks.

### When to use globals

| Use globals for | Keep in nodes/edges |
|---|---|
| PDE coefficients (Re, ν, κ) | Local field values (velocity, pressure at a node) |
| Simulation time `t` | Edge geometry (displacement, distance) |
| Boundary condition summary | Per-node boundary flags |
| Geometry-level invariants | Per-node coordinates |
| Cross-graph conditioning | Local connectivity |

### When to leave globals as `None`

If all conditioning information is already encoded per-node (e.g., the PDE parameters have been appended to each node's input features), use `GraphNetBlock` / `GraphNetProcessor` directly — they carry no global machinery at all, so there is zero overhead and no dead code paths.

---

## Usage Patterns

### Research Development (Lean API)

```python
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.core import MLP
from gnn_pde_v2.components import GraphNetBlock, GlobalGraphNetBlock, Residual

# Node/edge-only (no global state)
encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128])
processor = Residual(GraphNetBlock(latent_dim=128))
decoder = MLP(in_dim=128, out_dim=2, hidden_dims=[64])

# With global state (PDE parameters, time, BCs)
processor_g = Residual(GlobalGraphNetBlock(latent_dim=128, global_latent_dim=32))
```

### Model Registry

```python
from gnn_pde_v2.core import AutoRegisterModel

# Register models for config-driven instantiation
class MyModel(AutoRegisterModel, name='my_model'):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Linear(hidden_dim, hidden_dim)

# Create by name
model = AutoRegisterModel.create('my_model', hidden_dim=256)
```

### Factory Functions

Factory functions are provided **only for runtime polymorphism** where the component type is determined at runtime (e.g., from configuration). For explicit construction, prefer direct class instantiation.

```python
from gnn_pde_v2.components import make_residual, make_spectral_conv
from gnn_pde_v2.components import Residual, GatedResidual, SpectralConv

# Use factories for runtime selection (config-driven)
residual_block = make_residual(module, residual_type=config.residual_type)  # 'add', 'gated', etc.
conv = make_spectral_conv(64, 64, modes, separable=config.use_separable)

# Use direct classes for explicit construction (preferred)
residual_block = Residual(module)
residual_block = GatedResidual(module, gate_bias=2.0)
conv = SpectralConv(64, 64, [16, 16])
```

### Paper Reproduction

```python
from gnn_pde_v2.examples.example_meshgraphnets import MeshGraphNets

# Exact reproduction with paper-equivalent behavior
model = MeshGraphNets(
    node_input_size=11,
    edge_input_size=3, 
    output_size=3,
    n_layers=15
)
```

## Extension Points

The framework supports several extension mechanisms:

1. **Component Extension**: Inherit from existing components
2. **Model Registration**: Use `AutoRegisterModel` for config-based instantiation
3. **Custom Conditioning**: Implement `ConditioningProtocol` for new modulation schemes
4. **Custom Aggregation**: Implement `Aggregation` protocol for new reduction methods
5. **Paper Reproduction**: Follow examples pattern for new papers

### Model Hierarchy

```
nn.Module
    └── BaseModel (core.base)           # Minimal marker class, no magic
            └── AutoRegisterModel       # Adds registry, create(), list_models()
                    └── YourModel       # Custom models with auto-registration
```

See `examples/` directory for detailed implementation patterns used in research reproductions.