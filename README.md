# GNN-PDE v2

A clean implementation of the Encode-Process-Decode architecture for PDE-GNNs with modular components and comprehensive research reproductions.

## Overview

**Architecture layering**: core → components → models

### Key Features

- **Modular EPD Architecture**: Clean separation of Encoder, Processor, Decoder components
- **Clean API Design**: Lean, composable core API with consistent patterns
- **Component-Based**: Reusable building blocks (MLP, GraphNetBlock, FNOBlock, etc.)
- **Pluggable Conditioning**: Protocol-based conditioning system (AdaLN, FiLM, DualAdaLN) for transformers
- **Research Reproductions**: 7+ paper implementations with exact equivalence
- **Optional Auto-Registration**: Convenience models can self-register for config-based instantiation
- **Flexible Dependencies**: Graceful fallbacks when optional dependencies unavailable
- **Comprehensive Testing**: Full test suite with pytest

### Architecture Philosophy

The framework provides two usage patterns:

1. **Lean Core API** (recommended for research): Direct component usage
2. **Model Registry** (optional): Auto-registration for config-driven workflows

## Architecture

```
gnn_pde_v2/
├── core/                    # Minimal core primitives
│   ├── graph.py            # GraphsTuple, batching utilities
│   ├── base.py             # Lean BaseModel marker class
│   └── functional.py       # Scatter operations, aggregation
├── components/              # Reusable building blocks (canonical lean API)
│   ├── encoders.py         # MLP and encoder helpers
│   ├── processors.py       # GraphNetBlock and GraphNetProcessor
│   ├── decoders.py         # MLPDecoder variants
│   ├── layers.py           # Residual layers
│   ├── fno.py              # FNO components (SpectralConv, etc.)
│   ├── transformer.py      # Transformer components
│   └── probe.py            # Probe-based components
├── models/                  # Complete model implementations
│   ├── encode_process_decode.py  # Clean EPD model
│   ├── gnn_model.py             # GNN models (GraphNet, MeshGraphNet)
│   └── fno_model.py             # FNO models (FNO, TFNO)
├── examples/                 # Research paper reproductions
│   ├── example_meshgraphnets.py     # MeshGraphNets (ICML 2021)
│   ├── example_deepxde.py           # DeepXDE (SIAM Review 2021)
│   ├── example_neuraloperator_fno.py # NeuralOperator FNO (ICLR 2021)
│   ├── example_transolver.py         # Transolver (ICML 2024)
│   ├── example_unisolver.py          # Unisolver (ICML 2024)
│   ├── example_windfarm_gno.py       # WindFarm GNO (2025)
│   └── example_graph_pde_gno.py      # Graph-PDE GNO (2020)
├── utils/                   # Utility functions
│   ├── graph_utils.py      # Graph processing utilities
│   └── spatial_utils.py    # Spatial computations
└── tests/                   # Comprehensive test suite
    ├── test_core.py        # Core functionality tests
    ├── test_components.py  # Component tests
    └── test_examples.py    # Example reproduction tests
```

`MLP` supports:
- arbitrary depth via `hidden_dims`
- hidden vs final normalization separately (`norm=` vs `final_norm=`)
- dense or pointwise-conv stacks via `linear_factory`

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
        self.node_encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128], use_layer_norm=False)
        self.edge_encoder = MLP(in_dim=3, out_dim=128, hidden_dims=[128], use_layer_norm=False)
        self.processor = GraphNetProcessor(
            node_dim=128,
            edge_dim=128,
            n_layers=4,
        )
        self.decoder = MLPDecoder(node_dim=128, out_dim=2, hidden_dims=[128, 64])
    
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

## Available Components

### Core Building Blocks

| Component | Description | Usage |
|-----------|-------------|-------|
| `MLP` | Flexible dense or pointwise-conv feedforward stack with pre-activation option | Encoder/Decoder |
| `BaseModel` | Minimal marker class for model hierarchy (no magic, no registry) | Base class |
| `SinActivation` | Sine activation for SIREN-style networks | Activation |
| `FourierFeatureEncoder` | Random Fourier feature lifting | Encoder |
| `GraphNetBlock` | Graph message passing block | Processor |
| `TransformerBlock` | Multi-head attention block with optional conditioning | Processor |
| `FNOBlock` | Fourier neural operator block | Processor |
| `Residual` | Residual connection wrapper | Any layer |
| `ProbeDecoder` | Arbitrary query point decoder | Decoder |

### Functional Operations (core.functional)

| Function | Description |
|----------|-------------|
| `scatter_sum` | Sum aggregation (scatter_add) |
| `scatter_mean` | Mean aggregation |
| `scatter_max` | Max aggregation |
| `aggregate_edges` | Aggregate edge features to receiver nodes |
| `broadcast_nodes_to_edges` | Broadcast node features to edges |

### Encoders (components.encoders)

| Component | Description |
|-----------|-------------|
| `MLPEncoder` | Simple graph encoder with separate node and optional edge MLPs |
| `MLPMeshEncoder` | MeshGraphNets-style encoder with node, edge, and optional global MLPs |
| `make_mlp_encoder()` | Factory function for creating MLP encoders |

### Decoders (components.decoders)

| Component | Description |
|-----------|-------------|
| `MLPDecoder` | Simple MLP decoder operating on node features |
| `IndependentMLPDecoder` | Separate MLPs for each output component (multi-task) |
| `ProbeDecoder` | Arbitrary query point decoder for continuous fields |

### Residual Connections (components.layers)

| Component | Description |
|-----------|-------------|
| `Residual` | Basic residual wrapper: `x + f(x)` |
| `ResidualBlock` | Configurable wrapper with 'add', 'scaled', or 'none' types |
| `GatedResidual` | Learnable gate: `(1-g)*x + g*f(x)` (Highway-style) |
| `PreNormResidual` | Pre-normalization block (Transformer Pre-LN style) |
| `ResidualSequence` | Sequence of residual blocks with consistent interface |
| `SkipConnection` | Flexible skip with optional projection for dimension changes |
| `make_residual()` | Factory function to create residual wrapper by type |

### Attention & Transformer Components (components.transformer)

| Component | Description |
|-----------|-------------|
| `MultiHeadAttention` | Standard multi-head self-attention |
| `TransformerBlock` | Full transformer block with attention, MLP, and optional conditioning |
| `PhysicsTokenAttention` | Transolver-style slice-attention-deslice (O(G²) vs O(N²)) |
| `TransformerProcessor` | Multi-layer transformer processor for graph nodes |

### Conditioning System (components.transformer)

The framework provides a pluggable conditioning protocol for transformer-based models:

| Component | Description |
|-----------|-------------|
| `Modulation` | Dataclass: shift, scale, gate, cross_kv parameters for attention modulation |
| `ConditioningProtocol` | ABC for implementing custom conditioning schemes |
| `ZeroConditioning` | Identity passthrough (no modulation) |
| `AdaLNConditioning` | Single-source Adaptive Layer Normalization |
| `DualAdaLNConditioning` | UniSolver-style dual conditioning (μ + f embeddings) |
| `FiLMConditioning` | Feature-wise Linear Modulation (γ, β) |

```python
from gnn_pde_v2.components import (
    Modulation, ConditioningProtocol, AdaLNConditioning, FiLMConditioning
)

# Use AdaLN conditioning with transformer
conditioner = AdaLNConditioning(cond_dim=64, out_dim=128)
block = TransformerBlock(dim=128, conditioner=conditioner)

# Or implement custom conditioning
class MyConditioning(ConditioningProtocol):
    def forward(self, cond: torch.Tensor) -> Modulation:
        # Custom logic to produce modulation parameters
        return Modulation(shift=..., scale=..., gate=..., cross_kv=...)
```

### FNO Components (components.fno)

| Component | Description |
|-----------|-------------|
| `SpectralConv` | Spectral convolution with learnable complex weights in Fourier space |
| `FNOBlock` | Standard Fourier Neural Operator block |
| `AFNOBlock` | Adaptive FNO with block-diagonal weights and soft-thresholding |
| `FNOProcessor` | Complete FNO processor with lifting, blocks, and projection |

### Probe Components (components.probe)

| Component | Description |
|-----------|-------------|
| `ProbeMessagePassingLayer` | Single message passing layer for probe graphs |
| `ProbeDecoder` | Decoder for arbitrary query points |

### Processors

| Type | Component | Key Features |
|------|-----------|--------------|
| **Graph-based** | `GraphNetBlock` | Message passing, edge updates |
| **Graph Processor** | `GraphNetProcessor` | Multiple GraphNet blocks in sequence |
| **Attention** | `TransformerBlock` | Multi-head attention, physics tokens, conditioning |
| **Transformer Processor** | `TransformerProcessor` | Multi-layer transformer for nodes |
| **Spectral** | `FNOBlock` | FFT-based global convolution |
| **Spectral Processor** | `FNOProcessor` | Complete FNO with lifting/projection |

### Research Reproductions

The framework includes exact reproductions of 7+ major PDE-GNN papers:

| Paper | Model | Key Innovation |
|-------|-------|----------------|
| MeshGraphNets (ICML 2021) | `meshgraphnets` | Unstructured mesh simulation |
| DeepXDE (SIAM 2021) | `deepxde` | Physics-informed neural networks |
| NeuralOperator FNO (ICLR 2021) | `fno` | Fourier neural operators |
| Transolver (ICML 2024) | `transolver` | Physics-attention mechanism |
| Unisolver (ICML 2024) | `unisolver` | PDE-conditional transformers |
| WindFarm GNO (2025) | `windfarm_gno` | Two-stage graph operator |
| Graph-PDE GNO (2020) | `graph_pde_gno` | Edge-conditioned convolution |

See `examples/README.md` for detailed implementation notes.

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

- **Core**: Graph processing, BaseModel, functional operations
- **Components**: All encoders, processors, decoders
- **Registry**: Auto-registration for model discovery
- **Examples**: Research paper reproduction accuracy

## Installation & Dependencies

### Core Dependencies (Required)

```bash
pip install torch numpy
```

### Optional Dependencies

```bash
# For config-based experiments (optional)
pip install pydantic

# For testing
pip install pytest pytest-cov

# For advanced examples
pip install torch-scatter torch-sparse
```

The framework gracefully handles missing optional dependencies - features are disabled with clear error messages.

## Design Principles

1. **Composability**: Small, chainable components that work together
2. **Clean API**: Lean, composable API with consistent patterns  
3. **Component over Framework**: Focus on reusable building blocks
4. **Reproducibility**: Exact paper implementations with equivalence guarantees
5. **Graceful Degradation**: Optional dependencies with clear fallbacks
6. **Testing First**: Comprehensive test coverage for all components

## Usage Patterns

### Research Development (Lean API)

```python
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.core import MLP
from gnn_pde_v2.components import GraphNetBlock, Residual

# Direct component composition for maximum flexibility
encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128], use_layer_norm=False)
processor = Residual(GraphNetBlock(128, 128))
decoder = MLP(in_dim=128, out_dim=2, hidden_dims=[64], use_layer_norm=False)
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
2. **Model Registration**: Use `AutoRegisterModel` (inherits from `BaseModel`) for config-based instantiation
3. **Custom Conditioning**: Implement `ConditioningProtocol` for new modulation schemes
4. **Configuration**: Extend Pydantic configs for new parameters
5. **Paper Reproduction**: Follow examples pattern for new papers

### Model Hierarchy

```
nn.Module
    └── BaseModel (core.base)           # Minimal marker class, no magic
            └── AutoRegisterModel       # Adds registry, create(), list_models()
                    └── YourModel       # Custom models with auto-registration
```

See `examples/README.md` for detailed extension guidelines and design patterns used in research reproductions.
