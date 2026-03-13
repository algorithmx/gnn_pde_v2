# GNN-PDE v2

A clean implementation of the Encode-Process-Decode architecture for PDE-GNNs with modular components and comprehensive research reproductions.

## Overview

**Total Implementation**: ~11,000 LOC (core + examples + tests) across 63 Python files

### Key Features

- **Modular EPD Architecture**: Clean separation of Encoder, Processor, Decoder components
- **Dual API Design**: Lean core API + optional convenient high-level API
- **Component-Based**: Reusable building blocks (MLP, GraphNetBlock, FNOBlock, etc.)
- **Research Reproductions**: 7+ paper implementations with exact equivalence
- **Optional Auto-Registration**: Convenience models can self-register for config-based instantiation
- **Flexible Dependencies**: Graceful fallbacks when optional dependencies unavailable
- **Comprehensive Testing**: Full test suite with pytest

### Architecture Philosophy

The framework provides two usage patterns:

1. **Lean Core API** (recommended for research): Direct component usage
2. **Convenient API** (optional): Configuration-based with auto-registration

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
├── convenient/              # High-level API (optional)
│   ├── registry.py         # Auto-registration base class
│   ├── config.py           # Pydantic configurations
│   ├── builder.py          # Configuration-based model building
│   ├── training.py         # Unified training wrapper
│   ├── aggregation.py      # Extended aggregation functions
│   └── initializers.py     # Weight initialization utilities
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
    ├── test_convenient.py  # High-level API tests
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

### Convenient API (Optional)

```python
from gnn_pde_v2.convenient import GNNConfig, ConfigBuilder, TrainingConfig

# Define config
config = GNNConfig(
    model_type='graphnet',
    node_in_dim=5,
    edge_in_dim=3,
    out_dim=2,
    hidden_dim=128,
    n_message_passing=4,
)

# Build and train model
builder = ConfigBuilder(config)
model = builder.build_unified_model(TrainingConfig())

# Training step
metrics = model.train_step((graph, target))
```

## Available Components

### Core Building Blocks

| Component | Description | Usage |
|-----------|-------------|-------|
| `MLP` | Flexible dense or pointwise-conv feedforward stack | Encoder/Decoder |
| `FourierFeatureEncoder` | Random Fourier feature lifting | Encoder |
| `GraphNetBlock` | Graph message passing block | Processor |
| `TransformerBlock` | Multi-head attention block | Processor |
| `FNOBlock` | Fourier neural operator block | Processor |
| `Residual` | Residual connection wrapper | Any layer |
| `ProbeDecoder` | Arbitrary query point decoder | Decoder |

### Processors

| Type | Component | Key Features |
|------|-----------|--------------|
| **Graph-based** | `GraphNetBlock` | Message passing, edge updates |
| **Attention** | `TransformerBlock` | Multi-head attention, physics tokens |
| **Spectral** | `FNOBlock` | FFT-based global convolution |
| **Hybrid** | `GraphNetProcessor` | Multiple GraphNet blocks |

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
- **Convenient**: Configuration, registry, training API
- **Examples**: Research paper reproduction accuracy

## Installation & Dependencies

### Core Dependencies (Required)

```bash
pip install torch numpy
```

### Optional Dependencies

```bash
# For convenient API (configuration, training)
pip install pydantic

# For testing
pip install pytest pytest-cov

# For advanced examples
pip install torch-scatter torch-sparse
```

The framework gracefully handles missing optional dependencies - features are disabled with clear error messages.

## Design Principles

1. **Composability**: Small, chainable components that work together
2. **Dual API**: Lean core for research + convenient API for experimentation  
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

### Rapid Experimentation (Convenient API)

```python
from gnn_pde_v2.convenient import GNNConfig, ConfigBuilder

# Configuration-based model building
config = GNNConfig(model_type='graphnet', hidden_dim=128)
model = ConfigBuilder(config).build_unified_model()
```

### Paper Reproduction

```python
from gnn_pde_v2.examples.example_meshgraphnets import MeshGraphNets

# Exact reproduction with paper-equivalent behavior
model = MeshGraphNets(
    node_input_size=11,
    edge_input_size=3, 
    output_size=3,
    message_passing_steps=15
)
```

## Extension Points

The framework supports several extension mechanisms:

1. **Component Extension**: Inherit from existing components
2. **Model Registration**: Use `BaseModel` for auto-registration
3. **Configuration**: Extend Pydantic configs for new parameters
4. **Paper Reproduction**: Follow examples pattern for new papers

See `examples/README.md` for detailed extension guidelines and design patterns used in research reproductions.
