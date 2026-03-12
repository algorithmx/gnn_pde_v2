# GNN-PDE v2

A clean implementation of the Encode-Process-Decode architecture for PDE-GNNs with modular components and comprehensive research reproductions.

## Overview

**Total Implementation**: ~11,000 LOC (core + examples + tests) across 63 Python files

### Key Features

- **Modular EPD Architecture**: Clean separation of Encoder, Processor, Decoder components
- **Dual API Design**: Lean core API + optional convenient high-level API
- **Component-Based**: Reusable building blocks (MLP, GraphNetBlock, FNOBlock, etc.)
- **Research Reproductions**: 7+ paper implementations with exact equivalence
- **Auto-Registration**: Models self-register for easy discovery
- **Flexible Dependencies**: Graceful fallbacks when optional dependencies unavailable
- **Comprehensive Testing**: Full test suite with pytest

### Architecture Philosophy

The framework provides two usage patterns:

1. **Lean Core API** (recommended for research): Direct component usage
2. **Convenient API** (optional): Configuration-based with auto-registration

## Architecture

```
gnn_pde_v2/
├── core/                    # Core data structures and base classes
│   ├── graph.py            # GraphsTuple, batching utilities
│   ├── base.py             # BaseModel for auto-registration
│   ├── base_model.py       # Enhanced BaseModel with registry
│   └── functional.py       # Scatter operations, aggregation
├── components/              # Reusable building blocks (lean API)
│   ├── encoders/           # Input encoders
│   │   ├── mlp_encoder.py  # Standard MLP encoder
│   │   └── fourier_encoder.py  # Fourier feature encoder
│   ├── processors/         # Message passing and processing
│   │   ├── graphnet_block.py    # GraphNet message passing
│   │   ├── transformer_block.py # Attention mechanisms
│   │   └── fno_block.py          # Fourier neural operators
│   ├── decoders/           # Output decoders
│   │   ├── mlp_decoder.py  # Standard MLP decoder
│   │   └── probe_decoder.py     # Arbitrary query point decoder
│   ├── layers/             # Utility layers
│   │   └── residual.py     # Residual connections
│   ├── fno.py              # FNO components (SpectralConv, etc.)
│   ├── transformer.py      # Transformer components
│   └── probe.py            # Probe-based components
├── convenient/              # High-level API (optional)
│   ├── registry.py         # Auto-registration decorators
│   ├── config.py           # Pydantic configurations
│   ├── builder.py          # Configuration-based model building
│   ├── training.py         # Unified training wrapper
│   ├── aggregation.py      # Extended aggregation functions
│   └── initializers/       # Weight initialization utilities
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
├── config/                  # Deprecated - use convenient.config
├── utils/                   # Utility functions
│   ├── graph_utils.py      # Graph processing utilities
│   ├── spatial_utils.py    # Spatial computations
│   └── aggregation.py     # Basic aggregation functions
└── tests/                   # Comprehensive test suite
    ├── test_core.py        # Core functionality tests
    ├── test_components.py  # Component tests
    ├── test_convenient.py  # High-level API tests
    └── test_examples.py    # Example reproduction tests
```

## Quick Start

### Lean Core API (Recommended)

```python
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import MLP, GraphNetBlock, Residual
import torch

# Build model from components
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128, 128])
        self.processor = Residual(GraphNetBlock(128, 128, n_layers=4))
        self.decoder = MLP(in_dim=128, out_dim=2, hidden_dims=[128, 64])
    
    def forward(self, graph):
        x = self.encoder(graph.nodes)
        x = self.processor(graph, x)
        return self.decoder(x)

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
from gnn_pde_v2.convenient import GNNConfig, ConfigBuilder, Model

# Define config
config = GNNConfig(
    model_type='graphnet',
    node_in_dim=5,
    edge_in_dim=3,
    out_dim=2,
    hidden_dim=128,
    n_layers=4,
)

# Build and train model
builder = ConfigBuilder(config)
model = builder.build_unified_model()

# Training step
metrics = model.train_step((graph, target))
```

## Available Components

### Core Building Blocks

| Component | Description | Usage |
|-----------|-------------|-------|
| `MLP` | Standard multi-layer perceptron | Encoder/Decoder |
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
from gnn_pde_v2.components import MLP, GraphNetBlock, Residual

# Direct component composition for maximum flexibility
encoder = MLP(in_dim=5, out_dim=128, hidden_dims=[128])
processor = Residual(GraphNetBlock(128, 128))
decoder = MLP(in_dim=128, out_dim=2, hidden_dims=[64])
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
