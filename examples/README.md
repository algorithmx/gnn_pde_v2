# GNN-PDE v2 Examples

This directory contains reference implementations of popular PDE-GNN architectures from research papers, rewritten to use the `gnn_pde_v2` framework components while maintaining exact equivalence to the original implementations.

## Overview

| Example | Paper | Key Innovation | Use Case |
|---------|-------|----------------|----------|
| [MeshGraphNets](#meshgraphnets) | Pfaff et al., ICML 2021 | Graph networks on unstructured meshes | CFD, cloth simulation |
| [DeepXDE](#deepxde) | Lu et al., SIAM Review 2021 | Physics-informed neural networks | PDE solving with constraints |
| [NeuralOperator FNO](#neuraloperator-fno) | Li et al., ICLR 2021 | Fourier neural operators | Function space learning |
| [Transolver](#transolver) | Wu et al., ICML 2024 | Physics-attention mechanism | Irregular geometries |
| [Unisolver](#unisolver) | Zhou et al., ICML 2024 | PDE-conditional transformers | Universal PDE solver |
| [WindFarm GNO](#windfarm-gno) | Schøler et al., 2025 | Two-stage graph operator | Wind farm wake modeling |
| [Graph-PDE GNO](#graph-pde-gno) | Li et al., 2020 | Edge-conditioned convolution | Irregular meshes |

---

## MeshGraphNets

**Paper**: "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICML 2021)

**Original Repo**: https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Encoder | `MLPMeshEncoder` | Separate MLPs for nodes and edges |
| Processor | `GraphNetBlock` × 15 | Message passing with residual connections |
| Decoder | `MLPDecoder` | 2-layer MLP for acceleration prediction |
| Base Class | `BaseModel` | Auto-registration as `meshgraphnets` |

### Custom Implementation

- **Residual connections**: Applied manually at model level (MeshGraphNets-style)
- **PyG Wrapper**: `MeshGraphNetsPyGWrapper` for PyTorch Geometric compatibility
- **No global features**: Explicitly set to `None` (unlike full GraphNet)

```python
from gnn_pde_v2.examples.example_meshgraphnets import MeshGraphNets

model = MeshGraphNets(
    node_input_size=11,      # position + velocity + node_type
    edge_input_size=3,       # relative displacement
    output_size=3,           # acceleration
    hidden_size=128,
    message_passing_steps=15,
)
```

---

## DeepXDE

**Paper**: "DeepXDE: A Deep Learning Library for Solving Differential Equations" (Lu et al., SIAM Review 2021)

**Original Repo**: https://github.com/lululxvi/deepxde

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Fourier Features | `FourierFeatureEncoder` | High-frequency PDE handling |
| MLP Backbone | `MLP` | Fully connected network |
| Training API | `Model` | Unified training interface |
| Base Class | `BaseModel` | Auto-registration as `deepxde` |

### Custom Implementation

- **Weight initialization**: Custom Xavier/Glorot initialization matching DeepXDE
- **Residual connections**: Optional manual implementation
- **Activation mapping**: Maps DeepXDE names to framework activations
- **Physics Loss**: Custom `PhysicsLoss` class for PDE/BC/IC constraints

```python
from gnn_pde_v2.examples.example_deepxde import DeepXDEModel, create_deepxde_model

# Standard FNN
model = DeepXDEModel(
    layer_sizes=[2, 64, 64, 64, 1],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)

# With Fourier features
model = DeepXDEModel(
    layer_sizes=[2, 128, 128, 1],
    use_fourier_features=True,
    num_fourier_features=256,
)

# Unified Model API
unified_model = create_deepxde_model(
    layer_sizes=[2, 64, 64, 1],
    learning_rate=1e-3,
)
metrics = unified_model.train_step((x, target))
```

---

## NeuralOperator FNO

**Paper**: "Fourier Neural Operator for Parametric Partial Differential Equations" (Li et al., ICLR 2021)

**Original Repo**: https://github.com/neuraloperator/neuraloperator

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Spectral Conv | `SpectralConv` | FFT-based global convolution |
| Alternative | `FNOProcessor` | Complete FNO pipeline (SimpleFNO) |
| Base Class | `BaseModel` | Auto-registration |

### Custom Implementation

- **ChannelMLP**: Custom 2-layer MLP with expansion ratios
- **Positional embedding**: Grid coordinates concatenation
- **Lifting/Projection**: 2-layer MLPs matching neuraloperator ratios

```python
from gnn_pde_v2.examples.example_neuraloperator_fno import NeuralOperatorFNO, SimpleFNO

# Full implementation matching neuraloperator
model = NeuralOperatorFNO(
    n_modes=(16, 16),
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
    lifting_channel_ratio=2.0,
    projection_channel_ratio=2.0,
)

# Simplified using framework's FNOProcessor
simple_model = SimpleFNO(
    in_channels=1,
    out_channels=1,
    width=64,
    modes=[16, 16],
    n_layers=4,
)
```

---

## Transolver

**Paper**: "Transolver: A Fast Transformer Solver for PDEs on General Geometries" (Wu et al., ICML 2024)

**Original Repo**: https://github.com/thuml/Transolver

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Preprocessor | `MLP` | Input feature projection |
| Feed-Forward | `MLP` | Transformer FFN blocks |
| Base Class | `BaseModel` | Auto-registration as `transolver` |

### Custom Implementation

- **Physics-Attention**: Custom `PhysicsAttentionIrregularMesh` (core innovation)
  - Slice-attention-deslice mechanism
  - Complexity reduction from O(N²) to O(N×G + G²)
- **Unified position encoding**: Distance-based position encoding
- **TransolverBlock**: Custom transformer block with physics-attention

```python
from gnn_pde_v2.examples.example_transolver import Transolver

model = Transolver(
    space_dim=2,
    n_layers=5,
    n_hidden=256,
    n_head=8,
    slice_num=32,      # Number of physics tokens
    ref=8,             # Reference grid size
    unified_pos=True,  # Use unified position encoding
)

output = model(x, pos)  # x: [B, N, fun_dim], pos: [B, N, space_dim]
```

---

## Unisolver

**Paper**: "Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers" (Zhou et al., ICML 2024)

**Original Repo**: https://github.com/thu-ml/Unisolver

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| MLP Backbone | `MLP` | Could be used for FFN (original uses custom) |
| Base Class | `BaseModel` | Auto-registration as `unisolver` |

### Custom Implementation

- **Decoupled AdaLN**: Separate conditioning for domain-wise (μ) and point-wise (f) parameters
- **VisEmbedder**: Scalar parameter embedding (similar to timestep embedding)
- **Transformer**: Custom dual-conditioning transformer
- **Patch embedding**: Image-to-patch conversion with reference grid distances

```python
from gnn_pde_v2.examples.example_unisolver import Unisolver

model = Unisolver(
    image_size=64,
    patch_size=4,
    dim=256,
    depth=8,
    heads=8,
    in_channels=10,
    out_channels=1,
)

output = model(x, mu, f)  # x: [B, H, W, C], mu: [B], f: [B, H, W]
```

---

## WindFarm GNO

**Paper**: "Graph Neural Operator for windfarm wake flow" (Schøler et al., 2025)

**Original Repo**: https://github.com/jenspeterschoeler/Wind-Farm-GNO

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Edge/Node Encoders | `MLP` | Feature encoding |
| Turbine GNN | `GraphNetBlock` | Message passing |
| Probe Decoder | `MLP` | Flow prediction |
| Base Class | `BaseModel` | Auto-registration as `windfarm_gno` |

### Custom Implementation

- **Two-stage architecture**:
  1. **T2T (Turbine-to-Turbine)**: Wake interaction modeling
  2. **P2T (Probe-to-Turbine)**: Flow field prediction at probe points
- **k-NN aggregation**: Custom `ProbeAggregation` for nearest turbine pooling
- **Batched processing**: Custom batch handling for variable graph sizes

```python
from gnn_pde_v2.examples.example_windfarm_gno import WindFarmGNO

model = WindFarmGNO(
    num_turbine_features=10,
    num_edge_features=4,
    turbine_output_dim=1,    # Power prediction
    num_probe_features=6,
    probe_output_dim=1,      # Flow velocity
    k_neighbors=5,
    n_hidden=128,
    n_layers=6,
)

output = model(
    turbine_features=turbine_features,
    turbine_positions=turbine_positions,
    probe_features=probe_features,
    probe_positions=probe_positions,
    edge_index_turbine=edge_index,
    edge_attr_turbine=edge_attr,
)
# Returns: {'turbine': [B, num_turbines, 1], 'probe': [B, num_probes, 1]}
```

---

## Graph-PDE GNO

**Paper**: "Neural Operator: Graph Kernel Network for Partial Differential Equations" (Li et al., 2020)

**Original Repo**: https://github.com/neuraloperator/graph-pde

### Framework Usage

| Component | Framework Class | Purpose |
|-----------|-----------------|---------|
| Node/Edge Encoders | `MLP` | Feature projection |
| Decoder | `MLP` | Output projection |
| EdgeConv | `MLP` | Edge convolution (alternative) |
| Base Class | `BaseModel` | Auto-registration as `graph_pde_gno` |

### Custom Implementation

- **Edge-conditioned weights**: `GraphConvBlock` generates weights from edge features
  - Scalar or vector weight options
  - Heterogeneous message passing
- **Aggregation**: Mean pooling over incoming edges
- **EdgeConv variant**: DGCNN-style edge convolution using framework MLP

```python
from gnn_pde_v2.examples.example_graph_pde_gno import GraphPDE_GNO, EdgeConvBlock

# Main model
model = GraphPDE_GNO(
    node_input_size=2,
    edge_input_size=3,
    output_size=1,
    hidden_size=128,
    num_layers=6,
    edge_weight_type='scalar',  # or 'vector'
)

output = model(node_features, edge_index, edge_attr)

# Alternative: EdgeConv block
edge_conv = EdgeConvBlock(in_channels=2, out_channels=64)
output = edge_conv(node_features, edge_index)
```

---

## Design Patterns

### 1. BaseModel Registration

All examples inherit from `BaseModel` for automatic registration:

```python
class MyModel(BaseModel, model_name='my_model'):
    def __init__(self, ...):
        super().__init__()
        ...

# Later access
from gnn_pde_v2.core.base_model import BaseModel
model_class = BaseModel.get_model('my_model')
```

### 2. Encoder-Processor-Decoder Pattern

Examples follow the EPD architecture where applicable:

```
Input → [Encoder] → Latent → [Processor] → Processed → [Decoder] → Output
           ↓              ↓
    Framework MLP   Framework GraphNetBlock/FNOBlock
```

### 3. Framework Component Selection

| When to Use | Framework Component | Custom Implementation |
|-------------|---------------------|----------------------|
| Standard MLP | `MLP` | Custom initialization only |
| Graph message passing | `GraphNetBlock` | Architecture-specific variants |
| Fourier convolution | `SpectralConv` | ChannelMLP customization |
| Fourier features | `FourierFeatureEncoder` | Scale and learnable options |
| Training loop | `Model` | Physics-informed losses |

### 4. Maintaining Equivalence

Each example preserves the original paper's:
- Hyperparameters (defaults match paper)
- Architecture details (layer configurations)
- Initialization schemes
- Forward pass behavior
- Output formats

The framework components provide building blocks while paper-specific logic (attention mechanisms, conditioning, etc.) is implemented custom.

---

## Running Examples

Each example can be run standalone:

```bash
# From repo root
cd gnn_pde_v2/examples

# Run individual examples
python example_meshgraphnets.py
python example_deepxde.py
python example_neuraloperator_fno.py
python example_transolver.py
python example_unisolver.py
python example_windfarm_gno.py
python example_graph_pde_gno.py
```

All examples include `example_usage()` functions demonstrating proper usage.

---

## Citation

If you use these examples, please cite the original papers:

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter W},
  booktitle={ICML},
  year={2021}
}

@article{lu2021deepxde,
  title={DeepXDE: A Deep Learning Library for Solving Differential Equations},
  author={Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  journal={SIAM Review},
  year={2021}
}

@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={ICLR},
  year={2021}
}
```
