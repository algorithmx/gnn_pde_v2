# GNN-PDE v2 Architecture & Dependency Map

## Module Dependency Graph

```mermaid
graph TB
    subgraph "External Dependencies"
        TORCH[torch]
        NUMPY[numpy]
        PYDANTIC[pydantic<br/>optional]
        TORCH_CLUSTER[torch_cluster<br/>optional]
        TORCH_SCATTER[torch_scatter<br/>optional]
    end

    subgraph "Core Layer"
        BASE[core.base<br/>BaseModel]
        GRAPH[core.graph<br/>GraphsTuple]
        MLP[core.mlp<br/>MLP, SinActivation]
        FUNC[core.functional<br/>scatter_*, aggregate_*]
        AGG[core.aggregation<br/>Aggregation, Sum, Mean, Max, Min]
        REG[core.registry<br/>AutoRegisterModel, MODEL_REGISTRY]
        PROT[core.protocols<br/>ConditioningProtocol, GraphProcessor, etc.]
    end

    subgraph "Components Layer"
        PROC[components.processors<br/>GraphNetBlock, MessagePassingBase, GENBlock]
        GCN[components.gcn<br/>GCNBlock, GCNBlockWithEdgeFeatures]
        TRANS[components.transformer<br/>TransformerBlock, TransformerProcessor]
        ATTN[components.attention<br/>MultiHeadAttention, PhysicsTokenAttention, QKNorm, SparseGraphAttention]
        COND[components.conditioning<br/>AdaLN, DualAdaLN, FiLM, ZeroConditioning]
        TEMP[components.temperature<br/>TemperatureBase, AdaptiveTemperature, AnnealedTemperature]
        SPECT[components.spectral<br/>SpectralConv, FNOBlock, AFNOBlock, FNOProcessor]
        DEC[components.decoders<br/>MLPDecoder, IndependentMLPDecoder]
        PROBE[components.probe<br/>ProbeDecoder, WindFarmGNO, ProbeGraphBuilder]
        RBF[components.rbf<br/>LearnableRBFEncoder, GaussianRBFEncoder]
        FOURIER[components.fourier_encoder<br/>FourierFeatureEncoder]
        LAYERS[components.layers<br/>Residual, GatedResidual, make_residual]
    end

    subgraph "Models Layer"
        EPD[models.encode_process_decode<br/>EncodeProcessDecode]
        GNN[models.gnn_model<br/>GraphNet, MeshGraphNet, MeshEncoder]
        FNO[models.fno_model<br/>FNO, TFNO, AFNO]
        MSFNO[models.multiscale_fno<br/>MultiscaleFNO]
    end

    subgraph "Examples"
        EX_MESH[examples.example_meshgraphnets]
        EX_GNN[examples.example_graph_pde_gno]
        EX_FNO[examples.example_neuraloperator_fno]
        EX_TRANS[examples.example_transolver, example_transolver_v3]
        EX_UNI[examples.example_unisolver]
        EX_WIND[examples.example_windfarm_gno]
        TRAINING[examples.training_utils]
    end

    subgraph "Utils"
        GRAPH_UTILS[utils.graph_utils<br/>knn_graph, radius_graph, compute_edge_features]
        SPATIAL[utils.spatial_utils<br/>grid_to_points, points_to_grid]
    end

    %% Core dependencies
    BASE --> TORCH
    GRAPH --> TORCH
    MLP --> TORCH
    FUNC --> TORCH
    FUNC -.->|optional| TORCH_SCATTER
    AGG --> TORCH
    REG --> BASE
    PROT --> TORCH

    %% Components depend on Core
    PROC --> GRAPH
    PROC --> MLP
    PROC --> FUNC
    PROC --> AGG

    GCN --> GRAPH
    GCN --> MLP

    TRANS --> MLP
    TRANS --> PROT

    ATTN --> MLP
    ATTN --> TORCH

    COND --> PROT
    COND --> MLP

    TEMP --> TORCH
    TEMP --> MLP

    SPECT --> TORCH
    SPECT --> NUMPY

    DEC --> MLP
    DEC --> GRAPH
    DEC --> PROT

    PROBE --> MLP
    PROBE --> GRAPH
    PROBE --> FUNC

    RBF --> MLP
    FOURIER --> TORCH
    FOURIER --> GRAPH

    LAYERS --> TORCH

    %% Models depend on Components AND Core
    EPD --> GRAPH
    EPD --> PROT

    GNN --> GRAPH
    GNN --> MLP
    GNN --> REG
    GNN --> PROC
    GNN --> DEC
    GNN --> EPD

    FNO --> REG
    FNO --> SPECT

    MSFNO --> REG
    MSFNO --> SPECT

    %% Examples depend on everything
    EX_MESH --> GNN
    EX_GNN --> PROC
    EX_GNN --> REG

    EX_FNO --> FNO

    EX_TRANS --> TRANS
    EX_TRANS --> ATTN

    EX_UNI --> TRANS

    EX_WIND --> PROBE

    EX_DEEP --> FOURIER
    EX_DEEP --> MLP

    TRAINING --> TORCH

    %% Utils dependencies
    GRAPH_UTILS --> TORCH
    GRAPH_UTILS --> GRAPH
    GRAPH_UTILS -.->|optional| TORCH_CLUSTER

    SPATIAL --> TORCH
    SPATIAL --> GRAPH

    %% Styling
    classDef core fill:#d0bfff,stroke:#7950f2
    classDef component fill:#a9e34b,stroke:#5c940d
    classDef model fill:#ffd43b,stroke:#fab005
    classDef example fill:#74c0fc,stroke:#339af0
    classDef optional fill:#ffe066,stroke:#fab005,stroke-dasharray: 5 5

    class BASE,GRAPH,MLP,FUNC,AGG,REG,PROT core
    class PROC,GCN,TRANS,ATTN,COND,TEMP,SPECT,DEC,PROBE,RBF,FOURIER,LAYERS component
    class EPD,GNN,FNO,MSFNO model
    class EX_MESH,EX_GNN,EX_FNO,EX_TRANS,EX_UNI,EX_WIND,EX_DEEP,TRAINING example
```

## Layer Architecture

```mermaid
graph LR
    direction TB
    C1[Core<br/>base, graph, mlp, functional<br/>aggregation, registry, protocols] --> C2[Components<br/>processors, transformers, attention<br/>spectral, decoders, etc.]
    C2 --> C3[Models<br/>encode_process_decode<br/>gnn_model, fno_model, multiscale_fno]
    C1 --> C4[Examples<br/>example_*.py]

    style C1 fill:#d0bfff
    style C2 fill:#a9e34b
    style C3 fill:#ffd43b
    style C4 fill:#74c0fc
```

**Note**: There is NO `convenient/` layer in this version. The registry (`AutoRegisterModel`, `MODEL_REGISTRY`) is in `core/`.

## Core Layer Details

### core/base.py - BaseModel
Minimal marker class for framework models. No auto-registration.

```python
from gnn_pde_v2.core import BaseModel
```

### core/registry.py - ModelRegistry & AutoRegisterModel
Two registration mechanisms:
1. **ModelRegistry** - Standalone registry object with decorator/imperative APIs
2. **AutoRegisterModel** - Base class that auto-registers subclasses

```python
from gnn_pde_v2.core import MODEL_REGISTRY, AutoRegisterModel

# Decorator style
@MODEL_REGISTRY.register('my_model', aliases=['mymodel'])
class MyModel(nn.Module):
    ...

# Base class style
class MyModel(AutoRegisterModel, name='my_model'):
    ...
```

### core/graph.py - GraphsTuple
Minimal graph representation with batching utilities.

```python
from gnn_pde_v2.core import GraphsTuple, batch_graphs, unbatch_graphs
```

### core/functional.py - Scatter Operations
Thin wrappers with torch_scatter fallback to pure PyTorch.

```python
from gnn_pde_v2.core import scatter_sum, scatter_mean, scatter_max, scatter_min
from gnn_pde_v2.core import aggregate_edges, broadcast_nodes_to_edges
```

### core/aggregation.py - Aggregation Protocol
Extensible aggregation system with pluggable reduction methods.

```python
from gnn_pde_v2.core import Aggregation, Sum, Mean, Max, Min, get_aggregation
```

### core/protocols.py - Structural Protocols
TypeScript-style structural protocols for component contracts.

```python
from gnn_pde_v2.core.protocols import (
    Modulation, ConditioningProtocol,
    GraphEncoder, GraphProcessor, NodeDecoder, QueryDecoder, Decoder,
    GraphModel, PositionEncoder, GridProcessor, GridModel
)
```

## Components Layer

### Processors (Graph-based)
| Module | Description |
|--------|-------------|
| `MessagePassingBase` | Abstract base for graph message passing |
| `GraphNetBlock` | DeepMind-style node/edge update |
| `EdgeConditionedConvBlock` | Edge-conditioned convolution |
| `GENBlock` | Graph Edges Networks block |
| `GlobalGraphNetBlock` | Full encoder-processor-decoder with globals |
| `GCNBlock`, `GCNBlockWithEdgeFeatures` | Graph Convolutional Networks |
| `TransformerBlock`, `TransformerProcessor` | Transformer for graphs |

### Attention Mechanisms
| Module | Description |
|--------|-------------|
| `MultiHeadAttention` | Standard multi-head attention |
| `PhysicsTokenAttention` | Token-based physics attention |
| `PhysicsTokenAttentionV3` | Improved physics attention |
| `QKNormMultiHeadAttention` | Attention with QK normalization |
| `SparseGraphAttention` | Sparse attention for graphs |
| `RelativePositionEncoding` | Relative position encoding |

### Conditioning
| Module | Description |
|--------|-------------|
| `ZeroConditioning` | No conditioning (passthrough) |
| `AdaLNConditioning`, `AdaLNConditioningNoGate` | Adaptive Layer Norm |
| `DualAdaLNConditioning`, `DualAdaLNConditioningNoGate` | Dual AdaLN |
| `FiLMConditioning` | Feature-wise Linear Modulation |

### Temperature Mechanisms
| Module | Description |
|--------|-------------|
| `FixedTemperature` | Fixed temperature scaling |
| `LearnableScalarTemperature` | Learnable scalar temperature |
| `PerHeadTemperature` | Per-head temperature |
| `AdaptiveTemperature` | Adaptive temperature |
| `AnnealedTemperature` | Annealed temperature |
| `create_temperature_module` | Factory function |

### Spectral (Grid-based)
| Module | Description |
|--------|-------------|
| `SpectralConv` | Standard spectral convolution |
| `SeparableSpectralConv` | Factorized spectral convolution |
| `FNOBlock` | FNO block |
| `AFNOBlock` | Adaptive FNO block |
| `FNOProcessor` | Full FNO processor |

### Decoders
| Module | Description |
|--------|-------------|
| `MLPDecoder` | MLP-based node decoder |
| `IndependentMLPDecoder` | Independent MLP decoder |
| `ProbeDecoder` | Query-based decoder |
| `WindFarmGNO` | Wind farm-specific GNO |

## Models Layer

### Lazy Loading
Models use lazy loading for optional dependencies:

```python
from gnn_pde_v2.models import FNO, TFNO, AFNO, GraphNet, MeshGraphNet

# These are lazily loaded - raises ImportError with helpful message if deps missing
```

### Registered Models
| Name | Aliases | Type |
|------|---------|------|
| `graphnet` | gnn, graph_net | Graph |
| `meshgraphnet` | mgn, mesh_graph_net | Graph |
| `fno` | fourier_no, fno2d | Grid |
| `tfno` | tensorized_fno | Grid |
| `afno` | adaptive_fno | Grid |

```python
from gnn_pde_v2.core import MODEL_REGISTRY

model = MODEL_REGISTRY.create('graphnet', node_in_dim=11, edge_in_dim=3, out_dim=3)
```

## Optional Dependencies

```mermaid
graph LR
    TORCH[torch] --> HARD[Hard]
    NUMPY[numpy] --> HARD
    SCATTER[torch_scatter] -.-> OPT[Optional]
    CLUSTER[torch_cluster] -.-> OPT
    PYDANTIC[pydantic] -.-> OPT

    style HARD fill:#d0bfff
    style OPT fill:#ffe066,stroke-dasharray: 5 5
```

| Package | Usage | Fallback |
|---------|-------|----------|
| `torch` | All | N/A |
| `numpy` | Spectral ops | N/A |
| `torch_scatter` | Fast scatter | Pure PyTorch |
| `torch_cluster` | knn/radius graph | Required |
| `pydantic` | Config (not used) | N/A |

**Note**: `torch_cluster` is currently required for `knn_graph` and `radius_graph` in `utils.graph_utils`.

## Import Reference

### Package Root (`gnn_pde_v2`)
```python
from gnn_pde_v2 import GraphsTuple, BaseModel
from gnn_pde_v2 import scatter_sum, scatter_mean, aggregate_edges
```

### Core (`gnn_pde_v2.core`)
```python
from gnn_pde_v2.core import (
    GraphsTuple, batch_graphs, unbatch_graphs,
    BaseModel, MLP, SinActivation,
    AutoRegisterModel, MODEL_REGISTRY,
    scatter_sum, scatter_mean, scatter_max, scatter_min, scatter_softmax,
    aggregate_edges, broadcast_nodes_to_edges,
    Aggregation, Sum, Mean, Max, Min, get_aggregation,
    # Protocols
    Modulation, ConditioningProtocol,
    GraphEncoder, GraphProcessor, NodeDecoder, QueryDecoder, Decoder, GraphModel,
    PositionEncoder, GridProcessor, GridModel,
)
```

### Components (`gnn_pde_v2.components`)
```python
from gnn_pde_v2.components import (
    # Encoders
    FourierFeatureEncoder,
    # Layers
    Residual, GatedResidual, make_residual,
    # Processors
    MessagePassingBase, GraphNetBlock, GraphNetProcessor,
    EdgeConditionedConvBlock, EdgeConvBlock, GENBlock,
    GlobalGraphNetBlock, GlobalGraphNetProcessor,
    GCNBlock, GCNBlockWithEdgeFeatures,
    TransformerBlock, TransformerProcessor,
    # Attention
    MultiHeadAttention, PhysicsTokenAttention, PhysicsTokenAttentionV3,
    TiledSliceOperation, QKNormMultiHeadAttention, SparseGraphAttention,
    RelativePositionEncoding,
    # Conditioning
    ZeroConditioning, AdaLNConditioning, AdaLNConditioningNoGate,
    DualAdaLNConditioning, DualAdaLNConditioningNoGate,
    FiLMConditioning, apply_modulation,
    # Temperature
    TemperatureBase, FixedTemperature, LearnableScalarTemperature,
    PerHeadTemperature, AdaptiveTemperature, AnnealedTemperature,
    create_temperature_module,
    # Spectral
    FNOProcessor, SpectralConv, SeparableSpectralConv, SpectralConvBase,
    make_spectral_conv, SpectralBlockBase, FNOBlock, FNOMLPBlock, AFNOBlock,
    # Decoders
    MLPDecoder, IndependentMLPDecoder,
    ProbeDecoder, WindFarmGNO, ProbeGraphBuilder,
    # RBF
    LearnableRBFEncoder, GaussianRBFEncoder,
    # Protocols (re-exported)
    GraphEncoder, GraphProcessor, NodeDecoder, QueryDecoder, Decoder, GraphModel,
    PositionEncoder, GridProcessor, GridModel,
)
```

### Models (`gnn_pde_v2.models`)
```python
from gnn_pde_v2.models import (
    EncodeProcessDecode,
    FNO, TFNO, AFNO,  # Lazy loaded
    GraphNet, MeshGraphNet,  # Lazy loaded
)
```

### Utils (`gnn_pde_v2.utils`)
```python
from gnn_pde_v2.utils import (
    compute_edge_features, knn_graph, radius_graph,
    grid_to_points, points_to_grid,
)
```

## Summary

| Layer | Depends On | Status |
|-------|------------|--------|
| `core/` | torch, numpy | ✅ Clean |
| `components/` | core | ✅ Clean |
| `models/` | core, components | ✅ Clean |
| `examples/` | all | ✅ Top-level |
| `utils/` | core, torch_cluster (optional) | ✅ Clean |

**Key Changes from v1**:
- Registry moved from `convenient/` to `core/`
- No `convenient/` layer exists
- Added `core/aggregation.py` for pluggable aggregations
- Added `components/temperature.py` for temperature mechanisms
- Models use lazy loading pattern
- Protocols defined in `core/protocols.py` and re-exported in components
