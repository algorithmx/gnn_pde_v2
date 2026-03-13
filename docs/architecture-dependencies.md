# GNN-PDE v2 Architecture & Dependency Map

## Module Dependency Graph

```mermaid
graph TB
    subgraph "External Dependencies"
        TORCH[torch]
        NUMPY[numpy]
        PYDANTIC[pydantic]
        TORCH_CLUSTER[torch_cluster]
        TORCH_SCATTER[torch_scatter]
    end

    subgraph "Core Layer"
        BASE[core.base<br/>BaseModel]
        GRAPH[core.graph<br/>GraphsTuple]
        MLP[core.mlp<br/>MLP, SinActivation]
        FUNC[core.functional<br/>scatter_*, aggregate_*]
    end

    subgraph "Components Layer"
        ENC[components.encoders<br/>MLPEncoder, MLPMeshEncoder]
        DEC[components.decoders<br/>MLPDecoder, IndependentMLPDecoder]
        PROC[components.processors<br/>GraphNetBlock, GraphNetProcessor]
        TRANS[components.transformer<br/>TransformerBlock, Conditioning]
        FNO[components.fno<br/>SpectralConv, FNOBlock, FNOProcessor]
        LAYERS[components.layers<br/>Residual variants]
        PROBE[components.probe<br/>ProbeDecoder]
        FOURIER[components.fourier_encoder<br/>FourierFeatureEncoder]
    end

    subgraph "Models Layer"
        EPD[models.encode_process_decode<br/>EncodeProcessDecode]
        GNN_MODEL[models.gnn_model<br/>GraphNet, MeshGraphNet]
        FNO_MODEL[models.fno_model<br/>FNO, TFNO, AFNO]
        UNIFIED[models.unified_model<br/>UnifiedModel]
    end

    subgraph "Convenient Layer"
        REGISTRY[convenient.registry<br/>AutoRegisterModel]
        CONFIG[convenient.config<br/>*Config classes]
        INIT[convenient.initializers<br/>get_initializer]
        AGG[convenient.aggregation<br/>scatter_min, scatter_softmax]
    end

    subgraph "Examples"
        EX_MESH[examples.example_meshgraphnets<br/>MeshGraphNets]
        EX_DEEP[examples.example_deepxde<br/>DeepXDE]
        EX_FNO[examples.example_neuraloperator_fno<br/>FNO]
        EX_TRANS[examples.example_transolver<br/>Transolver]
        EX_UNI[examples.example_unisolver<br/>Unisolver]
    end

    subgraph "Utils"
        GRAPH_UTILS[utils.graph_utils<br/>knn_graph, radius_graph]
        SPATIAL[utils.spatial_utils<br/>spatial computations]
    end

    %% Core dependencies
    BASE --> TORCH
    GRAPH --> TORCH
    MLP --> TORCH
    MLP --> BASE
    FUNC --> TORCH
    FUNC --> TORCH_SCATTER

    %% Components depend on Core
    ENC --> MLP
    ENC --> GRAPH
    DEC --> MLP
    DEC --> GRAPH
    PROC --> MLP
    PROC --> GRAPH
    PROC --> FUNC
    TRANS --> MLP
    TRANS --> GRAPH
    FNO --> MLP
    LAYERS --> TORCH
    PROBE --> MLP
    PROBE --> FUNC
    FOURIER --> TORCH

    %% Models depend on Components AND Core
    EPD --> GRAPH
    EPD --> ENC
    EPD --> PROC
    EPD --> DEC

    GNN_MODEL --> GRAPH
    GNN_MODEL --> ENC
    GNN_MODEL --> PROC
    GNN_MODEL --> DEC

    FNO_MODEL --> FNO

    %% CRITICAL: Models depend on Convenient (violation!)
    GNN_MODEL -.->|VIOLATION| REGISTRY
    FNO_MODEL -.->|VIOLATION| REGISTRY

    %% Convenient dependencies
    REGISTRY --> BASE
    REGISTRY --> PYDANTIC
    CONFIG --> PYDANTIC
    BUILDER --> CONFIG
    BUILDER --> GNN_MODEL
    BUILDER --> FNO_MODEL
    BUILDER --> EPD
    BUILDER --> REGISTRY
    TRAINING --> TORCH
    INIT --> TORCH
    AGG --> TORCH_SCATTER

    %% Examples depend on everything
    EX_MESH --> GRAPH
    EX_MESH --> PROC
    EX_MESH --> REGISTRY

    EX_DEEP --> MLP
    EX_DEEP --> FOURIER

    EX_FNO --> FNO_MODEL

    EX_TRANS --> TRANS
    EX_UNI --> TRANS

    %% Utils dependencies
    GRAPH_UTILS --> TORCH
    GRAPH_UTILS --> TORCH_CLUSTER

    %% Styling
    classDef violation fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    classDef core fill:#d0bfff,stroke:#7950f2
    classDef component fill:#a9e34b,stroke:#5c940d
    classDef model fill:#ffd43b,stroke:#fab005
    classDef convenient fill:#ff922b,stroke:#e8590c

    class GNN_MODEL,FNO_MODEL violation
    class BASE,GRAPH,MLP,FUNC core
    class ENC,DEC,PROC,TRANS,FNO,LAYERS,PROBE,FOURIER component
    class EPD,UNIFIED model
    class REGISTRY,CONFIG,BUILDER,TRAINING,INIT,AGG convenient
```

## Intended vs Actual Layering

```mermaid
graph LR
    subgraph "Intended Layering"
        direction TB
        C1[Core] --> C2[Components] --> C3[Models] --> C4[Convenient]
    end

    subgraph "Actual Layering (with violations)"
        direction TB
        A1[Core] --> A2[Components]
        A2 --> A3[Models]
        A3 -.->|backwards!| A4[Convenient]
        A4 -.->|used by| A3
    end
```

## Component Internal Dependencies

```mermaid
graph TB
    subgraph "Encoder Dependencies"
        MLPE[MLPEncoder] --> MLP
        MESHE[MLPMeshEncoder] --> MLP
        MAKE_E[make_mlp_encoder] --> MLP
    end

    subgraph "Decoder Dependencies"
        MLPD[MLPDecoder] --> MLP
        INDD[IndependentMLPDecoder] --> MLP
        PROBED[ProbeDecoder] --> MLP
        PROBED --> scatter_mean
    end

    subgraph "Processor Dependencies"
        GN_B[GraphNetBlock] --> MLP
        GN_B --> scatter_sum
        GN_B --> scatter_mean
        GNP[GraphNetProcessor] --> GN_B
        GNP --> Residual

        T_B[TransformerBlock] --> MLP
        T_B --> MultiHeadAttention
        T_B --> Conditioning
        TP[TransformerProcessor] --> T_B

        SPEC[SpectralConv] --> torch.fft
        FNO_B[FNOBlock] --> SPEC
        FNOP[FNOProcessor] --> FNO_B
        FNOP --> MLP
    end
```

## Conditioning System Dependencies

```mermaid
graph TB
    subgraph "Conditioning Protocol"
        PROTO[ConditioningProtocol<br/>ABC]
        MOD[Modulation<br/>dataclass]
    end

    ZERO[ZeroConditioning]
    ADA[AdaLNConditioning]
    DUAL[DualAdaLNConditioning]
    FILM[FiLMConditioning]

    ZERO --> PROTO
    ADA --> PROTO
    DUAL --> PROTO
    FILM --> PROTO

    ADA --> MOD
    DUAL --> MOD
    FILM --> MOD
    ZERO --> MOD

    TRANS_C[TransformerBlock] --> PROTO
```

## Registry & Auto-Registration Flow

```mermaid
sequenceDiagram
    participant User
    participant AutoRegisterModel
    participant Registry
    participant ConfigBuilder

    User->>AutoRegisterModel: Define class with name='foo'
    AutoRegisterModel->>Registry: Register on class creation
    Registry-->>Registry: _registry['foo'] = MyClass

    User->>ConfigBuilder: build_model(name='foo')
    ConfigBuilder->>Registry: create('foo', **kwargs)
    Registry->>AutoRegisterModel: MyClass(**kwargs)
    AutoRegisterModel-->>User: instance
```

## Import Graph (Simplified)

```mermaid
graph TD
    ROOT[__init__.py]

    ROOT --> CORE[core/]
    ROOT --> COMP[components/]
    ROOT --> MODELS[models/]
    ROOT --> CONV[convenient/]

    CORE --> BASE[base.py]
    CORE --> GRAPH[graph.py]
    CORE --> MLP[mlp.py]
    CORE --> FUNC[functional.py]

    COMP --> INIT_COMP[__init__.py]
    INIT_COMP --> ENC[encoders.py]
    INIT_COMP --> PROC[processors.py]
    INIT_COMP --> DEC[decoders.py]
    INIT_COMP --> TRANS[transformer.py]
    INIT_COMP --> FNO[fno.py]
    INIT_COMP --> LAYERS[layers.py]

    MODELS --> INIT_MODELS[__init__.py]
    INIT_MODELS --> EPD[encode_process_decode.py]
    INIT_MODELS --> GNN[gnn_model.py]
    INIT_MODELS --> FNO_M[fno_model.py]

    %% Cross-layer imports
    GNN -.->|imports| CONV
    FNO_M -.->|imports| CONV
    INIT_MODELS -.->|imports| TRAINING[training.py]

    style GNN fill:#ff6b6b
    style FNO_M fill:#ff6b6b
```

## Key Issues Highlighted

### 1. Circular Dependency Risk
```mermaid
graph LR
    MODELS[models/] -->|imports| CONV[convenient/]
    CONV -->|imports| MODELS

    style MODELS fill:#ff6b6b
    style CONV fill:#ff6b6b
```

### 2. Two Model Classes Confusion (RESOLVED)
```mermaid
graph TD
    BASE[core.base.BaseModel<br/>marker class]
    TRAINING_EXAMPLES[examples.training_utils.Model<br/>training wrapper]
    AUTO[convenient.registry.AutoRegisterModel<br/>registry mixin]

    AUTO -->|extends| BASE
    MODELS[models/ classes] -->|extend| AUTO
    MODELS2[models/__init__.py] -->|no longer exports| TRAINING

    style TRAINING fill:#a9e34b,stroke:#e8590c
    style BASE fill:#d0bfff
```

### 3. FNO vs Graph Processor Mismatch
```mermaid
graph TD
    subgraph "Graph Processors"
        GNP[GraphNetProcessor<br/>forward: GraphsTuple → GraphsTuple]
        TP[TransformerProcessor<br/>forward: GraphsTuple → GraphsTuple]
    end

    subgraph "Grid Processors"
        FNOP[FNOProcessor<br/>forward: Tensor → Tensor]
        FNOP_G[FNOProcessor.forward_graph<br/>raises NotImplementedError!]
    end

    style FNOP_G fill:#ff6b6b,stroke:#c92a2a
```

## Summary

| Layer | Depends On | Should Depend On | Issue |
|-------|------------|------------------|-------|
| `core/` | torch, numpy | torch, numpy | ✅ OK |
| `components/` | core | core | ✅ OK |
| `models/` | components | components only | ✅ FIXED (training moved to examples) |
| `convenient/` | core, components, models | core, components | ⚠️ Acceptable |
| `examples/` | all | all | ✅ OK (top level) |
| `utils/` | torch_cluster (hard) | optional | ⚠️ Should be optional |
