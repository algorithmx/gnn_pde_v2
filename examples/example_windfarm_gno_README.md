# Wind-Farm-GNO Implementation Notes

## Overview

This is a complete rewrite of the Wind-Farm GNO example for the gnn_pde_v2 framework, based on:

- **Paper**: Schøler et al. (2025) "Graph Neural Operator for windfarm wake flow"
- **Repository**: https://github.com/jenspeterschoeler/Wind-Farm-GNO

## Architecture

The Wind-Farm-GNO is a two-stage Graph Neural Operator designed for wind farm aerodynamic modeling:

### Stage 1: Turbine-to-Turbine (T2T)

Encodes turbine interactions into a latent representation:

```
Turbine Features + Edge Features
        ↓
   RBF Encoding (optional)
        ↓
   Node/Edge Encoders (MLP)
        ↓
   GEN Blocks × N (message passing)
        ↓
   Turbine Latent Features
```

### Stage 2: Probe-to-Turbine (P2T)

Decodes flow field at arbitrary probe locations:

```
Probe Positions + Turbine Latent Features
        ↓
   k-NN Graph Construction
        ↓
   Probe GEN Blocks × M (message passing)
        ↓
   Decoder MLP
        ↓
   Flow Predictions at Probes
```

## Key Components

### 1. RBF Encoder

Encodes scalar distances into high-dimensional features using Gaussian basis functions:

- **num_kernels**: Number of RBF basis functions (default: 20)
- **d_min/d_max**: Distance range for kernel centers (default: -1.0 to 1.0)
- **learnable**: Whether μ and β are trainable parameters
- **Cosine cutoff**: Smooth spatial truncation at d_max

Reference: Behler & Parrinello (2007)

### 2. GEN Block

GEneralized aggregation Network block from Li et al. (2020) "DeeperGCN":

```python
# Message computation
m_ij = ReLU(e_ij + h_j) + epsilon

# Softmax aggregation
agg_i = sum_j softmax_j(m_ij) * m_ij

# Node update
h'_i = MLP(h_i + agg_i)
```

The softmax aggregation learns dynamic attention weights for neighbor messages.

### 3. WindFarm GNN

Complete GNN stage with:
- Encoder: MLPs for nodes and edges
- Processor: Stack of GEN blocks
- Decoder: MLP for predictions

Can be configured to skip encoding/decoding for use in multi-stage architectures.

### 4. WindFarm GNO

Complete two-stage model combining:
- T2T processor (turbine graph)
- P2T processor (probe graph construction + processing)
- Shared decoders for final predictions

## Paper Hyperparameters

From the grid search in Section 3.1 of the paper:

```python
latent_dim = 128
hidden_dim = 128
num_mlp_layers = 6
wt_message_passing_steps = 6  # Stage 1
probe_message_passing_steps = 6  # Stage 2
k_neighbors = 5

# RBF encoding
rbf_kwargs = {
    'num_kernels': 20,
    'd_min': -1.0,
    'd_max': 1.0,
    'learnable': True,
}
```

## Framework Integration

### Components Used

- `gnn_pde_v2.core.MLP`: Flexible MLP with per-layer normalization control
- `gnn_pde_v2.core.GraphsTuple`: Graph representation
- `gnn_pde_v2.core.functional.scatter_softmax`: Softmax aggregation
- `gnn_pde_v2.core.AutoRegisterModel`: Model registration

### Key Design Decisions

1. **No framework processors used directly**: The GEN block is implemented from scratch to match the paper's specific message passing equation with epsilon stability term.

2. **GraphsTuple for both stages**: The probe graph is constructed as a proper GraphsTuple for consistency with the framework.

3. **Explicit probe graph construction**: The `_build_probe_graph` method explicitly creates k-NN edges from probes to turbines.

4. **RBF encoding optional**: Can be disabled for comparison or when edge features are already encoded.

## Usage

```python
from gnn_pde_v2.examples.example_windfarm_gno import WindFarmGNO
from gnn_pde_v2.core.graph import GraphsTuple

# Create model
model = WindFarmGNO(
    num_turbine_features=10,
    num_edge_features=4,
    turbine_output_dim=1,
    num_probe_features=6,
    probe_output_dim=1,
    latent_dim=128,
    hidden_dim=128,
    num_mlp_layers=6,
    wt_message_passing_steps=6,
    probe_message_passing_steps=6,
    k_neighbors=5,
    use_rbf=True,
)

# Create turbine graph
turbine_graph = GraphsTuple(
    nodes=turbine_features,      # [N_turbines, features]
    edges=edge_attr,             # [N_edges, edge_features]
    receivers=edge_index[1],     # [N_edges]
    senders=edge_index[0],       # [N_edges]
    positions=turbine_positions, # [N_turbines, 2]
    ...
)

# Forward pass
output = model(
    turbine_graph=turbine_graph,
    probe_positions=probe_positions,  # [N_probes, 2]
    probe_features=probe_features,    # [N_probes, features]
)

# Results
output['turbine']  # [N_turbines, turbine_output_dim]
output['probe']    # [N_probes, probe_output_dim]
```

## Comparison with Original

| Aspect | Original (JAX/Flax) | This Implementation (PyTorch) |
|--------|---------------------|------------------------------|
| Framework | JAX + jraph + flax | PyTorch + gnn_pde_v2 |
| GEN Block | Custom jraph block | Custom nn.Module |
| RBF Encoder | Flax module | PyTorch nn.Module |
| Message Passing | GEN with softmax agg | GEN with softmax agg |
| Batching | jraph batching | GraphsTuple batching |
| Edge Features | Explicit | Explicit |

## Testing

Run the example:

```bash
python examples/example_windfarm_gno.py
```

This will:
1. Create a WindFarmGNO model with paper defaults
2. Run a forward pass with dummy data
3. Display model statistics

## References

1. Schøler et al. (2025) - Wind-Farm-GNO paper
2. Li et al. (2020) - DeeperGCN (GEN block)
3. Behler & Parrinello (2007) - RBF encoding
4. Seidman et al. (2022) - NOMAD framework (theoretical foundation)
