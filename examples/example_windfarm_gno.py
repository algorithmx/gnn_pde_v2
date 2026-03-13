"""
Example: Wind-Farm GNO (Graph Neural Operator for Wind Farm Modeling)

This example recreates the Wind-Farm-GNO model from:
https://github.com/jenspeterschoeler/Wind-Farm-GNO

Original Work Reference:
------------------------
Schøler, J. P., Peder Weilmann Rasmussen, F., Quick, J., & Réthoré, P.-E. (2025).
"Graph Neural Operator for windfarm wake flow."
Wind Energy Science Discussions (preprint).
Paper: https://doi.org/10.5194/wes-2025-261

Key Innovation:
---------------
Wind-Farm-GNO introduces a two-stage graph neural operator specifically
designed for wind farm aerodynamic modeling.

This implementation uses the gnn_pde_v2 framework components where applicable
while maintaining exact equivalence to the original Wind-Farm-GNO architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List

# Import framework components
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.components import GraphNetBlock
from gnn_pde_v2.core import MLP


def make_windfarm_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden_layers: int,
    activation: str = "relu",
    layer_norm_out: bool = False,
    dropout: float = 0.0,
) -> MLP:
    """Create Wind-Farm-GNO faithful MLP using framework MLP.

    Matches `external_research/Wind-Farm-GNO/models/mlp.py`:
    - num_hidden_layers hidden layers of same size
    - activation after each hidden layer
    - optional output LayerNorm applied once at the end
    """
    total_layers = num_hidden_layers + 1  # hidden layers + output layer
    return MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=[hidden_dim] * num_hidden_layers,
        activation=activation,
        dropout=dropout,
        norms=[None] * (total_layers - 1) + ['layer' if layer_norm_out else None],
    )


class WindFarmGNO(AutoRegisterModel, name='windfarm_gno', namespace='example'):
    """
    Wind-Farm-GNO implementation using gnn_pde_v2 framework components.
    
    Precise equivalent of Wind-Farm-GNO model.
    Original implementation: Wind-Farm-GNO/src/models/gnn/windfarm_gnn.py
    
    Architecture (Two-Stage):
        Stage 1: Turbine-to-Turbine (T2T)
            turbine_features [B, num_turbines, num_turbine_features]
                ↓
            EdgeEncoder: rel_pos → edge_features (framework's MLP)
            TurbineGNN: n_layers × GN blocks (framework's GraphNetBlock)
                ↓
            TurbineDecoder: predictions [B, num_turbines, 1]
        
        Stage 2: Probe-to-Turbine (P2T)
            probe_positions [B, num_probes, 2]
            probe_features [B, num_probes, probe_feature_dim]
                ↓
            Find k-NN: Connect probes to nearest turbines
            Aggregate: Pool information from connected turbines
            Decoder: Flow predictions [B, num_probes, out_channels]
    
    Key insight: Decoupled graph construction allows generalization
    across different turbine layouts.
    """
    
    def __init__(
        self,
        num_turbine_features: int,
        num_edge_features: int,
        turbine_output_dim: int,
        num_probe_features: int,
        probe_output_dim: int,
        k_neighbors: int = 5,
        n_hidden: int = 128,
        n_layers: int = 6,
        num_iterations: int = 6,
        output_normalizer: Optional[callable] = None,
    ):
        super().__init__()
        
        self.num_turbine_features = num_turbine_features
        self.num_edge_features = num_edge_features
        self.turbine_output_dim = turbine_output_dim
        self.num_probe_features = num_probe_features
        self.probe_output_dim = probe_output_dim
        self.k_neighbors = k_neighbors
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_iterations = num_iterations
        self.output_normalizer = output_normalizer
        
        # ==================== Stage 1: Turbine-to-Turbine ====================
        
        # Edge encoder for T2T graph (output-only LayerNorm semantics)
        self.edge_encoder = make_windfarm_mlp(
            in_dim=num_edge_features,
            hidden_dim=n_hidden,
            out_dim=n_hidden,
            num_hidden_layers=n_layers,
            activation='relu',
            layer_norm_out=True,
        )

        # Turbine encoder (output-only LayerNorm semantics)
        self.turbine_encoder = make_windfarm_mlp(
            in_dim=num_turbine_features,
            hidden_dim=n_hidden,
            out_dim=n_hidden,
            num_hidden_layers=n_layers,
            activation='relu',
            layer_norm_out=True,
        )
        
        # Graph Network for turbine-to-turbine message passing
        # Using framework's GraphNetBlock
        self.turbine_gnn = TurbineGNN(
            node_dim=n_hidden,
            edge_dim=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers,
            num_iterations=num_iterations,
        )
        
        # Turbine decoder (Wind-Farm-GNO sets layer_norm_decoder=False; keep output LN disabled)
        self.turbine_decoder = make_windfarm_mlp(
            in_dim=n_hidden,
            hidden_dim=n_hidden,
            out_dim=turbine_output_dim,
            num_hidden_layers=n_layers,
            activation='relu',
            layer_norm_out=False,
        )
        
        # ==================== Stage 2: Probe-to-Turbine ====================
        
        # Probe aggregation
        self.probe_aggregation = ProbeAggregation(
            turbine_hidden=n_hidden,
            probe_feature_dim=num_probe_features,
            k=k_neighbors,
            n_hidden=n_hidden,
        )
        
        # Probe decoder (keep output LayerNorm disabled; original default layer_norm_decoder=False)
        self.probe_decoder = make_windfarm_mlp(
            in_dim=n_hidden + num_probe_features,
            hidden_dim=n_hidden,
            out_dim=probe_output_dim,
            num_hidden_layers=n_layers,
            activation='relu',
            layer_norm_out=False,
        )
    
    def forward(
        self,
        turbine_features: torch.Tensor,
        turbine_positions: torch.Tensor,
        probe_features: torch.Tensor,
        probe_positions: torch.Tensor,
        edge_index_turbine: torch.Tensor,
        edge_attr_turbine: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through two-stage architecture.
        
        Args:
            turbine_features: [B, num_turbines, num_turbine_features]
            turbine_positions: [B, num_turbines, 2]
            probe_features: [B, num_probes, num_probe_features]
            probe_positions: [B, num_probes, 2]
            edge_index_turbine: [2, num_edges_turbine] or [B, 2, num_edges_turbine]
            edge_attr_turbine: Edge attributes
            batch: Batch assignment for turbines
            
        Returns:
            Dict with 'turbine' and 'probe' predictions
        """
        # ==================== Stage 1: Turbine-to-Turbine ====================
        
        # Encode edge attributes
        B = turbine_features.shape[0]
        num_edges = edge_attr_turbine.shape[1] if edge_attr_turbine.dim() == 3 else edge_attr_turbine.shape[0]
        edge_emb = self.edge_encoder(edge_attr_turbine.reshape(-1, self.num_edge_features))
        edge_emb = edge_emb.reshape(B, num_edges, self.n_hidden)
        
        # Encode turbine node features
        turbine_emb = self.turbine_encoder(turbine_features.reshape(-1, self.num_turbine_features))
        turbine_emb = turbine_emb.reshape(B, -1, self.n_hidden)
        
        # Apply GNN for message passing using framework components
        turbine_emb = self.turbine_gnn(
            turbine_emb, edge_index_turbine, edge_emb, batch
        )
        
        # Decode turbine predictions
        turbine_pred = self.turbine_decoder(turbine_emb.reshape(-1, self.n_hidden))
        turbine_pred = turbine_pred.reshape(B, -1, self.turbine_output_dim)
        
        # ==================== Stage 2: Probe-to-Turbine ====================
        
        # Aggregate turbine information for each probe
        probe_emb = self.probe_aggregation(
            turbine_emb, turbine_positions, probe_positions
        )
        
        # Concatenate probe features with aggregated turbine info
        probe_input = torch.cat([probe_emb, probe_features], dim=-1)
        
        # Decode probe predictions
        probe_pred = self.probe_decoder(probe_input.reshape(-1, self.n_hidden + self.num_probe_features))
        probe_pred = probe_pred.reshape(B, -1, self.probe_output_dim)
        
        # Apply normalization if provided
        if self.output_normalizer is not None:
            probe_pred = self.output_normalizer(probe_pred)
        
        return {
            'turbine': turbine_pred,
            'probe': probe_pred,
        }
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'windfarm_gno',
            'num_turbine_features': self.num_turbine_features,
            'turbine_output_dim': self.turbine_output_dim,
            'probe_output_dim': self.probe_output_dim,
            'k_neighbors': self.k_neighbors,
            'n_hidden': self.n_hidden,
            'n_layers': self.n_layers,
        }


class TurbineGNN(nn.Module):
    """
    Multi-layer Graph Network for turbine-to-turbine communication.
    
    Uses framework's GraphNetBlock for message passing.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_hidden: int,
        n_layers: int,
        num_iterations: int,
    ):
        super().__init__()
        
        self.num_iterations = num_iterations
        
        # Stack of GN blocks using framework components
        self.blocks = nn.ModuleList([
            GraphNetBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=None,
                hidden_dim=n_hidden,
                activation='relu',
            )
            for _ in range(num_iterations)
        ])
    
    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multiple GN blocks."""
        B, num_nodes, _ = node_emb.shape
        
        # Flatten batch for processing
        nodes_flat = node_emb.reshape(-1, node_emb.shape[-1])
        edges_flat = edge_attr.reshape(-1, edge_attr.shape[-1])
        
        # Handle edge_index for batched processing
        if edge_index.dim() == 3:
            # [B, 2, num_edges] -> [2, B*num_edges]
            edge_index_list = []
            for b in range(B):
                offset = b * num_nodes
                edge_index_b = edge_index[b] + offset
                edge_index_list.append(edge_index_b)
            edge_index_flat = torch.cat(edge_index_list, dim=1)
        else:
            edge_index_flat = edge_index
        
        # Create n_node and n_edge for batched graph
        n_node = torch.full((B,), num_nodes, dtype=torch.long, device=node_emb.device)
        n_edge = torch.full((B,), edge_attr.shape[1], dtype=torch.long, device=node_emb.device)
        
        # Create GraphsTuple
        graph = GraphsTuple(
            nodes=nodes_flat,
            edges=edges_flat,
            senders=edge_index_flat[0],
            receivers=edge_index_flat[1],
            globals=None,
            n_node=n_node,
            n_edge=n_edge,
        )
        
        # Process through blocks
        for block in self.blocks:
            graph = block(graph)
        
        # Reshape back
        return graph.nodes.reshape(B, num_nodes, -1)


class ProbeAggregation(nn.Module):
    """
    Probe-to-Turbine aggregation.
    
    For each probe point:
    1. Find k nearest turbines (using Euclidean distance)
    2. Aggregate features from those turbines (mean pooling)
    3. Output: Probe representation
    """
    
    def __init__(
        self,
        turbine_hidden: int,
        probe_feature_dim: int,
        k: int = 5,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.k = k
    
    def forward(
        self,
        turbine_emb: torch.Tensor,
        turbine_positions: torch.Tensor,
        probe_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate turbine information for each probe.
        
        Args:
            turbine_emb: [B, num_turbines, turbine_hidden]
            turbine_positions: [B, num_turbines, 2]
            probe_positions: [B, num_probes, 2]
            
        Returns:
            [B, num_probes, turbine_hidden]
        """
        B, num_turbines, hidden = turbine_emb.shape
        num_probes = probe_positions.shape[1]
        
        # Compute distances between probes and turbines
        distances = torch.cdist(probe_positions, turbine_positions)
        
        # Find k nearest turbines for each probe
        _, nearest_indices = torch.topk(distances, self.k, dim=-1, largest=False)
        
        # Aggregate turbine features
        probe_emb = []
        for b in range(B):
            # Get features of k nearest turbines for each probe
            nearest_features = turbine_emb[b][nearest_indices[b]]
            
            # Mean pooling over k turbines
            aggregated = nearest_features.mean(dim=1)
            probe_emb.append(aggregated)
        
        return torch.stack(probe_emb)


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the Wind-Farm-GNO equivalent.
    
    This configuration is suitable for wind farm modeling with:
    - 5 turbines
    - 30 probe points (10×3 grid)
    """
    print("=" * 60)
    print("Wind-Farm-GNO Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Model configuration
    num_turbines = 5
    num_probes = 30
    
    model = WindFarmGNO(
        num_turbine_features=10,
        num_edge_features=4,
        turbine_output_dim=1,
        num_probe_features=6,
        probe_output_dim=1,
        k_neighbors=5,
        n_hidden=128,
        n_layers=6,
        num_iterations=6,
    )
    
    # Example inputs
    batch_size = 2
    
    # Turbine information
    turbine_features = torch.randn(batch_size, num_turbines, 10)
    turbine_positions = torch.randn(batch_size, num_turbines, 2)
    
    # Probe information
    probe_features = torch.randn(batch_size, num_probes, 6)
    probe_positions = torch.randn(batch_size, num_probes, 2)
    
    # Turbine graph edges (fully connected for wake interactions)
    edge_index = []
    for i in range(num_turbines):
        for j in range(num_turbines):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Edge attributes
    edge_attr = torch.randn(batch_size, edge_index.shape[1], 4)
    
    # Expand edge_index for batch
    edge_index_batch = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Forward pass
    output = model(
        turbine_features=turbine_features,
        turbine_positions=turbine_positions,
        probe_features=probe_features,
        probe_positions=probe_positions,
        edge_index_turbine=edge_index_batch,
        edge_attr_turbine=edge_attr,
    )
    
    print(f"\nModel Configuration:")
    print(f"  k_neighbors: {model.k_neighbors}")
    print(f"  Hidden size: {model.n_hidden}")
    print(f"  Number of layers: {model.n_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Turbine features: {turbine_features.shape}")
    print(f"  Turbine positions: {turbine_positions.shape}")
    print(f"  Probe features: {probe_features.shape}")
    print(f"  Probe positions: {probe_positions.shape}")
    print(f"  Turbine predictions: {output['turbine'].shape}")
    print(f"  Probe predictions: {output['probe'].shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    model, output = example_usage()
