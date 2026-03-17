"""
Example: Graph U-Nets (Gao & Ji, ICML 2019)

This example implements the Graph U-Nets model from:
"Graph U-Nets", Hongyang Gao & Shuiwang Ji, ICML 2019
Paper: https://arxiv.org/abs/1905.05178

The paper proposes:
1. gPool (graph pooling): Adaptively selects nodes based on scalar projection
2. gUnpool (graph unpooling): Restores graph structure using stored indices
3. Graph U-Net architecture: Encoder-decoder with skip connections

This implementation uses the gnn_pde_v2 framework components:
- GraphPool and GraphUnpool from components.multiscale.graph_pooling
- GraphsTuple from core.graph
- MLP from core.mlp

Reference:
---------
Gao, H., & Ji, S. (2019). Graph U-Nets. 
In Proceedings of the 36th International Conference on Machine Learning (ICML).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import replace

# Framework imports
from gnn_pde_v2.core.graph import GraphsTuple
from gnn_pde_v2.core.mlp import MLP
from gnn_pde_v2.components.multiscale.graph_pooling import GraphPool, GraphUnpool
from gnn_pde_v2.components import GCNBlock

# For backwards compatibility, we keep ImprovedGCN as an alias to GCNBlock
# This uses the built-in framework implementation which matches the paper's formula:
#   X' = D^(-1/2)(A + kI)D^(-1/2) X W
# where k=2 for Improved GCN (A + 2I)
ImprovedGCN = lambda in_dim, out_dim, activation=None: GCNBlock(
    in_dim=in_dim,
    out_dim=out_dim,
    self_loop_weight=2.0,  # A + 2I (improved GCN from paper)
    activation=activation,
)


class GraphUNets(nn.Module):
    """
    Unified Graph U-Nets architecture for both node and graph classification.

    Encoder-decoder architecture with:
    - GCN layers for message passing
    - gPool layers for downsampling in encoder
    - gUnpool layers for upsampling in decoder
    - Skip connections between encoder and decoder

    Architecture (from paper):
        Input
          ↓
    [Embedding GCN]
          ↓
    [GCN → gPool] ───────────┐  (Encoder Block 1)
          ↓                  │
    [GCN → gPool] ───────────┤  (Encoder Block 2)
          ↓                  │ Skip
           ...               │ Connections
          ↓                  │
    [GCN → gPool] ───────────┤  (Encoder Block N)
          ↓                  │
      [Bottleneck GCN]       │
          ↓                  │
    [gUnpool → GCN] ←────────┘  (Decoder Block N)
          ↓
    [gUnpool → GCN]             (Decoder Block 2)
          ↓
    [gUnpool → GCN]             (Decoder Block 1)
          ↓
    [Final GCN / Classifier]
          ↓
        Output

    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden/feature dimension for GCN layers
        out_dim: Output dimension (number of classes)
        nodes_per_level: List of fixed node counts for each pooling level.
            If None, uses pool_ratios instead.
        pool_ratios: List of pooling ratios for each level (e.g., [0.9, 0.7, 0.6, 0.5]).
            Used when nodes_per_level is None.
        n_encoder_layers: Number of GCN layers per encoder block (default: 1)
        n_decoder_layers: Number of GCN layers per decoder block (default: 1)
        skip_connection: Type of skip connection ('add' or 'concat')
        connectivity_augmentation: Graph power for connectivity (default: 2)
        activation: Activation for GCN layers (default: None/identity)
        global_pool: Global pooling method for graph classification.
            None for node classification (default), 'mean', 'sum', or 'max' for graph-level output.

    Reference:
        Gao & Ji, "Graph U-Nets", ICML 2019
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        nodes_per_level: Optional[List[int]] = None,
        pool_ratios: Optional[List[float]] = None,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        skip_connection: str = "add",
        connectivity_augmentation: int = 2,
        activation: Optional[str] = None,
        global_pool: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.skip_connection = skip_connection
        self.connectivity_augmentation = connectivity_augmentation
        self.global_pool = global_pool

        # Determine pooling specification
        if nodes_per_level is not None:
            self.nodes_per_level = nodes_per_level
            self.pool_ratios = None
            self.use_ratios = False
        elif pool_ratios is not None:
            self.pool_ratios = pool_ratios
            self.nodes_per_level = None
            self.use_ratios = True
        else:
            raise ValueError("Either nodes_per_level or pool_ratios must be provided")

        self.n_levels = len(self.nodes_per_level) if nodes_per_level else len(pool_ratios)

        # Embedding layer
        self.embedding = ImprovedGCN(in_dim, hidden_dim, activation=activation)

        # Encoder: GCN layers followed by gPool
        self.encoder_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(self.n_levels):
            # GCN layers for this encoder level
            level_gcns = nn.ModuleList([
                ImprovedGCN(hidden_dim, hidden_dim, activation=activation)
                for _ in range(n_encoder_layers)
            ])
            self.encoder_gcns.append(level_gcns)

            # gPool layer (except for last level which goes to bottleneck)
            if i < self.n_levels - 1:
                if self.use_ratios:
                    # Placeholder: will compute k dynamically in forward
                    self.pools.append(
                        GraphPool(
                            k=1,  # Will be overridden
                            feature_dim=hidden_dim,
                            use_gate=True,
                            connectivity_augmentation=connectivity_augmentation,
                        )
                    )
                else:
                    self.pools.append(
                        GraphPool(
                            k=nodes_per_level[i],
                            feature_dim=hidden_dim,
                            use_gate=True,
                            connectivity_augmentation=connectivity_augmentation,
                        )
                    )

        # Bottleneck GCN
        self.bottleneck_gcn = ImprovedGCN(hidden_dim, hidden_dim, activation=activation)

        # Decoder: gUnpool followed by GCN
        self.decoder_gcns = nn.ModuleList()

        for i in range(self.n_levels):
            level_gcns = nn.ModuleList([
                ImprovedGCN(hidden_dim, hidden_dim, activation=activation)
                for _ in range(n_decoder_layers)
            ])
            self.decoder_gcns.append(level_gcns)

        # gUnpool (shared across all levels)
        self.unpool = GraphUnpool()

        # Skip connection projections (for concat mode)
        if skip_connection == "concat":
            self.skip_projections = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim)
                for _ in range(self.n_levels)
            ])

        # Output layer: GCN for node classification, Linear for graph classification
        if global_pool is not None:
            self.classifier = nn.Linear(hidden_dim, out_dim)
        else:
            self.output_gcn = ImprovedGCN(hidden_dim, out_dim, activation=None)

    def forward(self, graph: GraphsTuple) -> torch.Tensor:
        """
        Forward pass through Graph U-Nets.

        Args:
            graph: Input GraphsTuple with:
                - nodes: [N, in_dim] node features
                - senders: [E] source node indices
                - receivers: [E] destination node indices

        Returns:
            - Node classification: [N, out_dim] output predictions for each node
            - Graph classification: [out_dim] graph-level prediction
        """
        x = graph

        # Embedding layer
        x = self.embedding(x)

        # Store encoder outputs for skip connections
        encoder_outputs = []
        indices_list = []
        original_sizes = []

        # Encoder path
        for i in range(self.n_levels):
            # Store original size for skip connection
            original_sizes.append(x.nodes.shape[0] if x.nodes is not None else 0)

            # Apply GCN layers
            for gcn in self.encoder_gcns[i]:
                x = gcn(x)

            # Store for skip connection
            encoder_outputs.append(x)

            # Pool if not the last level
            if i < self.n_levels - 1:
                if self.use_ratios:
                    # Compute k dynamically based on ratio
                    num_nodes = x.nodes.shape[0] if x.nodes is not None else 0
                    k = max(1, int(num_nodes * self.pool_ratios[i]))

                    # Create pool with computed k (reuse module, just update k)
                    pool = GraphPool(
                        k=k,
                        feature_dim=self.hidden_dim,
                        use_gate=True,
                        connectivity_augmentation=self.connectivity_augmentation,
                    )
                    x, indices = pool(x)
                else:
                    x, indices = self.pools[i](x)
                indices_list.append(indices)

        # Bottleneck
        x = self.bottleneck_gcn(x)

        # Decoder path (reverse order)
        for i in range(self.n_levels - 1, -1, -1):
            # Unpool if not the first decoder level
            if i < len(indices_list):
                target_size = original_sizes[i]
                x = self.unpool(x, indices_list[i], target_size)

            # Skip connection
            skip = encoder_outputs[i]
            if self.skip_connection == "add" and x.nodes is not None and skip.nodes is not None:
                x = x.replace(nodes=x.nodes + skip.nodes)
            elif self.skip_connection == "concat" and x.nodes is not None and skip.nodes is not None:
                combined = torch.cat([x.nodes, skip.nodes], dim=-1)
                x = x.replace(nodes=self.skip_projections[i](combined))

            # Apply GCN layers
            for gcn in self.decoder_gcns[i]:
                x = gcn(x)

        # Output based on mode
        if self.global_pool is not None:
            # Graph classification: global pooling + linear
            if x.nodes is None:
                raise ValueError("Graph must have nodes")

            if self.global_pool == "mean":
                graph_feat = x.nodes.mean(dim=0)
            elif self.global_pool == "sum":
                graph_feat = x.nodes.sum(dim=0)
            elif self.global_pool == "max":
                graph_feat = x.nodes.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown global_pool: {self.global_pool}")

            return self.classifier(graph_feat)
        else:
            # Node classification: GCN output
            x = self.output_gcn(x)
            return x.nodes


# Backward compatibility alias
GraphUNetsForGraphClassification = GraphUNets


def create_synthetic_graph(
    num_nodes: int = 500,
    num_edges: int = 2000,
    feature_dim: int = 1433,  # Like Cora
    num_classes: int = 7,
) -> Tuple[GraphsTuple, torch.Tensor]:
    """
    Create a synthetic graph for testing.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        feature_dim: Dimension of node features
        num_classes: Number of classes for labels

    Returns:
        graph: GraphsTuple
        labels: Node labels [num_nodes]
    """
    # Create random node features
    nodes = torch.randn(num_nodes, feature_dim)

    # Create random edges (ensure no self-loops for initial graph)
    senders = torch.randint(0, num_nodes, (num_edges,))
    receivers = torch.randint(0, num_nodes, (num_edges,))

    # Remove self-loops
    mask = senders != receivers
    senders = senders[mask]
    receivers = receivers[mask]

    # Create symmetric edges (undirected graph) - need to track original pairs
    # Add reverse edges: receiver -> sender
    sym_senders = torch.cat([senders, receivers])
    sym_receivers = torch.cat([receivers, senders])

    # Remove duplicates
    edge_pairs = torch.stack([sym_senders, sym_receivers], dim=1)
    edge_pairs = torch.unique(edge_pairs, dim=0)
    final_senders = edge_pairs[:, 0]
    final_receivers = edge_pairs[:, 1]

    # Create graph
    graph = GraphsTuple.from_flat(
        nodes=nodes,
        edges=None,
        senders=final_senders,
        receivers=final_receivers,
        globals=None,
        n_node=torch.tensor([num_nodes]),
        n_edge=torch.tensor([final_senders.shape[0]]),
    )

    # Create random labels
    labels = torch.randint(0, num_classes, (num_nodes,))

    return graph, labels


def example_node_classification():
    """
    Example: Node classification with Graph U-Nets.

    Configuration similar to Cora dataset experiments from the paper:
    - 4 encoder/decoder blocks
    - gPool sampling: 2000, 1000, 500, 200 nodes
    - Skip connections: Addition
    - Identity activation
    """
    print("=" * 70)
    print("Graph U-Nets: Node Classification Example")
    print("=" * 70)

    # Configuration (from paper Section 4.2)
    config = {
        'in_dim': 1433,      # Like Cora
        'hidden_dim': 128,
        'out_dim': 7,        # Number of classes
        'nodes_per_level': [2000, 1000, 500, 200],
        'n_encoder_layers': 1,
        'n_decoder_layers': 1,
        'skip_connection': 'add',
        'connectivity_augmentation': 2,
        'activation': None,  # Identity as per paper
        'global_pool': None, # Node classification
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = GraphUNets(**config)

    # Create synthetic data
    graph, labels = create_synthetic_graph(
        num_nodes=2708,  # Like Cora
        num_edges=10556,
        feature_dim=1433,
        num_classes=7,
    )

    print(f"\nInput Graph:")
    print(f"  Nodes: {graph.nodes.shape}")
    print(f"  Edges: {graph.senders.shape[0]}")
    print(f"  Labels: {labels.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(graph)

    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Expected: [2708, 7]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {n_params:,}")

    # Test backward pass
    model.train()
    output = model(graph)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print("  Backward pass: OK")

    return model, graph, output


def example_graph_classification():
    """
    Example: Graph classification with Graph U-Nets.

    Configuration similar to D&D/PROTEINS experiments from the paper.
    Uses unified API with pool_ratios and global_pool.
    """
    print("\n" + "=" * 70)
    print("Graph U-Nets: Graph Classification Example")
    print("=" * 70)

    config = {
        'in_dim': 50,        # Common feature dimension
        'hidden_dim': 128,
        'out_dim': 2,        # Binary classification
        'pool_ratios': [0.9, 0.7, 0.6, 0.5],  # From paper
        'n_encoder_layers': 1,
        'n_decoder_layers': 1,
        'skip_connection': 'add',
        'connectivity_augmentation': 2,
        'activation': None,
        'global_pool': 'mean',
    }

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = GraphUNets(**config)

    # Create synthetic data
    graph, _ = create_synthetic_graph(
        num_nodes=284,  # Average from D&D
        num_edges=1500,
        feature_dim=50,
        num_classes=2,
    )

    print(f"\nInput Graph:")
    print(f"  Nodes: {graph.nodes.shape}")
    print(f"  Edges: {graph.senders.shape[0]}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(graph)

    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Expected: [2] (graph-level prediction)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {n_params:,}")

    return model, graph, output


def verify_components():
    """
    Verify individual components work correctly.
    """
    print("\n" + "=" * 70)
    print("Component Verification")
    print("=" * 70)

    # Create simple test graph
    num_nodes = 10
    senders = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
    receivers = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

    graph = GraphsTuple.from_flat(
        nodes=torch.randn(num_nodes, 16),
        edges=None,
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=torch.tensor([num_nodes]),
        n_edge=torch.tensor([senders.shape[0]]),
    )

    print(f"\nTest Graph: {num_nodes} nodes, {senders.shape[0]} edges")

    # Test ImprovedGCN
    print("\n1. ImprovedGCN Layer:")
    gcn = ImprovedGCN(in_dim=16, out_dim=32)
    out = gcn(graph)
    print(f"   Input:  {graph.nodes.shape}")
    print(f"   Output: {out.nodes.shape}")
    assert out.nodes.shape == (num_nodes, 32), "GCN output shape mismatch"
    print("   ✓ Pass")

    # Test gPool
    print("\n2. GraphPool Layer:")
    pool = GraphPool(k=5, feature_dim=16, connectivity_augmentation=2)
    pooled, indices = pool(graph)
    print(f"   Input:  {graph.nodes.shape[0]} nodes")
    print(f"   Output: {pooled.nodes.shape[0]} nodes")
    print(f"   Indices: {indices.shape[0]}")
    assert pooled.nodes.shape[0] == 5, "Pool output node count mismatch"
    print("   ✓ Pass")

    # Test gUnpool
    print("\n3. GraphUnpool Layer:")
    unpool = GraphUnpool()
    unpooled = unpool(pooled, indices, num_nodes)
    print(f"   Input:  {pooled.nodes.shape[0]} nodes")
    print(f"   Output: {unpooled.nodes.shape[0]} nodes")
    assert unpooled.nodes.shape[0] == num_nodes, "Unpool output node count mismatch"
    print("   ✓ Pass")

    print("\nAll component tests passed!")


if __name__ == "__main__":
    # Verify components
    verify_components()

    # Run examples
    print("\n")
    model_node, graph_node, output_node = example_node_classification()
    model_graph, graph_graph, output_graph = example_graph_classification()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)