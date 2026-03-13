"""
Probe-based decoder for arbitrary query points.

Based on Wind-Farm-GNO: two-stage processing with probe graph.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple, batch_graphs
from ..core.mlp import MLP


class ProbeDecoder(nn.Module):
    """
    Decoder for arbitrary query points using probe mechanism.

    Two-stage approach:
    1. Process source graph (already done by processor)
    2. Message pass from source to probe locations
    3. Decode at probe locations

    Reference: Wind-Farm-GNO probe-based decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int = 32,
        out_dim: int = 3,
        hidden_dim: int = 128,
        n_probe_layers: int = 2,
        k_nearest: int = 3,
        distance_encoding: str = 'rbf',
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.k_nearest = k_nearest
        self.distance_encoding = distance_encoding

        # Edge encoder: encode distance to edge features
        if distance_encoding == 'rbf':
            # RBF encoding dimension
            edge_input_dim = k_nearest  # One feature per neighbor
        else:
            edge_input_dim = k_nearest * 2  # Position differences

        self.edge_encoder = MLP(
            in_dim=edge_input_dim,
            out_dim=edge_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )

        # Probe processor layers
        self.probe_layers = nn.ModuleList([
            ProbeMessagePassingLayer(latent_dim, edge_dim, hidden_dim, activation)
            for _ in range(n_probe_layers)
        ])

        # Output MLP
        self.output_mlp = MLP(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
    
    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode at query positions.
        
        Args:
            graph: Processed source GraphsTuple with node features
            query_positions: [N_queries, n_dim] - Query point coordinates.
                Required for ProbeDecoder; raises ``ValueError`` if ``None``.
            
        Returns:
            [N_queries, out_dim] - Output at query positions

        Raises:
            ValueError: If ``query_positions`` is ``None``.
            ValueError: If ``graph.nodes`` or ``graph.positions`` is ``None``.
        """
        if query_positions is None:
            raise ValueError(
                "ProbeDecoder requires query_positions to be provided. "
                "Pass the [N_queries, n_dim] tensor of query coordinates."
            )
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for ProbeDecoder")
        if graph.positions is None:
            raise ValueError("Graph must have positions for ProbeDecoder")
        
        # Create probe graph: edges from source nodes to query points
        probe_graph = self._construct_probe_graph(
            source_positions=graph.positions,
            source_features=graph.nodes,
            query_positions=query_positions,
        )
        
        # Process probe graph
        for layer in self.probe_layers:
            probe_graph = layer(probe_graph)
        
        # Decode only at probe nodes (query points), which are appended after source nodes
        n_queries = query_positions.shape[0]
        return self.output_mlp(probe_graph.nodes[-n_queries:])
    
    def _construct_probe_graph(
        self,
        source_positions: torch.Tensor,
        source_features: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> GraphsTuple:
        """
        Construct probe graph with edges from source to query points.
        
        Args:
            source_positions: [N_source, n_dim]
            source_features: [N_source, node_dim]
            query_positions: [N_queries, n_dim]
            
        Returns:
            GraphsTuple with probe nodes and edges
        """
        n_queries = query_positions.shape[0]
        n_source = source_positions.shape[0]
        
        # Find k nearest source nodes for each query point
        distances = torch.cdist(query_positions, source_positions)  # [N_queries, N_source]
        _, nearest_indices = torch.topk(distances, k=self.k_nearest, largest=False, dim=-1)
        # nearest_indices: [N_queries, k_nearest]
        
        # Create edges: source nodes -> probe nodes
        # Receivers: probe nodes appended after source nodes
        # Senders: source nodes (nearest_indices flattened)
        local_receivers = torch.arange(n_queries, device=query_positions.device).repeat_interleave(self.k_nearest)
        receivers = local_receivers + n_source
        senders = nearest_indices.reshape(-1)
        
        # Edge features: distances
        nearest_distances = torch.gather(
            distances, 1, nearest_indices
        )  # [N_queries, k_nearest]
        
        if self.distance_encoding == 'rbf':
            # RBF encoding: use distances directly
            edge_features = nearest_distances.reshape(-1, 1).expand(-1, self.k_nearest)
        else:
            # Position difference encoding
            query_expanded = query_positions.repeat_interleave(self.k_nearest, dim=0)
            source_nearest = source_positions[senders]
            pos_diff = query_expanded - source_nearest
            edge_features = torch.cat([pos_diff, nearest_distances.reshape(-1, 1).expand(-1, pos_diff.shape[1])], dim=-1)
        
        # Encode edge features
        edge_features = self.edge_encoder(edge_features)
        
        # Create probe graph
        # Nodes: initialized from aggregated source features
        probe_nodes = torch.zeros(n_queries, source_features.shape[1], device=query_positions.device)
        probe_nodes.index_add_(0, local_receivers, source_features[senders])
        
        # Normalize by number of incoming edges
        counts = torch.zeros(n_queries, device=query_positions.device)
        counts.index_add_(0, local_receivers, torch.ones_like(local_receivers, dtype=torch.float))
        probe_nodes = probe_nodes / (counts.unsqueeze(1) + 1e-8)
        
        all_nodes = torch.cat([source_features, probe_nodes], dim=0)
        all_positions = torch.cat([source_positions, query_positions], dim=0)

        return GraphsTuple(
            nodes=all_nodes,
            edges=edge_features,
            receivers=receivers,
            senders=senders,
            globals=None,
            n_node=torch.tensor([n_source + n_queries], device=query_positions.device),
            n_edge=torch.tensor([len(receivers)], device=query_positions.device),
            positions=all_positions,
        )


class ProbeMessagePassingLayer(nn.Module):
    """
    Single message passing layer for probe graph.
    
    Performs one step of message passing from source nodes to probe nodes:
    1. Update edges based on sender/receiver nodes and current edge features
    2. Aggregate messages to receiver (probe) nodes
    3. Update node features
    
    Args:
        latent_dim: Dimension for node features
        edge_dim: Dimension for edge features
        hidden_dim: Hidden dimension for MLPs
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh')
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.edge_dim = edge_dim

        # Edge update: [sender_node, receiver_node, edge] -> new_edge
        self.edge_mlp = MLP(
            in_dim=2 * latent_dim + edge_dim,
            out_dim=edge_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )

        # Node update: [node, aggregated_edges] -> new_node
        self.node_mlp = MLP(
            in_dim=latent_dim + edge_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        One message passing step.
        
        Args:
            graph: Input GraphsTuple with nodes, edges, senders, receivers
            
        Returns:
            Updated GraphsTuple with new node and edge features (with residual)
        """
        nodes = graph.nodes
        edges = graph.edges
        receivers = graph.receivers
        senders = graph.senders
        
        # Edge update
        sender_features = nodes[senders]
        receiver_features = nodes[receivers]
        edge_inputs = torch.cat([sender_features, receiver_features, edges], dim=-1)
        new_edges = self.edge_mlp(edge_inputs)
        
        # Node update (aggregate to receivers)
        edge_dim = new_edges.shape[-1]
        aggregated = torch.zeros(nodes.shape[0], edge_dim, device=nodes.device, dtype=nodes.dtype)
        aggregated.index_add_(0, receivers, new_edges)
        
        node_inputs = torch.cat([nodes, aggregated], dim=-1)
        new_nodes = self.node_mlp(node_inputs)
        
        # Residual connection
        return graph.replace(nodes=nodes + new_nodes, edges=edges + new_edges)
