"""
GraphNet processor: DeepMind-style message passing.

Standard 3-step update: edges → nodes → globals.
"""

from typing import Optional
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..core.functional import scatter_sum
from ..core.mlp import MLP


class GraphNetBlock(nn.Module):
    """
    Single GraphNet message passing block.

    Performs a complete message passing step following DeepMind's GraphNet
    architecture. Update order:
    1. Update edges: aggregate sender nodes + edge features
    2. Update nodes: aggregate incoming edges + node features
    3. Update globals: aggregate all nodes/edges

    Args:
        latent_dim: Dimension for node, edge, and output features
        global_latent_dim: Optional dimension for global features. If None,
            no global features are used.
        hidden_dim: Hidden dimension for internal MLPs
        activation: Activation function name ('relu', 'gelu', 'silu', 'tanh')
    """

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: Optional[int] = None,
        hidden_dim: int = 128,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.global_latent_dim = global_latent_dim

        # Edge update: [sender_node, receiver_node, edge, global] -> new_edge
        edge_input_dim = 3 * latent_dim  # sender + receiver + edge (all latent_dim)
        if global_latent_dim is not None:
            edge_input_dim += global_latent_dim

        self.edge_mlp = MLP(
            in_dim=edge_input_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

        # Node update: [node, aggregated_edges, global] -> new_node
        node_input_dim = 2 * latent_dim  # node + aggregated_edges
        if global_latent_dim is not None:
            node_input_dim += global_latent_dim

        self.node_mlp = MLP(
            in_dim=node_input_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

        # Global update (optional)
        if global_latent_dim is not None:
            global_input_dim = 2 * latent_dim + global_latent_dim
            self.global_mlp = MLP(
                in_dim=global_input_dim,
                out_dim=global_latent_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                activation=activation,
            )
        else:
            self.global_mlp = None
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Single message passing step.

        Args:
            graph: Input GraphsTuple with nodes, edges, and optionally globals

        Returns:
            Updated GraphsTuple with new node, edge, and global features

        Raises:
            ValueError: If graph.nodes is None (required for message passing)
        """
        nodes = graph.nodes
        edges = graph.edges
        receivers = graph.receivers
        senders = graph.senders
        globals_ = graph.globals
        n_node = graph.n_node
        n_edge = graph.n_edge
        
        # --- Edge update ---
        # Gather sender and receiver node features
        sender_features = nodes[senders]  # [E, node_dim]
        receiver_features = nodes[receivers]  # [E, node_dim]
        
        # Concatenate for edge update
        edge_inputs = [sender_features, receiver_features, edges]
        if globals_ is not None:
            # Map global to each edge
            global_per_edge = self._map_global_to_edges(globals_, n_edge)
            edge_inputs.append(global_per_edge)
        
        edge_inputs = torch.cat(edge_inputs, dim=-1)
        new_edges = self.edge_mlp(edge_inputs)
        
        # --- Node update ---
        # Aggregate edges to nodes (sum aggregation)
        aggregated_edges = scatter_sum(new_edges, receivers, dim=0, dim_size=nodes.shape[0])
        
        node_inputs = [nodes, aggregated_edges]
        if globals_ is not None:
            global_per_node = self._map_global_to_nodes(globals_, n_node)
            node_inputs.append(global_per_node)
        
        node_inputs = torch.cat(node_inputs, dim=-1)
        new_nodes = self.node_mlp(node_inputs)
        
        # --- Global update (optional) ---
        if self.global_mlp is not None and globals_ is not None:
            # Aggregate nodes and edges to global
            aggregated_nodes = self._aggregate_to_global(new_nodes, n_node)
            aggregated_edges_global = self._aggregate_to_global(new_edges, n_edge)
            
            global_inputs = torch.cat([aggregated_nodes, aggregated_edges_global, globals_], dim=-1)
            new_globals = self.global_mlp(global_inputs)
        else:
            new_globals = globals_
        
        return graph.replace(
            nodes=new_nodes,
            edges=new_edges,
            globals=new_globals,
        )
    
    def _map_global_to_edges(self, globals_: torch.Tensor, n_edge: torch.Tensor) -> torch.Tensor:
        """
        Map global features to each edge.

        Args:
            globals_: [batch_size, global_latent_dim] - Global features per graph
            n_edge: [batch_size] - Number of edges per graph in batch

        Returns:
            [total_edges, global_latent_dim] - Global features broadcast to each edge
        """
        # globals_: [batch_size, global_latent_dim]
        # n_edge: [batch_size]
        # output: [total_edges, global_latent_dim]
        global_latent_dim = globals_.shape[-1]
        global_per_edge = []

        for i, n_e in enumerate(n_edge):
            global_per_edge.append(globals_[i:i+1].expand(n_e, global_latent_dim))

        return torch.cat(global_per_edge, dim=0)

    def _map_global_to_nodes(self, globals_: torch.Tensor, n_node: torch.Tensor) -> torch.Tensor:
        """
        Map global features to each node.

        Args:
            globals_: [batch_size, global_latent_dim] - Global features per graph
            n_node: [batch_size] - Number of nodes per graph in batch

        Returns:
            [total_nodes, global_latent_dim] - Global features broadcast to each node
        """
        global_latent_dim = globals_.shape[-1]
        global_per_node = []

        for i, n_n in enumerate(n_node):
            global_per_node.append(globals_[i:i+1].expand(n_n, global_latent_dim))

        return torch.cat(global_per_node, dim=0)
    
    def _aggregate_to_global(self, features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features to global (mean pooling).

        Args:
            features: [total, feature_dim] - Features to aggregate
            counts: [batch_size] - Number of items per graph

        Returns:
            [batch_size, feature_dim] - Mean-pooled features per graph
        """
        # features: [total, feature_dim]
        # counts: [batch_size] - items per graph
        feature_dim = features.shape[-1]
        aggregated = []
        
        start = 0
        for count in counts:
            end = start + count.item()
            graph_features = features[start:end]
            aggregated.append(graph_features.mean(dim=0, keepdim=True))
            start = end
        
        return torch.cat(aggregated, dim=0)


class GraphNetProcessor(nn.Module):
    """
    Multi-layer GraphNet processor.

    Stacks multiple GraphNetBlocks with optional residual connections.
    This is the main processing component in Encode-Process-Decode architectures.

    Args:
        latent_dim: Dimension for node and edge features
        global_latent_dim: Optional dimension for global features. If None,
            no global features are processed.
        n_layers: Number of GraphNetBlocks to stack
        hidden_dim: Hidden dimension for internal MLPs in each block
        activation: Activation function name ('relu', 'gelu', 'silu', 'tanh')
        residual: Whether to use residual connections between blocks

    Example:
        >>> processor = GraphNetProcessor(
        ...     latent_dim=128,
        ...     n_layers=15,
        ...     residual=True
        ... )
        >>> output = processor(graph)
    """

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: Optional[int] = None,
        n_layers: int = 15,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            GraphNetBlock(
                latent_dim=latent_dim,
                global_latent_dim=global_latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
            )
            for _ in range(n_layers)
        ])

        self.residual = residual
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all GraphNetBlocks.

        Args:
            graph: Input GraphsTuple with node and edge features

        Returns:
            Processed GraphsTuple with updated features
        """
        for block in self.blocks:
            new_graph = block(graph)
            
            if self.residual:
                # Residual connection
                new_graph = graph.replace(
                    nodes=graph.nodes + new_graph.nodes if graph.nodes is not None else new_graph.nodes,
                    edges=graph.edges + new_graph.edges if graph.edges is not None else new_graph.edges,
                    globals=graph.globals + new_graph.globals if graph.globals is not None else new_graph.globals,
                )
            
            graph = new_graph
        
        return graph
