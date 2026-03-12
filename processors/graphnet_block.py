"""
GraphNet processor: DeepMind-style message passing.

Standard 3-step update: edges → nodes → globals.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..encoders.mlp_encoder import MLP


class GraphNetBlock(nn.Module):
    """
    Single GraphNet message passing block.
    
    Update order:
    1. Update edges: aggregate sender nodes + edge features
    2. Update nodes: aggregate incoming edges + node features
    3. Update globals: aggregate all nodes/edges
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: Optional[int] = None,
        hidden_dim: int = 128,
        activation: str = 'gelu',
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        
        # Edge update: [sender_node, receiver_node, edge, global] -> new_edge
        edge_input_dim = 2 * node_dim + edge_dim
        if global_dim is not None:
            edge_input_dim += global_dim
        
        self.edge_mlp = MLP(
            in_dim=edge_input_dim,
            out_dim=edge_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
        
        # Node update: [node, aggregated_edges, global] -> new_node
        node_input_dim = node_dim + edge_dim
        if global_dim is not None:
            node_input_dim += global_dim
        
        self.node_mlp = MLP(
            in_dim=node_input_dim,
            out_dim=node_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
        
        # Global update (optional)
        if global_dim is not None:
            global_input_dim = node_dim + edge_dim + global_dim
            self.global_mlp = MLP(
                in_dim=global_input_dim,
                out_dim=global_dim,
                hidden_dims=[hidden_dim, hidden_dim],
                activation=activation,
            )
        else:
            self.global_mlp = None
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Single message passing step."""
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
        aggregated_edges = self._aggregate_edges_to_nodes(new_edges, receivers, nodes.shape[0])
        
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
        """Map global features to each edge."""
        # globals_: [batch_size, global_dim]
        # n_edge: [batch_size]
        # output: [total_edges, global_dim]
        global_dim = globals_.shape[-1]
        global_per_edge = []
        
        for i, n_e in enumerate(n_edge):
            global_per_edge.append(globals_[i:i+1].expand(n_e, global_dim))
        
        return torch.cat(global_per_edge, dim=0)
    
    def _map_global_to_nodes(self, globals_: torch.Tensor, n_node: torch.Tensor) -> torch.Tensor:
        """Map global features to each node."""
        global_dim = globals_.shape[-1]
        global_per_node = []
        
        for i, n_n in enumerate(n_node):
            global_per_node.append(globals_[i:i+1].expand(n_n, global_dim))
        
        return torch.cat(global_per_node, dim=0)
    
    def _aggregate_edges_to_nodes(
        self,
        edges: torch.Tensor,
        receivers: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Aggregate incoming edges to nodes (sum)."""
        # edges: [E, edge_dim]
        # receivers: [E] - node indices
        edge_dim = edges.shape[-1]
        aggregated = torch.zeros(num_nodes, edge_dim, device=edges.device, dtype=edges.dtype)
        aggregated.index_add_(0, receivers, edges)
        return aggregated
    
    def _aggregate_to_global(self, features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Aggregate features to global (mean)."""
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
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: Optional[int] = None,
        n_layers: int = 15,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            GraphNetBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                global_dim=global_dim,
                hidden_dim=hidden_dim,
                activation=activation,
            )
            for _ in range(n_layers)
        ])
        
        self.residual = residual
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Process through all blocks."""
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
