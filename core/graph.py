"""
GraphsTuple: Minimal graph representation.

Based on DeepMind's Graph Nets library but simplified:
- No validation in __post_init__
- Use dataclasses.replace() instead of custom replace()
- Minimal methods
"""

from dataclasses import dataclass, replace
from typing import Optional, List
import torch
from torch import Tensor


@dataclass
class GraphsTuple:
    """
    Minimal graph representation for batched graphs.
    
    Attributes:
        nodes: [total_nodes, node_feat_dim] - Node features
        edges: [total_edges, edge_feat_dim] - Edge features
        receivers: [total_edges] - Destination node indices
        senders: [total_edges] - Source node indices
        globals: [batch_size, global_feat_dim] - Global features
        n_node: [batch_size] - Number of nodes per graph
        n_edge: [batch_size] - Number of edges per graph
        positions: Optional [total_nodes, n_dim] - Node positions
    """
    nodes: Optional[Tensor] = None
    edges: Optional[Tensor] = None
    receivers: Optional[Tensor] = None
    senders: Optional[Tensor] = None
    globals: Optional[Tensor] = None
    n_node: Optional[Tensor] = None
    n_edge: Optional[Tensor] = None
    positions: Optional[Tensor] = None
    
    def to(self, device) -> 'GraphsTuple':
        """Move all tensors to device."""
        return GraphsTuple(
            nodes=self.nodes.to(device) if self.nodes is not None else None,
            edges=self.edges.to(device) if self.edges is not None else None,
            receivers=self.receivers.to(device) if self.receivers is not None else None,
            senders=self.senders.to(device) if self.senders is not None else None,
            globals=self.globals.to(device) if self.globals is not None else None,
            n_node=self.n_node.to(device) if self.n_node is not None else None,
            n_edge=self.n_edge.to(device) if self.n_edge is not None else None,
            positions=self.positions.to(device) if self.positions is not None else None,
        )
    
    @property
    def num_graphs(self) -> int:
        """Number of graphs in batch."""
        if self.n_node is not None:
            return len(self.n_node)
        return 1 if self.nodes is not None else 0
    
    @property
    def num_nodes(self) -> int:
        """Total number of nodes across all graphs."""
        return self.nodes.shape[0] if self.nodes is not None else 0
    
    @property
    def num_edges(self) -> int:
        """Total number of edges across all graphs."""
        return self.edges.shape[0] if self.edges is not None else 0
    
    @property
    def device(self) -> torch.device:
        """Get device from nodes tensor."""
        if self.nodes is not None:
            return self.nodes.device
        if self.edges is not None:
            return self.edges.device
        return torch.device('cpu')
    
    def replace(self, **kwargs) -> 'GraphsTuple':
        """Return a new GraphsTuple with replaced fields."""
        return replace(self, **kwargs)


def batch_graphs(graphs: List[GraphsTuple]) -> GraphsTuple:
    """
    Batch a list of graphs into a single GraphsTuple.
    
    Args:
        graphs: List of GraphsTuple to batch
        
    Returns:
        Batched GraphsTuple
    """
    if not graphs:
        raise ValueError("Cannot batch empty list of graphs")
    
    device = graphs[0].device
    
    all_nodes = []
    all_edges = []
    all_receivers = []
    all_senders = []
    all_globals = []
    all_positions = []
    n_nodes = []
    n_edges = []
    
    node_offset = 0
    
    for g in graphs:
        if g.nodes is not None:
            all_nodes.append(g.nodes)
            n_nodes.append(g.nodes.shape[0])
        else:
            n_nodes.append(0)
        
        if g.edges is not None:
            all_edges.append(g.edges)
            n_edges.append(g.edges.shape[0])
        else:
            n_edges.append(0)
        
        if g.receivers is not None:
            all_receivers.append(g.receivers + node_offset)
            all_senders.append(g.senders + node_offset)
        
        if g.globals is not None:
            all_globals.append(g.globals)
        
        if g.positions is not None:
            all_positions.append(g.positions)
        
        node_offset += n_nodes[-1]
    
    nodes = torch.cat(all_nodes, dim=0) if all_nodes else None
    edges = torch.cat(all_edges, dim=0) if all_edges else None
    receivers = torch.cat(all_receivers, dim=0) if all_receivers else None
    senders = torch.cat(all_senders, dim=0) if all_senders else None
    globals_ = torch.cat(all_globals, dim=0) if all_globals else None
    positions = torch.cat(all_positions, dim=0) if all_positions else None
    
    n_node = torch.tensor(n_nodes, dtype=torch.long, device=device)
    n_edge = torch.tensor(n_edges, dtype=torch.long, device=device)
    
    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
        positions=positions,
    )


def unbatch_graphs(graph: GraphsTuple) -> List[GraphsTuple]:
    """
    Unbatch a GraphsTuple into a list of individual graphs.
    
    Args:
        graph: Batched GraphsTuple
        
    Returns:
        List of individual GraphsTuples
    """
    if graph.n_node is None:
        return [graph]
    
    num_graphs = len(graph.n_node)
    n_nodes = graph.n_node.cpu().tolist()
    n_edges = graph.n_edge.cpu().tolist() if graph.n_edge is not None else [0] * num_graphs
    
    graphs = []
    node_start = 0
    edge_start = 0
    
    for i in range(num_graphs):
        n_node = n_nodes[i]
        n_edge = n_edges[i]
        
        nodes = graph.nodes[node_start:node_start + n_node] if graph.nodes is not None else None
        positions = graph.positions[node_start:node_start + n_node] if graph.positions is not None else None
        
        edges = graph.edges[edge_start:edge_start + n_edge] if graph.edges is not None else None
        receivers = graph.receivers[edge_start:edge_start + n_edge] - node_start if graph.receivers is not None else None
        senders = graph.senders[edge_start:edge_start + n_edge] - node_start if graph.senders is not None else None
        
        globals_ = graph.globals[i:i+1] if graph.globals is not None else None
        
        graphs.append(GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=torch.tensor([n_node], device=graph.device),
            n_edge=torch.tensor([n_edge], device=graph.device),
            positions=positions,
        ))
        
        node_start += n_node
        edge_start += n_edge
    
    return graphs
