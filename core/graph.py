"""
GraphsTuple: Minimal graph representation.

Based on DeepMind's Graph Nets library but simplified:
- Use dataclasses.replace() instead of custom replace()
- Minimal methods
- Validation via __post_init__ when VALIDATE_GRAPHS is True (default: False)
"""

from dataclasses import dataclass, replace
from typing import Optional, List
import torch
from torch import Tensor

# Set to True to enable fail-fast validation in GraphsTuple.__post_init__.
# Useful for debugging; keep False in production to avoid overhead from
# dataclasses.replace() calls inside processor forward passes.
VALIDATE_GRAPHS: bool = False


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

    def __post_init__(self) -> None:
        """Optionally validate the graph on construction.

        Validation is skipped by default to avoid overhead from
        ``dataclasses.replace()`` calls inside forward passes.  Enable it by
        setting ``core.graph.VALIDATE_GRAPHS = True`` before creating graphs,
        or call :meth:`validate` explicitly.
        """
        if VALIDATE_GRAPHS:
            self.validate()

    def validate(self) -> None:
        """Validate internal consistency of this GraphsTuple.

        Checks:
        1. ``n_node`` sum matches ``nodes.shape[0]``.
        2. ``n_edge`` sum matches ``edges.shape[0]``.
        3. ``senders`` and ``receivers`` lie within ``[0, num_nodes)``.
        4. ``senders`` and ``receivers`` have equal length and match ``edges``.
        5. ``positions`` has the same leading dimension as ``nodes``.
        6. All tensors reside on the same device.

        Raises:
            ValueError: on any consistency violation.
        """
        # --- shape / count checks ---
        if self.n_node is not None and self.nodes is not None:
            expected = int(self.n_node.sum().item())
            actual = self.nodes.shape[0]
            if expected != actual:
                raise ValueError(
                    f"GraphsTuple: n_node sums to {expected} but "
                    f"nodes.shape[0] == {actual}"
                )

        if self.n_edge is not None and self.edges is not None:
            expected = int(self.n_edge.sum().item())
            actual = self.edges.shape[0]
            if expected != actual:
                raise ValueError(
                    f"GraphsTuple: n_edge sums to {expected} but "
                    f"edges.shape[0] == {actual}"
                )

        # --- sender / receiver consistency ---
        if self.senders is not None and self.receivers is not None:
            if self.senders.shape != self.receivers.shape:
                raise ValueError(
                    f"GraphsTuple: senders.shape {self.senders.shape} != "
                    f"receivers.shape {self.receivers.shape}"
                )
            if self.edges is not None and self.senders.shape[0] != self.edges.shape[0]:
                raise ValueError(
                    f"GraphsTuple: senders length {self.senders.shape[0]} != "
                    f"edges length {self.edges.shape[0]}"
                )
        elif (self.senders is None) != (self.receivers is None):
            raise ValueError(
                "GraphsTuple: senders and receivers must both be set or both be None"
            )

        # --- node-index bounds check ---
        if self.nodes is not None and self.senders is not None:
            num_nodes = self.nodes.shape[0]
            if num_nodes > 0:
                s_min = int(self.senders.min().item()) if self.senders.numel() > 0 else 0
                s_max = int(self.senders.max().item()) if self.senders.numel() > 0 else 0
                r_min = int(self.receivers.min().item()) if self.receivers.numel() > 0 else 0
                r_max = int(self.receivers.max().item()) if self.receivers.numel() > 0 else 0
                if s_min < 0 or s_max >= num_nodes:
                    raise ValueError(
                        f"GraphsTuple: senders out of bounds — values in "
                        f"[{s_min}, {s_max}] but num_nodes={num_nodes}"
                    )
                if r_min < 0 or r_max >= num_nodes:
                    raise ValueError(
                        f"GraphsTuple: receivers out of bounds — values in "
                        f"[{r_min}, {r_max}] but num_nodes={num_nodes}"
                    )

        # --- positions shape ---
        if self.positions is not None and self.nodes is not None:
            if self.positions.shape[0] != self.nodes.shape[0]:
                raise ValueError(
                    f"GraphsTuple: positions.shape[0]={self.positions.shape[0]} != "
                    f"nodes.shape[0]={self.nodes.shape[0]}"
                )

        # --- device consistency ---
        tensors = {
            "nodes": self.nodes,
            "edges": self.edges,
            "receivers": self.receivers,
            "senders": self.senders,
            "globals": self.globals,
            "n_node": self.n_node,
            "n_edge": self.n_edge,
            "positions": self.positions,
        }
        devices = {
            name: t.device for name, t in tensors.items() if t is not None
        }
        unique_devices = set(devices.values())
        if len(unique_devices) > 1:
            details = ", ".join(f"{n}={d}" for n, d in devices.items())
            raise ValueError(
                f"GraphsTuple: tensors on multiple devices — {details}"
            )

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
