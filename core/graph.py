"""
GraphsTuple: Minimal graph representation.

Based on DeepMind's Graph Nets library but simplified.

Architecture
------------
graph.topology ──▶ GraphTopology  (immutable graph structure)
                   • n_node, senders, receivers, n_edge, positions
graph.nodes    ──▶ learned node features
graph.edges    ──▶ learned edge features (optional)
graph.globals  ──▶ global features (optional)

Constructing graphs
-------------------
Canonical form (recommended for new code)::

    graph = GraphsTuple(
        nodes=node_feats,
        topology=GraphTopology(n_node=n_node, senders=senders, receivers=receivers, n_edge=n_edge),
        edges=edge_feats,
    )

Flat form (migration / convenience)::

    graph = GraphsTuple.from_flat(
        nodes=node_feats,
        n_node=n_node,
        edges=edge_feats,
        senders=senders,
        receivers=receivers,
        n_edge=n_edge,
    )

Updating graphs
---------------
Feature update (nodes, edges, globals)::

    graph = graph.replace(nodes=new_nodes)

Topology swap (after pooling/unpooling)::

    graph = graph.with_topology(encoder_outputs[i])   # copy from another graph
    graph = graph.with_topology(                       # explicit fields
        senders=self_loop_idx,
        receivers=self_loop_idx,
        n_edge=torch.tensor([n]),
        edges=self_loop_feats,
    )
"""

from dataclasses import dataclass, replace
from typing import Optional, List, Union
import torch
from torch import Tensor

# Set to True to enable fail-fast validation in __post_init__.
# Useful for debugging; keep False in production to avoid overhead from
# dataclasses.replace() calls inside processor forward passes.
VALIDATE_GRAPHS: bool = False

# Topology field identifiers blocked from GraphsTuple.replace().
# Includes both the nested field name and the flat backward-compat property names.
_TOPOLOGY_GUARD: frozenset = frozenset(
    {"topology", "n_node", "senders", "receivers", "n_edge", "positions"}
)


@dataclass(frozen=True, eq=False)
class GraphTopology:
    """Immutable structural description of a batched graph.

    All index and geometry information lives here; learned feature tensors live
    on :class:`GraphsTuple`.

    Fields
    ------
    n_node : Tensor, shape [batch_size]
        Number of nodes per graph.  Always required.
    senders : Optional Tensor, shape [total_edges]
        Source node index for each edge.
    receivers : Optional Tensor, shape [total_edges]
        Destination node index for each edge.
    n_edge : Optional Tensor, shape [batch_size]
        Number of edges per graph.
    positions : Optional Tensor, shape [total_nodes, n_dim]
        Node positions in physical space (mesh coordinates, query points, …).
    """

    n_node: Tensor
    senders: Optional[Tensor] = None
    receivers: Optional[Tensor] = None
    n_edge: Optional[Tensor] = None
    positions: Optional[Tensor] = None

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch."""
        return len(self.n_node)

    @property
    def device(self) -> torch.device:
        return self.n_node.device

    def to(self, device) -> 'GraphTopology':
        """Move all tensors to *device*."""
        return GraphTopology(
            n_node=self.n_node.to(device),
            senders=self.senders.to(device) if self.senders is not None else None,
            receivers=self.receivers.to(device) if self.receivers is not None else None,
            n_edge=self.n_edge.to(device) if self.n_edge is not None else None,
            positions=self.positions.to(device) if self.positions is not None else None,
        )

    def validate(self, nodes: Tensor) -> None:
        """Validate topology consistency against a node feature tensor.

        Raises:
            ValueError: on any consistency violation.
        """
        expected = int(self.n_node.sum().item())
        actual = nodes.shape[0]
        if expected != actual:
            raise ValueError(
                f"GraphTopology: n_node sums to {expected} but "
                f"nodes.shape[0] == {actual}"
            )

        if (self.senders is None) != (self.receivers is None):
            raise ValueError(
                "GraphTopology: senders and receivers must both be set or both be None"
            )

        if self.senders is not None and self.receivers is not None:
            if self.senders.shape != self.receivers.shape:
                raise ValueError(
                    f"GraphTopology: senders.shape {self.senders.shape} != "
                    f"receivers.shape {self.receivers.shape}"
                )
            n = nodes.shape[0]
            if n > 0 and self.senders.numel() > 0:
                s_min = int(self.senders.min().item())
                s_max = int(self.senders.max().item())
                r_min = int(self.receivers.min().item())
                r_max = int(self.receivers.max().item())
                if s_min < 0 or s_max >= n:
                    raise ValueError(
                        f"GraphTopology: senders out of bounds — [{s_min}, {s_max}] "
                        f"but num_nodes={n}"
                    )
                if r_min < 0 or r_max >= n:
                    raise ValueError(
                        f"GraphTopology: receivers out of bounds — [{r_min}, {r_max}] "
                        f"but num_nodes={n}"
                    )

        if self.n_edge is not None and self.senders is not None:
            expected_e = int(self.n_edge.sum().item())
            actual_e = self.senders.shape[0]
            if expected_e != actual_e:
                raise ValueError(
                    f"GraphTopology: n_edge sums to {expected_e} but "
                    f"senders.shape[0] == {actual_e}"
                )

        if self.positions is not None and self.positions.shape[0] != nodes.shape[0]:
            raise ValueError(
                f"GraphTopology: positions.shape[0]={self.positions.shape[0]} != "
                f"nodes.shape[0]={nodes.shape[0]}"
            )


@dataclass(frozen=True, eq=False)
class GraphsTuple:
    """Minimal batched graph representation.

    Fields
    ------
    nodes : Tensor, shape [total_nodes, node_feat_dim]
        Node feature matrix.
    topology : GraphTopology
        Immutable structural information (node counts, edge indices, positions).
    edges : Optional Tensor, shape [total_edges, edge_feat_dim]
        Edge feature matrix.  ``None`` for edgeless / point-cloud graphs.
    globals : Optional Tensor, shape [batch_size, global_feat_dim]
        Per-graph global features.

    Backward-compat properties
    --------------------------
    ``n_node``, ``senders``, ``receivers``, ``n_edge``, ``positions`` are
    delegated to ``self.topology`` so code written before Phase 2 continues to
    work without changes.
    """

    # ── Core fields ──────────────────────────────────────────────────────────
    nodes: Tensor
    topology: GraphTopology

    # ── Optional learned features ─────────────────────────────────────────────
    edges: Optional[Tensor] = None
    globals: Optional[Tensor] = None

    # ── Backward-compat properties ────────────────────────────────────────────
    @property
    def n_node(self) -> Tensor:
        return self.topology.n_node

    @property
    def senders(self) -> Optional[Tensor]:
        return self.topology.senders

    @property
    def receivers(self) -> Optional[Tensor]:
        return self.topology.receivers

    @property
    def n_edge(self) -> Optional[Tensor]:
        return self.topology.n_edge

    @property
    def positions(self) -> Optional[Tensor]:
        return self.topology.positions

    # ── Constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_flat(
        cls,
        nodes: Tensor,
        n_node: Tensor,
        edges: Optional[Tensor] = None,
        senders: Optional[Tensor] = None,
        receivers: Optional[Tensor] = None,
        n_edge: Optional[Tensor] = None,
        globals: Optional[Tensor] = None,  # noqa: A002
        positions: Optional[Tensor] = None,
    ) -> 'GraphsTuple':
        """Construct from flat kwargs — backward-compat / migration form.

        Equivalent to::

            GraphsTuple(
                nodes=nodes,
                topology=GraphTopology(n_node=n_node, senders=senders, ...),
                edges=edges,
                globals=globals,
            )
        """
        return cls(
            nodes=nodes,
            topology=GraphTopology(
                n_node=n_node,
                senders=senders,
                receivers=receivers,
                n_edge=n_edge,
                positions=positions,
            ),
            edges=edges,
            globals=globals,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Optionally validate on construction.

        Enable with ``core.graph.VALIDATE_GRAPHS = True`` or call
        :meth:`validate` explicitly.
        """
        if VALIDATE_GRAPHS:
            self.validate()

    def validate(self) -> None:
        """Validate internal consistency.

        Delegates structural checks to :meth:`GraphTopology.validate` and
        additionally checks:

        - ``edges`` shape matches ``topology.n_edge``.
        - All tensors on the same device.

        Raises:
            ValueError: on any consistency violation.
        """
        self.topology.validate(self.nodes)

        if self.topology.n_edge is not None and self.edges is not None:
            expected = int(self.topology.n_edge.sum().item())
            actual = self.edges.shape[0]
            if expected != actual:
                raise ValueError(
                    f"GraphsTuple: n_edge sums to {expected} but "
                    f"edges.shape[0] == {actual}"
                )

        tensors = {
            "nodes": self.nodes,
            "topology.n_node": self.topology.n_node,
            "topology.senders": self.topology.senders,
            "topology.receivers": self.topology.receivers,
            "topology.n_edge": self.topology.n_edge,
            "topology.positions": self.topology.positions,
            "edges": self.edges,
            "globals": self.globals,
        }
        devices = {k: t.device for k, t in tensors.items() if t is not None}
        unique = set(devices.values())
        if len(unique) > 1:
            detail = ", ".join(f"{k}={d}" for k, d in devices.items())
            raise ValueError(f"GraphsTuple: tensors on multiple devices — {detail}")

    # ── Device movement ───────────────────────────────────────────────────────

    def to(self, device) -> 'GraphsTuple':
        """Move all tensors to *device*."""
        return GraphsTuple(
            nodes=self.nodes.to(device),
            topology=self.topology.to(device),
            edges=self.edges.to(device) if self.edges is not None else None,
            globals=self.globals.to(device) if self.globals is not None else None,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch."""
        return self.topology.num_graphs

    @property
    def num_nodes(self) -> int:
        """Total number of nodes across all graphs."""
        return self.nodes.shape[0]

    @property
    def num_edges(self) -> int:
        """Total number of edges across all graphs."""
        return self.edges.shape[0] if self.edges is not None else 0

    @property
    def device(self) -> torch.device:
        """Device of the nodes tensor."""
        return self.nodes.device

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def replace(self, **kwargs) -> 'GraphsTuple':
        """Return a new GraphsTuple with *feature* fields replaced.

        Allowed fields: ``nodes``, ``edges``, ``globals``.

        Topology fields are blocked — use :meth:`with_topology` instead.

        Raises:
            TypeError: if any topology field name is passed.
        """
        bad = set(kwargs) & _TOPOLOGY_GUARD
        if bad:
            raise TypeError(
                f"GraphsTuple.replace() cannot update topology fields "
                f"{sorted(bad)!r}.  Use with_topology() to swap topology."
            )
        return replace(self, **kwargs)

    def with_topology(
        self,
        source: Optional[Union['GraphTopology', 'GraphsTuple']] = None,
        *,
        senders: Optional[Tensor] = None,
        receivers: Optional[Tensor] = None,
        n_edge: Optional[Tensor] = None,
        edges: Optional[Tensor] = None,
    ) -> 'GraphsTuple':
        """Return a new GraphsTuple with topology (and optionally edges) replaced.

        Two calling styles:

        **Copy from another graph** (U-Net / MGKN upward-pass pattern)::

            # Copies topology + edges from a GraphsTuple source
            x = x.with_topology(encoder_outputs[i])
            # Copies topology only from a GraphTopology source
            x = x.with_topology(some_topology_object)

        **Explicit fields** (e.g. synthetic self-loops)::

            x = x.with_topology(
                senders=self_idx,
                receivers=self_idx,
                n_edge=torch.tensor([n], device=device),
                edges=self_loop_feats,
            )

        Args:
            source: :class:`GraphsTuple` (copies topology + edges) or
                :class:`GraphTopology` (copies topology only).
            senders: New senders tensor (explicit form only).
            receivers: New receivers tensor (explicit form only).
            n_edge: New n_edge tensor (explicit form only).
            edges: New edge features (explicit form only).
        """
        if source is not None:
            if isinstance(source, GraphsTuple):
                return replace(self, topology=source.topology, edges=source.edges)
            if isinstance(source, GraphTopology):
                return replace(self, topology=source)
            raise TypeError(
                f"with_topology() source must be GraphsTuple or GraphTopology, "
                f"got {type(source).__name__}"
            )
        # Explicit kwargs: update only the fields that were provided.
        topo_updates: dict = {}
        if senders is not None:
            topo_updates["senders"] = senders
        if receivers is not None:
            topo_updates["receivers"] = receivers
        if n_edge is not None:
            topo_updates["n_edge"] = n_edge
        new_topo = replace(self.topology, **topo_updates)
        new_edges = edges if edges is not None else self.edges
        return replace(self, topology=new_topo, edges=new_edges)


# ── Batching utilities ────────────────────────────────────────────────────────

def batch_graphs(graphs: List[GraphsTuple]) -> GraphsTuple:
    """Batch a list of :class:`GraphsTuple` instances into one.

    Node offsets in ``senders`` / ``receivers`` are adjusted automatically.

    Args:
        graphs: Non-empty list of GraphsTuples to batch.

    Returns:
        A single batched GraphsTuple.
    """
    if not graphs:
        raise ValueError("Cannot batch empty list of graphs")

    device = graphs[0].device

    all_nodes: List[Tensor] = []
    all_edges: List[Tensor] = []
    all_senders: List[Tensor] = []
    all_receivers: List[Tensor] = []
    all_globals: List[Tensor] = []
    all_positions: List[Tensor] = []
    n_nodes: List[int] = []
    n_edges: List[int] = []

    node_offset = 0

    for g in graphs:
        all_nodes.append(g.nodes)
        n_nodes.append(g.nodes.shape[0])

        if g.edges is not None:
            all_edges.append(g.edges)
            n_edges.append(g.edges.shape[0])
        else:
            n_edges.append(0)

        if g.topology.senders is not None:
            all_senders.append(g.topology.senders + node_offset)
            all_receivers.append(g.topology.receivers + node_offset)

        if g.globals is not None:
            all_globals.append(g.globals)

        if g.topology.positions is not None:
            all_positions.append(g.topology.positions)

        node_offset += g.nodes.shape[0]

    has_edges = bool(all_edges)
    return GraphsTuple(
        nodes=torch.cat(all_nodes, dim=0),
        topology=GraphTopology(
            n_node=torch.tensor(n_nodes, dtype=torch.long, device=device),
            senders=torch.cat(all_senders, dim=0) if all_senders else None,
            receivers=torch.cat(all_receivers, dim=0) if all_receivers else None,
            n_edge=torch.tensor(n_edges, dtype=torch.long, device=device) if has_edges else None,
            positions=torch.cat(all_positions, dim=0) if all_positions else None,
        ),
        edges=torch.cat(all_edges, dim=0) if has_edges else None,
        globals=torch.cat(all_globals, dim=0) if all_globals else None,
    )


def unbatch_graphs(graph: GraphsTuple) -> List[GraphsTuple]:
    """Split a batched :class:`GraphsTuple` into individual graphs.

    Args:
        graph: Batched GraphsTuple.

    Returns:
        List of individual GraphsTuples.
    """
    topo = graph.topology
    num_graphs = topo.num_graphs
    n_nodes = topo.n_node.cpu().tolist()
    n_edges = topo.n_edge.cpu().tolist() if topo.n_edge is not None else [0] * num_graphs

    result: List[GraphsTuple] = []
    node_start = 0
    edge_start = 0

    for i in range(num_graphs):
        n_node = int(n_nodes[i])
        n_edge = int(n_edges[i])

        nodes_i = graph.nodes[node_start:node_start + n_node]
        edges_i = graph.edges[edge_start:edge_start + n_edge] if graph.edges is not None else None
        globs_i = graph.globals[i:i + 1] if graph.globals is not None else None

        send_i = (
            topo.senders[edge_start:edge_start + n_edge] - node_start
            if topo.senders is not None else None
        )
        recv_i = (
            topo.receivers[edge_start:edge_start + n_edge] - node_start
            if topo.receivers is not None else None
        )
        pos_i = (
            topo.positions[node_start:node_start + n_node]
            if topo.positions is not None else None
        )
        n_edge_i = (
            torch.tensor([n_edge], device=graph.device)
            if topo.n_edge is not None else None
        )

        result.append(GraphsTuple(
            nodes=nodes_i,
            topology=GraphTopology(
                n_node=torch.tensor([n_node], device=graph.device),
                senders=send_i,
                receivers=recv_i,
                n_edge=n_edge_i,
                positions=pos_i,
            ),
            edges=edges_i,
            globals=globs_i,
        ))

        node_start += n_node
        edge_start += n_edge

    return result


