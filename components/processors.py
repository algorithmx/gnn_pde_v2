"""
GraphNet processors: DeepMind-style message passing.

Two independent block classes reflect the two distinct use cases:

- ``GraphNetBlock`` / ``GraphNetProcessor``:
    Node/edge-only message passing.  Use when all conditioning (PDE
    parameters, time, BCs) has already been encoded into per-node or
    per-edge features.

- ``GlobalGraphNetBlock`` / ``GlobalGraphNetProcessor``:
    Full DeepMind Graph Nets block with globals as a first-class
    participant.  Use when system-level state (Reynolds number, viscosity,
    simulation time, boundary-condition summary, …) must flow through a
    dedicated global channel rather than being copied into every node.

The two families are independent siblings; they are NOT related by
inheritance.  Both satisfy the ``GraphProcessor`` structural protocol
defined in ``core.protocols``.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..core.graph import GraphsTuple
from ..core.functional import aggregate_edges, broadcast_global, aggregate_to_global
from ..core.mlp import MLP


# ---------------------------------------------------------------------------
# Node/edge-only block
# ---------------------------------------------------------------------------

class GraphNetBlock(nn.Module):
    """
    Single node/edge message-passing step (no globals).

    Performs a complete 2-step update:

    1. **Edge update**: ``new_e_ij = MLP([v_i, v_j, e_ij])``
    2. **Node update**: ``new_v_i  = MLP([v_i, Σ_j new_e_ij])``

    Use this when all conditioning information (PDE parameters, time, BCs)
    has already been encoded into per-node or per-edge features before
    entering the processor.  If you need a dedicated global channel, use
    :class:`GlobalGraphNetBlock` instead.

    Args:
        latent_dim: Dimension for node, edge, and output features.  All
            three feature channels are assumed to have this dimension after
            encoding.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name (``'relu'``, ``'gelu'``,
            ``'silu'``, ``'tanh'``).
        aggregate_fn: Callable with signature
            ``(edge_features, receivers, num_nodes) -> aggregated``
            used to pool incoming edge messages to each node.  Defaults to
            ``functional.aggregate_edges`` (sum).  Override for mean/max
            pooling or attention-based aggregation.

    Example::

        block = GraphNetBlock(latent_dim=128)
        out_graph = block(graph)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        aggregate_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.aggregate_fn: Callable = aggregate_fn if aggregate_fn is not None else aggregate_edges

        # Edge update: [sender_node, receiver_node, edge] → new_edge
        self.edge_mlp = MLP(
            in_dim=3 * latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

        # Node update: [node, aggregated_edges] → new_node
        self.node_mlp = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Single node/edge message-passing step.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.
                ``nodes`` and ``edges`` must not be ``None``.

        Returns:
            Updated :class:`~gnn_pde_v2.core.GraphsTuple` with new node
            and edge features.  ``globals`` is passed through unchanged.
        """
        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        receivers = graph.receivers

        # --- Edge update ---
        edge_inputs = torch.cat(
            [nodes[senders], nodes[receivers], edges], dim=-1
        )
        new_edges = self.edge_mlp(edge_inputs)

        # --- Node update ---
        agg = self.aggregate_fn(new_edges, receivers, nodes.shape[0])
        node_inputs = torch.cat([nodes, agg], dim=-1)
        new_nodes = self.node_mlp(node_inputs)

        return graph.replace(nodes=new_nodes, edges=new_edges)


# ---------------------------------------------------------------------------
# Full Graph Nets block with globals
# ---------------------------------------------------------------------------

class GlobalGraphNetBlock(nn.Module):
    """
    Single full Graph Nets message-passing step with globals.

    Performs the complete 3-step update from DeepMind's Graph Nets library:

    1. **Edge update**: ``new_e_ij = MLP([v_i, v_j, e_ij, g])``
    2. **Node update**: ``new_v_i  = MLP([v_i, Σ_j new_e_ij, g])``
    3. **Global update**: ``new_g   = MLP([pool(new_v), pool(new_e), g])``

    Globals are first-class participants: they are broadcast *down* to
    every edge and node in steps 1–2, and aggregated *up* from nodes and
    edges in step 3, creating a complete bidirectional information loop.

    Use this when system-level state (Reynolds number, viscosity, simulation
    time, boundary-condition summary, …) must flow through the processor as
    a dedicated channel.  If no such global state exists, use the lighter
    :class:`GraphNetBlock` instead.

    Args:
        latent_dim: Dimension for node and edge features.
        global_latent_dim: Dimension for global features.  **Required** —
            not optional.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name.
        aggregate_fn: Edge-to-node aggregation callable
            ``(edge_features, receivers, num_nodes) -> aggregated``.
            Defaults to sum aggregation.
        global_pool: Pooling method used when aggregating nodes/edges back
            to global (``'mean'``, ``'sum'``, ``'max'``).  Defaults to
            ``'mean'``.

    Example::

        block = GlobalGraphNetBlock(latent_dim=128, global_latent_dim=32)
        out_graph = block(graph)   # graph.globals must not be None
    """

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: int,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        aggregate_fn: Optional[Callable] = None,
        global_pool: str = 'mean',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.global_latent_dim = global_latent_dim
        self.global_pool = global_pool
        self.aggregate_fn: Callable = aggregate_fn if aggregate_fn is not None else aggregate_edges

        # Edge update: [sender, receiver, edge, global] → new_edge
        self.edge_mlp = MLP(
            in_dim=3 * latent_dim + global_latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

        # Node update: [node, aggregated_edges, global] → new_node
        self.node_mlp = MLP(
            in_dim=2 * latent_dim + global_latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

        # Global update: [pooled_nodes, pooled_edges, global] → new_global
        self.global_mlp = MLP(
            in_dim=2 * latent_dim + global_latent_dim,
            out_dim=global_latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Single full message-passing step with globals.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.
                ``nodes``, ``edges``, ``globals``, ``n_node``, and
                ``n_edge`` must not be ``None``.

        Returns:
            Updated :class:`~gnn_pde_v2.core.GraphsTuple` with new node,
            edge, and global features.

        Raises:
            AssertionError: If ``graph.globals`` is ``None``.
        """
        assert graph.globals is not None, (
            "GlobalGraphNetBlock requires graph.globals to be a tensor. "
            "If your graph has no global state, use GraphNetBlock instead."
        )

        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        receivers = graph.receivers
        globals_ = graph.globals
        n_node = graph.n_node
        n_edge = graph.n_edge

        # --- Edge update ---
        g_e = broadcast_global(globals_, n_edge)
        edge_inputs = torch.cat(
            [nodes[senders], nodes[receivers], edges, g_e], dim=-1
        )
        new_edges = self.edge_mlp(edge_inputs)

        # --- Node update ---
        agg = self.aggregate_fn(new_edges, receivers, nodes.shape[0])
        g_n = broadcast_global(globals_, n_node)
        node_inputs = torch.cat([nodes, agg, g_n], dim=-1)
        new_nodes = self.node_mlp(node_inputs)

        # --- Global update ---
        pooled_nodes = aggregate_to_global(new_nodes, n_node, method=self.global_pool)
        pooled_edges = aggregate_to_global(new_edges, n_edge, method=self.global_pool)
        global_inputs = torch.cat([pooled_nodes, pooled_edges, globals_], dim=-1)
        new_globals = self.global_mlp(global_inputs)

        return graph.replace(nodes=new_nodes, edges=new_edges, globals=new_globals)


# ---------------------------------------------------------------------------
# Processors (multi-layer stacks)
# ---------------------------------------------------------------------------

class GraphNetProcessor(nn.Module):
    """
    Multi-layer node/edge-only GraphNet processor.

    Stacks multiple :class:`GraphNetBlock` instances with optional residual
    connections.  No global state is maintained or expected.

    Args:
        latent_dim: Node and edge feature dimension.
        n_layers: Number of :class:`GraphNetBlock` layers.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name.
        residual: Whether to add residual connections between blocks.
        aggregate_fn: Edge-to-node aggregation callable passed to each
            block.  Defaults to sum aggregation.

    Example::

        processor = GraphNetProcessor(latent_dim=128, n_layers=15)
        out_graph = processor(graph)
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 15,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
        aggregate_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.residual = residual
        self.blocks = nn.ModuleList([
            GraphNetBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                aggregate_fn=aggregate_fn,
            )
            for _ in range(n_layers)
        ])

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all :class:`GraphNetBlock` layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        for block in self.blocks:
            new_graph = block(graph)
            if self.residual:
                new_graph = new_graph.replace(
                    nodes=graph.nodes + new_graph.nodes,
                    edges=graph.edges + new_graph.edges,
                )
            graph = new_graph
        return graph


class GlobalGraphNetProcessor(nn.Module):
    """
    Multi-layer full Graph Nets processor with globals.

    Stacks multiple :class:`GlobalGraphNetBlock` instances with optional
    residual connections.  All three feature channels — nodes, edges, and
    globals — are updated at every layer.

    Args:
        latent_dim: Node and edge feature dimension.
        global_latent_dim: Global feature dimension.  **Required.**
        n_layers: Number of :class:`GlobalGraphNetBlock` layers.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name.
        residual: Whether to add residual connections between blocks.
        aggregate_fn: Edge-to-node aggregation callable.
        global_pool: Pooling method for node/edge → global aggregation.

    Example::

        processor = GlobalGraphNetProcessor(
            latent_dim=128, global_latent_dim=32, n_layers=15
        )
        out_graph = processor(graph)   # graph.globals must not be None
    """

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: int,
        n_layers: int = 15,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
        aggregate_fn: Optional[Callable] = None,
        global_pool: str = 'mean',
    ):
        super().__init__()

        self.residual = residual
        self.blocks = nn.ModuleList([
            GlobalGraphNetBlock(
                latent_dim=latent_dim,
                global_latent_dim=global_latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                aggregate_fn=aggregate_fn,
                global_pool=global_pool,
            )
            for _ in range(n_layers)
        ])

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all :class:`GlobalGraphNetBlock` layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.
                ``graph.globals`` must not be ``None``.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        for block in self.blocks:
            new_graph = block(graph)
            if self.residual:
                new_graph = new_graph.replace(
                    nodes=graph.nodes + new_graph.nodes,
                    edges=graph.edges + new_graph.edges,
                    globals=graph.globals + new_graph.globals,
                )
            graph = new_graph
        return graph
