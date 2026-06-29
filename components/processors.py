"""
GraphNet processors: DeepMind-style and edge-conditioned message passing.

All processor blocks share a common ``GraphBlockBase`` interface:

- ``MessagePassingBase`` subclasses (``GraphNetBlock``,
  ``EdgeConditionedConvBlock``, ``EdgeConvBlock``, ``GENBlock``):
    Node/edge message passing without explicit global updates.

- ``GlobalGraphNetBlock``:
    Full DeepMind Graph Nets block with globals as a first-class
    participant.

The node/edge-only and global-aware variants still use different internal
update templates, but now live under a unified block hierarchy so processors
and higher-level code can reason about them consistently.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, final, Final, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.graph import GraphsTuple
from ..core.functional import aggregate_edges, broadcast_global, aggregate_to_global, scatter_softmax
from ..core.mlp import MLP
from ..core.aggregation import Aggregation, Sum, get_aggregation
from .edge_assemblers import NodeDifferenceAssembler, EdgeFeatureAssembler
from .edge_processors import (
    EdgeMessageProcessor,
    FullEdgeMessageProcessor,
    VectorEdgeMessageProcessor,
    ScalarEdgeMessageProcessor,
    LowRankEdgeMessageProcessor,
    _default_edge_message_processor,
)
from .node_updaters import (
    NodeUpdateStrategy,
    ConcatMLPNodeUpdater,
    RootWeightNodeUpdater,
    PassThroughNodeUpdater,
    ResidualMLPNodeUpdater,
    _default_node_updater,
)
from .node_updaters import (
    build_concat_mlp_node_updater,
    build_root_weight_node_updater,
    build_pass_through_node_updater,
    build_residual_mlp_node_updater,
)
from .processor_validators import (
    reset_linear_layers,
    validate_edge_message_processor,
    validate_node_update_strategy,
    verify_edge_message_pipeline,
    verify_edge_transform_output,
)


# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

__all__ = [
    # Base classes and protocols
    "GraphBlockBase",
    "MessagePassingBase",
    # GraphNet-style blocks
    "GraphNetBlock",
    "EdgeConditionedConvBlock",
    "EdgeConvBlock",
    "GENBlock",
    "GlobalGraphNetBlock",
    "GlobalGraphNetProcessor",
]


# Default width for internal MLPs across processor blocks; chosen as the
# project-wide baseline capacity for message/update networks.
DEFAULT_HIDDEN_DIM: Final[int] = 128
# Edge-update MLPs concatenate three latent-sized inputs
# [sender_node, receiver_node, edge_features].
EDGE_UPDATE_INPUT_PARTS: Final[int] = 3
# Eager shape checks only need a minimal non-trivial edge batch; two edges are
# enough to verify [E, ...] behavior without adding unnecessary overhead.
PIPELINE_VALIDATION_NUM_EDGES: Final[int] = 2


class GraphBlockBase(nn.Module, ABC):
    """Common runtime contract for graph processor blocks.

    All graph blocks consume a :class:`GraphsTuple` and return an updated
    :class:`GraphsTuple`, regardless of whether they update edges, globals, or
    only nodes. The class attributes expose these capabilities explicitly so
    processors do not need fragile reflective lookups.
    """

    updates_edges: bool = False
    updates_globals: bool = False

    @abstractmethod
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        ...


class MessagePassingBase(GraphBlockBase, ABC):
    """
    Abstract base class for graph message passing.
    
    FRAMEWORK CONSTRAINS (cannot be changed by subclasses):
    - Aggregation: MUST use Aggregation Protocol via self.aggregate_fn
    - Input: GraphsTuple with nodes, edges, senders, receivers
    - Output: GraphsTuple with updated nodes (and optionally edges)  
    - Template: compute_messages → _aggregate → update_nodes
    
    SUBCLASSES IMPLEMENT (abstract methods):
    - compute_messages(graph) → (messages, new_edges)
    - update_nodes(nodes, aggregated, graph) → new_nodes
    
    SUBCLASSES CAN CUSTOMIZE (but not required):
    - Edge feature source: graph.edges, node difference, or none
    - Message transformation: MLP, weighted, identity, etc.
    - Node update logic (with or without root weights, bias, etc.)
    
    The key insight is that edge feature computation is left to subclasses
    while aggregation is standardized by the framework.
    
    Example:
        block = GraphNetBlock(latent_dim=128)
        out_graph = block(in_graph)
    """

    # Whether this block produces updated edge features.
    updates_edges: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'forward' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override final method 'forward' from "
                f"MessagePassingBase. Implement compute_messages() and "
                f"supply a node_updater instead."
            )
        if 'update_nodes' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override 'update_nodes' from "
                f"MessagePassingBase. Supply a node_updater to __init__ instead."
            )

    def __init__(
        self,
        latent_dim: int,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        node_updater: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.aggregate_fn = get_aggregation(aggregate)
        
        # Node-update strategy (composable)
        if node_updater is not None:
            validate_node_update_strategy(node_updater, latent_dim)
            self.node_updater = node_updater
        else:
            self.node_updater = _default_node_updater(latent_dim)

    def _aggregate(
        self,
        messages: torch.Tensor,
        receivers: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        return self.aggregate_fn(messages, receivers, num_nodes)

    @abstractmethod
    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            messages: [E, msg_dim]
            new_edges: [E, edge_dim] or None to keep current edges
        """
        ...

    @final
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        if graph.senders is None:
            # Edgeless graph (e.g. after aggressive pooling strips all edges).
            # No messages to aggregate — return graph with only the node updater
            # applied to a zero-aggregation signal.
            zero_agg = torch.zeros_like(graph.nodes)
            new_nodes = self.node_updater(graph.nodes, zero_agg)
            return graph.replace(nodes=new_nodes)
        # Standard message passing flow
        # 1. Compute messages and optionally new edges
        messages, new_edges = self.compute_messages(graph)
        # 2. Aggregate messages to receiver nodes
        aggregated = self._aggregate(messages, graph.receivers, graph.nodes.shape[0])
        # 3. Update node features via composed node_updater
        new_nodes = self.node_updater(graph.nodes, aggregated)
        # 4. Return updated graph (with new edges if provided)
        return graph.replace(nodes=new_nodes, edges=new_edges) \
                    if new_edges is not None \
                    else graph.replace(nodes=new_nodes)


class GraphNetBlock(MessagePassingBase):
    """
    Single DeepMind GraphNets message-passing step.  Updates both edges and nodes.

    1. **Edge update**: ``new_e_ij = MLP([v_j; v_i; e_ij])``  — ``[E, 3D] → [E, D]``
    2. **Aggregation**: ``a_i = Σ_{j∈N(i)} new_e_ij``         — ``[E, D] → [N, D]``
    3. **Node update**: ``new_v_i = MLP([v_i; a_i])``          — ``[N, 2D] → [N, D]``

    All MLPs are pointwise (row-wise); graph topology enters only through
    gather (step 1) and scatter-aggregate (step 2).  Nodes, edges, and
    outputs share the same dimension D (``latent_dim``).

    Args:
        latent_dim: Common feature dimension D for nodes, edges, and outputs.
        hidden_dim: Hidden dimension for the two internal MLPs.
        activation: Activation function name.
        aggregate: Aggregation strategy (``'sum'``, ``'mean'``, ``'max'``, ``'min'``).
        node_updater: Optional custom node updater. If None, uses
            ConcatMLPNodeUpdater via factory.
    """

    updates_edges = True

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        activation: str = 'gelu',
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = build_concat_mlp_node_updater(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
            )
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            node_updater=node_updater,
        )

        # Edge update: [sender_node, receiver_node, edge] → new_edge
        self.edge_mlp = MLP(
            in_dim=EDGE_UPDATE_INPUT_PARTS * latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        nodes = graph.nodes
        senders = graph.senders
        receivers = graph.receivers

        # ``new_e_ij = MLP([v_j, v_i, e_ij])``  for every edge in parallel
        edge_inputs = torch.cat(
            [nodes[senders], nodes[receivers], graph.edges], dim=-1
        )
        new_edges = self.edge_mlp(edge_inputs)
        # Messages == new edges: aggregated for node update AND stored as updated edges
        return new_edges, new_edges


class EdgeConditionedConvBlock(MessagePassingBase):
    """Edge-conditioned message passing with pluggable edge transforms.

    The block first maps edge features to per-edge weights using
    ``edge_weight_net``, then delegates message formation to ``edge_processor``.
    Messages are aggregated to receiver nodes and optionally combined with a
    learned root projection and bias.

    By default, ``edge_processor`` is :class:`FullEdgeMessageProcessor`, which
    preserves the original full-rank behavior.

    Args:
        latent_dim: Node latent dimension.
        edge_latent_dim: Edge feature dimension consumed by ``edge_weight_net``.
            Used for eager pipeline verification at construction time.
        edge_weight_net: ``nn.Module`` mapping edge features
            ``[E, edge_latent_dim]`` to edge weights
            ``[E, edge_processor.weight_out_dim]``.  **Required** — the
            caller is responsible for building this network with the
            correct input/output dimensions.
        aggregate: Aggregation strategy for incoming messages.
        root_weight: Whether to add a learned root projection.
        bias: Whether to add a learned bias term.
        edge_processor: Edge-message processor module. Must satisfy
            :class:`~gnn_pde_v2.core.protocols.EdgeMessageProcessor`.
    """

    updates_edges = False

    def __init__(
        self,
        latent_dim: int,
        edge_latent_dim: int,
        edge_weight_net: nn.Module,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        root_weight: bool = True,
        bias: bool = True,
        edge_processor: Optional[EdgeMessageProcessor] = None,
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = build_root_weight_node_updater(
                latent_dim=latent_dim,
                root_weight=root_weight,
                bias=bias,
            )
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            node_updater=node_updater,
        )

        resolved_processor = (
            edge_processor
            if edge_processor is not None
            else _default_edge_message_processor(latent_dim)
        )
        validate_edge_message_processor(
            edge_processor=resolved_processor,
            latent_dim=latent_dim,
        )

        self.edge_processor = resolved_processor
        self.low_rank = getattr(resolved_processor, 'low_rank', 0)

        if not isinstance(edge_weight_net, nn.Module):
            raise TypeError(
                "edge_weight_net must be an nn.Module for proper parameter "
                "registration and torch.compile() support"
            )
        self.edge_weight_net = edge_weight_net
        verify_edge_message_pipeline(
            edge_weight_net=self.edge_weight_net,
            edge_processor=self.edge_processor,
            latent_dim=self.latent_dim,
            edge_latent_dim=edge_latent_dim,
            num_edges=PIPELINE_VALIDATION_NUM_EDGES,
        )

    def reset_parameters(self):
        reset_linear_layers(self.edge_weight_net)

    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_x = graph.nodes[graph.senders]  # shape [E, H]
        w = self.edge_weight_net(graph.edges)
        msg = self.edge_processor(src_x, w)
        return msg, None  # Keep edges unchanged


# ---------------------------------------------------------------------------
# EdgeConv block (DGCNN-style)
# ----------------------------------------------------------------------------

class EdgeConvBlock(MessagePassingBase):
    """EdgeConv-style message passing (DGCNN / Point Cloud Networks).

    ``m_ij = edge_transform(assembler(graph))`` → aggregate → node update.
    Edges are **not** updated. Default aggregation is **max**.

    This block uses pluggable edge feature assemblers for flexible edge
    feature construction:

    - ``NodeDifferenceAssembler`` (default): ``[v_i; v_j - v_i]`` — DGCNN style
    - ``ConcatAssembler``: ``[v_i; v_j]`` — simple concatenation
    - ``DifferenceOnlyAssembler``: ``v_j - v_i`` — difference only
    - ``ConcatWithEdgesAssembler``: ``[v_i; v_j - v_i; e_ij]`` — with edge attrs

    Args:
        latent_dim: Node feature dimension (``D``).
        hidden_dim: Hidden dim for default edge transform MLP
            (ignored if ``edge_transform`` is provided).
        aggregate: ``'sum'``, ``'mean'``, ``'max'`` (default), ``'min'``,
            or ``Aggregation``.
        activation: Activation for default edge transform MLP.
        edge_assembler: Edge feature assembler. If None, uses
            ``NodeDifferenceAssembler(latent_dim)`` (DGCNN default).
        edge_transform: Custom ``nn.Module`` mapping assembled edge features
            ``[E, assembler.out_dim]`` to messages ``[E, D]``. If None, uses
            a default MLP with ``hidden_dim`` and ``activation``.
        node_updater: Optional custom node update strategy. If None, uses
            ``PassThroughNodeUpdater`` (messages become new node features).

    Example::

        # DGCNN default (node difference)
        block = EdgeConvBlock(latent_dim=128)

        # Explicit assembler
        from gnn_pde_v2.components import NodeDifferenceAssembler
        block = EdgeConvBlock(
            latent_dim=128,
            edge_assembler=NodeDifferenceAssembler(128),
        )

        # With edge attributes
        from gnn_pde_v2.components import ConcatWithEdgesAssembler
        block = EdgeConvBlock(
            latent_dim=128,
            edge_assembler=ConcatWithEdgesAssembler(128, edge_dim=3),
        )

        # Custom transform
        from gnn_pde_v2.core import MLP
        block = EdgeConvBlock(
            latent_dim=128,
            edge_assembler=ConcatWithEdgesAssembler(128, edge_dim=3),
            edge_transform=MLP(259, 128, [128, 128], 'relu'),
        )
    """

    updates_edges = False

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'max',
        activation: str = 'relu',
        edge_assembler: Optional[EdgeFeatureAssembler] = None,
        edge_transform: Optional[nn.Module] = None,
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = build_pass_through_node_updater(latent_dim=latent_dim)
        
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            node_updater=node_updater,
        )

        # Edge assembler (default: NodeDifferenceAssembler for DGCNN compatibility)
        if edge_assembler is not None:
            self.edge_assembler = edge_assembler
        else:
            self.edge_assembler = NodeDifferenceAssembler(latent_dim)

        # Edge transform (default: MLP)
        if edge_transform is not None:
            self.edge_transform = edge_transform
        else:
            self.edge_transform = MLP(
                in_dim=self.edge_assembler.out_dim,
                out_dim=latent_dim,
                hidden_dims=[hidden_dim],
                activation=activation,
            )

        # Eager validation: ensure transform output matches latent_dim
        verify_edge_transform_output(
            edge_transform=self.edge_transform,
            input_dim=self.edge_assembler.out_dim,
            expected_dim=latent_dim,
            num_edges=PIPELINE_VALIDATION_NUM_EDGES,
        )

    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, None]:
        edge_features = self.edge_assembler(graph)
        messages = self.edge_transform(edge_features)
        return messages, None  # Don't update edges


# ---------------------------------------------------------------------------
# GEN Block (GEneralized aggregation Network)
# ---------------------------------------------------------------------------

class GENBlock(MessagePassingBase):
    """
    GEneralized aggregation Network block from Li et al. (2020).
    
    From "DeeperGCN: All You Need to Train Deeper GCNs" (http://arxiv.org/abs/2006.07739).
    Used in Wind-Farm-GNO for wake interaction modeling.
    
    Performs message passing with softmax aggregation and epsilon stability:
    
    1. **Message computation**: ``m_ij = ReLU(e_ij + h_j) + epsilon``
       - Adds edge features to sender node features
       - Applies ReLU and small epsilon for numerical stability
       
    2. **Softmax aggregation**: ``agg_i = sum_j softmax_j(m_ij) * m_ij``
       - Computes attention weights via softmax over messages
       - Aggregates weighted messages to receiver nodes
       
    3. **Node update**: ``h'_i = MLP(h_i + agg_i)``
       - Residual-style update: adds aggregated messages to original features
       - Transforms via MLP to produce new node features
    
    Unlike GraphNetBlock:
    - Does NOT update edge features (edges are static)
    - Uses softmax aggregation instead of sum/mean/max
    - Has epsilon stability term in message computation
    
    Args:
        latent_dim: Dimension for node and edge features
        hidden_dim: Hidden dimension for node update MLP
        num_mlp_layers: Number of layers in node update MLP
        activation: Activation function for MLP
        epsilon: Small constant for numerical stability (default: 1e-6)
        message_norm: Whether to apply message normalization from DeeperGCN
        
    Example::
    
        from gnn_pde_v2.components import GENBlock
        
        block = GENBlock(latent_dim=128, hidden_dim=128, num_mlp_layers=2)
        out_graph = block(graph)  # graph.edges are NOT updated
    """
    
    updates_edges = False  # GEN blocks do not update edge features
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_mlp_layers: int = 2,
        activation: str = 'relu',
        epsilon: float = 1e-6,
        message_norm: bool = False,
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = build_residual_mlp_node_updater(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_mlp_layers,
                activation=activation,
                message_norm=message_norm,
                epsilon=epsilon,
            )
        super().__init__(
            latent_dim=latent_dim,
            aggregate='sum',  # Softmax is applied before aggregation
            node_updater=node_updater,
        )
        self.epsilon = epsilon
    
    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute GEN messages with softmax weighting.
        
        Raw messages ``m_ij = ReLU(e_ij + h_j) + epsilon`` are weighted by
        per-receiver softmax attention before being returned, so the base
        class ``forward`` can aggregate them directly via sum.
        
        Returns:
            messages: [E, latent_dim] softmax-weighted messages
            new_edges: None (edges are not updated in GEN)
        """

        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        
        # Gather sender node features
        sender_features = nodes[senders]
        
        # Compute raw messages: m_ij = ReLU(e_ij + h_j) + epsilon
        messages = torch.relu(edges + sender_features) + self.epsilon
        
        # Softmax aggregation: weight messages before the base-class sum
        attention_weights = scatter_softmax(
            messages, graph.receivers, dim=0, dim_size=nodes.shape[0]
        )
        weighted_messages = attention_weights * messages
        
        return weighted_messages, None  # Edges are NOT updated


# ---------------------------------------------------------------------------
# Full Graph Nets block with globals
# ---------------------------------------------------------------------------

class GlobalGraphNetBlock(GraphBlockBase):
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
        aggregate: Aggregation strategy for node updates (``'sum'``, ``'mean'``, ``'max'``, ``'min'``).
        global_pool: Pooling method used when aggregating nodes/edges back
            to global (``'mean'``, ``'sum'``, ``'max'``).  Defaults to
            ``'mean'``.

    Example::

        block = GlobalGraphNetBlock(latent_dim=128, global_latent_dim=32)
        out_graph = block(graph)   # graph.globals must not be None
    """

    updates_edges = True
    updates_globals = True

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        activation: str = 'gelu',
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        global_pool: str = 'mean',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.global_latent_dim = global_latent_dim
        self.global_pool = global_pool
        
        # Normalize aggregation: support new aggregate param
        self.aggregate_fn = get_aggregation(aggregate)

        # Edge update: [sender, receiver, edge, global] → new_edge
        self.edge_mlp = MLP(
            in_dim=EDGE_UPDATE_INPUT_PARTS * latent_dim + global_latent_dim,
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


class _BlockProcessorBase(nn.Module, ABC):
    """Shared residual/checkpoint orchestration for processor stacks."""

    def __init__(self, *, residual: bool, use_checkpoint: bool):
        super().__init__()
        self.residual = residual
        self.use_checkpoint = use_checkpoint

    @abstractmethod
    def _make_block_step(
        self,
        graph: GraphsTuple,
        block: GraphBlockBase,
        layer_index: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        ...

    @abstractmethod
    def _step_inputs(self, graph: GraphsTuple) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def _replace_graph(
        self,
        graph: GraphsTuple,
        step_outputs: tuple[torch.Tensor, ...],
    ) -> GraphsTuple:
        ...

    def _apply_block(
        self,
        graph: GraphsTuple,
        block: GraphBlockBase,
        layer_index: int,
    ) -> GraphsTuple:
        step = self._make_block_step(graph, block, layer_index)
        step_inputs = self._step_inputs(graph)

        if self.use_checkpoint:
            step_outputs = checkpoint(step, *step_inputs, use_reentrant=False)
        else:
            step_outputs = step(*step_inputs)

        return self._replace_graph(graph, step_outputs)

    def _run_blocks(self, graph: GraphsTuple) -> GraphsTuple:
        for layer_index, block in enumerate(self.blocks):
            graph = self._apply_block(graph, block, layer_index)
        return graph


class GraphNetProcessor(_BlockProcessorBase):
    """
    Multi-layer node/edge-only GraphNet processor.

    Stacks multiple :class:`GraphNetBlock` instances with optional residual
    connections.  No global state is maintained or expected.

    Uses **pre-norm residual connections** for numerical stability in deep
    networks: LayerNorm is applied *before* each block, and the residual
    adds the block's output to the normalized input.

    Args:
        latent_dim: Node and edge feature dimension.
        n_layers: Number of :class:`GraphNetBlock` layers.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name.
        residual: Whether to add residual connections between blocks.
            When True, uses pre-norm residuals for stability.
        aggregate: Aggregation strategy passed to each block.
            Options: ``'sum'``, ``'mean'``, ``'max'``, ``'min'``,
            or an Aggregation instance.  Defaults to ``'sum'``.
        use_checkpoint: If ``True``, applies gradient checkpointing to
            each block during the forward pass.  Trades compute for
            memory — each block's activations are recomputed during the
            backward pass instead of being stored.  Requires PyTorch
            2.0+ (``use_reentrant=False`` convention).

    Example::

        processor = GraphNetProcessor(latent_dim=128, n_layers=15,
                                      use_checkpoint=True)
        out_graph = processor(graph)
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 15,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        activation: str = 'gelu',
        residual: bool = True,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        use_checkpoint: bool = False,
        block_factory: Optional[Callable[[], GraphBlockBase]] = None,
    ):
        super().__init__(residual=residual, use_checkpoint=use_checkpoint)

        if block_factory is not None:
            self.blocks = nn.ModuleList([block_factory() for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([
                GraphNetBlock(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    aggregate=aggregate,
                )
                for _ in range(n_layers)
            ])

        # Pre-norm layers for numerical stability in deep networks
        if self.residual:
            self.node_norms = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(n_layers)
            ])
            self.edge_norms = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(n_layers)
            ])

    @staticmethod
    def _graph_block_step(
        nodes: torch.Tensor,
        edges: torch.Tensor,
        *,
        block: GraphBlockBase,
        base_graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        block_output = block(base_graph.replace(nodes=nodes, edges=edges))
        return block_output.nodes, block_output.edges

    @staticmethod
    def _graph_block_residual_step(
        nodes: torch.Tensor,
        edges: torch.Tensor,
        *,
        block: GraphBlockBase,
        base_graph: GraphsTuple,
        node_norm: nn.Module,
        edge_norm: nn.Module,
        updates_edges: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_edges = edge_norm(edges) if updates_edges else edges
        normalized_graph = base_graph.replace(
            nodes=node_norm(nodes),
            edges=normalized_edges,
        )
        block_output = block(normalized_graph)
        next_nodes = nodes + block_output.nodes
        next_edges = edges + block_output.edges if updates_edges else edges
        return next_nodes, next_edges

    def _make_block_step(
        self,
        graph: GraphsTuple,
        block: GraphBlockBase,
        layer_index: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        if self.residual:
            return partial(
                self._graph_block_residual_step,
                block=block,
                base_graph=graph,
                node_norm=self.node_norms[layer_index],
                edge_norm=self.edge_norms[layer_index],
                updates_edges=block.updates_edges,
            )
        return partial(
            self._graph_block_step,
            block=block,
            base_graph=graph,
        )

    def _step_inputs(self, graph: GraphsTuple) -> tuple[torch.Tensor, ...]:
        return graph.nodes, graph.edges

    def _replace_graph(
        self,
        graph: GraphsTuple,
        step_outputs: tuple[torch.Tensor, ...],
    ) -> GraphsTuple:
        new_nodes, new_edges = step_outputs
        return graph.replace(nodes=new_nodes, edges=new_edges)

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all message-passing layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        return self._run_blocks(graph)


class GlobalGraphNetProcessor(_BlockProcessorBase):
    """
    Multi-layer full Graph Nets processor with globals.

    Stacks multiple :class:`GlobalGraphNetBlock` instances with optional
    residual connections.  All three feature channels — nodes, edges, and
    globals — are updated at every layer.

    Uses **pre-norm residual connections** for numerical stability in deep
    networks: LayerNorm is applied *before* each block, and the residual
    adds the block's output to the normalized input.

    Args:
        latent_dim: Node and edge feature dimension.
        global_latent_dim: Global feature dimension.  **Required.**
        n_layers: Number of :class:`GlobalGraphNetBlock` layers.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name.
        residual: Whether to add residual connections between blocks.
            When True, uses pre-norm residuals for stability.
        global_pool: Pooling method for node/edge → global aggregation.
        use_checkpoint: If ``True``, applies gradient checkpointing to
            each block.  Reduces peak memory by recomputing activations
            during the backward pass.  Requires PyTorch 2.0+.

    Example::

        processor = GlobalGraphNetProcessor(
            latent_dim=128, global_latent_dim=32, n_layers=15,
            use_checkpoint=True
        )
        out_graph = processor(graph)   # graph.globals must not be None
    """

    def __init__(
        self,
        latent_dim: int,
        global_latent_dim: int,
        n_layers: int = 15,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        activation: str = 'gelu',
        residual: bool = True,
        global_pool: str = 'mean',
        use_checkpoint: bool = False,
    ):
        super().__init__(residual=residual, use_checkpoint=use_checkpoint)
        self.blocks = nn.ModuleList([
            GlobalGraphNetBlock(
                latent_dim=latent_dim,
                global_latent_dim=global_latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                global_pool=global_pool,
            )
            for _ in range(n_layers)
        ])

        # Pre-norm layers for numerical stability in deep networks
        if self.residual:
            self.node_norms = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(n_layers)
            ])
            self.edge_norms = nn.ModuleList([
                nn.LayerNorm(latent_dim) for _ in range(n_layers)
            ])
            self.global_norms = nn.ModuleList([
                nn.LayerNorm(global_latent_dim) for _ in range(n_layers)
            ])

    @staticmethod
    def _global_block_step(
        nodes: torch.Tensor,
        edges: torch.Tensor,
        globals_: torch.Tensor,
        *,
        block: GraphBlockBase,
        base_graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_output = block(
            base_graph.replace(nodes=nodes, edges=edges, globals=globals_)
        )
        return block_output.nodes, block_output.edges, block_output.globals

    @staticmethod
    def _global_block_residual_step(
        nodes: torch.Tensor,
        edges: torch.Tensor,
        globals_: torch.Tensor,
        *,
        block: GraphBlockBase,
        base_graph: GraphsTuple,
        node_norm: nn.Module,
        edge_norm: nn.Module,
        global_norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_graph = base_graph.replace(
            nodes=node_norm(nodes),
            edges=edge_norm(edges),
            globals=global_norm(globals_),
        )
        block_output = block(normalized_graph)
        return (
            nodes + block_output.nodes,
            edges + block_output.edges,
            globals_ + block_output.globals,
        )

    def _make_block_step(
        self,
        graph: GraphsTuple,
        block: GraphBlockBase,
        layer_index: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        if self.residual:
            return partial(
                self._global_block_residual_step,
                block=block,
                base_graph=graph,
                node_norm=self.node_norms[layer_index],
                edge_norm=self.edge_norms[layer_index],
                global_norm=self.global_norms[layer_index],
            )
        return partial(
            self._global_block_step,
            block=block,
            base_graph=graph,
        )

    def _step_inputs(self, graph: GraphsTuple) -> tuple[torch.Tensor, ...]:
        assert graph.globals is not None, (
            "GlobalGraphNetProcessor requires graph.globals to be a tensor. "
            "If your graph has no global state, use GraphNetProcessor instead."
        )
        return graph.nodes, graph.edges, graph.globals

    def _replace_graph(
        self,
        graph: GraphsTuple,
        step_outputs: tuple[torch.Tensor, ...],
    ) -> GraphsTuple:
        new_nodes, new_edges, new_globals = step_outputs
        return graph.replace(
            nodes=new_nodes,
            edges=new_edges,
            globals=new_globals,
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all :class:`GlobalGraphNetBlock` layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.
                ``graph.globals`` must not be ``None``.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        return self._run_blocks(graph)


