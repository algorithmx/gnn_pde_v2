"""
GraphNet processors: DeepMind-style and edge-conditioned message passing.

Two independent block families reflect distinct use cases:

- ``MessagePassingBase`` subclasses (``GraphNetBlock``,
  ``EdgeConditionedConvBlock``):
    Node/edge message passing without explicit globals.

- ``GlobalGraphNetBlock`` / ``GlobalGraphNetProcessor``:
    Full DeepMind Graph Nets block with globals as a first-class
    participant.

``GlobalGraphNetBlock`` is intentionally not derived from
``MessagePassingBase`` because it performs a 3-step update
(edge → node → global) rather than the base 2-step template.
"""

from abc import ABC, abstractmethod
from typing import Callable, final, Final, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.graph import GraphsTuple
from ..core.functional import aggregate_edges, broadcast_global, aggregate_to_global, scatter_softmax
from ..core.mlp import MLP
from ..core.aggregation import Aggregation, Sum, get_aggregation
from ..core.protocols import EdgeMessageProcessor
from .edge_processors import (
    _EdgeMessageProcessorBase,
    FullEdgeMessageProcessor,
    VectorEdgeMessageProcessor,
    ScalarEdgeMessageProcessor,
    LowRankEdgeMessageProcessor,
    _default_edge_message_processor,
)
from .node_updaters import (
    _NodeUpdaterBase,
    ConcatMLPNodeUpdater,
    RootWeightNodeUpdater,
    PassThroughNodeUpdater,
    ResidualMLPNodeUpdater,
    _default_node_updater,
)
from .node_updaters import (
    concat_mlp_factory,
    root_weight_factory,
    pass_through_factory,
    residual_mlp_factory,
)


# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

__all__ = [
    # Base classes and protocols
    "MessagePassingBase",
    # GraphNet-style blocks
    "GraphNetBlock",
    "EdgeConditionedConvBlock",
    "EdgeConvBlock",
    "GENBlock",
    "GlobalGraphNetBlock",
    "GlobalGraphNetProcessor",
]

class MessagePassingBase(ABC, nn.Module):
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
        hidden_dim: int = 128,
        activation: str = 'gelu',
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = concat_mlp_factory(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                activation=activation,
            )()
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            node_updater=node_updater,
        )

        # Edge update: [sender_node, receiver_node, edge] → new_edge
        self.edge_mlp = MLP(
            in_dim=3 * latent_dim,
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
            node_updater = root_weight_factory(
                latent_dim=latent_dim,
                root_weight=root_weight,
                bias=bias,
            )()
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
        self._validate_edge_processor(
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
        self._verify_edge_message_pipeline(edge_latent_dim=edge_latent_dim)

    @staticmethod
    def _validate_edge_processor(
        edge_processor: EdgeMessageProcessor,
        latent_dim: int,
    ) -> int:
        if not isinstance(edge_processor, nn.Module):
            raise TypeError(
                "edge_processor must be an nn.Module to preserve parameter registration "
                "and torch.compile() specialization"
            )
        if not isinstance(edge_processor, EdgeMessageProcessor):
            raise TypeError(
                "edge_processor must satisfy EdgeMessageProcessor protocol: "
                "provide weight_out_dim and forward(src_x, edge_weights)"
            )
        weight_out_dim = edge_processor.weight_out_dim
        if not isinstance(weight_out_dim, int) or weight_out_dim <= 0:
            raise ValueError(
                f"edge_processor.weight_out_dim must be a positive int, got {weight_out_dim!r}"
            )
        processor_latent_dim = edge_processor.latent_dim
        if processor_latent_dim != latent_dim:
            raise ValueError(
                "edge_processor latent_dim must match block latent_dim: "
                f"got {processor_latent_dim} vs {latent_dim}"
            )
        return weight_out_dim

    def _verify_edge_message_pipeline(self, edge_latent_dim: int, num_edges: int = 2) -> None:
        """Eagerly verify the full edge-weight-net → edge-processor pipeline."""
        if num_edges <= 0:
            raise ValueError(f"num_edges must be positive, got {num_edges}")

        tensor_kwargs = self._example_tensor_kwargs()
        edge_features = torch.randn(num_edges, edge_latent_dim, **tensor_kwargs)
        src_x = torch.randn(num_edges, self.latent_dim, **tensor_kwargs)

        with torch.no_grad():
            edge_weights = self.edge_weight_net(edge_features)
            out = self.edge_processor(src_x, edge_weights)

        if edge_weights.ndim != 2:
            raise ValueError(
                "edge_weight_net must return rank-2 tensor [E, weight_out_dim] during verification"
            )
        if edge_weights.shape != (num_edges, self.edge_processor.weight_out_dim):
            raise ValueError(
                "edge_weight_net and edge_processor disagree on weight shape during verification: "
                f"got {tuple(edge_weights.shape)} vs ({num_edges}, {self.edge_processor.weight_out_dim})"
            )
        if out.ndim != 2 or out.shape != (num_edges, self.latent_dim):
            raise ValueError(
                "edge message pipeline must return shape [E, latent_dim] during verification: "
                f"got {tuple(out.shape)} vs ({num_edges}, {self.latent_dim})"
            )

    def _example_tensor_kwargs(self) -> dict[str, torch.device | torch.dtype]:
        """Infer device/dtype for eager verification tensors."""
        ref: Optional[torch.Tensor] = next(self.edge_weight_net.parameters(), None)
        if ref is None:
            ref = next(self.edge_weight_net.buffers(), None)
        if ref is None:
            return {}
        tensor_kwargs: dict[str, torch.device | torch.dtype] = {"device": ref.device}
        if torch.is_floating_point(ref):
            tensor_kwargs["dtype"] = ref.dtype
        return tensor_kwargs

    def reset_parameters(self):
        for m in self.edge_weight_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
    """
    EdgeConv-style message passing (DGCNN / Point Cloud Networks).

    ``m_ij = edge_mlp(features(v_i, v_j, e_ij))`` → aggregate → pass-through node update.
    Edges are **not** updated.  Default aggregation is **max**.

    Edge Feature Modes (``edge_feature_mode``):
        - ``'node_difference'`` (default): ``[v_i; v_j - v_i]``  — dim ``2D``
        - ``'concat'``: ``[v_i; v_j]``  — dim ``2D``
        - ``'difference_only'``: ``v_j - v_i``  — dim ``D``
        - ``'concat_with_edges'``: ``[v_i; v_j - v_i; e_ij]``  — dim ``2D + edge_input_dim``
          (requires ``edge_input_dim``)

    Args:
        latent_dim: Node feature dimension (``D``).
        hidden_dim: Hidden dim for the default edge MLP (ignored if ``edge_mlp`` given).
        aggregate: ``'sum'``, ``'mean'``, ``'max'`` (default), ``'min'``, or ``Aggregation``.
        activation: Activation for the default edge MLP.
        edge_feature_mode: Feature assembly mode (see above).
        edge_input_dim: Explicit edge attribute dim; required for ``'concat_with_edges'``.
        edge_mlp: Custom ``nn.Module`` mapping assembled features → ``[E, D]``.

    Example::

        block = EdgeConvBlock(latent_dim=128)                        # DGCNN defaults
        block = EdgeConvBlock(latent_dim=128, aggregate='sum')       # different aggregation
        block = EdgeConvBlock(latent_dim=128, edge_feature_mode='concat_with_edges',
                              edge_input_dim=3)                      # with edge attrs
    """

    #: Supported edge-feature mode labels.
    EDGE_FEATURE_MODES = (
        'node_difference',    # [v_i; v_j - v_i]           → 2 * latent_dim
        'concat',             # [v_i; v_j]                 → 2 * latent_dim
        'difference_only',    # v_j - v_i                  → latent_dim
        'concat_with_edges',  # [v_i; v_j - v_i; e_ij]    → 2 * latent_dim + edge_input_dim
    )
    
    updates_edges = False
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'max',  # Default: max (original)
        activation: str = 'relu',
        edge_feature_mode: str = 'node_difference',
        edge_input_dim: Optional[int] = None,
        edge_mlp: Optional[nn.Module] = None,
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = pass_through_factory(latent_dim=latent_dim)()
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            node_updater=node_updater,
        )
        
        if edge_feature_mode not in self.EDGE_FEATURE_MODES:
            raise ValueError(
                f"Unknown edge_feature_mode={edge_feature_mode!r}. "
                f"Supported: {self.EDGE_FEATURE_MODES}"
            )
        self.edge_feature_mode = edge_feature_mode
        self.edge_input_dim = edge_input_dim
        
        mlp_in_dim = self._edge_feature_dim(
            latent_dim, edge_feature_mode, edge_input_dim,
        )
        
        if edge_mlp is not None:
            self.edge_mlp = edge_mlp
        else:
            # Default MLP: assembled edge features → message
            self.edge_mlp = MLP(
                in_dim=mlp_in_dim,
                out_dim=latent_dim,
                hidden_dims=[hidden_dim],
                activation=activation,
            )

    @staticmethod
    def _edge_feature_dim(
        latent_dim: int,
        mode: str,
        edge_input_dim: Optional[int] = None,
    ) -> int:
        """Return the feature dimension produced by *mode*."""
        if mode == 'node_difference':
            return 2 * latent_dim
        elif mode == 'concat':
            return 2 * latent_dim
        elif mode == 'difference_only':
            return latent_dim
        elif mode == 'concat_with_edges':
            if edge_input_dim is None:
                raise ValueError(
                    "edge_input_dim is required when "
                    "edge_feature_mode='concat_with_edges'"
                )
            return 2 * latent_dim + edge_input_dim
        else:
            raise ValueError(
                f"Unknown edge_feature_mode={mode!r}. "
                f"Supported: {EdgeConvBlock.EDGE_FEATURE_MODES}"
            )

    def _compute_edge_features(self, graph: GraphsTuple) -> torch.Tensor:
        """Assemble per-edge feature vectors according to ``edge_feature_mode``."""
        nodes = graph.nodes
        v_i = nodes[graph.receivers]   # receiver node features
        v_j = nodes[graph.senders]     # sender node features
        
        if self.edge_feature_mode == 'node_difference':
            return torch.cat([v_i, v_j - v_i], dim=-1)
        elif self.edge_feature_mode == 'concat':
            return torch.cat([v_i, v_j], dim=-1)
        elif self.edge_feature_mode == 'difference_only':
            return v_j - v_i
        elif self.edge_feature_mode == 'concat_with_edges':
            return torch.cat([v_i, v_j - v_i, graph.edges], dim=-1)
        else:
            # Defensive; __init__ already validates
            raise ValueError(f"Unknown edge_feature_mode={self.edge_feature_mode!r}")
    
    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, None]:
        edge_features = self._compute_edge_features(graph)
        messages = self.edge_mlp(edge_features)
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
        hidden_dim: int = 128,
        num_mlp_layers: int = 2,
        activation: str = 'relu',
        epsilon: float = 1e-6,
        message_norm: bool = False,
        node_updater: Optional[nn.Module] = None,
    ):
        if node_updater is None:
            node_updater = residual_mlp_factory(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_mlp_layers,
                activation=activation,
                message_norm=message_norm,
                epsilon=epsilon,
            )()
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
        aggregate: Aggregation strategy for node updates (``'sum'``, ``'mean'``, ``'max'``, ``'min'``).
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
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min']] = 'sum',
        use_checkpoint: bool = False,
        block_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()

        self.residual = residual
        self.use_checkpoint = use_checkpoint

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

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all message-passing layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        for i, block in enumerate(self.blocks):
            if self.residual:
                nn_ = self.node_norms[i]
                en_ = self.edge_norms[i]
                _ue = getattr(block, 'updates_edges', True)
                def _step(nodes, edges,
                          _b=block, _g=graph, _nn=nn_, _en=en_,
                          _updates_edges=_ue):
                    normed_edges = _en(edges) if _updates_edges else edges
                    out = _b(_g.replace(nodes=_nn(nodes), edges=normed_edges))
                    new_nodes = nodes + out.nodes
                    new_edges = (edges + out.edges) if _updates_edges else edges
                    return new_nodes, new_edges
            else:
                def _step(nodes, edges, _b=block, _g=graph):
                    out = _b(_g.replace(nodes=nodes, edges=edges))
                    return out.nodes, out.edges

            if self.use_checkpoint:
                new_nodes, new_edges = checkpoint(_step, graph.nodes, graph.edges, use_reentrant=False)
            else:
                new_nodes, new_edges = _step(graph.nodes, graph.edges)
            graph = graph.replace(nodes=new_nodes, edges=new_edges)
        return graph


class GlobalGraphNetProcessor(nn.Module):
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
        hidden_dim: int = 128,
        activation: str = 'gelu',
        residual: bool = True,
        global_pool: str = 'mean',
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.residual = residual
        self.use_checkpoint = use_checkpoint
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

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process graph through all :class:`GlobalGraphNetBlock` layers.

        Args:
            graph: Input :class:`~gnn_pde_v2.core.GraphsTuple`.
                ``graph.globals`` must not be ``None``.

        Returns:
            Processed :class:`~gnn_pde_v2.core.GraphsTuple`.
        """
        for i, block in enumerate(self.blocks):
            if self.use_checkpoint:
                if self.residual:
                    node_norm = self.node_norms[i]
                    edge_norm = self.edge_norms[i]
                    global_norm = self.global_norms[i]
                    def _step(nodes, edges, globs,
                              _b=block, _g=graph,
                              _nn=node_norm, _en=edge_norm, _gn=global_norm):
                        norm_g = _g.replace(
                            nodes=_nn(nodes),
                            edges=_en(edges),
                            globals=_gn(globs),
                        )
                        out = _b(norm_g)
                        return nodes + out.nodes, edges + out.edges, globs + out.globals
                    new_nodes, new_edges, new_globals = checkpoint(
                        _step, graph.nodes, graph.edges, graph.globals,
                        use_reentrant=False
                    )
                else:
                    def _step(nodes, edges, globs, _b=block, _g=graph):
                        out = _b(_g.replace(nodes=nodes, edges=edges, globals=globs))
                        return out.nodes, out.edges, out.globals
                    new_nodes, new_edges, new_globals = checkpoint(
                        _step, graph.nodes, graph.edges, graph.globals,
                        use_reentrant=False
                    )
                graph = graph.replace(
                    nodes=new_nodes, edges=new_edges, globals=new_globals
                )
            else:
                if self.residual:
                    # Pre-norm: normalize before block
                    normalized_nodes = self.node_norms[i](graph.nodes)
                    normalized_edges = self.edge_norms[i](graph.edges)
                    normalized_globals = self.global_norms[i](graph.globals)
                    normalized_graph = graph.replace(
                        nodes=normalized_nodes,
                        edges=normalized_edges,
                        globals=normalized_globals,
                    )
                    # Apply block to normalized input
                    new_graph = block(normalized_graph)
                    # Residual: add block output to original (unnormalized) input
                    new_graph = new_graph.replace(
                        nodes=graph.nodes + new_graph.nodes,
                        edges=graph.edges + new_graph.edges,
                        globals=graph.globals + new_graph.globals,
                    )
                else:
                    new_graph = block(graph)
                graph = new_graph
        return graph


