"""
GraphNet processors: DeepMind-style and edge-conditioned message passing.

Two independent block families reflect distinct use cases:

- ``MessagePassingBlock`` subclasses (``GraphNetBlock``,
  ``EdgeConditionedConvBlock``):
    Node/edge message passing without explicit globals.

- ``GlobalGraphNetBlock`` / ``GlobalGraphNetProcessor``:
    Full DeepMind Graph Nets block with globals as a first-class
    participant.

``GlobalGraphNetBlock`` is intentionally not derived from
``MessagePassingBlock`` because it performs a 3-step update
(edge → node → global) rather than the base 2-step template.
"""

from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.graph import GraphsTuple
from ..core.functional import aggregate_edges, broadcast_global, aggregate_to_global
from ..core.mlp import MLP
from ..core.aggregation import Aggregation, Sum, get_aggregation


# ---------------------------------------------------------------------------
# Node/edge-only blocks
# ---------------------------------------------------------------------------
class MessagePassingBlock(ABC, nn.Module):
    """
    Abstract base class for graph message passing.
    
    FRAMEWORK CONSTRAINS (cannot be changed by subclasses):
    - Aggregation: MUST use Aggregation Protocol via self._aggregate_fn
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

    def __init__(
        self,
        latent_dim: int,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min'], Callable] = 'sum',
        aggregate_fn: Optional[Callable] = None,  # Deprecated, use aggregate
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Normalize aggregation input: support multiple types
        # Priority: aggregate_fn (deprecated) > aggregate (new) > default
        if aggregate_fn is not None:
            # Legacy: aggregate_fn takes precedence
            self._aggregate_fn = aggregate_fn
        elif callable(aggregate) and not isinstance(aggregate, str):
            # New: Aggregation Protocol instance or custom callable
            self._aggregate_fn = get_aggregation(aggregate)
        elif isinstance(aggregate, str):
            # New: String shortcut
            self._aggregate_fn = get_aggregation(aggregate)
        else:
            # Fallback (shouldn't happen)
            raise TypeError(
                f"aggregate must be Aggregation, string, or callable, got {type(aggregate)}"
            )

    def _aggregate(
        self,
        messages: torch.Tensor,
        receivers: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        return self._aggregate_fn(messages, receivers, num_nodes)

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

    @abstractmethod
    def update_nodes(
        self,
        nodes: torch.Tensor,
        aggregated: torch.Tensor,
        graph: GraphsTuple,
    ) -> torch.Tensor:
        ...

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        if graph.senders is None:
            # Edgeless graph (e.g. after aggressive pooling strips all edges).
            # No messages to aggregate — return graph with only the node MLP
            # applied to a zero-aggregation signal.
            zero_agg = torch.zeros_like(graph.nodes)
            new_nodes = self.update_nodes(graph.nodes, zero_agg, graph)
            return graph.replace(nodes=new_nodes)
        messages, new_edges = self.compute_messages(graph)
        aggregated = self._aggregate(messages, graph.receivers, graph.nodes.shape[0])
        new_nodes = self.update_nodes(graph.nodes, aggregated, graph)
        return graph.replace(
            nodes=new_nodes,
            edges=new_edges if new_edges is not None else graph.edges,
        )


class GraphNetBlock(MessagePassingBlock):
    """
    Single node/edge message-passing step (DeepMind Graph Nets style).
    
    This is the most general-purpose block in the framework. It performs
    a complete 2-step update:
    
    1. **Edge update**: ``new_e_ij = MLP([v_i, v_j; e_ij])``
       - Concatenates sender node, receiver node, and edge features
       - Transforms via MLP to produce new edge features
       - **Updates edges**: Yes (output includes updated edges)
    2. **Node update**: ``new_v_i = MLP([v_i; a_i])``
       - Aggregated messages are concatenated with original node features
       - Transformed via MLP to produce new node features
    
    Edge Feature Strategy:
        Uses explicit edge attributes: [sender_node, receiver_node, edge_attr]
        - sender_node: features of the source node (v_j)
        - receiver_node: features of the target node (v_i)
        - edge_attr: the edge feature tensor (e_ij)
    
    Aggregation:
        Configurable via ``aggregate`` parameter. Default is sum aggregation.
        Use ``aggregate='max'`` or ``aggregate='mean'`` for different behaviors.
    
    Comparison with other MessagePassingBlocks:
    
    +---------------------+---------------------------+--------------------------+---------------------------+
    | Aspect              | GraphNetBlock             | EdgeConditionedConvBlock| EdgeConvBlock             |
    +=====================+===========================+==========================+===========================+
    | Edge features       | [v_i, v_j, e_ij] concat   | e_ij → weight matrix    | [v_i, v_j - v_i] diff    |
    | Updates edges       | Yes                      | No                       | No                        |
    | Default aggregation | sum                      | sum                      | max                       |
    | Use case            | General purpose          | Edge-weighted conv      | Point cloud / geometric   |
    +---------------------+---------------------------+--------------------------+---------------------------+
    
    Use this when all conditioning information (PDE parameters, time, BCs)
    has already been encoded into per-node or per-edge features before
    entering the processor. If you need a dedicated global channel, use
    :class:`GlobalGraphNetBlock` instead.
    
    Args:
        latent_dim: Dimension for node, edge, and output features. All
            three feature channels are assumed to have this dimension after
            encoding.
        hidden_dim: Hidden dimension for internal MLPs.
        activation: Activation function name (``'relu'``, ``'gelu'``, ``'silu'``, ``'tanh'``).
        aggregate: Aggregation strategy. Options: ``'sum'``, ``'mean'``, ``'max'``, ``'min'``,
            or an Aggregation instance (e.g., ``Sum()``, ``Max()``), or a custom callable.
            Default is ``'sum'``.
        aggregate_fn: Deprecated. Use ``aggregate`` instead.
    
    Example::
    
        from gnn_pde_v2.components import GraphNetBlock
        from gnn_pde_v2.core import GraphsTuple
        import torch
        
        # Basic usage (sum aggregation)
        block = GraphNetBlock(latent_dim=128)
        graph = GraphsTuple(...)  # Your input graph
        out_graph = block(graph)
        
        # Custom aggregation
        block = GraphNetBlock(latent_dim=128, aggregate='max')
        block = GraphNetBlock(latent_dim=128, aggregate=Max())
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        activation: str = 'gelu',
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min'], Callable] = 'sum',
        aggregate_fn: Optional[Callable] = None,  # Deprecated, use aggregate
    ):
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            aggregate_fn=aggregate_fn,
        )

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

    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        nodes = graph.nodes
        senders = graph.senders
        receivers = graph.receivers

        # --- Edge update ---
        edge_inputs = torch.cat(
            [nodes[senders], nodes[receivers], graph.edges], dim=-1
        )
        new_edges = self.edge_mlp(edge_inputs)
        return new_edges, new_edges

    def update_nodes(
        self,
        nodes: torch.Tensor,
        aggregated: torch.Tensor,
        graph: GraphsTuple,
    ) -> torch.Tensor:
        node_inputs = torch.cat([nodes, aggregated], dim=-1)
        return self.node_mlp(node_inputs)


class EdgeConditionedConvBlock(MessagePassingBlock):
    """
    Edge-conditioned convolution block (NNConv-style / Neural Operator).
    
    This block uses edge attributes to dynamically compute message weights,
    making it ideal for scenarios where edge features carry important 
    conditioning information (e.g., edge positions, physical properties).
    
    Two-step process:
    
    1. **Message computation**: ``m_ij = v_j ⊙ f(e_ij)``
       - Uses edge features e_ij to generate a weight tensor f(e_ij)
       - Applies weight to sender node features v_j
       - Does NOT use receiver node features in message (only in update)
       - **Updates edges**: No (edges passed through unchanged)
    
    2. **Node update**: ``new_v_i = a_i + (optional) v_i @ W + b``
       - Aggregated messages optionally combined with:
         - Root weight: v_i @ root_matrix (learnable skip connection)
         - Bias term
    
    Edge Feature Strategy:
        Uses explicit edge attributes as input to weight network:
        - edge_attr → MLP → weight tensor → multiply with sender node
        - Four weight types available: 'full', 'vector', 'scalar', 'low_rank'
    
    Weight Generation Modes:
        - ``'full'``: Per-edge [H, H] matrix (full transformation)
        - ``'vector'``: Per-edge [H] vector (channel-wise scaling)
        - ``'scalar'``: Per-edge [1] scalar (global scaling)
        - ``'low_rank'``: Symmetric low-rank approximation W_e ≈ U_e · U_e^T
    
    Low-Rank Mode:
        Memory-efficient symmetric factorization for large latent dimensions.
        Instead of computing full d×d weight matrices, computes factorized 
        U_e ∈ R^{d×r} where r << d.
        
        Message computation: ``M_e = U_e · U_e^T · x_j``
        
        Memory reduction: d×r vs d² (ratio = r/d)
        For d=64, r=8: 512 values vs 4096 values (8× reduction)
        
        The symmetric factorization produces positive semi-definite weight 
        matrices and may better match physical symmetries in Green's function 
        kernels.
    
    Aggregation:
        Configurable via ``aggregate`` parameter. Default is sum aggregation.
    
    Comparison with other MessagePassingBlocks:
    
    +---------------------+---------------------------+--------------------------+---------------------------+
    | Aspect              | GraphNetBlock             | EdgeConditionedConvBlock| EdgeConvBlock             |
    +=====================+===========================+==========================+===========================+
    | Edge features       | [v_i, v_j, e_ij] concat   | e_ij → weight matrix    | [v_i, v_j - v_i] diff    |
    | Updates edges       | Yes                      | No                       | No                        |
    | Default aggregation | sum                      | sum                      | max                       |
    | Use case            | General purpose          | Edge-weighted conv      | Point cloud / geometric   |
    +---------------------+---------------------------+--------------------------+---------------------------+
    
    Args:
        latent_dim: Dimension for node features.
        edge_latent_dim: Dimension for edge features.
        hidden_dim: Hidden dimension for edge weight network.
        edge_weight_type: Weight generation mode. Options: ``'full'``, ``'vector'``, 
            ``'scalar'``, ``'low_rank'``.
        low_rank: Rank for low-rank approximation when ``edge_weight_type='low_rank'``.
            Must be <= latent_dim. Use values like latent_dim//8 to latent_dim//4 
            for memory savings. Ignored for other weight types.
        aggregate: Aggregation strategy. Options: ``'sum'``, ``'mean'``, ``'max'``, ``'min'``,
            or an Aggregation instance, or a custom callable. Default is ``'sum'``.
        aggregate_fn: Deprecated. Use ``aggregate`` instead.
        root_weight: Whether to add skip connection via learned root matrix.
            Default is True.
        bias: Whether to add bias term. Default is True.
        activation: Activation function name for weight network.
    
    Example::
    
        from gnn_pde_v2.components import EdgeConditionedConvBlock
        
        # Full weight matrix (most expressive)
        block = EdgeConditionedConvBlock(latent_dim=128, edge_latent_dim=16, edge_weight_type='full')
        
        # Vector gating (channel-wise scaling)
        block = EdgeConditionedConvBlock(latent_dim=128, edge_latent_dim=16, edge_weight_type='vector')
        
        # Scalar gating (global scaling)
        block = EdgeConditionedConvBlock(latent_dim=128, edge_latent_dim=16, edge_weight_type='scalar')
        
        # Low-rank symmetric factorization (memory-efficient)
        block = EdgeConditionedConvBlock(
            latent_dim=128, 
            edge_latent_dim=16, 
            edge_weight_type='low_rank',
            low_rank=16,  # r=16 for d=128 gives 8× memory reduction
        )
        
        # Custom aggregation
        block = EdgeConditionedConvBlock(latent_dim=128, edge_latent_dim=16, aggregate='max')
    """

    updates_edges = False

    def __init__(
        self,
        latent_dim: int,
        edge_latent_dim: int,
        hidden_dim: int = 128,
        edge_weight_type: str = 'full',
        low_rank: int = 0,
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min'], Callable] = 'sum',
        aggregate_fn: Optional[Callable] = None,  # Deprecated, use aggregate
        root_weight: bool = True,
        bias: bool = True,
        activation: str = 'relu',
    ):
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            aggregate_fn=aggregate_fn,
        )

        if edge_weight_type == 'full':
            out_dim = latent_dim * latent_dim
        elif edge_weight_type == 'vector':
            out_dim = latent_dim
        elif edge_weight_type == 'scalar':
            out_dim = 1
        elif edge_weight_type == 'low_rank':
            if low_rank <= 0:
                raise ValueError(f"low_rank must be positive when edge_weight_type='low_rank', got {low_rank}")
            if low_rank > latent_dim:
                raise ValueError(f"low_rank ({low_rank}) must be <= latent_dim ({latent_dim})")
            out_dim = latent_dim * low_rank
            self.low_rank = low_rank
        else:
            raise ValueError(f"Unknown edge_weight_type: {edge_weight_type}")

        self.edge_weight_type = edge_weight_type
        self.edge_weight_net = MLP(
            in_dim=edge_latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
            use_layer_norm=False,
        )

        if root_weight:
            self.root = nn.Parameter(torch.empty(latent_dim, latent_dim))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(latent_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.edge_weight_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.root is not None:
            nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H = self.latent_dim
        src_x = graph.nodes[graph.senders]  # [E, H]

        w = self.edge_weight_net(graph.edges)
        
        if self.edge_weight_type == 'full':
            W = w.view(-1, H, H)
            msg = torch.bmm(src_x.unsqueeze(1), W).squeeze(1)  # [E, H]
        elif self.edge_weight_type == 'low_rank':
            # Symmetric low-rank message computation: M_e = U_e · U_e^T · x_j
            # Step 1: Reshape to get U_e factors [E, d, r]
            edge_u = w.view(-1, H, self.low_rank)  # [E, d, r]
            # Step 2: Project to rank-r space: h_e = U_e^T · x_j
            # x_j: [E, d], U_e: [E, d, r] -> h_e: [E, r]
            h_e = torch.einsum('ed,edr->er', src_x, edge_u)
            # Step 3: Project back to d-dim: M_e = U_e · h_e
            msg = torch.einsum('er,edr->ed', h_e, edge_u)  # [E, d]
        else:
            msg = src_x * w  # [E, H]

        return msg, None  # Keep edges unchanged

    def update_nodes(
        self,
        nodes: torch.Tensor,
        aggregated: torch.Tensor,
        graph: GraphsTuple,
    ) -> torch.Tensor:
        out = aggregated
        if self.root is not None:
            out = out + nodes @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# EdgeConv block (DGCNN-style)
# ----------------------------------------------------------------------------

class EdgeConvBlock(MessagePassingBlock):
    """
    EdgeConv-style message passing (DGCNN-style / Point Cloud Networks).
    
    This block is designed for capturing local geometric structure in graphs
    where edge features are implicitly derived from node feature differences
    rather than from explicit edge attributes.
    
    Two-step process:
    
    1. **Message computation**: ``m_ij = MLP([v_i; v_j - v_i])``
       - Computes edge features from node difference: (v_j - v_i)
       - Concatenates with receiver node: [v_i; v_j - v_i]
       - Transforms via MLP to produce message
       - **Does NOT use explicit edge attributes** (graph.edges is ignored)
       - **Updates edges**: No (edges passed through unchanged)
    
    2. **Node update**: ``new_v_i = a_i``
       - Direct pass-through: aggregated messages become new node features
       - No additional transformation (unlike GraphNetBlock which uses MLP)
    
    Edge Feature Strategy:
        Implicit derivation from node features:
        - Uses node difference: v_j - v_i (relative feature change)
        - Does NOT use explicit edge attributes from graph.edges
        - This is ideal for point clouds where edges represent spatial proximity
    
    Aggregation:
        Default is **max** aggregation (original EdgeConv / DGCNN behavior).
        This captures the strongest signal from neighbors, which is particularly
        useful for point cloud tasks. However, aggregation is fully configurable.
    
    Comparison with other MessagePassingBlocks:
    
    +---------------------+---------------------------+--------------------------+---------------------------+
    | Aspect              | GraphNetBlock             | EdgeConditionedConvBlock| EdgeConvBlock             |
    +=====================+===========================+==========================+===========================+
    | Edge features       | [v_i, v_j, e_ij] concat   | e_ij → weight matrix    | [v_i, v_j - v_i] diff    |
    | Updates edges       | Yes                      | No                       | No                        |
    | Default aggregation | sum                      | sum                      | max                       |
    | Use case            | General purpose          | Edge-weighted conv      | Point cloud / geometric   |
    +---------------------+---------------------------+--------------------------+---------------------------+
    
    Args:
        latent_dim: Dimension for node features.
        hidden_dim: Hidden dimension for edge feature MLP.
        aggregate: Aggregation strategy. Options: ``'sum'``, ``'mean'``, ``'max'``, ``'min'``,
            or an Aggregation instance, or a custom callable. Default is ``'max'``.
        aggregate_fn: Deprecated. Use ``aggregate`` instead.
        activation: Activation function name for edge MLP.
    
    Example::
    
        from gnn_pde_v2.components import EdgeConvBlock
        from gnn_pde_v2.core import Max  # Protocol instance
        
        # Default: max aggregation (original EdgeConv / DGCNN)
        block = EdgeConvBlock(latent_dim=128)
        
        # Configurable aggregation
        block = EdgeConvBlock(latent_dim=128, aggregate='sum')
        block = EdgeConvBlock(latent_dim=128, aggregate=Max())  # Explicit
        
        # With custom hidden dimension
        block = EdgeConvBlock(latent_dim=128, hidden_dim=256)
    """
    
    updates_edges = False
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        aggregate: Union[Aggregation, str, Callable] = 'max',  # Default: max (original)
        aggregate_fn: Optional[Callable] = None,  # Deprecated, use aggregate
        activation: str = 'relu',
    ):
        super().__init__(
            latent_dim=latent_dim,
            aggregate=aggregate,
            aggregate_fn=aggregate_fn,
        )
        
        # Edge feature MLP: [v_i, v_j - v_i] → message
        self.edge_mlp = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
    
    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, None]:
        nodes = graph.nodes
        senders = graph.senders
        receivers = graph.receivers
        
        # Edge features: [v_i, v_j - v_i]
        v_i = nodes[receivers]  # receiver node features
        v_j = nodes[senders]     # sender node features
        edge_features = torch.cat([v_i, v_j - v_i], dim=-1)
        
        messages = self.edge_mlp(edge_features)
        return messages, None  # Don't update edges
    
    def update_nodes(
        self,
        nodes: torch.Tensor,
        aggregated: torch.Tensor,
        graph: GraphsTuple,
    ) -> torch.Tensor:
        return aggregated  # Direct pass-through


# ---------------------------------------------------------------------------
# GEN Block (GEneralized aggregation Network)
# ---------------------------------------------------------------------------

class GENBlock(MessagePassingBlock):
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
    ):
        super().__init__(
            latent_dim=latent_dim,
            aggregate='sum',  # Softmax is applied before aggregation
        )
        self.epsilon = epsilon
        self.message_norm = message_norm
        
        # Node update MLP: transforms concatenated [node + aggregated_messages]
        self.node_mlp = MLP(
            in_dim=latent_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim] * num_mlp_layers,
            activation=activation,
            use_layer_norm=False,
        )
        
        if message_norm:
            # Learnable scale parameter for message normalization
            self.message_scale = nn.Parameter(torch.ones(1))
    
    def compute_messages(
        self,
        graph: GraphsTuple,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute GEN messages: m_ij = ReLU(e_ij + h_j) + epsilon
        
        Returns:
            messages: [E, latent_dim] computed messages
            new_edges: None (edges are not updated in GEN)
        """
        nodes = graph.nodes
        edges = graph.edges
        senders = graph.senders
        
        # Gather sender node features
        sender_features = nodes[senders]
        
        # Compute messages: m_ij = ReLU(e_ij + h_j) + epsilon
        messages = torch.relu(edges + sender_features) + self.epsilon
        
        return messages, None  # Edges are NOT updated
    
    def update_nodes(
        self,
        nodes: torch.Tensor,
        aggregated: torch.Tensor,
        graph: GraphsTuple,
    ) -> torch.Tensor:
        """
        Update nodes: h'_i = MLP(h_i + agg_i)
        
        Args:
            nodes: [N, latent_dim] original node features
            aggregated: [N, latent_dim] aggregated messages
            graph: Input graph
            
        Returns:
            [N, latent_dim] updated node features
        """
        # Optional message normalization
        if self.message_norm:
            agg_norm = torch.norm(aggregated, dim=-1, keepdims=True)
            node_norm = torch.norm(nodes, dim=-1, keepdims=True)
            aggregated = self.message_scale * node_norm * aggregated / (agg_norm + self.epsilon)
        
        # Residual-style update: h_i + agg_i
        node_input = nodes + aggregated
        return self.node_mlp(node_input)
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Apply GEN block with softmax aggregation.
        
        This overrides the base forward to insert softmax before aggregation.
        """
        # Compute messages (edges not updated)
        messages, _ = self.compute_messages(graph)
        
        # Softmax aggregation: compute attention weights
        from ..core.functional import scatter_softmax
        attention_weights = scatter_softmax(
            messages, graph.receivers, dim=0, dim_size=graph.nodes.shape[0]
        )
        weighted_messages = attention_weights * messages
        
        # Aggregate weighted messages
        aggregated = self._aggregate(weighted_messages, graph.receivers, graph.nodes.shape[0])
        
        # Update nodes
        new_nodes = self.update_nodes(graph.nodes, aggregated, graph)
        
        return graph.replace(nodes=new_nodes)


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
        aggregate: Union[Aggregation, Literal['sum', 'mean', 'max', 'min'], Callable] = 'sum',
        aggregate_fn: Optional[Callable] = None,  # Deprecated, use aggregate
        global_pool: str = 'mean',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.global_latent_dim = global_latent_dim
        self.global_pool = global_pool
        
        # Normalize aggregation: support new aggregate param + legacy aggregate_fn
        if aggregate_fn is not None:
            self.aggregate_fn = aggregate_fn
        else:
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
        aggregate_fn: Edge-to-node aggregation callable passed to each
            block.  Defaults to sum aggregation.
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
        aggregate_fn: Optional[Callable] = None,
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
                    aggregate_fn=aggregate_fn,
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
        aggregate_fn: Edge-to-node aggregation callable.
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
        aggregate_fn: Optional[Callable] = None,
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
                aggregate_fn=aggregate_fn,
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
