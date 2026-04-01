"""
Composable node-update strategies for message-passing blocks.

Each strategy is an ``nn.Module`` satisfying
:class:`~gnn_pde_v2.core.protocols.NodeUpdateStrategy`.  Inject one into any
:class:`~gnn_pde_v2.components.processors.MessagePassingBase` subclass to
control how a node combines its own features with aggregated messages.

Built-in strategies
-------------------

==========================  =========================================  ====================
Strategy                    Update rule                                 Default for
==========================  =========================================  ====================
``ConcatMLPNodeUpdater``    ``MLP([v_i; a_i])``                        ``GraphNetBlock``
``RootWeightNodeUpdater``   ``a_i + v_i @ W + b``                      ``EdgeConditionedConvBlock``
``PassThroughNodeUpdater``  ``a_i``                                    ``EdgeConvBlock``
``ResidualMLPNodeUpdater``  ``MLP(v_i + a_i)`` (+ optional msg-norm)   ``GENBlock``
==========================  =========================================  ====================

Custom strategies
-----------------

Any ``nn.Module`` with ``latent_dim: int`` attribute and
``forward(nodes, aggregated) -> Tensor`` works::

    class MyUpdater(nn.Module):
        def __init__(self, latent_dim: int):
            super().__init__()
            self.latent_dim = latent_dim
            self.lin = nn.Linear(latent_dim, latent_dim)

        def forward(self, nodes, aggregated):
            return self.lin(nodes) + aggregated

    block = GraphNetBlock(latent_dim=128, node_updater=MyUpdater(128))
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.mlp import MLP
from ..core.protocols import NodeUpdateStrategy

from typing import Callable

#: Type alias for factory functions that create node updaters.
#: The factory takes no arguments and returns an nn.Module.
NodeUpdaterFactory = Callable[[], nn.Module]

__all__ = [
    # Node updater classes
    "ConcatMLPNodeUpdater",
    "RootWeightNodeUpdater",
    "PassThroughNodeUpdater",
    "ResidualMLPNodeUpdater",
    # Direct builders
    "build_concat_mlp_node_updater",
    "build_root_weight_node_updater",
    "build_pass_through_node_updater",
    "build_residual_mlp_node_updater",
    # Factory functions
    "NodeUpdaterFactory",
    "concat_mlp_factory",
    "root_weight_factory",
    "pass_through_factory",
    "residual_mlp_factory",
    "default_node_updater_factory",
    # Internal
    "_default_node_updater",
]

class _NodeUpdaterBase(nn.Module, ABC):
    """Base class for compile-friendly node-update strategies."""

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, nodes: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        ...


class ConcatMLPNodeUpdater(_NodeUpdaterBase):
    """Concatenate ``[v_i; a_i]`` and pass through an MLP.

    This is the classic DeepMind Graph Nets node update.

    Args:
        latent_dim: Node feature dimension (also the output dimension).
        hidden_dims: Hidden layer sizes for the internal MLP.
        activation: Activation function name.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = 'gelu',
    ):
        super().__init__(latent_dim)
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.mlp = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    def forward(self, nodes: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([nodes, aggregated], dim=-1))


class RootWeightNodeUpdater(_NodeUpdaterBase):
    """Affine root-weight update: ``a_i + v_i @ W + b``.

    Used by edge-conditioned convolution blocks where the aggregated
    messages are combined with a learned linear projection of the
    original node features plus an optional bias.

    Args:
        latent_dim: Node feature dimension.
        root_weight: Whether to include the learned ``W`` projection.
        bias: Whether to include the learned bias ``b``.
    """

    def __init__(
        self,
        latent_dim: int,
        root_weight: bool = True,
        bias: bool = True,
    ):
        super().__init__(latent_dim)
        if root_weight:
            self.root = nn.Parameter(torch.empty(latent_dim, latent_dim))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(latent_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.root is not None:
            nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, nodes: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        out = aggregated
        if self.root is not None:
            out = out + nodes @ self.root
        if self.bias is not None:
            out = out + self.bias
        return out


class PassThroughNodeUpdater(_NodeUpdaterBase):
    """Return aggregated messages directly, ignoring current node features.

    Suitable for blocks like EdgeConv where the edge MLP already
    incorporates receiver node information.
    """

    def forward(self, nodes: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        return aggregated


class ResidualMLPNodeUpdater(_NodeUpdaterBase):
    """Residual add then MLP: ``MLP(v_i + a_i)``, with optional message normalization.

    Used by GEN (generalized aggregation) blocks.

    Args:
        latent_dim: Node feature dimension.
        hidden_dims: Hidden layer sizes for the internal MLP.
        activation: Activation function name.
        message_norm: Whether to normalize aggregated messages before
            the residual addition.
        epsilon: Small constant for numerical stability in message norm.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = 'relu',
        message_norm: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__(latent_dim)
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.epsilon = epsilon
        self.message_norm = message_norm

        self.mlp = MLP(
            in_dim=latent_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            use_layer_norm=False,
        )

        if message_norm:
            self.message_scale = nn.Parameter(torch.ones(1))

    def forward(self, nodes: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        if self.message_norm:
            agg_norm = torch.norm(aggregated, dim=-1, keepdims=True)
            node_norm = torch.norm(nodes, dim=-1, keepdims=True)
            aggregated = self.message_scale * node_norm * aggregated / (agg_norm + self.epsilon)
        return self.mlp(nodes + aggregated)


def _default_node_updater(latent_dim: int, hidden_dim: int = 128, activation: str = 'gelu') -> ConcatMLPNodeUpdater:
    """Default node updater: concatenate + MLP (Graph Nets style)."""
    return ConcatMLPNodeUpdater(latent_dim, hidden_dims=[hidden_dim, hidden_dim], activation=activation)


def build_concat_mlp_node_updater(
    latent_dim: int,
    hidden_dim: int = 128,
    activation: str = 'gelu',
) -> ConcatMLPNodeUpdater:
    """Construct a :class:`ConcatMLPNodeUpdater` directly.

    Prefer this helper for one-off instantiation at module construction time.
    Use :func:`concat_mlp_factory` only when a reusable zero-argument factory
    is specifically required.
    """
    return concat_mlp_factory(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )()


def build_root_weight_node_updater(
    latent_dim: int,
    root_weight: bool = True,
    bias: bool = True,
) -> RootWeightNodeUpdater:
    """Construct a :class:`RootWeightNodeUpdater` directly."""
    return root_weight_factory(
        latent_dim=latent_dim,
        root_weight=root_weight,
        bias=bias,
    )()


def build_pass_through_node_updater(
    latent_dim: int,
) -> PassThroughNodeUpdater:
    """Construct a :class:`PassThroughNodeUpdater` directly."""
    return pass_through_factory(latent_dim=latent_dim)()


def build_residual_mlp_node_updater(
    latent_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    activation: str = 'relu',
    message_norm: bool = False,
    epsilon: float = 1e-6,
) -> ResidualMLPNodeUpdater:
    """Construct a :class:`ResidualMLPNodeUpdater` directly."""
    return residual_mlp_factory(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        message_norm=message_norm,
        epsilon=epsilon,
    )()


# =============================================================================
# Factory Functions
# =============================================================================
# These factories return callable factories that create node updater instances
# when invoked. This pattern allows blocks to create default updaters while:
# 1. Keeping parameter logic localized (no parameter forwarding issues)
# 2. Maintaining torch.compile() compatibility (factories called at __init__ time)
# 3. Allowing users to create consistent updaters across multiple blocks
# =============================================================================

def concat_mlp_factory(
    latent_dim: int,
    hidden_dim: int = 128,
    activation: str = 'gelu',
) -> NodeUpdaterFactory:
    """Factory for creating ConcatMLPNodeUpdater instances.

    Used by :class:`~gnn_pde_v2.components.GraphNetBlock` as its default
    node updater. Creates an updater that concatenates node features with
    aggregated messages and passes them through an MLP.

    Args:
        latent_dim: Node feature dimension (input and output).
        hidden_dim: Hidden dimension for the internal MLP layers.
        activation: Activation function name for the MLP.

    Returns:
        A factory function that creates ConcatMLPNodeUpdater when called.

    Example::

        factory = concat_mlp_factory(latent_dim=128, hidden_dim=256)
        updater = factory()  # Creates the actual updater
    """
    def factory() -> ConcatMLPNodeUpdater:
        return ConcatMLPNodeUpdater(
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )
    return factory


def root_weight_factory(
    latent_dim: int,
    root_weight: bool = True,
    bias: bool = True,
) -> NodeUpdaterFactory:
    """Factory for creating RootWeightNodeUpdater instances.

    Used by :class:`~gnn_pde_v2.components.EdgeConditionedConvBlock` as its
    default node updater. Creates an updater that adds a learned linear
    projection of node features (plus optional bias) to aggregated messages.

    Args:
        latent_dim: Node feature dimension.
        root_weight: Whether to include the learned root projection matrix.
        bias: Whether to include a learned bias term.

    Returns:
        A factory function that creates RootWeightNodeUpdater when called.

    Example::

        factory = root_weight_factory(latent_dim=128, root_weight=True)
        updater = factory()  # Creates the actual updater
    """
    def factory() -> RootWeightNodeUpdater:
        return RootWeightNodeUpdater(
            latent_dim=latent_dim,
            root_weight=root_weight,
            bias=bias,
        )
    return factory


def pass_through_factory(
    latent_dim: int,
) -> NodeUpdaterFactory:
    """Factory for creating PassThroughNodeUpdater instances.

    Used by :class:`~gnn_pde_v2.components.EdgeConvBlock` as its default
    node updater. Creates an updater that simply returns the aggregated
    messages without modification (ignoring current node features).

    Args:
        latent_dim: Node feature dimension (required for interface consistency).

    Returns:
        A factory function that creates PassThroughNodeUpdater when called.

    Example::

        factory = pass_through_factory(latent_dim=128)
        updater = factory()  # Creates the actual updater
    """
    def factory() -> PassThroughNodeUpdater:
        return PassThroughNodeUpdater(latent_dim)
    return factory


def residual_mlp_factory(
    latent_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    activation: str = 'relu',
    message_norm: bool = False,
    epsilon: float = 1e-6,
) -> NodeUpdaterFactory:
    """Factory for creating ResidualMLPNodeUpdater instances.

    Used by :class:`~gnn_pde_v2.components.GENBlock` as its default node
    updater. Creates an updater that adds aggregated messages to node features,
    then passes the result through an MLP (with optional message normalization).

    Args:
        latent_dim: Node feature dimension (input and output).
        hidden_dim: Hidden dimension for each MLP layer.
        num_layers: Number of hidden layers in the MLP.
        activation: Activation function name for the MLP.
        message_norm: Whether to apply message normalization before the residual
            addition (from DeeperGCN).
        epsilon: Small constant for numerical stability in message norm.

    Returns:
        A factory function that creates ResidualMLPNodeUpdater when called.

    Example::

        factory = residual_mlp_factory(
            latent_dim=128, hidden_dim=256, num_layers=3, message_norm=True
        )
        updater = factory()  # Creates the actual updater
    """
    def factory() -> ResidualMLPNodeUpdater:
        return ResidualMLPNodeUpdater(
            latent_dim=latent_dim,
            hidden_dims=[hidden_dim] * num_layers,
            activation=activation,
            message_norm=message_norm,
            epsilon=epsilon,
        )
    return factory


def default_node_updater_factory(
    latent_dim: int,
    hidden_dim: int = 128,
    activation: str = 'gelu',
) -> NodeUpdaterFactory:
    """Default factory for general-purpose node updaters.

    Creates a ConcatMLPNodeUpdater, which is the most general-purpose default
    and matches the original Graph Nets paper implementation.

    Args:
        latent_dim: Node feature dimension (input and output).
        hidden_dim: Hidden dimension for the internal MLP layers.
        activation: Activation function name for the MLP.

    Returns:
        A factory function that creates the default node updater when called.
    """
    return concat_mlp_factory(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        activation=activation,
    )
