from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.graph import GraphsTuple
from ..core.functional import aggregate_edges, broadcast_global, aggregate_to_global
from ..core.mlp import MLP
from ..core.aggregation import Aggregation, Sum, get_aggregation

__all__ = [
    "EdgeMessageProcessor",
    "FullEdgeMessageProcessor",
    "VectorEdgeMessageProcessor",
    "ScalarEdgeMessageProcessor",
    "LowRankEdgeMessageProcessor",
    "_default_edge_message_processor",
]


class EdgeMessageProcessor(nn.Module, ABC):
    """Abstract base for compile-friendly edge-message processors.

    This abstract base is the single contract for edge-message processors:
    concrete processors subclass it, instantiation fails unless
    ``weight_out_dim`` and ``forward`` are implemented, and the
    construction-time validator ``validate_edge_message_processor`` checks
    membership via ``isinstance(p, EdgeMessageProcessor)``.  A
    ``__subclasshook__`` keeps the contract open for extension: any
    ``nn.Module`` exposing ``weight_out_dim`` and ``forward`` passes
    ``isinstance`` without explicit inheritance.
    """

    @classmethod
    def __subclasshook__(cls, candidate):
        if cls is EdgeMessageProcessor:
            if all(hasattr(candidate, attr) for attr in ("weight_out_dim", "forward")):
                return True
        return NotImplemented

    def __init__(self, latent_dim: int):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = latent_dim

    @property
    @abstractmethod
    def weight_out_dim(self) -> int:
        """Required output size of the edge weight network."""
        ...

    def verify_shape_contract(self, num_edges: int = 2) -> None:
        """Exercise ``forward`` on synthetic inputs and assert output shape.

        This performs an eager construction-time compatibility check so that
        malformed custom processors fail early, before the first real graph is
        processed.
        """
        if num_edges <= 0:
            raise ValueError(f"num_edges must be positive, got {num_edges}")

        src_x = torch.randn(num_edges, self.latent_dim)
        edge_weights = torch.randn(num_edges, self.weight_out_dim)

        with torch.no_grad():
            out = self(src_x, edge_weights)

        if out.ndim != 2:
            raise ValueError(
                f"edge_processor forward must return rank-2 tensor [E, H], got ndim={out.ndim}"
            )
        if out.shape[0] != num_edges:
            raise ValueError(
                "edge_processor forward must preserve edge count during shape verification: "
                f"got {out.shape[0]} vs {num_edges}"
            )
        if out.shape[-1] != self.latent_dim:
            raise ValueError(
                "edge_processor forward must return last dimension equal to latent_dim "
                f"during shape verification: got {out.shape[-1]} vs {self.latent_dim}"
            )


class FullEdgeMessageProcessor(EdgeMessageProcessor):
    """Applies a full per-edge ``[H, H]`` weight matrix.

    This is the most expressive built-in processor. The edge weight network
    must emit ``H * H`` values per edge, which are reshaped into a dense
    transformation matrix and applied to the sender node features.
    """

    @property
    def weight_out_dim(self) -> int:
        return self.latent_dim * self.latent_dim

    def forward(self, src_x: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        H = self.latent_dim
        # Reshape edge weights into [E, H, H] and apply to src_x with batch matmul.
        W = edge_weights.view(-1, H, H)
        return torch.bmm(src_x.unsqueeze(1), W).squeeze(1)


class VectorEdgeMessageProcessor(EdgeMessageProcessor):
    """Applies per-channel vector gating ``src_x * w``.

    The edge weight network emits one weight per latent channel, producing a
    lightweight diagonal-style modulation of each sender feature vector.
    """

    @property
    def weight_out_dim(self) -> int:
        return self.latent_dim

    def forward(self, src_x: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        # Elementwise multiplication with broadcasting over the latent dimension.
        return src_x * edge_weights


class ScalarEdgeMessageProcessor(EdgeMessageProcessor):
    """Applies scalar gating with broadcast over channels.

    The edge weight network emits a single scalar per edge, which is
    broadcast across all latent channels. This is the cheapest built-in mode.
    """

    @property
    def weight_out_dim(self) -> int:
        return 1

    def forward(self, src_x: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        # Elementwise multiplication with broadcasting over the latent dimension.
        return src_x * edge_weights


class LowRankEdgeMessageProcessor(EdgeMessageProcessor):
    """Applies symmetric low-rank message transforms ``U U^T x``.

    Instead of materializing a full ``[H, H]`` matrix per edge, this processor
    uses a factorized representation ``U ∈ R^{H×r}``. The edge weight network
    therefore emits ``H * r`` values per edge, reducing memory and compute when
    ``r << H``.
    """

    def __init__(self, latent_dim: int, low_rank: int):
        super().__init__(latent_dim=latent_dim)
        if low_rank <= 0:
            raise ValueError(
                f"low_rank must be positive for LowRankEdgeMessageProcessor, got {low_rank}"
            )
        if low_rank > latent_dim:
            raise ValueError(
                f"low_rank ({low_rank}) must be <= latent_dim ({latent_dim})"
            )
        self.low_rank = low_rank

    @property
    def weight_out_dim(self) -> int:
        return self.latent_dim * self.low_rank

    def forward(self, src_x: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        H = self.latent_dim
        # Reshape edge weights into [E, H, r] and apply symmetric low-rank transform.
        edge_u = edge_weights.view(-1, H, self.low_rank)
        # First compute U^T x for each edge, resulting in [E, r]. Then compute U (U^T x) with another batch matmul, resulting in [E,
        h_e = torch.einsum('ed,edr->er', src_x, edge_u)
        return torch.einsum('er,edr->ed', h_e, edge_u)


def _default_edge_message_processor(latent_dim: int) -> FullEdgeMessageProcessor:
    """Default processor: full per-edge weight matrices.

    This preserves the original default semantics of
    :class:`EdgeConditionedConvBlock`.
    """
    return FullEdgeMessageProcessor(latent_dim)
