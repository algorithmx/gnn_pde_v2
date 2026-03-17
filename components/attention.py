"""
Attention components for transformer architectures.

Includes multi-head attention variants, physics token attention, and relative
position encoding for spatial awareness in PDE applications.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..core.functional import scatter_softmax, aggregate_edges
from .temperature import create_temperature_module, TemperatureBase


# =============================================================================
# QKV Projection Helpers
# =============================================================================

class QKVProjectionType:
    """Enum-like options for QKV projection strategies."""
    COMBINED = "combined"      # Single linear: dim -> 3*dim
    SEPARATE = "separate"       # Three linears: dim -> dim each
    TOKEN_SLICE = "token_slice"  # Custom for token-based attention


def create_qkv_projection(
    dim: int,
    projection_type: str = "combined",
) -> nn.Module:
    """
    Factory function to create QKV projection module.
    
    Args:
        dim: Model dimension
        projection_type: One of "combined", "separate", or a custom Module
        
    Returns:
        A module with forward(x) returning (q, k, v) tensors
    """
    if projection_type == QKVProjectionType.COMBINED:
        return _CombinedQKVProjection(dim)
    elif projection_type == QKVProjectionType.SEPARATE:
        return _SeparateQKVProjection(dim)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")


class _CombinedQKVProjection(nn.Module):
    """Combined QKV projection: single linear layer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [*, D] input features
            
        Returns:
            (q, k, v) each [*, D]
        """
        qkv = self.qkv(x)
        # Split along feature dimension
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v


class _SeparateQKVProjection(nn.Module):
    """Separate QKV projections: three linear layers."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [*, D] input features
            
        Returns:
            (q, k, v) each [*, D]
        """
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)



# =============================================================================
# Relative Position Encoding
# =============================================================================

def _get_position_bucket(distance: torch.Tensor, num_buckets: int, max_distance: float) -> torch.Tensor:
    """
    Map continuous distances to discrete buckets using logarithmic bucketing.
    
    This follows the T5/Transformer-XL style bucketing where:
    - Bucket 0: distance in [0, 1)
    - Bucket 1: distance in [1, 2)
    - Bucket 2: distance in [2, 4)
    - Bucket 3: distance in [4, 8)
    - etc.
    
    This allows fine-grained local interactions and coarse long-range interactions.
    
    Args:
        distance: [..., N, N] pairwise distances
        num_buckets: Number of discrete buckets
        max_distance: Maximum distance to consider
        
    Returns:
        [..., N, N] bucket indices (0 to num_buckets-1)
    """
    # Clip distance to max_distance
    distance = distance.clamp(max=max_distance)
    
    # Bucket 0 is reserved for exact matches (distance < 1)
    # For distance >= 1, use logarithmic bucketing
    log_distance = torch.log(distance + 1e-8) / math.log(max_distance + 1e-8)
    log_buckets = torch.floor(log_distance * (num_buckets - 1))
    
    # Distance < 1 goes to bucket 0, distance >= 1 uses log buckets starting from 1
    buckets = torch.where(distance < 1.0, torch.zeros_like(distance), log_buckets + 1)
    
    # Clamp to valid range
    buckets = buckets.clamp(0, num_buckets - 1).long()
    
    return buckets


class RelativePositionEncoding(nn.Module):
    """
    Relative position encoding for attention mechanisms.
    
    Computes learnable or sinusoidal position biases based on pairwise distances
    between node positions. These biases are added to attention scores before
    softmax, allowing the model to attend based on spatial relationships.
    
    Reference:
        - T5: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
        - Graph Transformer with positional encoding variants
    
    Args:
        num_heads: Number of attention heads
        num_buckets: Number of distance buckets
        position_dim: Dimension of position vectors (2 for 2D, 3 for 3D, etc.)
        max_distance: Maximum distance to consider
        encoding_type: 'learned' or 'sinusoidal'
    """
    
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        position_dim: int = 2,
        max_distance: float = 10.0,
        encoding_type: str = 'learned',
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.position_dim = position_dim
        self.max_distance = max_distance
        self.encoding_type = encoding_type
        
        if encoding_type == 'learned':
            # Learnable bias per head per bucket
            self.position_bias = nn.Parameter(torch.randn(num_heads, num_buckets) * 0.02)
        elif encoding_type == 'sinusoidal':
            # Fixed sinusoidal encodings
            self._init_sinusoidal_encodings()
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}. Use 'learned' or 'sinusoidal'.")
    
    def _init_sinusoidal_encodings(self):
        """Initialize sinusoidal position encodings."""
        # Create sinusoidal encodings for each bucket
        position = torch.arange(self.num_buckets).unsqueeze(1).float()  # [num_buckets, 1]
        div_term = torch.exp(torch.arange(0, self.num_heads, 2).float() * 
                            (-math.log(10000.0) / self.num_heads))  # [num_heads//2]
        
        encodings = torch.zeros(self.num_heads, self.num_buckets)
        # For each pair of heads (sin, cos), compute the encoding
        # position * div_term has shape [num_buckets, num_heads//2]
        sin_enc = torch.sin(position * div_term)  # [num_buckets, num_heads//2]
        cos_enc = torch.cos(position * div_term)  # [num_buckets, num_heads//2]
        
        # Assign sin to even indices, cos to odd indices
        encodings[0::2, :] = sin_enc.T  # [num_heads//2, num_buckets]
        if self.num_heads > 1:
            encodings[1::2, :] = cos_enc.T  # [num_heads//2, num_buckets]
        
        self.register_buffer('sinusoidal_encodings', encodings)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute relative position bias from node positions.
        
        Args:
            positions: [B, N, position_dim] or [N, position_dim] - Node positions
            
        Returns:
            [B, num_heads, N, N] or [num_heads, N, N] position bias
        """
        single_batch = False
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            single_batch = True
        
        # Compute pairwise distances: [B, N, N]
        # dist[i, j] = ||pos[i] - pos[j]||
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, position_dim]
        distance = torch.norm(diff, dim=-1)  # [B, N, N]
        
        # Map distances to buckets: [B, N, N]
        buckets = _get_position_bucket(distance, self.num_buckets, self.max_distance)
        
        # Get bias for each bucket
        if self.encoding_type == 'learned':
            # [num_heads, num_buckets] -> gather -> [B, num_heads, N, N]
            bias = self.position_bias[:, buckets]  # [num_heads, B, N, N]
            bias = bias.permute(1, 0, 2, 3)  # [B, num_heads, N, N]
        else:  # sinusoidal
            bias = self.sinusoidal_encodings[:, buckets]  # [num_heads, B, N, N]
            bias = bias.permute(1, 0, 2, 3)  # [B, num_heads, N, N]
        
        if single_batch:
            bias = bias.squeeze(0)  # [num_heads, N, N]
        
        return bias


# =============================================================================
# Attention Components
# =============================================================================

class BaseMultiHeadAttention(nn.Module):
    """
    Base class for multi-head attention mechanisms.
    
    Provides common functionality for:
    - QKV projection
    - Relative position encoding
    - Batch/unbatched input handling
    - Attention mask application
    - Output projection
    
    Subclasses must implement the ``_compute_attention_scores`` method.
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_positions: Whether to use relative position encoding
        position_dim: Dimension of position vectors (2 for 2D, 3 for 3D)
        max_distance: Maximum distance to consider for position encoding
        num_position_buckets: Number of discrete distance buckets
        position_encoding_type: 'learned' or 'sinusoidal'
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_relative_positions: bool = False,
        position_dim: int = 2,
        max_distance: float = 10.0,
        num_position_buckets: int = 32,
        position_encoding_type: str = 'learned',
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_relative_positions = use_relative_positions
        
        # QKV projection
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position encoding
        if use_relative_positions:
            self.position_encoding = RelativePositionEncoding(
                num_heads=n_heads,
                num_buckets=num_position_buckets,
                position_dim=position_dim,
                max_distance=max_distance,
                encoding_type=position_encoding_type,
            )
        else:
            self.position_encoding = None
    
    def _compute_attention_scores(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores from Q, K, V tensors.
        
        Subclasses must implement this method.
        
        Args:
            q: [B, H, N, d] Query tensor
            k: [B, H, N, d] Key tensor  
            v: [B, H, N, d] Value tensor
            
        Returns:
            Attention scores [B, H, N, N]
        """
        raise NotImplementedError("Subclasses must implement _compute_attention_scores")
    
    def _apply_position_bias(
        self, 
        scores: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Apply relative position bias to attention scores."""
        if self.position_encoding is None:
            return scores
            
        position_bias = self.position_encoding(positions)
        if position_bias.dim() == 3:
            position_bias = position_bias.unsqueeze(0)  # [1, H, N, N]
        return scores + position_bias
    
    def _apply_mask(
        self, 
        scores: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply attention mask to scores."""
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        return scores
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] or [N, D] - Input features
            mask: Optional attention mask [B, N, N] or [N, N]
            positions: [B, N, position_dim] or [N, position_dim] - Node positions
                      Required if use_relative_positions=True
                      
        Returns:
            [B, N, D] or [N, D] - Output features
        """
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
            if positions is not None and positions.dim() == 2:
                positions = positions.unsqueeze(0)
        
        B, N, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]
        
        # Subclass-specific attention score computation
        scores = self._compute_attention_scores(q, k, v)
        
        # Add relative position bias if enabled
        if self.use_relative_positions:
            if positions is None:
                raise ValueError(
                    "positions must be provided when use_relative_positions=True"
                )
            scores = self._apply_position_bias(scores, positions)
        
        # Apply mask
        scores = self._apply_mask(scores, mask)
        
        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.out_proj(out)
        
        if single_batch:
            out = out.squeeze(0)
        
        return out


class MultiHeadAttention(BaseMultiHeadAttention):
    """
    Standard multi-head self-attention with optional relative position encoding.
    
    Implements scaled dot-product attention:
        attention = softmax(Q @ K^T / sqrt(d)) * V
    
    For PDE applications, relative position encoding allows the model to attend
    based on spatial relationships between nodes. Position biases are computed
    from pairwise distances and added to attention scores before softmax.
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_positions: Whether to use relative position encoding
        position_dim: Dimension of position vectors (2 for 2D, 3 for 3D)
        max_distance: Maximum distance to consider for position encoding
        num_position_buckets: Number of discrete distance buckets
        position_encoding_type: 'learned' or 'sinusoidal'
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_relative_positions: bool = False,
        position_dim: int = 2,
        max_distance: float = 10.0,
        num_position_buckets: int = 32,
        position_encoding_type: str = 'learned',
    ):
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            use_relative_positions=use_relative_positions,
            position_dim=position_dim,
            max_distance=max_distance,
            num_position_buckets=num_position_buckets,
            position_encoding_type=position_encoding_type,
        )
    
    def _compute_attention_scores(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        return (q @ k.transpose(-2, -1)) / self.scale


class QKNormMultiHeadAttention(BaseMultiHeadAttention):
    """
    Multi-head self-attention with Query-Key Normalization (QK-Norm).
    
    Implements the QK-Norm technique from the paper:
    "Query-Key Normalization for Transformers" (Henry et al., 2020)
    
    Key differences from standard scaled dot-product attention:
    1. Applies L2 normalization along the head dimension to Q and K
    2. Uses a learnable scalar parameter g instead of fixed sqrt(d) scaling
    
    This converts dot products to cosine similarities scaled by g, which:
    - Makes attention scores less prone to saturation
    - Enables more diffuse attention patterns
    - Improves performance on low-resource translation tasks
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        use_relative_positions: Whether to use relative position encoding
        position_dim: Dimension of position vectors
        max_distance: Maximum distance for position encoding
        num_position_buckets: Number of distance buckets for position encoding
        position_encoding_type: 'learned' or 'sinusoidal'
        init_g: Initial value for learnable scaling parameter g.
               Defaults to sqrt(dim) for better initialization.
        learnable_g: Whether g is learnable (default True)
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_relative_positions: bool = False,
        position_dim: int = 2,
        max_distance: float = 10.0,
        num_position_buckets: int = 32,
        position_encoding_type: str = 'learned',
        init_g: Optional[float] = None,
        learnable_g: bool = True,
    ):
        # Initialize base with use_relative_positions=False, then set it in subclass
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            use_relative_positions=False,
            position_dim=position_dim,
            max_distance=max_distance,
            num_position_buckets=num_position_buckets,
            position_encoding_type=position_encoding_type,
        )
        
        # Override position_encoding with the correct setting
        if use_relative_positions:
            self.position_encoding = RelativePositionEncoding(
                num_heads=n_heads,
                num_buckets=num_position_buckets,
                position_dim=position_dim,
                max_distance=max_distance,
                encoding_type=position_encoding_type,
            )
        
        self.use_relative_positions = use_relative_positions
        
        # =================================================================
        # Improved design: Use softplus instead of exp for positivity
        # 
        # Benefits:
        # 1. softplus is more numerically stable than exp (no overflow)
        # 2. More consistent with temperature pattern in PhysicsTokenAttention  
        # 3. Avoids computing exp() every forward pass
        # 4. Better gradient flow near zero
        # =================================================================
        
        if init_g is None:
            # Better default: sqrt(dim) ≈ expected cosine similarity magnitude
            init_g = math.sqrt(dim)
        
        self.learnable_g = learnable_g
        if learnable_g:
            # Use raw parameter + softplus for positivity constraint
            # softplus(x) = log(1 + exp(x)) ensures g > 0
            self._g_raw = nn.Parameter(torch.tensor(init_g))
        else:
            self.register_buffer('g', torch.tensor(init_g))
    
    @property
    def g(self) -> torch.Tensor:
        """Get current g value (ensured positive via softplus)."""
        if self.learnable_g:
            # softplus(x) = log(1 + exp(x)) ensures positivity
            return F.softplus(self._g_raw)
        return self.g
    
    def _compute_attention_scores(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        QK-Norm attention: cosine similarity scaled by learnable g.
        
        This converts dot products to cosine similarities:
            attention = softmax(g * cosine(Q, K)) * V
        
        Where cosine(Q, K) = Q_norm @ K_norm^T (L2 normalized vectors).
        """
        # Apply L2 normalization along head dimension
        q_norm = F.normalize(q, p=2, dim=-1)  # [B, H, N, d]
        k_norm = F.normalize(k, p=2, dim=-1)  # [B, H, N, d]
        
        # Compute cosine similarity scaled by g
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) * self.g
        
        return scores
    
    def get_g_value(self) -> float:
        """Get current value of learnable parameter g."""
        return self.g.item()



class SparseGraphAttention(nn.Module):
    """
    Sparse graph attention over explicit edge-defined neighbourhoods.

    Graph-topology-aware counterpart to :class:`MultiHeadAttention`. Computes
    attention only over the edges provided by ``senders``/``receivers`` rather
    than all N² pairs, making it suitable for large sparse graphs.

    Uses canonical temperature system from ``temperature.py`` for controlling
    attention distribution sharpness. Supports multiple temperature modes:
    ``fixed``, ``learnable_scalar``, ``per_head``, ``adaptive``, ``annealed``.

    Key design choices (Shirzad et al., NeurIPS 2023 — *Low-Width
    Approximations and Sparsification for Scaling Graph Transformers*):

    * **Edge type bias** — a learnable per-type additive bias is added to the
      value vectors for each edge (e.g. type 0 = original graph edge, 1 =
      expander edge, 2 = self-loop).  Disabled by default (``num_edge_types=0``).
    * **V normalisation** — L2-normalises value vectors and rescales by a
      learnable global scalar ``v_scale``.  Stabilises training of very
      narrow (width-4) networks.  Disabled by default.
    * **Temperature** — Uses canonical ``TemperatureBase`` system for learnable,
      adaptive, and annealed temperature modes.

    Args:
        dim: Model dimension (must be divisible by ``n_heads``).
        n_heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
        num_edge_types: Number of distinct edge type classes for the per-type
            value bias.  Set to 0 (default) to disable.
        use_v_norm: If ``True``, L2-normalise value vectors and rescale by a
            learnable scalar. Recommended for low-width (dim ≤ 8) settings.
        temperature_mode: Temperature mode for ``create_temperature_module()``.
            Options: "fixed", "learnable_scalar", "per_head", "adaptive", "annealed".
        min_temperature: Minimum temperature when using learnable modes.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        dropout: float = 0.0,
        num_edge_types: int = 0,
        use_v_norm: bool = False,
        temperature_mode: str = "fixed",
        min_temperature: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, (
            f"dim {dim} must be divisible by n_heads {n_heads}"
        )

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.num_edge_types = num_edge_types
        self.use_v_norm = use_v_norm

        # QKV projection
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Per-type additive bias on value vectors: [num_edge_types, n_heads, head_dim]
        if num_edge_types > 0:
            self.edge_type_bias = nn.Parameter(
                torch.zeros(num_edge_types, n_heads, self.head_dim)
            )
        else:
            self.edge_type_bias = None

        # Learnable global scale for V normalisation
        if use_v_norm:
            self.v_scale = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.v_scale = None

        # Temperature module (canonical approach from temperature.py)
        self.temperature_module = create_temperature_module(
            mode=temperature_mode,
            dim=dim,
            n_heads=n_heads,
            temperature=1.0,
            min_temperature=min_temperature,
        )

    def forward(
        self,
        x: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``[N, D]`` — Node features (unbatched; graph attention is
               defined over a single graph at a time).
            senders: ``[E]`` int64 — Source node index for each edge.
            receivers: ``[E]`` int64 — Destination node index for each edge.
            edge_type: ``[E]`` int64 — Edge type index in
                ``[0, num_edge_types)``.  Required when
                ``num_edge_types > 0``, ignored otherwise.

        Returns:
            ``[N, D]`` — Aggregated attention output *without* residual
            connection (add residual in the enclosing block, as with
            :class:`MultiHeadAttention`).
        """
        num_nodes = x.shape[0]
        H, d = self.n_heads, self.head_dim

        # Project to Q, K, V — [N, H, d]
        q = self.q_proj(x).view(num_nodes, H, d)
        k = self.k_proj(x).view(num_nodes, H, d)
        v = self.v_proj(x).view(num_nodes, H, d)

        # Gather per-edge sender Q, receiver K and receiver V — [E, H, d]
        q_e = q[senders]
        k_e = k[receivers]
        v_e = v[receivers]

        # Optional per-type additive bias on value vectors
        if self.edge_type_bias is not None:
            if edge_type is None:
                raise ValueError(
                    "edge_type must be provided when num_edge_types > 0"
                )
            v_e = v_e + self.edge_type_bias[edge_type]  # [E, H, d]

        # Optional V normalisation with learnable global scale
        if self.use_v_norm:
            v_norms = v_e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v_e = self.v_scale * v_e / v_norms

        # Attention scores: ⟨Q_sender, K_receiver⟩ — [E, H]
        attn_scores = (q_e * k_e).sum(dim=-1)

        # Apply temperature using canonical temperature module
        # Temperature module expects [B, H, N, G], we have [E, H]
        attn_scores_4d = attn_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, E, H]
        _, attn_scores_4d = self.temperature_module(attn_scores_4d, x.unsqueeze(0))
        attn_scores = attn_scores_4d.squeeze(0).squeeze(0)  # [E, H]

        # Sparse softmax: normalise over all edges arriving at the same receiver
        # scatter_softmax groups by receivers along dim=0 (the edge dimension)
        attn_weights = scatter_softmax(
            attn_scores, receivers, dim=0, dim_size=num_nodes
        )  # [E, H]
        attn_weights = self.dropout(attn_weights)

        # Weighted values: [E, H, d] collapsed to [E, D]
        weighted_v = (attn_weights.unsqueeze(-1) * v_e).view(-1, self.dim)

        # Aggregate to receiver nodes: [N, D]
        aggregated = aggregate_edges(weighted_v, receivers, num_nodes, method='sum')

        return self.out_proj(aggregated)

    def set_epoch(self, epoch: int):
        """Set current epoch for temperature annealing (if using annealed mode)."""
        if hasattr(self.temperature_module, 'set_epoch'):
            self.temperature_module.set_epoch(epoch)


class PhysicsTokenAttention(nn.Module):
    """
    Transolver-style slice-attention-deslice attention.
    
    Reduces complexity from O(N^2) to O(G^2) where G << N (physics tokens).
    This implementation supports both paper-faithful mode and extended features.
    
    Paper Reference:
        Wu et al. "Transolver: A Fast Transformer Solver for PDEs on General 
        Geometries." ICML 2024.
    
    Architecture (from paper):
        1. Slice: Project N mesh points to G physics-aware tokens
        2. Attention: Self-attention among the G tokens (O(G^2))
        3. Deslice: Map tokens back to N points
    
    Args:
        dim: Model dimension
        n_tokens: Number of physics tokens (G in paper, called slice_num in original)
        n_heads: Number of attention heads
        dropout: Dropout rate
        temperature: Initial temperature value
        temperature_mode: Temperature scheduling mode
        use_gumbel_softmax: Enable Gumbel-Softmax for differentiable slicing (Transolver++)
        min_temperature: Minimum temperature for learnable modes
        
        # Paper Fidelity Options
        use_slice_normalization: If True (default), apply slice normalization from paper.
            Critical for proper token aggregation - divides by number of points per slice.
        use_learnable_tokens: If False (default, paper-faithful), tokens are computed
            purely from input. If True, adds learnable token bias (framework extension).
        qkv_mode: 'direct' (paper-faithful) uses per-head Q/K/V linears,
            'multihead' uses framework's MultiHeadAttention wrapper.
        use_orthogonal_init: If True, apply orthogonal initialization to slice
            projection (recommended in paper for better geometric structure).
            
        # Annealing parameters (for 'annealed' temperature mode)
        anneal_warmup_epochs: Epochs before annealing starts
        anneal_factor: Temperature decay factor per epoch
        anneal_final_temp: Final temperature after annealing
    """
    
    def __init__(
        self,
        dim: int,
        n_tokens: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
        temperature: float = 0.5,
        temperature_mode: str = 'learnable_scalar',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        # Paper fidelity options
        use_slice_normalization: bool = True,
        use_learnable_tokens: bool = False,
        qkv_mode: str = 'direct',
        use_orthogonal_init: bool = True,
        # Annealing parameters (for 'annealed' mode)
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
    ):
        super().__init__()
        
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        
        self.dim = dim
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.inner_dim = self.head_dim * n_heads
        self.temperature_mode = temperature_mode
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_slice_normalization = use_slice_normalization
        self.use_learnable_tokens = use_learnable_tokens
        self.qkv_mode = qkv_mode
        
        # Paper uses two-branch projection:
        # - in_project_x: for computing slice weights
        # - in_project_fx: for computing slice content
        if qkv_mode == 'direct':
            # Paper-faithful: two separate projections
            self.in_project_x = nn.Linear(dim, self.inner_dim)
            self.in_project_fx = nn.Linear(dim, self.inner_dim)
        else:
            # Framework style: single content projection
            self.slice_content_proj = nn.Linear(dim, dim)
            
        # Slice projection: head_dim -> n_tokens (G)
        self.slice_weight_proj = nn.Linear(self.head_dim, n_tokens)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.slice_weight_proj.weight)
        
        # Optional learnable token bias (framework extension, NOT in paper)
        if use_learnable_tokens:
            self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        else:
            self.register_buffer('tokens', None)
        
        # Token attention
        if qkv_mode == 'direct':
            # Paper-faithful: per-head Q/K/V linears
            self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.scale = self.head_dim ** -0.5
            self.token_attention = None
        else:
            # Framework style: MultiHeadAttention wrapper
            self.to_q = self.to_k = self.to_v = None
            self.token_attention = MultiHeadAttention(dim, n_heads, dropout)
            
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Temperature mechanism
        self.temperature_module = create_temperature_module(
            mode=temperature_mode,
            dim=dim,
            n_heads=n_heads,
            temperature=temperature,
            min_temperature=min_temperature,
            anneal_warmup_epochs=anneal_warmup_epochs,
            anneal_factor=anneal_factor,
            anneal_final_temp=anneal_final_temp,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] or [B, N, D] - Input features
            
        Returns:
            [N, D] or [B, N, D] - Processed features
        """
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
        
        B, N, D = x.shape
        H = self.n_heads
        G = self.n_tokens
        d = self.head_dim
        
        # --- Slice: Project N points to G tokens ---
        
        if self.qkv_mode == 'direct':
            # Paper-faithful two-branch projection
            # Branch 1: for slice weights (x)
            x_mid = self.in_project_x(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
            # Branch 2: for slice content (fx)
            fx_mid = self.in_project_fx(x).reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
        else:
            # Framework style: single projection for content
            x_mid = x.reshape(B, N, H, d).permute(0, 2, 1, 3)  # [B, H, N, d]
            fx_mid = self.slice_content_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        
        # Compute slice weights: [B, H, N, G]
        slice_logits = self.slice_weight_proj(x_mid)
        
        # Optional Gumbel-Softmax noise (Transolver++ feature)
        if self.use_gumbel_softmax and self.training:
            epsilon = torch.rand_like(slice_logits)
            gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
            slice_logits = slice_logits - gumbel_noise
        
        # Apply temperature mechanism
        temperature, slice_logits = self.temperature_module(slice_logits, x)
        
        slice_weights = torch.softmax(slice_logits, dim=-1)  # [B, H, N, G]
        
        # Slice to tokens: weighted aggregation
        # [B, H, N, d] @ [B, H, N, G].T -> [B, H, G, d]
        tokens = torch.einsum('bhnd,bhng->bhgd', fx_mid, slice_weights)
        
        # Slice normalization (CRITICAL from paper!)
        # Normalizes by number of points assigned to each slice
        if self.use_slice_normalization:
            slice_norm = slice_weights.sum(dim=2, keepdim=True)  # [B, H, 1, G]
            tokens = tokens / (slice_norm.transpose(-2, -1) + 1e-5)  # [B, H, G, d]
        
        # --- Attention: Process G tokens ---
        
        if self.qkv_mode == 'direct':
            # Paper-faithful: direct QKV attention on tokens
            # tokens: [B, H, G, d]
            q = self.to_q(tokens)  # [B, H, G, d]
            k = self.to_k(tokens)  # [B, H, G, d]
            v = self.to_v(tokens)  # [B, H, G, d]
            
            # Scaled dot-product attention
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, H, G, G]
            attn = torch.softmax(dots, dim=-1)
            tokens_out = torch.matmul(attn, v)  # [B, H, G, d]
        else:
            # Framework style: use MultiHeadAttention wrapper
            tokens = tokens.permute(0, 2, 1, 3).reshape(B, G, D)  # [B, G, D]
            if self.tokens is not None:
                tokens = tokens + self.tokens
            tokens_out = self.token_attention(tokens)  # [B, G, D]
            tokens_out = tokens_out.reshape(B, G, H, d).permute(0, 2, 1, 3)  # [B, H, G, d]
        
        # --- Deslice: Distribute tokens back to N points ---
        
        # [B, H, G, d] @ [B, H, N, G].T -> [B, H, N, d]
        out = torch.einsum('bhgd,bhng->bhnd', tokens_out, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.inner_dim)  # [B, N, inner_dim]
        
        out = self.to_out(out)
        
        if single_batch:
            out = out.squeeze(0)
        
        return out
    
    def set_epoch(self, epoch: int):
        """Set current epoch for temperature annealing schedules."""
        if hasattr(self.temperature_module, 'set_epoch'):
            self.temperature_module.set_epoch(epoch)



# =============================================================================
# Transolver-3: Scaling to Industrial-Scale Geometries
# =============================================================================

class TiledSliceOperation(nn.Module):
    """
    Geometry Slice Tiling for Transolver-3.
    
    Partitions the computation of slice weights into tiles to avoid materializing
    the full N x M slice weight matrix in memory. This reduces memory from 
    O(N * M) to O(N * tile_size).
    
    Reference: Transolver-3 Section 3.2 "Geometry Slice Tiling"
    (arXiv:2602.04940)
    
    Args:
        tile_size: Size of each tile along the mesh dimension N.
            Smaller values save more memory but increase computation time.
            Default 100000 is paper's recommended balance point.
        use_gradient_checkpointing: If True, use gradient checkpointing for tiles
            to trade computation for memory during training.
    """
    
    def __init__(
        self,
        tile_size: int = 100000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    def forward(
        self,
        fx_mid: torch.Tensor,  # [B, H, N, d] - content features
        x_mid: torch.Tensor,   # [B, H, N, d] - weight features
        slice_weight_proj: nn.Linear,  # Projects d -> G
        temperature_module: Optional[nn.Module] = None,
        use_gumbel_softmax: bool = False,
        training: bool = True,
        use_slice_normalization: bool = True,
    ):
        """
        Compute sliced tokens using tiling.
        
        Args:
            fx_mid: Content features [B, H, N, d]
            x_mid: Weight features [B, H, N, d]  
            slice_weight_proj: Linear layer mapping d -> n_tokens
            temperature_module: Optional temperature scheduling module
            use_gumbel_softmax: Whether to add Gumbel noise
            training: Whether in training mode
            use_slice_normalization: Whether to normalize by slice counts
            
        Returns:
            tokens: [B, H, G, d] - Sliced tokens
            slice_weights_full: [B, H, N, G] - Full slice weights (for deslice)
        """
        B, H, N, d = x_mid.shape
        G = slice_weight_proj.out_features
        device = x_mid.device
        
        # If N is small enough, don't tile
        if N <= self.tile_size:
            return self._compute_slice(
                fx_mid, x_mid, slice_weight_proj,
                temperature_module, use_gumbel_softmax, training,
                use_slice_normalization
            )
        
        # Initialize accumulated tokens and normalization
        tokens = torch.zeros(B, H, G, d, device=device, dtype=x_mid.dtype)
        slice_norm = torch.zeros(B, H, 1, G, device=device, dtype=x_mid.dtype)
        
        # Store slice weights for deslice (if needed)
        slice_weights_list = []
        
        # Process in tiles
        num_tiles = (N + self.tile_size - 1) // self.tile_size
        
        for i in range(num_tiles):
            start_idx = i * self.tile_size
            end_idx = min((i + 1) * self.tile_size, N)
            
            # Get tile
            fx_tile = fx_mid[:, :, start_idx:end_idx, :]  # [B, H, tile_N, d]
            x_tile = x_mid[:, :, start_idx:end_idx, :]    # [B, H, tile_N, d]
            
            # Compute tile contribution with optional gradient checkpointing
            if self.use_gradient_checkpointing and training:
                from torch.utils.checkpoint import checkpoint
                tile_tokens, tile_norm, tile_weights = checkpoint(
                    self._compute_tile,
                    fx_tile, x_tile, slice_weight_proj,
                    temperature_module, use_gumbel_softmax, training,
                    use_slice_normalization,
                    use_reentrant=False,
                )
            else:
                tile_tokens, tile_norm, tile_weights = self._compute_tile(
                    fx_tile, x_tile, slice_weight_proj,
                    temperature_module, use_gumbel_softmax, training,
                    use_slice_normalization,
                )
            
            # Accumulate
            tokens += tile_tokens
            slice_norm += tile_norm
            slice_weights_list.append(tile_weights)
        
        # Final normalization
        if use_slice_normalization:
            tokens = tokens / (slice_norm.transpose(-2, -1) + 1e-5)
        
        # Concatenate slice weights for deslice
        slice_weights_full = torch.cat(slice_weights_list, dim=2)  # [B, H, N, G]
        
        return tokens, slice_weights_full
    
    def _compute_tile(
        self,
        fx_tile: torch.Tensor,
        x_tile: torch.Tensor,
        slice_weight_proj: nn.Linear,
        temperature_module: Optional[nn.Module],
        use_gumbel_softmax: bool,
        training: bool,
        use_slice_normalization: bool,
    ):
        """Compute slice contribution for a single tile."""
        # Compute slice weights for tile
        slice_logits = slice_weight_proj(x_tile)  # [B, H, tile_N, G]
        
        # Optional Gumbel-Softmax
        if use_gumbel_softmax and training:
            epsilon = torch.rand_like(slice_logits)
            gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
            slice_logits = slice_logits - gumbel_noise
        
        # Apply temperature
        if temperature_module is not None:
            _, slice_logits = temperature_module(slice_logits, fx_tile)
        
        slice_weights = torch.softmax(slice_logits, dim=-1)  # [B, H, tile_N, G]
        
        # Compute token contribution: [B, H, tile_N, d] @ [B, H, tile_N, G].T -> sum
        tile_tokens = torch.einsum('bhnd,bhng->bhgd', fx_tile, slice_weights)
        
        # Compute normalization contribution
        tile_norm = slice_weights.sum(dim=2, keepdim=True)  # [B, H, 1, G]
        
        return tile_tokens, tile_norm, slice_weights
    
    def _compute_slice(
        self,
        fx_mid: torch.Tensor,
        x_mid: torch.Tensor,
        slice_weight_proj: nn.Linear,
        temperature_module: Optional[nn.Module],
        use_gumbel_softmax: bool,
        training: bool,
        use_slice_normalization: bool,
    ):
        """Compute slice without tiling (for small N)."""
        # Compute slice weights
        slice_logits = slice_weight_proj(x_mid)  # [B, H, N, G]
        
        # Optional Gumbel-Softmax
        if use_gumbel_softmax and training:
            epsilon = torch.rand_like(slice_logits)
            gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
            slice_logits = slice_logits - gumbel_noise
        
        # Apply temperature
        if temperature_module is not None:
            _, slice_logits = temperature_module(slice_logits, x_mid)
        
        slice_weights = torch.softmax(slice_logits, dim=-1)  # [B, H, N, G]
        
        # Slice to tokens
        tokens = torch.einsum('bhnd,bhng->bhgd', fx_mid, slice_weights)
        
        # Slice normalization
        if use_slice_normalization:
            slice_norm = slice_weights.sum(dim=2, keepdim=True)  # [B, H, 1, G]
            tokens = tokens / (slice_norm.transpose(-2, -1) + 1e-5)
        
        return tokens, slice_weights


class PhysicsTokenAttentionV3(PhysicsTokenAttention):
    """
    Transolver-3 optimized Physics Token Attention.
    
    Implements the architectural optimizations from Transolver-3 (arXiv:2602.04940):
    1. **Geometry Slice Tiling**: Partitions computation to avoid materializing 
       the full N x M slice weight matrix, reducing memory from O(N*M) to O(N*tile_size).
    
    These optimizations enable single-GPU training on meshes up to ~2.9M cells and
    inference on industrial-scale geometries exceeding 100M cells.
    
    Paper: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"
    arXiv:2602.04940
    
    Args:
        Same as PhysicsTokenAttention, plus:
        use_tiling: Enable geometry slice tiling for memory efficiency
        tile_size: Size of each tile. Default 100k is paper's recommended balance.
        use_gradient_checkpointing: Trade computation for memory during training
    """
    
    def __init__(
        self,
        dim: int,
        n_tokens: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
        temperature: float = 0.5,
        temperature_mode: str = 'learnable_scalar',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        # Paper fidelity options
        use_slice_normalization: bool = True,
        use_learnable_tokens: bool = False,
        qkv_mode: str = 'direct',
        use_orthogonal_init: bool = True,
        # Annealing parameters
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
        # Transolver-3 specific optimizations
        use_tiling: bool = True,
        tile_size: int = 100000,
        use_gradient_checkpointing: bool = True,
    ):
        # Initialize parent class
        super().__init__(
            dim=dim,
            n_tokens=n_tokens,
            n_heads=n_heads,
            dropout=dropout,
            temperature=temperature,
            temperature_mode=temperature_mode,
            use_gumbel_softmax=use_gumbel_softmax,
            min_temperature=min_temperature,
            use_slice_normalization=use_slice_normalization,
            use_learnable_tokens=use_learnable_tokens,
            qkv_mode=qkv_mode,
            use_orthogonal_init=use_orthogonal_init,
            anneal_warmup_epochs=anneal_warmup_epochs,
            anneal_factor=anneal_factor,
            anneal_final_temp=anneal_final_temp,
        )
        
        self.use_tiling = use_tiling
        
        # Initialize tiling module
        if use_tiling:
            self.tiled_slice = TiledSliceOperation(
                tile_size=tile_size,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            self.tiled_slice = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Transolver-3 optimizations.
        
        Args:
            x: [N, D] or [B, N, D] - Input features
            
        Returns:
            [N, D] or [B, N, D] - Processed features
        """
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
        
        B, N, D = x.shape
        H = self.n_heads
        G = self.n_tokens
        d = self.head_dim
        
        # --- Slice: Project N points to G tokens ---
        
        if self.qkv_mode == 'direct':
            # Two-branch projection (paper-faithful)
            x_mid = self.in_project_x(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
            fx_mid = self.in_project_fx(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        else:
            x_mid = x.reshape(B, N, H, d).permute(0, 2, 1, 3)
            fx_mid = self.slice_content_proj(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        
        # Apply tiling or standard slice
        if self.use_tiling and self.tiled_slice is not None and N > self.tiled_slice.tile_size:
            tokens, slice_weights = self.tiled_slice(
                fx_mid, x_mid, self.slice_weight_proj,
                self.temperature_module, self.use_gumbel_softmax,
                self.training, self.use_slice_normalization,
            )
            # Normalization already applied in tiled slice if enabled
            if not self.use_slice_normalization:
                # Need to recompute slice weights for deslice
                slice_logits = self.slice_weight_proj(x_mid)
                if self.use_gumbel_softmax and self.training:
                    epsilon = torch.rand_like(slice_logits)
                    gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
                    slice_logits = slice_logits - gumbel_noise
                _, slice_logits = self.temperature_module(slice_logits, x)
                slice_weights = torch.softmax(slice_logits, dim=-1)
        else:
            # Standard slice (from parent class logic)
            slice_logits = self.slice_weight_proj(x_mid)
            
            if self.use_gumbel_softmax and self.training:
                epsilon = torch.rand_like(slice_logits)
                gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
                slice_logits = slice_logits - gumbel_noise
            
            temperature, slice_logits = self.temperature_module(slice_logits, x)
            slice_weights = torch.softmax(slice_logits, dim=-1)
            
            tokens = torch.einsum('bhnd,bhng->bhgd', fx_mid, slice_weights)
            
            if self.use_slice_normalization:
                slice_norm = slice_weights.sum(dim=2, keepdim=True)
                tokens = tokens / (slice_norm.transpose(-2, -1) + 1e-5)
        
        # --- Attention: Process G tokens ---
        
        if self.qkv_mode == 'direct':
            q = self.to_q(tokens)
            k = self.to_k(tokens)
            v = self.to_v(tokens)
            
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = torch.softmax(dots, dim=-1)
            tokens_out = torch.matmul(attn, v)
        else:
            tokens = tokens.permute(0, 2, 1, 3).reshape(B, G, D)
            if self.tokens is not None:
                tokens = tokens + self.tokens
            tokens_out = self.token_attention(tokens)
            tokens_out = tokens_out.reshape(B, G, H, d).permute(0, 2, 1, 3)
        
        # --- Deslice: Distribute tokens back to N points ---
        
        out = torch.einsum('bhgd,bhng->bhnd', tokens_out, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.inner_dim)
        
        out = self.to_out(out)
        
        if single_batch:
            out = out.squeeze(0)
        
        return out
