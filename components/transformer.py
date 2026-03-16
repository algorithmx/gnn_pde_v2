"""
Transformer processor with optional physics tokens.

Standard multi-head attention or Transolver-style slice-attention-deslice.
"""

from typing import Any, Optional, Union

from torch import Tensor

import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.checkpoint import checkpoint

from ..core.mlp import MLP
from ..core.graph import GraphsTuple
from ..core.protocols import Modulation, ConditioningProtocol
from .temperature import create_temperature_module, TemperatureBase


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
        position = torch.arange(self.num_buckets).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.num_heads, 2).float() * 
                            (-math.log(10000.0) / self.num_heads))
        
        encodings = torch.zeros(self.num_heads, self.num_buckets)
        encodings[0::2, :] = torch.sin(position * div_term.unsqueeze(1)).T
        if self.num_heads > 1:
            encodings[1::2, :] = torch.cos(position * div_term.unsqueeze(1)).T
        
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
# Conditioning Protocol
# =============================================================================
# Modulation and ConditioningProtocol are defined in core/protocols.py.
# Import them from there: ``from gnn_pde_v2.core.protocols import ConditioningProtocol``


class ZeroConditioning(ConditioningProtocol[object]):
    """Identity conditioning — no modulation applied.

    Accepts (and ignores) any condition value, including ``None``.
    Suitable as a drop-in for any slot typed as
    ``ConditioningProtocol[T]`` for any ``T``.
    """

    def forward(self, condition: object = None) -> Modulation:  # type: ignore[override]
        return Modulation()


class AdaLNConditioning(ConditioningProtocol[Tensor]):
    """Single-source AdaLN conditioning.

    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        # 6 * out_dim: (shift, scale, gate) x 2 for attn+mlp
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * out_dim)
        )
        # Zero init for identity start
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, condition: Tensor) -> Modulation:
        params = self.proj(condition).chunk(6, dim=-1)
        return Modulation(
            shift=torch.cat([params[0], params[3]], dim=-1),
            scale=torch.cat([params[1], params[4]], dim=-1),
            gate=torch.cat([params[2], params[5]], dim=-1),
        )


class DualAdaLNConditioning(ConditioningProtocol[Tensor]):
    """Dual-source AdaLN conditioning (Unisolver-style: μ + f).

    Condition type: ``Tensor`` of shape ``[..., mu_dim + f_dim]``.
    The last dimension is split as ``condition[..., :mu_dim]`` (mean
    statistics μ) and ``condition[..., mu_dim:]`` (local field features
    f).  A ``ValueError`` will be raised at runtime if the last
    dimension does not equal ``mu_dim + f_dim``.
    """

    def __init__(
        self,
        mu_dim: int,
        f_dim: int,
        out_dim: int,
        split_ratio: float = 0.25,
    ):
        super().__init__()
        self.mu_dim = mu_dim
        self.f_dim = f_dim
        self.split_ratio = split_ratio

        mu_out = int(out_dim * split_ratio)
        f_out = out_dim - mu_out

        self.proj_mu = nn.Sequential(nn.SiLU(), nn.Linear(mu_dim, 6 * mu_out))
        self.proj_f = nn.Sequential(nn.SiLU(), nn.Linear(f_dim, 6 * f_out))

        # Zero init for identity start
        for proj in [self.proj_mu, self.proj_f]:
            nn.init.zeros_(proj[1].weight)
            nn.init.zeros_(proj[1].bias)

    def forward(self, condition: Tensor) -> Modulation:
        expected = self.mu_dim + self.f_dim
        if condition.shape[-1] != expected:
            raise ValueError(
                f"{type(self).__name__}.forward expects condition.shape[-1] == "
                f"mu_dim + f_dim == {expected}, got {condition.shape[-1]}."
            )
        mu = condition[..., : self.mu_dim]
        f = condition[..., self.mu_dim :]

        params_mu = self.proj_mu(mu).chunk(6, dim=-1)
        params_f = self.proj_f(f).chunk(6, dim=-1)

        return Modulation(
            shift=torch.cat([params_mu[0], params_f[0]], dim=-1),
            scale=torch.cat([params_mu[1], params_f[1]], dim=-1),
            gate=torch.cat([params_mu[2], params_f[2]], dim=-1),
        )


class FiLMConditioning(ConditioningProtocol[Tensor]):
    """FiLM-style conditioning (feature-wise linear modulation).

    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, out_dim)
        self.beta_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, condition: Tensor) -> Modulation:
        return Modulation(
            shift=self.beta_proj(condition),
            scale=self.gamma_proj(condition),
            gate=None,
        )


def _apply_modulation(x: Tensor, mod: Modulation) -> Tensor:
    """Apply modulation to tensor."""
    if mod.scale is not None:
        x = x * (1 + mod.scale)
    if mod.shift is not None:
        x = x + mod.shift
    return x


# =============================================================================
# Attention Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional relative position encoding.
    
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
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_relative_positions = use_relative_positions
        
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
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        
        # Add relative position bias if enabled
        if self.use_relative_positions:
            if positions is None:
                raise ValueError(
                    "positions must be provided when use_relative_positions=True"
                )
            position_bias = self.position_encoding(positions)  # [B, H, N, N] or [H, N, N]
            if position_bias.dim() == 3:
                position_bias = position_bias.unsqueeze(0)  # [1, H, N, N]
            scores = scores + position_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        out = self.out_proj(out)
        
        if single_batch:
            out = out.squeeze(0)
        
        return out


class PhysicsTokenAttention(nn.Module):
    """
    Transolver-style slice-attention-deslice attention.
    
    Reduces complexity from O(N^2) to O(G^2) where G << N (learnable physics tokens).
    
    Supports adaptive temperature mechanisms for controlling attention distribution
    sharpness based on local physical properties.
    
    Temperature modes:
        - 'fixed': Fixed temperature (backward compatible, default)
        - 'learnable_scalar': Global learnable temperature
        - 'per_head': Per-head learnable temperature
        - 'adaptive': Per-point adaptive temperature (Ada-Temp from Transolver++)
        - 'annealed': Training-time temperature annealing schedule
    """
    
    def __init__(
        self,
        dim: int,
        n_tokens: int = 32,
        n_heads: int = 8,
        temperature: float = 1.0,
        temperature_mode: str = 'fixed',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        # Annealing parameters (for 'annealed' mode)
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
    ):
        super().__init__()
        
        self.dim = dim
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.temperature_mode = temperature_mode
        self.use_gumbel_softmax = use_gumbel_softmax
        
        # Learnable physics tokens
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        
        # Two-branch projection for slice weights
        self.slice_weight_proj = nn.Linear(dim, n_heads * n_tokens)
        self.slice_content_proj = nn.Linear(dim, dim)
        
        # Attention on tokens
        self.token_attention = MultiHeadAttention(dim, n_heads)
        
        # Deslice projection
        self.deslice_proj = nn.Linear(dim, dim)
        
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
        
        # --- Slice: Project N points to G tokens ---
        
        # Slice weights: [B, N, H*G] -> [B, H, N, G]
        slice_logits = self.slice_weight_proj(x).reshape(B, N, H, G).permute(0, 2, 1, 3)
        
        # Optional Gumbel-Softmax noise (applied before temperature scaling)
        # Following Transolver++ Eq. 4: Rep-Slice(x, τ) = Softmax((Linear(x) - log(-log ε)) / τ)
        if self.use_gumbel_softmax and self.training:
            epsilon = torch.rand_like(slice_logits)
            gumbel_noise = -torch.log(-torch.log(epsilon + 1e-10) + 1e-10)
            slice_logits = slice_logits - gumbel_noise
        
        # Apply temperature mechanism
        temperature, slice_logits = self.temperature_module(slice_logits, x)
        
        slice_weights = torch.softmax(slice_logits, dim=-1)  # [B, H, N, G]
        
        # Content projection: [B, N, D] -> [B, N, D]
        content = self.slice_content_proj(x)
        content = content.reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]
        
        # Slice to tokens: [B, H, N, d] @ [B, H, N, G].T -> [B, H, G, d]
        tokens = torch.einsum('bhnd,bhng->bhgd', content, slice_weights)
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, G, D)  # [B, G, D]
        
        # Add learnable token bias
        tokens = tokens + self.tokens
        
        # --- Attention: Process G tokens ---
        tokens = self.token_attention(tokens)  # [B, G, D]
        
        # --- Deslice: Distribute tokens back to N points ---
        tokens = tokens.reshape(B, G, H, self.head_dim).permute(0, 2, 1, 3)  # [B, H, G, d]
        
        # Deslice: [B, H, G, d] @ [B, H, N, G] -> [B, H, N, d]
        out = torch.einsum('bhgd,bhng->bhnd', tokens, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        
        out = self.deslice_proj(out)
        
        if single_batch:
            out = out.squeeze(0)
        
        return out
    
    def set_epoch(self, epoch: int):
        """Set current epoch for temperature annealing schedules."""
        if hasattr(self.temperature_module, 'set_epoch'):
            self.temperature_module.set_epoch(epoch)


class TransformerBlock(nn.Module):
    """
    Transformer block with optional physics token attention and relative position encoding.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_physics_tokens: bool = False,
        n_tokens: int = 32,
        use_relative_positions: bool = False,
        position_dim: int = 2,
        max_distance: float = 10.0,
        num_position_buckets: int = 32,
        position_encoding_type: str = 'learned',
        # Temperature parameters (only used when use_physics_tokens=True)
        temperature: float = 1.0,
        temperature_mode: str = 'fixed',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.use_physics_tokens = use_physics_tokens
        self.use_relative_positions = use_relative_positions
        
        if use_physics_tokens:
            self.attn = PhysicsTokenAttention(
                dim=dim,
                n_tokens=n_tokens,
                n_heads=n_heads,
                temperature=temperature,
                temperature_mode=temperature_mode,
                use_gumbel_softmax=use_gumbel_softmax,
                min_temperature=min_temperature,
                anneal_warmup_epochs=anneal_warmup_epochs,
                anneal_factor=anneal_factor,
                anneal_final_temp=anneal_final_temp,
            )
        else:
            self.attn = MultiHeadAttention(
                dim, n_heads, dropout,
                use_relative_positions=use_relative_positions,
                position_dim=position_dim,
                max_distance=max_distance,
                num_position_buckets=num_position_buckets,
                position_encoding_type=position_encoding_type,
            )
        
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)

        # Use framework's MLP
        self.mlp = MLP(
            in_dim=dim,
            out_dim=dim,
            hidden_dims=[mlp_dim],
            activation='gelu',
            dropout=dropout,
            final_dropout=dropout,
            use_layer_norm=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, D] or [B, N, D] - Input features
            positions: [N, position_dim] or [B, N, position_dim] - Node positions
                      Required if use_relative_positions=True
        """
        if self.use_relative_positions and not self.use_physics_tokens:
            x = x + self.attn(self.norm1(x), positions=positions)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
    def set_epoch(self, epoch: int):
        """Set current epoch for temperature annealing schedules."""
        if self.use_physics_tokens and hasattr(self.attn, 'set_epoch'):
            self.attn.set_epoch(epoch)


class TransformerProcessor(nn.Module):
    """
    Transformer-based processor for graph nodes.

    Can use full attention or physics-token attention for efficiency.
    Supports relative position encoding when graph.positions is available.
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_physics_tokens: bool = False,
        n_tokens: int = 32,
        use_checkpoint: bool = False,
        use_relative_positions: bool = False,
        position_dim: int = 2,
        max_distance: float = 10.0,
        num_position_buckets: int = 32,
        position_encoding_type: str = 'learned',
        # Temperature parameters (only used when use_physics_tokens=True)
        temperature: float = 1.0,
        temperature_mode: str = 'fixed',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_relative_positions = use_relative_positions
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=latent_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_physics_tokens=use_physics_tokens,
                n_tokens=n_tokens,
                use_relative_positions=use_relative_positions,
                position_dim=position_dim,
                max_distance=max_distance,
                num_position_buckets=num_position_buckets,
                position_encoding_type=position_encoding_type,
                temperature=temperature,
                temperature_mode=temperature_mode,
                use_gumbel_softmax=use_gumbel_softmax,
                min_temperature=min_temperature,
                anneal_warmup_epochs=anneal_warmup_epochs,
                anneal_factor=anneal_factor,
                anneal_final_temp=anneal_final_temp,
            )
            for _ in range(n_layers)
        ])

        self.use_physics_tokens = use_physics_tokens
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Process nodes through transformer blocks.
        
        Args:
            graph: GraphsTuple with nodes and optionally positions
            
        Returns:
            GraphsTuple with updated nodes
        """
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for TransformerProcessor")
        
        nodes = graph.nodes
        positions = graph.positions if self.use_relative_positions else None
        
        # Check if positions are required but not provided
        if self.use_relative_positions and positions is None:
            raise ValueError(
                "use_relative_positions=True but graph.positions is None. "
                "Please provide positions in the GraphsTuple."
            )
        
        # Process through transformer blocks
        for block in self.blocks:
            if self.use_checkpoint:
                nodes = checkpoint(block, nodes, positions, use_reentrant=False)
            else:
                nodes = block(nodes, positions)
        
        return graph.replace(nodes=nodes)
    
    def set_epoch(self, epoch: int):
        """
        Set current epoch for temperature annealing schedules.
        
        Call this at the beginning of each training epoch when using
        temperature_mode='annealed'.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        for block in self.blocks:
            if hasattr(block, 'set_epoch'):
                block.set_epoch(epoch)
