"""
Transformer processor with optional physics tokens.

Standard multi-head attention or Transolver-style slice-attention-deslice.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import warnings

from ..core.mlp import MLP
from ..core.graph import GraphsTuple
from . import attention as _attention

__all__ = [
    'PhysicsTokenConfig',
    'RelativePositionConfig',
    'TransformerBlock',
    'TransformerProcessor',
]


@dataclass
class PhysicsTokenConfig:
    """Settings for Transolver-style physics-token (slice) attention.

    Consumed by :class:`TransformerBlock` / :class:`TransformerProcessor` only
    when ``use_physics_tokens=True``; ignored otherwise.
    """

    n_tokens: int = 32
    temperature: float = 0.5
    temperature_mode: str = 'learnable_scalar'
    use_gumbel_softmax: bool = False
    min_temperature: float = 0.1
    anneal_warmup_epochs: int = 5
    anneal_factor: float = 0.98
    anneal_final_temp: float = 0.05
    use_slice_normalization: bool = True
    use_learnable_tokens: bool = False
    qkv_mode: str = 'direct'
    use_orthogonal_init: bool = True


@dataclass
class RelativePositionConfig:
    """Settings for relative position encoding in standard attention.

    Consumed by :class:`TransformerBlock` / :class:`TransformerProcessor` only
    when ``use_physics_tokens=False``; ignored when physics tokens are active.
    """

    position_dim: int = 2
    max_distance: float = 10.0
    num_position_buckets: int = 32
    position_encoding_type: str = 'learned'


# Field names backing the legacy flat-kwarg API, derived from the configs so the
# two never drift apart.
_PHYSICS_TOKEN_PARAMS = frozenset(f.name for f in fields(PhysicsTokenConfig))
_POSITION_PARAMS = frozenset(f.name for f in fields(RelativePositionConfig))
_warned_configs = set()


def _resolve_transformer_configs(
    use_physics_tokens: bool,
    physics_token_config: Optional[PhysicsTokenConfig],
    position_config: Optional[RelativePositionConfig],
    legacy_kwargs: dict,
) -> tuple[PhysicsTokenConfig, RelativePositionConfig]:
    """Merge config objects with legacy per-parameter kwargs into a config pair.

    This is the single place that understands the relationship between the two
    attention modes and their parameter groups. It:

    * routes each legacy kwarg into its owning config (or raises ``TypeError``
      for unknown names),
    * rejects passing both a config object and overlapping legacy kwargs,
    * emits one ``UserWarning`` listing parameters that are silently ignored
      given the active mode.

    Returns the resolved ``(PhysicsTokenConfig, RelativePositionConfig)`` pair.
    """
    physics_overrides = {}
    position_overrides = {}
    unknown = []
    for key, value in legacy_kwargs.items():
        if key in _PHYSICS_TOKEN_PARAMS:
            physics_overrides[key] = value
        elif key in _POSITION_PARAMS:
            position_overrides[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise TypeError(f"Unexpected keyword arguments: {sorted(unknown)}")

    if physics_token_config is not None and physics_overrides:
        raise TypeError(
            "Pass physics-token settings via physics_token_config OR as keyword "
            f"arguments, not both. Conflicting keys: {sorted(physics_overrides)}"
        )
    if position_config is not None and position_overrides:
        raise TypeError(
            "Pass position settings via position_config OR as keyword arguments, "
            f"not both. Conflicting keys: {sorted(position_overrides)}"
        )

    physics_cfg = physics_token_config or PhysicsTokenConfig()
    if physics_overrides:
        physics_cfg = replace(physics_cfg, **physics_overrides)
    position_cfg = position_config or RelativePositionConfig()
    if position_overrides:
        position_cfg = replace(position_cfg, **position_overrides)

    # Diagnose settings that do not apply to the active attention mode.
    if use_physics_tokens:
        ignored = sorted(position_overrides)
        if position_config is not None:
            ignored.append('position_config')
    else:
        ignored = sorted(physics_overrides)
        if physics_token_config is not None:
            ignored.append('physics_token_config')
    if ignored:
        key = (use_physics_tokens, tuple(ignored))
        if key not in _warned_configs:
            _warned_configs.add(key)
            warnings.warn(
                f"Ignored parameters when use_physics_tokens={use_physics_tokens}: {ignored}",
                UserWarning,
                stacklevel=3,
            )
    return physics_cfg, position_cfg



class TransformerBlock(nn.Module):
    """
    Transformer block with optional physics token attention and relative position encoding.
    
    When use_physics_tokens=True, uses Transolver-style slice-attention-deslice
    attention which reduces complexity from O(N^2) to O(G^2) where G << N.

    Mode-specific settings are grouped into two config objects:

    * ``physics_token_config`` (:class:`PhysicsTokenConfig`) — active when
      ``use_physics_tokens=True``.
    * ``position_config`` (:class:`RelativePositionConfig`) — active when
      ``use_physics_tokens=False``.

    The individual fields of these configs may also be passed as flat keyword
    arguments for backward compatibility; passing a setting that does not apply
    to the active mode raises a ``UserWarning``.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_physics_tokens: bool = False,
        use_relative_positions: bool = False,
        physics_token_config: Optional[PhysicsTokenConfig] = None,
        position_config: Optional[RelativePositionConfig] = None,
        **legacy_kwargs,
    ):
        super().__init__()

        # Resolve the two parameter groups (config objects + legacy flat kwargs)
        # and warn about anything that does not apply to the active mode.
        physics_cfg, position_cfg = _resolve_transformer_configs(
            use_physics_tokens, physics_token_config, position_config, legacy_kwargs,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.use_physics_tokens = use_physics_tokens
        self.use_relative_positions = use_relative_positions
        
        if use_physics_tokens:
            self.attn = _attention.PhysicsTokenAttention(
                dim=dim,
                n_tokens=physics_cfg.n_tokens,
                n_heads=n_heads,
                dropout=dropout,
                temperature=physics_cfg.temperature,
                temperature_mode=physics_cfg.temperature_mode,
                use_gumbel_softmax=physics_cfg.use_gumbel_softmax,
                min_temperature=physics_cfg.min_temperature,
                anneal_warmup_epochs=physics_cfg.anneal_warmup_epochs,
                anneal_factor=physics_cfg.anneal_factor,
                anneal_final_temp=physics_cfg.anneal_final_temp,
                use_slice_normalization=physics_cfg.use_slice_normalization,
                use_learnable_tokens=physics_cfg.use_learnable_tokens,
                qkv_mode=physics_cfg.qkv_mode,
                use_orthogonal_init=physics_cfg.use_orthogonal_init,
            )
        else:
            self.attn = _attention.MultiHeadAttention(
                dim, n_heads, dropout,
                use_relative_positions=use_relative_positions,
                position_dim=position_cfg.position_dim,
                max_distance=position_cfg.max_distance,
                num_position_buckets=position_cfg.num_position_buckets,
                position_encoding_type=position_cfg.position_encoding_type,
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

    Mode-specific settings are grouped into ``physics_token_config``
    (:class:`PhysicsTokenConfig`) and ``position_config``
    (:class:`RelativePositionConfig`); the active config is forwarded to every
    block. Their fields may also be passed as flat keyword arguments for
    backward compatibility.
    """

    def __init__(
        self,
        latent_dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_physics_tokens: bool = False,
        use_checkpoint: bool = False,
        use_relative_positions: bool = False,
        physics_token_config: Optional[PhysicsTokenConfig] = None,
        position_config: Optional[RelativePositionConfig] = None,
        **legacy_kwargs,
    ):
        super().__init__()

        # Resolve parameter groups once, then hand the active config to every
        # block so the per-block kwargs are not duplicated at the call site.
        physics_cfg, position_cfg = _resolve_transformer_configs(
            use_physics_tokens, physics_token_config, position_config, legacy_kwargs,
        )
        block_physics_cfg = physics_cfg if use_physics_tokens else None
        block_position_cfg = None if use_physics_tokens else position_cfg

        self.use_checkpoint = use_checkpoint
        self.use_relative_positions = use_relative_positions
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=latent_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_physics_tokens=use_physics_tokens,
                use_relative_positions=use_relative_positions,
                physics_token_config=block_physics_cfg,
                position_config=block_position_cfg,
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
