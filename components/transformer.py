"""
Transformer processor with optional physics tokens.

Standard multi-head attention or Transolver-style slice-attention-deslice.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.mlp import MLP
from ..core.graph import GraphsTuple
from . import attention as _attention

__all__ = ['TransformerBlock', 'TransformerProcessor']


# =============================================================================
# Helper Functions
# =============================================================================



class TransformerBlock(nn.Module):
    """
    Transformer block with optional physics token attention and relative position encoding.
    
    When use_physics_tokens=True, uses Transolver-style slice-attention-deslice
    attention which reduces complexity from O(N^2) to O(G^2) where G << N.
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
        temperature: float = 0.5,
        temperature_mode: str = 'learnable_scalar',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
        # Paper-fidelity parameters (only used when use_physics_tokens=True)
        use_slice_normalization: bool = True,
        use_learnable_tokens: bool = False,
        qkv_mode: str = 'direct',
        use_orthogonal_init: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.use_physics_tokens = use_physics_tokens
        self.use_relative_positions = use_relative_positions
        
        if use_physics_tokens:
            self.attn = _attention.PhysicsTokenAttention(
                dim=dim,
                n_tokens=n_tokens,
                n_heads=n_heads,
                dropout=dropout,
                temperature=temperature,
                temperature_mode=temperature_mode,
                use_gumbel_softmax=use_gumbel_softmax,
                min_temperature=min_temperature,
                anneal_warmup_epochs=anneal_warmup_epochs,
                anneal_factor=anneal_factor,
                anneal_final_temp=anneal_final_temp,
                use_slice_normalization=use_slice_normalization,
                use_learnable_tokens=use_learnable_tokens,
                qkv_mode=qkv_mode,
                use_orthogonal_init=use_orthogonal_init,
            )
        else:
            self.attn = _attention.MultiHeadAttention(
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
        temperature: float = 0.5,
        temperature_mode: str = 'learnable_scalar',
        use_gumbel_softmax: bool = False,
        min_temperature: float = 0.1,
        anneal_warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
        anneal_final_temp: float = 0.05,
        # Paper-fidelity parameters (only used when use_physics_tokens=True)
        use_slice_normalization: bool = True,
        use_learnable_tokens: bool = False,
        qkv_mode: str = 'direct',
        use_orthogonal_init: bool = True,
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
                use_slice_normalization=use_slice_normalization,
                use_learnable_tokens=use_learnable_tokens,
                qkv_mode=qkv_mode,
                use_orthogonal_init=use_orthogonal_init,
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
