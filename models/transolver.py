"""
Transolver: physics-attention transformer for PDEs on general geometries.

Reusable model wrapping the framework's :class:`PhysicsTokenAttention`
(slice-attend-deslice) so that examples can build a Transolver in one call
instead of redefining blocks and preprocessing inline.

Reference:
    Wu, H., Hu, T., Luo, H., Wang, J., & Long, M. (2024).
    "Transolver: A Fast Transformer Solver for PDEs on General Geometries."
    ICML 2024. https://arxiv.org/abs/2402.02366

A ``variant='v3'`` option selects the tiled, memory-efficient attention
(:class:`PhysicsTokenAttentionV3`) from Transolver-3 for industrial-scale meshes.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..core import MLP, AutoRegisterModel
from ..components.attention import PhysicsTokenAttention, PhysicsTokenAttentionV3

_ACTIVATIONS = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


class TransolverBlock(nn.Module):
    """Transolver transformer block: physics-attention + feed-forward, both residual.

    Args:
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        dropout: Dropout rate.
        act: Activation name for the feed-forward MLP.
        mlp_ratio: Feed-forward hidden expansion ratio.
        slice_num: Number of physics tokens (slices).
        last_layer: If True, append a final projection to ``out_dim``.
        out_dim: Output dimension when ``last_layer`` is True.
        variant: ``'v2'`` for :class:`PhysicsTokenAttention`, ``'v3'`` for the
            tiled :class:`PhysicsTokenAttentionV3`.
        tile_size / use_tiling / use_gradient_checkpointing: forwarded to the v3 attention.
    """

    def __init__(
        self,
        n_hidden: int,
        n_head: int = 8,
        dropout: float = 0.0,
        act: str = 'gelu',
        mlp_ratio: float = 1.0,
        slice_num: int = 32,
        last_layer: bool = False,
        out_dim: int = 1,
        variant: str = 'v2',
        use_tiling: bool = True,
        tile_size: int = 100_000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(n_hidden)
        attn_kwargs = dict(
            dim=n_hidden,
            n_tokens=slice_num,
            n_heads=n_head,
            dropout=dropout,
            temperature=0.5,
            temperature_mode='learnable_scalar',
            use_slice_normalization=True,
            use_learnable_tokens=False,
            qkv_mode='direct',
            use_orthogonal_init=True,
        )
        if variant == 'v3':
            self.attn = PhysicsTokenAttentionV3(
                use_tiling=use_tiling,
                tile_size=tile_size,
                use_gradient_checkpointing=use_gradient_checkpointing,
                **attn_kwargs,
            )
        else:
            self.attn = PhysicsTokenAttention(**attn_kwargs)

        self.ln_2 = nn.LayerNorm(n_hidden)
        self.mlp = MLP(
            in_dim=n_hidden,
            out_dim=n_hidden,
            hidden_dims=[int(n_hidden * mlp_ratio)],
            activation=act,
            use_layer_norm=False,
        )

        if last_layer:
            self.ln_3 = nn.LayerNorm(n_hidden)
            self.mlp2 = nn.Linear(n_hidden, out_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(AutoRegisterModel, name='transolver', aliases=['physics_attention']):
    """Transolver model using the framework's physics-attention components.

    Architecture: optional unified position encoding (distances to a reference
    grid) -> MLP preprocessor -> ``n_layers`` Transolver blocks -> projection.

    Args:
        space_dim: Spatial dimension of node positions.
        n_layers: Number of Transolver blocks.
        n_hidden: Hidden dimension.
        dropout: Dropout rate.
        n_head: Number of attention heads.
        act: Activation name.
        mlp_ratio: Feed-forward expansion ratio.
        fun_dim: Input function (field) dimension.
        out_dim: Output dimension.
        slice_num: Number of physics tokens.
        ref: Reference grid size per axis for unified position encoding.
        unified_pos: Enable distance-to-grid position encoding (2D only).
        grid_bounds: ((xmin, xmax), (ymin, ymax)) for the reference grid.
        variant: ``'v2'`` (paper) or ``'v3'`` (tiled, industrial scale).
        use_tiling / tile_size / use_gradient_checkpointing: v3 options.

    Example:
        >>> model = Transolver(space_dim=2, fun_dim=1, out_dim=1, n_hidden=128)
        >>> x = torch.randn(2, 1000, 1); pos = torch.randn(2, 1000, 2)
        >>> y = model(x, pos)  # [2, 1000, 1]
    """

    def __init__(
        self,
        space_dim: int = 2,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = 'gelu',
        mlp_ratio: float = 1.0,
        fun_dim: int = 1,
        out_dim: int = 1,
        slice_num: int = 32,
        ref: int = 8,
        unified_pos: bool = True,
        grid_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2.0, 4.0), (-1.5, 1.5)),
        variant: str = 'v2',
        use_tiling: bool = True,
        tile_size: int = 100_000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if unified_pos and space_dim != 2:
            raise ValueError("unified_pos position encoding is only supported for space_dim=2")
        if variant not in ('v2', 'v3'):
            raise ValueError(f"variant must be 'v2' or 'v3', got {variant!r}")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.ref = ref
        self.unified_pos = unified_pos
        self.grid_bounds = grid_bounds
        self.variant = variant

        self.act = _ACTIVATIONS.get(act, nn.GELU)()

        preproc_input = fun_dim + space_dim + (ref * ref if unified_pos else 0)
        self.preprocess = MLP(
            in_dim=preproc_input,
            out_dim=n_hidden,
            hidden_dims=[n_hidden * 2],
            activation=act,
            use_layer_norm=False,
        )

        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

        self.blocks = nn.ModuleList([
            TransolverBlock(
                n_hidden=n_hidden,
                n_head=n_head,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1),
                out_dim=out_dim,
                variant=variant,
                use_tiling=use_tiling,
                tile_size=tile_size,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            for i in range(n_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_grid(self, pos: torch.Tensor) -> torch.Tensor:
        """Distances from each node to each point of a ref x ref reference grid."""
        batchsize = pos.shape[0]
        device = pos.device
        (xmin, xmax), (ymin, ymax) = self.grid_bounds

        gridx = torch.tensor(np.linspace(xmin, xmax, self.ref), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(ymin, ymax, self.ref), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref ** 2, 2)

        diff = pos[:, :, None, :] - grid_ref[:, None, :, :]
        return torch.sqrt(torch.sum(diff ** 2, dim=-1))

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, fun_dim] input function values.
            pos: [B, N, space_dim] node positions.
        Returns:
            [B, N, out_dim] predictions.
        """
        B, N, _ = x.shape
        if self.unified_pos:
            x = torch.cat((x, self.get_grid(pos)), dim=-1)
        x = torch.cat((x, pos), dim=-1)

        fx = self.preprocess(x.reshape(-1, x.shape[-1])).reshape(B, N, self.n_hidden)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)
        return fx

    def set_epoch(self, epoch: int):
        """Forward epoch to attention modules for temperature scheduling."""
        for block in self.blocks:
            if hasattr(block.attn, 'set_epoch'):
                block.attn.set_epoch(epoch)

    def save_config(self):
        return {
            'model_type': 'transolver',
            'variant': self.variant,
            'space_dim': self.space_dim,
            'n_hidden': self.n_hidden,
            'n_layers': len(self.blocks),
            'ref': self.ref,
            'unified_pos': self.unified_pos,
        }
