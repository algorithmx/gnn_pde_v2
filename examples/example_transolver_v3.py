"""
Transolver-3 Example: Scaling to Industrial-Scale Geometries

Demonstrates the ``variant='v3'`` mode of :class:`gnn_pde_v2.models.Transolver`,
which uses the tiled :class:`PhysicsTokenAttentionV3`, plus two scaling helpers:
geometry-amortized training and physical-state caching for massive-mesh inference.

Paper: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"
Authors: Hang Zhou, Haixu Wu, et al. (Tsinghua University)

Key Innovation:
- Handles meshes with 160M+ cells via geometry slice tiling
- Memory complexity O(N * tile_size) instead of O(N * M)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from gnn_pde_v2.components import PhysicsTokenAttentionV3
from gnn_pde_v2.models import Transolver


class TransolverV3(nn.Module):
    """Thin wrapper over ``Transolver(variant='v3')`` accepting a combined input.

    Input is ``[B, N, space_dim + input_dim]`` (or unbatched ``[N, ...]``) where
    the first ``space_dim`` columns are coordinates; the rest are fields.
    """

    def __init__(
        self,
        space_dim: int = 3,
        input_dim: int = 4,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_tiling: bool = True,
        tile_size: int = 100000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = Transolver(
            space_dim=space_dim, fun_dim=input_dim, out_dim=output_dim,
            n_hidden=hidden_dim, n_layers=num_layers, n_head=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, slice_num=slice_num,
            unified_pos=False, variant='v3', use_tiling=use_tiling,
            tile_size=tile_size, use_gradient_checkpointing=use_gradient_checkpointing,
        )

    @property
    def blocks(self):
        return self.model.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        pos, fields = x[..., :self.space_dim], x[..., self.space_dim:]
        out = self.model(fields, pos)
        return out.squeeze(0) if single else out

    def set_epoch(self, epoch: int):
        self.model.set_epoch(epoch)


class GeometryAmortizedTraining:
    """Sample random mesh subsets per iteration to train on meshes exceeding GPU memory."""

    def __init__(self, full_mesh_size: int, subset_size: int, seed: Optional[int] = None):
        self.full_mesh_size = full_mesh_size
        self.subset_size = min(subset_size, full_mesh_size)
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def get_subset_indices(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.randperm(self.full_mesh_size, generator=self.rng, device=device)[:self.subset_size]

    def apply_subset(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        idx = self.get_subset_indices(x.device)
        return x[idx], (y[idx] if y is not None else None)


class PhysicalStateCache:
    """Build per-layer physics-token states via chunked processing for huge meshes."""

    def __init__(self, model: nn.Module, chunk_size: int = 50000, device: Optional[torch.device] = None):
        self.model = model
        self.chunk_size = chunk_size
        self.device = device or next(model.parameters()).device

    def build_cache(self, x: torch.Tensor, num_layers: int) -> List[torch.Tensor]:
        return [self._compute_layer_state(x, self._get_layer(i)) for i in range(num_layers)]

    def _compute_layer_state(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        N = x.shape[0]
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        acc_state = acc_norm = None
        for i in range(num_chunks):
            chunk = x[i * self.chunk_size: min((i + 1) * self.chunk_size, N)].to(self.device)
            s, n = self._compute_chunk_contribution(chunk, layer)
            acc_state = s if acc_state is None else acc_state + s
            acc_norm = n if acc_norm is None else acc_norm + n
        return acc_state / (acc_norm + 1e-5)

    def _compute_chunk_contribution(self, chunk: torch.Tensor, layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = next((m for m in layer.modules() if isinstance(m, PhysicsTokenAttentionV3)), None)
        if attn is None:
            raise ValueError("Layer does not contain PhysicsTokenAttentionV3")
        with torch.no_grad():
            B, N, D = 1, chunk.shape[0], chunk.shape[1]
            H, d = attn.n_heads, attn.head_dim
            cb = chunk.unsqueeze(0)
            x_mid = attn.in_project_x(cb).reshape(B, N, H, d).permute(0, 2, 1, 3)
            fx_mid = attn.in_project_fx(cb).reshape(B, N, H, d).permute(0, 2, 1, 3)
            logits = attn.slice_weight_proj(x_mid)
            _, logits = attn.temperature_module(logits, cb)
            w = torch.softmax(logits, dim=-1)
            tokens = torch.einsum('bhnd,bhng->bhgd', fx_mid, w).sum(dim=1)
            norm = w.sum(dim=2, keepdim=True).sum(dim=1)
            return tokens.squeeze(0), norm.squeeze(0)

    def _get_layer(self, idx: int) -> nn.Module:
        for attr in ('blocks', 'transformer_blocks', 'layers'):
            if hasattr(self.model, attr):
                return getattr(self.model, attr)[idx]
        raise ValueError("Cannot find transformer layers in model")


def demo_memory_efficiency():
    print("=" * 60)
    print("Transolver-3 Memory Efficiency Demo")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for mesh_size in (10000, 100000, 500000):
        attn = PhysicsTokenAttentionV3(dim=256, n_heads=8, n_tokens=64,
                                       use_tiling=True, tile_size=min(50000, mesh_size // 2 + 1)).to(device)
        out = attn(torch.randn(1, mesh_size, 256, device=device))
        print(f"  Mesh {mesh_size:,}: tiled forward ok -> {tuple(out.shape)}")


def demo_amortized_training():
    print("\nTransolver-3 Geometry Amortized Training Demo")
    amortizer = GeometryAmortizedTraining(10_000_000, 400_000, seed=42)
    for it in range(3):
        print(f"  Iter {it+1}: sampled {len(amortizer.get_subset_indices()):,} points")


def demo_model_creation():
    print("\nTransolver-3 Model Creation Demo")
    model = TransolverV3(space_dim=3, input_dim=4, output_dim=4, hidden_dim=256,
                         num_layers=8, num_heads=8, slice_num=64)
    out = model(torch.randn(1, 10000, 7))
    print(f"  Forward: input (1, 10000, 7) -> {tuple(out.shape)}, tiling={model.blocks[0].attn.use_tiling}")


if __name__ == "__main__":
    demo_memory_efficiency()
    demo_amortized_training()
    demo_model_creation()
    print("\nAll demos completed!")
