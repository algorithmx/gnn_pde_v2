"""
Example: Transolver (Physics-Attention Mechanism)

Demonstrates the reusable :class:`gnn_pde_v2.models.Transolver` model, which
recreates Transolver from https://github.com/thuml/Transolver using the
framework's :class:`PhysicsTokenAttention`.

Reference:
    Wu, H., Hu, T., Luo, H., Wang, J., & Long, M. (2024).
    "Transolver: A Fast Transformer Solver for PDEs on General Geometries."
    ICML 2024. https://arxiv.org/abs/2402.02366

Key Innovation:
    Physics-Attention reduces complexity from O(N^2) to O(N*G + G^2) where
    G << N (number of physics tokens).
"""

import torch

from gnn_pde_v2.core import AutoRegisterModel
from gnn_pde_v2.models import Transolver, TransolverBlock  # re-export for tests

__all__ = ["Transolver", "TransolverBlock", "example_usage"]


def example_usage():
    """Build a paper-default Transolver (AirfRANS-style) and run one forward pass."""
    print("=" * 60)
    print("Transolver Example using gnn_pde_v2 Framework")
    print("=" * 60)

    model = Transolver(
        space_dim=2, n_layers=5, n_hidden=256, n_head=8, act='gelu',
        mlp_ratio=1.0, fun_dim=1, out_dim=1, slice_num=32, ref=8, unified_pos=True,
    )

    x = torch.randn(2, 1000, 1)
    pos = torch.randn(2, 1000, 2)
    output = model(x, pos)

    print(f"\nHidden dim: {model.n_hidden} | Layers: {len(model.blocks)} | Ref: {model.ref}x{model.ref}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input {tuple(x.shape)} + pos {tuple(pos.shape)} -> {tuple(output.shape)}")

    attn = model.blocks[0].attn
    print(f"Attention: {type(attn).__name__} (qkv_mode={attn.qkv_mode}, temp={attn.temperature_mode})")
    print("Available models:", AutoRegisterModel.list_models())
    return model, x, pos, output


if __name__ == "__main__":
    example_usage()
