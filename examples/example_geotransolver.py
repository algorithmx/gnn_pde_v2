"""
Example: GeoTransolver (Geometry-Aware Physics Attention Transformer)

This example implements the GeoTransolver model from:
https://arxiv.org/abs/2512.20399

Original Work Reference:
------------------------
Adams, C., Ranade, R., Cherukuri, R., & Choudhry, S. (2025).
"GeoTransolver: Learning Physics on Irregular Domains using Multi-scale
Geometry Aware Physics Attention Transformer."
arXiv:2512.20399. NVIDIA Corporation.

Key Innovation:
---------------
GeoTransolver extends Transolver's Physics-Attention with GALE (Geometry-Aware
Latent Embeddings), which combines:
1. Self-attention on learned physical state slices (from Transolver)
2. Cross-attention to geometry/global context embeddings
3. Learnable mixing between self- and cross-attention outputs
4. Multi-scale local geometric features via ball queries

This implementation uses gnn_pde_v2 framework components where available:
- PhysicsTokenAttention: For the core slice-attend-deslice mechanism
- MLP: For feed-forward networks and context projection
- AutoRegisterModel: For model registration

Insufficient support from the present package:
----------------------------------------------
1. **No cross-attention module for context**: PhysicsTokenAttention only supports
   self-attention among slice tokens. GeoTransolver needs cross-attention between
   slice tokens and external context (geometry/global embeddings). Implemented
   manually with linear projections + scaled_dot_product_attention.

2. **No context projector / ContextProjector**: The framework lacks a mechanism to
   project geometry/global features onto learned physical state slices (the
   "slice-only" half of PhysicsTokenAttention without deslice). Implemented as
   a lightweight custom module reusing the temperature system.

3. **No multi-scale ball query / BQWarp**: GeoTransolver uses radius-based ball
   queries at multiple scales for local geometric feature extraction. The framework
   has knn_graph (torch_cluster) but no radius-based neighbor search. Implemented
   via pairwise distance + topk approximation.

4. **No GALE attention module**: The framework provides PhysicsTokenAttention
   (slice-attend-deslice) but no variant that accepts external context for
   cross-attention blending. Implemented as GeoPhysicsTokenAttention wrapping
   the framework's PhysicsTokenAttention with added cross-attention.

5. **No global context builder**: No orchestrator for combining geometry
   tokenization, global embedding tokenization, and multi-scale local features
   into a unified context tensor. Implemented as MinimalContextBuilder.

6. **No concrete dropout**: The reference implementation uses ConcreteDropout;
   the framework provides standard nn.Dropout only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from gnn_pde_v2.core import AutoRegisterModel, MLP
from gnn_pde_v2.components.attention import PhysicsTokenAttention


class ContextProjector(nn.Module):
    """Project context features onto learned physical state slices.

    This is the "slice-only" half of physics attention: it projects input
    context (geometry features, global embeddings) onto learned physical states
    without projecting back (no deslice). The resulting slice tokens serve as
    cross-attention context in every GALE block.

    Similar to the ContextProjector in the reference implementation but
    using framework components (MLP, temperature) where possible.

    Args:
        dim: Input feature dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        slice_num: Number of learned physical state slices.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        slice_num: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] - Context features (geometry or global)

        Returns:
            [B, H, S, D] - Slice tokens for cross-attention context
        """
        B, N, C = x.shape
        H = self.heads
        d = self.dim_head
        S = self.in_project_slice.out_features

        x_mid = self.in_project_x(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        fx_mid = self.in_project_fx(x).reshape(B, N, H, d).permute(0, 2, 1, 3)

        slice_logits = self.in_project_slice(x_mid)
        temp = torch.clamp(self.temperature, min=0.1, max=5.0).to(slice_logits.dtype)
        slice_weights = torch.softmax(slice_logits / temp, dim=-1)

        # Weighted aggregation: (B, H, S, N) @ (B, H, N, d) -> (B, H, S, d)
        slice_norm = slice_weights.sum(dim=2)  # [B, H, S]
        normed_weights = slice_weights / (slice_norm[:, :, None, :] + 1e-5)
        slice_tokens = torch.matmul(normed_weights.transpose(2, 3), fx_mid)

        return slice_tokens


class MultiScaleBallQuery(nn.Module):
    """Approximate multi-scale ball query for local geometric features.

    Uses pairwise distance + topk to approximate radius-based neighbor search
    (the framework lacks BQWarp or radius_graph). For each scale (radius),
    finds the k nearest neighbors within the radius and processes their
    relative features via MLP.

    Args:
        radii: List of query radii for each scale.
        neighbors: List of max neighbors per radius (same length as radii).
        feature_dim: Dimension of input features.
        hidden_dim: Output dimension per scale.
    """

    def __init__(
        self,
        radii: List[float],
        neighbors: List[int],
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        assert len(radii) == len(neighbors)
        self.radii = radii
        self.neighbors = neighbors
        self.num_scales = len(radii)

        self.mlps = nn.ModuleList([
            MLP(
                in_dim=(feature_dim + 3) * neighbors[i],
                out_dim=hidden_dim,
                hidden_dims=[hidden_dim * 2],
                activation='gelu',
                use_layer_norm=False,
            )
            for i in range(self.num_scales)
        ])

    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_pos: [B, N, 3] - Query point positions
            key_pos: [B, M, 3] - Key point positions (geometry)
            key_features: [B, M, F] - Key point features

        Returns:
            [B, N, num_scales * hidden_dim] - Multi-scale local features
        """
        B, N, _ = query_pos.shape
        M = key_pos.shape[1]
        dists = torch.cdist(query_pos, key_pos)

        outputs = []
        for s in range(self.num_scales):
            radius = self.radii[s]
            k = min(self.neighbors[s], M)

            masked_dists = dists.clone()
            masked_dists[dists > radius] = float('inf')

            valid_counts = (masked_dists < float('inf')).sum(dim=-1)
            if (valid_counts == 0).any():
                masked_dists = dists

            _, indices = masked_dists.topk(k, dim=-1, largest=False)

            # key_features: [B, M, F] -> gather -> [B, N, k, F]
            nb_features = torch.gather(
                key_features.unsqueeze(1).expand(-1, N, -1, -1),
                dim=2,
                index=indices.unsqueeze(-1).expand(-1, -1, -1, key_features.shape[-1]),
            )
            nb_pos = torch.gather(
                key_pos.unsqueeze(1).expand(-1, N, -1, -1),
                dim=2,
                index=indices.unsqueeze(-1).expand(-1, -1, -1, 3),
            )
            rel_pos = nb_pos - query_pos.unsqueeze(2)

            nb_input = torch.cat([nb_features, rel_pos], dim=-1)
            nb_flat = nb_input.reshape(B, N, -1)

            out = self.mlps[s](nb_flat)
            outputs.append(out)

        return torch.cat(outputs, dim=-1)


class GeoPhysicsTokenAttention(nn.Module):
    """Geometry-Aware Physics Token Attention (GALE-like).

    Combines:
    1. Self-attention on learned physics tokens (via PhysicsTokenAttention)
    2. Cross-attention to geometry/global context
    3. Learnable mixing between the two

    This wraps the framework's PhysicsTokenAttention for self-attention and
    adds a custom cross-attention path for context integration.

    Args:
        dim: Model dimension.
        n_tokens: Number of physics tokens (slices).
        n_heads: Number of attention heads.
        context_dim: Dimension of context features for cross-attention.
        dropout: Dropout rate.
        temperature: Initial temperature for slice weight computation.
    """

    def __init__(
        self,
        dim: int,
        n_tokens: int = 32,
        n_heads: int = 8,
        context_dim: int = 0,
        dropout: float = 0.0,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        dim_head = self.head_dim
        self.has_context = context_dim > 0

        self.physics_attn = PhysicsTokenAttention(
            dim=dim,
            n_tokens=n_tokens,
            n_heads=n_heads,
            dropout=dropout,
            temperature=temperature,
            temperature_mode='learnable_scalar',
            use_slice_normalization=True,
            use_learnable_tokens=False,
            qkv_mode='direct',
            use_orthogonal_init=True,
        )

        if self.has_context:
            self.cross_q = nn.Linear(dim, dim)
            self.cross_k = nn.Linear(context_dim, dim_head)
            self.cross_v = nn.Linear(context_dim, dim_head)
            self.cross_out = nn.Linear(dim, dim)
            self.cross_dropout = nn.Dropout(dropout)
            self.state_mixing = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - Input features
            context: [B, H, S, D_c] - Context slice tokens (from ContextProjector)
                or None for self-attention only.

        Returns:
            [B, N, D] - Output features
        """
        self_out = self.physics_attn(x)

        if not self.has_context or context is None:
            return self_out

        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim
        S = context.shape[2]

        # Q from input: [B, N, D] -> [B, H, N, d]
        q = self.cross_q(x).reshape(B, N, H, d).permute(0, 2, 1, 3)
        # K, V from context: [B, H, S, D_c] -> project -> [B, H, S, d]
        k = self.cross_k(context)
        v = self.cross_v(context)

        scale = d ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, S]
        attn = torch.softmax(attn, dim=-1)
        cross_out = torch.matmul(attn, v)  # [B, H, N, d]

        cross_out = cross_out.permute(0, 2, 1, 3).reshape(B, N, D)
        cross_out = self.cross_out(cross_out)
        cross_out = self.cross_dropout(cross_out)

        alpha = torch.sigmoid(self.state_mixing)
        return (1 - alpha) * self_out + alpha * cross_out


class GeoTransolverBlock(nn.Module):
    """Single GeoTransolver transformer block (GALE block).

    Architecture: LayerNorm -> GALE attention -> residual -> LayerNorm -> FFN -> residual

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of attention heads.
        slice_num: Number of physics tokens.
        context_dim: Context dimension for cross-attention.
        dropout: Dropout rate.
        mlp_ratio: FFN expansion ratio.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        slice_num: int = 32,
        context_dim: int = 0,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = GeoPhysicsTokenAttention(
            dim=hidden_dim,
            n_tokens=slice_num,
            n_heads=num_heads,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dims=[int(hidden_dim * mlp_ratio)],
            activation='gelu',
            dropout=dropout,
            use_layer_norm=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            context: [B, H, S, D_c] or None

        Returns:
            [B, N, D]
        """
        x = self.attn(self.ln_1(x), context) + x
        x = self.mlp(self.ln_2(x)) + x
        return x


class GeoTransolver(AutoRegisterModel, name='geotransolver', namespace='example'):
    """GeoTransolver: Geometry-Aware Physics Attention Transformer.

    Extends Transolver with GALE (Geometry-Aware Latent Embeddings):
    - Physics-Attention for self-attention on learned state slices
    - Cross-attention to geometry/global context in every block
    - Multi-scale local geometric features
    - Persistent geometry conditioning throughout depth

    Architecture:
        1. Context building:
           - Geometry/global features -> ContextProjector -> shared context tokens
           - Multi-scale ball queries -> local features (appended to input)
        2. Input projection: MLP(input_features) -> hidden_dim
        3. GALE blocks x N:
           - Physics-Attention (self-attention on slices)
           - Cross-attention to shared context
           - Learnable mixing
           - FFN with residual
        4. Output projection: LayerNorm -> Linear -> output

    Args:
        functional_dim: Input feature dimension.
        out_dim: Output dimension.
        geometry_dim: Geometry feature dimension (for context). None to disable.
        global_dim: Global embedding dimension (for context). None to disable.
        n_layers: Number of GALE transformer blocks.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        slice_num: Number of learned physics tokens per head.
        dropout: Dropout rate.
        mlp_ratio: FFN expansion ratio.
        include_local_features: Whether to use multi-scale local features.
        radii: Ball query radii for local features.
        neighbors_in_radius: Max neighbors per radius.
        n_hidden_local: Hidden dim for local feature MLPs.
    """

    def __init__(
        self,
        functional_dim: int = 4,
        out_dim: int = 4,
        geometry_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        n_head: int = 8,
        slice_num: int = 32,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        include_local_features: bool = False,
        radii: Optional[List[float]] = None,
        neighbors_in_radius: Optional[List[int]] = None,
        n_hidden_local: int = 32,
    ):
        super().__init__()

        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]

        assert n_hidden % n_head == 0, (
            f"n_hidden ({n_hidden}) must be divisible by n_head ({n_head})"
        )

        self.n_hidden = n_hidden
        self.include_local_features = include_local_features
        self.geometry_dim = geometry_dim
        self.global_dim = global_dim

        dim_head = n_hidden // n_head
        context_dim = 0

        if include_local_features and geometry_dim is not None:
            self.local_feature_extractor = MultiScaleBallQuery(
                radii=radii,
                neighbors=neighbors_in_radius,
                feature_dim=geometry_dim,
                hidden_dim=n_hidden_local,
            )
            local_feat_dim = n_hidden_local * len(radii)
        else:
            self.local_feature_extractor = None
            local_feat_dim = 0

        if geometry_dim is not None:
            self.geometry_projector = ContextProjector(
                dim=geometry_dim,
                heads=n_head,
                dim_head=dim_head,
                slice_num=slice_num,
                dropout=dropout,
            )
            context_dim += dim_head
        else:
            self.geometry_projector = None

        if global_dim is not None:
            self.global_projector = ContextProjector(
                dim=global_dim,
                heads=n_head,
                dim_head=dim_head,
                slice_num=slice_num,
                dropout=dropout,
            )
            context_dim += dim_head
        else:
            self.global_projector = None

        input_dim = functional_dim + local_feat_dim
        self.preprocess = MLP(
            in_dim=input_dim,
            out_dim=n_hidden,
            hidden_dims=[n_hidden * 2],
            activation='gelu',
            use_layer_norm=False,
        )

        self.blocks = nn.ModuleList([
            GeoTransolverBlock(
                hidden_dim=n_hidden,
                num_heads=n_head,
                slice_num=slice_num,
                context_dim=context_dim,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(n_hidden)
        self.output_proj = nn.Linear(n_hidden, out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        local_embedding: torch.Tensor,
        local_positions: Optional[torch.Tensor] = None,
        global_embedding: Optional[torch.Tensor] = None,
        geometry: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            local_embedding: [B, N, functional_dim] - Point features
            local_positions: [B, N, 3] - Point positions (required for local features)
            global_embedding: [B, 1, global_dim] - Global conditioning
            geometry: [B, M, geometry_dim] - Geometry point features (e.g., surface normals)

        Returns:
            [B, N, out_dim] - Predictions
        """
        B, N, _ = local_embedding.shape

        context_parts = []

        x = local_embedding
        if (self.local_feature_extractor is not None
                and geometry is not None
                and local_positions is not None):
            local_feats = self.local_feature_extractor(
                local_positions, geometry, geometry
            )
            x = torch.cat([x, local_feats], dim=-1)

        # Geometry context: [B, H, S, d]
        if self.geometry_projector is not None and geometry is not None:
            context_parts.append(self.geometry_projector(geometry))

        # Global context: [B, H, S, d]
        if self.global_projector is not None and global_embedding is not None:
            context_parts.append(self.global_projector(global_embedding))

        # Concatenate along last dim: [B, H, S, D_c]
        if context_parts:
            context_for_blocks = torch.cat(context_parts, dim=-1)
        else:
            context_for_blocks = None

        fx = self.preprocess(x)

        for block in self.blocks:
            fx = block(fx, context_for_blocks)

        return self.output_proj(self.output_norm(fx))

    def save_config(self):
        return {
            'model_type': 'geotransolver',
            'n_hidden': self.n_hidden,
            'n_layers': len(self.blocks),
            'geometry_dim': self.geometry_dim,
            'global_dim': self.global_dim,
            'include_local_features': self.include_local_features,
        }


def example_usage():
    """Demonstrate GeoTransolver on a synthetic unstructured mesh problem."""
    print("=" * 60)
    print("GeoTransolver Example using gnn_pde_v2 Framework")
    print("=" * 60)

    model = GeoTransolver(
        functional_dim=4,
        out_dim=4,
        geometry_dim=3,
        global_dim=8,
        n_layers=4,
        n_hidden=128,
        n_head=8,
        slice_num=32,
        dropout=0.0,
        mlp_ratio=4.0,
        include_local_features=True,
        radii=[0.1, 0.5],
        neighbors_in_radius=[8, 32],
        n_hidden_local=16,
    )

    batch_size = 2
    n_points = 500
    n_geom = 200

    local_embedding = torch.randn(batch_size, n_points, 4)
    local_positions = torch.randn(batch_size, n_points, 3)
    geometry = torch.randn(batch_size, n_geom, 3)
    global_embedding = torch.randn(batch_size, 1, 8)

    output = model(
        local_embedding,
        local_positions=local_positions,
        geometry=geometry,
        global_embedding=global_embedding,
    )

    print(f"\nModel Configuration:")
    print(f"  Hidden dimension: {model.n_hidden}")
    print(f"  Number of layers: {len(model.blocks)}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nInput/Output:")
    print(f"  Local embedding:  {local_embedding.shape}")
    print(f"  Local positions:  {local_positions.shape}")
    print(f"  Geometry:         {geometry.shape}")
    print(f"  Global embedding: {global_embedding.shape}")
    print(f"  Output:           {output.shape}")

    block = model.blocks[0]
    print(f"\nFramework Integration:")
    print(f"  Physics-Attention: {type(block.attn.physics_attn).__name__}")
    print(f"  Cross-attention:   {'enabled' if block.attn.has_context else 'disabled'}")
    print(f"  Context projector: {'geometry' if model.geometry_projector else 'none'}"
          f" + {'global' if model.global_projector else 'none'}")
    print(f"  Local features:    {'enabled' if model.local_feature_extractor else 'disabled'}")

    print(f"\nInsufficient Package Support:")
    print(f"  1. No built-in cross-attention for context integration")
    print(f"     -> Implemented manually: GeoPhysicsTokenAttention")
    print(f"  2. No ContextProjector (slice-only physics attention)")
    print(f"     -> Implemented: ContextProjector")
    print(f"  3. No multi-scale ball query (BQWarp)")
    print(f"     -> Implemented: MultiScaleBallQuery (distance + topk)")
    print(f"  4. No GALE attention module")
    print(f"     -> Implemented: GeoPhysicsTokenAttention")
    print(f"  5. No global context builder orchestrator")
    print(f"     -> Implemented inline in GeoTransolver.forward()")

    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", [m for m in AutoRegisterModel.list_models() if 'geo' in m])
    print("=" * 60)

    return model, output


if __name__ == "__main__":
    model, output = example_usage()
