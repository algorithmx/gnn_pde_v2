"""
Transformer processor with optional physics tokens.

Standard multi-head attention or Transolver-style slice-attention-deslice.
"""

from typing import Any, Optional

from torch import Tensor

import torch
import torch.nn as nn
import math

from ..core.mlp import MLP
from ..core.graph import GraphsTuple
from ..core.protocols import Modulation, ConditioningProtocol  # re-exported for backwards compat


# =============================================================================
# Conditioning Protocol
# =============================================================================
# Modulation and ConditioningProtocol are defined in core/protocols.py and
# imported above. They are re-exported here so that existing code that does
# ``from gnn_pde_v2.components.transformer import ConditioningProtocol``
# continues to work without modification.


class ZeroConditioning(ConditioningProtocol):
    """Identity conditioning - no modulation applied."""

    def forward(self, condition: Any = None) -> Modulation:
        return Modulation()


class AdaLNConditioning(ConditioningProtocol):
    """Single-source AdaLN conditioning."""

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


class DualAdaLNConditioning(ConditioningProtocol):
    """Dual-source AdaLN conditioning (Unisolver-style: μ + f)."""

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
        mu = condition[..., : self.mu_dim]
        f = condition[..., self.mu_dim :]

        params_mu = self.proj_mu(mu).chunk(6, dim=-1)
        params_f = self.proj_f(f).chunk(6, dim=-1)

        return Modulation(
            shift=torch.cat([params_mu[0], params_f[0]], dim=-1),
            scale=torch.cat([params_mu[1], params_f[1]], dim=-1),
            gate=torch.cat([params_mu[2], params_f[2]], dim=-1),
        )


class FiLMConditioning(ConditioningProtocol):
    """FiLM-style conditioning (feature-wise linear modulation)."""

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
    """Standard multi-head self-attention."""
    
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, N, D] or [N, D]
            mask: Optional attention mask
        """
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
        
        B, N, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, N, N]
        
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
    """
    
    def __init__(
        self,
        dim: int,
        n_tokens: int = 32,
        n_heads: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.temperature = temperature
        
        # Learnable physics tokens
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        
        # Two-branch projection for slice weights
        self.slice_weight_proj = nn.Linear(dim, n_heads * n_tokens)
        self.slice_content_proj = nn.Linear(dim, dim)
        
        # Attention on tokens
        self.token_attention = MultiHeadAttention(dim, n_heads)
        
        # Deslice projection
        self.deslice_proj = nn.Linear(dim, dim)
    
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
        slice_logits = slice_logits / self.temperature
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


class TransformerBlock(nn.Module):
    """
    Transformer block with optional physics token attention.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_physics_tokens: bool = False,
        n_tokens: int = 32,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        
        if use_physics_tokens:
            self.attn = PhysicsTokenAttention(dim, n_tokens, n_heads)
        else:
            self.attn = MultiHeadAttention(dim, n_heads, dropout)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, D] or [B, N, D]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerProcessor(nn.Module):
    """
    Transformer-based processor for graph nodes.

    Can use full attention or physics-token attention for efficiency.
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
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=latent_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_physics_tokens=use_physics_tokens,
                n_tokens=n_tokens,
            )
            for _ in range(n_layers)
        ])

        self.use_physics_tokens = use_physics_tokens
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """Process nodes through transformer blocks."""
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for TransformerProcessor")
        
        nodes = graph.nodes
        
        # Process through transformer blocks
        for block in self.blocks:
            nodes = block(nodes)
        
        return graph.replace(nodes=nodes)
