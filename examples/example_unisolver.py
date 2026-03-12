"""
Example: Unisolver (PDE-Conditional Transformer)

This example recreates the Unisolver model from:
https://github.com/thu-ml/Unisolver

Original Work Reference:
------------------------
Zhou, H., Ma, Y., Wu, H., Wang, H., & Long, M. (2024).
"Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers."
International Conference on Machine Learning (ICML 2024).
Paper: https://arxiv.org/abs/2405.17527

Key Innovation:
---------------
Unisolver introduces decoupled AdaLN (Adaptive Layer Normalization) conditioning
for handling PDE parameters with different characteristics.

This implementation uses the gnn_pde_v2 framework components where applicable
while maintaining exact equivalence to the original Unisolver architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Import framework components
from gnn_pde_v2.core.base_model import BaseModel
from gnn_pde_v2.encoders.mlp_encoder import MLP


def modulate(x, shift, scale):
    """AdaLN modulation function."""
    return x * (1 + scale) + shift


class Unisolver(BaseModel, model_name='unisolver'):
    """
    Unisolver implementation using gnn_pde_v2 framework components.
    
    Precise equivalent of Unisolver model.
    Original implementation: HeterNS/models/Unisolver_HeterNS.py
    
    Key innovation: Decoupled AdaLN for domain-wise vs point-wise conditioning
    """
    
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_dim: int = 256,
        in_channels: int = 10,
        out_channels: int = 1,
        dim_head: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_height = patch_size
        self.patch_width = patch_size
        self.out_channels = out_channels
        
        self.patch_num_height = image_size // patch_size
        self.patch_num_width = image_size // patch_size
        
        # Reference grid size for unified position encoding
        self.ref = 4
        
        # Precompute position grid
        self.register_buffer('pos', self._get_grid([image_size, image_size], 'cpu'))
        
        # Patch dimension calculation
        patch_dim = (in_channels + self.ref * self.ref) * patch_size * patch_size
        
        # Mu embedder: Embeds scalar coefficient (domain-wise)
        self.mu_embedder = VisEmbedder(1, dim // 4)
        
        # Patch embedding for input field
        self.to_patch_embedding = nn.Sequential(
            self._rearrange_patches(patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Patch embedding for force field (point-wise)
        patch_dim_force = 1 * patch_size * patch_size
        self.to_patch_embedding_force = nn.Sequential(
            self._rearrange_patches_force(patch_size),
            nn.Linear(patch_dim_force, dim // 4),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer with dual conditioning
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        
        # Final layer with AdaLN modulation
        self.mlp_head = FinalLayer(mlp_dim, patch_size, out_channels)
    
    def _rearrange_patches(self, patch_size):
        """Create rearrange layer for patches."""
        return Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    
    def _rearrange_patches_force(self, patch_size):
        """Create rearrange layer for force field patches."""
        return Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    
    def _get_grid(self, shape, device):
        """Generate position grid with unified encoding."""
        size_x, size_y = shape[0], shape[1]
        batchsize = 1
        
        # Regular grid
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        
        grid = torch.cat((gridx, gridy), dim=-1).to(device)
        
        # Reference grid for unified encoding
        gridx_ref = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx_ref = gridx_ref.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        
        gridy_ref = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy_ref = gridy_ref.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        
        grid_ref = torch.cat((gridx_ref, gridy_ref), dim=-1).to(device)
        
        # Compute distances from each grid point to each reference point
        diff = grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]
        pos = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        pos = pos.reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        
        return pos
    
    def forward(self, x: torch.Tensor, mu: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, H, W, C] - Input field
            mu: [B] or [B, 1] - Scalar coefficient (domain-wise)
            f: [B, H, W] - Force field (point-wise)
            
        Returns:
            [B, H, W, out_channels] - Output prediction
        """
        B, H, W, C = x.shape
        
        # Embed mu (domain-wise conditioning)
        mu = mu[:, None] * 3200
        mu = self.mu_embedder(mu)[:, None, :]
        
        # Embed force field (point-wise conditioning)
        f = f[..., None]
        f = self.to_patch_embedding_force(f)
        
        # Broadcast mu to all patches
        mu = mu.repeat(1, f.shape[1], 1)
        
        # Get position grid and add to input
        if H != self.pos.shape[1] or W != self.pos.shape[2]:
            grid = self._get_grid([H, W], x.device)
            grid = grid.repeat(B, 1, 1, 1)
        else:
            grid = self.pos.repeat(B, 1, 1, 1)
        x = torch.cat((x, grid), dim=-1)
        
        # Permute for patch embedding
        x = x.permute(0, 3, 1, 2)
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        x = self.dropout(x)
        
        # Transformer with dual conditioning
        x = self.transformer(x, mu, f)
        
        # Final layer with AdaLN
        b, l, ch = x.shape
        x = self.mlp_head(x, mu, f)
        
        # Reshape back to image
        x = x.reshape(
            b, self.patch_num_height, self.patch_num_width,
            self.patch_height, self.patch_width, self.out_channels
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(b, self.patch_num_height * self.patch_height,
                     self.patch_num_width * self.patch_width, self.out_channels)
        
        return x
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'unisolver',
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'dim': self.transformer.layers_x[0][0].to_qkv.in_features,
            'depth': len(self.transformer.layers_x),
            'out_channels': self.out_channels,
        }


class Rearrange(nn.Module):
    """Rearrange tensor dimensions (einops-style)."""
    
    def __init__(self, pattern, **kwargs):
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs
    
    def forward(self, x):
        p1 = self.kwargs['p1']
        p2 = self.kwargs['p2']
        
        if self.pattern == 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)':
            B, C, H, W = x.shape
            h = H // p1
            w = W // p2
            x = x.reshape(B, C, h, p1, w, p2)
            x = x.permute(0, 2, 4, 3, 5, 1)
            x = x.reshape(B, h * w, p1 * p2 * C)
        elif self.pattern == 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)':
            B, H, W, C = x.shape
            h = H // p1
            w = W // p2
            x = x.reshape(B, h, p1, w, p2, C)
            x = x.permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B, h * w, p1 * p2 * C)
        return x


class VisEmbedder(nn.Module):
    """Visual embedder for scalar values (domain-wise parameters)."""
    
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.emb = nn.Linear(in_dim, hidden_size)
        self.mlp = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def forward(self, t):
        t_freq = self.emb(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    """Standard transformer feed-forward network using framework's MLP."""
    
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Could use framework's MLP here, but keeping exact original structure
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer with decoupled AdaLN conditioning.
    
    Each layer has TWO modulation networks for domain-wise and point-wise conditioning.
    """
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        
        self.layers_x = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                # AdaLN modulation for domain-wise (μ) conditioning
                nn.Sequential(nn.SiLU(), nn.Linear(dim // 4, 6 * dim // 4, bias=True)),
                # AdaLN modulation for point-wise (f) conditioning
                nn.Sequential(nn.SiLU(), nn.Linear(dim // 4, 6 * dim * 3 // 4, bias=True)),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))
        
        # Initialize modulation layers to identity
        for i in range(depth):
            nn.init.zeros_(self.layers_x[i][2][1].weight)
            nn.init.zeros_(self.layers_x[i][2][1].bias)
            nn.init.zeros_(self.layers_x[i][3][1].weight)
            nn.init.zeros_(self.layers_x[i][3][1].bias)
    
    def forward(self, x, mu, f):
        """
        Args:
            x: [B, N, dim] - Input tokens
            mu: [B, N, dim//4] - Domain-wise conditioning
            f: [B, N, dim//4] - Point-wise conditioning
        """
        for attn, ff, adaLN_mu, adaLN_f, norm1, norm2 in self.layers_x:
            # Compute AdaLN parameters from both conditions
            shift_msa_mu, scale_msa_mu, gate_msa_mu, shift_mlp_mu, scale_mlp_mu, gate_mlp_mu = \
                adaLN_mu(mu).chunk(6, dim=-1)
            shift_msa_f, scale_msa_f, gate_msa_f, shift_mlp_f, scale_mlp_f, gate_mlp_f = \
                adaLN_f(f).chunk(6, dim=-1)
            
            # Concatenate μ and f conditioning
            shift_msa = torch.cat([shift_msa_mu, shift_msa_f], dim=-1)
            scale_msa = torch.cat([scale_msa_mu, scale_msa_f], dim=-1)
            gate_msa = torch.cat([gate_msa_mu, gate_msa_f], dim=-1)
            shift_mlp = torch.cat([shift_mlp_mu, shift_mlp_f], dim=-1)
            scale_mlp = torch.cat([scale_mlp_mu, scale_mlp_f], dim=-1)
            gate_mlp = torch.cat([gate_mlp_mu, gate_mlp_f], dim=-1)
            
            # Attention block with modulation
            x_attn = attn(modulate(norm1(x), shift_msa, scale_msa))
            x = x + gate_msa * x_attn
            
            # MLP block with modulation
            x_mlp = ff(modulate(norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp * x_mlp
        
        return x


class FinalLayer(nn.Module):
    """Final output layer with AdaLN modulation."""
    
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        # Modulation networks for domain and point conditions
        self.adaLN_modulation_mu = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 2 * hidden_size // 4, bias=True)
        )
        self.adaLN_modulation_f = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 2 * hidden_size * 3 // 4, bias=True)
        )
        
        # Initialize to zero
        nn.init.constant_(self.adaLN_modulation_mu[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_mu[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_f[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_f[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x, mu, f):
        """Forward with AdaLN modulation."""
        shift_mu, scale_mu = self.adaLN_modulation_mu(mu).chunk(2, dim=-1)
        shift_f, scale_f = self.adaLN_modulation_f(f).chunk(2, dim=-1)
        
        shift = torch.cat([shift_mu, shift_f], dim=-1)
        scale = torch.cat([scale_mu, scale_f], dim=-1)
        
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate creating and using the Unisolver equivalent.
    
    This configuration is suitable for heterogeneous Navier-Stokes
    with varying Reynolds numbers (HeterNS dataset).
    """
    print("=" * 60)
    print("Unisolver Example using gnn_pde_v2 Framework")
    print("=" * 60)
    
    # Create model with paper defaults
    model = Unisolver(
        image_size=64,
        patch_size=4,
        dim=256,
        depth=8,
        heads=8,
        mlp_dim=256,
        in_channels=10,
        out_channels=1,
        dim_head=32,
        dropout=0.0,
    )
    
    # Example input
    batch_size = 2
    H, W = 64, 64
    
    # Input field
    x = torch.randn(batch_size, H, W, 10)
    
    # Scalar coefficient (domain-wise, e.g., Reynolds number)
    mu = torch.randn(batch_size)
    
    # Force field (point-wise, e.g., external forces)
    f = torch.randn(batch_size, H, W)
    
    # Forward pass
    output = model(x, mu, f)
    
    print(f"\nModel Configuration:")
    print(f"  Image size: {model.image_size}")
    print(f"  Patch size: {model.patch_size}")
    print(f"  Transformer depth: {len(model.transformer.layers_x)}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input x shape: {x.shape}")
    print(f"  Input mu shape: {mu.shape}")
    print(f"  Input f shape: {f.shape}")
    print(f"  Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model._model_name)
    print("Available models:", BaseModel.list_models())
    print("=" * 60)
    
    return model, x, mu, f, output


if __name__ == "__main__":
    model, x, mu, f, output = example_usage()
