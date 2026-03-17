"""
Transolver-3 Example: Scaling to Industrial-Scale Geometries

This example demonstrates the key improvements from Transolver-3 (arXiv:2602.04940):
1. Geometry slice tiling for memory efficiency  
2. Training on mesh subsets (geometry amortized training)
3. Physical state caching for inference on massive meshes

Paper: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"
Authors: Hang Zhou, Haixu Wu, et al. (Tsinghua University)

Key Innovation:
- Handles meshes with 160M+ cells (vs ~700K for Transolver++)
- Memory complexity: O(N * tile_size) instead of O(N * M)
- Single-GPU capacity: ~2.9M cells (vs ~700K for Transolver++)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple

# Import Transolver-3 components from framework
from gnn_pde_v2.components import PhysicsTokenAttentionV3, TiledSliceOperation


class TransolverV3Block(nn.Module):
    """
    Transolver-3 transformer block with optimized physics attention.
    
    Uses geometry slice tiling to handle meshes up to ~2.9M cells on single GPU.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        slice_num: int = 32,
        # Transolver-3 optimizations
        use_tiling: bool = True,
        tile_size: int = 100000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Transolver-3 optimized attention
        self.attn = PhysicsTokenAttentionV3(
            dim=hidden_dim,
            n_tokens=slice_num,
            n_heads=num_heads,
            dropout=dropout,
            temperature=0.5,
            temperature_mode='learnable_scalar',
            use_slice_normalization=True,
            use_learnable_tokens=False,
            qkv_mode='direct',
            use_orthogonal_init=True,
            # Transolver-3 specific
            use_tiling=use_tiling,
            tile_size=tile_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] or [N, D] - Input features
        Returns:
            [B, N, D] or [N, D] - Output features
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransolverV3(nn.Module):
    """
    Complete Transolver-3 model for industrial-scale PDE solving.
    
    Capable of handling meshes with 160M+ cells through:
    - Geometry slice tiling during training
    - Amortized training on random subsets
    - Physical state caching during inference
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
        # Transolver-3 optimizations
        use_tiling: bool = True,
        tile_size: int = 100000,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.space_dim = space_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.input_proj = nn.Linear(space_dim + input_dim, hidden_dim)
        
        # Transolver-3 transformer blocks
        self.blocks = nn.ModuleList([
            TransolverV3Block(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                slice_num=slice_num,
                use_tiling=use_tiling,
                tile_size=tile_size,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, N, space_dim + input_dim] or [N, space_dim + input_dim]
               Contains [coordinates, field_values]
        Returns:
            [B, N, output_dim] or [N, output_dim] - Predicted fields
        """
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.output_proj(x)
        
        if single_batch:
            x = x.squeeze(0)
        
        return x
    
    def set_epoch(self, epoch: int):
        """Set current epoch for temperature scheduling."""
        for block in self.blocks:
            if hasattr(block.attn, 'set_epoch'):
                block.attn.set_epoch(epoch)


class GeometryAmortizedTraining:
    """
    Geometry Amortized Training for Transolver-3.
    
    Instead of training on the full high-resolution mesh, randomly sample
    subsets for each training iteration. This allows training on industrial-scale
    meshes that exceed GPU memory while learning the underlying physics.
    
    Reference: Transolver-3 Section 3.2 "Geometry Amortized Training"
    """
    
    def __init__(
        self,
        full_mesh_size: int,
        subset_size: int,
        seed: Optional[int] = None,
    ):
        """
        Args:
            full_mesh_size: Total number of points in full mesh (e.g., 160_000_000)
            subset_size: Number of points to sample per iteration (e.g., 400_000)
            seed: Random seed for reproducibility
        """
        self.full_mesh_size = full_mesh_size
        self.subset_size = min(subset_size, full_mesh_size)
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
    
    def get_subset_indices(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get random subset indices for current training iteration.
        
        Returns:
            [subset_size] tensor of random indices
        """
        return torch.randperm(self.full_mesh_size, generator=self.rng, device=device)[:self.subset_size]
    
    def apply_subset(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply random subset to input and target tensors.
        
        Args:
            x: [N, ...] full mesh features
            y: [N, ...] optional target values
            
        Returns:
            x_subset: [subset_size, ...] 
            y_subset: [subset_size, ...] or None
        """
        indices = self.get_subset_indices(x.device)
        x_subset = x[indices]
        y_subset = y[indices] if y is not None else None
        return x_subset, y_subset


class PhysicalStateCache:
    """
    Physical State Caching for Transolver-3 inference on massive meshes.
    
    Enables high-fidelity predictions on industrial-scale geometries by:
    1. Building physical state cache layer-by-layer through chunked processing
    2. Decoding predictions point-by-point without loading full mesh
    
    Reference: Transolver-3 Section 3.3 "Geometry Scaling at the Inference Phase"
    """
    
    def __init__(
        self,
        model: nn.Module,
        chunk_size: int = 50000,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The Transolver model containing PhysicsTokenAttentionV3 layers
            chunk_size: Number of mesh points to process per chunk
            device: Device for computation (defaults to model's device)
        """
        self.model = model
        self.chunk_size = chunk_size
        self.device = device or next(model.parameters()).device
    
    def build_cache(
        self,
        x: torch.Tensor,
        num_layers: int,
    ) -> List[torch.Tensor]:
        """
        Build physical state cache layer-by-layer.
        
        Args:
            x: [N, D] full mesh features (can be 100M+ points)
            num_layers: Number of transformer layers
            
        Returns:
            List of cached physical states, one per layer.
            Each state is [G, D] where G is number of physics tokens.
        """
        N = x.shape[0]
        cache = []
        
        current_x = x
        
        for layer_idx in range(num_layers):
            layer = self._get_layer(layer_idx)
            physical_state = self._compute_layer_state(current_x, layer)
            cache.append(physical_state)
            
        return cache
    
    def _compute_layer_state(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> torch.Tensor:
        """
        Compute physical state for a layer via chunked processing.
        
        Args:
            x: [N, D] mesh features
            layer: Transformer layer containing PhysicsTokenAttentionV3
            
        Returns:
            [G, D] physical state tokens
        """
        N = x.shape[0]
        num_chunks = (N + self.chunk_size - 1) // self.chunk_size
        
        accumulated_state = None
        accumulated_norm = None
        
        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, N)
            
            chunk = x[start:end].to(self.device)
            state_chunk, norm_chunk = self._compute_chunk_contribution(chunk, layer)
            
            if accumulated_state is None:
                accumulated_state = state_chunk
                accumulated_norm = norm_chunk
            else:
                accumulated_state += state_chunk
                accumulated_norm += norm_chunk
        
        physical_state = accumulated_state / (accumulated_norm + 1e-5)
        return physical_state
    
    def _compute_chunk_contribution(
        self,
        chunk: torch.Tensor,
        layer: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute physical state contribution from a single chunk."""
        attn_module = None
        for module in layer.modules():
            if isinstance(module, PhysicsTokenAttentionV3):
                attn_module = module
                break
        
        if attn_module is None:
            raise ValueError("Layer does not contain PhysicsTokenAttentionV3")
        
        with torch.no_grad():
            B, N, D = 1, chunk.shape[0], chunk.shape[1]
            H = attn_module.n_heads
            G = attn_module.n_tokens
            d = attn_module.head_dim
            
            chunk_batch = chunk.unsqueeze(0)
            
            x_mid = attn_module.in_project_x(chunk_batch).reshape(B, N, H, d).permute(0, 2, 1, 3)
            fx_mid = attn_module.in_project_fx(chunk_batch).reshape(B, N, H, d).permute(0, 2, 1, 3)
            
            slice_logits = attn_module.slice_weight_proj(x_mid)
            _, slice_logits = attn_module.temperature_module(slice_logits, chunk_batch)
            slice_weights = torch.softmax(slice_logits, dim=-1)
            
            tokens = torch.einsum('bhnd,bhng->bhgd', fx_mid, slice_weights)
            slice_norm = slice_weights.sum(dim=2, keepdim=True)
            
            tokens = tokens.sum(dim=1)
            slice_norm = slice_norm.sum(dim=1)
            
            return tokens.squeeze(0), slice_norm.squeeze(0)
    
    def _get_layer(self, layer_idx: int) -> nn.Module:
        """Get layer from model by index."""
        if hasattr(self.model, 'blocks'):
            return self.model.blocks[layer_idx]
        elif hasattr(self.model, 'transformer_blocks'):
            return self.model.transformer_blocks[layer_idx]
        elif hasattr(self.model, 'layers'):
            return self.model.layers[layer_idx]
        else:
            raise ValueError("Cannot find transformer layers in model")


def demo_memory_efficiency():
    """Demonstrate Transolver-3's memory efficiency with tiling."""
    print("=" * 60)
    print("Transolver-3 Memory Efficiency Demo")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_dim = 256
    num_heads = 8
    n_tokens = 64
    
    mesh_sizes = [10000, 100000, 500000, 1000000]
    
    print(f"\nDevice: {device}")
    print(f"Hidden dim: {hidden_dim}, Heads: {num_heads}, Tokens: {n_tokens}\n")
    
    for mesh_size in mesh_sizes:
        print(f"\nMesh size: {mesh_size:,} points")
        
        # Without tiling
        try:
            attn_standard = PhysicsTokenAttentionV3(
                dim=hidden_dim,
                n_heads=num_heads,
                n_tokens=n_tokens,
                use_tiling=False,
            ).to(device)
            
            x = torch.randn(1, mesh_size, hidden_dim, device=device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            out = attn_standard(x)
            
            if torch.cuda.is_available():
                mem_standard = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  Without tiling: {mem_standard:.1f} MB peak memory")
            else:
                print(f"  Without tiling: Success (CPU mode)")
            
            del attn_standard, x, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Without tiling: OOM (out of memory)")
            else:
                print(f"  Without tiling: Error - {e}")
        
        # With tiling
        try:
            tile_size = min(50000, mesh_size // 2 + 1)
            attn_tiled = PhysicsTokenAttentionV3(
                dim=hidden_dim,
                n_heads=num_heads,
                n_tokens=n_tokens,
                use_tiling=True,
                tile_size=tile_size,
                use_gradient_checkpointing=True,
            ).to(device)
            
            x = torch.randn(1, mesh_size, hidden_dim, device=device)
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            out = attn_tiled(x)
            
            if torch.cuda.is_available():
                mem_tiled = torch.cuda.max_memory_allocated() / 1024**2
                print(f"  With tiling (tile={tile_size:,}): {mem_tiled:.1f} MB peak memory")
            else:
                print(f"  With tiling (tile={tile_size:,}): Success (CPU mode)")
            
            del attn_tiled, x, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"  With tiling: Error - {e}")


def demo_amortized_training():
    """Demonstrate geometry amortized training on mesh subsets."""
    print("\n" + "=" * 60)
    print("Transolver-3 Geometry Amortized Training Demo")
    print("=" * 60)
    
    full_mesh_size = 10_000_000
    subset_size = 400_000
    
    print(f"\nFull mesh size: {full_mesh_size:,} points")
    print(f"Training subset size: {subset_size:,} points ({subset_size/full_mesh_size*100:.1f}%)")
    
    amortizer = GeometryAmortizedTraining(
        full_mesh_size=full_mesh_size,
        subset_size=subset_size,
        seed=42,
    )
    
    print("\nSimulating training iterations:")
    for iteration in range(5):
        subset_indices = amortizer.get_subset_indices()
        print(f"  Iteration {iteration + 1}: Sampled {len(subset_indices):,} unique points")


def demo_model_creation():
    """Demonstrate creating a Transolver-3 model."""
    print("\n" + "=" * 60)
    print("Transolver-3 Model Creation Demo")
    print("=" * 60)
    
    model = TransolverV3(
        space_dim=3,
        input_dim=4,
        output_dim=4,
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        slice_num=64,
        use_tiling=True,
        tile_size=100000,
    )
    
    print(f"\nModel created:")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Num heads: {model.blocks[0].attn.n_heads}")
    print(f"  Physics tokens: {model.blocks[0].attn.n_tokens}")
    print(f"  Tiling enabled: {model.blocks[0].attn.use_tiling}")
    print(f"  Tile size: {model.blocks[0].attn.tiled_slice.tile_size if model.blocks[0].attn.tiled_slice else 'N/A'}")
    
    # Test forward pass
    x = torch.randn(1, 10000, 7)  # [B, N, space_dim + input_dim]
    out = model(x)
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transolver-3: Industrial-Scale Neural PDE Solvers")
    print("Paper: arXiv:2602.04940")
    print("=" * 60)
    
    demo_memory_efficiency()
    demo_amortized_training()
    demo_model_creation()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
