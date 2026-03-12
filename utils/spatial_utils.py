"""
Spatial utility functions for grid/point conversions.
"""

from typing import Tuple
import torch
import torch.nn.functional as F
from ..core.graph import GraphsTuple


def grid_to_points(
    grid_values: torch.Tensor,
    grid_extent: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert regular grid values to point cloud representation.
    
    Args:
        grid_values: [B, C, H, W] or [B, C, D, H, W] - Grid values
        grid_extent: (min, max) spatial extent
        
    Returns:
        (positions, values) where positions are [N, n_dim] and values are [N, C]
    """
    B, C = grid_values.shape[0], grid_values.shape[1]
    spatial_dims = grid_values.shape[2:]
    n_dim = len(spatial_dims)
    
    # Create coordinate grid
    coords = []
    for i, size in enumerate(spatial_dims):
        coord = torch.linspace(grid_extent[0], grid_extent[1], size, device=grid_values.device)
        # Expand to full grid shape
        shape = [1] * (n_dim + 1)
        shape[i + 1] = size
        coord = coord.view(*shape)
        # Broadcast
        broadcast_shape = [1] + list(spatial_dims)
        coord = coord.expand(*broadcast_shape)
        coords.append(coord)
    
    # Stack coordinates: [n_dim, *spatial_dims]
    positions_grid = torch.stack(coords, dim=0)
    
    # Flatten: [n_dim, N] -> [N, n_dim]
    N = torch.prod(torch.tensor(spatial_dims)).item()
    positions = positions_grid.reshape(n_dim, -1).T  # [N, n_dim]
    
    # Flatten values: [B, C, *spatial] -> [B, C, N] -> [B*N, C]
    values = grid_values.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]
    values = values.reshape(B * N, C)  # [B*N, C]
    
    # Repeat positions for batch
    positions = positions.repeat(B, 1)  # [B*N, n_dim]
    
    return positions, values


def points_to_grid(
    positions: torch.Tensor,
    values: torch.Tensor,
    grid_shape: Tuple[int, ...],
    grid_extent: Tuple[float, float],
    mode: str = 'bilinear',
) -> torch.Tensor:
    """
    Convert point cloud to regular grid via interpolation.
    
    Args:
        positions: [N, n_dim] - Point positions
        values: [N, C] - Point values
        grid_shape: Target grid shape (H, W) or (D, H, W)
        grid_extent: (min, max) spatial extent
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        [1, C, *grid_shape] - Grid values
    """
    n_dim = len(grid_shape)
    C = values.shape[1]
    
    # Normalize positions to [-1, 1] for grid_sample
    min_val, max_val = grid_extent
    positions_norm = 2 * (positions - min_val) / (max_val - min_val) - 1
    
    if n_dim == 2:
        # Reshape for grid_sample: [1, N, 1, 2]
        grid = positions_norm.view(1, 1, -1, 2)
        
        # Reshape values: [1, C, N, 1]
        values_for_sample = values.T.view(1, C, -1, 1)
        
        # Interpolate to grid
        # First, create a sparse grid representation
        H, W = grid_shape
        grid_dense = torch.zeros(1, C, H, W, device=values.device)
        
        # Simple scatter (can be improved with proper interpolation)
        # Convert normalized positions to grid indices
        indices = ((positions_norm + 1) / 2 * torch.tensor([H - 1, W - 1], device=positions.device)).long()
        indices = indices.clamp(0, torch.tensor([H - 1, W - 1], device=positions.device))
        
        for i in range(len(indices)):
            h, w = indices[i]
            grid_dense[0, :, h, w] = values[i]
        
        return grid_dense
    else:
        raise NotImplementedError(f"points_to_grid not implemented for {n_dim}D")


def normalize_positions(
    positions: torch.Tensor,
    method: str = 'minmax',
) -> Tuple[torch.Tensor, dict]:
    """
    Normalize positions.
    
    Args:
        positions: [N, n_dim] - Positions to normalize
        method: 'minmax' or 'standard'
        
    Returns:
        (normalized_positions, stats) where stats contains normalization parameters
    """
    if method == 'minmax':
        min_val = positions.min(dim=0, keepdim=True)[0]
        max_val = positions.max(dim=0, keepdim=True)[0]
        scale = max_val - min_val
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        normalized = (positions - min_val) / scale
        stats = {'min': min_val, 'max': max_val, 'scale': scale}
    elif method == 'standard':
        mean = positions.mean(dim=0, keepdim=True)
        std = positions.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        normalized = (positions - mean) / std
        stats = {'mean': mean, 'std': std}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def denormalize_positions(
    normalized: torch.Tensor,
    stats: dict,
    method: str = 'minmax',
) -> torch.Tensor:
    """Denormalize positions."""
    if method == 'minmax':
        return normalized * stats['scale'] + stats['min']
    elif method == 'standard':
        return normalized * stats['std'] + stats['mean']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
