"""
Radial Basis Function (RBF) encoders for distance features.

These encoders transform scalar distances into high-dimensional feature
representations using Gaussian basis functions. Widely used in:
- Graph neural networks for molecular/crystal systems
- Wind farm flow modeling (Wind-Farm-GNO)
- Neural network potentials for materials

Reference:
---------
Behler, J., & Parrinello, M. (2007). Generalized Neural-Network Representation
of High-Dimensional Potential-Energy Surfaces. Physical Review Letters, 98(14), 146401.
https://doi.org/10.1103/PhysRevLett.98.146401
"""

import torch
import torch.nn as nn
import math


class LearnableRBFEncoder(nn.Module):
    """
    Learnable Radial Basis Function encoder with cosine cutoff.
    
    Encodes scalar distances into high-dimensional features using Gaussian
    basis functions with learnable centers (mu) and widths (beta). Includes
    a smooth cosine cutoff function that truncates the encoding at d_max.
    
    The encoding for a distance d is:
        RBF_i(d) = cutoff(d) * exp(-beta_i * (d - mu_i)^2)
    
    where cutoff(d) is the cosine cutoff:
        cutoff(d) = 0.5 * (cos(pi * d / d_max) + 1)  for d < d_max
                    0                                   otherwise
    
    Design Decisions:
    -----------------
    1. Centers (mu) initialized uniformly between d_min and d_max
    2. Widths (beta) initialized based on kernel spacing for good coverage
    3. Both parameters are learnable by default (can be frozen via learnable=False)
    4. Cutoff applied after RBF computation for smooth spatial truncation
    
    Args:
        num_kernels: Number of RBF basis functions (output dimension)
        d_min: Minimum distance value for kernel centers
        d_max: Maximum distance value (also cutoff radius)
        learnable: Whether centers (mu) and widths (beta) are trainable
        
    Example:
        >>> encoder = LearnableRBFEncoder(num_kernels=20, d_min=0.0, d_max=5.0)
        >>> distances = torch.tensor([0.5, 1.0, 2.0, 3.0])
        >>> rbf_features = encoder(distances)  # [4, 20]
    """
    
    def __init__(
        self,
        num_kernels: int = 20,
        d_min: float = -1.0,
        d_max: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.d_min = d_min
        self.d_max = d_max
        
        # Initialize RBF centers uniformly between d_min and d_max
        mu = torch.linspace(d_min, d_max, num_kernels)
        
        # Initialize beta (width) based on kernel spacing
        # Heuristic: beta ~ num_kernels / (d_max - d_min) for good coverage
        if d_max != d_min:
            beta = torch.ones(num_kernels) * (num_kernels / (d_max - d_min))
        else:
            beta = torch.ones(num_kernels)
        
        if learnable:
            self.mu = nn.Parameter(mu)
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('beta', beta)
    
    def cosine_cutoff(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Smooth cosine cutoff function.
        
        Returns a smooth cutoff that goes to zero at d_max:
            cutoff(d) = 0.5 * (cos(pi * d / d_max) + 1)  for d < d_max
                        0                                   otherwise
        
        Args:
            distances: [...] scalar distances
            
        Returns:
            [...] cutoff values in [0, 1]
        """
        # Compute cosine cutoff
        cutoff = 0.5 * (torch.cos(math.pi * distances / self.d_max) + 1.0)
        # Zero out values beyond d_max
        cutoff = torch.where(distances < self.d_max, cutoff, 0.0)
        return cutoff
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Encode distances to RBF features.
        
        Args:
            distances: [...] scalar distances. Can be any shape.
            
        Returns:
            [..., num_kernels] RBF-encoded features
        """
        # Expand distances for broadcasting: [..., 1]
        d = distances.unsqueeze(-1)
        
        # Compute RBF: exp(-beta * (d - mu)^2)
        # Shape: [..., num_kernels]
        rbf = torch.exp(-self.beta * (d - self.mu) ** 2)
        
        # Apply cutoff (shape [...] broadcasts to [..., 1])
        cutoff = self.cosine_cutoff(distances)
        
        return cutoff.unsqueeze(-1) * rbf
    
    def extra_repr(self) -> str:
        return (
            f"num_kernels={self.num_kernels}, "
            f"d_min={self.d_min}, d_max={self.d_max}, "
            f"learnable={self.mu.requires_grad}"
        )


class GaussianRBFEncoder(nn.Module):
    """
    Fixed Gaussian RBF encoder without learnable parameters.
    
    Similar to LearnableRBFEncoder but with fixed centers and widths.
    Useful when you want deterministic, non-trainable distance encoding.
    
    Args:
        num_kernels: Number of RBF basis functions
        d_min: Minimum distance for kernel centers
        d_max: Maximum distance for kernel centers
        gamma: Width parameter for Gaussian (smaller = wider)
    """
    
    def __init__(
        self,
        num_kernels: int = 20,
        d_min: float = 0.0,
        d_max: float = 5.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.d_min = d_min
        self.d_max = d_max
        self.gamma = gamma
        
        # Fixed centers
        centers = torch.linspace(d_min, d_max, num_kernels)
        self.register_buffer('centers', centers)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Encode distances to RBF features.
        
        Args:
            distances: [...] scalar distances
            
        Returns:
            [..., num_kernels] RBF-encoded features
        """
        d = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


__all__ = [
    "LearnableRBFEncoder",
    "GaussianRBFEncoder",
]
