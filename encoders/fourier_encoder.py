"""
Fourier feature encoding for high-frequency functions.

Random Fourier Features for coordinate-based networks.
"""

import torch
import torch.nn as nn
import numpy as np
from ..core.graph import GraphsTuple


class FourierFeatureEncoder(nn.Module):
    """
    Random Fourier Feature encoding.
    
    Maps input coordinates to Fourier features using random Gaussian matrix.
    Useful for learning high-frequency functions in low-dimensional spaces.
    
    Reference: Tancik et al. "Fourier Features Let Networks Learn High Frequency 
    Functions in Low Dimensional Domains" (NeurIPS 2020)
    
    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for 2D)
        num_fourier_features: Number of Fourier features (output_dim = 2 * this)
        scale: Standard deviation of Gaussian matrix (higher = higher frequencies)
        learnable: If True, learn the frequency matrix; otherwise fixed
        include_input: If True, concatenate original input with Fourier features
    """
    
    def __init__(
        self,
        input_dim: int,
        num_fourier_features: int = 128,
        scale: float = 1.0,
        learnable: bool = False,
        include_input: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_fourier_features = num_fourier_features
        self.scale = scale
        self.learnable = learnable
        self.include_input = include_input
        
        # Initialize B matrix (input_dim x num_fourier_features)
        B = torch.randn(input_dim, num_fourier_features) * scale
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates to Fourier features.
        
        Args:
            x: [..., input_dim] - Input coordinates
            
        Returns:
            [..., 2*num_fourier_features] or [..., input_dim + 2*num_fourier_features]
            - Sine and cosine features (and optionally original input)
        """
        # Project to frequency space: [..., num_fourier_features]
        phase = 2 * np.pi * (x @ self.B)
        
        # Apply sin and cos
        features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        
        if self.include_input:
            features = torch.cat([x, features], dim=-1)
        
        return features
    
    def encode_graph(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Encode node positions to Fourier features.
        
        Uses graph.positions if available, otherwise graph.nodes (assumed to be coords).
        """
        if graph.positions is not None:
            features = self.forward(graph.positions)
            # Concatenate with existing node features if present
            if graph.nodes is not None:
                nodes = torch.cat([graph.nodes, features], dim=-1)
            else:
                nodes = features
        elif graph.nodes is not None:
            # Assume nodes are coordinates
            nodes = self.forward(graph.nodes)
        else:
            raise ValueError("Graph must have nodes or positions for Fourier encoding")
        
        return graph.replace(nodes=nodes)
    
    def get_output_dim(self) -> int:
        """Get output dimension."""
        out = 2 * self.num_fourier_features
        if self.include_input:
            out += self.input_dim
        return out
