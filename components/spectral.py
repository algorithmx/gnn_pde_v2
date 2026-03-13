"""
Spectral processors for regular grids.

This module contains Fourier Neural Operator (FNO) components that operate
on regular grid data (tensors), NOT graph-structured data.

For graph-based processing, use processors.py (GraphNetProcessor) or
transformer.py (TransformerProcessor).

Classes:
    SpectralConv: Spectral convolution layer
    FNOBlock: Single FNO block (spectral conv + MLP)
    AFNOBlock: Adaptive FNO block with soft thresholding
    FNOProcessor: Complete FNO pipeline with lifting/projection
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compl_mul1d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 1D Fourier space."""
    # input: [B, C, L], weights: [C, C', L]
    return torch.einsum("bcl,ccl->bcl", input, weights)


def compl_mul2d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 2D Fourier space."""
    # input: [B, C, H, W], weights: [C, C', H, W]
    return torch.einsum("bchw,cchw->bchw", input, weights)


def compl_mul3d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 3D Fourier space."""
    # input: [B, C, D, H, W], weights: [C, C', D, H, W]
    return torch.einsum("bcdhw,ccdhw->bcdhw", input, weights)


class SpectralConv(nn.Module):
    """
    Spectral convolution layer.

    Performs convolution in Fourier space with learnable complex weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        separable: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dim = len(modes)
        self.separable = separable

        # Scale for weight initialization
        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        if separable:
            # Separable weights: one per dimension
            self.weights = nn.ParameterList([
                nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes[i], 2))
                for i in range(self.n_dim)
            ])
        else:
            # Full weights: all dimensions together
            weights_shape = [in_channels, out_channels] + list(modes) + [2]
            self.weights = nn.Parameter(self.scale * torch.rand(*weights_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, *spatial_dims] - Input on regular grid

        Returns:
            [B, C', *spatial_dims] - Output
        """
        batch_size = x.shape[0]

        # FFT
        x_ft = torch.fft.rfftn(x, dim=list(range(2, 2 + self.n_dim)), norm='ortho')

        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            *x_ft.shape[2:],
            dtype=torch.cfloat,
            device=x.device,
        )

        # Multiply relevant Fourier modes
        if self.n_dim == 1:
            out_ft[:, :, :self.modes[0]] = compl_mul1d(
                x_ft[:, :, :self.modes[0]],
                torch.view_as_complex(self.weights)
            )
        elif self.n_dim == 2:
            out_ft[:, :, :self.modes[0], :self.modes[1]] = compl_mul2d(
                x_ft[:, :, :self.modes[0], :self.modes[1]],
                torch.view_as_complex(self.weights)
            )
        elif self.n_dim == 3:
            out_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]] = compl_mul3d(
                x_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]],
                torch.view_as_complex(self.weights)
            )

        # IFFT
        x = torch.fft.irfftn(out_ft, s=x.shape[2:], dim=list(range(2, 2 + self.n_dim)), norm='ortho')

        return x


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator block.

    Uses block-diagonal weights and soft-thresholding for sparsity.
    Reference: Guibas et al. "Adaptive Fourier Neural Operators" (2021)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        n_dim: int = 2,
    ):
        super().__init__()

        assert hidden_dim % num_blocks == 0
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.block_size = hidden_dim // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.n_dim = n_dim

        # Block-diagonal weights
        self.scale = 0.02
        if n_dim == 1:
            self.weights = nn.Parameter(
                self.scale * torch.randn(num_blocks, self.block_size, self.block_size, 2)
            )
        else:
            # For 2D/3D, use separable block-diagonal
            self.weights_real = nn.Parameter(
                self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
            )
            self.weights_imag = nn.Parameter(
                self.scale * torch.randn(num_blocks, self.block_size, self.block_size)
            )

        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, *spatial_dims]
        """
        B, C = x.shape[0], x.shape[1]
        spatial_dims = x.shape[2:]

        # FFT
        x_ft = torch.fft.rfftn(x, dim=list(range(2, 2 + self.n_dim)), norm='ortho')

        # Reshape to blocks: [B, num_blocks, block_size, *spatial_ft]
        x_ft_blocks = x_ft.reshape(B, self.num_blocks, self.block_size, *x_ft.shape[2:])

        # Apply block-diagonal weights
        if self.n_dim == 1:
            weights_complex = torch.view_as_complex(self.weights)
            out_ft_blocks = torch.einsum('bnc...,ncl->bnl...', x_ft_blocks, weights_complex)
        else:
            # Real and imaginary parts
            x_ft_real = x_ft_blocks.real
            x_ft_imag = x_ft_blocks.imag

            out_real = torch.einsum('bnc...,ncl->bnl...', x_ft_real, self.weights_real) - \
                       torch.einsum('bnc...,ncl->bnl...', x_ft_imag, self.weights_imag)
            out_imag = torch.einsum('bnc...,ncl->bnl...', x_ft_real, self.weights_imag) + \
                       torch.einsum('bnc...,ncl->bnl...', x_ft_imag, self.weights_real)

            out_ft_blocks = torch.complex(out_real, out_imag)

        # Soft thresholding for sparsity
        if self.sparsity_threshold > 0:
            out_ft_blocks = torch.complex(
                F.softshrink(out_ft_blocks.real, self.sparsity_threshold),
                F.softshrink(out_ft_blocks.imag, self.sparsity_threshold)
            )

        # Reshape back
        out_ft = out_ft_blocks.reshape(B, C, *x_ft.shape[2:])

        # IFFT
        x = torch.fft.irfftn(out_ft, s=spatial_dims, dim=list(range(2, 2 + self.n_dim)), norm='ortho')

        return x + self.bias.view(1, -1, *([1] * self.n_dim))


class FNOBlock(nn.Module):
    """
    FNO block: Spectral convolution + pointwise MLP.
    """

    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int,
        activation: str = 'gelu',
        use_afno: bool = False,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.width = width
        self.n_dim = n_dim

        # Spectral convolution
        if use_afno:
            self.spectral_conv = AFNOBlock(width, num_blocks, n_dim=n_dim)
        else:
            self.spectral_conv = SpectralConv(width, width, modes, separable=False)

        # Pointwise MLP
        self.mlp = nn.Conv1d(width, width, 1) if n_dim == 1 else \
                   nn.Conv2d(width, width, 1) if n_dim == 2 else \
                   nn.Conv3d(width, width, 1)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, *spatial_dims]
        """
        # Spectral branch
        x1 = self.spectral_conv(x)

        # Pointwise branch
        x2 = self.mlp(x)

        return self.activation(x1 + x2)


class FNOProcessor(nn.Module):
    """
    FNO processor for regular grids.

    Lifts input to hidden space, applies FNO blocks, projects to output.

    NOTE: This processor operates on regular grids (tensors), NOT graphs.
    Use GraphNetProcessor or TransformerProcessor for graph data.

    Input/Output:
        x: [B, in_channels, *spatial_dims] - regular grid data
        returns: [B, out_channels, *spatial_dims]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes: List[int] = [16, 16],
        n_layers: int = 4,
        n_dim: int = 2,
        use_afno: bool = False,
        num_blocks: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_dim = n_dim

        # Lifting layer
        if n_dim == 1:
            self.lifting = nn.Conv1d(in_channels, width, 1)
        elif n_dim == 2:
            self.lifting = nn.Conv2d(in_channels, width, 1)
        elif n_dim == 3:
            self.lifting = nn.Conv3d(in_channels, width, 1)
        else:
            raise ValueError(f"n_dim must be 1, 2, or 3, got {n_dim}")

        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock(
                width=width,
                modes=modes,
                n_dim=n_dim,
                use_afno=use_afno,
                num_blocks=num_blocks,
            )
            for _ in range(n_layers)
        ])

        # Projection layer
        if n_dim == 1:
            self.projection = nn.Conv1d(width, out_channels, 1)
        elif n_dim == 2:
            self.projection = nn.Conv2d(width, out_channels, 1)
        elif n_dim == 3:
            self.projection = nn.Conv3d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, *spatial_dims]

        Returns:
            [B, out_channels, *spatial_dims]
        """
        # Lift
        x = self.lifting(x)

        # Process
        for block in self.blocks:
            x = block(x)

        # Project
        x = self.projection(x)

        return x
