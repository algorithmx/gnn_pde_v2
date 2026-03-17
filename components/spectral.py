"""
Spectral processors for regular grids.

This module contains Fourier Neural Operator (FNO) components that operate
on regular grid data (tensors), NOT graph-structured data.

For graph-based processing, use processors.py (GraphNetProcessor) or
transformer.py (TransformerProcessor).

Classes:
    SpectralConvBase: Abstract base for Fourier-domain mode-multiplication layers
    SpectralConv: Full weight-tensor spectral (Fourier-domain) layer
    SeparableSpectralConv: Factorized spectral layer for memory efficiency
    SpectralBlockBase: Abstract base for FNO-style dual-branch (K + W) blocks
    FNOBlock: Classic FNO block — σ(K(x) + W(x))
    FNOMLPBlock: FNO block with channel MLP — channel_mlp(K(x) + W(x))
    AFNOBlock: Adaptive FNO token mixer with block-diagonal weights
    FNOProcessor: Complete FNO pipeline with lifting/projection

Factory:
    make_spectral_conv: Convenience factory for creating spectral conv layers
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.mlp import MLP


def compl_mul_nd(input: torch.Tensor, weights: torch.Tensor, n_dim: int) -> torch.Tensor:
    """
    Complex multiplication in n-dimensional Fourier space.
    
    Unified implementation that works for any dimension (1D, 2D, 3D, etc.)
    using dynamic einsum string generation.
    
    Args:
        input: [B, C_in, *spatial_dims] - Input in Fourier space
        weights: [C_in, C_out, *spatial_dims] - Complex weights
        n_dim: Number of spatial dimensions
        
    Returns:
        [B, C_out, *spatial_dims] - Output in Fourier space
    """
    # Generate einsum subscripts: b=batch, i=in_channels, o=out_channels
    # spatial dims use letters x, y, z, w (up to 4D)
    spatial_subs = "xyzw"[:n_dim]
    # einsum: b i {spatial}, i o {spatial} -> b o {spatial}
    einsum_str = f"bi{spatial_subs},io{spatial_subs}->bo{spatial_subs}"
    return torch.einsum(einsum_str, input, weights)


def compl_mul1d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 1D Fourier space."""
    # input: [B, C, L], weights: [C, C', L]
    return compl_mul_nd(input, weights, n_dim=1)


def compl_mul2d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 2D Fourier space."""
    # input: [B, C, H, W], weights: [C, C', H, W]
    return compl_mul_nd(input, weights, n_dim=2)


def compl_mul3d(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Complex multiplication in 3D Fourier space."""
    # input: [B, C, D, H, W], weights: [C, C', D, H, W]
    return compl_mul_nd(input, weights, n_dim=3)


def _get_conv_nd(n_dim: int, in_ch: int, out_ch: int, kernel_size: int = 1, padding: int = 0) -> nn.Module:
    """
    Factory function to get the appropriate Conv layer for n-dimensional data.
    
    Args:
        n_dim: Spatial dimension (1, 2, or 3)
        in_ch: Number of input channels
        out_ch: Number of output channels
        kernel_size: Kernel size for the convolution (default: 1 for pointwise)
        padding: Padding for the convolution (default: 0)
        
    Returns:
        nn.Conv1d, nn.Conv2d, or nn.Conv3d instance
        
    Raises:
        ValueError: If n_dim is not 1, 2, or 3
    """
    conv_classes = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if n_dim not in conv_classes:
        raise ValueError(f"n_dim must be 1, 2, or 3, got {n_dim}")
    return conv_classes[n_dim](in_ch, out_ch, kernel_size, padding=padding)


# ---------------------------------------------------------------------------
# SpectralConv Base Class and Implementations
# ---------------------------------------------------------------------------

class SpectralConvBase(nn.Module, ABC):
    """
    Abstract base class for spectral convolution layers.

    Provides the common FFT/IFFT framework; subclasses implement the
    specific spectral multiplication strategy.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep per dimension
    """

    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.n_dim = len(modes)
        self.scale = 1 / (in_channels * out_channels)

        self._init_weights()

    @abstractmethod
    def _init_weights(self) -> None:
        """Initialize spectral weights. Subclasses define weight structure."""
        ...

    @abstractmethod
    def _apply_spectral_conv(self, x_ft: torch.Tensor, out_ft: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution in Fourier space.

        Args:
            x_ft: Input in Fourier space [B, C, *ft_dims]
            out_ft: Pre-allocated output tensor [B, C', *ft_dims]

        Returns:
            Modified out_ft with spectral convolution applied
        """
        ...

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

        # Apply spectral convolution (subclass-specific)
        out_ft = self._apply_spectral_conv(x_ft, out_ft)

        # IFFT
        x = torch.fft.irfftn(out_ft, s=x.shape[2:], dim=list(range(2, 2 + self.n_dim)), norm='ortho')

        return x


class SpectralConv(SpectralConvBase):
    """
    Standard (non-separable) spectral convolution layer.

    Performs convolution in Fourier space with a full weight tensor.
    Uses O(C^2 * prod(modes)) parameters.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep per dimension
    """

    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        super().__init__(in_channels, out_channels, modes)

    def _init_weights(self) -> None:
        """Initialize full spectral weight tensor."""
        weights_shape = [self.in_channels, self.out_channels] + list(self.modes) + [2]
        self.weights = nn.Parameter(self.scale * torch.rand(*weights_shape))

    def _apply_spectral_conv(self, x_ft: torch.Tensor, out_ft: torch.Tensor) -> torch.Tensor:
        """Apply standard (non-separable) spectral convolution."""
        weights_complex = torch.view_as_complex(self.weights)
        
        # Build slice tuple for indexing: [:, :, :modes[0], :modes[1], ...]
        slice_idx = (slice(None), slice(None)) + tuple(slice(0, m) for m in self.modes)
        
        # Apply unified n-dimensional complex multiplication
        out_ft[slice_idx] = compl_mul_nd(x_ft[slice_idx], weights_complex, self.n_dim)

        return out_ft


class SeparableSpectralConv(SpectralConvBase):
    """
    Separable (factorized) spectral convolution layer.

    Applies 1D spectral convolution along each dimension independently,
    then combines the results. This factorization reduces parameters from
    O(C^2 * prod(modes)) to O(n_dim * C^2 * max(modes)).

    This is particularly beneficial for high-dimensional problems (2D, 3D)
    where the full spectral weight tensor becomes prohibitively large.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep per dimension
    """

    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        super().__init__(in_channels, out_channels, modes)

    def _init_weights(self) -> None:
        """Initialize per-dimension weight tensors."""
        self.weights = nn.ParameterList([
            nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes[i], 2))
            for i in range(self.n_dim)
        ])

    def _apply_spectral_conv(self, x_ft: torch.Tensor, out_ft: torch.Tensor) -> torch.Tensor:
        """Apply separable spectral convolution."""
        # For each dimension, apply the corresponding 1D weight
        # We accumulate contributions from each dimension
        for dim_idx, weight in enumerate(self.weights):
            weights_complex = torch.view_as_complex(weight)
            mode = self.modes[dim_idx]
            
            # Build slice for input: slice the current dimension up to mode
            # For dim_idx=0 in 2D: [:, :, :mode, :] -> slice all of dim 0 (H)
            # For dim_idx=1 in 2D: [:, :, :, :mode] -> slice all of dim 1 (W)
            input_slices = [slice(None), slice(None)]  # batch and channels
            for d in range(self.n_dim):
                if d == dim_idx:
                    input_slices.append(slice(0, mode))
                else:
                    input_slices.append(slice(None))
            
            # Build slice for output (same pattern)
            output_slices = tuple(input_slices)
            
            # Extract input slice
            slice_in = x_ft[output_slices]
            
            # Build einsum string for this dimension
            # Pattern: contract along the current dimension only
            # For 1D: "bix,iox->box" (full contraction since only 1 dim)
            # For 2D dim0: "bixy,iox->boxy" (contract x, keep y)
            # For 2D dim1: "bixy,ioy->boxy" (contract y, keep x)
            spatial_letters = "xyzw"[:self.n_dim]
            current_letter = spatial_letters[dim_idx]
            
            # einsum: bi{spatial}, io{current} -> bo{spatial}
            einsum_str = f"bi{spatial_letters},io{current_letter}->bo{spatial_letters}"
            
            # Apply einsum - this contracts along the current dimension
            out_contribution = torch.einsum(einsum_str, slice_in, weights_complex)
            
            # Accumulate into output
            out_ft[output_slices] = out_ft[output_slices] + out_contribution

        return out_ft


def make_spectral_conv(
    in_channels: int,
    out_channels: int,
    modes: List[int],
    separable: bool = False,
) -> SpectralConvBase:
    """
    Factory function for creating spectral convolution layers.

    Use this factory when the spectral conv type is determined at runtime
    (e.g., from configuration). For fixed implementations, prefer direct
    class instantiation for better type clarity.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes to keep per dimension
        separable: If True, use SeparableSpectralConv; otherwise SpectralConv

    Returns:
        SpectralConv or SeparableSpectralConv instance

    Example:
        >>> # Use factory for runtime selection (config-driven)
        >>> conv = make_spectral_conv(64, 64, [16, 16], separable=config.use_separable)
        >>>
        >>> # Use direct class for explicit construction
        >>> conv = SpectralConv(64, 64, [16, 16])
        >>> conv = SeparableSpectralConv(64, 64, [16, 16])  # Memory-efficient
    """
    if separable:
        return SeparableSpectralConv(in_channels, out_channels, modes)
    return SpectralConv(in_channels, out_channels, modes)


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator block.

    Uses block-diagonal weights and soft-thresholding for sparsity.
    Reference: Guibas et al. "Adaptive Fourier Neural Operators" (2021)

    Args:
        hidden_dim: Number of hidden channels (must be divisible by num_blocks)
        num_blocks: Number of blocks for block-diagonal weight matrix
        sparsity_threshold: Threshold for soft-thresholding sparsity (0 to disable)
        n_dim: Spatial dimension (1, 2, or 3)
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


class SpectralBlockBase(nn.Module, ABC):
    """
    Abstract base for FNO-style dual-branch blocks.

    Encapsulates the shared structure common to all FNO block variants
    (Li et al. 2021, Eq. 3)::

        out = K(x) + W(x)         # K = spectral_conv, W = pointwise linear
        out = _post_branch(out)   # subclass-defined post-processing
        if residual: out = x + out

    Subclasses implement ``_post_branch`` to define what happens after the
    two parallel branches are summed.

    Attributes:
        spectral_conv: The K operator — Fourier-domain learned mode
            multiplications (a ``SpectralConvBase`` subclass).
        W: The W operator — a pointwise 1×1 linear map running in parallel
            with ``spectral_conv``.  This is **not** a residual/skip
            connection; it is a learned linear transform.

    Args:
        width: Hidden channel dimension.  Both branches map
            ``width → width``, so input channels must equal ``width``.
        modes: Number of Fourier modes to keep per dimension.
        n_dim: Spatial dimension (1, 2, or 3).
        separable: Use factorized spectral convolution. Reduces parameters from
            O(C² · prod(modes)) to O(n_dim · C² · max(modes)).
        residual: If ``True``, apply an outer residual: ``output = x + block(x)``.
    """

    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int,
        separable: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.width = width
        self.n_dim = n_dim
        self.residual = residual

        # Spectral branch K: FFT → learned mode multiplication → IFFT
        self.spectral_conv: SpectralConvBase = make_spectral_conv(
            width, width, modes, separable=separable
        )
        # W operator: pointwise 1×1 linear map (Li et al. 2021, Eq. 3).
        # NOT a skip/residual connection — this is a learned linear transform
        # that runs in parallel with the spectral branch.
        self.W: nn.Module = _get_conv_nd(n_dim, width, width, kernel_size=1)

    @abstractmethod
    def _post_branch(self, out: torch.Tensor) -> torch.Tensor:
        """Post-process the result of K(x) + W(x).

        Args:
            out: [B, width, *spatial_dims] — the summed output of
                ``spectral_conv(x) + W(x)``.

        Returns:
            Same shape as ``out``.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, width, *spatial_dims]

        Returns:
            [B, width, *spatial_dims]
        """
        out = self.spectral_conv(x) + self.W(x)
        out = self._post_branch(out)
        if self.residual:
            out = x + out
        return out


class FNOBlock(SpectralBlockBase):
    """
    Classic FNO block: activation(spectral_conv(x) + skip(x)).

    Original formulation from Li et al. (2021)::

        output = σ(K(x) + W(x))

    where K is the spectral convolution and W is the pointwise skip.

    Args:
        width: Hidden channel dimension.
        modes: Number of Fourier modes to keep per dimension.
        n_dim: Spatial dimension (1, 2, or 3).
        activation: Pointwise activation applied to the branch sum.
            Supports 'relu', 'gelu', 'silu', 'tanh', 'sigmoid', 'sin', or a
            callable/Module. Defaults to 'gelu'.
        separable: Use factorized spectral convolutions.
        residual: If ``True``, apply an outer skip: ``output = x + block(x)``.
    """

    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int,
        activation: Union[str, nn.Module, Callable, None] = 'gelu',
        separable: bool = False,
        residual: bool = False,
    ):
        super().__init__(width, modes, n_dim, separable=separable, residual=residual)
        self.activation = MLP._make_activation(activation)

    def _post_branch(self, out: torch.Tensor) -> torch.Tensor:
        return self.activation(out)


class FNOMLPBlock(SpectralBlockBase):
    """
    FNO block with a channel MLP replacing the plain pointwise activation.

    A sibling of :class:`FNOBlock` sharing the same :class:`SpectralBlockBase`
    skeleton.  Where ``FNOBlock`` applies a single activation after the branch
    sum, this class applies a two-layer pointwise MLP — the pattern used in the
    neuraloperator library::

        out = channel_mlp(spectral_conv(x) + skip(x))   [+ x if residual]

    The channel MLP uses 1×1 convolutions (spatial dimensions are never mixed),
    acting purely as a per-point nonlinear channel mixer.

    Args:
        width: Hidden channel dimension.
        modes: Number of Fourier modes to keep per dimension.
        n_dim: Spatial dimension (1, 2, or 3).
        channel_mlp_ratio: Hidden-to-width ratio for the channel MLP.
            Hidden size = ``max(1, int(ratio * width))``. Defaults to ``0.5``.
        channel_mlp_dropout: Dropout rate inside the channel MLP.
        activation: Internal activation used by the channel MLP (applied
            between the two pointwise layers). Defaults to 'gelu'.
        separable: Use factorized spectral convolutions.
        residual: If ``True``, apply an outer skip: ``output = x + block(x)``.
    """

    def __init__(
        self,
        width: int,
        modes: List[int],
        n_dim: int,
        channel_mlp_ratio: float = 0.5,
        channel_mlp_dropout: float = 0.0,
        activation: Union[str, nn.Module, Callable, None] = 'gelu',
        separable: bool = False,
        residual: bool = False,
    ):
        super().__init__(width, modes, n_dim, separable=separable, residual=residual)
        mlp_hidden = max(1, int(width * channel_mlp_ratio))
        self.channel_mlp = MLP(
            in_dim=width,
            out_dim=width,
            hidden_dims=[mlp_hidden],
            activation=activation,
            dropout=channel_mlp_dropout,
            norm=None,
            linear_factory=lambda a, b: _get_conv_nd(n_dim, a, b),
            use_layer_norm=False,
        )

    def _post_branch(self, out: torch.Tensor) -> torch.Tensor:
        return self.channel_mlp(out)


class FNOProcessor(nn.Module):
    """
    FNO processor for regular grids.

    Lifts input to hidden space, applies FNO blocks, projects to output.

    NOTE: This processor operates on regular grids (tensors), NOT graphs.
    Use GraphNetProcessor or TransformerProcessor for graph data.

    Input/Output:
        x: [B, in_channels, *spatial_dims] - regular grid data
        returns: [B, out_channels, *spatial_dims]

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden channel dimension
        modes: Number of Fourier modes to keep per dimension
        n_layers: Number of FNO blocks
        n_dim: Spatial dimension (1, 2, or 3)
        use_afno: Use Adaptive FNO blocks with block-diagonal weights
        num_blocks: Number of blocks for AFNO (only used if use_afno=True)
        separable: Use separable (factorized) spectral convolutions. Reduces
            memory from O(C^2 * prod(modes)) to O(n_dim * C^2 * max(modes)).
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
        separable: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_dim = n_dim

        # Lifting layer
        self.lifting = _get_conv_nd(n_dim, in_channels, width, kernel_size=1)

        # FNO blocks — AFNOBlock is instantiated directly to keep it separate
        # from the FNOBlock abstraction (which owns the spectral+skip+activation
        # combinatorics for standard FNO variants).
        if use_afno:
            self.blocks: nn.ModuleList = nn.ModuleList([
                AFNOBlock(width, num_blocks, n_dim=n_dim)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                FNOBlock(
                    width=width,
                    modes=modes,
                    n_dim=n_dim,
                    separable=separable,
                )
                for _ in range(n_layers)
            ])

        # Projection layer
        self.projection = _get_conv_nd(n_dim, width, out_channels, kernel_size=1)

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


__all__ = [
    # Complex multiplication functions
    "compl_mul_nd",
    "compl_mul1d",
    "compl_mul2d",
    "compl_mul3d",
    # Factory functions
    "_get_conv_nd",
    "make_spectral_conv",
    # Spectral convolution classes
    "SpectralConvBase",
    "SpectralConv",
    "SeparableSpectralConv",
    # FNO block base and variants
    "SpectralBlockBase",
    "FNOBlock",
    "FNOMLPBlock",
    "AFNOBlock",
    "FNOProcessor",
]
