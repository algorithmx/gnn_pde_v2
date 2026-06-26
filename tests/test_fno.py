"""
Tests for FNO (Fourier Neural Operator) components.

Covers: SpectralConvBase, SpectralConv, SeparableSpectralConv,
        SpectralBlockBase, FNOBlock, FNOMLPBlock, AFNOBlock, FNOProcessor,
        and the model classes (FNO, TFNO, AFNO).
"""

import pytest
import torch
import torch.nn as nn

from gnn_pde_v2.components.spectral import (
    compl_mul_nd,
    compl_mul1d,
    compl_mul2d,
    compl_mul3d,
    _get_conv_nd,
    SpectralConv,
    SeparableSpectralConv,
    SpectralConvBase,
    FNOBlock,
    FNOMLPBlock,
    AFNOBlock,
    FNOProcessor,
    make_spectral_conv,
    SpectralBlockBase,
)
from gnn_pde_v2.models.fno_model import FNO, TFNO, AFNO


class TestComplMulND:
    """Test unified complex multiplication function."""

    def test_1d(self):
        """Test 1D complex multiplication."""
        B, C_in, C_out, L = 2, 4, 6, 16
        x = torch.randn(B, C_in, L, dtype=torch.cfloat)
        w = torch.randn(C_in, C_out, L, dtype=torch.cfloat)
        
        out = compl_mul_nd(x, w, n_dim=1)
        
        assert out.shape == (B, C_out, L)
        # Compare with legacy function
        out_legacy = compl_mul1d(x, w)
        assert torch.allclose(out, out_legacy)

    def test_2d(self):
        """Test 2D complex multiplication."""
        B, C_in, C_out, H, W = 2, 4, 6, 16, 16
        x = torch.randn(B, C_in, H, W, dtype=torch.cfloat)
        w = torch.randn(C_in, C_out, H, W, dtype=torch.cfloat)
        
        out = compl_mul_nd(x, w, n_dim=2)
        
        assert out.shape == (B, C_out, H, W)
        out_legacy = compl_mul2d(x, w)
        assert torch.allclose(out, out_legacy)

    def test_3d(self):
        """Test 3D complex multiplication."""
        B, C_in, C_out, D, H, W = 2, 4, 6, 8, 8, 8
        x = torch.randn(B, C_in, D, H, W, dtype=torch.cfloat)
        w = torch.randn(C_in, C_out, D, H, W, dtype=torch.cfloat)
        
        out = compl_mul_nd(x, w, n_dim=3)
        
        assert out.shape == (B, C_out, D, H, W)
        out_legacy = compl_mul3d(x, w)
        assert torch.allclose(out, out_legacy)


class TestGetConvND:
    """Test Conv factory function."""

    def test_1d(self):
        """Test 1D convolution factory."""
        conv = _get_conv_nd(1, 4, 8, kernel_size=1)
        assert isinstance(conv, nn.Conv1d)
        assert conv.in_channels == 4
        assert conv.out_channels == 8
        assert conv.kernel_size == (1,)

    def test_2d(self):
        """Test 2D convolution factory."""
        conv = _get_conv_nd(2, 4, 8, kernel_size=1)
        assert isinstance(conv, nn.Conv2d)
        assert conv.in_channels == 4
        assert conv.out_channels == 8
        assert conv.kernel_size == (1, 1)

    def test_3d(self):
        """Test 3D convolution factory."""
        conv = _get_conv_nd(3, 4, 8, kernel_size=1)
        assert isinstance(conv, nn.Conv3d)
        assert conv.in_channels == 4
        assert conv.out_channels == 8
        assert conv.kernel_size == (1, 1, 1)

    def test_invalid_n_dim(self):
        """Test error handling for invalid n_dim."""
        with pytest.raises(ValueError, match="n_dim must be 1, 2, or 3"):
            _get_conv_nd(4, 4, 8)


class TestSpectralConv:
    """Test standard spectral convolution."""

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_forward(self, n_dim, device):
        """Test forward pass for different dimensions."""
        modes = [8] * n_dim
        spatial = [32] * n_dim
        
        conv = SpectralConv(4, 6, modes).to(device)
        x = torch.randn(2, 4, *spatial, device=device)
        
        out = conv(x)
        
        assert out.shape == (2, 6, *spatial)

    def test_gradient_flow(self, device):
        """Test gradients flow through the layer."""
        conv = SpectralConv(4, 6, [8, 8]).to(device)
        x = torch.randn(2, 4, 32, 32, device=device, requires_grad=True)
        
        out = conv(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert conv.weights.grad is not None

    def test_weight_shape(self):
        """Test weight tensor shape."""
        conv = SpectralConv(4, 6, [8, 10])
        # Shape: [in_ch, out_ch, mode0, mode1, 2] for real/imag
        assert conv.weights.shape == (4, 6, 8, 10, 2)


class TestSeparableSpectralConv:
    """Test separable spectral convolution."""

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_forward(self, n_dim, device):
        """Test forward pass for different dimensions."""
        modes = [8] * n_dim
        spatial = [32] * n_dim
        
        conv = SeparableSpectralConv(4, 6, modes).to(device)
        x = torch.randn(2, 4, *spatial, device=device)
        
        out = conv(x)
        
        assert out.shape == (2, 6, *spatial)

    def test_weight_structure(self):
        """Test per-dimension weight structure."""
        conv = SeparableSpectralConv(4, 6, [8, 10, 12])
        
        # Should have 3 separate weight tensors
        assert len(conv.weights) == 3
        
        # Each weight: [in_ch, out_ch, mode, 2] for real/imag
        assert conv.weights[0].shape == (4, 6, 8, 2)
        assert conv.weights[1].shape == (4, 6, 10, 2)
        assert conv.weights[2].shape == (4, 6, 12, 2)

    def test_vs_standard_conv(self, device):
        """Test that separable conv produces different but valid output."""
        modes = [8, 8]
        x = torch.randn(2, 4, 32, 32, device=device)
        
        standard = SpectralConv(4, 6, modes).to(device)
        separable = SeparableSpectralConv(4, 6, modes).to(device)
        
        out_std = standard(x)
        out_sep = separable(x)
        
        # Both should have same shape
        assert out_std.shape == out_sep.shape
        # But different values (different parameterizations)
        assert not torch.allclose(out_std, out_sep)


class TestMakeSpectralConv:
    """Test spectral conv factory."""

    def test_standard(self):
        """Test factory creates standard conv."""
        conv = make_spectral_conv(4, 6, [8, 8], separable=False)
        assert isinstance(conv, SpectralConv)
        assert not isinstance(conv, SeparableSpectralConv)

    def test_separable(self):
        """Test factory creates separable conv."""
        conv = make_spectral_conv(4, 6, [8, 8], separable=True)
        assert isinstance(conv, SeparableSpectralConv)


class TestSpectralBlockBase:
    """
    Tests for the SpectralBlockBase abstract class.

    Verifies the design contract: both branches (spectral_conv = K,
    W = pointwise linear) are always present, the outer residual is
    optional and controlled by ``residual``, and W is a *learned
    transform* rather than an identity / skip connection.
    """

    def test_is_abstract(self):
        """SpectralBlockBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SpectralBlockBase(width=32, modes=[8, 8], n_dim=2)  # type: ignore[abstract]

    def test_fno_block_is_subclass(self):
        """FNOBlock is a concrete SpectralBlockBase subclass."""
        assert issubclass(FNOBlock, SpectralBlockBase)

    def test_fno_mlp_block_is_subclass(self):
        """FNOMLPBlock is a concrete SpectralBlockBase subclass."""
        assert issubclass(FNOMLPBlock, SpectralBlockBase)

    def test_siblings_share_base_but_differ(self, device):
        """FNOBlock and FNOMLPBlock share SpectralBlockBase but are distinct."""
        fno = FNOBlock(width=32, modes=[8, 8], n_dim=2).to(device)
        fno_mlp = FNOMLPBlock(width=32, modes=[8, 8], n_dim=2).to(device)

        assert isinstance(fno, SpectralBlockBase)
        assert isinstance(fno_mlp, SpectralBlockBase)
        assert type(fno) is not type(fno_mlp)

        # Structurally distinct post-branch components
        assert hasattr(fno, 'activation') and not hasattr(fno, 'channel_mlp')
        assert hasattr(fno_mlp, 'channel_mlp') and not hasattr(fno_mlp, 'activation')

    def test_W_is_learned_transform_not_identity(self, device):
        """
        W is a learned 1x1 linear operator, not a skip/identity.
        Its output must differ from its input (with non-trivial weights).
        """
        block = FNOBlock(width=32, modes=[8, 8], n_dim=2).to(device)
        # W is a Conv2d — has explicit weight and bias parameters
        assert isinstance(block.W, nn.Conv2d)
        assert block.W.kernel_size == (1, 1)
        assert block.W.in_channels == block.W.out_channels == 32
        # Output should differ from input (W transforms channels)
        x = torch.randn(1, 32, 8, 8, device=device)
        out_W = block.W(x)
        assert not torch.allclose(x, out_W)

    def test_W_attribute_not_named_skip(self):
        """W must be named 'W', not 'skip' or 'mlp' — avoid misnomer."""
        block = FNOBlock(width=32, modes=[8, 8], n_dim=2)
        assert hasattr(block, 'W')
        assert not hasattr(block, 'skip')
        assert not hasattr(block, 'mlp')

    def test_residual_false_by_default(self, device):
        """By default, no outer residual is applied."""
        block = FNOBlock(width=32, modes=[8, 8], n_dim=2).to(device)
        assert block.residual is False

        # With zero-weight W and zero spectral_conv, output should be
        # activation(0) not x + activation(0) — i.e. no x addback.
        nn.init.zeros_(block.W.weight)
        nn.init.zeros_(block.W.bias)
        for p in block.spectral_conv.parameters():
            nn.init.zeros_(p)

        x = torch.ones(1, 32, 8, 8, device=device)
        out = block(x)
        # activation(0) != x; residual was not applied
        assert not torch.allclose(out, x)

    def test_residual_true_adds_input(self, device):
        """With residual=True, output = x + post_branch(K(x) + W(x))."""
        block = FNOBlock(width=32, modes=[8, 8], n_dim=2, residual=True).to(device)
        assert block.residual is True

        # Zero out all learnable weights → K(x)=0, W(x)=0 → post_branch(0)
        nn.init.zeros_(block.W.weight)
        nn.init.zeros_(block.W.bias)
        for p in block.spectral_conv.parameters():
            nn.init.zeros_(p)

        x = torch.ones(1, 32, 8, 8, device=device)
        out = block(x)
        # With gelu(0)=0, residual means out = x + 0 = x
        assert torch.allclose(out, x, atol=1e-6)


class TestFNOBlock:
    """Test FNO block."""

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_forward(self, n_dim, device):
        """Test forward pass for different dimensions."""
        modes = [8] * n_dim
        spatial = [32] * n_dim
        
        block = FNOBlock(
            width=64,
            modes=modes,
            n_dim=n_dim,
        ).to(device)
        
        x = torch.randn(2, 64, *spatial, device=device)
        out = block(x)
        
        assert out.shape == x.shape

    @pytest.mark.parametrize("activation", ['relu', 'gelu', 'silu', 'tanh'])
    def test_activations(self, activation, device):
        """Test different activation functions."""
        block = FNOBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
            activation=activation,
        ).to(device)
        
        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)
        
        assert out.shape == x.shape

    def test_separable(self, device):
        """Test FNO block with separable convolutions."""
        block = FNOBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
            separable=True,
        ).to(device)
        
        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)
        
        assert out.shape == x.shape
        assert isinstance(block.spectral_conv, SeparableSpectralConv)

    def test_residual(self, device):
        """Test FNO block with outer residual connection."""
        block = FNOBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
            residual=True,
        ).to(device)

        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)

        assert out.shape == x.shape
        assert block.residual is True

    def test_is_spectral_block_base(self, device):
        """Test that FNOBlock is a SpectralBlockBase subclass."""
        block = FNOBlock(width=32, modes=[8, 8], n_dim=2)
        assert isinstance(block, SpectralBlockBase)
        # W is the pointwise linear operator (NOT a skip/residual connection)
        assert hasattr(block, 'W')
        assert hasattr(block, 'spectral_conv')
        assert hasattr(block, 'activation')
        assert not hasattr(block, 'channel_mlp')


class TestFNOMLPBlock:
    """Test FNOMLPBlock (channel-MLP sibling of FNOBlock)."""

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_forward(self, n_dim, device):
        """Test forward pass for different dimensions."""
        modes = [8] * n_dim
        spatial = [32] * n_dim

        block = FNOMLPBlock(
            width=64,
            modes=modes,
            n_dim=n_dim,
        ).to(device)

        x = torch.randn(2, 64, *spatial, device=device)
        out = block(x)

        assert out.shape == x.shape

    def test_is_spectral_block_base(self, device):
        """Test that FNOMLPBlock is a SpectralBlockBase subclass."""
        block = FNOMLPBlock(width=32, modes=[8, 8], n_dim=2)
        assert isinstance(block, SpectralBlockBase)
        assert hasattr(block, 'W')
        assert hasattr(block, 'spectral_conv')
        assert hasattr(block, 'channel_mlp')
        assert not hasattr(block, 'activation')

    def test_channel_mlp_ratio(self, device):
        """Test that channel_mlp_ratio controls MLP hidden size."""
        block = FNOMLPBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
            channel_mlp_ratio=0.5,
        ).to(device)

        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)

        assert out.shape == x.shape

    def test_residual_with_channel_mlp(self, device):
        """Test neuraloperator-style block: residual + channel MLP."""
        block = FNOMLPBlock(
            width=64,
            modes=[8, 8],
            n_dim=2,
            channel_mlp_ratio=0.5,
            channel_mlp_dropout=0.1,
            residual=True,
        ).to(device)

        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)

        assert out.shape == x.shape
        assert block.residual is True

    def test_gradient_flow(self, device):
        """Test gradients flow through K, W, and channel_mlp."""
        block = FNOMLPBlock(width=32, modes=[8, 8], n_dim=2).to(device)
        x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)

        out = block(x)
        out.sum().backward()

        assert x.grad is not None
        assert block.W.weight.grad is not None
        assert all(p.grad is not None for p in block.channel_mlp.parameters())

    @pytest.mark.parametrize("activation", ['relu', 'gelu', 'silu'])
    def test_activations(self, activation, device):
        """Test internal MLP activation options."""
        block = FNOMLPBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
            activation=activation,
        ).to(device)

        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)

        assert out.shape == x.shape


class TestAFNOBlock:
    """Test Adaptive FNO block."""

    def test_forward_2d(self, device):
        """Test 2D forward pass."""
        block = AFNOBlock(
            hidden_dim=64,
            num_blocks=8,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        
        assert out.shape == x.shape

    def test_forward_1d(self, device):
        """Test 1D forward pass."""
        block = AFNOBlock(
            hidden_dim=64,
            num_blocks=8,
            n_dim=1,
        ).to(device)
        
        x = torch.randn(2, 64, 32, device=device)
        out = block(x)
        
        assert out.shape == x.shape

    def test_num_blocks_assertion(self):
        """Test assertion for hidden_dim % num_blocks == 0."""
        with pytest.raises(AssertionError):
            AFNOBlock(hidden_dim=63, num_blocks=8)  # 63 % 8 != 0

    def test_sparsity_threshold(self, device):
        """Test soft thresholding effect."""
        block = AFNOBlock(
            hidden_dim=64,
            num_blocks=8,
            n_dim=2,
            sparsity_threshold=0.1,
        ).to(device)
        
        x = torch.randn(2, 64, 16, 16, device=device)
        out = block(x)
        
        assert out.shape == x.shape


class TestFNOProcessor:
    """Test FNO processor."""

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_forward(self, n_dim, device):
        """Test forward pass for different dimensions."""
        modes = [8] * n_dim
        spatial = [32] * n_dim
        
        processor = FNOProcessor(
            in_channels=3,
            out_channels=1,
            width=64,
            modes=modes,
            n_layers=2,
            n_dim=n_dim,
        ).to(device)
        
        x = torch.randn(2, 3, *spatial, device=device)
        out = processor(x)
        
        assert out.shape == (2, 1, *spatial)

    def test_multiple_layers(self, device):
        """Test with multiple layers."""
        processor = FNOProcessor(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[8, 8],
            n_layers=4,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, 1, 16, 16, device=device)
        out = processor(x)
        
        assert out.shape == (2, 1, 16, 16)
        assert len(processor.blocks) == 4

    def test_separable(self, device):
        """Test with separable convolutions."""
        processor = FNOProcessor(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[8, 8],
            n_layers=2,
            n_dim=2,
            separable=True,
        ).to(device)
        
        x = torch.randn(2, 1, 16, 16, device=device)
        out = processor(x)
        
        assert out.shape == (2, 1, 16, 16)

    def test_afno(self, device):
        """Test with AFNO blocks."""
        processor = FNOProcessor(
            in_channels=1,
            out_channels=1,
            width=64,
            modes=[8, 8],
            n_layers=2,
            n_dim=2,
            use_afno=True,
            num_blocks=8,
        ).to(device)
        
        x = torch.randn(2, 1, 16, 16, device=device)
        out = processor(x)
        
        assert out.shape == (2, 1, 16, 16)


class TestFNOModel:
    """Test FNO model class."""

    def test_forward_2d(self, device):
        """Test 2D FNO model."""
        model = FNO(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[8, 8],
            n_layers=2,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, 1, 32, 32, device=device)
        out = model(x)
        
        assert out.shape == (2, 1, 32, 32)

    def test_forward_1d(self, device):
        """Test 1D FNO model."""
        model = FNO(
            in_channels=2,
            out_channels=1,
            width=32,
            modes=[16],
            n_layers=2,
            n_dim=1,
        ).to(device)
        
        x = torch.randn(2, 2, 64, device=device)
        out = model(x)
        
        assert out.shape == (2, 1, 64)

    def test_model_registry(self):
        """Test model is registered."""
        from gnn_pde_v2.core.registry import MODEL_REGISTRY
        
        assert 'fno' in MODEL_REGISTRY
        # Check aliases
        assert 'fourier_no' in MODEL_REGISTRY
        assert 'fno2d' in MODEL_REGISTRY

    def test_is_base_model(self):
        """FNO is registered via AutoRegisterModel, so it must be a BaseModel.

        This locks in the discriminator fix documented in
        ``docs/investigation-report-model-base-class-and-registration.md``
        §3.3: FNO used to inherit from plain nn.Module and failed the
        ``isinstance(BaseModel)`` check; it now inherits from
        ``AutoRegisterModel`` and must pass it.
        """
        from gnn_pde_v2.core import BaseModel

        assert isinstance(FNO(in_channels=1, out_channels=1, width=8), BaseModel)


class TestTFNOModel:
    """Test TFNO (Tensorized FNO) model class."""

    def test_forward(self, device):
        """Test TFNO uses separable convolutions."""
        model = TFNO(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[8, 8],
            n_layers=2,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, 1, 16, 16, device=device)
        out = model(x)
        
        assert out.shape == (2, 1, 16, 16)
        
        # Verify separable convolutions are used
        processor = model.fno
        for block in processor.blocks:
            assert isinstance(block.spectral_conv, SeparableSpectralConv)

    def test_model_registry(self):
        """Test model is registered."""
        from gnn_pde_v2.core.registry import MODEL_REGISTRY
        
        assert 'tfno' in MODEL_REGISTRY
        assert 'tensorized_fno' in MODEL_REGISTRY

    def test_is_base_model(self):
        """TFNO is registered via AutoRegisterModel, so it must be a BaseModel."""
        from gnn_pde_v2.core import BaseModel

        assert isinstance(TFNO(in_channels=1, out_channels=1, width=8), BaseModel)


class TestAFNOModel:
    """Test AFNO (Adaptive FNO) model class."""

    def test_forward(self, device):
        """Test AFNO uses adaptive blocks."""
        model = AFNO(
            in_channels=1,
            out_channels=1,
            width=64,
            modes=[8, 8],
            n_layers=2,
            n_dim=2,
            num_blocks=8,
        ).to(device)
        
        x = torch.randn(2, 1, 16, 16, device=device)
        out = model(x)
        
        assert out.shape == (2, 1, 16, 16)

        # FNOProcessor instantiates AFNOBlock directly in processor.blocks
        # (not wrapped inside FNOBlock) — verify this design contract.
        processor = model.fno
        for block in processor.blocks:
            assert isinstance(block, AFNOBlock)
            assert not isinstance(block, SpectralBlockBase)

    def test_model_registry(self):
        """Test model is registered (previously missing for AFNO)."""
        from gnn_pde_v2.core.registry import MODEL_REGISTRY

        assert 'afno' in MODEL_REGISTRY
        assert 'adaptive_fno' in MODEL_REGISTRY

    def test_is_base_model(self):
        """AFNO is registered via AutoRegisterModel, so it must be a BaseModel."""
        from gnn_pde_v2.core import BaseModel

        assert isinstance(AFNO(in_channels=1, out_channels=1, width=64, num_blocks=8), BaseModel)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        """Test on CUDA if available."""
        conv = SpectralConv(4, 6, [8, 8]).cuda()
        x = torch.randn(2, 4, 16, 16, device='cuda')
        out = conv(x)
        assert out.device.type == 'cuda'


class TestEdgeCases:
    """Test edge cases."""

    def test_different_modes_per_dim(self, device):
        """Test with different number of modes per dimension."""
        conv = SpectralConv(4, 6, [8, 16]).to(device)
        x = torch.randn(2, 4, 32, 64, device=device)
        out = conv(x)
        
        assert out.shape == (2, 6, 32, 64)
        # Weight shape should reflect different modes
        assert conv.weights.shape == (4, 6, 8, 16, 2)

    def test_single_batch(self, device):
        """Test with batch size 1."""
        conv = SpectralConv(4, 6, [8, 8]).to(device)
        x = torch.randn(1, 4, 16, 16, device=device)
        out = conv(x)
        
        assert out.shape == (1, 6, 16, 16)

    def test_large_modes(self, device):
        """Test with modes close to spatial size."""
        conv = SpectralConv(4, 6, [15, 15]).to(device)
        x = torch.randn(2, 4, 32, 32, device=device)
        out = conv(x)
        
        assert out.shape == (2, 6, 32, 32)

    def test_fno_block_output_differs_from_input(self, device):
        """FNOBlock output must differ from input — it is not an identity."""
        block = FNOBlock(
            width=32,
            modes=[8, 8],
            n_dim=2,
        ).to(device)

        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)

        # Output should not be the same tensor as input
        assert out is not x
        assert out.shape == x.shape
        # With random weights, output should genuinely differ
        assert not torch.allclose(out, x)


class TestRegistryWriteApiRemoved:
    """AutoRegisterModel is the unique registration method.

    The decorator / imperative registration API has been removed from
    ``MODEL_REGISTRY``: the only way to register a model is to subclass
    ``AutoRegisterModel``. These assertions guard against re-introducing a
    competing registration surface.
    """

    def test_model_registry_has_no_register_decorator(self):
        from gnn_pde_v2.core.registry import MODEL_REGISTRY

        assert not hasattr(MODEL_REGISTRY, 'register')

    def test_model_registry_has_no_add_method(self):
        from gnn_pde_v2.core.registry import MODEL_REGISTRY

        assert not hasattr(MODEL_REGISTRY, 'add')

    def test_all_fno_family_models_are_base_model(self):
        """FNO/TFNO/AFNO all register via AutoRegisterModel and thus are BaseModel."""
        from gnn_pde_v2.core import BaseModel

        assert isinstance(FNO(in_channels=1, out_channels=1, width=8), BaseModel)
        assert isinstance(TFNO(in_channels=1, out_channels=1, width=8), BaseModel)
        assert isinstance(AFNO(in_channels=1, out_channels=1, width=64, num_blocks=8), BaseModel)
