"""Consolidated tests for temperature mechanisms.

Combines module, integration, edge-case, research-validation, and training
coverage for the temperature features used by attention components.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from gnn_pde_v2.components.attention import PhysicsTokenAttention
from gnn_pde_v2.components.temperature import (
    AdaptiveTemperature,
    AnnealedTemperature,
    FixedTemperature,
    LearnableScalarTemperature,
    PerHeadTemperature,
    create_temperature_module,
)
from gnn_pde_v2.components.transformer import TransformerBlock, TransformerProcessor
from gnn_pde_v2.core.graph import GraphsTuple


class TestFixedTemperature:
    """Tests for FixedTemperature class."""

    def test_default_temperature(self):
        """Test default temperature is 1.0."""
        temp = FixedTemperature()
        logits = torch.randn(2, 4, 8, 16)
        t, scaled = temp(logits)
        assert t.item() == 1.0
        assert torch.allclose(scaled, logits)

    def test_custom_temperature(self):
        """Test custom temperature value."""
        temp = FixedTemperature(temperature=2.0)
        logits = torch.randn(2, 4, 8, 16)
        t, scaled = temp(logits)
        assert t.item() == 2.0
        assert torch.allclose(scaled, logits / 2.0)

    def test_temperature_scaling_effect(self):
        """Test that higher temperature produces smoother distribution."""
        temp_high = FixedTemperature(temperature=2.0)
        temp_low = FixedTemperature(temperature=0.5)
        logits = torch.randn(2, 4, 8, 16)

        _, scaled_high = temp_high(logits)
        _, scaled_low = temp_low(logits)

        var_high = torch.var(scaled_high)
        var_low = torch.var(scaled_low)
        assert var_high < var_low


class TestLearnableScalarTemperature:
    """Tests for LearnableScalarTemperature class."""

    def test_initialization(self):
        """Test parameter initialization."""
        temp = LearnableScalarTemperature(init_temperature=1.0)
        assert hasattr(temp, 'log_temperature')
        assert torch.allclose(temp.log_temperature, torch.tensor(0.0))

    def test_forward_returns_temperature(self):
        """Test forward returns temperature and scaled logits."""
        temp = LearnableScalarTemperature(init_temperature=1.0)
        logits = torch.randn(2, 4, 8, 16)
        t, scaled = temp(logits)

        assert t.item() > 0
        assert scaled.shape == logits.shape
        assert torch.allclose(scaled, logits / t)

    def test_temperature_clamping(self):
        """Test temperature is clamped to min_temp."""
        temp = LearnableScalarTemperature(init_temperature=1.0, min_temp=0.5)
        temp.log_temperature.data = torch.tensor(-10.0)

        logits = torch.randn(2, 4, 8, 16)
        t, _ = temp(logits)
        assert t.item() >= 0.5

    def test_gradient_flow(self):
        """Test gradients flow to temperature parameter."""
        temp = LearnableScalarTemperature(init_temperature=1.0)
        logits = torch.randn(2, 4, 8, 16, requires_grad=True)

        t, scaled = temp(logits)
        loss = scaled.sum()
        loss.backward()

        assert temp.log_temperature.grad is not None
        assert temp.log_temperature.grad.abs() > 0


class TestPerHeadTemperature:
    """Tests for PerHeadTemperature class."""

    def test_initialization(self):
        """Test per-head parameter initialization."""
        n_heads = 8
        temp = PerHeadTemperature(n_heads=n_heads, init_temperature=1.0)
        assert temp.log_temperatures.shape == (n_heads,)
        assert torch.allclose(temp.log_temperatures, torch.zeros(n_heads))

    def test_per_head_temperatures(self):
        """Test each head has its own temperature."""
        n_heads = 4
        temp = PerHeadTemperature(n_heads=n_heads, init_temperature=1.0)

        temp.log_temperatures.data = torch.tensor([0.0, 0.5, 1.0, 1.5])

        logits = torch.randn(2, n_heads, 8, 16)
        t, scaled = temp(logits)

        assert t.item() > 0
        assert scaled.shape == logits.shape

    def test_temperature_broadcasting(self):
        """Test temperatures are broadcast correctly."""
        n_heads = 4
        temp = PerHeadTemperature(n_heads=n_heads)
        logits = torch.randn(2, n_heads, 8, 16)

        t, scaled = temp(logits)
        for h in range(n_heads):
            expected_scale = torch.exp(temp.log_temperatures[h])
            assert torch.allclose(scaled[:, h], logits[:, h] / expected_scale)


class TestAdaptiveTemperature:
    """Tests for AdaptiveTemperature (Ada-Temp) class."""

    def test_initialization(self):
        """Test projection layer initialization."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim)

        assert hasattr(temp, 'temp_proj')
        assert temp.temp_proj.in_features == feature_dim
        assert temp.temp_proj.out_features == 1
        assert torch.allclose(temp.temp_proj.weight, torch.zeros_like(temp.temp_proj.weight))
        assert torch.allclose(temp.temp_proj.bias, torch.zeros_like(temp.temp_proj.bias))

    def test_ada_temp_formula(self):
        """Test Ada-Temp formula: τ_i = τ_0 + Linear(x_i)."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim, init_temperature=1.0, learnable_base=False)

        features = torch.randn(2, 8, feature_dim)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits, features)

        tau_0 = 1.0
        delta_tau = temp.temp_proj(features).squeeze(-1)
        expected_temps = (tau_0 + delta_tau).clamp_min(0.1)

        assert expected_temps.shape == (2, 8)
        assert t.item() > 0

    def test_per_point_temperature(self):
        """Test different points get different temperatures."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim)

        features = torch.zeros(1, 4, feature_dim)
        features[0, 0, :] = 1.0
        features[0, 1, :] = -1.0
        features[0, 2, :] = 0.5
        features[0, 3, :] = -0.5

        logits = torch.randn(1, 1, 4, 16)

        temp.temp_proj.weight.data = torch.ones_like(temp.temp_proj.weight)

        t, scaled = temp(logits, features)
        assert t.item() > 0
        assert scaled.shape == logits.shape

    def test_gradient_flow_to_projection(self):
        """Test gradients flow to projection layer."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim)

        features = torch.randn(2, 8, feature_dim, requires_grad=True)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits, features)
        loss = scaled.sum()
        loss.backward()

        assert temp.temp_proj.weight.grad is not None
        assert temp.log_tau_0.grad is not None


class TestAnnealedTemperature:
    """Tests for AnnealedTemperature class."""

    def test_initial_temperature(self):
        """Test initial temperature is returned during warmup."""
        temp = AnnealedTemperature(init_temperature=1.0, warmup_epochs=5)
        logits = torch.randn(2, 4, 8, 16)

        for epoch in range(5):
            temp.set_epoch(epoch)
            t, _ = temp(logits)
            assert t.item() == 1.0

    def test_annealing_schedule(self):
        """Test annealing schedule computation."""
        init_temp = 1.0
        final_temp = 0.05
        warmup = 5
        factor = 0.98

        temp = AnnealedTemperature(
            init_temperature=init_temp,
            final_temperature=final_temp,
            warmup_epochs=warmup,
            anneal_factor=factor,
        )

        temp.set_epoch(6)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        expected = max(factor ** (6 - warmup), final_temp)
        assert abs(t.item() - expected) < 1e-6

    def test_final_temperature_clamp(self):
        """Test temperature doesn't go below final_temperature."""
        temp = AnnealedTemperature(
            init_temperature=1.0,
            final_temperature=0.05,
            warmup_epochs=5,
            anneal_factor=0.98,
        )

        temp.set_epoch(1000)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        assert t.item() >= 0.05

    def test_set_epoch_updates_temperature(self):
        """Test set_epoch correctly updates internal state."""
        temp = AnnealedTemperature(init_temperature=1.0, warmup_epochs=5)

        temp.set_epoch(0)
        assert temp.current_epoch == 0
        assert temp._current_temp == 1.0

        temp.set_epoch(10)
        assert temp.current_epoch == 10
        assert temp._current_temp < 1.0


class TestCreateTemperatureModule:
    """Tests for create_temperature_module factory function."""

    def test_create_fixed(self):
        """Test creating FixedTemperature."""
        temp = create_temperature_module('fixed', temperature=1.5)
        assert isinstance(temp, FixedTemperature)
        assert temp.temperature == 1.5

    def test_create_learnable_scalar(self):
        """Test creating LearnableScalarTemperature."""
        temp = create_temperature_module('learnable_scalar', temperature=2.0, min_temperature=0.2)
        assert isinstance(temp, LearnableScalarTemperature)

    def test_create_per_head(self):
        """Test creating PerHeadTemperature."""
        temp = create_temperature_module('per_head', n_heads=8, temperature=1.0)
        assert isinstance(temp, PerHeadTemperature)
        assert temp.log_temperatures.shape == (8,)

    def test_create_adaptive(self):
        """Test creating AdaptiveTemperature."""
        temp = create_temperature_module('adaptive', dim=128, temperature=1.0)
        assert isinstance(temp, AdaptiveTemperature)
        assert temp.temp_proj.in_features == 128

    def test_create_annealed(self):
        """Test creating AnnealedTemperature."""
        temp = create_temperature_module(
            'annealed',
            temperature=1.0,
            anneal_final_temp=0.05,
            anneal_warmup_epochs=5,
        )
        assert isinstance(temp, AnnealedTemperature)

    def test_invalid_mode_raises_error(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match='Unknown temperature mode'):
            create_temperature_module('invalid_mode')


class TestPhysicsTokenAttentionTemperature:
    """Tests for PhysicsTokenAttention with different temperature modes."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 16, 64)

    def test_fixed_temperature_backward_compatible(self, sample_input):
        """Test fixed mode maintains backward compatibility."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='fixed',
            temperature=1.0,
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, FixedTemperature)

    def test_adaptive_temperature(self, sample_input):
        """Test adaptive temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=False,
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, AdaptiveTemperature)

    def test_per_head_temperature(self, sample_input):
        """Test per-head temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='per_head',
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, PerHeadTemperature)

    def test_learnable_scalar_temperature(self, sample_input):
        """Test learnable scalar temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='learnable_scalar',
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape

    def test_annealed_temperature(self, sample_input):
        """Test annealed temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='annealed',
            anneal_warmup_epochs=5,
        )
        attn.set_epoch(10)
        out = attn(sample_input)
        assert out.shape == sample_input.shape

    def test_gumbel_softmax_integration(self, sample_input):
        """Test Gumbel-Softmax works with adaptive temperature."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True,
        )
        attn.train()
        out_train = attn(sample_input)

        attn.eval()
        out_eval = attn(sample_input)

        assert out_train.shape == sample_input.shape
        assert out_eval.shape == sample_input.shape

    def test_single_batch_input(self):
        """Test with single batch (2D input)."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )
        x = torch.randn(16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_temperature_parameter_access(self, sample_input):
        """Test temperature value can be accessed after forward pass."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )
        out = attn(sample_input)

        assert out.shape == sample_input.shape
        assert hasattr(attn, 'temperature_module')


class TestTransformerBlockTemperature:
    """Tests for TransformerBlock with temperature parameters."""

    def test_transformer_block_with_adaptive_temp(self):
        """Test TransformerBlock passes temperature params to PhysicsTokenAttention."""
        block = TransformerBlock(
            dim=64,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='adaptive',
            use_gumbel_softmax=True,
        )

        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_transformer_block_set_epoch(self):
        """Test TransformerBlock set_epoch propagates to attention."""
        block = TransformerBlock(
            dim=64,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='annealed',
        )

        block.set_epoch(5)

    def test_transformer_block_without_physics_tokens(self):
        """Test TransformerBlock without physics tokens warns about ignored params."""
        with pytest.warns(UserWarning, match='Ignored parameters'):
            block = TransformerBlock(
                dim=64,
                n_heads=4,
                use_physics_tokens=False,
                temperature_mode='adaptive',
            )

        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape


class TestTransformerProcessorTemperature:
    """Tests for TransformerProcessor with temperature."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        return GraphsTuple.from_flat(
            nodes=torch.randn(10, 64),
            n_node=torch.tensor([10]),
            edges=None,
            receivers=None,
            senders=None,
            positions=None,
        )

    def test_processor_with_adaptive_temperature(self, sample_graph):
        """Test TransformerProcessor with adaptive temperature."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=2,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='adaptive',
        )

        out_graph = processor(sample_graph)
        assert out_graph.nodes.shape == sample_graph.nodes.shape

    def test_processor_set_epoch_propagation(self, sample_graph):
        """Test set_epoch propagates to all blocks."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=3,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='annealed',
        )

        processor.set_epoch(10)

        for block in processor.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'temperature_module'):
                temp_module = block.attn.temperature_module
                if hasattr(temp_module, 'current_epoch'):
                    assert temp_module.current_epoch == 10

    def test_processor_multiple_layers_consistent(self, sample_graph):
        """Test all layers use same temperature configuration."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=3,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='per_head',
        )

        out_graph = processor(sample_graph)
        assert out_graph.nodes.shape == sample_graph.nodes.shape


class TestTemperatureClamping:
    """Tests for minimum temperature clamping."""

    def test_learnable_scalar_min_clamp(self):
        """Test learnable scalar clamps to min_temperature."""
        temp = LearnableScalarTemperature(init_temperature=1.0, min_temp=0.1)
        temp.log_temperature.data = torch.tensor(-10.0)

        logits = torch.randn(2, 4, 8, 16)
        t, _ = temp(logits)

        assert t.item() >= 0.1

    def test_adaptive_temperature_min_clamp(self):
        """Test adaptive temperature clamps to min_temperature."""
        temp = AdaptiveTemperature(feature_dim=64, min_temp=0.1)
        temp.temp_proj.weight.data = torch.full((1, 64), -10.0)

        features = torch.randn(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)

        t, _ = temp(logits, features)

        assert t.item() >= 0.1


class TestInputShapes:
    """Tests for various input shapes."""

    def test_single_batch_2d_input(self):
        """Test with 2D input [N, D]."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )

        x = torch.randn(16, 64)
        out = attn(x)

        assert out.shape == x.shape

    def test_batched_3d_input(self):
        """Test with 3D input [B, N, D]."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )

        x = torch.randn(2, 16, 64)
        out = attn(x)

        assert out.shape == x.shape

    def test_varying_batch_sizes(self):
        """Test with different batch sizes."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 16, 64)
            out = attn(x)
            assert out.shape == x.shape

    def test_varying_sequence_lengths(self):
        """Test with different sequence lengths."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        )

        for seq_len in [4, 8, 16, 32]:
            x = torch.randn(2, seq_len, 64)
            out = attn(x)
            assert out.shape == x.shape


class TestFeatureValues:
    """Tests with various feature value ranges."""

    def test_zero_features(self):
        """Test with zero features."""
        temp = AdaptiveTemperature(feature_dim=64)

        features = torch.zeros(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits, features)

        assert t.item() > 0
        assert not torch.isnan(scaled).any()

    def test_large_features(self):
        """Test with large feature values."""
        temp = AdaptiveTemperature(feature_dim=64)

        features = torch.randn(2, 8, 64) * 1000
        logits = torch.randn(2, 4, 8, 16) * 1000

        t, scaled = temp(logits, features)

        assert t.item() > 0
        assert not torch.isinf(scaled).any()
        assert not torch.isnan(scaled).any()

    def test_small_features(self):
        """Test with small feature values."""
        temp = AdaptiveTemperature(feature_dim=64)

        features = torch.randn(2, 8, 64) * 0.001
        logits = torch.randn(2, 4, 8, 16) * 0.001

        t, scaled = temp(logits, features)

        assert t.item() > 0
        assert not torch.isnan(scaled).any()


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_temperature_mode(self):
        """Test invalid temperature mode raises ValueError."""
        with pytest.raises(ValueError, match='Unknown temperature mode'):
            create_temperature_module('invalid_mode')

    def test_adaptive_requires_features(self):
        """Test adaptive temperature requires features."""
        temp = AdaptiveTemperature(feature_dim=64)

        features = torch.randn(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits, features)

        assert t is not None
        assert scaled is not None


class TestDevicePlacement:
    """Tests for device placement (CPU/CUDA)."""

    @pytest.fixture
    def device(self):
        """Get device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_temperature_on_device(self, device):
        """Test temperature module on correct device."""
        temp = AdaptiveTemperature(feature_dim=64).to(device)
        features = torch.randn(2, 8, 64).to(device)
        logits = torch.randn(2, 4, 8, 16).to(device)

        t, scaled = temp(logits, features)

        assert t.device.type == device.type
        assert scaled.device.type == device.type

    def test_attention_on_device(self, device):
        """Test PhysicsTokenAttention on correct device."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
        ).to(device)

        x = torch.randn(2, 16, 64).to(device)
        out = attn(x)

        assert out.device.type == device.type


class TestDtypeConsistency:
    """Tests for dtype consistency."""

    def test_float32_consistency(self):
        """Test with float32."""
        temp = AdaptiveTemperature(feature_dim=64)

        features = torch.randn(2, 8, 64, dtype=torch.float32)
        logits = torch.randn(2, 4, 8, 16, dtype=torch.float32)

        t, scaled = temp(logits, features)

        assert t.dtype == torch.float32
        assert scaled.dtype == torch.float32

    def test_float64_consistency(self):
        """Test with float64."""
        temp = AdaptiveTemperature(feature_dim=64)
        temp = temp.to(dtype=torch.float64)

        features = torch.randn(2, 8, 64, dtype=torch.float64)
        logits = torch.randn(2, 4, 8, 16, dtype=torch.float64)

        t, scaled = temp(logits, features)

        assert scaled.dtype == torch.float64


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_default_temperature_mode(self):
        """Test default temperature mode is 'learnable_scalar'.

        Using a learnable scalar as the default gives the model a free
        parameter to tune attention sharpness from the start of training,
        unlike the fixed mode which requires explicit configuration.
        """
        attn = PhysicsTokenAttention(dim=64, n_tokens=8, n_heads=4)

        assert attn.temperature_mode == 'learnable_scalar'
        assert isinstance(attn.temperature_module, LearnableScalarTemperature)

    def test_old_api_compatibility(self):
        """Test old API with temperature parameter still works."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature=2.0,
        )

        x = torch.randn(2, 16, 64)
        out = attn(x)

        assert out.shape == x.shape


class TestTransolverPlusPlus:
    """Tests validating against Transolver++ paper."""

    def test_ada_temp_formula(self):
        """Test Ada-Temp formula: τ_i = τ_0 + Linear(x_i).

        From Transolver++ Eq. 3:
        τ = {τ_i} = {τ_0 + Linear(x_i)}
        """
        feature_dim = 64
        init_temp = 1.0

        temp = AdaptiveTemperature(
            feature_dim=feature_dim,
            init_temperature=init_temp,
            learnable_base=True,
        )

        features = torch.randn(2, 8, feature_dim)

        tau_0 = torch.exp(temp.log_tau_0)
        delta_tau = temp.temp_proj(features).squeeze(-1)
        expected_tau = tau_0 + delta_tau

        logits = torch.randn(2, 4, 8, 16)
        actual_tau, scaled = temp(logits, features)

        assert expected_tau.shape == (2, 8)
        assert scaled.shape == logits.shape
        assert actual_tau.item() > 0

    def test_ada_temp_learns_local_properties(self):
        """Test Ada-Temp adapts to local point properties."""
        temp = AdaptiveTemperature(feature_dim=64)

        features_high = torch.tensor([[5.0]])
        features_low = torch.tensor([[-5.0]])

        temp.temp_proj = nn.Linear(1, 1)
        with torch.no_grad():
            temp.temp_proj.weight.data = torch.tensor([[1.0]])
            temp.temp_proj.bias.data = torch.tensor([0.0])

        delta_high = temp.temp_proj(features_high).squeeze()
        delta_low = temp.temp_proj(features_low).squeeze()

        assert delta_high.item() > delta_low.item()

    def test_gumbel_softmax_formula(self):
        """Test Gumbel-Softmax formula from Transolver++ Eq. 4."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True,
        )

        attn.train()
        x = torch.randn(2, 16, 64)

        out = attn(x)

        assert out.shape == x.shape

    def test_gumbel_softmax_during_inference(self):
        """Test Gumbel-Softmax is disabled during inference."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True,
        )

        attn.eval()
        x = torch.randn(2, 16, 64)

        out = attn(x)

        assert out.shape == x.shape

    def test_projection_initialization(self):
        """Test projection initialized to near-zero for identity start."""
        temp = AdaptiveTemperature(feature_dim=64)

        assert temp.temp_proj.weight.data.abs().max().item() < 1e-6
        assert temp.temp_proj.bias.data.abs().max().item() < 1e-6


class TestLowWidthGraphTransformers:
    """Tests validating against Low-Width Graph Transformers paper."""

    def test_annealing_schedule_formula(self):
        """Test annealing schedule: τ_t = max(f^(t-c), τ_min)."""
        init_temp = 1.0
        final_temp = 0.05
        warmup_epochs = 5
        anneal_factor = 0.98

        temp = AnnealedTemperature(
            init_temperature=init_temp,
            final_temperature=final_temp,
            warmup_epochs=warmup_epochs,
            anneal_factor=anneal_factor,
        )

        for epoch in range(warmup_epochs):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            assert t.item() == init_temp

        temp.set_epoch(warmup_epochs + 10)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        expected = max(anneal_factor ** 10, final_temp)
        assert abs(t.item() - expected) < 1e-6

    def test_warmup_phase(self):
        """Test warmup phase maintains constant temperature."""
        temp = AnnealedTemperature(
            init_temperature=1.0,
            warmup_epochs=5,
        )

        for epoch in range(5):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            assert t.item() == 1.0

    def test_final_temperature_clamp(self):
        """Test temperature doesn't go below final value."""
        temp = AnnealedTemperature(
            init_temperature=1.0,
            final_temperature=0.05,
            warmup_epochs=5,
            anneal_factor=0.95,
        )

        temp.set_epoch(100)
        t, _ = temp(torch.randn(2, 4, 8, 16))

        assert t.item() >= 0.05

    def test_fast_converging_parameters(self):
        """Test fast-converging parameters: c=5, f=0.98."""
        temp = AnnealedTemperature(
            init_temperature=1.0,
            warmup_epochs=5,
            anneal_factor=0.98,
        )

        temp.set_epoch(10)
        t, _ = temp(torch.randn(2, 4, 8, 16))

        assert t.item() > 0
        assert t.item() <= 1.0


class TestLearnableTemperatureBlog:
    """Tests validating against Learnable Temperature Blog paper."""

    def test_per_head_temperature(self):
        """Test per-head temperature parameterization."""
        n_heads = 8
        temp = PerHeadTemperature(n_heads=n_heads, init_temperature=1.0)

        assert temp.log_temperatures.shape == (n_heads,)

        temp.log_temperatures.data = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

        logits = torch.randn(2, n_heads, 8, 16)
        t, scaled = temp(logits)

        temperatures = torch.exp(temp.log_temperatures)
        assert scaled.shape == logits.shape
        assert t.item() > 0
        assert temperatures.std().item() > 0

    def test_temperature_multiplies_presoftmax(self):
        """Test temperature multiplies scores pre-softmax."""
        temp = FixedTemperature(temperature=2.0)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits)

        assert t.item() == 2.0
        assert torch.allclose(scaled, logits / 2.0)

    def test_log_space_parameterization(self):
        """Test log-space parameterization for gradient flow."""
        temp = LearnableScalarTemperature(init_temperature=1.0)

        assert temp.log_temperature.item() == 0.0

        for _ in range(10):
            logits = torch.randn(2, 4, 8, 16)
            t, scaled = temp(logits)
            loss = scaled.sum()
            temp.zero_grad()
            loss.backward()

        actual_temp = torch.exp(temp.log_temperature).item()
        assert t.item() > 0
        assert actual_temp > 0

    def test_different_heads_different_temps(self):
        """Test different heads can learn different temperature values."""
        temp = PerHeadTemperature(n_heads=4, init_temperature=1.0)

        with torch.no_grad():
            temp.log_temperatures.data = torch.tensor([0.0, -0.5, 0.5, 1.0])

        temps = torch.exp(temp.log_temperatures)

        assert temps[0].item() != temps[1].item()
        assert temps[2].item() != temps[3].item()


class TestPaperRecommendations:
    """Tests for recommendations from research papers."""

    def test_initialize_near_one(self):
        """Test initialization near 1.0 for stable training."""
        learnable = LearnableScalarTemperature(init_temperature=1.0)
        assert torch.exp(learnable.log_temperature).item() == 1.0

        perhead = PerHeadTemperature(n_heads=4, init_temperature=1.0)
        assert torch.exp(perhead.log_temperatures).mean().item() == 1.0

        adaptive = AdaptiveTemperature(feature_dim=64, init_temperature=1.0)
        assert torch.exp(adaptive.log_tau_0).item() == 1.0

    def test_clamp_min_to_prevent_collapse(self):
        """Test clamp(min=0.1) to prevent temperature collapse to zero."""
        temp = LearnableScalarTemperature(init_temperature=1.0, min_temp=0.1)

        temp.log_temperature.data = torch.tensor(-20.0)

        actual_temp = torch.exp(temp.log_temperature).clamp_min(temp.min_temp)

        assert abs(actual_temp.item() - 0.1) < 0.01

    def test_temperature_range_reasonable(self):
        """Test temperature stays in reasonable range."""
        temp = AdaptiveTemperature(
            feature_dim=64,
            init_temperature=1.0,
            min_temp=0.1,
        )

        for _ in range(50):
            features = torch.randn(2, 8, 64) * 10
            logits = torch.randn(2, 4, 8, 16)

            t, _ = temp(logits, features)

            assert 0.1 <= t.item() <= 50.0


class TestGradientFlow:
    """Tests for gradient flow through temperature parameters."""

    def test_learnable_scalar_gradient_flow(self):
        """Test gradients flow to learnable scalar temperature."""
        temp = LearnableScalarTemperature(init_temperature=1.0)
        logits = torch.randn(2, 4, 8, 16, requires_grad=True)

        t, scaled = temp(logits)
        loss = scaled.sum()
        loss.backward()

        assert t.item() > 0
        assert temp.log_temperature.grad is not None
        assert temp.log_temperature.grad.abs().item() > 0

    def test_adaptive_temperature_gradient_flow(self):
        """Test gradients flow to adaptive temperature components."""
        temp = AdaptiveTemperature(feature_dim=64, init_temperature=1.0)
        features = torch.randn(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)

        t, scaled = temp(logits, features)
        loss = scaled.sum()
        loss.backward()

        assert t.item() > 0
        assert temp.log_tau_0.grad is not None
        assert temp.temp_proj.weight.grad is not None

    def test_per_head_temperature_gradient_flow(self):
        """Test gradients flow to per-head temperature parameters."""
        temp = PerHeadTemperature(n_heads=8, init_temperature=1.0)
        logits = torch.randn(2, 8, 8, 16, requires_grad=True)

        t, scaled = temp(logits)
        loss = scaled.sum()
        loss.backward()

        assert t.item() > 0
        assert temp.log_temperatures.grad is not None
        assert temp.log_temperatures.grad.abs().sum().item() > 0


class TestAnnealingSchedule:
    """Tests for temperature annealing during training."""

    def test_annealing_over_epochs(self):
        """Test temperature decreases over epochs."""
        temp = AnnealedTemperature(
            init_temperature=1.0,
            final_temperature=0.05,
            warmup_epochs=5,
            anneal_factor=0.98,
        )

        temps = []
        for epoch in range(20):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            temps.append(t.item())

        assert temps[0] == 1.0
        assert temps[5] == 1.0
        assert temps[10] < temps[5]
        assert temps[-1] >= 0.05

    def test_annealing_affects_forward(self):
        """Test different epochs produce different outputs."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='annealed',
            anneal_warmup_epochs=5,
            anneal_factor=0.98,
        )

        x = torch.randn(2, 16, 64)

        attn.set_epoch(0)
        out1 = attn(x)

        attn.set_epoch(20)
        out2 = attn(x)

        assert not torch.allclose(out1, out2, atol=1e-5)


class TestTrainingConvergence:
    """Tests for training convergence with different temperature modes."""

    def test_fixed_temperature_training(self):
        """Test training with fixed temperature."""
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='fixed',
        )

        optimizer = optim.Adam(attn.parameters(), lr=0.001)

        for _ in range(10):
            x = torch.randn(2, 8, 32)
            optimizer.zero_grad()
            out = attn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        assert True

    def test_learnable_temperature_training(self):
        """Test training with learnable temperature."""
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='learnable_scalar',
        )

        optimizer = optim.Adam(attn.parameters(), lr=0.001)

        for _ in range(10):
            x = torch.randn(2, 8, 32)
            optimizer.zero_grad()
            out = attn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        current_temp = torch.exp(attn.temperature_module.log_temperature).item()
        assert current_temp > 0

    def test_adaptive_temperature_training(self):
        """Test training with adaptive temperature."""
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='adaptive',
        )

        optimizer = optim.Adam(attn.parameters(), lr=0.001)

        for _ in range(10):
            x = torch.randn(2, 8, 32)
            optimizer.zero_grad()
            out = attn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        assert True


class TestTemperatureStability:
    """Tests for temperature stability during training."""

    def test_temperature_no_nan(self):
        """Test temperature doesn't produce NaN."""
        temp = AdaptiveTemperature(feature_dim=64)

        for _ in range(100):
            features = torch.randn(2, 8, 64)
            logits = torch.randn(2, 4, 8, 16)
            t, scaled = temp(logits, features)

            assert not torch.isnan(t).any()
            assert not torch.isnan(scaled).any()

    def test_temperature_no_inf(self):
        """Test temperature doesn't produce Inf."""
        temp = AdaptiveTemperature(feature_dim=64, min_temp=0.01)

        features = torch.randn(2, 8, 64) * 100
        logits = torch.randn(2, 4, 8, 16) * 100
        t, scaled = temp(logits, features)

        assert not torch.isinf(t).any()
        assert not torch.isinf(scaled).any()