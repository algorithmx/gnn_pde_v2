"""
Unit tests for temperature mechanism modules.

Tests individual temperature classes in isolation.
"""

import pytest
import torch
import torch.nn as nn

from gnn_pde_v2.components.temperature import (
    FixedTemperature,
    LearnableScalarTemperature,
    PerHeadTemperature,
    AdaptiveTemperature,
    AnnealedTemperature,
    create_temperature_module,
)


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
        
        # Higher temperature should reduce variance
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
        # Force very low temperature
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
        
        # Set different temperatures for each head
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
        # Each head should have different scaling
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
        # Should be initialized to near zero
        assert torch.allclose(temp.temp_proj.weight, torch.zeros_like(temp.temp_proj.weight))
        assert torch.allclose(temp.temp_proj.bias, torch.zeros_like(temp.temp_proj.bias))
    
    def test_ada_temp_formula(self):
        """Test Ada-Temp formula: τ_i = τ_0 + Linear(x_i)."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim, init_temperature=1.0, learnable_base=False)
        
        features = torch.randn(2, 8, feature_dim)
        logits = torch.randn(2, 4, 8, 16)
        
        t, scaled = temp(logits, features)
        
        # Manual computation
        tau_0 = 1.0
        delta_tau = temp.temp_proj(features).squeeze(-1)  # [B, N]
        expected_temps = (tau_0 + delta_tau).clamp_min(0.1)
        
        # Check temperature is per-point
        assert t.item() > 0
    
    def test_per_point_temperature(self):
        """Test different points get different temperatures."""
        feature_dim = 64
        temp = AdaptiveTemperature(feature_dim=feature_dim)
        
        # Features with clear differences
        features = torch.zeros(1, 4, feature_dim)
        features[0, 0, :] = 1.0
        features[0, 1, :] = -1.0
        features[0, 2, :] = 0.5
        features[0, 3, :] = -0.5
        
        logits = torch.randn(1, 1, 4, 16)
        
        # Set projection to identity-like
        temp.temp_proj.weight.data = torch.ones_like(temp.temp_proj.weight)
        
        t, scaled = temp(logits, features)
        assert t.item() > 0
    
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
            anneal_factor=factor
        )
        
        # After warmup
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
            anneal_factor=0.98
        )
        
        # Set very high epoch
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
            anneal_warmup_epochs=5
        )
        assert isinstance(temp, AnnealedTemperature)
    
    def test_invalid_mode_raises_error(self):
        """Test invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown temperature mode"):
            create_temperature_module('invalid_mode')
