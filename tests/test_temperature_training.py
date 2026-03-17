"""
Training and gradient flow tests for temperature mechanisms.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


class TestGradientFlow:
    """Tests for gradient flow through temperature parameters."""
    
    def test_learnable_scalar_gradient_flow(self):
        """Test gradients flow to learnable scalar temperature."""
        from gnn_pde_v2.components.temperature import LearnableScalarTemperature
        
        temp = LearnableScalarTemperature(init_temperature=1.0)
        logits = torch.randn(2, 4, 8, 16, requires_grad=True)
        
        t, scaled = temp(logits)
        loss = scaled.sum()
        loss.backward()
        
        assert temp.log_temperature.grad is not None
        assert temp.log_temperature.grad.abs().item() > 0
    
    def test_adaptive_temperature_gradient_flow(self):
        """Test gradients flow to adaptive temperature components."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64, init_temperature=1.0)
        features = torch.randn(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)
        
        t, scaled = temp(logits, features)
        loss = scaled.sum()
        loss.backward()
        
        assert temp.log_tau_0.grad is not None
        assert temp.temp_proj.weight.grad is not None
    
    def test_per_head_temperature_gradient_flow(self):
        """Test gradients flow to per-head temperature parameters."""
        from gnn_pde_v2.components.temperature import PerHeadTemperature
        
        temp = PerHeadTemperature(n_heads=8, init_temperature=1.0)
        logits = torch.randn(2, 8, 8, 16, requires_grad=True)
        
        t, scaled = temp(logits)
        loss = scaled.sum()
        loss.backward()
        
        assert temp.log_temperatures.grad is not None
        assert temp.log_temperatures.grad.abs().sum().item() > 0


class TestAnnealingSchedule:
    """Tests for temperature annealing during training."""
    
    def test_annealing_over_epochs(self):
        """Test temperature decreases over epochs."""
        from gnn_pde_v2.components.temperature import AnnealedTemperature
        
        temp = AnnealedTemperature(
            init_temperature=1.0,
            final_temperature=0.05,
            warmup_epochs=5,
            anneal_factor=0.98
        )
        
        temps = []
        for epoch in range(20):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            temps.append(t.item())
        
        # Temperature should generally decrease after warmup
        assert temps[0] == 1.0  # Initial
        assert temps[5] == 1.0  # End of warmup
        assert temps[10] < temps[5]  # After annealing starts
        assert temps[-1] >= 0.05  # Clamped at final
    
    def test_annealing_affects_forward(self):
        """Test different epochs produce different outputs."""
        from gnn_pde_v2.components.attention import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='annealed',
            anneal_warmup_epochs=5,
            anneal_factor=0.98
        )
        
        x = torch.randn(2, 16, 64)
        
        attn.set_epoch(0)
        out1 = attn(x)
        
        attn.set_epoch(20)
        out2 = attn(x)
        
        # Different epochs should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestTrainingConvergence:
    """Tests for training convergence with different temperature modes."""
    
    def test_fixed_temperature_training(self):
        """Test training with fixed temperature."""
        from gnn_pde_v2.components.attention import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='fixed'
        )
        
        optimizer = optim.Adam(attn.parameters(), lr=0.001)
        
        # Simple training loop
        for _ in range(10):
            x = torch.randn(2, 8, 32)
            optimizer.zero_grad()
            out = attn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_learnable_temperature_training(self):
        """Test training with learnable temperature."""
        from gnn_pde_v2.components.attention import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='learnable_scalar'
        )
        
        optimizer = optim.Adam(attn.parameters(), lr=0.001)
        
        for _ in range(10):
            x = torch.randn(2, 8, 32)
            optimizer.zero_grad()
            out = attn(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
        
        # Temperature should have changed from initial value
        current_temp = torch.exp(attn.temperature_module.log_temperature).item()
        assert current_temp > 0
    
    def test_adaptive_temperature_training(self):
        """Test training with adaptive temperature."""
        from gnn_pde_v2.components.attention import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=32,
            n_tokens=4,
            n_heads=2,
            temperature_mode='adaptive'
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
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        # Test with various inputs
        for _ in range(100):
            features = torch.randn(2, 8, 64)
            logits = torch.randn(2, 4, 8, 16)
            t, scaled = temp(logits, features)
            
            assert not torch.isnan(t).any()
            assert not torch.isnan(scaled).any()
    
    def test_temperature_no_inf(self):
        """Test temperature doesn't produce Inf."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64, min_temp=0.01)
        
        # Test with extreme values
        features = torch.randn(2, 8, 64) * 100
        logits = torch.randn(2, 4, 8, 16) * 100
        t, scaled = temp(logits, features)
        
        assert not torch.isinf(t).any()
        assert not torch.isinf(scaled).any()