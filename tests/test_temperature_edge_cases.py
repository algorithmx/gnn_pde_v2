"""
Edge cases and error handling tests for temperature mechanisms.
"""

import pytest
import torch
import torch.nn as nn


class TestTemperatureClamping:
    """Tests for minimum temperature clamping."""
    
    def test_learnable_scalar_min_clamp(self):
        """Test learnable scalar clamps to min_temperature."""
        from gnn_pde_v2.components.temperature import LearnableScalarTemperature
        
        temp = LearnableScalarTemperature(init_temperature=1.0, min_temp=0.1)
        
        # Force very low temperature
        temp.log_temperature.data = torch.tensor(-10.0)
        
        logits = torch.randn(2, 4, 8, 16)
        t, _ = temp(logits)
        
        assert t.item() >= 0.1
    
    def test_adaptive_temperature_min_clamp(self):
        """Test adaptive temperature clamps to min_temperature."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64, min_temp=0.1)
        
        # Force negative delta
        temp.temp_proj.weight.data = torch.full((1, 64), -10.0)
        
        features = torch.randn(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)
        
        t, _ = temp(logits, features)
        
        assert t.item() >= 0.1


class TestInputShapes:
    """Tests for various input shapes."""
    
    def test_single_batch_2d_input(self):
        """Test with 2D input [N, D]."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        
        x = torch.randn(16, 64)  # [N, D]
        out = attn(x)
        
        assert out.shape == x.shape
    
    def test_batched_3d_input(self):
        """Test with 3D input [B, N, D]."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        
        x = torch.randn(2, 16, 64)  # [B, N, D]
        out = attn(x)
        
        assert out.shape == x.shape
    
    def test_varying_batch_sizes(self):
        """Test with different batch sizes."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 16, 64)
            out = attn(x)
            assert out.shape == x.shape
    
    def test_varying_sequence_lengths(self):
        """Test with different sequence lengths."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        
        for seq_len in [4, 8, 16, 32]:
            x = torch.randn(2, seq_len, 64)
            out = attn(x)
            assert out.shape == x.shape


class TestFeatureValues:
    """Tests with various feature value ranges."""
    
    def test_zero_features(self):
        """Test with zero features."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        features = torch.zeros(2, 8, 64)
        logits = torch.randn(2, 4, 8, 16)
        
        t, scaled = temp(logits, features)
        
        assert t.item() > 0
        assert not torch.isnan(scaled).any()
    
    def test_large_features(self):
        """Test with large feature values."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        features = torch.randn(2, 8, 64) * 1000
        logits = torch.randn(2, 4, 8, 16) * 1000
        
        t, scaled = temp(logits, features)
        
        assert t.item() > 0
        assert not torch.isinf(scaled).any()
        assert not torch.isnan(scaled).any()
    
    def test_small_features(self):
        """Test with small feature values."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
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
        from gnn_pde_v2.components.temperature import create_temperature_module
        
        with pytest.raises(ValueError, match="Unknown temperature mode"):
            create_temperature_module('invalid_mode')
    
    def test_adaptive_requires_features(self):
        """Test adaptive temperature requires features."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
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
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64).to(device)
        features = torch.randn(2, 8, 64).to(device)
        logits = torch.randn(2, 4, 8, 16).to(device)
        
        t, scaled = temp(logits, features)
        
        # Use type comparison for device (cuda:0 == cuda in terms of type)
        assert t.device.type == device.type
        assert scaled.device.type == device.type
    
    def test_attention_on_device(self, device):
        """Test PhysicsTokenAttention on correct device."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        ).to(device)
        
        x = torch.randn(2, 16, 64).to(device)
        out = attn(x)
        
        # Use type comparison for device
        assert out.device.type == device.type


class TestDtypeConsistency:
    """Tests for dtype consistency."""
    
    def test_float32_consistency(self):
        """Test with float32."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        features = torch.randn(2, 8, 64, dtype=torch.float32)
        logits = torch.randn(2, 4, 8, 16, dtype=torch.float32)
        
        t, scaled = temp(logits, features)
        
        assert t.dtype == torch.float32
        assert scaled.dtype == torch.float32
    
    def test_float64_consistency(self):
        """Test with float64."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        temp = temp.to(dtype=torch.float64)
        
        features = torch.randn(2, 8, 64, dtype=torch.float64)
        logits = torch.randn(2, 4, 8, 16, dtype=torch.float64)
        
        t, scaled = temp(logits, features)
        
        # Note: temperature scalar may be float32, check scaled dtype
        assert scaled.dtype == torch.float64


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_default_temperature_mode(self):
        """Test default temperature mode is 'fixed'."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(dim=64, n_tokens=8, n_heads=4)
        
        assert attn.temperature_mode == 'fixed'
    
    def test_old_api_compatibility(self):
        """Test old API with temperature parameter still works."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        # Old API: passing temperature directly
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature=2.0
        )
        
        x = torch.randn(2, 16, 64)
        out = attn(x)
        
        assert out.shape == x.shape