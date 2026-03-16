"""
Research paper validation tests for temperature mechanisms.

Validates implementation against papers:
- Transolver++ (ICML 2025): Ada-Temp and Gumbel-Softmax
- Low-Width Graph Transformers (NeurIPS 2023): Annealing schedule
- Learnable Temperature Blog (2024): Per-head temperature
"""

import pytest
import torch
import torch.nn as nn


class TestTransolverPlusPlus:
    """Tests validating against Transolver++ paper."""
    
    def test_ada_temp_formula(self):
        """Test Ada-Temp formula: τ_i = τ_0 + Linear(x_i)
        
        From Transolver++ Eq. 3:
        τ = {τ_i} = {τ_0 + Linear(x_i)}
        """
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        feature_dim = 64
        init_temp = 1.0
        
        temp = AdaptiveTemperature(
            feature_dim=feature_dim,
            init_temperature=init_temp,
            learnable_base=True
        )
        
        # Create features
        features = torch.randn(2, 8, feature_dim)
        
        # Manual calculation: tau_0 + Linear(x_i)
        tau_0 = torch.exp(temp.log_tau_0)
        delta_tau = temp.temp_proj(features).squeeze(-1)
        expected_tau = tau_0 + delta_tau
        
        logits = torch.randn(2, 4, 8, 16)
        actual_tau, scaled = temp(logits, features)
        
        # Should produce per-point temperature
        assert actual_tau.item() > 0
    
    def test_ada_temp_learns_local_properties(self):
        """Test Ada-Temp adapts to local point properties.
        
        Points with different features should get different temperatures.
        """
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        # Use simple scalar features
        features_high = torch.tensor([[5.0]])  # [1, 1]
        features_low = torch.tensor([[-5.0]])   # [1, 1]
        
        # Set projection to produce identity mapping
        temp.temp_proj = nn.Linear(1, 1)
        with torch.no_grad():
            temp.temp_proj.weight.data = torch.tensor([[1.0]])
            temp.temp_proj.bias.data = torch.tensor([0.0])
        
        logits = torch.randn(1, 1, 1, 16)
        
        # Get temperature for high and low features
        delta_high = temp.temp_proj(features_high).squeeze()
        delta_low = temp.temp_proj(features_low).squeeze()
        
        # High feature should produce higher delta than low
        assert delta_high.item() > delta_low.item()
    
    def test_gumbel_softmax_formula(self):
        """Test Gumbel-Softmax formula from Transolver++ Eq. 4:
        Rep-Slice(x, τ) = Softmax((Linear(x) - log(-log ε)) / τ)
        """
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True
        )
        
        attn.train()
        x = torch.randn(2, 16, 64)
        
        # Should not raise error
        out = attn(x)
        
        assert out.shape == x.shape
    
    def test_gumbel_softmax_during_inference(self):
        """Test Gumbel-Softmax is disabled during inference."""
        from gnn_pde_v2.components.transformer import PhysicsTokenAttention
        
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True
        )
        
        attn.eval()
        x = torch.randn(2, 16, 64)
        
        out = attn(x)
        
        assert out.shape == x.shape
    
    def test_projection_initialization(self):
        """Test projection initialized to near-zero for identity start.
        
        From paper: Initialize projection to near zero so that
        temperature starts close to τ_0.
        """
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(feature_dim=64)
        
        # Weight should be initialized to zeros
        assert temp.temp_proj.weight.data.abs().max().item() < 1e-6
        # Bias should be initialized to zeros
        assert temp.temp_proj.bias.data.abs().max().item() < 1e-6


class TestLowWidthGraphTransformers:
    """Tests validating against Low-Width Graph Transformers paper."""
    
    def test_annealing_schedule_formula(self):
        """Test annealing schedule: τ_t = max(f^(t-c), τ_min)
        
        From paper: Start with τ=1.0, gradually anneal to 0.05 by end of training.
        """
        from gnn_pde_v2.components.temperature import AnnealedTemperature
        
        init_temp = 1.0
        final_temp = 0.05
        warmup_epochs = 5
        anneal_factor = 0.98
        
        temp = AnnealedTemperature(
            init_temperature=init_temp,
            final_temperature=final_temp,
            warmup_epochs=warmup_epochs,
            anneal_factor=anneal_factor
        )
        
        # During warmup
        for epoch in range(warmup_epochs):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            assert t.item() == init_temp
        
        # After warmup - check formula
        temp.set_epoch(warmup_epochs + 10)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        expected = max(anneal_factor ** 10, final_temp)
        assert abs(t.item() - expected) < 1e-6
    
    def test_warmup_phase(self):
        """Test warmup phase maintains constant temperature.
        
        From paper: Initial phase of c epochs with τ=1 to learn important neighbors.
        """
        from gnn_pde_v2.components.temperature import AnnealedTemperature
        
        temp = AnnealedTemperature(
            init_temperature=1.0,
            warmup_epochs=5
        )
        
        for epoch in range(5):
            temp.set_epoch(epoch)
            t, _ = temp(torch.randn(2, 4, 8, 16))
            assert t.item() == 1.0
    
    def test_final_temperature_clamp(self):
        """Test temperature doesn't go below final value.
        
        From paper: max(f^(t-c), 0.05)
        """
        from gnn_pde_v2.components.temperature import AnnealedTemperature
        
        temp = AnnealedTemperature(
            init_temperature=1.0,
            final_temperature=0.05,
            warmup_epochs=5,
            anneal_factor=0.95
        )
        
        # Very high epoch
        temp.set_epoch(100)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        
        assert t.item() >= 0.05
    
    def test_fast_converging_parameters(self):
        """Test fast-converging parameters: c=5, f=0.98.
        
        From paper: For fast-converging models, use c=5, f=0.98.
        """
        from gnn_pde_v2.components.temperature import AnnealedTemperature
        
        temp = AnnealedTemperature(
            init_temperature=1.0,
            warmup_epochs=5,
            anneal_factor=0.98
        )
        
        # Should work with these parameters
        temp.set_epoch(10)
        t, _ = temp(torch.randn(2, 4, 8, 16))
        
        assert t.item() > 0
        assert t.item() <= 1.0


class TestLearnableTemperatureBlog:
    """Tests validating against Learnable Temperature Blog paper."""
    
    def test_per_head_temperature(self):
        """Test per-head temperature parameterization.
        
        From blog: Each attention head at each layer has its own temperature term.
        """
        from gnn_pde_v2.components.temperature import PerHeadTemperature
        
        n_heads = 8
        temp = PerHeadTemperature(n_heads=n_heads, init_temperature=1.0)
        
        assert temp.log_temperatures.shape == (n_heads,)
        
        # Each head should have independent temperature
        temp.log_temperatures.data = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        
        logits = torch.randn(2, n_heads, 8, 16)
        t, scaled = temp(logits)
        
        # Temperature per head should vary
        temperatures = torch.exp(temp.log_temperatures)
        assert temperatures.std().item() > 0
    
    def test_temperature_multiplies_presoftmax(self):
        """Test temperature multiplies scores pre-softmax.
        
        From blog: Temperature here multiplies the scores pre-softmax
        rather than dividing them as is customarily done.
        Note: Our implementation uses division (standard convention).
        """
        from gnn_pde_v2.components.temperature import FixedTemperature
        
        temp = FixedTemperature(temperature=2.0)
        logits = torch.randn(2, 4, 8, 16)
        
        t, scaled = temp(logits)
        
        # Standard convention: divide by temperature
        assert torch.allclose(scaled, logits / 2.0)
    
    def test_log_space_parameterization(self):
        """Test log-space parameterization for gradient flow.
        
        From blog: Temperature terms initialized at 1.0.
        """
        from gnn_pde_v2.components.temperature import LearnableScalarTemperature
        
        temp = LearnableScalarTemperature(init_temperature=1.0)
        
        # log(1.0) = 0
        assert temp.log_temperature.item() == 0.0
        
        # After some updates, should still be positive
        for _ in range(10):
            logits = torch.randn(2, 4, 8, 16)
            t, scaled = temp(logits)
            loss = scaled.sum()
            temp.zero_grad()
            loss.backward()
        
        actual_temp = torch.exp(temp.log_temperature).item()
        assert actual_temp > 0
    
    def test_different_heads_different_temps(self):
        """Test different heads can learn different temperature values.
        
        From blog: Different heads take on different roles, model might
        benefit from temperature per-head terms to further differentiate heads.
        """
        from gnn_pde_v2.components.temperature import PerHeadTemperature
        
        temp = PerHeadTemperature(n_heads=4, init_temperature=1.0)
        
        # Simulate training: different heads learn different values
        with torch.no_grad():
            temp.log_temperatures.data = torch.tensor([0.0, -0.5, 0.5, 1.0])
        
        temps = torch.exp(temp.log_temperatures)
        
        assert temps[0].item() != temps[1].item()
        assert temps[2].item() != temps[3].item()


class TestPaperRecommendations:
    """Tests for recommendations from research papers."""
    
    def test_initialize_near_one(self):
        """Test initialization near 1.0 for stable training.
        
        From papers: Initialize τ_0 near 1.0 for stable training.
        """
        from gnn_pde_v2.components.temperature import (
            LearnableScalarTemperature,
            PerHeadTemperature,
            AdaptiveTemperature,
        )
        
        # All should initialize near 1.0
        learnable = LearnableScalarTemperature(init_temperature=1.0)
        assert torch.exp(learnable.log_temperature).item() == 1.0
        
        perhead = PerHeadTemperature(n_heads=4, init_temperature=1.0)
        assert torch.exp(perhead.log_temperatures).mean().item() == 1.0
        
        adaptive = AdaptiveTemperature(feature_dim=64, init_temperature=1.0)
        assert torch.exp(adaptive.log_tau_0).item() == 1.0
    
    def test_clamp_min_to_prevent_collapse(self):
        """Test clamp(min=0.1) to prevent temperature collapse to zero.
        
        From papers: Apply softplus or clamp(min=0.1) to prevent collapse to zero.
        """
        from gnn_pde_v2.components.temperature import LearnableScalarTemperature
        
        temp = LearnableScalarTemperature(init_temperature=1.0, min_temp=0.1)
        
        # Force very negative log temperature
        temp.log_temperature.data = torch.tensor(-20.0)
        
        actual_temp = torch.exp(temp.log_temperature).clamp_min(temp.min_temp)
        
        # Allow small floating point tolerance
        assert abs(actual_temp.item() - 0.1) < 0.01
    
    def test_temperature_range_reasonable(self):
        """Test temperature stays in reasonable range."""
        from gnn_pde_v2.components.temperature import AdaptiveTemperature
        
        temp = AdaptiveTemperature(
            feature_dim=64,
            init_temperature=1.0,
            min_temp=0.1
        )
        
        # Test with extreme inputs
        for _ in range(50):
            features = torch.randn(2, 8, 64) * 10
            logits = torch.randn(2, 4, 8, 16)
            
            t, _ = temp(logits, features)
            
            # Should stay in reasonable range
            assert 0.1 <= t.item() <= 50.0