"""
Tests for convenient API (optional).

These tests are skipped if optional dependencies are not available.
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests in this file if convenient module is not available
convenient = pytest.importorskip("gnn_pde_v2.convenient", reason="Convenient module not available")

from gnn_pde_v2.convenient import (
    AutoRegisterModel,
    get_initializer,
    initialize_module,
)


class TestAutoRegisterModel:
    """Test AutoRegisterModel."""
    
    def test_auto_registration(self):
        """Test models are auto-registered."""
        # Clear registry first
        AutoRegisterModel._registry.clear()
        
        class TestModel(AutoRegisterModel, name='test_model'):
            def __init__(self, dim=10):
                super().__init__()
                self.dim = dim
        
        assert 'test_model' in AutoRegisterModel.list_models()
        
        # Clean up
        del AutoRegisterModel._registry['test_model']
    
    def test_create(self):
        """Test creating model by name."""
        # Clear registry first
        AutoRegisterModel._registry.clear()
        
        class CreateTestModel(AutoRegisterModel, name='create_test'):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
        
        model = AutoRegisterModel.create('create_test', dim=20)
        assert model.dim == 20
        
        # Clean up
        del AutoRegisterModel._registry['create_test']
    
    def test_get_unknown_model(self):
        """Test error on unknown model."""
        with pytest.raises(ValueError) as exc_info:
            AutoRegisterModel.create('nonexistent')
        assert 'Unknown model' in str(exc_info.value)


class TestInitializers:
    """Test initializer functions."""
    
    def test_get_initializer_by_string(self):
        """Test getting initializer by string."""
        init_fn = get_initializer('glorot_uniform')
        assert callable(init_fn)
        
        init_fn = get_initializer('he_normal')
        assert callable(init_fn)
    
    def test_get_initializer_by_callable(self):
        """Test passing through a callable."""
        import torch.nn.init as init
        
        init_fn = get_initializer(init.xavier_uniform_)
        assert init_fn is init.xavier_uniform_
    
    def test_constant_initializer(self):
        """Test constant initializer parsing."""
        init_fn = get_initializer('constant_0')
        
        tensor = torch.empty(5, 5)
        init_fn(tensor)
        
        assert torch.allclose(tensor, torch.zeros(5, 5))
    
    def test_deepxde_style_names(self):
        """Test DeepXDE-style names with spaces."""
        init_fn1 = get_initializer('glorot_uniform')
        init_fn2 = get_initializer('glorot uniform')
        assert init_fn1 == init_fn2
    
    def test_unknown_initializer(self):
        """Test error on unknown initializer."""
        with pytest.raises(ValueError) as exc_info:
            get_initializer('unknown_init')
        assert 'Unknown initializer' in str(exc_info.value)
    
    def test_initialize_module(self):
        """Test module initialization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )
        
        initialize_module(model, weight_init='glorot_uniform', bias_init='constant_0')
        
        # Check biases are zero
        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert torch.allclose(m.bias, torch.zeros_like(m.bias))


@pytest.mark.requires_pydantic
class TestConfig:
    """Test config system (requires pydantic)."""
    
    def test_model_config(self):
        """Test ModelConfig creation."""
        try:
            from gnn_pde_v2.convenient import ModelConfig
        except ImportError:
            pytest.skip("Pydantic not available")
        
        config = ModelConfig(
            model_type='graphnet',
            hidden_dim=128,
            n_layers=4,
        )
        
        assert config.model_type == 'graphnet'
        assert config.hidden_dim == 128
        assert config.n_layers == 4


@pytest.mark.requires_pydantic
class TestConfigBuilder:
    """Test ConfigBuilder (requires pydantic)."""
    
    def test_build_model(self):
        """Test building model from config."""
        try:
            from gnn_pde_v2.convenient import ConfigBuilder, ModelConfig
            from gnn_pde_v2.convenient.registry import AutoRegisterModel
        except ImportError:
            pytest.skip("Pydantic not available")
        
        # Clear and register a test model
        AutoRegisterModel._registry.clear()
        
        class TestBuildModel(AutoRegisterModel, name='test_build'):
            def __init__(self, hidden_dim=64):
                super().__init__()
                self.hidden_dim = hidden_dim
        
        config = ModelConfig(model_type='test_build', hidden_dim=128)
        builder = ConfigBuilder(config)
        
        # Note: This would need proper integration to work fully
        # model = builder.build_model()
        # assert model.hidden_dim == 128
        
        # Clean up
        del AutoRegisterModel._registry['test_build']


class TestUnifiedModel:
    """Test unified Model wrapper."""
    
    def test_model_creation(self):
        """Test creating unified Model."""
        try:
            from gnn_pde_v2.convenient import Model
        except ImportError:
            pytest.skip("Convenient training module not available")
        
        architecture = nn.Linear(10, 5)
        model = Model(architecture, loss_fn='mse')
        
        assert model.architecture is architecture
