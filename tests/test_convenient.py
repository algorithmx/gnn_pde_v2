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
)


class TestAutoRegisterModel:
    """Test AutoRegisterModel."""
    
    def test_auto_registration(self):
        """Test models are auto-registered."""
        # Clear registry first
        AutoRegisterModel.clear_registry()
        
        class TestModel(AutoRegisterModel, name='test_model'):
            def __init__(self, dim=10):
                super().__init__()
                self.dim = dim
        
        assert 'test_model' in AutoRegisterModel.list_models()
        
        # Clean up
        AutoRegisterModel.unregister('test_model')
    
    def test_create(self):
        """Test creating model by name."""
        # Clear registry first
        AutoRegisterModel.clear_registry()
        
        class CreateTestModel(AutoRegisterModel, name='create_test'):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
        
        model = AutoRegisterModel.create('create_test', dim=20)
        assert model.dim == 20
        
        # Clean up
        AutoRegisterModel.unregister('create_test')
    
    def test_get_unknown_model(self):
        """Test error on unknown model."""
        with pytest.raises(ValueError) as exc_info:
            AutoRegisterModel.create('nonexistent')
        assert 'Unknown model' in str(exc_info.value)


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

