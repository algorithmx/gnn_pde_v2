"""
Auto-registration for model classes (optional convenience).

This is an OPTIONAL feature. For lean usage, use plain nn.Module.
"""

from typing import Dict, Type, ClassVar, Optional
import torch.nn as nn


class AutoRegisterModel(nn.Module):
    """
    Base class with automatic model registration.
    
    This is OPTIONAL sugar for users who want config-based instantiation.
    For lean usage, use standard nn.Module or gnn_pde_v2.core.BaseModel.
    
    Example:
        class MyModel(AutoRegisterModel, name='my_model'):
            def __init__(self, dim=128):
                super().__init__()
                self.net = nn.Linear(dim, dim)
        
        # Later:
        model = AutoRegisterModel.create('my_model', dim=256)
    """
    
    _registry: ClassVar[Dict[str, Type['AutoRegisterModel']]] = {}
    _model_name: ClassVar[Optional[str]] = None
    
    def __init_subclass__(cls, name: Optional[str] = None, **kwargs):
        """Automatically register subclass."""
        super().__init_subclass__(**kwargs)
        
        reg_name = name or cls.__name__
        reg_name = reg_name.lower()
        
        cls._model_name = reg_name
        AutoRegisterModel._registry[reg_name] = cls
    
    @classmethod
    def create(cls, name: str, **kwargs) -> 'AutoRegisterModel':
        """
        Create model instance by name.
        
        Args:
            name: Registered model name
            **kwargs: Arguments passed to model constructor
            
        Returns:
            Instantiated model
        """
        name = name.lower()
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown model: '{name}'. Available: [{available}]")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return sorted(cls._registry.keys())
