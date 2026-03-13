"""
Auto-registration for model classes.

Provides optional registry-based model instantiation for config-driven workflows.
"""

from typing import Dict, Type, ClassVar, Optional
import warnings

from .base import BaseModel


class AutoRegisterModel(BaseModel):
    """
    Base class with automatic model registration.

    Inherits from BaseModel to provide a unified type hierarchy while adding
    optional registry-based instantiation for config-driven workflows.

    For lean usage, use standard nn.Module or gnn_pde_v2.core.BaseModel directly.

    Args:
        name: Custom registration name (defaults to class name, lowercased)
        namespace: Optional namespace prefix (e.g., 'example' -> 'example.modelname')
        allow_overwrite: If True, silently overwrite existing registration

    Example:
        class MyModel(AutoRegisterModel, name='my_model'):
            def __init__(self, dim=128):
                super().__init__()
                self.net = nn.Linear(dim, dim)

        # Later:
        model = AutoRegisterModel.create('my_model', dim=256)

        # With namespace (avoids collisions):
        class ExampleModel(AutoRegisterModel, name='model', namespace='example'):
            ...
        # Registers as 'example.model'
    """

    _registry: ClassVar[Dict[str, Type['AutoRegisterModel']]] = {}
    _model_name: ClassVar[Optional[str]] = None

    def __init_subclass__(
        cls,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        allow_overwrite: bool = False,
        **kwargs
    ):
        """Automatically register subclass."""
        super().__init_subclass__(**kwargs)

        reg_name = name or cls.__name__
        reg_name = reg_name.lower()

        # Add namespace prefix if provided
        if namespace:
            reg_name = f"{namespace.lower()}.{reg_name}"

        # Check for conflicts and warn
        if reg_name in cls._registry and not allow_overwrite:
            existing = cls._registry[reg_name]
            warnings.warn(
                f"Model '{reg_name}' already registered by "
                f"{existing.__module__}.{existing.__qualname__}. "
                f"Overwriting with {cls.__module__}.{cls.__qualname__}. "
                f"Use namespace= to avoid conflicts or allow_overwrite=True to suppress.",
                UserWarning,
                stacklevel=2
            )

        cls._model_name = reg_name
        AutoRegisterModel._registry[reg_name] = cls

    @classmethod
    def create(cls, name: str, **kwargs) -> 'AutoRegisterModel':
        """
        Create model instance by name.

        Args:
            name: Registered model name (can include namespace prefix)
            **kwargs: Arguments passed to model constructor

        Returns:
            Instantiated model

        Raises:
            ValueError: If model name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown model: '{name}'. Available: [{available}]")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls, namespace: Optional[str] = None) -> list:
        """
        List all registered model names.

        Args:
            namespace: If provided, filter to models in this namespace

        Returns:
            Sorted list of registered model names
        """
        models = sorted(cls._registry.keys())
        if namespace:
            prefix = f"{namespace.lower()}."
            models = [m for m in models if m.startswith(prefix)]
        return models

    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """
        Get information about a registered model.

        Args:
            name: Registered model name

        Returns:
            Dict with model class and registration info
        """
        name = name.lower()
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown model: '{name}'. Available: [{available}]")
        model_cls = cls._registry[name]
        return {
            'name': name,
            'class': model_cls,
            'module': model_cls.__module__,
            'qualname': model_cls.__qualname__,
        }

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a model from the registry.

        Args:
            name: Registered model name to remove

        Raises:
            KeyError: If model name is not registered
        """
        name = name.lower()
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' is not registered")
        del AutoRegisterModel._registry[name]

    @classmethod
    def clear_registry(cls, namespace: Optional[str] = None) -> None:
        """
        Clear all registered models, or only those in a given namespace.

        Args:
            namespace: If provided, only remove models whose names start with
                ``'<namespace>.'``.  If ``None``, clear the entire registry.
        """
        if namespace is None:
            AutoRegisterModel._registry.clear()
        else:
            prefix = f"{namespace.lower()}."
            keys = [k for k in AutoRegisterModel._registry if k.startswith(prefix)]
            for k in keys:
                del AutoRegisterModel._registry[k]
