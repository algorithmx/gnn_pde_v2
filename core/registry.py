"""
Model registry for the GNN-PDE framework.

Two complementary registration mechanisms are provided:

1. **`ModelRegistry`** — a standalone registry object that can be instantiated,
   injected, and replaced independently of any model base class.  The
   module-level singleton :data:`MODEL_REGISTRY` is the default registry used
   by the whole framework::

       from gnn_pde_v2.core import MODEL_REGISTRY

       # Instantiate a registered model by name
       model = MODEL_REGISTRY.create('graphnet', node_in_dim=11, edge_in_dim=3, out_dim=3)

       # Register an external class (no inheritance required)
       @MODEL_REGISTRY.register('my_model', aliases=['mymodel'])
       class MyModel(nn.Module):
           ...

       # Register with namespace
       @MODEL_REGISTRY.register('solver', namespace='mylib')
       class MySolver(nn.Module):
           ...

       # Inspect
       print(MODEL_REGISTRY.list_models())
       print(MODEL_REGISTRY)   # ModelRegistry(afno, fno, graphnet, ...)

2. **`AutoRegisterModel`** — opt-in base class that auto-registers subclasses
   in :data:`MODEL_REGISTRY` via ``__init_subclass__``.  All class-method
   operations (``create``, ``list_models``, …) delegate to
   :data:`MODEL_REGISTRY` so both entry-points are always in sync::

       class MyModel(AutoRegisterModel, name='my_model', aliases=['mymodel']):
           def __init__(self, dim: int = 128) -> None:
               super().__init__()
               self.net = nn.Linear(dim, dim)

       # Either entry-point works
       m1 = AutoRegisterModel.create('my_model', dim=256)
       m2 = MODEL_REGISTRY.create('mymodel', dim=256)   # alias
"""

from __future__ import annotations

import warnings
from typing import Callable, ClassVar, Dict, List, Optional, Type

import torch.nn as nn

from .base import BaseModel


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Standalone model registry.

    Can be instantiated to create independent registries (e.g. for testing,
    plugins, or domain-specific subsets), or used via the module-level
    singleton :data:`MODEL_REGISTRY`.

    Example::

        from gnn_pde_v2.core import MODEL_REGISTRY

        # Decorator registration — no inheritance required
        @MODEL_REGISTRY.register('unet', aliases=['u_net'])
        class UNet(nn.Module):
            def __init__(self, channels: int = 64) -> None:
                super().__init__()

        model = MODEL_REGISTRY.create('u_net', channels=32)
        assert 'unet' in MODEL_REGISTRY
        print(MODEL_REGISTRY)          # ModelRegistry(..., unet, ...)
        print(MODEL_REGISTRY['unet'])  # <class 'UNet'>
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[nn.Module]] = {}

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _register_key(
        self,
        key: str,
        cls: Type[nn.Module],
        allow_overwrite: bool,
    ) -> None:
        """Write ``cls`` under ``key``, warning on conflict unless suppressed."""
        if key in self._registry and not allow_overwrite:
            existing = self._registry[key]
            warnings.warn(
                f"Model '{key}' already registered by "
                f"{existing.__module__}.{existing.__qualname__}. "
                f"Overwriting with {cls.__module__}.{cls.__qualname__}. "
                f"Use namespace= or aliases= to avoid conflicts, or "
                f"allow_overwrite=True to suppress this warning.",
                UserWarning,
                stacklevel=4,
            )
        self._registry[key] = cls

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def add(
        self,
        cls: Type[nn.Module],
        name: str,
        namespace: Optional[str] = None,
        allow_overwrite: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register ``cls`` imperatively.

        Args:
            cls: The class to register.
            name: Primary registration name (lowercased automatically).
            namespace: Optional prefix; produces ``'<namespace>.<name>'``.
            allow_overwrite: If ``True``, silently overwrite existing entries.
            aliases: Extra names that resolve to the same class.
        """
        primary = name.lower()
        if namespace:
            primary = f"{namespace.lower()}.{primary}"
        self._register_key(primary, cls, allow_overwrite)
        for alias in (aliases or []):
            alias_key = alias.lower()
            if namespace:
                alias_key = f"{namespace.lower()}.{alias_key}"
            self._register_key(alias_key, cls, allow_overwrite)

    def register(
        self,
        name: str,
        namespace: Optional[str] = None,
        allow_overwrite: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        """Decorator that registers an ``nn.Module`` subclass.

        The class is returned unmodified, so normal usage after decoration
        is unaffected.  No inheritance from :class:`AutoRegisterModel` is
        required.

        Example::

            @MODEL_REGISTRY.register('resnet', namespace='vision', aliases=['res'])
            class ResNet(nn.Module):
                def __init__(self, layers: int = 50) -> None:
                    super().__init__()

            model = MODEL_REGISTRY.create('vision.res', layers=18)

        Args:
            name: Primary name to register under.
            namespace: Optional namespace prefix.
            allow_overwrite: If ``True``, suppress overwrite warnings.
            aliases: Additional names that resolve to the same class.

        Returns:
            The original class, unmodified.
        """
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            self.add(
                cls,
                name=name,
                namespace=namespace,
                allow_overwrite=allow_overwrite,
                aliases=aliases,
            )
            return cls
        return decorator

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def create(self, name: str, **kwargs: object) -> nn.Module:
        """Instantiate a registered model by name.

        Args:
            name: Registered name (case-insensitive; aliases accepted).
            **kwargs: Forwarded verbatim to the model constructor.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(f"Unknown model: '{key}'. Available: [{available}]")
        return self._registry[key](**kwargs)

    def list_models(self, namespace: Optional[str] = None) -> List[str]:
        """Return a sorted list of registered names.

        Args:
            namespace: If provided, filter to names that start with
                ``'<namespace>.'``.
        """
        keys = sorted(self._registry)
        if namespace:
            prefix = f"{namespace.lower()}."
            keys = [k for k in keys if k.startswith(prefix)]
        return keys

    def get_model_info(self, name: str) -> dict:
        """Return metadata about a registered model.

        Args:
            name: Registered name (case-insensitive).

        Returns:
            Dict with keys ``name``, ``class``, ``module``, ``qualname``.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(f"Unknown model: '{key}'. Available: [{available}]")
        cls = self._registry[key]
        return {
            "name": key,
            "class": cls,
            "module": cls.__module__,
            "qualname": cls.__qualname__,
        }

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Registered name to remove (case-insensitive).

        Raises:
            KeyError: If ``name`` is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Model '{key}' is not registered")
        del self._registry[key]

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all registered models, or only those in a given namespace.

        Args:
            namespace: If provided, only remove models whose names start with
                ``'<namespace>.'``.  If ``None``, clear the entire registry.
        """
        if namespace is None:
            self._registry.clear()
        else:
            prefix = f"{namespace.lower()}."
            for key in [k for k in list(self._registry) if k.startswith(prefix)]:
                del self._registry[key]

    # Backwards-compatible alias for AutoRegisterModel.clear_registry callers
    def clear_registry(self, namespace: Optional[str] = None) -> None:
        """Alias for :meth:`clear` (backwards compatibility)."""
        self.clear(namespace)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Type[nn.Module]:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Model '{key}' is not registered")
        return self._registry[key]

    def __contains__(self, name: object) -> bool:
        return str(name).lower() in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._registry)) or "<empty>"
        return f"ModelRegistry({names})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

MODEL_REGISTRY = ModelRegistry()
"""Module-level :class:`ModelRegistry` singleton.

The framework's built-in models (:class:`~gnn_pde_v2.models.GraphNet`,
:class:`~gnn_pde_v2.models.MeshGraphNet`, :class:`~gnn_pde_v2.models.FNO`,
etc.) are all registered here automatically when their modules are imported.

Import and use directly for decorator-based or imperative registration::

    from gnn_pde_v2.core import MODEL_REGISTRY

    @MODEL_REGISTRY.register('my_solver', aliases=['solver'])
    class MySolver(nn.Module):
        ...

    model = MODEL_REGISTRY.create('solver')
"""


# ---------------------------------------------------------------------------
# AutoRegisterModel
# ---------------------------------------------------------------------------

class AutoRegisterModel(BaseModel):
    """Base class that auto-registers subclasses in :data:`MODEL_REGISTRY`.

    Subclass and supply ``name=`` to register.  All class-method operations
    delegate to :data:`MODEL_REGISTRY`, so both entry-points are always in sync::

        class MyModel(AutoRegisterModel, name='my_model', aliases=['mymodel']):
            def __init__(self, dim: int = 128) -> None:
                super().__init__()
                self.net = nn.Linear(dim, dim)

        # Both work:
        m1 = AutoRegisterModel.create('my_model', dim=256)
        m2 = MODEL_REGISTRY.create('mymodel', dim=256)

    .. tip::

        For external or third-party classes that you cannot subclass, use the
        :meth:`ModelRegistry.register` decorator on :data:`MODEL_REGISTRY`
        directly instead.

    Args:
        name: Registration name.  Omitting this argument emits a
            :class:`UserWarning` and defaults to the lowercased class name.
            Pass ``name=`` explicitly, or inherit from :class:`BaseModel`
            directly if you do not want the class to be registered.
        namespace: Optional namespace prefix.
        allow_overwrite: If ``True``, suppress overwrite warnings.
        aliases: Additional names that resolve to this class.
    """

    # Set per-subclass in __init_subclass__; useful for serialisation.
    _model_name: ClassVar[Optional[str]] = None

    # Backwards-compatible _registry attribute: points at the same dict
    # object as MODEL_REGISTRY._registry so that any code reading
    # ``AutoRegisterModel._registry`` or ``GraphNet._registry`` still works.
    _registry: ClassVar[Dict[str, Type["AutoRegisterModel"]]] = (
        MODEL_REGISTRY._registry  # type: ignore[assignment]
    )

    def __init_subclass__(
        cls,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        allow_overwrite: bool = False,
        aliases: Optional[List[str]] = None,
        **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if name is None:
            warnings.warn(
                f"{cls.__qualname__} subclasses AutoRegisterModel without an "
                f"explicit name= keyword argument. Defaulting to "
                f"'{cls.__name__.lower()}'. Pass "
                f"name='{cls.__name__.lower()}' to silence this warning, or "
                f"inherit from BaseModel directly if you don't want "
                f"this class to be registered.",
                UserWarning,
                stacklevel=2,
            )

        reg_name = (name or cls.__name__).lower()
        if namespace:
            reg_name = f"{namespace.lower()}.{reg_name}"
        cls._model_name = reg_name

        MODEL_REGISTRY.add(
            cls,
            name=(name or cls.__name__),
            namespace=namespace,
            allow_overwrite=allow_overwrite,
            aliases=aliases,
        )

    # ------------------------------------------------------------------
    # Backwards-compatible class-method API — all delegate to MODEL_REGISTRY
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs: object) -> "AutoRegisterModel":
        """Create a registered model by name (delegates to :data:`MODEL_REGISTRY`)."""
        return MODEL_REGISTRY.create(name, **kwargs)  # type: ignore[return-value]

    @classmethod
    def list_models(cls, namespace: Optional[str] = None) -> List[str]:
        """List registered model names (delegates to :data:`MODEL_REGISTRY`)."""
        return MODEL_REGISTRY.list_models(namespace)

    @classmethod
    def get_model_info(cls, name: str) -> dict:
        """Return metadata for a registered model (delegates to :data:`MODEL_REGISTRY`)."""
        return MODEL_REGISTRY.get_model_info(name)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a model from the registry (delegates to :data:`MODEL_REGISTRY`)."""
        MODEL_REGISTRY.unregister(name)

    @classmethod
    def clear_registry(cls, namespace: Optional[str] = None) -> None:
        """Clear the registry (delegates to :meth:`ModelRegistry.clear`)."""
        MODEL_REGISTRY.clear(namespace)
