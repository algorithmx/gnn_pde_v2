"""
Model registry for the GNN-PDE framework.

There is exactly **one** registration mechanism: subclass
:class:`AutoRegisterModel` and pass ``name=`` (plus optional ``aliases=`` and
``namespace=``). Every registered model is written to the shared
:data:`MODEL_REGISTRY` store, which exposes only lookup / creation /
introspection APIs — it is no longer mutated directly by user code::

    from gnn_pde_v2.core import AutoRegisterModel, MODEL_REGISTRY

    class MyModel(AutoRegisterModel, name='my_model', aliases=['mymodel']):
        def __init__(self, dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Linear(dim, dim)

    # Instantiate a registered model by name
    model = MODEL_REGISTRY.create('my_model', dim=256)
    model = MODEL_REGISTRY.create('mymodel', dim=256)   # alias

    # Register under a namespace
    class MySolver(AutoRegisterModel, name='solver', namespace='mylib'):
        ...

    # Inspect
    print(MODEL_REGISTRY.list_models())
    print(MODEL_REGISTRY)   # ModelRegistry(afno, fno, graphnet, ...)

Subclassing :class:`AutoRegisterModel` is the only supported way to add a model
to the registry. To opt a class *out* of registration, inherit from
:class:`~gnn_pde_v2.core.BaseModel` (or plain ``nn.Module``) instead.
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Dict, List, Optional, Type

import torch.nn as nn

from .base import BaseModel


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Shared model registry for name -> class lookup.

    Holds the ``name -> class`` mapping that :class:`AutoRegisterModel`
    writes to at subclass-definition time. It exposes lookup / creation /
    introspection / removal APIs, but does **not** expose a public write
    surface: the only way to add a model is to subclass
    :class:`AutoRegisterModel`.

    Can also be instantiated to build independent registries (e.g. for
    testing or plugins) that an :class:`AutoRegisterModel` subclass can
    target via the ``registry=`` class keyword.

    Example::

        from gnn_pde_v2.core import MODEL_REGISTRY

        model = MODEL_REGISTRY.create('graphnet', node_in_dim=11, edge_in_dim=3, out_dim=3)
        assert 'graphnet' in MODEL_REGISTRY
        print(MODEL_REGISTRY)              # ModelRegistry(..., graphnet, ...)
        print(MODEL_REGISTRY['graphnet'])  # <class 'GraphNet'>
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[nn.Module]] = {}

    # ------------------------------------------------------------------
    # Internal helper — only AutoRegisterModel should call this
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
                stacklevel=3,
            )
        self._registry[key] = cls

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
"""Module-level :class:`ModelRegistry` singleton — the shared name -> class
store for the framework.

The framework's built-in models (:class:`~gnn_pde_v2.models.GraphNet`,
:class:`~gnn_pde_v2.models.MeshGraphNet`, :class:`~gnn_pde_v2.models.FNO`,
:class:`~gnn_pde_v2.models.MultiscaleFNO`, etc.) register themselves here
automatically when their modules are imported, because they subclass
:class:`AutoRegisterModel`.

Models are added **only** by subclassing :class:`AutoRegisterModel`; this
singleton exposes lookup / creation / introspection, not direct registration::


    from gnn_pde_v2.core import AutoRegisterModel, MODEL_REGISTRY

    class MySolver(AutoRegisterModel, name='my_solver', aliases=['solver']):
        def __init__(self) -> None:
            super().__init__()

    model = MODEL_REGISTRY.create('solver')
"""


# ---------------------------------------------------------------------------
# AutoRegisterModel
# ---------------------------------------------------------------------------

class AutoRegisterModel(BaseModel):
    """The single, canonical way to register a model.

    Subclass and supply ``name=`` to register the class in
    :data:`MODEL_REGISTRY` at class-definition (import) time. All
    class-method operations delegate to :data:`MODEL_REGISTRY`, so the mixin
    and the singleton are always in sync::

        class MyModel(AutoRegisterModel, name='my_model', aliases=['mymodel']):
            def __init__(self, dim: int = 128) -> None:
                super().__init__()
                self.net = nn.Linear(dim, dim)

        # Both work:
        m1 = AutoRegisterModel.create('my_model', dim=256)
        m2 = MODEL_REGISTRY.create('mymodel', dim=256)

    To add a class that should **not** be registered (e.g. an internal building
    block, or the :class:`~gnn_pde_v2.models.EncodeProcessDecode` combinator),
    inherit from :class:`~gnn_pde_v2.core.BaseModel` (or plain ``nn.Module``)
    instead — those classes never touch the registry.

    Args:
        name: Registration name.  Omitting this argument emits a
            :class:`UserWarning` and defaults to the lowercased class name.
            Always pass ``name=`` for real models.
        namespace: Optional namespace prefix (produces ``'<namespace>.<name>'``).
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

        # Compute the primary registration key (lowercased, namespaced).
        reg_name = (name or cls.__name__).lower()
        if namespace:
            reg_name = f"{namespace.lower()}.{reg_name}"
        cls._model_name = reg_name

        # Register the primary name and every alias in the shared store.
        MODEL_REGISTRY._register_key(reg_name, cls, allow_overwrite)
        for alias in (aliases or []):
            alias_key = alias.lower()
            if namespace:
                alias_key = f"{namespace.lower()}.{alias_key}"
            MODEL_REGISTRY._register_key(alias_key, cls, allow_overwrite)

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
