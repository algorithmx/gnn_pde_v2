"""Model implementations."""

# Core model always available
from .encode_process_decode import EncodeProcessDecode

# Optional models with lazy loading
_LAZY_MODELS = {
    'FNO': ('.fno_model', 'FNO'),
    'TFNO': ('.fno_model', 'TFNO'),
    'AFNO': ('.fno_model', 'AFNO'),
    'GraphNet': ('.gnn_model', 'GraphNet'),
    'MeshGraphNet': ('.gnn_model', 'MeshGraphNet'),
}

__all__ = [
    "EncodeProcessDecode",
    "FNO",
    "TFNO",
    "AFNO",
    "GraphNet",
    "MeshGraphNet",
]


def __getattr__(name: str):
    """Lazy import models with helpful error messages."""
    if name in _LAZY_MODELS:
        module_name, class_name = _LAZY_MODELS[name]
        try:
            module = __import__(
                module_name, globals(), locals(), fromlist=[class_name], level=1
            )
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires additional dependencies. "
                f"Original error: {e}\n"
                f"Install with: pip install gnn-pde-v2[full]"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
