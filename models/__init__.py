"""Model implementations."""

from .encode_process_decode import EncodeProcessDecode

# Models using convenient registry
# These require gnn_pde_v2.convenient to be importable
try:
    from .fno_model import FNO, TFNO
    from .gnn_model import GraphNet, MeshGraphNet
except ImportError:
    FNO = TFNO = GraphNet = MeshGraphNet = None

# Unified training model
try:
    from ..convenient.training import Model
except ImportError:
    Model = None

__all__ = [
    "EncodeProcessDecode",
    "Model",
    "FNO",
    "TFNO",
    "GraphNet",
    "MeshGraphNet",
]
