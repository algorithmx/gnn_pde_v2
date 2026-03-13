"""Model implementations."""

from .encode_process_decode import EncodeProcessDecode
from .gnn_model import GraphNet, MeshGraphNet
from .fno_model import FNO, TFNO

# NOTE: Training utilities (Model, LossFunction) have been moved to examples/training_utils.py
# to keep the framework lean. For training wrappers, use:
#     from gnn_pde_v2.examples.training_utils import Model, LossFunction

__all__ = [
    "EncodeProcessDecode",
    "FNO",
    "TFNO",
    "GraphNet",
    "MeshGraphNet",
]
