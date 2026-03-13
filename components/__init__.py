"""
Components for building GNN and neural operator architectures.

These are building blocks that can be composed directly.
No magic, no registry - just standard PyTorch modules.

Graph-based processors (work with GraphsTuple):
    - GraphNetBlock, GraphNetProcessor
    - TransformerBlock, TransformerProcessor

Spectral/Grid-based processors (work with regular grid tensors):
    - FNOProcessor, SpectralConv, FNOBlock, AFNOBlock

Note: Spectral processors are NOT compatible with graph data structures.
For graph data, use GraphNetProcessor or TransformerProcessor instead.

Note: MLP is now in core. Import with:
    from gnn_pde_v2.core import MLP

Example:
    from gnn_pde_v2.core import MLP
    from gnn_pde_v2.components import Residual
    from gnn_pde_v2.components.processors import GraphNetBlock

    class MyModel(nn.Module):
        def __init__(self):
            self.encoder = MLP(10, 128, [128, 128])
            self.processor = Residual(GraphNetBlock(128, 128))
"""

from .fourier_encoder import FourierFeatureEncoder
from .encoders import MeshEncoder
from .layers import (
    Residual,
    GatedResidual,
    make_residual,
)
from .processors import GraphNetBlock, GraphNetProcessor
from .decoders import MLPDecoder, IndependentMLPDecoder
from .probe import ProbeDecoder, ProbeMessagePassingLayer
from .transformer import (
    TransformerBlock, TransformerProcessor, MultiHeadAttention, PhysicsTokenAttention,
    # Conditioning (defined in core/protocols, re-exported through transformer)
    Modulation, ConditioningProtocol,
    ZeroConditioning, AdaLNConditioning, DualAdaLNConditioning, FiLMConditioning,
)
from .spectral import FNOProcessor, SpectralConv, FNOBlock, AFNOBlock

# Structural protocols — also available from gnn_pde_v2.core
from ..core.protocols import (
    GraphEncoder,
    GraphProcessor,
    Decoder,
    GraphModel,
    PositionEncoder,
    GridProcessor,
    GridModel,
)

__all__ = [
    # Encoders
    "FourierFeatureEncoder",
    "MeshEncoder",
    # Layers (residual connections)
    "Residual",
    "GatedResidual",
    "make_residual",
    # Processors
    "GraphNetBlock",
    "GraphNetProcessor",
    "TransformerBlock",
    "TransformerProcessor",
    "MultiHeadAttention",
    "PhysicsTokenAttention",
    "FNOProcessor",
    "SpectralConv",
    "FNOBlock",
    "AFNOBlock",
    # Decoders
    "MLPDecoder",
    "IndependentMLPDecoder",
    "ProbeDecoder",
    "ProbeMessagePassingLayer",
    # Conditioning
    "Modulation",
    "ConditioningProtocol",
    "ZeroConditioning",
    "AdaLNConditioning",
    "DualAdaLNConditioning",
    "FiLMConditioning",
    # Structural protocols
    "GraphEncoder",
    "GraphProcessor",
    "Decoder",
    "GraphModel",
    "PositionEncoder",
    "GridProcessor",
    "GridModel",
]
