"""
Reusable components for building GNN-PDE models.

These are building blocks that can be composed directly.
No magic, no registry - just standard PyTorch modules.

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

from .encoders import MLPEncoder, MLPMeshEncoder, make_mlp_encoder
from .fourier_encoder import FourierFeatureEncoder
from .layers import (
    Residual,
    ResidualBlock,
    GatedResidual,
    PreNormResidual,
    ResidualSequence,
    SkipConnection,
    make_residual,
)
from .processors import GraphNetBlock, GraphNetProcessor
from .decoders import MLPDecoder, IndependentMLPDecoder
from .probe import ProbeDecoder, ProbeMessagePassingLayer
from .transformer import TransformerBlock, TransformerProcessor, MultiHeadAttention, PhysicsTokenAttention
from .fno import FNOProcessor, SpectralConv, FNOBlock

__all__ = [
    # Encoders
    "MLPEncoder",
    "MLPMeshEncoder",
    "make_mlp_encoder",
    "FourierFeatureEncoder",
    # Layers (residual connections)
    "Residual",
    "ResidualBlock",
    "GatedResidual",
    "PreNormResidual",
    "ResidualSequence",
    "SkipConnection",
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
    # Decoders
    "MLPDecoder",
    "IndependentMLPDecoder",
    "ProbeDecoder",
    "ProbeMessagePassingLayer",
]
