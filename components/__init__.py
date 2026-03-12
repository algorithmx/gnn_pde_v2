"""
Reusable components for building GNN-PDE models.

These are building blocks that can be composed directly.
No magic, no registry - just standard PyTorch modules.

Example:
    from gnn_pde_v2.components import MLP, Residual
    from gnn_pde_v2.components.processors import GraphNetBlock
    
    class MyModel(nn.Module):
        def __init__(self):
            self.encoder = MLP(10, 128, [128, 128])
            self.processor = Residual(GraphNetBlock(128, 128))
"""

from .encoders import MLP, MLPEncoder, MLPMeshEncoder, make_mlp_encoder
from .fourier_encoder import FourierFeatureEncoder
from .layers import Residual
from .processors import GraphNetBlock, GraphNetProcessor
from .decoders import MLPDecoder, IndependentMLPDecoder
from .probe import ProbeDecoder, ProbeMessagePassingLayer
from .transformer import TransformerBlock, TransformerProcessor, MultiHeadAttention, PhysicsTokenAttention
from .fno import FNOProcessor, SpectralConv, FNOBlock

__all__ = [
    # Encoders
    "MLP",
    "MLPEncoder",
    "MLPMeshEncoder",
    "make_mlp_encoder",
    "FourierFeatureEncoder",
    # Layers
    "Residual",
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
