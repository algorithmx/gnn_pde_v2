"""
Components for building GNN and neural operator architectures.

These are building blocks that can be composed directly.
No magic, no registry - just standard PyTorch modules.

Graph-based processors (work with GraphsTuple):
    - GraphNetBlock, GraphNetProcessor
    - TransformerBlock, TransformerProcessor (with optional relative position encoding)
    - MultiHeadAttention, PhysicsTokenAttention, RelativePositionEncoding

Spectral/Grid-based processors (work with regular grid tensors):
    - FNOProcessor, SpectralConv, SeparableSpectralConv, SpectralConvBase
    - FNOBlock, AFNOBlock

Note: Spectral processors are NOT compatible with graph data structures.
For graph data, use GraphNetProcessor or TransformerProcessor instead.

Note: MLP is now in core. Import with:
    from gnn_pde_v2.core import MLP

Factory Functions:
    Factory functions are provided ONLY for runtime polymorphism scenarios
    where the component type is determined at runtime (e.g., from config).
    For explicit construction, prefer direct class instantiation.
    
    Available factories:
    - make_residual: Select residual connection type ('add', 'gated', 'scaled', etc.)
    - make_spectral_conv: Select spectral conv implementation (standard vs separable)
    
    Prefer direct classes for:
    - MeshEncoder, MLPDecoder, GraphNetProcessor, FNOProcessor

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
from .layers import (
    Residual,
    GatedResidual,
    make_residual,
)
from .processors import (
    MessagePassingBlock,
    GraphNetBlock, GraphNetProcessor,
    EdgeConditionedConvBlock,
    GlobalGraphNetBlock, GlobalGraphNetProcessor,
)
from .decoders import MLPDecoder, IndependentMLPDecoder
from .probe import ProbeDecoder, ProbeMessagePassingLayer
from .transformer import (
    TransformerBlock, TransformerProcessor,
)
from .attention import (
    MultiHeadAttention, PhysicsTokenAttention,
    QKNormMultiHeadAttention, SparseGraphAttention, RelativePositionEncoding,
)
from .conditioning import (
    ZeroConditioning,
    AdaLNConditioning, AdaLNConditioningNoGate,
    DualAdaLNConditioning, DualAdaLNConditioningNoGate,
    FiLMConditioning,
    apply_modulation,
)
from .temperature import (
    TemperatureBase,
    FixedTemperature,
    LearnableScalarTemperature,
    PerHeadTemperature,
    AdaptiveTemperature,
    AnnealedTemperature,
    create_temperature_module,
)
from .spectral import FNOProcessor, SpectralConv, SeparableSpectralConv, SpectralConvBase, make_spectral_conv, FNOBlock, AFNOBlock

# Structural protocols — re-exported here for convenience since they describe
# component contracts. Conditioning types (Modulation, ConditioningProtocol)
# are NOT re-exported; import them directly from gnn_pde_v2.core.protocols.
from ..core.protocols import (
    GraphEncoder,
    GraphProcessor,
    NodeDecoder,
    QueryDecoder,
    Decoder,
    GraphModel,
    PositionEncoder,
    GridProcessor,
    GridModel,
)

__all__ = [
    # Encoders
    "FourierFeatureEncoder",
    # Layers (residual connections)
    "Residual",
    "GatedResidual",
    "make_residual",
    # Processors
    "MessagePassingBlock",
    "GraphNetBlock",
    "EdgeConditionedConvBlock",
    "GlobalGraphNetBlock",
    "GraphNetProcessor",
    "GlobalGraphNetProcessor",
    "TransformerBlock",
    "TransformerProcessor",
    "MultiHeadAttention",
    "PhysicsTokenAttention",
    "QKNormMultiHeadAttention",
    "SparseGraphAttention",
    "RelativePositionEncoding",
    # Temperature mechanisms
    "TemperatureBase",
    "FixedTemperature",
    "LearnableScalarTemperature",
    "PerHeadTemperature",
    "AdaptiveTemperature",
    "AnnealedTemperature",
    "create_temperature_module",
    "FNOProcessor",
    "SpectralConv",
    "SeparableSpectralConv",
    "SpectralConvBase",
    "make_spectral_conv",
    "FNOBlock",
    "AFNOBlock",
    # Decoders
    "MLPDecoder",
    "IndependentMLPDecoder",
    "ProbeDecoder",
    "ProbeMessagePassingLayer",
    # Conditioning (Modulation & ConditioningProtocol live in gnn_pde_v2.core.protocols)
    "ZeroConditioning",
    "AdaLNConditioning",
    "AdaLNConditioningNoGate",
    "DualAdaLNConditioningNoGate",
    "DualAdaLNConditioning",
    "FiLMConditioning",
    "apply_modulation",
    # Structural protocols
    "GraphEncoder",
    "GraphProcessor",
    "NodeDecoder",
    "QueryDecoder",
    "Decoder",
    "GraphModel",
    "PositionEncoder",
    "GridProcessor",
    "GridModel",
]
