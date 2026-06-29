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
    GraphBlockBase,
    MessagePassingBase,
    GraphNetBlock, GraphNetProcessor,
    EdgeConditionedConvBlock,
    EdgeConvBlock,
    GENBlock,
    GlobalGraphNetBlock, GlobalGraphNetProcessor,
)
from .edge_processors import (
    FullEdgeMessageProcessor,
    VectorEdgeMessageProcessor,
    ScalarEdgeMessageProcessor,
    LowRankEdgeMessageProcessor,
)
from .edge_assemblers import (
    EdgeFeatureAssembler,
    NodeDifferenceAssembler,
    ConcatAssembler,
    DifferenceOnlyAssembler,
    ConcatWithEdgesAssembler,
)
from .node_updaters import (
    ConcatMLPNodeUpdater,
    RootWeightNodeUpdater,
    PassThroughNodeUpdater,
    ResidualMLPNodeUpdater,
    build_concat_mlp_node_updater,
    build_root_weight_node_updater,
    build_pass_through_node_updater,
    build_residual_mlp_node_updater,
    NodeUpdaterFactory,
    concat_mlp_factory,
    root_weight_factory,
    pass_through_factory,
    residual_mlp_factory,
    default_node_updater_factory,
)
from .gcn import GCNBlock, GCNBlockWithEdgeFeatures
from .decoders import MLPDecoder, IndependentMLPDecoder
from .probe import ProbeDecoder, WindFarmGNO, ProbeGraphBuilder
from .rbf import LearnableRBFEncoder, GaussianRBFEncoder
from .transformer import (
    TransformerBlock, TransformerProcessor,
    PhysicsTokenConfig, RelativePositionConfig,
)
from .attention import (
    MultiHeadAttention, PhysicsTokenAttention, PhysicsTokenAttentionV3,
    TiledSliceOperation,
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
from .spectral import (
    FNOProcessor, SpectralConv, SeparableSpectralConv, SpectralConvBase,
    make_spectral_conv, SpectralBlockBase, FNOBlock, FNOMLPBlock, AFNOBlock,
)

# Structural protocols — re-exported here for convenience since they describe
# component contracts. Conditioning types (Modulation, ConditioningProtocol)
# are NOT re-exported; import them directly from gnn_pde_v2.core.protocols.
from ..core.protocols import (
    GraphEncoder,
    GraphProcessor,
    EdgeMessageProcessor,
    NodeDecoder,
    QueryDecoder,
    GraphModel,
)

__all__ = [
    # Encoders
    "FourierFeatureEncoder",
    # Node updaters
    "ConcatMLPNodeUpdater",
    "RootWeightNodeUpdater",
    "PassThroughNodeUpdater",
    "ResidualMLPNodeUpdater",
    "build_concat_mlp_node_updater",
    "build_root_weight_node_updater",
    "build_pass_through_node_updater",
    "build_residual_mlp_node_updater",
    # Node updater factories
    "NodeUpdaterFactory",
    "concat_mlp_factory",
    "root_weight_factory",
    "pass_through_factory",
    "residual_mlp_factory",
    "default_node_updater_factory",
    # Layers (residual connections)
    "Residual",
    "GatedResidual",
    "make_residual",
    # Processors
    "GraphBlockBase",
    "MessagePassingBase",
    "GraphNetBlock",
    "EdgeConditionedConvBlock",
    "FullEdgeMessageProcessor",
    "VectorEdgeMessageProcessor",
    "ScalarEdgeMessageProcessor",
    "LowRankEdgeMessageProcessor",
    # Edge assemblers
    "EdgeFeatureAssembler",
    "NodeDifferenceAssembler",
    "ConcatAssembler",
    "DifferenceOnlyAssembler",
    "ConcatWithEdgesAssembler",
    "EdgeConvBlock",
    "GENBlock",
    "GlobalGraphNetBlock",
    "GraphNetProcessor",
    "GlobalGraphNetProcessor",
    "GCNBlock",
    "GCNBlockWithEdgeFeatures",
    "TransformerBlock",
    "TransformerProcessor",
    "PhysicsTokenConfig",
    "RelativePositionConfig",
    "MultiHeadAttention",
    "PhysicsTokenAttention",
    "PhysicsTokenAttentionV3",
    "TiledSliceOperation",
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
    "SpectralBlockBase",
    "FNOBlock",
    "FNOMLPBlock",
    "AFNOBlock",
    # Decoders
    "MLPDecoder",
    "IndependentMLPDecoder",
    "ProbeDecoder",
    "WindFarmGNO",
    "ProbeGraphBuilder",
    # RBF Encoders
    "LearnableRBFEncoder",
    "GaussianRBFEncoder",
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
    "EdgeMessageProcessor",
    "NodeDecoder",
    "QueryDecoder",
    "GraphModel",
]
