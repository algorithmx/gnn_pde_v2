# GNN-PDE v2 Architectural Review Report

**Review Date**: 2026-03-13
**Framework Version**: 2.1.0
**Reviewer**: Code Auditor

---

## Executive Summary

The gnn_pde_v2 framework demonstrates a thoughtful layered architecture with clear separation between core, components, and convenience layers. However, the codebase exhibits several significant issues that warrant attention:

3. **Major**: Inconsistent naming conventions between similar components
4. **Major**: Missing type hints in multiple modules
5. **Moderate**: Circular dependency risks in models module
6. **Moderate**: Inconsistent API patterns for similar operations

The framework is well-structured overall but would benefit from consolidation and standardization efforts.

---

## 1. API Consistency Issues

### 1.1 Critical: Two `Model` Classes with Different Purposes

**Location**:
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/training.py:43` - `Model` class (unified training wrapper)
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/__init__.py:15` - Import of same `Model`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/base.py:11` - `BaseModel` class (marker class)

**Issue**: The naming collision between `Model` (training wrapper) and `BaseModel` (architecture marker) creates confusion. The framework exports `Model` from both `models` and `convenient` modules:

```python
# models/__init__.py:15-17
try:
    from ..convenient.training import Model
except ImportError:
    Model = None
```

```python
# convenient/__init__.py:44-46
try:
    from .training import Model, LossFunction
except ImportError:
    Model = LossFunction = None
```

**Impact**: Users may confuse the unified training `Model` with the concept of a neural network model (architecture).

**Recommendation**: Rename the training wrapper to `TrainingModel` or `ModelTrainer` to disambiguate.

---

### 1.2 Critical: Duplicate `MeshGraphNet` Implementations

**Locations**:
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/gnn_model.py:71` - `MeshGraphNet` (convenience model)
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/examples/example_meshgraphnets.py:103` - `MeshGraphNet` (faithful implementation with namespace)

**Issue**: Two different `MeshGraphNet` implementations exist:

1. **models/gnn_model.py** - Simplified version using `MLPMeshEncoder` + `GraphNetProcessor`
2. **examples/example_meshgraphnets.py** - Faithful recreation with custom `MeshGraphNetsGNBlock`

```python
# models/gnn_model.py:71-122
class MeshGraphNet(AutoRegisterModel, name='meshgraphnet'):
    def __init__(self, ...):
        encoder = MLPMeshEncoder(...)
        processor = GraphNetProcessor(...)
        decoder = MLPDecoder(...)
```

```python
# examples/example_meshgraphnets.py:103
class MeshGraphNets(AutoRegisterModel, name='meshgraphnets', namespace='example'):
    def __init__(self, ...):
        # Uses custom MeshGraphNetsGNBlock with different MLP structure
```

**Impact**: Users get different behavior depending on which they import. The example version is more faithful to the original DeepMind paper but uses `namespace='example'`.

**Recommendation**: Consolidate into a single implementation. If keeping both, clearly document the difference and provide factory methods to select the variant.

---

### 1.3 Inconsistent Parameter Names

**Issue**: Similar components use different parameter names for the same concept:

| Component | Parameter Name | File:Line |
|-----------|---------------|-----------|
| `GraphNetProcessor` | `n_layers` | components/processors.py:181 |
| `TransformerProcessor` | `n_layers` | components/transformer.py:344 |
| `FNOProcessor` | `n_layers` | components/fno.py:268 |
| `MeshGraphNet` (models) | `n_message_passing` | models/gnn_model.py:84 |
| `MeshGraphNets` (example) | `message_passing_steps` | examples/example_meshgraphnets.py:112 |
| `GraphNet` | `n_layers` | models/gnn_model.py:28 |
| `GNNConfig` | `n_message_passing` | convenient/config.py:64 |

**Impact**: API inconsistency forces users to memorize different parameter names for similar operations.

**Recommendation**: Standardize on `n_layers` for all processor/multi-layer components. Keep `message_passing_steps` as an alias for backward compatibility in MeshGraphNet specifically.

---

### 1.4 Inconsistent Input Dimension Parameter Names

| Component | Input Dim Parameter | File:Line |
|-----------|-------------------|-----------|
| `MLPEncoder` | `node_in_dim`, `edge_in_dim` | components/encoders.py:24-27 |
| `MLPMeshEncoder` | `node_in_dim`, `edge_in_dim` | components/encoders.py:78-80 |
| `GraphNetBlock` | `node_dim`, `edge_dim` | components/processors.py:26-27 |
| `GraphNet` | `node_in_dim`, `edge_in_dim` | models/gnn_model.py:23-24 |
| `MeshGraphNet` | `node_in_dim`, `edge_in_dim` | models/gnn_model.py:79-80 |
| `FNO` | `in_channels`, `out_channels` | models/fno_model.py:21-22 |
| `FNOProcessor` | `in_channels`, `out_channels` | components/fno.py:264-265 |

**Impact**: Graph-based components use `*_in_dim` while FNO uses `in_channels`. Graph processing components use `node_dim` (latent) vs `node_in_dim` (input).

**Recommendation**:
- Use `in_dim`/`out_dim` for graph node/edge features
- Use `in_channels`/`out_channels` for grid-based data (FNO)
- Use `latent_dim` consistently for hidden representations

---

### 1.5 Multiple Ways to Do the Same Thing

**Issue**: Multiple APIs exist for creating similar components:

**Creating an MLP:**
```python
# Option 1: Core MLP directly
from gnn_pde_v2.core import MLP
mlp = MLP(10, 20, [128, 128])

# Option 2: make_mlp_encoder helper
from gnn_pde_v2.components import make_mlp_encoder
mlp = make_mlp_encoder(10, 20, hidden_dim=128, num_layers=3)
```

**Creating a MeshGraphNet:**
```python
# Option 1: Convenience model from models/
from gnn_pde_v2.models import MeshGraphNet
model = MeshGraphNet(node_in_dim=11, edge_in_dim=3, out_dim=3)

# Option 2: EPD with explicit components
from gnn_pde_v2.components import MLPMeshEncoder, GraphNetProcessor, MLPDecoder
from gnn_pde_v2.models import EncodeProcessDecode
encoder = MLPMeshEncoder(...)
processor = GraphNetProcessor(...)
decoder = MLPDecoder(...)
model = EncodeProcessDecode(encoder, processor, decoder)

# Option 3: Registry-based creation
from gnn_pde_v2.convenient import AutoRegisterModel
model = AutoRegisterModel.create('meshgraphnet', node_in_dim=11, ...)

# Option 4: Config-based creation
from gnn_pde_v2.convenient import ConfigBuilder, GNNConfig
config = GNNConfig(model_type='meshgraphnet', ...)
model = ConfigBuilder(config).build_model()
```

**Impact**: While flexibility is good, having 4+ ways to create the same model increases cognitive load.

**Recommendation**: Document clear usage patterns for different scenarios. The current structure is acceptable but needs better documentation of when to use which approach.

---

## 2. Module Organization Issues

### 2.1 Circular Dependency Risk in Models Module

**Location**: `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/__init__.py:7-17`

```python
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
```

**Issue**: Models in `models/` depend on `convenient.registry.AutoRegisterModel` and `convenient.training.Model`. This creates a dependency from core models to the convenience layer.

**Evidence**:
- `models/fno_model.py:8` imports from `convenient.registry`
- `models/gnn_model.py:8` imports from `convenient.registry`
- `models/__init__.py:15` imports from `convenient.training`

**Impact**:
- Cannot import models without the entire convenient layer
- Violates the intended layering (core → components → models → convenient)
- Makes optional dependencies harder to manage

**Recommendation**: Move registry-free model variants to `models/` and keep registry-dependent versions in `convenient/`. Alternatively, make `AutoRegisterModel` a core feature rather than a convenience.

---

### 2.2 Misplaced Components

**Issue**: Some components are in unexpected locations:

1. **`FNOProcessor` in components** (`components/fno.py:255`):
   - FNO operates on grids, not graphs
   - Inconsistent with the GNN-focused components
   - Has a `forward_graph` method that raises `NotImplementedError` (line 336-341)

2. **`_MeshEncoder` in models** (`models/gnn_model.py:146`):
   - This is a helper class used only by `ConfigBuilder`
   - Should be in `components/encoders.py` or be private to `convenient/builder.py`

3. **`Segment*` functions in convenient/aggregation** (`convenient/aggregation.py:97-100`):
   - These are just aliases to `scatter_*` functions
   - Provide no additional value over the core implementations
   - Pollutes the API with redundant names

```python
# convenient/aggregation.py:97-100
segment_sum = scatter_sum
segment_mean = scatter_mean
segment_max = scatter_max
segment_min = scatter_min
```

**Recommendation**:
- Move grid-based processors (FNO) to a separate `components/grid.py` or `components/spectral.py`
- Remove or consolidate `segment_*` aliases

---

### 2.3 Inconsistent Core vs Components Split

**Issue**: The boundary between `core/` and `components/` is unclear:

**In core:**
- `MLP` (core/mlp.py) - Basic building block
- `BaseModel` (core/base.py) - Marker class
- `GraphsTuple` (core/graph.py) - Data structure
- Functional utilities (core/functional.py)

**In components:**
- Encoders (use MLP internally)
- Decoders (use MLP internally)
- Processors (use MLP internally)
- Layers (wrappers around modules)

**Problem**: `MLP` is in core, but `make_mlp_encoder` is in components. The split suggests that "core" = minimal primitives, "components" = composable blocks, but the boundary is blurry.

**Recommendation**: Document the layering explicitly:
- **Core**: Data structures, functional ops, minimal primitives (MLP)
- **Components**: Reusable architecture blocks (Encoder, Decoder, Processor)
- **Models**: Complete architectures (EPD, GraphNet, FNO)
- **Convenient**: Optional sugar (registry, config, training)

---

## 3. Code Duplication

### 3.1 Critical: Multiple MLP Patterns

**Issue**: While there's a single `MLP` class in `core/mlp.py`, the framework has multiple patterns for creating MLPs:

1. **Direct MLP instantiation** (recommended):
```python
MLP(in_dim, out_dim, hidden_dims=[128, 128])
```

2. **make_mlp_encoder helper** (`components/encoders.py:137`):
```python
def make_mlp_encoder(in_dim, out_dim, hidden_dim=128, num_layers=2, **mlp_kwargs):
    if num_layers == 1:
        hidden_dims = []
    else:
        hidden_dims = [hidden_dim] * (num_layers - 1)
    return MLP(in_dim, out_dim, hidden_dims, **mlp_kwargs)
```

3. **Encoder classes that just wrap MLP**:
```python
# components/encoders.py:18-69
class MLPEncoder(nn.Module):
    def __init__(self, node_in_dim, node_out_dim, ...):
        self.node_encoder = MLP(node_in_dim, node_out_dim, ...)
        # Optionally also self.edge_encoder = MLP(...)
```

**Impact**: The `make_mlp_encoder` helper provides marginal value over direct `MLP` instantiation but adds cognitive overhead.

**Recommendation**: Either remove `make_mlp_encoder` or make it significantly more useful (e.g., support multiple initialization schemes, activation schedules, etc.).

---

### 3.2 Duplicate Aggregation Functions

**Issue**: Aggregation functions are duplicated between `core/functional.py` and `convenient/aggregation.py`:

**core/functional.py:**
- `scatter_sum`
- `scatter_mean`
- `scatter_max`
- `aggregate_edges`
- `broadcast_nodes_to_edges`

**convenient/aggregation.py:**
- Re-exports `scatter_sum`, `scatter_mean`, `scatter_max`
- Adds `scatter_min`, `scatter_softmax`
- Adds aliases `segment_sum`, `segment_mean`, `segment_max`, `segment_min`

**Impact**: Users must choose between two modules for similar functionality. The `convenient/aggregation.py` adds value (`scatter_min`, `scatter_softmax`) but also creates confusion.

**Recommendation**:
1. Move all scatter operations to `core/functional.py`
2. Keep `convenient/aggregation.py` only for advanced operations not in core
3. Remove `segment_*` aliases (or keep only one set of names)

---

### 3.3 Duplicate Residual Patterns

**Issue**: Multiple residual wrapper classes with overlapping functionality:

**components/layers.py:**
- `Residual` - Simple wrapper (line 13)
- `ResidualBlock` - Configurable residual type (line 55)
- `GatedResidual` - Learnable gate (line 123)
- `PreNormResidual` - Transformer-style (line 194)
- `ResidualSequence` - Sequence of residual blocks (line 245)
- `SkipConnection` - With optional projection (line 289)
- `make_residual` - Factory function (line 345)

**Problem**:
- `Residual` and `ResidualBlock` have overlapping functionality
- `PreNormResidual` is just `Residual` with norm always set to LayerNorm
- The distinction between these classes is not clearly documented

**Evidence**:
```python
# components/layers.py:13-52 (Residual)
class Residual(nn.Module):
    def __init__(self, module: nn.Module, norm: Optional[nn.Module] = None):
        # output = x + module(norm(x)) if norm else x + module(x)

# components/layers.py:55-121 (ResidualBlock)
class ResidualBlock(nn.Module):
    def __init__(self, module, residual_type='add', scale=None, learnable_scale=False):
        # More complex with scaled/gated options
```

**Recommendation**: Consolidate to fewer, more clearly differentiated classes:
- Keep `Residual` as the simple, primary interface
- Keep `GatedResidual` for gated variants
- Keep `make_residual` factory for convenience
- Consider deprecating `ResidualBlock`, `PreNormResidual`, `SkipConnection` in favor of explicit composition

---

### 3.4 Duplicate Initialization Logic

**Issue**: Weight initialization is handled in multiple places:

1. **MLP built-in** (`core/mlp.py:86-87, 158-165`):
```python
def __init__(self, ..., weight_init=init.xavier_uniform_, bias_init=...):
    # ...
    self.apply(self._init_weights)
```

2. **String-based initializer** (`convenient/initializers.py:15-76`):
```python
def get_initializer(name: Union[str, Callable]) -> Callable:
    # Maps 'glorot_uniform' -> init.xavier_uniform_
```

3. **Module initializer** (`convenient/initializers.py:79-101`):
```python
def initialize_module(module, weight_init='glorot_uniform', bias_init='constant_0'):
    # Applies initialization to all parameters
```

**Problem**: The MLP class uses callable init functions directly, while the convenience layer uses string-based initialization. Users must learn two systems.

**Recommendation**: Standardize on one approach. The callable approach (used by MLP) is more Pythonic and type-safe.

---

## 4. Type Safety & Documentation

### 4.1 Missing Type Hints

**Issue**: Inconsistent type annotation coverage across modules.

**Modules with complete type hints:**
- `core/mlp.py` - Complete annotations
- `core/graph.py` - Complete annotations
- `components/transformer.py` - Complete annotations
- `convenient/config.py` - Complete annotations

**Modules with partial type hints:**
- `components/processors.py` - Missing return types on some methods
- `components/fno.py` - Missing return types on `forward_graph`
- `utils/spatial_utils.py` - Missing tensor shape annotations

**Modules with minimal/no type hints:**
- `utils/graph_utils.py` - No type hints at all

**Example from utils/graph_utils.py:**
```python
# Line 17-42: No type hints
def knn_graph(positions, k, batch=None):
    if batch is None:
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    edge_index = torch_knn_graph(positions, k, batch, loop=False)
    # ...
```

**Should be:**
```python
def knn_graph(
    positions: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Recommendation**: Add comprehensive type hints to all public APIs, especially utility modules.

---

### 4.2 Inconsistent Return Type Documentation

**Issue**: Return types are inconsistently documented in docstrings.

**Good example** (core/functional.py:126-144):
```python
def broadcast_nodes_to_edges(
    node_features: Tensor,
    senders: Tensor,
    receivers: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    ...
    Returns:
        (sender_features, receiver_features) both [E, feat_dim]
    """
```

**Poor example** (components/processors.py:75-127):
```python
def forward(self, graph: GraphsTuple) -> GraphsTuple:
    """Single message passing step."""
    # No return type documentation in docstring
```

**Recommendation**: Standardize docstring format to always include:
- Args section with types
- Returns section with types and shapes
- Raises section if applicable

---

### 4.3 Missing Docstrings

**Issue**: Some public functions lack docstrings.

**Example** (`convenient/builder.py:200-209`):
```python
def _get_optimizer_class(self, name: str):  # No return type annotation
    """Get optimizer class by name."""  # Minimal docstring
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    return optimizers[name]
```

**Should include:**
```python
def _get_optimizer_class(self, name: str) -> Type[Optimizer]:
    """
    Get optimizer class by name.

    Args:
        name: Optimizer name ('adam', 'adamw', 'sgd')

    Returns:
        Optimizer class (not instantiated)

    Raises:
        ValueError: If optimizer name is unknown
    """
```

---

### 4.4 Inconsistent Docstring Formats

**Issue**: Multiple docstring formats are used:

1. **Google style** (most common):
```python
# core/mlp.py:25-69
"""
Flexible feedforward network...

Args:
    in_dim: Input dimension
    out_dim: Output dimension
    ...

Example:
    >>> mlp = MLP(64, 64, hidden_dims=[128, 128])
"""
```

2. **Minimal docstrings**:
```python
# components/processors.py:169-175
"""
Multi-layer GraphNet processor.

Stacks multiple GraphNetBlocks with optional residual connections.
"""
# No Args, no Example
```

3. **Detailed with references**:
```python
# examples/example_meshgraphnets.py:1-22
"""
Example: MeshGraphNets (DeepMind)

Original Work Reference:
------------------------
Pfaff, T., et al. (2021).
...
"""
```

**Recommendation**: Standardize on Google-style docstrings for all modules with:
- One-line summary
- Extended description (if needed)
- Args section
- Returns section
- Raises section
- Example section (for frequently used APIs)

---

## 5. Abstraction Leaks

### 5.1 FNOProcessor's Graph Method

**Location**: `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/fno.py:330-341`

```python
def forward_graph(self, graph: GraphsTuple) -> GraphsTuple:
    """
    Process graph by treating nodes as grid points.

    Note: This requires the graph to be on a regular grid.
    """
    # This is a placeholder - actual implementation would need
    # to map graph nodes to/from a regular grid
    raise NotImplementedError(
        "FNOProcessor.forward_graph requires grid mapping. "
        "Use forward() with regular grid inputs instead."
    ")
```

**Issue**: This method exists only to satisfy some interface expectation but is not implemented. It's a leaky abstraction that confuses the API.

**Impact**: Users might expect all "processors" to work with graphs, but FNO doesn't.

**Recommendation**: Either:
1. Remove the method entirely (breaking change but cleaner)
2. Implement it properly with grid mapping
3. Create a separate `GridProcessor` protocol for non-graph processors

---

### 5.2 Exposed Internal Attributes

**Issue**: Some classes expose internal implementation details:

**Example 1** - `MLP.net` attribute (`core/mlp.py:154`):
```python
class MLP(nn.Module):
    def __init__(self, ...):
        # ...
        self.net = nn.Sequential(*layers)  # Exposed internal
```

Users might access `model.net[0]` directly instead of using the public API.

**Example 2** - `AutoRegisterModel._registry` (`convenient/registry.py:42`):
```python
class AutoRegisterModel(BaseModel):
    _registry: ClassVar[Dict[str, Type['AutoRegisterModel']]] = {}
```

While prefixed with `_`, this is a class variable that users might modify directly, leading to unexpected behavior.

**Example 3** - Tests access internal state:
```python
# tests/test_convenient.py:28-37
def test_auto_registration(self):
    """Test models are auto-registered."""
    # Clear registry first
    AutoRegisterModel._registry.clear()  # Direct access to private member
```

**Recommendation**:
- Make `MLP.net` private (`_net`) and expose layers via a property if needed
- Provide official API for registry management instead of direct access

---

### 5.3 Query Position Parameter Inconsistency

**Issue**: Decoder interfaces are inconsistent about `query_positions`:

**MLPDecoder** (`components/decoders.py:37-55`):
```python
def forward(
    self,
    graph: GraphsTuple,
    query_positions: Optional[torch.Tensor] = None  # Accepts but ignores
) -> torch.Tensor:
    """
    Args:
        query_positions: Ignored for this decoder (outputs at nodes)
    """
```

**ProbeDecoder** (`components/probe.py:73-106`):
```python
def forward(
    self,
    graph: GraphsTuple,
    query_positions: torch.Tensor,  # Required
) -> torch.Tensor:
```

**Impact**: The interface isn't consistent - some decoders require query positions, some ignore them, some don't accept them at all.

**Recommendation**: Define a clear `Decoder` protocol:
```python
class Decoder(Protocol):
    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
```

---

## 6. Testing Gaps

### 6.1 Missing Test Coverage

**Issue**: Several components lack comprehensive test coverage:

**Modules with tests:**
- `core/` - test_core.py (comprehensive)
- `components/` - test_components.py (good coverage)
- `convenient/` - test_convenient.py (basic coverage)
- Conditioning - test_conditioning.py (minimal)

**Modules without tests:**
- `models/fno_model.py` - No direct tests
- `models/gnn_model.py` - No direct tests
- `components/fno.py` - Not tested in test_components.py
- `components/transformer.py` - Not tested
- `components/fourier_encoder.py` - Not tested
- `utils/` - No tests at all

**Recommendation**: Add test files:
- `tests/test_models.py` - For FNO, GraphNet, MeshGraphNet models
- `tests/test_transformer.py` - For transformer components
- `tests/test_fno.py` - For FNO components
- `tests/test_utils.py` - For utility functions

---

### 6.2 Missing Edge Case Tests

**Issue**: Existing tests don't cover edge cases:

**Example from test_components.py:**
```python
# Line 150-175: TestGraphNetBlock
def test_forward(self, device):
    """Test basic forward pass."""
    block = GraphNetBlock(node_dim=16, edge_dim=8, global_dim=None)
    # ... only tests happy path

def test_with_globals(self, device):
    """Test with global features."""
    # ... only tests with globals
```

**Missing tests:**
- What happens with empty graphs?
- What happens with disconnected nodes?
- What happens with mismatched dimensions?
- What happens with batch_size=0?
- GPU/CPU consistency?

**Recommendation**: Add edge case tests for:
- Empty graphs (0 nodes, 0 edges)
- Single node graphs
- Disconnected graphs
- Dimension mismatches
- Device placement edge cases

---

### 6.3 No Integration Tests

**Issue**: The test suite lacks integration tests that verify the complete EPD pipeline works end-to-end.

**Current test structure:**
- Unit tests for individual components
- No tests that verify encoder + processor + decoder work together
- No tests for training loops
- No tests for save/load

**Recommendation**: Add integration tests:
```python
# tests/test_integration.py
def test_encode_process_decode_training():
    """Test complete EPD training loop."""
    encoder = MLPEncoder(...)
    processor = GraphNetProcessor(...)
    decoder = MLPDecoder(...)
    model = EncodeProcessDecode(encoder, processor, decoder)

    # Test forward pass
    # Test backward pass
    # Test gradient flow
    # Test save/load
```

---

## 7. Dependency Issues

### 7.1 Hard Dependencies in Models

**Issue**: Models in `models/` have hard dependencies on `convenient` layer:

```python
# models/fno_model.py:8
from ..convenient.registry import AutoRegisterModel

# models/gnn_model.py:8
from ..convenient.registry import AutoRegisterModel
```

**Impact**: Cannot use models without pulling in the entire convenient layer, even if you just want the architecture.

**Recommendation**: Create two versions:
1. `models/fno_core.py` - Plain nn.Module, no registry
2. `convenient/fno.py` - Wrapped with AutoRegisterModel

Or move AutoRegisterModel to core (since it's just a metaclass).

---

### 7.2 Optional Dependency Handling

**Issue**: The codebase handles optional dependencies with try/except but doesn't provide clear error messages:

**Good pattern** (`models/__init__.py:7-17`):
```python
try:
    from .fno_model import FNO, TFNO
    from .gnn_model import GraphNet, MeshGraphNet
except ImportError:
    FNO = TFNO = GraphNet = MeshGraphNet = None
```

**Problem**: When imports fail, users get `None` with no explanation of what's missing.

**Recommendation**: Provide lazy imports with helpful error messages:
```python
def __getattr__(name: str):
    if name in ('FNO', 'TFNO', 'GraphNet', 'MeshGraphNet'):
        raise ImportError(
            f"'{name}' requires the 'convenient' module. "
            f"Install with: pip install gnn-pde-v2[convenient]"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

### 7.3 Missing Dependency Declarations

**Issue**: Some imports assume optional dependencies without declaring them:

**utils/graph_utils.py:27, 63, 148**:
```python
from torch_cluster import knn_graph as torch_knn_graph
from torch_cluster import radius_graph as torch_radius_graph
```

These are imported without try/except, meaning importing `utils/graph_utils.py` requires `torch_cluster`.

**Recommendation**: Either:
1. Make these imports optional with fallbacks
2. Document `torch_cluster` as a required dependency
3. Move these utilities to an optional submodule

---

## 8. Design Pattern Issues

### 8.1 Over-Engineering: Multiple Residual Variants

**Issue**: As noted in Section 3.3, there are 6 residual wrapper classes plus a factory function. This is over-engineered for the actual use cases.

**Evidence**: Most examples only use `Residual`:
```python
# examples/core/meshgraphnets_core.py:54-64
# No Residual wrapper used - manual residual connection
for block in self.processor_blocks:
    processed = block(latent)
    latent = replace(latent, nodes=latent.nodes + processed.nodes, ...)
```

**Recommendation**: Reduce to 3 variants:
1. `Residual` - Simple wrapper (covers 80% of cases)
2. `GatedResidual` - When gating is needed
3. `make_residual` - Factory for convenience

Deprecate `ResidualBlock`, `PreNormResidual`, `SkipConnection`, `ResidualSequence` in favor of explicit composition.

---

### 8.2 Under-Engineering: Processor Protocol Missing

**Issue**: There's no formal protocol for what a "Processor" should implement:

```python
# components/processors.py:169
class GraphNetProcessor(nn.Module):
    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...

# components/transformer.py:333
class TransformerProcessor(nn.Module):
    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...

# components/fno.py:255
class FNOProcessor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # Different signature!
    def forward_graph(self, graph: GraphsTuple) -> GraphsTuple: ...  # Raises NotImplementedError
```

**Impact**: Cannot write code that works with any "Processor" because they have different interfaces.

**Recommendation**: Define a Protocol:
```python
from typing import Protocol

class GraphProcessor(Protocol):
    """Protocol for graph processors."""
    def forward(self, graph: GraphsTuple) -> GraphsTuple: ...

class GridProcessor(Protocol):
    """Protocol for grid-based processors."""
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

---

### 8.3 SOLID Principle Violations

**Single Responsibility Principle (SRP) Violations:**

1. **`MLPEncoder`** (`components/encoders.py:18-69`):
   - Encodes both nodes AND edges
   - Should be two separate concerns

2. **`MLPMeshEncoder`** (`components/encoders.py:71-133`):
   - Encodes nodes, edges, AND globals
   - Three responsibilities in one class

3. **`Model`** (`convenient/training.py:43-230`):
   - Handles architecture wrapping
   - Handles loss computation
   - Handles optimizer management
   - Handles checkpointing
   - Handles history tracking
   - At least 5 responsibilities

**Open/Closed Principle (OCP) Violations:**

1. **`ConfigBuilder._build_fno`** (`convenient/builder.py:106-139`):
   - Must modify this method to add new FNO variants
   - Should use registry pattern

2. **`ConfigBuilder._build_gnn`** (`convenient/builder.py:141-197`):
   - Hardcoded model types
   - Not extensible without modification

**Liskov Substitution Principle (LSP) Violations:**

1. **`FNOProcessor.forward_graph`** raises `NotImplementedError`
   - Subclasses/users can't rely on the base class contract
   - See Section 5.1

**Interface Segregation Principle (ISP) Violations:**

1. **Decoders** - Some require `query_positions`, some ignore it, some don't accept it
   - See Section 5.3

**Dependency Inversion Principle (DIP) Violations:**

1. **Models depend on concrete registry** (`models/fno_model.py:8`)
   - Should depend on abstraction (e.g., Protocol)

---

### 8.4 Inconsistent Factory Pattern Usage

**Issue**: Factory functions are used inconsistently:

**Has factory:**
- `make_mlp_encoder` (components/encoders.py:137)
- `make_residual` (components/layers.py:345)

**No factory:**
- No `make_encoder` for encoder creation
- No `make_processor` for processor creation
- No `make_decoder` for decoder creation

**Impact**: Inconsistent API forces users to memorize which components have factories and which don't.

**Recommendation**: Either add factories for all component categories or remove existing factories for consistency.

---

## 9. Specific Issues by File

### 9.1 core/base.py

**Issue**: `BaseModel` is essentially a marker class with no functionality.

```python
# core/base.py:11-30
class BaseModel(nn.Module):
    """
    Base class for all models in the GNN-PDE framework.

    This is intentionally minimal - just a marker class...
    """
    pass
```

**Impact**: Provides no value over using `nn.Module` directly.

**Recommendation**: Either add useful functionality (e.g., `save`, `load`, `num_parameters`) or remove it and use `nn.Module` directly.

---

### 9.3 components/fno.py

**Issue**: Multiple issues in this file:

1. **Complex multiplication functions** (lines 15-30):
   - Could be replaced with `torch.einsum` directly
   - No need for separate 1D/2D/3D functions

2. **Hardcoded n_dim check** (line 287-288):
```python
if n_dim not in {1, 2, 3}:
    raise ValueError(f"n_dim must be 1, 2, or 3, got {n_dim}")
```
   - Should be validated earlier or use type system

3. **Unimplemented forward_graph** (see Section 5.1)

---

### 9.5 models/gnn_model.py

**Issue**: Private `_MeshEncoder` class exposed for `ConfigBuilder`:

```python
# Line 146
class _MeshEncoder(nn.Module):  # Private class (underscore prefix)
    ...
```

But it's imported by `ConfigBuilder`:
```python
# convenient/builder.py:146
from ..models.gnn_model import GraphNet, MeshGraphNet, _MeshEncoder
```

**Impact**: Private class is part of the public API, violating encapsulation.

**Recommendation**: Move `_MeshEncoder` to `components/encoders.py` as `MeshEncoder` (public).

---

## 10. Recommendations Summary

### Critical (Fix Immediately)

2. **Consolidate MeshGraphNet implementations** - Choose one authoritative implementation

3. **Fix models/ dependency on convenient/** - Either move `AutoRegisterModel` to core or make models independent

### Major (Fix Soon)

4. **Standardize parameter names** - Use `n_layers` consistently, document exceptions

5. **Add type hints** to all public APIs, especially `utils/` module

6. **Create Processor protocols** to enable polymorphic processor usage

7. **Consolidate residual wrappers** - Reduce from 6 to 3 variants

### Moderate (Improve Over Time)

8. **Standardize docstring format** - Adopt Google style everywhere

9. **Add missing tests** - For models, transformer, FNO, utils

10. **Add integration tests** - Test complete EPD pipelines

11. **Improve optional dependency handling** - Better error messages, lazy imports

### Minor (Consider for Next Version)

12. **Remove redundant code** - `make_mlp_encoder`, `segment_*` aliases

13. **Add factory functions** - For encoder/decoder/processor creation

14. **Document layering** - Clear guidance on core vs components vs models vs convenient

---

## Appendix: File Reference

### Core Module Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/__init__.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/base.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/functional.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/graph.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/core/mlp.py`

### Component Module Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/__init__.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/decoders.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/encoders.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/fno.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/fourier_encoder.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/layers.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/probe.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/processors.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/components/transformer.py`

### Convenience Module Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/__init__.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/aggregation.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/builder.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/config.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/initializers.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/registry.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/convenient/training.py`

### Model Module Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/__init__.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/encode_process_decode.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/fno_model.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/gnn_model.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/models/unified_model.py`

### Test Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/tests/conftest.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/tests/test_components.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/tests/test_conditioning.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/tests/test_convenient.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/tests/test_core.py`

### Utility Files
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/utils/graph_utils.py`
- `/home/dabajabaza/Nutstore/Work/Project/gnn_pde_v2/utils/spatial_utils.py`

---

**End of Report**
