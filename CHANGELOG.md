# Changelog

All notable changes to the GNN-PDE framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.8.1] - 2026-03-16 - Planned

### TODO split `GraphTopology` from `GraphsTuple` 

## [2.8.0] - 2026-03-16

Jump to next version.

## [2.7.6] - 2026-03-16

### Added

- **Multiscale Components Test Suite** — `tests/test_multiscale.py`:
  - Comprehensive test coverage for all multiscale components (59 tests total)
  - `TestGraphPool` / `TestGraphUnpool`: Graph U-Nets pooling/unpooling (12 tests)
  - `TestGraphUNetProcessor`: Graph U-Net processor with encoder-decoder architecture
  - `TestMGKNProcessor`: MGKN V-cycle processor
  - `TestMultiResolutionFNOBlock`: Parallel frequency band processing
  - `TestUFNOBlock` / `TestMiniUNet`: U-FNO with spectral + local branches (11 tests)
  - `TestHierarchicalFNOBlock`: Coarse-to-fine FNO processing
  - `TestHierarchicalGraph`: MGKN hierarchy utilities
  - `TestIntegration`: End-to-end multiscale model tests

### Fixed

- `components/spectral.py`: Added `padding` parameter to `_get_conv_nd()` for MiniUNet support
- `components/multiscale/graph_pooling.py`: Fixed device handling and edge preservation during pooling
- `components/multiscale/graph_unet.py`: Fixed device placement for dynamically created pool layers
- `components/multiscale/mgkn_processor.py`: Fixed initialization and forward pass logic
- `components/multiscale/spectral_multiscale.py`: Fixed MiniUNet skip connections (concatenate channels)
- `components/multiscale/hierarchy.py`: Fixed device handling and removed duplicate code
- `components/multiscale/__init__.py`: Added missing exports (`MiniUNet`, `restrict_to_coarse`, `prolong_to_fine`)

## [2.7.5] - 2026-03-16

### Added

- **Temperature mechanisms for attention-based models** — `components/temperature.py`:
  - `TemperatureBase`: Abstract base class for temperature mechanisms
  - `FixedTemperature`: Fixed temperature (backward compatible)
  - `LearnableScalarTemperature`: Global learnable scalar temperature
  - `PerHeadTemperature`: Per-head learnable temperature (from research papers)
  - `AdaptiveTemperature`: Per-point adaptive temperature (Ada-Temp from Transolver++)
  - `AnnealedTemperature`: Training-time temperature annealing (from Low-Width Graph Transformers)
  - `create_temperature_module()`: Factory function for creating temperature modules

### Changed

- `components/transformer.py` — Integrated temperature support in transformer models:
  - Added `temperature` and `temperature_mode` parameters to `SliceAttention`, `PhysicsTokenTransformer`, and `GraphTransformer`
  - Supported temperature modes: `'fixed'`, `'learnable_scalar'`, `'per_head'`, `'adaptive'`, `'annealed'`
  - Added optional Gumbel-Softmax noise support (following Transolver++ Eq. 4)
  - Added `set_epoch()` method for temperature annealing schedules
  - New parameters: `min_temperature`, `anneal_warmup_epochs`, `anneal_factor`, `anneal_final_temp`


## [2.7.4] - 2026-03-16

### Added

- **Multiscale Architecture Support** — Complete implementation addressing code review issue 2.4:
  - New module `components/multiscale/` with comprehensive multiscale processing capabilities.
  
  - **Graph Pooling/Unpooling** (`graph_pooling.py`):
    - `GraphPool`: Adaptive pooling with trainable projection (Graph U-Nets style).
    - `GraphUnpool`: Position-aware restoration using stored indices.
    - Graph power augmentation (A^n) for improved connectivity.
    - Based on Gao & Ji, "Graph U-Nets", ICML 2019.
  
  - **Hierarchical Graph Builder** (`hierarchy.py`):
    - `HierarchicalGraph`: Container for multi-level graph representations.
    - `build_hierarchical_graphs`: Automated hierarchy construction.
    - `restrict_to_coarse` / `prolong_to_fine`: Level transition utilities.
    - Based on Li et al., "Multipole Graph Neural Operator", NeurIPS 2020.
  
  - **Graph U-Net Processor** (`graph_unet.py`):
    - `GraphUNetProcessor`: Encoder-decoder architecture with skip connections.
    - Configurable pooling ratios and skip connection types (add/concat).
    - Multi-level feature encoding for large-domain simulations.
  
  - **MGKN Processor** (`mgkn_processor.py`):
    - `MGKNProcessor`: V-cycle algorithm with linear complexity O(N).
    - Downward pass (fine→coarse) and upward pass (coarse→fine).
    - Multi-resolution kernel decomposition for long-range interactions.
  
  - **Multi-Resolution Spectral Components** (`spectral_multiscale.py`):
    - `MultiResolutionFNOBlock`: Parallel frequency band processing.
    - `UFNOBlock`: U-FNO combining spectral (global) + U-Net (local) processing.
    - `HierarchicalFNOBlock`: Coarse-to-fine FNO processing.
    - `MiniUNet`: Local feature extraction for U-FNO.
    - Super-resolution utilities for zero-shot resolution transfer.
    - Based on Wen et al., "U-FNO", 2022.
  
  - **Complete Multiscale Models** (`models/multiscale_fno.py`):
    - `MultiscaleFNO`: End-to-end model with architecture options:
      - `"multiband"`: Multi-band frequency processing.
      - `"ufno"`: U-FNO with local+global processing.
      - `"hierarchical"`: Coarse-to-fine processing.
    - Super-resolution inference support.

### Fixed

- **Code Review Issue 2.4 — No Multi-Scale Architecture Support**:
  - ✅ Added U-Net style skip connections via `GraphUNetProcessor`.
  - ✅ Added hierarchical graph pooling/unpooling via `GraphPool`, `GraphUnpool`, `MGKNProcessor`.
  - ✅ Added multi-resolution processing via `MultiResolutionFNOBlock`, `HierarchicalFNOBlock`.
  - Critical for turbulence, multi-physics, and large-domain simulations.

## [2.7.3] - 2026-03-16

### Fixed

- **Protocol Inconsistency (Issue 1.1)** — Strict Protocol Enforcement:
  - Changed `EncodeProcessDecode` to use strict protocol types instead of `Union[Protocol, nn.Module]`.
  - Split `Decoder` protocol into `NodeDecoder` (fixed-node decoders) and `QueryDecoder` (query-based decoders).
  - `Decoder` is now a backwards-compatible Union type alias: `Union[NodeDecoder, QueryDecoder]`.
  - Updated exports in `core/__init__.py` and `components/__init__.py`.
  - See `docs/PROTOCOL_FIX_SUMMARY.md` for detailed migration guide.

### Changed

- **Type Safety Improvements**:
  - `EncodeProcessDecode.__init__()` now enforces structural typing via Protocols.
  - `ProbeDecoder` now requires `query_positions: Tensor` (not Optional) at the protocol level.
  - All existing components satisfy their protocols via duck typing (no inheritance changes needed).


## [2.7.2] - 2026-03-16

### Added

- **`core/mlp.py` — Extended Normalization Options**:
  - Added support for `BatchNorm1d` (`'batch'`), `InstanceNorm1d` (`'instance'`), and
    `GroupNorm` (`'group'`) in addition to existing `LayerNorm` support.
  - Dict-based configuration for fine-grained control (e.g., `{'type': 'group', 'num_groups': 4}`).
  - Auto-adjustment of `num_groups` for GroupNorm when dimension is not divisible.
  - Useful for PDE solvers: BatchNorm for stable training, InstanceNorm for style-transfer-like
    physics, GroupNorm for small batch regimes.

## [2.7.1] - 2026-03-16

### Added

- **`components/transformer.py` — Relative Position Encoding (RPE)**:
  - Added `RelativePositionEncoding` class with support for learned and sinusoidal
    position embeddings based on pairwise spatial distances.
  - Modified `MultiHeadAttention` to optionally use RPE via `use_relative_positions`
    parameter, enabling spatially-aware attention for PDE applications.
  - Updated `TransformerBlock` and `TransformerProcessor` to propagate positions
    through the attention mechanism.
  - Logarithmic distance bucketing for fine-grained local and coarse long-range
    interactions (following T5/Transformer-XL style).

- **`tests/test_components.py`** — Added comprehensive test coverage for RPE:
  - `TestMultiHeadAttention`, `TestTransformerBlock`, `TestTransformerProcessor`
    with tests for learned/sinusoidal encodings, batched inputs, gradient flow,
    and backward compatibility.

- **`examples/example_relative_position_attention.py`** — Usage examples for RPE.

## [2.7.0] - 2026-03-15

New start.

## [2.6.5] - 2026-03-15

### Added

- **`tests/test_fno.py`** — comprehensive test suite for FNO components:
  - Tests for `SpectralConv`, `SeparableSpectralConv`, `FNOBlock`, `AFNOBlock`,
    and `FNOProcessor`.
  - Dimensionality tests (1D, 2D, 3D), gradient flow verification, and
    device-agnostic execution (CPU/CUDA).

- **`components/spectral.py` — `SpectralConvBase` abstract base class**:
  - Unified `compl_mul_nd()` using dynamic einsum for n-dimensional Fourier space
    multiplication (replaces separate 1D/2D/3D implementations).
  - `make_spectral_conv()` factory for runtime selection between standard and
    separable spectral convolutions.

- **`.gitignore`** — added `.pytest*/` and `.benchmark*/` to ignore test and
  benchmark artifacts.

### Changed

- **`README.md`** — added "Global Features: Design Philosophy" section explaining
  the DeepMind Graph Nets `globals` design pattern, when to use globals vs
  node/edge-only features, and how they flow through `GlobalGraphNetBlock`.
  Updated usage patterns with factory function guidance.

- **`components/layers.py` — `make_residual()` documentation**:
  - Clarified factory usage for runtime polymorphism vs direct class
    instantiation for explicit construction.

- **`components/processors.py`** — removed unused imports (`broadcast_global`,
  `aggregate_to_global`) from `GraphNetBlock` (now internal to globals-enabled
  processors only).

### Removed

- **`architectural-review-report.md`** — deleted obsolete design review document.

- **`components/encoders.py` — `MeshEncoder` class**:
  - Consolidated on `MLPMeshEncoder` as the canonical mesh encoder.

---

## [2.6.4] - 2026-03-15

### Added

- **`core/functional.py` — two new graph-level broadcast/aggregate utilities**:
  - `broadcast_global(globals_, counts)` — repeats per-graph global vectors to
    every node or edge (`[B, D] → [total, D]`); replaces inline
    `torch.repeat_interleave` calls that were scattered across `GraphNetBlock`.
  - `aggregate_to_global(features, counts, method)` — pools per-node or
    per-edge features back to graph level (`[total, D] → [B, D]`); supports
    `'mean'`, `'sum'`, `'max'`, `'min'`.  Both functions exported from
    `gnn_pde_v2` root.

- **`components/processors.py` — `GlobalGraphNetBlock` and
  `GlobalGraphNetProcessor`**:
  - Full DeepMind Graph Nets block with globals as a first-class participant.
    Performs the complete three-step update — edges → nodes → globals — with
    `globals` broadcast downward to every edge/node and aggregated back upward.
  - `GlobalGraphNetBlock(latent_dim, global_latent_dim, ...)` — `global_latent_dim`
    is a required `int`, not `Optional`; raises `AssertionError` if
    `graph.globals is None`.
  - `GlobalGraphNetProcessor(latent_dim, global_latent_dim, n_layers, ...)` —
    stacks `GlobalGraphNetBlock` layers; residual connections apply to all three
    feature streams (nodes, edges, globals).
  - Both accept an optional `aggregate_fn` for edge-to-node aggregation and a
    `global_pool` parameter (`'mean'` default) for node/edge → global pooling.

### Changed

- **`components/processors.py` — `GraphNetBlock` and `GraphNetProcessor` are now
  node/edge-only** (no globals):
  - `global_latent_dim` parameter **removed** from both classes.  Passing it now
    raises `TypeError`.  Use `GlobalGraphNetBlock` / `GlobalGraphNetProcessor`
    for graphs that carry global state.
  - `forward` contains zero `if globals_ is not None` branches; `graph.globals`
    is passed through unchanged.
  - Private methods `_map_global_to_edges`, `_map_global_to_nodes`,
    `_aggregate_to_global` removed; logic lives in `core.functional`.
  - `scatter_sum` import replaced by `aggregate_edges` from `core.functional`.
  - Added `aggregate_fn` injection parameter (default: sum) for custom
    edge-to-node pooling.
  - Residual path in `GraphNetProcessor` no longer guards on
    `graph.nodes is not None`.

- **`models/gnn_model.py` — `GraphNet` and `MeshGraphNet`**:
  - `GraphNet` now branches on `global_in_dim`: selects `GlobalGraphNetProcessor`
    when globals are present, `GraphNetProcessor` otherwise.
  - `MeshGraphNet` uses plain `GraphNetProcessor` with no `global_latent_dim`
    argument (was previously passing `global_latent_dim=None`).

- **`components/__init__.py`** — exports `GlobalGraphNetBlock`,
  `GlobalGraphNetProcessor`.

### Tests

- `TestGraphNetBlock` — updated: `global_latent_dim=None` call removed; added
  `test_globals_passed_through` and `test_batched`.
- `TestGlobalGraphNetBlock` — new: `test_forward`, `test_globals_updated`,
  `test_batched`, `test_missing_globals_raises`.
- `TestGlobalGraphNetProcessor` — new: `test_forward`, `test_residual_globals`.

---

## [2.6.3] - 2026-03-14

### Changed

- **`core/protocols.py` — `ConditioningProtocol` is now generic over its
  condition type** (`ConditioningProtocol[CondT]`):
  - Added `CondT = TypeVar("CondT")` exported from `core.protocols`.
  - `forward(self, condition: CondT) -> Modulation` replaces the uncontracted
    `condition: Any` signature; static analysers now enforce the correct
    condition type at every call site.
- **`components/transformer.py` — concrete conditioners bind `CondT`**:
  - `ZeroConditioning(ConditioningProtocol[object])` — accepts any value / `None`.
  - `AdaLNConditioning(ConditioningProtocol[Tensor])` — requires `Tensor[..., cond_dim]`.
  - `DualAdaLNConditioning(ConditioningProtocol[Tensor])` — requires
    `Tensor[..., mu_dim + f_dim]`; adds a runtime shape guard that raises
    `ValueError` when the last dimension does not match.
  - `FiLMConditioning(ConditioningProtocol[Tensor])` — requires `Tensor[..., cond_dim]`.

---

## [2.6.2] - 2026-03-14

### Added

- **`core/registry.py` — `ModelRegistry` standalone class**:
  - Extracted registry state and management API into a standalone `ModelRegistry`
    object, independent of any model base class.
  - `MODEL_REGISTRY` module-level singleton is now the single source of truth;
    exported from `gnn_pde_v2.core`.
  - `ModelRegistry.register(name, namespace=, aliases=)` decorator registers any
    `nn.Module` subclass without requiring inheritance from `AutoRegisterModel`.
  - `ModelRegistry.add(cls, name, ...)` for imperative registration.
  - Container protocol: `name in registry`, `registry[name]`, `len(registry)`,
    `repr(registry)`.
  - `aliases=` parameter on both `ModelRegistry.add` / `.register` and
    `AutoRegisterModel.__init_subclass__` — multiple names map to the same class.

- **Built-in aliases** registered for all framework models:
  - `GraphNet`: `gnn`, `graph_net`
  - `MeshGraphNet`: `mgn`, `mesh_graph_net`
  - `FNO`: `fourier_no`, `fno2d`
  - `TFNO`: `tensorized_fno`
  - `AFNO`: `adaptive_fno`

### Changed

- **`AutoRegisterModel` now warns when `name=` is omitted** in a subclass
  definition; defaulting to the lowercased class name silently was a source of
  accidental public-API leakage.
- **`AutoRegisterModel` classmethods** (`create`, `list_models`, `get_model_info`,
  `unregister`, `clear_registry`) delegate to `MODEL_REGISTRY`; `_registry` is the
  same dict object for full backwards compatibility.

### Fixed

- **`models/__init__.py` lazy-import** (`__getattr__`): replaced `__import__` with
  `importlib.import_module` — the previous `__import__('.fno_model', level=1)`
  call produced a trailing-dot module name (`gnn_pde_v2.models.`) that silently
  failed for direct `from gnn_pde_v2.models import FNO` calls.

## [2.6.1] - 2026-03-14

### Changed

- **`components/probe.py` — `ProbeDecoder` refactor and batch support**:
  - **Batch processing**: `forward` now accepts a batched `GraphsTuple` (output of
    `batch_graphs`) plus an optional `n_query: Tensor[B]` describing per-graph query
    counts; defaults to single-graph behaviour when `n_query` is `None`.
  - **`_construct_probe_graph`**: contract narrowed to single-graph; `forward` loops
    over `B` graphs only for `cdist`/`topk` k-NN (inherently local); all subsequent
    MLP and scatter ops run on the packed `batch_graphs` result.
  - **`_encode_edge_features`** new method: extracts the `rbf`/`pos-diff` branching
    from `_construct_probe_graph`; fixed RBF bug where
    `nearest_distances.reshape(-1,1).expand(-1, k)` repeated a single scalar per edge
    — replaced with `repeat_interleave(k, dim=0)` to broadcast each query's full
    k-distance row.
  - **`_segment_bounds`** new static utility: returns `(starts, ends)` tensors via
    `cumsum`; replaces the old `_packed_slices` generator; no Python loop.
  - **`_extract_probe_nodes`** fully vectorised: replaced `B`-iteration list
    comprehension with a single `all_nodes[base + local_offsets]` gather using the
    cumshift trick (`arange(total_q) - seg_starts.repeat_interleave(n_query)`).
  - **Scatter consistency**: replaced all `index_add_` calls with `scatter_sum` /
    `scatter_mean` from `core.functional`.

## [2.6.0] - 2026-03-13

### Added

- **Improved dependency error handling** in `utils/graph_utils.py`:
  - Added module-level check for `torch_cluster` availability
  - Added helpful error messages with installation instructions for:
    - `knn_graph()` - raises `ImportError` with pip instructions
    - `radius_graph()` - raises `ImportError` with pip instructions
    - `mesh_to_graph()` (when `faces=None`) - raises `ImportError` with pip instructions
  - Error messages include both standard pip command and CUDA-specific installation URL

- **Implemented lazy loading with helpful errors** in `models/__init__.py`:
  - Added `__getattr__` for lazy imports of optional models
  - Models are only imported when accessed, not at module load time
  - Provides clear error messages when dependencies are missing

- **Eliminated `convenient/` layer entirely**:
  - Removed `convenient/config.py` — Pydantic-based configurations deleted
  - Removed `convenient/__init__.py`
  - Architecture now simplified to: `core → components → models`
  - `AutoRegisterModel` remains available from `gnn_pde_v2.core` (already moved in v2.4.0)

### Changed

- **Updated all imports** across 6 example files:
  - Changed `from gnn_pde_v2.convenient import AutoRegisterModel` 
  - To `from gnn_pde_v2.core import AutoRegisterModel`
- **Updated documentation**:
  - Removed "Convenient API" section from README.md
  - Updated examples/README.md to reference `core.AutoRegisterModel`
  - Removed `test_convenient.py` test file

### Migration

```python
# Before (v2.5.0 and earlier)
from gnn_pde_v2.convenient import AutoRegisterModel

# After (v2.6.0+)
from gnn_pde_v2.core import AutoRegisterModel
```

## [2.5.0] - 2026-03-13

### Added

- **`core/protocols.py` — structural protocol system**:
  - Seven `runtime_checkable` `typing.Protocol` definitions covering all component
    boundaries in both the graph-world and grid-world pipelines:
    - `GraphEncoder`: `(GraphsTuple) → GraphsTuple`
    - `GraphProcessor`: `(GraphsTuple) → GraphsTuple`
    - `Decoder`: `(GraphsTuple, Optional[Tensor]) → Tensor`
    - `GraphModel`: `(GraphsTuple) → Tensor`
    - `PositionEncoder`: `(Tensor) → Tensor`
    - `GridProcessor`: `(Tensor) → Tensor`
    - `GridModel`: `(Tensor) → Tensor`
  - `Modulation` dataclass and `ConditioningProtocol` ABC relocated here from
    `components/transformer.py`; re-exported from `transformer.py` for backwards
    compatibility
  - All protocols use `@runtime_checkable`, so `isinstance(obj, GraphProcessor)`
    works at runtime without inheritance
- **`AutoRegisterModel.unregister(name)` class method**: removes a single entry
  from the registry; raises `KeyError` with a descriptive message if absent
- **`AutoRegisterModel.clear_registry(namespace=None)` class method**: clears
  the entire registry, or only entries within a given namespace prefix
- **`MLP.in_features` property**: returns the `in_features` of the first
  `nn.Linear` layer; eliminates the need to index into internal layers
- **`MLP.out_features` property**: returns the `out_features` of the last
  `nn.Linear` layer
- **`MLP.layers` property**: read-only accessor for the underlying
  `nn.Sequential`, replacing direct `.net` access

### Changed

- **`MLP.net` renamed to `MLP._net`** (**breaking checkpoint change**):
  - Internal `nn.Sequential` is now private; use `.layers`, `.in_features`,
    or `.out_features` instead
  - `state_dict` keys change from `net.N.*` to `_net.N.*`; existing saved
    checkpoints must be remapped (see migration note below)
- **`EncodeProcessDecode.__init__` parameters** now typed as
  `Union[GraphEncoder, nn.Module]`, `Union[GraphProcessor, nn.Module]`, and
  `Union[Decoder, nn.Module]`, making the expected protocol explicit in the
  signature
- **`ProbeDecoder.forward` signature** changed from `query_positions: Tensor`
  (required positional) to `query_positions: Optional[Tensor] = None` (optional
  keyword), now matching the `Decoder` protocol; a clear `ValueError` is raised
  when `None` is passed, preserving the original runtime behaviour
- **`components/transformer.py`** no longer defines `Modulation` or
  `ConditioningProtocol`; both are imported from `core.protocols` and re-exported
  so all existing imports continue to work unchanged
- **`core/__init__.py`** now exports all seven protocols plus `Modulation` and
  `ConditioningProtocol`
- **`components/__init__.py`** now exports all seven protocols, `MeshEncoder`,
  and updated conditioning symbols

### Fixed

- **Abstraction leak — `MLP.net`**: internal `nn.Sequential` was publicly
  accessible and used directly in `examples/example_meshgraphnets.py`
  (`.net[0].in_features`, `.net[-1].out_features`); replaced with
  `.in_features` / `.out_features` properties
- **Abstraction leak — registry direct mutation**: `tests/test_convenient.py`
  called `._registry.clear()` and `del ._registry[key]` directly; updated to
  use `clear_registry()` and `unregister()` instead
- **`Decoder` protocol inconsistency**: `ProbeDecoder` required `query_positions`
  as a mandatory positional argument while `MLPDecoder` and `IndependentMLPDecoder`
  made it optional, preventing the three decoders from being used
  interchangeably; all three now satisfy the `Decoder` protocol

### Migration Notes

**`MLP` checkpoint migration** (only relevant if loading saved state dicts):
```python
# Remap old keys to new keys
old_state = torch.load("checkpoint.pt")
new_state = {k.replace("net.", "_net.", 1): v for k, v in old_state.items()}
model.load_state_dict(new_state)
```

**Registry management in tests / custom code:**
```python
# Before
AutoRegisterModel._registry.clear()
del AutoRegisterModel._registry['my_model']

# After
AutoRegisterModel.clear_registry()
AutoRegisterModel.unregister('my_model')
```

**`ProbeDecoder` call sites** (no change required; signature is backwards compatible):
```python
# Still works — query_positions passed as keyword or positional
output = probe_decoder(graph, query_positions=coords)
output = probe_decoder(graph, coords)

# Now raises ValueError with a clear message instead of TypeError
output = probe_decoder(graph)  # ValueError: ProbeDecoder requires query_positions
```

**Protocol type annotations:**
```python
from gnn_pde_v2.core import GraphProcessor, Decoder

def build_pipeline(proc: GraphProcessor, dec: Decoder): ...
```

---

## [2.4.0] - 2026-03-13

### Added

- **`scatter_min` and `scatter_softmax` in `core.functional`**: Extended aggregation operations for graph message passing
- **`AFNOBlock` in `components/spectral`**: Adaptive Fourier Neural Operator block with soft-thresholding sparsity (Guibas et al. 2021)
- **`docs/architecture-dependencies.md`**: Documentation for module dependency structure

### Changed

- **Moved `AutoRegisterModel` from `convenient` to `core`**:
  - `convenient/registry.py` → `core/registry.py`
  - Registry is now a core feature, not convenience layer
  - Eliminates circular dependency: models no longer depend on convenient layer
  
- **Renamed `components/fno.py` → `components/spectral.py`**:
  - Clearer intent: spectral methods for regular grids (not graph-compatible)
  - Added explicit documentation that spectral processors work on tensors, not GraphsTuple
  - Exported `AFNOBlock` alongside existing FNO components

- **Simplified encoders (`components/encoders.py`)**:
  - Removed `MLPEncoder`, `MLPMeshEncoder`, and `make_mlp_encoder` helper
  - Consolidated into single `MeshEncoder` class with clean API:
    ```python
    MeshEncoder(node_in_dim, edge_in_dim, global_in_dim, latent_dim, hidden_dim=128)
    ```
  - Simplified MLP configuration (removed per-parameter weight_init, bias_init, dropout)

- **Consolidated residual connections (`components/layers.py`)**:
  - Removed `ResidualBlock`, `PreNormResidual`, `ResidualSequence`, `SkipConnection`
  - Consolidated into unified `Residual` class supporting:
    - Optional normalization (any `nn.Module`)
    - Optional scaling (fixed or learnable)
  - `make_residual` factory updated to use simplified classes
  - `GatedResidual` preserved for learnable gating use cases

- **Standardized processor API (`components/processors.py`)**:
  - Unified dimension parameters: `node_dim` + `edge_dim` → single `latent_dim`
  - `global_dim` → `global_latent_dim` for consistency
  - All features (nodes, edges, globals) share the same latent dimension
  - Better docstrings with Args/Returns documentation

- **Standardized decoder API (`components/decoders.py`)**:
  - `node_dim` → `latent_dim` parameter name
  - Consistent with encoder/processor terminology

- **Standardized probe decoder (`components/probe.py`)**:
  - `node_dim` → `latent_dim` parameter name
  - Added comprehensive docstrings

- **Standardized model parameters**:
  - `n_message_passing` / `message_passing_steps` → `n_layers` consistently across all models
  - Updated `GNNConfig`, `GraphNet`, `MeshGraphNet`, `MeshGraphNets`
  - README and examples updated

- **Updated `__init__.py` exports**:
  - Now exports `AutoRegisterModel` from core
  - Exports `scatter_min`, `scatter_softmax` from core.functional

### Removed

- **`convenient/aggregation.py`**: Aggregation functions now exclusively in `core/functional`
- **`convenient/initializers.py`**: String-based initialization removed (use functional init directly)
- **`convenient/registry.py`**: Moved to `core/registry.py`
- **`make_mlp_encoder` helper**: Direct `MLP` instantiation is clearer
- **`MLPEncoder` class**: Use `MLP` directly or `MeshEncoder` for graph encoding
- **`MLPMeshEncoder` class**: Replaced by simplified `MeshEncoder`
- **Redundant residual classes**: `ResidualBlock`, `PreNormResidual`, `ResidualSequence`, `SkipConnection`
- **`FNOProcessor.forward_graph` placeholder**: Method was unimplemented; FNO is for grids only

### Fixed

- **Circular dependency resolved**: Models module no longer imports from convenient layer
- **API consistency**: All processors use `n_layers` for layer count
- **Dimensional consistency**: All components use `latent_dim` for hidden representation size

### Migration Notes

**Encoder usage:**
```python
# Before
from gnn_pde_v2.components import MLPMeshEncoder
encoder = MLPMeshEncoder(node_in_dim=11, edge_in_dim=3, global_in_dim=None, 
                         latent_dim=128, hidden_dims=[128], dropout=0.0)

# After
from gnn_pde_v2.components import MeshEncoder
encoder = MeshEncoder(node_in_dim=11, edge_in_dim=3, global_in_dim=None,
                      latent_dim=128, hidden_dim=128)
```

**Residual connections:**
```python
# Before
from gnn_pde_v2.components import ResidualBlock, PreNormResidual
block = ResidualBlock(module, residual_type='scaled', learnable_scale=True)
pre_norm = PreNormResidual(module, dim=128)

# After
from gnn_pde_v2.components import Residual
block = Residual(module, scale=1.0, learnable_scale=True)
pre_norm = Residual(module, norm=nn.LayerNorm(128))
```

**Registry import:**
```python
# Before
from gnn_pde_v2.convenient import AutoRegisterModel

# After
from gnn_pde_v2.core import AutoRegisterModel
```

**Model configuration:**
```python
# Before
config = GNNConfig(n_message_passing=15, ...)
model = MeshGraphNet(n_message_passing=15, ...)

# After
config = GNNConfig(n_layers=15, ...)
model = MeshGraphNet(n_layers=15, ...)
```

## [2.3.4] - 2026-03-13

### Added

- **Generic conditioning protocol for transformer components**:
  - `Modulation` dataclass: encapsulates shift, scale, gate, and cross_kv parameters for flexible attention modulation
  - `ConditioningProtocol` ABC: pluggable interface for implementing custom conditioning schemes
  - `ZeroConditioning`: identity passthrough (no modulation applied)
  - `AdaLNConditioning`: single-source Adaptive Layer Normalization conditioning
  - `DualAdaLNConditioning`: UniSolver-style dual conditioning with μ (mean) and f (forcing) embeddings
  - `FiLMConditioning`: Feature-wise Linear Modulation (γ, β) for feature space transformation
  - `_apply_modulation` helper: unified tensor modulation application
- **Exported conditioning components** from `gnn_pde_v2.components`:
  - `Modulation`, `ConditioningProtocol`, `ZeroConditioning`, `AdaLNConditioning`, `DualAdaLNConditioning`, `FiLMConditioning`

### Changed

- **Transformer blocks now support pluggable conditioning**: enables research on various modulation strategies (AdaLN, FiLM, dual-source)

## [2.3.3] - 2026-03-13

### Changed

- **Consolidated residual connections** into `components/layers.py`:
  - Moved all residual connection implementations from `layers/residual.py` to `components/layers.py`
  - Removed the now-redundant `layers/residual.py` file
  - Provides comprehensive collection: `Residual`, `PreNormResidual`, `PostNormResidual`, `GRUGated`, `Highway`, `StochasticDepth`, etc.
- **Extended `components/layers.py`** with enhanced residual connection support and improved documentation

### Removed

- `layers/residual.py` (consolidated into `components/layers.py`)

## [2.3.2] - 2026-03-13

### Changed

- **`AutoRegisterModel` now inherits from `BaseModel`**:
  - Unifies the model hierarchy: all registered models now share common `BaseModel` interface
  - Improves type consistency across the codebase
  - Enhanced `convenient/registry.py` with better inheritance structure
- **Updated all example models** to use the unified `AutoRegisterModel` pattern consistently 

## [2.3.1] - 2026-03-13

### Added

- **`pre_activation` option in `core.MLP`**: Supports pre-activation pattern (Activation → Linear per layer)
  useful for architectures like AdaLN modulation networks in UniSolver

### Changed

- **Refactored DeepXDE examples** to use framework components:
  - `example_deepxde_style.py`: Uses `core.MLP`, `FourierFeatureEncoder`, `get_initializer`
  - `example_deepxde.py`: Removed redundant `FNN` class, uses `MLP` directly, added `DeepONet`
- **Replaced custom MLP implementations** with framework `core.MLP`:
  - `components/transformer.py`: `TransformerBlock.mlp`
  - `examples/example_graph_pde_gno.py`: `GraphConvBlock.edge_weight_net`

## [2.3.0] - 2026-03-12

### Added

- **Renovated `MLP` in `components/encoders.py`** with a richer, research-oriented API:
  - Separate `norm` / `final_norm` specs for hidden vs. output normalization
  - Separate `activation` / `final_activation` and `dropout` / `final_dropout` for independent per-phase control
  - New `linear_factory` parameter: pass a callable `(in_dim, out_dim) -> nn.Module` to build pointwise-conv stacks (e.g. `nn.Conv2d(..., kernel_size=1)`) instead of dense `nn.Linear`
  - New activations: `sigmoid` (`nn.Sigmoid`) and `sin` (new `SinActivation` module)
  - Backward-compatible: `use_layer_norm=True` maps to `norm='layer'` for existing call-sites
- **`SinActivation` module** (`components/encoders.py`) — trivial `nn.Module` wrapping `torch.sin`
- **Architecture-faithful example rewrites** for all seven supported papers:
  - `example_meshgraphnets.py`: local 4-linear MLP blocks with terminal `LayerNorm`, explicit GraphNet encoder/processor/decoder structure
  - `example_windfarm_gno.py`: output-only `LayerNorm` MLP matching original GNO design
  - `example_graph_pde_gno.py`: `NNConv`-style edge-conditioned convolution with full kernel matrices; removed incorrect scalar gating
  - `example_transolver.py`: locked FFN parity to original, fixed device placement
  - `example_deepxde.py`: new `DeepXDEFNN` with `sin`/`sigmoid` support, per-layer dropout, faithful residual connections, and proper Glorot/He initializers; no spurious `LayerNorm`
  - `example_neuraloperator_fno.py`: added `linear_skip` (1×1 conv) branch inside `FNOBlockFramework`; residual now applied internally per original; lifting/projection use `MLP` with `linear_factory`
  - `example_unisolver.py`: `FeedForward` uses framework `MLP`; removed unused import
- **Backward-compatible model aliases** added to example modules:
  - `PINNModel = DeepXDEModel`
  - `FNO = NeuralOperatorFNO`
  - `UniSolver = Unisolver`
  - `GraphPDEGNO = GraphPDE_GNO`
- `examples/core/__init__.py` added so `core.meshgraphnets_core` is importable in tests
- Expanded `tests/test_components.py` to cover new `MLP` API: hidden-only norm, final-only norm, `sigmoid`/`sin` activations, `linear_factory` with 1×1 convolutions, and legacy `use_layer_norm` compatibility

### Fixed

- **`ProbeDecoder` indexing bug**: receiver indices for aggregation were not offset by source node count; aggregation now uses correct local indices before offsetting
- `example_deepxde.py`: `sin` and `sigmoid` activations were silently remapped to `tanh`; now supported natively
- `example_neuraloperator_fno.py`: external `x = x + block(x)` residual was incorrect; residual is now internal to each block

### Changed

- `MLP` constructor extended with `norm`, `final_norm`, `final_activation`, `final_dropout`, `linear_factory` keyword arguments (fully additive; no existing positional arguments changed)
- README updated to document the renovated `MLP` API and new activation/norm options

## [2.2.0] - 2026-03-12

### Removed

- Removed v2.1 backward-compatibility shims for deprecated package-level import paths:
  - `gnn_pde_v2.core.base_model`
  - `gnn_pde_v2.core.protocols`
  - `gnn_pde_v2.config`
  - `gnn_pde_v2.encoders`
  - `gnn_pde_v2.processors`
  - `gnn_pde_v2.decoders`
  - `gnn_pde_v2.layers`
  - `gnn_pde_v2.initializers`
  - `gnn_pde_v2.utils.aggregation`
- Removed the legacy duplicate implementation directories after consolidation:
  - `encoders/`
  - `processors/`
  - `decoders/`

### Changed

- Documentation now reflects the post-shim canonical API only.
- The package tree now exposes only the lean `core`, canonical `components`,
  optional `convenient` API, and remaining non-deprecated utility/example code.
- Completed internal consolidation so reusable encoder, processor, and decoder
  implementations now live only under `components/`.
- Canonicalized `MLPEncoder` and `MLPMeshEncoder` under `gnn_pde_v2.components`
  and updated `models/gnn_model.py` to use the canonical wrapper implementation.

### Migration Notes

Use these canonical replacements:

- `gnn_pde_v2.core.base.BaseModel` instead of `gnn_pde_v2.core.base_model.BaseModel`
- `gnn_pde_v2.convenient.*` instead of `gnn_pde_v2.config.*`
- `gnn_pde_v2.components` instead of `gnn_pde_v2.encoders`, `processors`, `decoders`, or `layers`
- `gnn_pde_v2.convenient` or `torch.nn.init` instead of `gnn_pde_v2.initializers`
- `gnn_pde_v2.core` or `gnn_pde_v2.convenient` instead of `gnn_pde_v2.utils.aggregation`

## [2.1.0] - 2026-03-12

### Added

- **Three-layer architecture**: Core, Components, and Convenient
  - `core/` - Minimal essentials (~430 lines)
  - `components/` - Reusable building blocks (~1,350 lines)
  - `convenient/` - Optional high-level API (~930 lines)
- New examples demonstrating both approaches:
  - `examples/core/meshgraphnets_core.py` - Lean approach
  - `examples/convenient/meshgraphnets_easy.py` - High-level API
- Documentation:
  - `MIGRATION.md` - Migration guide from v2.0
  - `ARCHITECTURE.md` - Architecture overview
  - `CHANGELOG.md` - This file

### Changed

- Restructured package layout:
  - Core functionality moved to `gnn_pde_v2.core`
  - Building blocks moved to `gnn_pde_v2.components`
  - Optional features moved to `gnn_pde_v2.convenient`
- Simplified Residual: Single `Residual` class instead of 5 variants
- MLP now uses functional initialization (pass functions, not strings)
- Updated canonical model modules and examples to use the new structure

### Deprecated

The following imports still work but show deprecation warnings:

- `from gnn_pde_v2.encoders import X` → Use `gnn_pde_v2.components`
- `from gnn_pde_v2.processors import X` → Use `gnn_pde_v2.components`
- `from gnn_pde_v2.decoders import X` → Use `gnn_pde_v2.components`
- `from gnn_pde_v2.layers import X` → Use `gnn_pde_v2.components`
- `from gnn_pde_v2.initializers import X` → Use `gnn_pde_v2.convenient`
- `from gnn_pde_v2.utils.aggregation import X` → Use `gnn_pde_v2.core`
- `from gnn_pde_v2.config import X` → Use `gnn_pde_v2.convenient`
- `from gnn_pde_v2.core.base_model import BaseModel` → Use `gnn_pde_v2.core.base`
- `BaseModel with model_name='x'` → Use `AutoRegisterModel with name='x'`

These will be removed in v2.2.

### Removed

- `examples_bak/` directory (was backup, no longer needed)

### Migration Notes

To migrate from v2.0:

1. Update imports:
   ```python
   # Old
   from gnn_pde_v2.encoders import MLP

   # Current
   from gnn_pde_v2.core import MLP
   ```

2. For auto-registration:
   ```python
   # Old
   from gnn_pde_v2 import BaseModel
   class MyModel(BaseModel, model_name='my_model'):
       ...
   
   # New
   from gnn_pde_v2.convenient import AutoRegisterModel
   class MyModel(AutoRegisterModel, name='my_model'):
       ...
   ```

See [MIGRATION.md](MIGRATION.md) for complete guide.

## [2.0.0] - 2026-03-01

### Added

- Initial release of GNN-PDE v2
- Unified framework supporting 7 research papers:
  - MeshGraphNets
  - DeepXDE
  - FNO (Neural Operator)
  - Transolver
  - UniSolver
  - WindFarm GNO
  - Graph-PDE GNO
- Core architecture: Encode-Process-Decode
- BaseModel with auto-registration
- Components:
  - Encoders: MLP, MLPEncoder, MLPMeshEncoder, FourierFeatureEncoder
  - Processors: GraphNetBlock, TransformerBlock, FNOProcessor
  - Decoders: MLPDecoder, IndependentMLPDecoder, ProbeDecoder
- Utilities:
  - GraphsTuple for graph representation
  - Aggregation functions (scatter_sum, scatter_mean, etc.)
  - Graph utilities (knn_graph, radius_graph)
  - Spatial utilities (grid_to_points, points_to_grid)
- Configuration system with Pydantic
- Unified Model wrapper for training
- Comprehensive test suite (59 tests)

## Future Versions

### [3.0.0] (Future)

- Potential breaking changes based on v2.x feedback
- Performance optimizations
- Additional processor types
- Extended documentation and tutorials

---

## Version Compatibility

| Version | Python | PyTorch | Status |
|---------|--------|---------|--------|
| 2.5.x | 3.9+ | 2.0+ | Current |
| 2.4.x | 3.9+ | 2.0+ | Deprecated |
| 2.3.x | 3.9+ | 2.0+ | Deprecated |
| 2.2.x | 3.9+ | 2.0+ | Deprecated |
| 2.1.x | 3.9+ | 2.0+ | Deprecated |
| 2.0.x | 3.9+ | 2.0+ | Deprecated |

## Deprecation Timeline

| Feature | Deprecated In | Will Be Removed In |
|---------|---------------|-------------------|
| Old import paths | 2.1.0 | 2.2.0 |
| `BaseModel` with `model_name` | 2.1.0 | 2.2.0 |
| `protocols.py` | 2.1.0 | 2.2.0 |
