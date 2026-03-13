# Changelog

All notable changes to the GNN-PDE framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
| 2.3.x | 3.9+ | 2.0+ | Current |
| 2.2.x | 3.9+ | 2.0+ | Deprecated |
| 2.1.x | 3.9+ | 2.0+ | Deprecated |
| 2.0.x | 3.9+ | 2.0+ | Deprecated |

## Deprecation Timeline

| Feature | Deprecated In | Will Be Removed In |
|---------|---------------|-------------------|
| Old import paths | 2.1.0 | 2.2.0 |
| `BaseModel` with `model_name` | 2.1.0 | 2.2.0 |
| `protocols.py` | 2.1.0 | 2.2.0 |
