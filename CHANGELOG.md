# Changelog

All notable changes to the GNN-PDE framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Changed

- Documentation now reflects the post-shim canonical API only.
- The package tree now exposes only the lean `core`, canonical `components`,
  optional `convenient` API, and remaining non-deprecated utility/example code.

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
   
   # New
   from gnn_pde_v2.components import MLP
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

### [2.3.0] (Planned)

- Add position encoding module
- Add physics-informed loss components
- Add conditioning layers (AdaLN, FiLM)

### [3.0.0] (Future)

- Potential breaking changes based on v2.x feedback
- Performance optimizations
- Additional processor types
- Extended documentation and tutorials

---

## Version Compatibility

| Version | Python | PyTorch | Status |
|---------|--------|---------|--------|
| 2.2.x | 3.9+ | 2.0+ | Current |
| 2.1.x | 3.9+ | 2.0+ | Deprecated |
| 2.0.x | 3.9+ | 2.0+ | Deprecated |

## Deprecation Timeline

| Feature | Deprecated In | Will Be Removed In |
|---------|---------------|-------------------|
| Old import paths | 2.1.0 | 2.2.0 |
| `BaseModel` with `model_name` | 2.1.0 | 2.2.0 |
| `protocols.py` | 2.1.0 | 2.2.0 |
