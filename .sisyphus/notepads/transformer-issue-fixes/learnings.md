# Learnings

## [2026-05-12] Session Start
- Test infrastructure: pytest, conda env `ml_env`, no CI
- V3 has ZERO dedicated tests — only base PhysicsTokenAttention tested
- SparseGraphAttention has ZERO tests — new test file required
- V3 forward() is ~90% copy-paste of parent
- PhysicalStateCache does isinstance(V3) + deep attribute access
- adaptive temperature mode is architecturally incompatible with SparseGraphAttention
- All 9 issues verified as still present (7 fully, 2 partially)

## 2026-05-12: Task 1 - Deleted QKVProjectionType.TOKEN_SLICE dead code

- Deleted `TOKEN_SLICE = "token_slice"` enum member from `components/attention.py:28`
- Confirmed zero references to TOKEN_SLICE in entire codebase (grep returned no matches)
- `create_qkv_projection()` only handles COMBINED and SEPARATE; TOKEN_SLICE fell through to ValueError — confirming it was truly dead code
- All 17 attention-related tests pass after deletion
- Pre-existing test failure in `test_prelu_string_creates_prelu_modules` (assert 1 == 3) is unrelated to this change
- Evidence saved: `.sisyphus/evidence/task-1-token-slice-gone.txt`, `.sisyphus/evidence/task-1-tests-pass.txt`

## 2026-05-12: Task 2 - Fix AdaptiveTemperature hasattr bug

- Added `self.learnable_base = learnable_base` attribute in `AdaptiveTemperature.__init__()` (line 134)
- Replaced `hasattr(self, 'log_tau_0')` with `self.learnable_base` in `forward()` (line 148)
- State_dict keys unchanged: `log_tau_0` when learnable_base=True, `tau_0` when learnable_base=False
- `learnable_base` is a plain instance attribute — does not appear in state_dict
- Bug: `conda run` picks up wrong `pytest` from `~/.local/bin/` (uses system Python, not env). Fix: use `python -m pytest` instead of bare `pytest`
- All 81 temperature tests pass
- Evidence saved: `.sisyphus/evidence/task-2-tests-pass.txt`

## 2026-05-12: Task 3 - Parameter validation warnings in TransformerBlock

- Added `import warnings` + module-level `_PHYSICS_TOKEN_PARAMS`, `_POSITION_PARAMS`, `_warned_blocks` to `components/transformer.py`
- Warning fires when use_physics_tokens=False + explicit physics-token params differ from defaults (12 params)
- Warning fires when use_physics_tokens=True + explicit position params differ from defaults (4 params)
- Key insight: use `_local[p] is not None and _local[p] != _defaults[p]` to detect explicit user-passed params
  - `_local[p] is not None` alone is insufficient because `TransformerProcessor` passes ALL resolved params explicitly to `TransformerBlock`, causing false positives
- Comparing against `_defaults[p]` correctly distinguishes "user explicitly passed this non-default value" from "TransformerProcessor passed a resolved default"
- Module-level `_warned_blocks` set of frozensets prevents warning spam on repeated identical instantiations
- All 14 temperature integration tests pass
- `pytest.warns(UserWarning)` used in updated test to validate warning fires
- Pre-existing failures in test_components.py (2) are unrelated
- Evidence saved: `.sisyphus/evidence/task-3-tests-pass.txt`
- 
## 2026-05-12: Task 3 (re-execution) - Clean implementation
- Avoided previous scope creep: no TransformerConfig, no constructor signature changes, no new files
- Used `_vals = locals()` to access init parameters for comparison against `_defaults` dict
- Inserted validation block between `super().__init__()` and `self.norm1 = nn.LayerNorm(dim)`
- Updated `test_transformer_block_without_physics_tokens` with `pytest.warns(UserWarning, match="Ignored parameters")`
- All 41 temperature tests pass (integration + edge cases + training)

## Task 7: Add epoch parameter to ScheduledTemperature.forward()

- Added `_compute_temperature(self, epoch: int) -> float` private method that encapsulates temperature computation logic.
- Refactored `set_epoch()` to call `_compute_temperature()` instead of duplicating the formula.
- Extended `forward()` signature with `epoch: Optional[int] = None` parameter.
- Key design: `if epoch is not None:` (not `if epoch:`) to handle epoch=0 as valid.
- When `epoch` is provided, computes on-the-fly WITHOUT mutating `self.current_epoch` or `self._current_temp`.
- `AnnealedTemperature` and `FixedTemperature` inherit the updated `forward()` automatically.
- All 81 existing tests pass unchanged — full backward compatibility.
- Manual verification confirmed: mutable state isolation, constant/scheduled modes, boundary case epoch=0.

## 2026-05-12: Task 4 - Fix SparseGraphAttention temperature reshape bug

- **Bug**: `attn_scores.unsqueeze(0).unsqueeze(0)` produced `[1, 1, E, H]` — heads and edges dimensions swapped relative to temperature module's expected `[B, H, N, G]` interface
- **Fix**: `attn_scores.T.unsqueeze(0).unsqueeze(-1)` → `[1, H, E, 1]` and reverse `.squeeze(0).squeeze(-1).T` → `[E, H]`
- **Why it matters**: For per_head mode with multiple heads, the old code applied per-head temperatures along the wrong axis — scalar modes happened to work only by broadcasting coincidence
- **Adaptive mode guard**: Added ValueError in `__init__` blocking `temperature_mode='adaptive'` because per-node temperature features are incompatible with per-edge attention scores (N nodes ≠ E edges)
- **Tests**: Created `tests/test_sparse_graph_attention.py` with 21 tests covering construction validation (8) and forward pass (13) for all supported modes + edge cases (dropout, v_norm, edge_type_bias, determinism)
- **All 21 tests pass**; 0 failures

## 2026-05-12: Task 4 - Extract _compute_slice_tokens from PhysicsTokenAttention

- Extracted `_compute_slice_tokens(self, x)` method from `PhysicsTokenAttention.forward()` (lines ~901-937 → standalone method)
- The method handles: shape unpacking, 2D→3D unsqueeze, QKV-mode projection, Gumbel-Softmax, temperature, softmax, einsum aggregation, slice normalization
- Updated base `forward()` to call `self._compute_slice_tokens(x)` — single-line replacement of ~37 lines
- Added override `PhysicsTokenAttentionV3._compute_slice_tokens()` with tiling path + non-tiling fallback to `super()._compute_slice_tokens(x)`
- V3 tiling path preserved exactly, including the recompute anomaly when `use_slice_normalization=False` (marked with `# NOTE:`)
- V3 `forward()` updated to call `self._compute_slice_tokens(x)` — eliminated ~44 lines of duplicated slice code
- `from typing import Tuple` added to existing import line (required by return type annotation)
- 302 pytest passed; isinstance check + attribute access verified
- Import verification requires `PYTHONPATH` set to parent dir due to pre-existing `components/__init__.py` cascade issue with `FourierFeatureEncoder`

### Verification results:
- `pytest tests/ -x -q --ignore=tests/test_components.py --ignore=tests/test_examples.py`: 302 passed
- `isinstance(V3, PhysicsTokenAttention)`: PASS
- All attributes (`n_heads`, `in_project_x`, `n_tokens`, `head_dim`, `in_project_fx`, `slice_weight_proj`, `temperature_module`): accessible
- Base forward, V3 forward (non-tiling), V3 forward (tiling), single-batch forward: all produce correct shapes
- LSP diagnostics: all errors pre-existing (torch import resolution, Optional typing on to_q/to_k/to_v)
