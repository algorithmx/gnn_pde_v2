# Code Review Summary: Transformer Architecture Issues

Reviewed against the current state of `components/transformer.py`,
`components/attention.py`, and `components/temperature.py`. Issues that were
already fixed or that did not match the code have been purged; remaining
entries reflect genuine, current shortcomings.

## Purged issues (verified resolved or inaccurate)

| Original claim | Reason for removal |
|----------------|--------------------|
| Dead code `QKVProjectionType.TOKEN_SLICE` | `QKVProjectionType` only defines `COMBINED` and `SEPARATE`; `TOKEN_SLICE` does not exist. |
| Silent parameter ignorance in `TransformerBlock` | Constructor already validates and emits `UserWarning` for ignored params via `_PHYSICS_TOKEN_PARAMS` / `_POSITION_PARAMS`. |
| `AdaptiveTemperature` `hasattr` bug | Constructor now stores `self.learnable_base` and branches on it directly — no `hasattr` involved. |
| `PhysicsTokenAttentionV3` Liskov violation | V3 preserves the `forward(x) -> Tensor` contract and only overrides `_compute_slice_tokens`/`forward`; it also composes `TiledSliceOperation` rather than duplicating logic. Substitutability holds. |
| `AnnealedTemperature` mutable-state bug | `ScheduledTemperature.forward` accepts an optional `epoch` argument for stateless use; the cached `_current_temp` is a convenience, not a correctness hazard (it's not in `state_dict`). |
| `TransformerProcessor` is a "parameter tunnel" | Forwarding shared kwargs into a `ModuleList` of `TransformerBlock`s is idiomatic; no indirection cost. |

---

## Active Issues, Ranked

_All previously-tracked issues below have been resolved (v2.9.6). They are kept
here for provenance; see "Resolution" notes._

### ✅ RESOLVED - Temperature-module interface mismatch in `SparseGraphAttention`
- **Where**: `components/attention.py`
- **Symptom**: The canonical temperature modules assume a `[B, H, N, G]` 4D
  logits tensor, but sparse attention produces `[E, H]`. The reshape lived
  inline in `forward`, coupling `SparseGraphAttention` to the temperature
  modules' internal layout.
- **Resolution**: Extracted `SparseGraphAttention._apply_temperature()`. The
  `[E, H] ↔ [1, H, E, 1]` layout adaptation now lives in a single documented
  method; `forward` just calls it.

### ✅ RESOLVED - Duplicated transformer constructor kwargs
- **Where**: `components/transformer.py`
- **Symptom**: `TransformerBlock` / `TransformerProcessor` each carried ~22
  duplicated constructor kwargs; adding a physics-token parameter meant
  touching both signatures, a `_defaults` dict, the `_PHYSICS_TOKEN_PARAMS`
  set, and the block-construction site.
- **Resolution**: Introduced `PhysicsTokenConfig` and `RelativePositionConfig`
  dataclasses. Both classes accept the config objects (legacy flat kwargs kept
  as a deprecation shim) and share one `_resolve_transformer_configs()` helper
  that merges sources, validates, and emits the ignored-parameter warning. The
  param-group sets are now derived from the dataclass fields, so they cannot
  drift.

### ✅ RESOLVED - `PhysicsTokenAttention` forward owned interleaved stages
- **Where**: `components/attention.py`
- **Symptom**: The forward path interleaved projection / slice / attend /
  deslice, and `PhysicsTokenAttentionV3` duplicated ~90% of that forward.
- **Resolution**: Extracted `_attend_tokens()` and `_deslice()` (mirroring the
  existing `_compute_slice_tokens()`), giving each stage a named, individually
  testable method. `PhysicsTokenAttentionV3` now inherits the base `forward()`
  via its polymorphic `_compute_slice_tokens` override, eliminating the
  duplication.

---

## Impact Assessment
After this round the transformer/attention stack has no outstanding correctness
bugs or tracked design smells. Each conceptual stage (slice / attend / deslice,
temperature adaptation, config resolution) now lives in a single named place.

