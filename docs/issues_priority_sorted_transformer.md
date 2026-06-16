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

### 🔧 MEDIUM - Genuine wart, low fix cost
**Issue**: Temperature-module interface mismatch in `SparseGraphAttention`
- **Where**: `components/attention.py:734-739`
- **Symptom**: The canonical temperature modules assume a `[B, H, N, G]` 4D
  logits tensor, but sparse attention produces `[E, H]`. The current code
  paper-masks this with `attn_scores.T.unsqueeze(0).unsqueeze(-1)` followed
  by a symmetric squeeze. This works but couples `SparseGraphAttention` to an
  internal layout assumption of every temperature module.
- **Fix**: Either (a) relax the temperature-module protocol to accept 2D
  `[E, H]` inputs natively, or (b) introduce a thin adapter inside
  `SparseGraphAttention` so the squeeze/unsqueeze dance lives in one named
  place.

### 📐 MEDIUM - Design smell, medium effort
**Issue**: `TransformerBlock` / `TransformerProcessor` carry ~22 constructor
kwargs each, all duplicated across the two classes.
- **Where**: `components/transformer.py:41-67`, `194-222`
- **Symptom**: Adding a new physics-token parameter requires touching both
  signatures, the `_defaults` dict in `TransformerBlock`, the
  `_PHYSICS_TOKEN_PARAMS` set, and the block-construction site in
  `TransformerProcessor`. Validation already exists, so this is no longer
  a correctness risk — just a maintenance tax.
- **Fix**: Bundle the physics-token kwargs into a `PhysicsTokenConfig`
  dataclass and the position-encoding kwargs into a
  `RelativePositionConfig`. Both classes then accept those objects; legacy
  kwargs can be retained as a deprecation shim for one release.

### 🧩 LOW - Optional polish, high effort
**Issue**: `PhysicsTokenAttention` is ~250 lines and still owns several
distinct responsibilities (QKV projection, slice-weight computation,
attention, deslice).
- **Where**: `components/attention.py:762-1011`
- **Symptom**: `TiledSliceOperation` was already extracted as a composed
  submodule, so the worst of the god-class problem is gone. What remains is
  a single forward path that interleaves projection / slice / attend /
  deslice — readable but not unit-testable in isolation.
- **Fix**: Optional further decomposition into `SliceProjector`,
  `TokenAttentionCore`, `DesliceProjector` classes — only worth doing if
  future variants (V4+) need to swap one stage independently.

---

## Quick Wins
1. Add a 2D-friendly adapter or relax the temperature protocol to clean up
   `SparseGraphAttention`'s reshape gymnastics.

## Strategic Refactors (only if maintenance cost grows)
1. Group the duplicated transformer kwargs into config dataclasses.
2. Further decompose `PhysicsTokenAttention` only when a new variant
   justifies it.

## Impact Assessment
After the prior purge of `QKVProjectionType.TOKEN_SLICE`, the
`AdaptiveTemperature` bug, and the addition of `TransformerBlock`'s
parameter-validation warnings, the transformer/attention stack has no
outstanding correctness bugs. The remaining items are design smells worth
tracking but not blocking.
