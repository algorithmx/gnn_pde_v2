# Decisions

## [2026-05-12] Planning Session
- Test strategy: tests-after (add tests after each fix)
- Issue #5: method extraction (_compute_slice_tokens) not full composition
- Issue #6: minimal reshape fix + ValueError for adaptive mode
- Issue #9: forward(epoch=N) is read-only, does NOT update mutable state
- Issue #4: warn only for explicitly-passed non-default kwargs, single warnings.warn()
- PhysicalStateCache update: explicitly excluded (scope creep)
- Task 3: compare against `_defaults` dict, not just `is not None`, to avoid false positives from TransformerProcessor passing resolved defaults
- Task 3: module-level dedup via `_warned_blocks = set()` keyed by frozenset of ignored param names, NOT by instance id
- V3 tiling recompute anomaly: preserve exactly with NOTE comment
