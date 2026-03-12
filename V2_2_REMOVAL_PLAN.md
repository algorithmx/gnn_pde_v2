# v2.2 Deprecation Removal Plan

Last updated: 2026-03-12
Status: Executed in workspace
Scope: Remove v2.1 compatibility shims in v2.2 with minimal risk

## Goal

Remove deprecated import paths and migration shims that were intentionally
retained during v2.1, while preserving the canonical API:

- `core/` for lean primitives
- `components/` for canonical building blocks
- `convenient/` for optional high-level features

This plan assumes the current workspace state, where canonical internals and
examples have already been migrated away from deprecated paths.

Execution result: the listed compatibility shims were deleted and primary docs
were updated to reflect the post-shim canonical API.

---

## Removal candidates

These files are now compatibility-only and are the primary v2.2 removal set:

- [core/base_model.py](core/base_model.py)
- [core/protocols.py](core/protocols.py)
- [config/__init__.py](config/__init__.py)
- [config/base_config.py](config/base_config.py)
- [config/config_builder.py](config/config_builder.py)
- [encoders/__init__.py](encoders/__init__.py)
- [processors/__init__.py](processors/__init__.py)
- [decoders/__init__.py](decoders/__init__.py)
- [layers/__init__.py](layers/__init__.py)
- [initializers/__init__.py](initializers/__init__.py)
- [utils/aggregation.py](utils/aggregation.py)

---

## Pre-removal gate

Before deleting any shim, verify all of the following:

1. No canonical module imports the shim.
2. No top-level example imports the shim.
3. README and migration docs point only to canonical paths.
4. The replacement import path is already stable and public.
5. Release notes clearly announce the removal.

If any one of these is false, defer deletion for that file.

---

## Recommended removal waves

### Wave 1 — Safe namespace removals

These are the lowest-risk removals because they now exist only to forward
imports to canonical modules:

- [encoders/__init__.py](encoders/__init__.py)
- [processors/__init__.py](processors/__init__.py)
- [decoders/__init__.py](decoders/__init__.py)
- [layers/__init__.py](layers/__init__.py)
- [initializers/__init__.py](initializers/__init__.py)
- [utils/aggregation.py](utils/aggregation.py)

### Wave 2 — Old config surface removal

These should be removed together so there is no partial old config API left:

- [config/__init__.py](config/__init__.py)
- [config/base_config.py](config/base_config.py)
- [config/config_builder.py](config/config_builder.py)

### Wave 3 — Old base/protocol compatibility removal

These are conceptually important removals and should happen only after release
notes and migration docs are explicit:

- [core/base_model.py](core/base_model.py)
- [core/protocols.py](core/protocols.py)

---

## Canonical replacements to document in v2.2 notes

| Removed path | Canonical replacement |
|---|---|
| `gnn_pde_v2.core.base_model.BaseModel` | `gnn_pde_v2.core.base.BaseModel` |
| `gnn_pde_v2.config.*` | `gnn_pde_v2.convenient.*` |
| `gnn_pde_v2.encoders.*` | `gnn_pde_v2.components` or specific canonical modules |
| `gnn_pde_v2.processors.*` | `gnn_pde_v2.components` or specific canonical modules |
| `gnn_pde_v2.decoders.*` | `gnn_pde_v2.components` or specific canonical modules |
| `gnn_pde_v2.layers.*` | `gnn_pde_v2.components` |
| `gnn_pde_v2.initializers.*` | `gnn_pde_v2.convenient` or `torch.nn.init` |
| `gnn_pde_v2.utils.aggregation.*` | `gnn_pde_v2.core` or `gnn_pde_v2.convenient` |
| `gnn_pde_v2.core.protocols.*` | duck typing / `nn.Module` typing |

---

## Suggested execution checklist

### Step 1 — Freeze docs first
- Update [CHANGELOG.md](CHANGELOG.md) with a v2.2 removal section.
- Update [MIGRATION.md](MIGRATION.md) to state that deprecated v2.1 paths are now removed.
- Keep [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md) as the historical migration record.

### Step 2 — Delete shims by wave
- Delete Wave 1 files.
- Re-scan workspace for stale imports.
- Delete Wave 2 files.
- Re-scan workspace for stale imports.
- Delete Wave 3 files.

### Step 3 — Tighten package documentation
- Remove references to deprecated namespaces from [README.md](README.md).
- Remove references to compatibility-only behavior from [examples/README.md](examples/README.md).

### Step 4 — Final release pass
- Confirm public examples still import only canonical paths.
- Confirm the remaining package tree reflects the intended architecture.

---

## Risks to watch

### External user breakage
Even if the workspace is clean, external users may still import old paths.
Mitigation:
- Make removals explicit in release notes.
- Include a short replacement table.

### Hidden stale imports in docs or notebooks
Markdown and auxiliary docs can lag behind code.
Mitigation:
- Search all `*.py` and `*.md` files before release.

### Partial removal confusion
Removing only some of `config/` or only one of the old namespace facades can
leave users with a mixed mental model.
Mitigation:
- Remove related shims in grouped waves.

---

## Definition of done for v2.2

The v2.2 removal is complete when:

- All listed shim files are deleted.
- No workspace file imports deprecated v2.1 paths.
- Public docs reference only canonical imports.
- The architecture description matches the actual package tree.
- Users can migrate using the documented replacement table without ambiguity.

---

## Suggested follow-up after v2.2

After removal, the next cleanup target should be API hardening rather than
further migration work:

- introduce domain-specific exceptions
- consider a non-global registry abstraction for the convenient API
- improve config-level validation for transformer/FNO constraints
- simplify docs by removing all compatibility discussion
