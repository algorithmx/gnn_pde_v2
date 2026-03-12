# Migration Completion Checklist

Last updated: 2026-03-12
Status: Historical migration record — v2.2 shim removal executed

## Purpose

This checklist distinguishes between:

- **Intentional compatibility shims** that may remain until v2.2
- **Actual unfinished migration work** where internal code, examples, or docs still assume the pre-v2.1 API

It is ordered by **risk first**, then **impact**.

## Priority Legend

- **P0** — High risk / high impact. Internal inconsistencies, stale code paths, or broken migration claims.
- **P1** — Medium risk / high user impact. Examples and docs that teach the wrong API.
- **P2** — Lower risk cleanup. Finish deprecations and simplify maintenance before v2.2.

---

## P0 — Fix internal migration inconsistencies first

### 1. config/config_builder.py
**Status:** Completed in current workspace state  
**Risk:** Very high  
**Impact:** Very high

**Why it matters**
- Still imports the old base-model path.
- Still assumes the old registry-bearing `BaseModel` API.
- This is more than a shim; it is stale implementation logic.

**Observed issues**
- Imports old path: [config/config_builder.py](config/config_builder.py#L9)
- Calls removed/old-style API: [config/config_builder.py](config/config_builder.py#L53)

**Required action**
- Stop using `core.base_model` assumptions.
- Replace implementation with one of these:
  1. **Preferred:** thin forwarding shim to [convenient/builder.py](convenient/builder.py)
  2. **Acceptable:** delete stale logic and re-export `ConfigBuilder` only
- Ensure no internal code path relies on `BaseModel.get_model()`.

**Completion criteria**
- No logic in this file depends on old registry behavior.
- File is a deprecation wrapper only, or removed in v2.2.

---

### 2. convenient/builder.py
**Status:** Completed in current workspace state  
**Risk:** High  
**Impact:** Very high

**Why it matters**
- This is the canonical high-level builder in v2.1.
- It still imports old namespace modules instead of canonical `components` modules.
- It reaches directly into registry internals.

**Observed issues**
- Direct registry access: [convenient/builder.py](convenient/builder.py#L53-L54)
- Old namespace imports: [convenient/builder.py](convenient/builder.py#L142-L143)

**Required action**
- Replace old-path imports with canonical imports from `components` where possible.
- Prefer registry API over `_registry` internals.
- Ensure unknown model handling uses one consistent error path.

**Completion criteria**
- No imports from deprecated namespaces remain.
- Builder does not depend on `_registry` implementation details.
- Error behavior for unknown models is explicit and stable.

---

### 3. models/encode_process_decode.py
**Status:** Completed in current workspace state  
**Risk:** Medium-high  
**Impact:** High

**Why it matters**
- The architecture has deprecated explicit protocols in favor of duck typing.
- This file still uses deprecated protocol annotations.

**Observed issues**
- Deprecated protocol references remain in constructor typing: [models/encode_process_decode.py](models/encode_process_decode.py#L36-L38)

**Required action**
- Replace deprecated protocol annotations with one of:
  1. `nn.Module`
  2. local non-deprecated `Protocol` definitions
  3. broader callable/module typing if needed
- Keep behavior unchanged.

**Completion criteria**
- No dependency on deprecated `core/protocols.py` from canonical internal code.

---

## P1 — Fix examples so they teach the new API

### 4. examples/example_meshgraphnets.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path: [examples/example_meshgraphnets.py](examples/example_meshgraphnets.py#L30)
- Old registration style: [examples/example_meshgraphnets.py](examples/example_meshgraphnets.py#L36)

**Required action**
- Convert to either:
  - lean example using `nn.Module` or lean `BaseModel`, or
  - convenience example using `AutoRegisterModel`
- Match the style documented in [MIGRATION.md](MIGRATION.md#L24-L81)

---

### 5. examples/example_deepxde.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style: [examples/example_deepxde.py](examples/example_deepxde.py#L30-L36)

**Required action**
- Same migration approach as item 4.

---

### 6. examples/example_transolver.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style: [examples/example_transolver.py](examples/example_transolver.py#L29-L33)

**Required action**
- Same migration approach as item 4.

---

### 7. examples/example_unisolver.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style: [examples/example_unisolver.py](examples/example_unisolver.py#L29-L38)

**Required action**
- Same migration approach as item 4.

---

### 8. examples/example_graph_pde_gno.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style: [examples/example_graph_pde_gno.py](examples/example_graph_pde_gno.py#L30-L34)

**Required action**
- Same migration approach as item 4.

---

### 9. examples/example_neuraloperator_fno.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style remain in multiple classes: [examples/example_neuraloperator_fno.py](examples/example_neuraloperator_fno.py#L28-L32) and [examples/example_neuraloperator_fno.py](examples/example_neuraloperator_fno.py#L301-L301)

**Required action**
- Same migration approach as item 4.
- Verify all model classes in the file use the chosen style consistently.

---

### 10. examples/example_windfarm_gno.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Observed issues**
- Old import path and old registration style: [examples/example_windfarm_gno.py](examples/example_windfarm_gno.py#L30-L35)

**Required action**
- Same migration approach as item 4.

---

## P1 — Fix documentation that still describes the old architecture

### 11. README.md
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** Very high

**Why it matters**
- README is the primary entry point.
- It still documents the old base-model/registry arrangement.

**Observed issues**
- Still claims `base.py` is for auto-registration: [README.md](README.md#L32)
- Still lists `base_model.py` as a current architecture file: [README.md](README.md#L33)

**Required action**
- Rewrite architecture tree to reflect canonical v2.1 structure.
- Mark compatibility modules as deprecated shims, not primary architecture.
- Align README language with [MIGRATION.md](MIGRATION.md#L18-L19) and [CHANGELOG.md](CHANGELOG.md#L32-L47)

**Completion criteria**
- A new reader following README uses only canonical imports.

---

### 12. CHANGELOG.md
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** High

**Why it matters**
- Changelog claims some migrations are already complete.

**Observed issues**
- Claim: “Updated all imports in models to use new structure”: [CHANGELOG.md](CHANGELOG.md#L32)
- That is true for some model modules, but not for examples and not consistently for adjacent high-level code.

**Required action**
- Refine wording so it distinguishes:
  - model internals migrated
  - compatibility layers retained
  - examples/docs still being finalized, if that remains true

**Completion criteria**
- Release notes match actual repository state.

---

### 13. MIGRATION.md
**Status:** Completed in current workspace state  
**Risk:** Low-medium  
**Impact:** High

**Why it matters**
- This file already describes the target state accurately.
- The problem is that repository examples do not yet consistently follow it.

**Observed anchor points**
- BaseModel import move: [MIGRATION.md](MIGRATION.md#L18-L19)
- New registration approach: [MIGRATION.md](MIGRATION.md#L135-L142)

**Required action**
- After code cleanup, re-check every example snippet against this guide.
- Update any examples in this guide only if the final chosen migration style changes.

---

## P2 — Convert deprecations into true shims and prepare v2.2 removal

### v2.2 removal result

The planned compatibility shims have now been removed from the workspace.
See [V2_2_REMOVAL_PLAN.md](V2_2_REMOVAL_PLAN.md) for the executed removal plan.

### 14. config/base_config.py
**Status:** Completed in current workspace state  
**Risk:** Medium  
**Impact:** Medium

**Why it matters**
- Duplicated implementation increases drift risk.

**Required action**
- Replace with forwarding shim to [convenient/config.py](convenient/config.py), or remove in v2.2.

---

### 15. encoders/__init__.py and similar deprecated package facades
**Status:** Intentional shim, but should remain shim-only  
**Risk:** Low-medium  
**Impact:** Medium

**Relevant file**
- [encoders/__init__.py](encoders/__init__.py#L1-L28)

**Required action**
- Confirm deprecated facades do not contain unique behavior.
- Keep warning text and forwarding only.
- Apply same rule to other deprecated namespaces before v2.2 removal.

---

### 16. core/protocols.py
**Status:** Completed in current workspace state  
**Risk:** Low-medium  
**Impact:** Medium

**Observed issues**
- Deprecated by design: [core/protocols.py](core/protocols.py#L4-L15)

**Required action**
- Once canonical code no longer references it, leave as compatibility-only until removal.
- Remove in v2.2 as planned.

---

### 17. utils/aggregation.py
**Status:** Completed in current workspace state  
**Risk:** Low-medium  
**Impact:** Medium

**Observed issues**
- Deprecated shim by design, forwarding to `core.functional` and optional convenient aggregation helpers.

**Required action**
- Keep shim-only behavior until v2.2.
- Remove in v2.2 once callers are migrated to [core/__init__.py](core/__init__.py) or [convenient/__init__.py](convenient/__init__.py).

---

## Suggested execution order

### Wave 1 — Canonical internals
1. [config/config_builder.py](config/config_builder.py)
2. [convenient/builder.py](convenient/builder.py)
3. [models/encode_process_decode.py](models/encode_process_decode.py)

### Wave 2 — User-facing correctness
4. [README.md](README.md)
5. [CHANGELOG.md](CHANGELOG.md)
6. All top-level example files under [examples](examples)

### Wave 3 — Deprecation cleanup
7. [config/base_config.py](config/base_config.py)
8. [encoders/__init__.py](encoders/__init__.py)
9. [core/protocols.py](core/protocols.py)

---

## Definition of done for the migration

The migration is effectively complete when all of the following are true:

- Canonical internal modules import only canonical v2.1 paths.
- Deprecated namespaces are forwarding shims only.
- No canonical module depends on deprecated `BaseModel` registry behavior.
- Examples use only the documented v2.1 styles.
- README and CHANGELOG describe the real architecture, not the pre-migration one.
- v2.2 can remove deprecated shims without breaking canonical code.
