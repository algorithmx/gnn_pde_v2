# Remediation Plan — Issue #4: Structural graph/grid protocols carry no enforceable contract

> **Status:** PLANNED (not yet executed)
> **Scope:** `core/protocols.py` structural protocols only (clusters A/B/C/D + `Decoder` Union).
>   The three "distinct" protocols (`EdgeMessageProcessor`, `NodeUpdateStrategy`,
>   `EdgeFeatureAssembler`) are **OUT OF SCOPE** — they belong to issue #3
>   (dual Protocol+ABC extension mechanism).
> **Source issue:** `docs/architecture_issues.md` §4
> **Strategy:** Option 2 (drop `@runtime_checkable` from graph-world protocols) +
>   Option 3 (delete orphaned grid trio) + remove the deprecated `Decoder` Union.

---

## 0. Problem summary (verified)

`core/protocols.py` defines 11 `@runtime_checkable` protocols. Of these, 7
"stage" protocols collapse into 4 indistinguishable clusters because
`@runtime_checkable` only checks method **existence**, not signatures:

| Cluster | Protocols | Signature | `isinstance` can distinguish? |
|---------|-----------|-----------|------------------------------|
| A | `GraphEncoder`, `GraphProcessor` | `GraphsTuple → GraphsTuple` | No |
| B | `NodeDecoder`, `GraphModel` | `GraphsTuple → Tensor` | No |
| C | `PositionEncoder`, `GridProcessor`, `GridModel` | `Tensor → Tensor` | No — `nn.ReLU()` satisfies all three |
| D | `QueryDecoder` | `(GraphsTuple, Tensor, …) → Tensor` | Marginal — can't tell from NodeDecoder |

Consequence: these protocols are **decorative**. They promise runtime contract
enforcement they cannot deliver. The codebase already works around them via an
explicit `is_query_decoder` class attribute (the *real* discriminator that
`EncodeProcessDecode.forward` trusts).

---

## 1. Verified evidence (4 parallel explore agents + direct reads)

### 1a. Zero runtime discrimination in production code
A grep for `isinstance(... GraphEncoder|GraphProcessor|GraphModel|GridProcessor|GridModel|PositionEncoder|NodeDecoder|QueryDecoder)` across the **entire** codebase hits **zero** production call sites. The only `isinstance` occurrences are in `core/protocols.py` *docstring examples*. The sole runtime discriminator is:

```python
# models/encode_process_decode.py:81 — the REAL dispatch
if getattr(self.decoder, "is_query_decoder", False):
    output = self.decoder(processed, query_positions)
else:
    output = self.decoder(processed)
```

This is locked in by `tests/test_protocol_conformance.py::test_dispatch_uses_flag_not_isinstance`
(line 138–142), which asserts the source contains **no** `isinstance(self.decoder, QueryDecoder)`.

### 1b. Grid trio (Cluster C) is fully orphaned
- **Zero production imports** of `PositionEncoder` / `GridProcessor` / `GridModel`.
- **Zero test references** (`tests/test_fno.py` and all other test files: 0 hits).
- No grid model (`FNO`, `TFNO`, `AFNO`, `MultiscaleFNO`) imports them; no grid
  component (`spectral.py`, `FNOProcessor`, `AFNOBlock`) imports them.
- They exist only as: definitions in `core/protocols.py`, re-exports in
  `core/__init__.py` + `components/__init__.py`, README/CHANGELOG lines, and one
  docstring reference in `components/fourier_encoder.py:26` (no import).
- Combined with issue #2 (no grid model uses `EncodeProcessDecode`), they are
  documentation shell classes. → **Safe to delete.**

### 1c. Protocol consumer map (where each name appears)

| Protocol | Imports (re-export) | Type annotations | Runtime checks | Tests |
|----------|---------------------|------------------|----------------|-------|
| `GraphEncoder` | `core/__init__.py`, `components/__init__.py` | `encode_process_decode.py:45` | **0** | 0 |
| `GraphProcessor` | `core/__init__.py`, `components/__init__.py` | `encode_process_decode.py:46` | **0** (docstring only) | 0 |
| `NodeDecoder` | `core/__init__.py`, `components/__init__.py` | `encode_process_decode.py:47` | **0** | helper `_StrictNodeDecoder` |
| `QueryDecoder` | `core/__init__.py`, `components/__init__.py` | `encode_process_decode.py:47` | **0** | string-check in `test_dispatch_uses_flag_not_isinstance` |
| `GraphModel` | `core/__init__.py`, `components/__init__.py` | **0** | **0** | 0 |
| `PositionEncoder` | `core/__init__.py`, `components/__init__.py` | **0** | **0** | 0 |
| `GridProcessor` | `core/__init__.py`, `components/__init__.py` | **0** | **0** | 0 |
| `GridModel` | `core/__init__.py`, `components/__init__.py` | **0** | **0** | 0 |
| `Decoder` (Union) | `core/__init__.py`, `components/__init__.py` | `encode_process_decode.py:47` | **0** | 0 (deprecated) |

### 1d. Test impact preview
Only **4** test files reference any of the in-scope protocols, and the references
are docstrings/helpers/string-literals — **no test performs `isinstance` on any
cluster A/B/C/D protocol**. Therefore dropping `@runtime_checkable` breaks
**zero** tests in scope. (The 4 `isinstance` calls in `test_components.py:522–525`
target `EdgeMessageProcessor`, which is **out of scope** — issue #3.)

---

## 2. Remediation strategy

Apply the issue's own recommendation: **Option 2 + Option 3 + Decoder removal.**

| Target | Action | Rationale |
|--------|--------|-----------|
| `GraphEncoder`, `GraphProcessor` (A) | Drop `@runtime_checkable`; keep `Protocol` class | Used as EPD type hints → static-typing value remains; runtime promise was false |
| `NodeDecoder`, `QueryDecoder` (B/D) | Drop `@runtime_checkable`; keep `Protocol` class | Used as EPD type hints; dispatch already uses `is_query_decoder` flag |
| `GraphModel` (B) | Drop `@runtime_checkable`; keep `Protocol` class | Graph-world documentation hint (consistency with A); not orphaned-grid-tier |
| `PositionEncoder`, `GridProcessor`, `GridModel` (C) | **DELETE entirely** | Fully orphaned (§1b); re-add with distinct methods if a grid EPD ever materializes |
| `Decoder = Union[NodeDecoder, QueryDecoder]` | **DELETE** | Already marked deprecated; `isinstance(x, Decoder)` is true for any `nn.Module` |

**Why "drop `@runtime_checkable`" rather than "convert to `Callable` alias"?**
A `Protocol` class without `@runtime_checkable` still gives mypy/pyright full
structural static typing — strictly more value than a `Callable[[GraphsTuple], GraphsTuple]`
alias, at the same zero runtime cost. The only thing removed is the *ability* to
write `isinstance(x, GraphEncoder)`, which was always a false promise.

---

## 3. Step-by-step execution plan

All steps are mechanical and independently verifiable. Estimated effort: ~1–2 hours.

### Step 1 — Edit `core/protocols.py` (the core change)

**1a. Remove `@runtime_checkable` from the 5 retained graph-world protocols.**
Remove the decorator (and its blank-line handling) from:
- `GraphEncoder` (line 73)
- `GraphProcessor` (line 92)
- `NodeDecoder` (line 193)
- `QueryDecoder` (line 217)
- `GraphModel` (line 265)

Leave `from typing import … runtime_checkable` in place — it is still used by
`EdgeMessageProcessor`, `NodeUpdateStrategy`, `EdgeFeatureAssembler` (out of scope).

**1b. Delete the grid trio.**
Remove the entire definitions of:
- `PositionEncoder` (lines 288–296)
- `GridProcessor` (lines 299–310)
- `GridProcessor` (lines 313–322)
- The "Grid-world protocols" section header comment block (lines 277–286)

**1c. Delete the `Decoder` Union.**
Remove lines 251–262 (the deprecated comment block + `Decoder = Union[NodeDecoder, QueryDecoder]`).

**1d. Update `__all__` in `core/protocols.py`.**
Remove `"Decoder"`, `"PositionEncoder"`, `"GridProcessor"`, `"GridModel"` from
the `__all__` list (lines 325–343). Keep all graph-world names + the 3
distinct protocols + conditioning re-exports.

**1e. Rewrite the module docstring.**
- Remove the `assert isinstance(my_processor, GraphProcessor)` example (lines 13–16)
  and the `GraphProcessor` docstring `isinstance` example (lines 101–108).
- Remove the `EdgeFeatureAssembler` `isinstance` example (lines 178–184) — *note:
  this is out of scope for #4 but the example is misleading; flag for #3, leave
  for now to keep scope tight.* → **Decision: leave EdgeFeatureAssembler example
  untouched (issue #3 scope).**
- Update the "warning" section: the retained graph protocols are now
  **non-runtime-checkable** Protocols (static typing only) by design, not by
  limitation. Remove language implying they are `runtime_checkable`.
- Remove all references to the deleted grid trio and `Decoder` Union.

**1f. Update per-protocol docstrings.**
- `GraphEncoder` / `GraphProcessor` / `NodeDecoder` / `QueryDecoder` / `GraphModel`:
  remove `.. warning::` blocks that say "structurally identical at runtime" /
  "Use it as a static-typing hint only, never as an `isinstance` discriminator" —
  replace with a positive one-liner: "Static structural-typing hint; not
  `runtime_checkable` — dispatch on explicit attributes (e.g. `is_query_decoder`)."

### Step 2 — Update re-export barrels

**2a. `core/__init__.py`:**
- Remove `PositionEncoder`, `GridProcessor`, `GridModel`, `Decoder` from the
  `from .protocols import (...)` block (lines 25–36).
- Remove the same 4 names from `__all__` (lines 38–73).

**2b. `components/__init__.py`:**
- Remove `PositionEncoder`, `GridProcessor`, `GridModel`, `Decoder` from the
  `from ..core.protocols import (...)` block (lines 126–137).
- Remove the same 4 names from `__all__` (lines 229–239).
- Keep `GraphEncoder`, `GraphProcessor`, `NodeDecoder`, `QueryDecoder`,
  `GraphModel`, `EdgeMessageProcessor` re-exports.

### Step 3 — Update `models/encode_process_decode.py`

- Line 11 import: remove `QueryDecoder` is **kept** (still used for the type hint);
  actually the import `from ..core.protocols import GraphEncoder, GraphProcessor,
  NodeDecoder, QueryDecoder` stays as-is — all 4 are retained protocols.
- Line 47 annotation `decoder: Union[NodeDecoder, QueryDecoder]`: **keep** the
  inline `Union[NodeDecoder, QueryDecoder]` (do **not** import the deleted
  `Decoder` alias; the file already uses the inline `Union[...]` form, so no
  change needed here). **Verify** the file does not import `Decoder` by name —
  it does not (line 11 imports only the 4 protocols). ✓ No change required.

### Step 4 — Update `components/fourier_encoder.py`

- Line 26 docstring: remove the `:class:`~gnn_pde_v2.core.protocols.PositionEncoder``
  reST reference (the protocol no longer exists). Reword to "satisfies the
  `Tensor → Tensor` position-encoder contract" without citing the deleted protocol.

### Step 5 — Update tests

**5a. `tests/test_protocol_conformance.py`:**
- Line 142: `assert "isinstance(self.decoder, QueryDecoder)" not in src` — this
  is a **string literal**, not a name reference, so it still works after
  `@runtime_checkable` is removed. **No change needed.** ✓
- Lines 1–13 module docstring: update the reference to
  `docs/protocol_issues_2026_06.md` §1/§4 to note the protocols are now
  non-runtime-checkable by design.
- **ADD** a new regression test class `TestProtocolsNotRuntimeCheckable`:
  ```python
  class TestProtocolsNotRuntimeCheckable:
      """Issue #4: structural stage protocols must NOT be runtime_checkable.

      They carry no enforceable contract (signatures are unchecked) and were
      decorative. The real dispatch uses the is_query_decoder flag.
      """
      @pytest.mark.parametrize("name", [
          "GraphEncoder", "GraphProcessor", "NodeDecoder",
          "QueryDecoder", "GraphModel",
      ])
      def test_graph_protocols_not_runtime_checkable(self, name):
          from gnn_pde_v2.core import protocols
          proto = getattr(protocols, name)
          # A non-runtime-checkable Protocol is not usable with isinstance.
          with pytest.raises(TypeError):
              isinstance(object(), proto)

      def test_decoder_union_removed(self):
          from gnn_pde_v2.core import protocols
          assert not hasattr(protocols, "Decoder")

      def test_grid_trio_removed(self):
          from gnn_pde_v2.core import protocols
          for name in ("PositionEncoder", "GridProcessor", "GridModel"):
              assert not hasattr(protocols, name), f"{name} should be deleted"
  ```
  This locks in the remediation so it cannot silently regress.

**5b. Other test files:** No changes. (`test_components.py`'s `EdgeMessageProcessor`
isinstance checks are out of scope.)

### Step 6 — Update documentation

**6a. `README.md`:** In the "Protocols" table (around lines 119–127), remove the
rows for `Decoder`, `GridProcessor`, `GridModel`, `PositionEncoder`. Add a note
that the retained graph protocols are static-typing hints, not runtime contracts.

**6b. `docs/architecture_issues.md`:** After execution, prepend a RESOLVED block
to §4 mirroring the style of §1's RESOLVED block (date, summary, evidence pointer
to this plan). Move the current content under a "Historical (pre-fix) record"
header.

**6c. `docs/architecture-dependencies.md`:** Remove any "`EdgeFeatureAssembler`
protocol"/grid-protocol references that cite the deleted types (verify with grep
during execution).

**6d. `CHANGELOG.md`:** Add an entry under Unreleased describing the removal of
the decorative `@runtime_checkable` + deletion of the grid trio + `Decoder` Union,
with migration guidance (use `is_query_decoder` flag; use `Union[NodeDecoder, QueryDecoder]`
inline if the alias was imported).

### Step 7 — Backwards-compatibility shim (OPTIONAL, decide before executing)

The deleted `Decoder` alias and grid trio were importable from `gnn_pde_v2.core`
and `gnn_pde_v2.components`. If external/downstream code imports them, the import
breaks. Two options:

- **A (recommended for a research codebase):** No shim. The names were documented
  as deprecated/useless; a clean break is honest. Note in CHANGELOG.
- **B (if downstream consumers matter):** Add a thin compatibility module that
  raises `ImportError` with a clear message on import of the removed names, e.g.
  via module `__getattr__` in `core/protocols.py`:
  ```python
  def __getattr__(name):
      if name in {"Decoder", "PositionEncoder", "GridProcessor", "GridModel"}:
          raise ImportError(
              f"{name!r} was removed (issue #4): it carried no enforceable "
              f"contract. See docs/remediation-plan-issue4-structural-protocols.md."
          )
      raise AttributeError(name)
  ```
  Decision deferred to the user. **Default: Option A** (no shim).

---

## 4. Verification plan (must pass before declaring done)

1. **LSP diagnostics clean** on every edited file:
   `core/protocols.py`, `core/__init__.py`, `components/__init__.py`,
   `models/encode_process_decode.py`, `components/fourier_encoder.py`,
   `tests/test_protocol_conformance.py`.
2. **Import smoke test** under `conda run -n ml_env`:
   ```python
   import gnn_pde_v2
   from gnn_pde_v2.core import protocols
   from gnn_pde_v2.components import GraphNetProcessor, MLPDecoder, FNOBlock
   from gnn_pde_v2.models import EncodeProcessDecode, FNO
   assert not hasattr(protocols, "Decoder")
   assert not hasattr(protocols, "GridModel")
   ```
3. **Full test suite** under `conda run -n ml_env`:
   ```bash
   conda run -n ml_env pytest tests/ -q
   ```
   Must be all green, including the new `TestProtocolsNotRuntimeCheckable` class.
4. **Grep guard** — confirm no stale references to deleted names remain in
   production code:
   ```bash
   rg -n "PositionEncoder|GridProcessor|GridModel|\bDecoder\b" \
     --glob '!docs/**' --glob '!CHANGELOG.md' --glob '!*.md'
   ```
   Expected: zero hits outside docs/changelog (and the new regression test's
   string literals).
5. **Static-typing spot check** — confirm mypy/pyright still accept the EPD type
   hints (they will: non-runtime-checkable Protocols are still valid structural types).

---

## 5. Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| External code imports `Decoder`/grid trio | Low (research repo) | Import error | CHANGELOG note; optional `__getattr__` shim (Step 7B) |
| A test relies on `isinstance` against a retained protocol | Very Low | Test failure | Verified: zero such tests in scope (§1d) |
| Static type checkers regress | Very Low | Type hints weaker | Non-runtime-checkable Protocols retain full static value |
| Docstring/reST references break Sphinx build | Low | Broken links | Step 6 cleans all known references; grep guard catches rest |

**Net risk: LOW.** This is a subtractive change (removing false promises and dead
code) with exhaustive verification that nothing depends on the removed behaviour.

---

## 6. Out of scope (tracked separately)

- **Issue #3** — the Protocol+ABC duplication for `EdgeMessageProcessor`,
  `NodeUpdateStrategy`, `EdgeFeatureAssembler`. Those three stay
  `@runtime_checkable` for now; their fate (collapse to ABC only, or keep
  Protocol) is issue #3's decision.
- **Issue #2** — grid models not using `EncodeProcessDecode`. If a grid EPD is
  ever built, re-add grid protocols **with distinct method names** (the issue's
  Option 1) rather than the deleted decorative form.

---

## 7. Execution order (dependency-safe)

```
Step 1 (core/protocols.py)   ← central edit; do first
  └─ Step 5a (add regression test)  ← can run immediately to validate Step 1
Step 2 (barrel re-exports)    ← depends on Step 1 names existing/removed
Step 3 (EPD model)            ← verify only; no edit expected
Step 4 (fourier_encoder.py)   ← independent docstring fix
Step 5b (other tests)         ← no-op
Step 6 (docs)                 ← last; cosmetic
Step 7 (shim decision)        ← decide before Step 6 changelog entry
Verification (§4)             ← gate
```

Steps 1, 4, 5a are parallelizable; Steps 2 and 6 must follow Step 1.
