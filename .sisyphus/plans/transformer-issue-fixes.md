# Fix Transformer Issues #3, #4, #5, #6, #8, #9

## TL;DR

> **Quick Summary**: Fix 6 specific code quality/bug issues in the transformer/attention/temperature modules — dead code removal, silent parameter validation, inheritance misuse via method extraction, temperature shape mismatch, hasattr fragility, and mutable state in nn.Module.
> 
> **Deliverables**:
> - Delete `QKVProjectionType.TOKEN_SLICE` dead code
> - Add parameter validation warnings to `TransformerBlock`
> - Extract `_compute_slice_tokens()` method to eliminate V3 code duplication
> - Fix temperature reshape bug in `SparseGraphAttention`
> - Replace `hasattr` with stored boolean in `AdaptiveTemperature`
> - Add `epoch` parameter to `ScheduledTemperature.forward()`
> - New test file `tests/test_sparse_graph_attention.py`
> - Regression test for V3 method extraction
> 
> **Estimated Effort**: Medium (6 focused tasks, ~90 min total)
> **Parallel Execution**: YES — 2 waves + final verification
> **Critical Path**: Task 1 → Task 3 → Task 6 → F1-F4

---

## Context

### Original Request
Fix 6 issues from `docs/issues_priority_sorted_transformer.md` that were verified as still present in the codebase. Issues #1 (parameter explosion), #2 (God Class decomposition), #7 (parameter tunnel) are explicitly excluded.

### Interview Summary
**Key Discussions**:
- Test strategy: **Tests-after** — add test tasks after each fix, rely on existing suite (~1424 lines) for regression
- Issue #5 approach: **Method extraction** — extract `_compute_slice_tokens()` in parent, V3 overrides it. Preserves `isinstance()` + attribute access from `PhysicalStateCache`.
- Issue #6 approach: **Minimal reshape fix** — correct the squeeze/unsqueeze to produce `[1, H, E, 1]` instead of `[1, 1, E, H]`. Raise clear error for architecturally incompatible `adaptive` mode.

**Research Findings**:
- Test infrastructure: pytest, comprehensive existing tests, conda env `ml_env`, no CI
- V3 has **ZERO dedicated tests** — only base PhysicsTokenAttention tested
- V3 has 4 instantiation sites, all in `examples/example_transolver_v3.py`
- `PhysicalStateCache` does `isinstance(V3)` + deep attribute access
- `TiledSliceOperation` is already a standalone nn.Module with clean API
- V3 forward() is ~90% copy-paste of parent's forward()
- `SparseGraphAttention` has **ZERO tests** — new test file required
- `adaptive` temperature mode is architecturally incompatible with SparseGraphAttention (per-node temp vs per-edge scores)

### Metis Review
**Identified Gaps** (all addressed):
- Issue #5 signature: `_compute_slice_tokens(self, x)` — single arg, computes shapes internally
- Issue #6 `adaptive` mode: Must raise `ValueError` at construction, not silently break
- Issue #5 V3 tiling recompute anomaly: Preserve exactly, flag with comment
- Issue #9 epoch=0 trap: Use `if epoch is not None:` not `if epoch:`
- Issue #4 warning scope: Warn only for explicitly-passed non-default kwargs
- Issue #4 warning format: Single `warnings.warn()` listing all ignored params
- PhysicalStateCache update: Explicitly excluded (scope creep flag)

---

## Work Objectives

### Core Objective
Fix 6 specific code quality/bug issues in transformer, attention, and temperature modules while preserving backward compatibility and passing the full existing test suite.

### Concrete Deliverables
- `components/attention.py`: TOKEN_SLICE deleted, `_compute_slice_tokens()` extracted, V3 override added, SparseGraphAttention reshape fixed
- `components/transformer.py`: Parameter validation warnings in TransformerBlock
- `components/temperature.py`: `self.learnable_base` stored, `epoch` param added to ScheduledTemperature.forward()
- `tests/test_sparse_graph_attention.py`: New test file for SparseGraphAttention
- Regression test for V3 output preservation

### Definition of Done
- [ ] `conda run -n ml_env pytest tests/ -x -q` → all pass
- [ ] `grep -rn "TOKEN_SLICE" --include="*.py" .` → zero results
- [ ] `grep -n "hasattr(self, 'log_tau_0')" components/temperature.py` → zero results
- [ ] V3 forward() contains `self._compute_slice_tokens` (no duplicated slice logic)
- [ ] `tests/test_sparse_graph_attention.py` exists and passes

### Must Have
- All 6 issues fixed with behavior preservation
- Full existing test suite passes after each fix
- Backward compatibility: state_dict keys unchanged, isinstance() checks preserved, set_epoch() still works
- New tests for previously untested code paths (SparseGraphAttention, V3 regression)

### Must NOT Have (Guardrails)
- MUST NOT touch issues #1, #2, #7 (explicitly excluded)
- MUST NOT delete entire `QKVProjectionType` class or `create_qkv_projection()` factory (out of scope)
- MUST NOT change state_dict key names (`log_tau_0`, `tau_0`)
- MUST NOT deprecate `set_epoch()` or migrate callers to `forward(epoch=...)`
- MUST NOT fix V3 tiling recompute anomaly (lines 1281-1288) — preserve behavior exactly
- MUST NOT refactor PhysicalStateCache to use `_compute_slice_tokens()` (scope creep)
- MUST NOT add 2D-native paths to temperature base class
- MUST NOT unify AdaptiveTemperature attribute names
- MUST NOT touch `docs/` files
- MUST NOT add dependencies or change imports beyond minimum needed
- MUST NOT use vague AI-slop comments — flag the V3 anomaly with a specific `# NOTE:` comment only

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, ~1424 lines)
- **Automated tests**: Tests-after — add tests after each fix
- **Framework**: pytest, conda env `ml_env`

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Library/Module**: Use Bash — run pytest, python -c for assertions
- **Regression**: Use Bash — compare outputs before/after with torch.allclose

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — trivial fixes, no dependencies):
├── Task 1: Delete TOKEN_SLICE dead code (Issue #3) [quick]
├── Task 2: Fix AdaptiveTemperature hasattr bug (Issue #8) [quick]
└── Task 3: Add parameter validation warnings (Issue #4) [unspecified-low]

Wave 2 (After Wave 1 — focused fixes with tests):
├── Task 4: Fix SparseGraphAttention temperature reshape + new tests (Issue #6) [unspecified-high]
├── Task 5: Add epoch parameter to ScheduledTemperature.forward() (Issue #9) [unspecified-low]
└── Task 6: Extract _compute_slice_tokens() + V3 override (Issue #5) [deep]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
→ Present results → Get explicit user okay

Critical Path: Task 1 → Task 4 → Task 6 → F1-F4
Parallel Speedup: ~40% faster than sequential
Max Concurrent: 3 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | - | - | 1 |
| 2 | - | - | 1 |
| 3 | - | - | 1 |
| 4 | - | - | 2 |
| 5 | - | - | 2 |
| 6 | - | - | 2 |
| F1-F4 | 1-6 | - | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 3 agents — T1 `quick`, T2 `quick`, T3 `unspecified-low`
- **Wave 2**: 3 agents — T4 `unspecified-high`, T5 `unspecified-low`, T6 `deep`
- **FINAL**: 4 agents — F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [x] 1. Delete `QKVProjectionType.TOKEN_SLICE` dead code (Issue #3)

  **What to do**:
  - In `components/attention.py`, delete line 28: `TOKEN_SLICE = "token_slice"  # Custom for token-based attention`
  - Verify with `lsp_find_references` that no code references `TOKEN_SLICE` before deleting
  - Run the full test suite to confirm no regressions

  **Must NOT do**:
  - Do NOT delete the entire `QKVProjectionType` class or `create_qkv_projection()` factory (out of scope)
  - Do NOT modify the `ValueError` in `create_qkv_projection()` or any other enum members
  - Do NOT update `docs/` files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single line deletion, zero risk, zero references
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/attention.py:22-30` — `QKVProjectionType` enum definition. TOKEN_SLICE is on line 28.

  **API/Type References**:
  - `components/attention.py:31-50` — `create_qkv_projection()` factory. It only handles COMBINED and SEPARATE, falling through to `ValueError` for anything else. TOKEN_SLICE would hit that ValueError, confirming it's unreachable.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TOKEN_SLICE is fully deleted
    Tool: Bash (grep)
    Preconditions: components/attention.py exists
    Steps:
      1. grep -rn "TOKEN_SLICE" --include="*.py" .
      2. Assert zero output
    Expected Result: grep returns exit code 1 (no matches found)
    Failure Indicators: Any line containing "TOKEN_SLICE" found
    Evidence: .sisyphus/evidence/task-1-token-slice-gone.txt

  Scenario: Full test suite passes after deletion
    Tool: Bash
    Preconditions: conda env ml_env available
    Steps:
      1. conda run -n ml_env pytest tests/ -x -q
      2. Assert exit code 0
    Expected Result: All tests pass, zero failures
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-1-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `fix(attention): remove unused TOKEN_SLICE enum member`
  - Files: `components/attention.py`
  - Pre-commit: `conda run -n ml_env pytest tests/ -x -q`

- [x] 2. Fix `AdaptiveTemperature` hasattr bug (Issue #8)

  **What to do**:
  - In `components/temperature.py`, `AdaptiveTemperature.__init__()` (around line 134):
    - Add `self.learnable_base = learnable_base` BEFORE the if/else block (store as plain instance attribute)
  - In `components/temperature.py`, `AdaptiveTemperature.forward()` (line 147):
    - Replace `hasattr(self, 'log_tau_0')` with `self.learnable_base`
    - The line becomes: `tau_0 = torch.exp(self.log_tau_0) if self.learnable_base else self.tau_0`
  - Run tests: `conda run -n ml_env pytest tests/test_temperature_modules.py tests/test_temperature_training.py tests/test_temperature_edge_cases.py tests/test_temperature_integration.py -x -q`

  **Must NOT do**:
  - Do NOT unify attribute names (`log_tau_0` / `tau_0`) — would break state_dict backward compatibility
  - Do NOT change state_dict key names
  - Do NOT rename or restructure the class

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Add one attribute, change one condition — 2 lines
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/temperature.py:125-150` — Full `AdaptiveTemperature` class. `__init__` at line 134, `forward` at line 147.
  - `core/registry.py:49,102,393` — Example `warnings.warn` patterns if needed for reference (not used in this task but shows codebase style)

  **API/Type References**:
  - `components/temperature.py:15` — `TemperatureBase(nn.Module, ABC)` base class. Forward signature is `forward(self, logits, features=None)` — do NOT change this.

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: learnable_base=True path works correctly
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import AdaptiveTemperature
attn = AdaptiveTemperature(feature_dim=64, learnable_base=True)
assert hasattr(attn, 'learnable_base') and attn.learnable_base is True
logits = torch.randn(1, 4, 10, 8)
temp, scaled = attn(logits)
assert temp.shape == logits.shape
assert not torch.isnan(temp).any()
print('PASS: learnable_base=True')
"
      2. Assert exit code 0 and output contains "PASS"
    Expected Result: learnable_base attribute stored, forward produces valid output
    Failure Indicators: AttributeError, NaN output
    Evidence: .sisyphus/evidence/task-2-learnable-true.txt

  Scenario: learnable_base=False path works correctly
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import AdaptiveTemperature
attn = AdaptiveTemperature(feature_dim=64, learnable_base=False)
assert attn.learnable_base is False
logits = torch.randn(1, 4, 10, 8)
temp, scaled = attn(logits)
assert temp.shape == logits.shape
assert not torch.isnan(temp).any()
print('PASS: learnable_base=False')
"
      2. Assert exit code 0
    Expected Result: learnable_base=False stored, forward uses tau_0 buffer
    Failure Indicators: AttributeError on tau_0, NaN output
    Evidence: .sisyphus/evidence/task-2-learnable-false.txt

  Scenario: state_dict keys unchanged (backward compat)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
from components.temperature import AdaptiveTemperature
attn_true = AdaptiveTemperature(feature_dim=64, learnable_base=True)
attn_false = AdaptiveTemperature(feature_dim=64, learnable_base=False)
sd_true = attn_true.state_dict()
sd_false = attn_false.state_dict()
assert 'log_tau_0' in sd_true, f'Expected log_tau_0, got {list(sd_true.keys())}'
assert 'tau_0' in sd_false, f'Expected tau_0, got {list(sd_false.keys())}'
assert 'learnable_base' not in sd_true, 'learnable_base should NOT be in state_dict'
print('PASS: state_dict keys correct')
"
      2. Assert exit code 0
    Expected Result: log_tau_0 and tau_0 keys preserved, learnable_base NOT in state_dict
    Failure Indicators: Missing keys or extra keys
    Evidence: .sisyphus/evidence/task-2-state-dict.txt

  Scenario: Existing temperature tests pass
    Tool: Bash (pytest)
    Preconditions: conda env ml_env
    Steps:
      1. conda run -n ml_env pytest tests/test_temperature_modules.py tests/test_temperature_training.py tests/test_temperature_edge_cases.py tests/test_temperature_integration.py -x -q
      2. Assert exit code 0
    Expected Result: All temperature tests pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-2-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `fix(temperature): replace hasattr with stored learnable_base flag`
  - Files: `components/temperature.py`
  - Pre-commit: `conda run -n ml_env pytest tests/test_temperature_modules.py -x -q`

- [x] 3. Add parameter validation warnings to TransformerBlock (Issue #4)

  **What to do**:
  - In `components/transformer.py`, `TransformerBlock.__init__()` (after parameter resolution, around line 160):
    - Add a validation block that checks for ignored parameters
    - When `use_physics_tokens=False`: if any physics-token-specific params were explicitly passed (non-sentinel), emit a single `warnings.warn()` listing all ignored params
    - When `use_physics_tokens=True`: if any position-encoding params were explicitly passed (non-sentinel), emit a single `warnings.warn()` listing all ignored params
  - Physics-token-specific params (ignored when `use_physics_tokens=False`): `n_tokens`, `temperature`, `temperature_mode`, `use_gumbel_softmax`, `min_temperature`, `anneal_warmup_epochs`, `anneal_factor`, `anneal_final_temp`, `use_slice_normalization`, `use_learnable_tokens`, `qkv_mode`, `use_orthogonal_init`
  - Position-encoding params (ignored when `use_physics_tokens=True`): `position_dim`, `max_distance`, `num_position_buckets`, `position_encoding_type`
  - Use `warnings.warn(msg, UserWarning, stacklevel=2)` following the pattern in `core/registry.py`
  - Use a module-level `_warned_blocks` set to prevent warning spam on repeated instantiation of identical configs
  - Update `tests/test_temperature_integration.py` line ~170: if the test creates a TransformerBlock with `use_physics_tokens=False` and passes ignored params, either wrap with `pytest.warns(UserWarning)` or use `warnings.filterwarnings("ignore")`

  **Must NOT do**:
  - Do NOT raise exceptions or change control flow — warnings are purely diagnostic
  - Do NOT validate params that ARE used — only warn about silently dropped ones
  - Do NOT add validation to TransformerProcessor (it delegates to TransformerBlock)
  - Do NOT warn when the ignored param has its default value (only warn for explicitly-passed non-defaults)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Add validation logic + update one test, moderate care needed
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `core/registry.py:49,102,393` — Existing `warnings.warn(msg, UserWarning, stacklevel=2)` patterns. Follow this exact style.
  - `components/transformer.py:67-160` — `TransformerBlock.__init__()` full constructor. Parameter resolution is the 3-tier system (explicit → config → default) at lines 95-134. Validation should go AFTER resolution but BEFORE the branch at line 163.

  **API/Type References**:
  - `components/transformer.py:22-57` — `TransformerConfig` dataclass. Contains defaults for all params.
  - `components/transformer.py:163-189` — Branching logic: lines 163-180 = `use_physics_tokens=True` path, lines 182-189 = `False` path. Params used in each branch are the ones that matter.

  **Test References**:
  - `tests/test_temperature_integration.py:~170` — Test `test_transformer_block_without_physics_tokens`. Currently has docstring "Should be ignored" — needs update to expect/suppress the warning.

  **WHY Each Reference Matters**:
  - The registry.py patterns show the canonical warning style — match it for consistency
  - The TransformerBlock constructor lines 95-134 show how params are resolved — the validation must check if the user explicitly passed a value (not just if it differs from default)
  - The branching at lines 163-189 shows exactly which params are used/ignored in each mode

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Warning fires when physics-token params passed with use_physics_tokens=False
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import warnings
from components.transformer import TransformerBlock
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    block = TransformerBlock(dim=64, n_heads=4, use_physics_tokens=False, n_tokens=16)
    assert len(w) >= 1, f'Expected warning, got {len(w)}'
    msg = str(w[0].message).lower()
    assert 'ignor' in msg or 'not used' in msg or 'incompatib' in msg, f'Unexpected message: {w[0].message}'
print('PASS: warning fires for ignored params')
"
      2. Assert exit code 0
    Expected Result: UserWarning emitted listing ignored params
    Failure Indicators: No warning raised, or wrong warning type
    Evidence: .sisyphus/evidence/task-3-warning-fires.txt

  Scenario: No warning when all params are defaults
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import warnings
from components.transformer import TransformerBlock
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    block = TransformerBlock(dim=64, n_heads=4, use_physics_tokens=False)
    # Filter to only UserWarning from our module
    relevant = [x for x in w if 'ignor' in str(x.message).lower() or 'not used' in str(x.message).lower()]
    assert len(relevant) == 0, f'Unexpected warning: {relevant}'
print('PASS: no false positives')
"
      2. Assert exit code 0
    Expected Result: No warnings for default-only params
    Failure Indicators: Warning raised for default params
    Evidence: .sisyphus/evidence/task-3-no-false-positive.txt

  Scenario: Warning fires for position params with use_physics_tokens=True
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import warnings
from components.transformer import TransformerBlock
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    block = TransformerBlock(dim=64, n_heads=4, use_physics_tokens=True, n_tokens=8, position_dim=3)
    assert len(w) >= 1, f'Expected warning for position_dim, got {len(w)} warnings'
print('PASS: warning for position params')
"
      2. Assert exit code 0
    Expected Result: UserWarning for position_dim being ignored
    Failure Indicators: No warning
    Evidence: .sisyphus/evidence/task-3-position-warning.txt

  Scenario: Updated test passes
    Tool: Bash (pytest)
    Preconditions: conda env ml_env
    Steps:
      1. conda run -n ml_env pytest tests/test_temperature_integration.py tests/test_components.py -x -q
      2. Assert exit code 0
    Expected Result: All tests pass including the updated test
    Failure Indicators: Test failure in updated test
    Evidence: .sisyphus/evidence/task-3-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `feat(transformer): add parameter validation warnings`
  - Files: `components/transformer.py`, `tests/test_temperature_integration.py`
  - Pre-commit: `conda run -n ml_env pytest tests/test_temperature_integration.py tests/test_components.py -x -q`

- [x] 4. Fix SparseGraphAttention temperature reshape + new tests (Issue #6)

  **What to do**:
  - In `components/attention.py`, `SparseGraphAttention.forward()` (lines 728-731):
    - Fix the reshape to produce `[1, H, E, 1]` (matching `[B, H, N, G]` convention) instead of `[1, 1, E, H]`
    - Change: `attn_scores.unsqueeze(0).unsqueeze(0)` → `attn_scores.T.unsqueeze(0).unsqueeze(-1)`
    - Change: `attn_scores_4d.squeeze(0).squeeze(0)` → `attn_scores_4d.squeeze(0).squeeze(-1).T`
    - This preserves the round-trip: `[E, H]` → `[1, H, E, 1]` → temperature module → `[1, H, E, 1]` → `[E, H]`
  - Add validation in `SparseGraphAttention.__init__()`: if `temperature_mode='adaptive'`, raise `ValueError` with clear message explaining the architectural incompatibility (per-node temperature features vs per-edge attention scores)
  - Create new file `tests/test_sparse_graph_attention.py` with tests for:
    - Construction with each temperature mode (`fixed`, `learnable_scalar`, `per_head`, `annealed`)
    - Forward pass with a small graph (5 nodes, 8 edges)
    - Verify output shape is correct
    - Verify no NaN/Inf in output
    - `adaptive` mode raises `ValueError` at construction
  - Run full test suite

  **Must NOT do**:
  - Do NOT add 2D-native paths to temperature base class (Approach B rejected)
  - Do NOT change the temperature module interface
  - Do NOT fix the deeper `adaptive` mode shape mismatch — just raise a clear error
  - Do NOT add tests for modes beyond the 5 temperature modes

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding 4D tensor conventions, creating a new test file, and handling an architectural incompatibility
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/attention.py:728-731` — The buggy squeeze/unsqueeze code. The comment says "Temperature module expects [B, H, N, G], we have [E, H]".
  - `tests/test_temperature_modules.py` — Example test file structure for temperature modules. Follow this pattern: class-based grouping, `assert` statements, `torch.manual_seed` for determinism.
  - `tests/conftest.py` — Shared fixtures. `sample_graph` provides a 5-node, 8-edge GraphsTuple that can be used directly.

  **API/Type References**:
  - `components/attention.py:587` — `SparseGraphAttention.__init__()`. Constructor signature includes `temperature_mode` parameter.
  - `components/temperature.py` — All temperature module classes. `FixedTemperature`, `LearnableScalarTemperature`, `PerHeadTemperature`, `AdaptiveTemperature`, `AnnealedTemperature`.

  **Test References**:
  - `tests/test_temperature_modules.py:TestFixedTemperature` — Example test class structure to follow.

  **WHY Each Reference Matters**:
  - The buggy lines 728-731 are the exact fix location — the reshape must map `[E, H]` to `[1, H, E, 1]` to match the `[B, H, N, G]` convention
  - The existing test patterns show how to construct test graphs and verify tensor shapes
  - The conftest fixtures provide pre-built graph objects for testing

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: per_head mode works without crash
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.attention import SparseGraphAttention
attn = SparseGraphAttention(dim=64, n_heads=4, temperature_mode='per_head')
edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)
x = torch.randn(5, 64)
out = attn(x, edge_index)
assert out.shape == (5, 64), f'Expected (5,64), got {out.shape}'
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
print('PASS: per_head mode')
"
      2. Assert exit code 0
    Expected Result: Valid output tensor with shape [5, 64], no NaN/Inf
    Failure Indicators: RuntimeError, NaN, Inf, wrong shape
    Evidence: .sisyphus/evidence/task-4-per-head.txt

  Scenario: adaptive mode raises clear error
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
from components.attention import SparseGraphAttention
try:
    attn = SparseGraphAttention(dim=64, n_heads=4, temperature_mode='adaptive')
    print('FAIL: should have raised ValueError')
except ValueError as e:
    assert 'adaptive' in str(e).lower() or 'per-node' in str(e).lower() or 'per-edge' in str(e).lower()
    print(f'PASS: correct error raised: {e}')
"
      2. Assert exit code 0
    Expected Result: ValueError with clear message about architectural incompatibility
    Failure Indicators: No error raised, or generic error message
    Evidence: .sisyphus/evidence/task-4-adaptive-error.txt

  Scenario: fixed mode produces same results (regression)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.attention import SparseGraphAttention
torch.manual_seed(42)
attn = SparseGraphAttention(dim=64, n_heads=4, temperature_mode='fixed')
edge_index = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)
x = torch.randn(5, 64)
out = attn(x, edge_index)
assert out.shape == (5, 64)
assert not torch.isnan(out).any()
print('PASS: fixed mode stable')
"
      2. Assert exit code 0
    Expected Result: Valid output, same as before the fix
    Failure Indicators: Shape mismatch, NaN
    Evidence: .sisyphus/evidence/task-4-fixed.txt

  Scenario: New test file passes
    Tool: Bash (pytest)
    Preconditions: conda env ml_env
    Steps:
      1. conda run -n ml_env pytest tests/test_sparse_graph_attention.py -v
      2. Assert exit code 0
    Expected Result: All tests in new file pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-4-new-tests.txt
  ```

  **Commit**: YES
  - Message: `fix(attention): correct temperature reshape in SparseGraphAttention`
  - Files: `components/attention.py`, `tests/test_sparse_graph_attention.py`
  - Pre-commit: `conda run -n ml_env pytest tests/test_sparse_graph_attention.py -v`

- [x] 5. Add epoch parameter to ScheduledTemperature.forward() (Issue #9)

  **What to do**:
  - In `components/temperature.py`, `ScheduledTemperature.forward()` (line 84):
    - Add optional parameter: `epoch: Optional[int] = None`
    - When `epoch is not None`: compute temperature on-the-fly using the same formula as `set_epoch()` (the annealing calculation)
    - When `epoch is None`: use existing `self._current_temp` (backward compatible)
    - IMPORTANT: Use `if epoch is not None:` NOT `if epoch:` — epoch=0 is a valid value
  - The computation logic (from `set_epoch()`) should be extracted into a private method `_compute_temperature(epoch)` that both `set_epoch()` and `forward()` can call
  - `forward(epoch=N)` must NOT update `self.current_epoch` or `self._current_temp` — it's a read-only override
  - `ScheduledTemperature` with `schedule='constant'` should return the initial temperature regardless of the epoch parameter
  - `AnnealedTemperature` inherits from `ScheduledTemperature` — no changes needed there
  - Run full test suite — existing tests for `current_epoch` and `_current_temp` must continue to pass (they use `set_epoch()` path)

  **Must NOT do**:
  - Do NOT deprecate `set_epoch()` or migrate any callers
  - Do NOT change any caller of `forward()` — no propagation up the call chain
  - Do NOT update `self.current_epoch` when `epoch` parameter is provided
  - Do NOT change the forward signature of any other temperature class (only `ScheduledTemperature`)
  - Do NOT use `if epoch:` — must use `if epoch is not None:` to handle epoch=0 correctly

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Add one optional parameter + extract a private method, moderate care for backward compat
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/temperature.py:33-90` — Full `ScheduledTemperature` class. `__init__` at line 39, `set_epoch()` at line 70, `forward()` at line 84. The annealing formula is in `set_epoch()` lines 76-80.
  - `components/temperature.py:154` — `AnnealedTemperature` class. Inherits from `ScheduledTemperature` with `schedule='scheduled'`.

  **API/Type References**:
  - `components/temperature.py:15` — `TemperatureBase(nn.Module, ABC)`. The abstract forward signature may need checking — if it defines `forward(self, logits, features=None)`, adding `epoch=None` is compatible since it's optional.

  **Test References**:
  - `tests/test_temperature_modules.py:255,259` — Tests that check `current_epoch` and `_current_temp` on `AnnealedTemperature`. These use `set_epoch()` and must continue to pass.
  - `tests/test_temperature_integration.py:232` — Integration test that calls `set_epoch()`.

  **WHY Each Reference Matters**:
  - The annealing formula in `set_epoch()` lines 76-80 must be duplicated (or extracted) for the on-the-fly computation in `forward()`
  - The existing tests for `current_epoch`/`_current_temp` verify backward compat — they must pass unchanged
  - `TemperatureBase` abstract signature may constrain the forward() parameter — need to check

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: forward(epoch=N) computes correct temperature without set_epoch
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import ScheduledTemperature
temp = ScheduledTemperature(temperature=1.0, schedule='scheduled',
                            final_temperature=0.05, warmup_epochs=5, anneal_factor=0.98)
logits = torch.randn(1, 4, 10, 8)
temp_val, scaled = temp(logits, epoch=20)
expected = max(0.98 ** (20 - 5), 0.05)
assert abs(temp_val.item() - expected) < 0.01, f'Expected {expected}, got {temp_val.item()}'
print(f'PASS: epoch=20 → temp={temp_val.item():.4f}')
"
      2. Assert exit code 0
    Expected Result: Temperature computed as 0.98^15 ≈ 0.738
    Failure Indicators: Wrong temperature value, AttributeError
    Evidence: .sisyphus/evidence/task-5-epoch-param.txt

  Scenario: forward(epoch=N) does NOT update mutable state
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import ScheduledTemperature
temp = ScheduledTemperature(temperature=1.0, schedule='scheduled',
                            final_temperature=0.05, warmup_epochs=5, anneal_factor=0.98)
logits = torch.randn(1, 4, 10, 8)
temp(logits, epoch=20)
assert temp.current_epoch == 0, f'current_epoch should be 0, got {temp.current_epoch}'
assert temp._current_temp == 1.0, f'_current_temp should be 1.0, got {temp._current_temp}'
print('PASS: mutable state unchanged')
"
      2. Assert exit code 0
    Expected Result: current_epoch=0, _current_temp=1.0 (unchanged)
    Failure Indicators: Mutable state was updated
    Evidence: .sisyphus/evidence/task-5-no-state-update.txt

  Scenario: set_epoch() still works (backward compat)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import ScheduledTemperature
temp = ScheduledTemperature(temperature=1.0, schedule='scheduled',
                            final_temperature=0.05, warmup_epochs=5, anneal_factor=0.98)
temp.set_epoch(20)
logits = torch.randn(1, 4, 10, 8)
temp_val, scaled = temp(logits)
expected = max(0.98 ** (20 - 5), 0.05)
assert abs(temp_val.item() - expected) < 0.01
assert temp.current_epoch == 20
print('PASS: set_epoch still works')
"
      2. Assert exit code 0
    Expected Result: Same temperature via set_epoch path
    Failure Indicators: Different temperature, AttributeError
    Evidence: .sisyphus/evidence/task-5-set-epoch-compat.txt

  Scenario: epoch=0 handled correctly (not treated as None)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.temperature import ScheduledTemperature
temp = ScheduledTemperature(temperature=1.0, schedule='scheduled',
                            final_temperature=0.05, warmup_epochs=5, anneal_factor=0.98)
logits = torch.randn(1, 4, 10, 8)
temp_val, scaled = temp(logits, epoch=0)
assert abs(temp_val.item() - 1.0) < 0.01, f'Expected 1.0 (warmup), got {temp_val.item()}'
print('PASS: epoch=0 handled correctly')
"
      2. Assert exit code 0
    Expected Result: Temperature = 1.0 (epoch 0 is within warmup period)
    Failure Indicators: Wrong temperature or epoch=0 treated as None
    Evidence: .sisyphus/evidence/task-5-epoch-zero.txt

  Scenario: Existing tests pass unchanged
    Tool: Bash (pytest)
    Preconditions: conda env ml_env
    Steps:
      1. conda run -n ml_env pytest tests/test_temperature_modules.py tests/test_temperature_integration.py tests/test_temperature_training.py tests/test_temperature_edge_cases.py tests/test_temperature_research.py -x -q
      2. Assert exit code 0
    Expected Result: All temperature tests pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-5-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `feat(temperature): add epoch parameter to ScheduledTemperature.forward()`
  - Files: `components/temperature.py`
  - Pre-commit: `conda run -n ml_env pytest tests/test_temperature_modules.py tests/test_temperature_integration.py -x -q`

- [x] 6. Extract `_compute_slice_tokens()` method + V3 override (Issue #5)

  **What to do**:
  - **Step 1: Create snapshot test BEFORE extraction**
    - Create a regression test: instantiate `PhysicsTokenAttentionV3` with fixed seed, capture forward output
    - This test will verify output preservation after the refactoring
    - Test should cover both `use_tiling=True` and `use_tiling=False` paths
  - **Step 2: Extract `_compute_slice_tokens(self, x)` in `PhysicsTokenAttention`**
    - Extract the slice phase from `forward()` (lines ~904-937) into a protected method
    - Signature: `def _compute_slice_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:`
    - Returns: `(tokens, slice_weights)` — same shapes as before
    - The method computes shapes internally from `x.shape` and `self` attributes
    - The extracted section includes: input projection, slice weight computation, Gumbel-Softmax, temperature application, softmax, einsum aggregation, slice normalization
  - **Step 3: Update `PhysicsTokenAttention.forward()`**
    - Replace the extracted slice section with a call to `self._compute_slice_tokens(x)`
    - The rest of forward() (attention + deslice) remains unchanged
  - **Step 4: Override `_compute_slice_tokens()` in `PhysicsTokenAttentionV3`**
    - V3's override checks `self.use_tiling`:
      - If tiling is active and N > tile_size: delegate to `self.tiled_slice(...)`, preserving the exact `use_slice_normalization` behavior (including the suspicious recompute anomaly at lines ~1281-1288)
      - Otherwise: `return super()._compute_slice_tokens(x)` — NO duplicated base class code
    - Add a `# NOTE:` comment on the recompute anomaly: `# NOTE: Preserved existing V3 behavior — recompute for non-normalized tiling path`
  - **Step 5: Update V3.forward()**
    - Replace the duplicated slice section with `self._compute_slice_tokens(x)`
    - The attention + deslice sections remain (they're identical to base class)
  - **Step 6: Verify snapshot test passes**
    - Run the regression test created in Step 1
    - Output must match: `torch.allclose(output_before, output_after, atol=1e-6)`
  - Run full test suite

  **Must NOT do**:
  - Do NOT fix the V3 tiling recompute anomaly — preserve behavior exactly with a NOTE comment
  - Do NOT refactor PhysicalStateCache to use `_compute_slice_tokens()` (scope creep — flag as recommended follow-up)
  - Do NOT change any public API — `_compute_slice_tokens` is protected (single underscore)
  - Do NOT change V3's tiling behavior or the tiling/non-tiling decision logic
  - Do NOT add V3 to any factory/registry
  - Do NOT create SliceOperation/TokenAttention/DesliceOperation classes (that's Issue #2, excluded)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex refactoring requiring careful line-by-line code extraction, method design, and regression testing across two classes
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Nothing
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `components/attention.py:754-977` — Full `PhysicsTokenAttention` class. `__init__` at line ~810, `forward()` at line ~884. The slice section to extract is lines ~904-937.
  - `components/attention.py:1170-1337` — Full `PhysicsTokenAttentionV3` class. `__init__` at line ~1199, `forward()` at line ~1245. The V3-specific slice section is lines ~1267-1301.
  - `components/attention.py:984-1167` — `TiledSliceOperation`. Already a standalone nn.Module with clean API: `forward(fx_mid, x_mid, slice_weight_proj, temperature_module, ...)`.

  **API/Type References**:
  - `examples/example_transolver_v3.py:51` — V3 instantiation site in `TransolverV3Block.__init__`. Must remain functional.
  - `examples/example_transolver_v3.py:354` — `isinstance(module, PhysicsTokenAttentionV3)` check in `PhysicalStateCache`. Must still return True.
  - `examples/example_transolver_v3.py:369-377` — Deep attribute access: `attn_module.n_heads`, `.n_tokens`, `.head_dim`, `.in_project_x`, `.in_project_fx`, `.slice_weight_proj`, `.temperature_module`. All must remain accessible.
  - `components/__init__.py:97,188` — Public export of `PhysicsTokenAttentionV3`.

  **Test References**:
  - `tests/test_temperature_integration.py` — Integration tests for PhysicsTokenAttention. These test the base class and should continue to pass.
  - `tests/test_components.py` — Component tests including transformer blocks. Should continue to pass.

  **WHY Each Reference Matters**:
  - The slice section lines ~904-937 are the exact code to extract — must be moved verbatim into `_compute_slice_tokens()`
  - The V3 forward() lines ~1245-1337 contain ~90% duplicated code — after extraction, only the `_compute_slice_tokens` override and the attention/deslice call should remain
  - The PhysicalStateCache attribute access is the constraint that prevents full composition — these attributes must stay on the V3 instance
  - The TiledSliceOperation API shows what parameters the tiling path needs — the V3 override must pass these correctly

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: _compute_slice_tokens method exists on base class
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import inspect
from components.attention import PhysicsTokenAttention, PhysicsTokenAttentionV3
assert hasattr(PhysicsTokenAttention, '_compute_slice_tokens')
sig = inspect.signature(PhysicsTokenAttention._compute_slice_tokens)
params = list(sig.parameters.keys())
assert 'x' in params, f'Expected x param, got {params}'
print(f'PASS: method exists with params {params}')
"
      2. Assert exit code 0
    Expected Result: Method exists with signature (self, x)
    Failure Indicators: AttributeError
    Evidence: .sisyphus/evidence/task-6-method-exists.txt

  Scenario: V3 overrides _compute_slice_tokens (not inherited)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
from components.attention import PhysicsTokenAttention, PhysicsTokenAttentionV3
assert PhysicsTokenAttentionV3._compute_slice_tokens is not PhysicsTokenAttention._compute_slice_tokens
print('PASS: V3 has its own override')
"
      2. Assert exit code 0
    Expected Result: V3 has a distinct override (not inherited from base)
    Failure Indicators: V3 uses base class method directly
    Evidence: .sisyphus/evidence/task-6-v3-override.txt

  Scenario: Regression test — V3 output preserved (no tiling)
    Tool: Bash (python)
    Preconditions: conda env ml_env, snapshot taken BEFORE extraction
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.attention import PhysicsTokenAttentionV3
torch.manual_seed(42)
v3 = PhysicsTokenAttentionV3(dim=64, n_heads=4, n_tokens=8, use_tiling=False)
x = torch.randn(2, 32, 64)
out = v3(x)
assert out.shape == (2, 32, 64), f'Wrong shape: {out.shape}'
assert not torch.isnan(out).any(), 'NaN in output'
assert not torch.isinf(out).any(), 'Inf in output'
print(f'PASS: no-tiling output shape={out.shape}, mean={out.mean().item():.4f}, std={out.std().item():.4f}')
"
      2. Assert exit code 0
    Expected Result: Valid output, no NaN/Inf, correct shape
    Failure Indicators: Shape mismatch, NaN, Inf
    Evidence: .sisyphus/evidence/task-6-regression-no-tiling.txt

  Scenario: Regression test — V3 output preserved (with tiling)
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
import torch
from components.attention import PhysicsTokenAttentionV3
torch.manual_seed(42)
v3 = PhysicsTokenAttentionV3(dim=64, n_heads=4, n_tokens=8, use_tiling=True)
v3.eval()
x = torch.randn(2, 200, 64)  # N=200 > default tile_size
out = v3(x)
assert out.shape == (2, 200, 64), f'Wrong shape: {out.shape}'
assert not torch.isnan(out).any(), 'NaN in output'
print(f'PASS: tiling output shape={out.shape}')
"
      2. Assert exit code 0
    Expected Result: Valid output with tiling, no NaN
    Failure Indicators: Shape mismatch, NaN (tiling path broken)
    Evidence: .sisyphus/evidence/task-6-regression-tiling.txt

  Scenario: isinstance(V3) check preserved for PhysicalStateCache compat
    Tool: Bash (python)
    Preconditions: conda env ml_env
    Steps:
      1. Run: conda run -n ml_env python -c "
from components.attention import PhysicsTokenAttention, PhysicsTokenAttentionV3
v3 = PhysicsTokenAttentionV3(dim=64, n_heads=4, n_tokens=8)
assert isinstance(v3, PhysicsTokenAttention), 'V3 must be instance of base class'
assert isinstance(v3, PhysicsTokenAttentionV3), 'V3 must be instance of V3'
# Check PhysicalStateCache attribute access
assert hasattr(v3, 'n_heads')
assert hasattr(v3, 'n_tokens')
assert hasattr(v3, 'head_dim')
assert hasattr(v3, 'in_project_x')
assert hasattr(v3, 'in_project_fx')
assert hasattr(v3, 'slice_weight_proj')
assert hasattr(v3, 'temperature_module')
print('PASS: isinstance + attribute access preserved')
"
      2. Assert exit code 0
    Expected Result: All attribute checks pass, isinstance works
    Failure Indicators: Missing attribute, isinstance fails
    Evidence: .sisyphus/evidence/task-6-isinstance-compat.txt

  Scenario: Full test suite passes
    Tool: Bash (pytest)
    Preconditions: conda env ml_env
    Steps:
      1. conda run -n ml_env pytest tests/ -x -q
      2. Assert exit code 0
    Expected Result: All tests pass
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-6-tests-pass.txt
  ```

  **Commit**: YES
  - Message: `refactor(attention): extract _compute_slice_tokens method`
  - Files: `components/attention.py`
  - Pre-commit: `conda run -n ml_env pytest tests/ -x -q`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `conda run -n ml_env pytest tests/ -v`. Review all changed files for: `as any`, empty catches, `console.log` equivalents (print statements in non-debug code), commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration: TransformerBlock with physics tokens + validation warnings, SparseGraphAttention with all temperature modes, V3 forward() with and without tiling. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Verify issues #1, #2, #7 were NOT touched. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Task 1**: `fix(attention): remove unused TOKEN_SLICE enum member` — components/attention.py
- **Task 2**: `fix(temperature): replace hasattr with stored learnable_base flag` — components/temperature.py
- **Task 3**: `feat(transformer): add parameter validation warnings` — components/transformer.py
- **Task 4**: `fix(attention): correct temperature reshape in SparseGraphAttention` — components/attention.py, tests/test_sparse_graph_attention.py
- **Task 5**: `feat(temperature): add epoch parameter to ScheduledTemperature.forward()` — components/temperature.py
- **Task 6**: `refactor(attention): extract _compute_slice_tokens method` — components/attention.py

---

## Success Criteria

### Verification Commands
```bash
# Full test suite passes
conda run -n ml_env pytest tests/ -x -q
# Expected: all pass, zero failures

# Issue #3: TOKEN_SLICE gone
grep -rn "TOKEN_SLICE" --include="*.py" .
# Expected: zero output

# Issue #8: hasattr gone
grep -n "hasattr(self, 'log_tau_0')" components/temperature.py
# Expected: zero output

# Issue #5: V3 uses extracted method
grep -n "_compute_slice_tokens" components/attention.py
# Expected: at least 3 lines (base method definition, base forward() call, V3 override)

# Issue #6: New test file exists and passes
conda run -n ml_env pytest tests/test_sparse_graph_attention.py -v
# Expected: all pass
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Issues #1, #2, #7 untouched
- [ ] No state_dict key names changed
- [ ] set_epoch() still works
- [ ] isinstance(V3) checks preserved
