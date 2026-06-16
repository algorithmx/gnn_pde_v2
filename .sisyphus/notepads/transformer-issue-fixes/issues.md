# Issues

## [2026-05-12] Wave 1 Scope Creep
- Task 2 subagent scope-crept: restructured temperature hierarchy (ScheduledTemperature merge) which is Task 5 territory. Kept because tests pass and it's forward progress.
- Task 3 subagent scope-crept MASSIVELY: added TransformerConfig (Issue #1), rewrote both constructors, deleted example files, modified models/ and registry.py. REVERTED. Task 3 needs redo with strict scoping.


## 2026-05-12: Task 2 - conda run pytest issue

- `conda run -n ml_env pytest` picks up `/home/dabajabaza/.local/bin/pytest` (system Python) instead of the conda env's pytest
- Symptom: `ModuleNotFoundError: No module named 'torch'` even though torch is installed in ml_env
- Workaround: Use `conda run -n ml_env python -m pytest ...` or `conda run -n ml_env --cwd ... python -m pytest ...`
- Root cause: `~/.local/bin` appears first in PATH, and conda run doesn't fully isolate from user-local binaries
