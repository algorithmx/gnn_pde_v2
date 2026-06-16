## Manual QA Session - 2026-05-12

### Task 4 issue
- The original test command for Task 4 passed `edge_index` directly as second arg, but `SparseGraphAttention.forward(x, senders, receivers, edge_type=None)` requires keyword args `senders` and `receivers` separately.
- Fix: `attn(x, senders=edge_index[0], receivers=edge_index[1])`

### Test environment
- Need to `unset GTK_MODULES` to avoid GLib-GIO-WARNING spam in pytest
- conda env: ml_env
- torch_scatter not available; fallback to pure PyTorch (benign warning)
