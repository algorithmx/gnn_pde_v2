# Code Review: GNN-PDE v2 Framework

## Executive Summary

This is a well-designed GNN/PDE solver framework with clean abstractions. The
codebase demonstrates good software engineering practices (protocols,
composability, clean separation). Several design issues remain to be addressed
from both architectural and ML research perspectives.

---
## 1. Software Architecture Issues

### 1.1 Inconsistent Protocol Enforcement ✓ Fixed

**Location:** `core/protocols.py` vs component implementations

**Status:** Fixed in commit <TODO>

The `GraphEncoder`, `GraphProcessor`, and `Decoder` protocols use `@runtime_checkable`
but the codebase didn't consistently validate protocol compliance. The
`EncodeProcessDecode` class previously accepted `Union[Protocol, nn.Module]` which undermined
type safety.

**Fix Applied:**
1. Changed `EncodeProcessDecode.__init__()` to use strict protocol types:
   ```python
   def __init__(
       self,
       encoder: GraphEncoder,
       processor: GraphProcessor,
       decoder: Decoder,
   ):
   ```

2. Split `Decoder` protocol into `NodeDecoder` and `QueryDecoder` for type safety:
   - `NodeDecoder`: For fixed-node decoders (MLPDecoder, IndependentMLPDecoder)
   - `QueryDecoder`: For query-based decoders (ProbeDecoder)
   - `Decoder` is now a backwards-compatible Union type alias

**Benefits:**
- Static type checking now catches incompatible modules at analysis time
- Clear separation between decoder types prevents runtime errors
- All existing components satisfy their protocols via structural typing

### 1.2 Tight Coupling via Re-exports ✓ Fixed

**Location:** `components/__init__.py`

`Modulation` and `ConditioningProtocol` are no longer re-exported from the
`components` namespace. They must be imported directly from their canonical
location:

```python
from gnn_pde_v2.core.protocols import Modulation, ConditioningProtocol
```

The `components/__init__.py` comment was updated to make this explicit.

### 1.3 Missing Abstract Method in Protocol Hierarchy

**Location:** `core/protocols.py:66-107`

`ConditioningProtocol` inherits from both `ABC` and `nn.Module`. While this works,
it creates an unusual inheritance diamond when subclasses like
`AdaLNConditioning` also inherit from `nn.Module` implicitly:

```python
class ConditioningProtocol(nn.Module, ABC, Generic[CondT]):
    # ABC's metaclass + nn.Module's metaclass interaction is complex
```

### 1.4 Mutable Default Arguments via replace()

**Location:** `core/graph.py:79-81`

The `replace()` method uses `dataclasses.replace()` which creates new objects, but
the `GraphsTuple` fields themselves are mutable tensors. This can lead to subtle bugs
where "immutable" updates actually share tensor references.

---
## 2. ML Research Issues

### 2.1 Missing Normalization Layer Options ✓ Fixed

**Location:** `core/mlp.py:230-300`

**Status:** Resolved in v2.7.1. The MLP now supports multiple normalization types:
- `'layer'` — `LayerNorm` (existing)
- `'batch'` — `BatchNorm1d` (for stable training across batch sizes)
- `'instance'` — `InstanceNorm1d` (for style-transfer-like physics)
- `'group'` — `GroupNorm` (for small batch regimes)
- Dict specs for fine-grained control: `{'type': 'group', 'num_groups': 4}`

GroupNorm auto-adjusts `num_groups` when dimension is not divisible.

### 2.2 No Spectral Bias Mitigation

**Location:** `components/spectral.py`

The FNO implementation lacks common spectral bias mitigation techniques:
- No mode truncation schedule (gradual increase during training)
- No spectral normalization
- Fixed mode selection (no adaptive mode learning)

### 2.3 Attention Without Relative Position Encoding ✓ Fixed

**Location:** `components/transformer.py:154-203`

`MultiHeadAttention` uses standard absolute attention. For PDEs, relative
positions are crucial:

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # No relative position bias - problematic for physics domains
        scores = (q @ k.transpose(-2, -1)) / self.scale
```

**Impact:** The model cannot inherently distinguish spatial relationships, relying
entirely on learned positional embeddings.

### 2.4 No Multi-Scale Architecture Support ✓ Fixed

**Location:** Overall architecture

The framework lacks native support for multi-scale PDEs:
- No U-Net style skip connections
- No hierarchical graph pooling/unpooling
- No multi-resolution processing

This is critical for turbulence, multi-physics, and large-domain simulations.

### 2.5 PhysicsTokenAttention Slice Weight Temperature ✓ Fixed

**Location:** `components/transformer.py:408-530`, `components/temperature.py`

**Status:** Resolved - Implemented configurable multi-mode temperature system

The original implementation used a fixed temperature constant. The enhanced implementation now supports 5 temperature modes:

```python
from gnn_pde_v2.components import PhysicsTokenAttention

PhysicsTokenAttention(
    dim=128,
    n_tokens=32,
    n_heads=8,
    temperature_mode='adaptive',      # 'fixed', 'learnable_scalar', 'per_head', 'adaptive', 'annealed'
    use_gumbel_softmax=True,          # Optional Gumbel-Softmax reparameterization
    min_temperature=0.1,              # Prevent collapse to zero
    # Annealing-specific (for 'annealed' mode):
    anneal_warmup_epochs=5,
    anneal_factor=0.98,
    anneal_final_temp=0.05,
)
```

**Implementation Details:**

1. **Ada-Temp (Transolver++)**: `τ_i = τ_0 + Linear(x_i)` per mesh point
   - Dynamically sharpens/softens slice weight distributions based on local physics
   - Prevents state homogenization on large meshes
   - File: `components/temperature.py:135-160`

2. **Gumbel-Softmax (Transolver++)**: Differentiable sampling from categorical distribution
   - Applied as: `Softmax((Linear(x) - log(-log ε)) / τ)` following Eq. 4
   - Provides stochasticity during training for sharper state assignments

3. **Per-Head Temperature (Blog paper)**: Each head learns its own temperature
   - Heads × layers additional parameters
   - Allows different heads to have different attention sharpness
   - File: `components/temperature.py:122-133`

4. **Annealed Temperature (Low-Width Graph Transformers)**: Training schedule
   - Formula: `τ_epoch = max(f^(epoch-c), τ_min)`
   - Start with τ=1.0, anneal to 0.05 by end of training
   - Call `processor.set_epoch(epoch)` at the start of each epoch
   - File: `components/temperature.py:162-189`

5. **Safety Constraints**: All learnable temperatures use `clamp(min=0.1)` to prevent collapse

**Files Added/Modified:**
- **NEW**: `components/temperature.py` - Temperature mechanism implementations
- **MODIFY**: `components/transformer.py` - Enhanced PhysicsTokenAttention with temperature support
- **MODIFY**: `components/__init__.py` - Export new temperature classes

**Backward Compatibility:** Default `temperature_mode='fixed'` maintains original behavior.

### 2.6 FNO Lacks Zero-Shot Super-Resolution

**Location:** `components/spectral.py:462-544`

`FNOProcessor` doesn't implement the key FNO super-resolution capability
(evaluating at arbitrary resolution). The processor assumes fixed spatial
dimensions:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, in_channels, *spatial_dims] - fixed dims
```


---


| Issue                           | Severity | Impact | Effort | Status    |
|--------------------------------|----------|--------|--------|-----------|
| Spectral bias mitigation (2.2) | Low      | Medium | Medium | Pending   |
| Adaptive temperature (2.5)     | Low      | Low    | Low    | ✓ Fixed   |
| No multi-scale support (2.4)   | Medium   | High   | High   | ✓ Fixed   |
| Protocol inconsistency (1.1)   | Medium   | Low    | Medium | ✓ Fixed   |
| Missing normalization opts(2.1)| Medium   | Medium | Low    | ✓ Fixed   |
| Missing relative position (2.3)| Medium   | High   | Medium | ✓ Fixed   |

---
## Recommendations

1. **Long-term:** Super-resolution for FNO, adaptive spectral modes
