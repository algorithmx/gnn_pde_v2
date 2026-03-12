# GNN-PDE v2.1 Migration Guide

> Note: As of v2.2, all deprecated v2.1 import shims have been removed.
> Use only the canonical imports shown in the “New Import” / “After” examples.

This guide helps you migrate from v2.0 to v2.1's new structure.

## Overview

Version 2.1 introduces a **lean core + convenient API** architecture:

- **`core/`** - Minimal, explicit (~250 lines)
- **`components/`** - Reusable building blocks (~700 lines)  
- **`convenient/`** - Optional high-level API (~400 lines)

## Quick Reference

| Old Import | New Import | Notes |
|------------|------------|-------|
| `from gnn_pde_v2 import BaseModel` | Same (now lean) | No auto-registration |
| `from gnn_pde_v2.core.base_model import BaseModel` | `from gnn_pde_v2.core.base import BaseModel` | Lean base class |
| `BaseModel with model_name='x'` | `from gnn_pde_v2.convenient import AutoRegisterModel` | Registry moved |
| `from gnn_pde_v2.encoders import MLP` | `from gnn_pde_v2.components import MLP` | Moved to components |
| `from gnn_pde_v2.processors import X` | `from gnn_pde_v2.components import X` | Moved to components |
| `from gnn_pde_v2.decoders import X` | `from gnn_pde_v2.components import X` | Moved to components |
| `from gnn_pde_v2.layers import X` | `from gnn_pde_v2.components import Residual` | Simplified |
| `from gnn_pde_v2.initializers import X` | `from gnn_pde_v2.convenient import X` | Optional now |
| `from gnn_pde_v2.utils.aggregation import X` | `from gnn_pde_v2.core import scatter_sum` | Core has essentials |
| `from gnn_pde_v2.config import X` | `from gnn_pde_v2.convenient import X` | Optional now |

## Migration Scenarios

### Scenario 1: I want the lean approach (recommended for research)

**Before:**
```python
from gnn_pde_v2 import BaseModel, GraphsTuple
from gnn_pde_v2.encoders import MLP, MLPMeshEncoder
from gnn_pde_v2.processors import GraphNetBlock

class MyModel(BaseModel, model_name='my_model'):
    def __init__(self):
        self.encoder = MLPMeshEncoder(...)
```

**After:**
```python
import torch.nn as nn
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import MLP, GraphNetBlock

class MyModel(nn.Module):  # Or BaseModel if you want the marker
    def __init__(self):
        self.node_encoder = MLP(...)
        self.edge_encoder = MLP(...)
```

### Scenario 2: I want the convenient API (quick experimentation)

**Before:**
```python
from gnn_pde_v2 import BaseModel
from gnn_pde_v2.config import ModelConfig, ConfigBuilder

class MyModel(BaseModel, model_name='my_model'):
    ...

config = ModelConfig(model_type='my_model')
model = ConfigBuilder(config).build_model()
```

**After:**
```python
from gnn_pde_v2.convenient import (
    AutoRegisterModel, ModelConfig, ConfigBuilder
)

class MyModel(AutoRegisterModel, name='my_model'):
    ...

config = ModelConfig(model_type='my_model')
model = ConfigBuilder(config).build_model()
```

### Scenario 3: I use string-based initializers

**Before:**
```python
from gnn_pde_v2.initializers import get_initializer

init = get_initializer('glorot_uniform')
```

**After:**
```python
# Option 1: Use convenient (optional dependency)
from gnn_pde_v2.convenient import get_initializer
init = get_initializer('glorot_uniform')

# Option 2: Use PyTorch directly (recommended)
import torch.nn.init as init
init.xavier_uniform_(tensor)
```

### Scenario 4: I use aggregation functions

**Before:**
```python
from gnn_pde_v2.utils.aggregation import scatter_sum, scatter_mean
```

**After:**
```python
# Option 1: Core has essentials
from gnn_pde_v2.core import scatter_sum, scatter_mean

# Option 2: Full API in convenient
from gnn_pde_v2.convenient import scatter_softmax, scatter_min

# Option 3: Use torch_scatter directly (recommended for production)
from torch_scatter import scatter
scatter(src, index, dim=0, reduce='sum')
```

## Deprecation Timeline

- **v2.1**: Old imports worked with `DeprecationWarning`
- **v2.2** (current): Old imports have been removed

## Key Changes

### BaseModel No Longer Has Auto-Registration

The lean `BaseModel` is now just a marker class. For auto-registration:

```python
# Old way (deprecated)
from gnn_pde_v2 import BaseModel
class MyModel(BaseModel, model_name='my_model'):
    ...

# New way
from gnn_pde_v2.convenient import AutoRegisterModel
class MyModel(AutoRegisterModel, name='my_model'):
    ...
```

### Components Moved

All building blocks are now in `gnn_pde_v2.components`:

```python
from gnn_pde_v2.components import (
    MLP,                    # Encoders
    Residual,               # Layers
    GraphNetBlock,          # Processors
    TransformerBlock,
    FNOProcessor,
    MLPDecoder,             # Decoders
    ProbeDecoder,
)
```

### Simplified Residual

Only one `Residual` class remains:

```python
from gnn_pde_v2.components import Residual

# Simple residual
block = Residual(nn.Sequential(...))

# With pre-normalization (Transformer-style)
block = Residual(
    module=nn.MultiheadAttention(128, 8),
    norm=nn.LayerNorm(128)
)
```

### MLP Uses Functional Init

```python
import torch.nn.init as init
from gnn_pde_v2.components import MLP

# Pass init functions directly
mlp = MLP(64, 64, [128], weight_init=init.kaiming_normal_)
```

## Benefits of Migrating

1. **Leaner imports**: Import only what you need
2. **Clearer dependencies**: Core has no optional deps
3. **Better performance**: Core uses torch_scatter when available
4. **More explicit**: Less magic, more understandable
5. **Flexibility**: Choose your level of abstraction

## Getting Help

- Check `examples/core/` for lean usage patterns
- Check `examples/convenient/` for high-level API usage
- File issues at: [your-repo-url]
