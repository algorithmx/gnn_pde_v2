# Convenient Examples (High-Level API)

These examples demonstrate the **convenient high-level API**.

## Philosophy

- **Quick experimentation**: Less boilerplate, faster iteration
- **Config-driven**: Define models via configuration
- **String-based**: Use names instead of function references

## Features

- `AutoRegisterModel` - Auto-registration for config-based instantiation
- `ModelConfig` - Pydantic configuration with validation
- `ConfigBuilder` - Build models from config
- `get_initializer()` - String-based weight initialization

## Examples

### `meshgraphnets_easy.py`

MeshGraphNets with auto-registration and config support.

## Usage

```python
from gnn_pde_v2.convenient import (
    AutoRegisterModel, ModelConfig, ConfigBuilder, get_initializer
)

# Define model with auto-registration
@AutoRegisterModel(name='my_model')
class MyModel(nn.Module):
    ...

# Create via config
config = ModelConfig(model_type='my_model', hidden_dim=128)
model = ConfigBuilder(config).build_model()

# String-based initialization
init_fn = get_initializer('glorot_uniform')
```

## When to Use Convenient

- Quick prototyping
- Hyperparameter sweeps
- Teaching/demonstrations
- When you want sensible defaults

## Note

This API is **optional**. The framework works perfectly without it.
Import from `gnn_pde_v2.convenient` only if you want these features.
