"""
Convenient high-level API for GNN-PDE v2 (optional).

This module provides optional sugar for quick experimentation:
- Auto-registration for models
- Pydantic-based configuration
- Unified training wrapper
- String-based initialization

For lean usage, import directly from core and components:
    from gnn_pde_v2 import GraphsTuple, BaseModel
    from gnn_pde_v2.components import MLP, GraphNetBlock

Example with convenient API:
    from gnn_pde_v2.convenient import (
        AutoRegisterModel, ModelConfig, ConfigBuilder, Model
    )
    
    class MyModel(AutoRegisterModel, name='my_model'):
        ...
    
    config = ModelConfig(model_type='my_model', hidden_dim=128)
    model = ConfigBuilder(config).build_model()
"""

# Optional imports - these modules handle their own dependencies gracefully
try:
    from .registry import AutoRegisterModel
except ImportError:
    AutoRegisterModel = None

try:
    from .config import ModelConfig, TrainingConfig, FNOConfig, GNNConfig, ExperimentConfig
except ImportError:
    ModelConfig = TrainingConfig = FNOConfig = GNNConfig = ExperimentConfig = None

try:
    from .builder import ConfigBuilder
except ImportError:
    ConfigBuilder = None

try:
    from .training import Model, LossFunction
except ImportError:
    Model = LossFunction = None

try:
    from .initializers import get_initializer, initialize_module
except ImportError:
    get_initializer = initialize_module = None

try:
    from .aggregation import scatter_softmax, scatter_min, segment_sum, segment_mean, segment_max, segment_min
except ImportError:
    scatter_softmax = scatter_min = segment_sum = segment_mean = segment_max = segment_min = None

__all__ = [
    # Registry
    'AutoRegisterModel',
    # Config
    'ModelConfig',
    'TrainingConfig',
    'FNOConfig',
    'GNNConfig',
    'ExperimentConfig',
    # Builder
    'ConfigBuilder',
    # Training
    'Model',
    'LossFunction',
    # Initializers
    'get_initializer',
    'initialize_module',
    # Aggregation
    'scatter_softmax',
    'scatter_min',
    'segment_sum',
    'segment_mean',
    'segment_max',
    'segment_min',
]
