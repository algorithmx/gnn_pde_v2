"""
Convenient high-level API for GNN-PDE v2 (optional).

This module provides optional sugar for quick experimentation:
- Auto-registration for models (re-exported from core)
- Pydantic-based configuration
- String-based initialization

NOTE: Training utilities (Model, LossFunction) are in examples/training_utils.py.
For training wrappers, use:
    from gnn_pde_v2.examples.training_utils import Model, LossFunction

For lean usage, import directly from core and components:
    from gnn_pde_v2 import GraphsTuple, BaseModel
    from gnn_pde_v2.core import MLP, AutoRegisterModel
    from gnn_pde_v2.components import GraphNetBlock

Example with convenient API:
    from gnn_pde_v2.convenient import AutoRegisterModel, ModelConfig

    class MyModel(AutoRegisterModel, name='my_model'):
        ...

    # Create model via registry
    model = AutoRegisterModel.create('my_model', hidden_dim=128)
"""

# Re-export AutoRegisterModel from core for backwards compatibility
from gnn_pde_v2.core import AutoRegisterModel

# Optional imports - these modules handle their own dependencies gracefully
try:
    from .config import ModelConfig, TrainingConfig, FNOConfig, GNNConfig, ExperimentConfig
except ImportError:
    ModelConfig = TrainingConfig = FNOConfig = GNNConfig = ExperimentConfig = None

__all__ = [
    # Registry (re-exported from core)
    'AutoRegisterModel',
    # Config
    'ModelConfig',
    'TrainingConfig',
    'FNOConfig',
    'GNNConfig',
    'ExperimentConfig',
]
