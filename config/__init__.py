"""
Backward compatibility for config.

DEPRECATED: This module is deprecated and will be removed in v2.2.
Use gnn_pde_v2.convenient.config instead.
"""

import warnings

warnings.warn(
    "gnn_pde_v2.config is deprecated. "
    "Use gnn_pde_v2.convenient.config instead. "
    "This module will be removed in v2.2.",
    DeprecationWarning,
    stacklevel=2,
)

# Forward imports from new location
try:
    from ..convenient.config import (
        ConfigBase,
        ModelConfig,
        FNOConfig,
        GNNConfig,
        TrainingConfig,
        ExperimentConfig,
    )
except ImportError:
    ConfigBase = ModelConfig = FNOConfig = GNNConfig = TrainingConfig = ExperimentConfig = None

try:
    from ..convenient.builder import ConfigBuilder
except ImportError:
    ConfigBuilder = None

__all__ = [
    "ConfigBase",
    "ModelConfig",
    "FNOConfig",
    "GNNConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "ConfigBuilder",
]
