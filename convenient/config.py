"""
Pydantic-based configuration system (optional convenience).

Note: This module requires pydantic. Install with: pip install pydantic
For lean usage without pydantic, use plain Python dataclasses.
"""

from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel as PydanticBaseModel, Field, validator


class ConfigBase(PydanticBaseModel):
    """Base configuration with validation."""
    
    class Config:
        validate_assignment = True
        extra = 'forbid'


class ModelConfig(ConfigBase):
    """Base model configuration."""
    
    model_type: str = Field(..., description="Model type identifier")
    hidden_dim: int = Field(default=128, ge=16, le=1024, description="Hidden dimension")
    n_layers: int = Field(default=4, ge=1, le=32, description="Number of layers")
    activation: Literal['relu', 'gelu', 'silu', 'tanh'] = Field(
        default='gelu', description="Activation function"
    )
    dropout: float = Field(default=0.0, ge=0.0, le=0.9, description="Dropout rate")
    
    @validator('model_type')
    def model_type_lowercase(cls, v):
        return v.lower()


class FNOConfig(ModelConfig):
    """FNO-specific configuration."""
    
    model_type: Literal['fno', 'tfno', 'afno'] = 'fno'
    width: int = Field(default=64, ge=16, le=512, description="FNO width")
    modes: List[int] = Field(default=[16, 16], description="Fourier modes per dimension")
    n_dim: int = Field(default=2, ge=1, le=3, description="Spatial dimension")
    in_channels: int = Field(default=1, ge=1, description="Input channels")
    out_channels: int = Field(default=1, ge=1, description="Output channels")
    use_afno: bool = Field(default=False, description="Use AFNO block-diagonal weights")
    num_blocks: int = Field(default=8, ge=1, description="Number of blocks for AFNO")
    
    @validator('modes')
    def validate_modes(cls, v, values):
        if 'n_dim' in values:
            if len(v) != values['n_dim']:
                raise ValueError(f"Number of modes ({len(v)}) must match n_dim ({values['n_dim']})")
        return v


class GNNConfig(ModelConfig):
    """GNN-specific configuration."""
    
    model_type: Literal['graphnet', 'meshgraphnet', 'transformer'] = 'graphnet'
    node_in_dim: int = Field(default=1, ge=1, description="Input node feature dimension")
    edge_in_dim: int = Field(default=3, ge=1, description="Input edge feature dimension")
    out_dim: int = Field(default=1, ge=1, description="Output dimension")
    latent_dim: int = Field(default=128, ge=16, description="Latent dimension")
    residual: bool = Field(default=True, description="Use residual connections")
    
    # Transformer-specific
    use_physics_tokens: bool = Field(default=False, description="Use physics token attention")
    n_tokens: int = Field(default=32, ge=4, description="Number of physics tokens")
    n_heads: int = Field(default=8, ge=1, description="Number of attention heads")


class TrainingConfig(ConfigBase):
    """Training configuration."""
    
    # Data
    batch_size: int = Field(default=32, ge=1, le=1024)
    num_workers: int = Field(default=4, ge=0, le=16)
    
    # Optimization
    optimizer: Literal['adam', 'adamw', 'sgd'] = 'adam'
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=1.0)
    scheduler: Optional[Literal['step', 'cosine', 'plateau']] = None
    scheduler_step: int = Field(default=100, ge=1)
    scheduler_gamma: float = Field(default=0.5, ge=0.1, le=1.0)
    
    # Training loop
    epochs: int = Field(default=100, ge=1)
    max_iterations: Optional[int] = Field(default=None, ge=1)
    
    # Loss
    loss_type: Literal['mse', 'l1', 'smooth_l1'] = 'mse'
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    checkpoint_freq: int = Field(default=10, ge=1)
    
    # Logging
    log_freq: int = Field(default=10, ge=1)
    
    # Device
    device: str = Field(default='auto', description="'auto', 'cpu', 'cuda', or 'cuda:N'")


class ExperimentConfig(ConfigBase):
    """Full experiment configuration."""
    
    model: ModelConfig
    training: TrainingConfig
    
    # Optional extra config
    data: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
