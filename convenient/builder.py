"""
Configuration builder for easy model instantiation.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

from .registry import AutoRegisterModel
from .training import Model
from .config import ModelConfig, TrainingConfig, FNOConfig, GNNConfig


class ConfigBuilder:
    """
    Builder for creating models from configurations.
    
    Example:
        config = GNNConfig(
            model_type='graphnet',
            node_in_dim=2,
            edge_in_dim=3,
            out_dim=1,
        )
        
        builder = ConfigBuilder(config)
        model = builder.build_model()
        
        # Or with training config
        train_config = TrainingConfig(learning_rate=1e-3)
        model, optimizer = builder.build_for_training(train_config)
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
    
    def build_model(self, device: Optional[str] = None) -> nn.Module:
        """
        Build model from configuration.
        
        Returns:
            Instantiated model
        """
        config = self.model_config
        model_type = config.model_type
        
        if isinstance(config, FNOConfig):
            model = self._build_fno(config)
        elif isinstance(config, GNNConfig):
            model = self._build_gnn(config)
        else:
            # Generic instantiation
            model = AutoRegisterModel.create(
                model_type,
                **config.dict(exclude={'model_type'})
            )
        
        if device:
            model = model.to(self._get_device(device))
        
        return model
    
    def build_for_training(
        self,
        train_config: TrainingConfig,
        device: Optional[str] = None,
    ) -> tuple:
        """
        Build model and optimizer for training.
        
        Returns:
            (model, optimizer) tuple
        """
        model = self.build_model(device)
        
        # Create optimizer
        optimizer_class = self._get_optimizer_class(train_config.optimizer)
        optimizer = optimizer_class(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        
        return model, optimizer
    
    def build_unified_model(
        self,
        train_config: TrainingConfig,
        device: Optional[str] = None,
    ) -> Model:
        """
        Build unified Model wrapper with architecture and optimizer.
        
        Returns:
            Unified Model instance ready for training
        """
        architecture, optimizer = self.build_for_training(train_config, device)
        
        return Model(
            architecture=architecture,
            loss_fn=train_config.loss_type,
            optimizer=optimizer,
        )
    
    def _build_fno(self, config: FNOConfig) -> nn.Module:
        """Build FNO model."""
        from ..models.fno_model import FNO, TFNO, AFNO
        
        if config.model_type == 'fno':
            return FNO(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                width=config.width,
                modes=config.modes,
                n_layers=config.n_layers,
                n_dim=config.n_dim,
            )
        elif config.model_type == 'tfno':
            return TFNO(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                width=config.width,
                modes=config.modes,
                n_layers=config.n_layers,
                n_dim=config.n_dim,
            )
        elif config.model_type == 'afno':
            return AFNO(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                width=config.width,
                modes=config.modes,
                n_layers=config.n_layers,
                n_dim=config.n_dim,
                num_blocks=config.num_blocks,
            )
        else:
            raise ValueError(f"Unknown FNO type: {config.model_type}")
    
    def _build_gnn(self, config: GNNConfig) -> nn.Module:
        """Build GNN model."""
        from ..components.decoders import MLPDecoder
        from ..components.transformer import TransformerProcessor
        from ..models.encode_process_decode import EncodeProcessDecode
        from ..models.gnn_model import GraphNet, MeshGraphNet, _MeshEncoder
        
        if config.model_type == 'graphnet':
            return GraphNet(
                node_in_dim=config.node_in_dim,
                edge_in_dim=config.edge_in_dim,
                out_dim=config.out_dim,
                latent_dim=config.latent_dim,
                n_layers=config.n_message_passing,
                hidden_dim=config.hidden_dim,
                activation=config.activation,
                residual=config.residual,
            )
        elif config.model_type == 'meshgraphnet':
            return MeshGraphNet(
                node_in_dim=config.node_in_dim,
                edge_in_dim=config.edge_in_dim,
                out_dim=config.out_dim,
                latent_dim=config.latent_dim,
                n_message_passing=config.n_message_passing,
                hidden_dim=config.hidden_dim,
                activation=config.activation,
            )
        elif config.model_type == 'transformer':
            # Build EPD with transformer processor
            encoder = _MeshEncoder(
                node_in_dim=config.node_in_dim,
                edge_in_dim=config.edge_in_dim,
                global_in_dim=None,
                latent_dim=config.latent_dim,
                hidden_dims=[config.hidden_dim],
                activation=config.activation,
            )
            
            processor = TransformerProcessor(
                node_dim=config.latent_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_physics_tokens=config.use_physics_tokens,
                n_tokens=config.n_tokens,
            )
            
            decoder = MLPDecoder(
                node_dim=config.latent_dim,
                out_dim=config.out_dim,
                hidden_dims=[config.hidden_dim],
                activation=config.activation,
            )
            
            return EncodeProcessDecode(encoder, processor, decoder)
        else:
            raise ValueError(f"Unknown GNN type: {config.model_type}")
    
    def _get_optimizer_class(self, name: str):
        """Get optimizer class by name."""
        optimizers = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
        }
        if name not in optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        return optimizers[name]
    
    def _get_device(self, device: str) -> torch.device:
        """Parse device string."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigBuilder':
        """Create builder from dictionary."""
        model_type = config_dict.get('model_type', 'graphnet').lower()
        
        if model_type in ['fno', 'tfno', 'afno']:
            config = FNOConfig(**config_dict)
        elif model_type in ['graphnet', 'meshgraphnet', 'transformer']:
            config = GNNConfig(**config_dict)
        else:
            config = ModelConfig(**config_dict)
        
        return cls(config)
