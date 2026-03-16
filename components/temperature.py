"""
Temperature mechanisms for attention-based models.

Implements various temperature scaling strategies for controlling
attention distribution sharpness in physics-informed neural networks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TemperatureBase(nn.Module, ABC):
    """Abstract base for temperature mechanisms."""
    
    @abstractmethod
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Input logits to scale
            features: Optional feature tensor for adaptive temperature
            
        Returns:
            Tuple of (temperature, scaled_logits)
        """
        pass


class FixedTemperature(TemperatureBase):
    """Fixed temperature (backward compatible)."""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.temperature, device=logits.device, dtype=logits.dtype), logits / self.temperature


class LearnableScalarTemperature(TemperatureBase):
    """Global learnable scalar temperature."""
    
    def __init__(self, init_temperature: float = 1.0, min_temp: float = 0.1):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        self.min_temp = min_temp
    
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        temperature = torch.exp(self.log_temperature).clamp_min(self.min_temp)
        return temperature, logits / temperature


class PerHeadTemperature(TemperatureBase):
    """Per-head learnable temperature (from blog paper)."""
    
    def __init__(self, n_heads: int, init_temperature: float = 1.0, min_temp: float = 0.1):
        super().__init__()
        self.log_temperatures = nn.Parameter(torch.full((n_heads,), init_temperature).log())
        self.min_temp = min_temp
    
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # logits: [B, H, N, G]
        temperatures = torch.exp(self.log_temperatures).clamp_min(self.min_temp)  # [H]
        temperatures = temperatures.view(1, -1, 1, 1)  # [1, H, 1, 1]
        return temperatures.mean(), logits / temperatures


class AdaptiveTemperature(TemperatureBase):
    """Per-point adaptive temperature (Ada-Temp from Transolver++).
    
    Formula: tau_i = tau_0 + Linear(x_i)
    """
    
    def __init__(self, feature_dim: int, init_temperature: float = 1.0, 
                 min_temp: float = 0.1, learnable_base: bool = True):
        super().__init__()
        if learnable_base:
            self.log_tau_0 = nn.Parameter(torch.log(torch.tensor(init_temperature)))
        else:
            self.register_buffer('tau_0', torch.tensor(init_temperature))
        self.temp_proj = nn.Linear(feature_dim, 1)
        self.min_temp = min_temp
        # Initialize projection to near zero for identity start
        nn.init.zeros_(self.temp_proj.weight)
        nn.init.zeros_(self.temp_proj.bias)
    
    def forward(self, logits: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features: [B, N, D]
        # logits: [B, H, N, G]
        tau_0 = torch.exp(self.log_tau_0) if hasattr(self, 'log_tau_0') else self.tau_0
        delta_tau = self.temp_proj(features).squeeze(-1)  # [B, N]
        temperatures = (tau_0 + delta_tau).clamp_min(self.min_temp)  # [B, N]
        temperatures = temperatures.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        return temperatures.mean(), logits / temperatures


class AnnealedTemperature(TemperatureBase):
    """Training-time temperature annealing (from Low-Width Graph Transformers).
    
    Schedule: tau_t = max(f^(t-c), tau_min)
    """
    
    def __init__(self, init_temperature: float = 1.0, final_temperature: float = 0.05,
                 warmup_epochs: int = 5, anneal_factor: float = 0.98):
        super().__init__()
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.warmup_epochs = warmup_epochs
        self.anneal_factor = anneal_factor
        self.current_epoch = 0
        self._current_temp = init_temperature
    
    def set_epoch(self, epoch: int):
        """Call at the start of each epoch."""
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            self._current_temp = self.init_temperature
        else:
            self._current_temp = max(
                self.anneal_factor ** (epoch - self.warmup_epochs),
                self.final_temperature
            )
    
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self._current_temp, device=logits.device, dtype=logits.dtype), logits / self._current_temp


def create_temperature_module(
    mode: str,
    dim: int = 128,
    n_heads: int = 8,
    temperature: float = 1.0,
    min_temperature: float = 0.1,
    anneal_warmup_epochs: int = 5,
    anneal_factor: float = 0.98,
    anneal_final_temp: float = 0.05,
) -> TemperatureBase:
    """
    Factory function to create temperature modules.
    
    Args:
        mode: One of 'fixed', 'learnable_scalar', 'per_head', 'adaptive', 'annealed'
        dim: Feature dimension (for adaptive mode)
        n_heads: Number of attention heads (for per_head mode)
        temperature: Initial temperature value
        min_temperature: Minimum temperature clamp value
        anneal_warmup_epochs: Warmup epochs for annealed mode
        anneal_factor: Annealing factor for annealed mode
        anneal_final_temp: Final temperature for annealed mode
        
    Returns:
        TemperatureBase instance
    """
    if mode == 'fixed':
        return FixedTemperature(temperature)
    elif mode == 'learnable_scalar':
        return LearnableScalarTemperature(temperature, min_temperature)
    elif mode == 'per_head':
        return PerHeadTemperature(n_heads, temperature, min_temperature)
    elif mode == 'adaptive':
        return AdaptiveTemperature(dim, temperature, min_temperature)
    elif mode == 'annealed':
        return AnnealedTemperature(temperature, anneal_final_temp, anneal_warmup_epochs, anneal_factor)
    else:
        raise ValueError(f"Unknown temperature mode: {mode}")
