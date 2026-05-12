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


class ScheduledTemperature(TemperatureBase):
    """Temperature with constant or scheduled (annealed) decay.
    
    Merges the former FixedTemperature and AnnealedTemperature into a single
    class controlled by the ``schedule`` parameter.
    
    Modes:
        'constant':  Fixed temperature value throughout training.
        'scheduled': Exponential decay after warmup: tau_t = max(f^(t-c), tau_min)
    
    Args:
        temperature: Temperature value (initial value for scheduled mode).
        schedule: 'constant' or 'scheduled'.
        final_temperature: Minimum temperature for scheduled mode.
        warmup_epochs: Epochs before annealing starts for scheduled mode.
        anneal_factor: Exponential decay factor for scheduled mode.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        schedule: str = 'constant',
        final_temperature: float = 0.05,
        warmup_epochs: int = 5,
        anneal_factor: float = 0.98,
    ):
        super().__init__()
        if schedule not in ('constant', 'scheduled'):
            raise ValueError(f"schedule must be 'constant' or 'scheduled', got '{schedule}'")
        self.temperature = temperature
        self.schedule = schedule
        self.final_temperature = final_temperature
        self.warmup_epochs = warmup_epochs
        self.anneal_factor = anneal_factor
        self.current_epoch = 0
        self._current_temp = temperature
    
    def _compute_temperature(self, epoch: int) -> float:
        if self.schedule == 'constant':
            return self.temperature
        if epoch < self.warmup_epochs:
            return self.temperature
        return max(
            self.anneal_factor ** (epoch - self.warmup_epochs),
            self.final_temperature
        )

    def set_epoch(self, epoch: int):
        """Update temperature schedule. No-op for constant mode."""
        if self.schedule == 'constant':
            return
        self.current_epoch = epoch
        self._current_temp = self._compute_temperature(epoch)
    
    def forward(self, logits: torch.Tensor, features: Optional[torch.Tensor] = None, epoch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if epoch is not None:
            temp_val = self._compute_temperature(epoch)
        else:
            temp_val = self._current_temp
        return torch.tensor(temp_val, device=logits.device, dtype=logits.dtype), logits / temp_val


class FixedTemperature(ScheduledTemperature):
    """Fixed temperature (backward compatible).
    
    Prefer :class:`ScheduledTemperature` with ``schedule='constant'`` for new code.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature=temperature, schedule='constant')


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
        self.learnable_base = learnable_base
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
        tau_0 = torch.exp(self.log_tau_0) if self.learnable_base else self.tau_0
        delta_tau = self.temp_proj(features).squeeze(-1)  # [B, N]
        temperatures = (tau_0 + delta_tau).clamp_min(self.min_temp)  # [B, N]
        temperatures = temperatures.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        return temperatures.mean(), logits / temperatures


class AnnealedTemperature(ScheduledTemperature):
    """Training-time temperature annealing (backward compatible).
    
    Prefer :class:`ScheduledTemperature` with ``schedule='scheduled'`` for new code.
    """
    
    def __init__(self, init_temperature: float = 1.0, final_temperature: float = 0.05,
                 warmup_epochs: int = 5, anneal_factor: float = 0.98):
        super().__init__(
            temperature=init_temperature,
            schedule='scheduled',
            final_temperature=final_temperature,
            warmup_epochs=warmup_epochs,
            anneal_factor=anneal_factor,
        )
        self.init_temperature = init_temperature


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
        mode: One of 'fixed'/'constant', 'learnable_scalar', 'per_head',
              'adaptive', 'annealed'/'scheduled'
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
    if mode in ('fixed', 'constant'):
        return FixedTemperature(temperature)
    elif mode == 'learnable_scalar':
        return LearnableScalarTemperature(temperature, min_temperature)
    elif mode == 'per_head':
        return PerHeadTemperature(n_heads, temperature, min_temperature)
    elif mode == 'adaptive':
        return AdaptiveTemperature(dim, temperature, min_temperature)
    elif mode in ('annealed', 'scheduled'):
        return AnnealedTemperature(temperature, anneal_final_temp, anneal_warmup_epochs, anneal_factor)
    else:
        raise ValueError(f"Unknown temperature mode: {mode}")
