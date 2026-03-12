"""
String-based weight initialization (optional convenience).

For lean usage, use torch.nn.init directly:
    import torch.nn.init as init
    init.xavier_uniform_(tensor)
"""

from typing import Union, Callable
import torch
import torch.nn as nn
import torch.nn.init as init


def get_initializer(name: Union[str, Callable]) -> Callable:
    """
    Get initializer function by name.
    
    This is OPTIONAL sugar. For lean usage, pass init functions directly.
    
    Args:
        name: Initializer name or callable
        
    Returns:
        Initialization function
        
    Example:
        >>> init_fn = get_initializer('glorot_uniform')
        >>> init_fn(tensor)
        >>> 
        >>> # Or just use PyTorch directly:
        >>> init.xavier_uniform_(tensor)
    """
    if callable(name):
        return name
    
    name = name.lower().strip()
    
    # Handle constant_X format
    if name.startswith('constant_'):
        try:
            val = float(name.split('_')[1])
            return lambda t: init.constant_(t, val)
        except (IndexError, ValueError):
            pass
    
    mapping = {
        # Glorot/Xavier
        'glorot_uniform': init.xavier_uniform_,
        'glorot_normal': init.xavier_normal_,
        'xavier_uniform': init.xavier_uniform_,
        'xavier_normal': init.xavier_normal_,
        # He/Kaiming
        'he_uniform': init.kaiming_uniform_,
        'he_normal': init.kaiming_normal_,
        'kaiming_uniform': init.kaiming_uniform_,
        'kaiming_normal': init.kaiming_normal_,
        # Others
        'orthogonal': init.orthogonal_,
        'uniform': init.uniform_,
        'normal': init.normal_,
        'zeros': init.zeros_,
        'ones': init.ones_,
    }
    
    # DeepXDE-style names with spaces
    mapping['glorot uniform'] = init.xavier_uniform_
    mapping['glorot normal'] = init.xavier_normal_
    mapping['he uniform'] = init.kaiming_uniform_
    mapping['he normal'] = init.kaiming_normal_
    
    if name not in mapping:
        available = ', '.join(sorted(set(mapping.keys())))
        raise ValueError(f"Unknown initializer: '{name}'. Available: [{available}]")
    
    return mapping[name]


def initialize_module(
    module: nn.Module,
    weight_init: Union[str, Callable] = 'glorot_uniform',
    bias_init: Union[str, Callable] = 'constant_0',
) -> None:
    """
    Initialize all parameters in a module.
    
    Args:
        module: PyTorch module to initialize
        weight_init: Initializer for weights (name or callable)
        bias_init: Initializer for biases (name or callable)
    """
    weight_fn = get_initializer(weight_init)
    bias_fn = get_initializer(bias_init)
    
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            weight_fn(m.weight)
            if m.bias is not None:
                bias_fn(m.bias)
        elif isinstance(m, nn.Embedding):
            weight_fn(m.weight)
