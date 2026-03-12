"""
Minimal base model for GNN-PDE v2.

This module provides a simple base class with no magic, no registry,
no metaclasses - just a marker class for framework models.
"""

import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models in the GNN-PDE framework.
    
    This is intentionally minimal - just a marker class that inherits
    from nn.Module. No auto-registration, no magic, no global state.
    
    For auto-registration features, use:
        from gnn_pde_v2.convenient import AutoRegisterModel
    
    Example:
        class MyModel(BaseModel):
            def __init__(self, dim=128):
                super().__init__()
                self.net = nn.Linear(dim, dim)
            
            def forward(self, x):
                return self.net(x)
    """
    pass
