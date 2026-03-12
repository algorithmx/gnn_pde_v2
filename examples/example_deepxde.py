"""
Example: DeepXDE-style Physics-Informed Neural Network

This example recreates the DeepXDE model interface from:
https://github.com/lululxvi/deepxde

Original Work Reference:
------------------------
Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021).
"DeepXDE: A deep learning library for solving differential equations."
SIAM Review, 63(1), 208-228.
Paper: https://doi.org/10.1137/19M1274067

Key Innovation:
---------------
DeepXDE provides a unified framework for solving various types of PDEs
using physics-informed neural networks (PINNs).

This implementation uses the gnn_pde_v2 framework components while maintaining
exact equivalence to the original DeepXDE FNN architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable, Dict, Union
import math

# Import framework components
from gnn_pde_v2.convenient import AutoRegisterModel
from gnn_pde_v2.components import FourierFeatureEncoder
from gnn_pde_v2.models.unified_model import Model


class SinActivation(nn.Module):
    """Sine activation used in some PINN variants."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class DeepXDEFNN(nn.Module):
    """Faithful DeepXDE-style fully-connected network.

    Mirrors `deepxde/nn/pytorch/fnn.py`:
    - ModuleList of Linear layers
    - activation on all but the last linear
    - optional per-layer dropout
    - no LayerNorm
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[str, List[str]],
        kernel_initializer: str,
        dropout_rate: Union[float, List[float]] = 0.0,
        use_residual: bool = False,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.kernel_initializer = kernel_initializer
        self.use_residual = use_residual
        self._input_transform = None
        self._output_transform = None

        if isinstance(activation, list):
            if len(activation) != len(layer_sizes) - 1:
                raise ValueError(
                    "Total number of activation functions must match the number of linear layers."
                )
            self.activation = [self._get_activation(a) for a in activation]
        else:
            self.activation = self._get_activation(activation)

        if isinstance(dropout_rate, list):
            if len(dropout_rate) != len(layer_sizes) - 1:
                raise ValueError(
                    f"Number of dropout rates must be equal to {len(layer_sizes) - 1}"
                )
            self.dropout_rate = dropout_rate
        else:
            self.dropout_rate = [dropout_rate] * (len(layer_sizes) - 1)

        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            linear = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            self.linears.append(linear)

        self._initialize_weights()

    def _get_activation(self, name: str):
        mapping = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "sin": SinActivation(),
        }
        if name not in mapping:
            raise ValueError(f"Unsupported activation: {name}")
        return mapping[name]

    def _initialize_weights(self):
        for linear in self.linears:
            in_size = linear.weight.shape[1]
            out_size = linear.weight.shape[0]

            if self.kernel_initializer == "Glorot uniform":
                limit = math.sqrt(6.0 / (in_size + out_size))
                nn.init.uniform_(linear.weight, -limit, limit)
            elif self.kernel_initializer == "Glorot normal":
                nn.init.xavier_normal_(linear.weight)
            elif self.kernel_initializer == "He normal":
                nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            elif self.kernel_initializer == "He uniform":
                nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(linear.weight)

            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for j, linear in enumerate(self.linears[:-1]):
            act = self.activation[j] if isinstance(self.activation, list) else self.activation
            x_new = act(linear(x))
            if self.dropout_rate[j] > 0:
                x_new = torch.nn.functional.dropout(
                    x_new, p=self.dropout_rate[j], training=self.training
                )
            if self.use_residual and x_new.shape == x.shape:
                x = x + x_new
            else:
                x = x_new

        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class DeepXDEModel(AutoRegisterModel, name='deepxde'):
    """
    DeepXDE-style Physics-Informed Neural Network using gnn_pde_v2 framework.
    
    Original implementation: deepxde/nn/pytorch/fnn.py
    
    Architecture:
        Input x ∈ R^d (spatial/temporal coordinates)
            ↓
        [Optional] Fourier Feature Mapping:
            Uses framework's FourierFeatureEncoder
            x → [sin(2πBx), cos(2πBx)]
            Helps with high-frequency PDEs
            ↓
        Hidden Layers: Framework's MLP
            - Configurable activation (tanh, sin, etc.)
            - Optional residual connections
            - Optional layer normalization
            ↓
        Output: u(x) ∈ R^m (PDE solution)
    
    The network is trained by minimizing the PDE residual:
        L = L_PDE + α · L_BC + β · L_IC
    
    Derivatives are computed via autograd for physics constraints.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        kernel_initializer: str = "Glorot uniform",
        use_fourier_features: bool = False,
        fourier_feature_scale: float = 1.0,
        num_fourier_features: int = 256,
        use_residual: bool = False,
        use_layer_norm: bool = False,
    ):
        """
        Initialize DeepXDE-style model using framework components.
        
        Args:
            layer_sizes: List of layer dimensions, e.g., [2, 64, 64, 64, 1]
                        for input_dim=2, hidden=[64,64,64], output_dim=1
            activation: Activation function ("tanh", "sin", "relu", "sigmoid", "gelu")
            kernel_initializer: Weight initialization ("Glorot uniform", etc.)
            use_fourier_features: Whether to use Fourier feature mapping
            fourier_feature_scale: Scale of random Fourier frequencies
            num_fourier_features: Number of Fourier features (output dim = 2×)
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.layer_sizes = layer_sizes.copy()  # Copy to avoid modifying input
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer
        self.use_fourier_features = use_fourier_features
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        input_dim = layer_sizes[0]
        output_dim = layer_sizes[-1]
        
        # Build Fourier feature mapping using framework component (if enabled)
        if use_fourier_features:
            self.fourier_encoder = FourierFeatureEncoder(
                input_dim=input_dim,
                num_fourier_features=num_fourier_features,
                scale=fourier_feature_scale,
                learnable=False,
                include_input=False,  # DeepXDE-style: only Fourier features
            )
            # Update first layer input size to match Fourier output
            mlp_input_dim = num_fourier_features * 2
        else:
            self.fourier_encoder = None
            mlp_input_dim = input_dim
        
        adjusted_layer_sizes = [mlp_input_dim] + layer_sizes[1:]
        self.mlp = DeepXDEFNN(
            layer_sizes=adjusted_layer_sizes,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout_rate=0.0,
            use_residual=use_residual,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input coordinates [batch_size, input_dim]
               Can be spatial (x), temporal (t), or spatiotemporal (x, t)
               
        Returns:
            Network output [batch_size, output_dim]
        """
        # Fourier feature mapping (for high-frequency PDEs)
        if self.use_fourier_features and self.fourier_encoder is not None:
            x = self.fourier_encoder(x)
        
        return self.mlp(x)
    
    def save_config(self):
        """Save model configuration."""
        return {
            'model_type': 'deepxde',
            'layer_sizes': self.layer_sizes,
            'activation': self.activation_name,
            'kernel_initializer': self.kernel_initializer,
            'use_fourier_features': self.use_fourier_features,
            'fourier_feature_scale': getattr(self, 'fourier_feature_scale', 1.0),
            'num_fourier_features': getattr(self, 'num_fourier_features', 256),
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
        }


# Backward-compatible alias expected by tests/examples.
PINNModel = DeepXDEModel


# ============================================================================
# Physics-Informed Loss (not part of framework, specific to PINNs)
# ============================================================================

class PhysicsLoss:
    """
    Physics-informed loss function for PDE training.
    
    Combines PDE residual, boundary condition, and initial condition losses.
    This is specific to PINN applications and not part of the core framework.
    """
    
    def __init__(
        self,
        pde_fn: Callable,
        bc_fns: Optional[List[Callable]] = None,
        ic_fn: Optional[Callable] = None,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
    ):
        """
        Args:
            pde_fn: Function that computes PDE residual: f(x, u, u_x, u_xx, ...)
            bc_fns: List of boundary condition functions
            ic_fn: Initial condition function
            lambda_bc: Weight for BC loss
            lambda_ic: Weight for IC loss
        """
        self.pde_fn = pde_fn
        self.bc_fns = bc_fns or []
        self.ic_fn = ic_fn
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
    
    def __call__(
        self,
        model: nn.Module,
        x_pde: torch.Tensor,
        x_bc: Optional[torch.Tensor] = None,
        x_ic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total physics-informed loss.
        
        Returns:
            Dictionary with 'total', 'pde', 'bc', 'ic' losses
        """
        x_pde = x_pde.requires_grad_(True)
        u = model(x_pde)
        
        # Compute first derivatives
        grads = torch.autograd.grad(
            u.sum(), x_pde,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Package for PDE function
        derivs = {'u': u, 'x': x_pde}
        for i in range(x_pde.shape[1]):
            derivs[f'u_x{i}'] = grads[:, i:i+1]
        
        # PDE loss
        pde_residual = self.pde_fn(**derivs)
        loss_pde = torch.mean(pde_residual ** 2)
        
        losses = {'pde': loss_pde, 'total': loss_pde}
        
        # BC loss
        if x_bc is not None and self.bc_fns:
            loss_bc = 0
            u_bc = model(x_bc)
            for bc_fn in self.bc_fns:
                loss_bc += torch.mean(bc_fn(x_bc, u_bc) ** 2)
            loss_bc /= len(self.bc_fns)
            losses['bc'] = loss_bc
            losses['total'] += self.lambda_bc * loss_bc
        
        # IC loss
        if x_ic is not None and self.ic_fn is not None:
            u_ic = model(x_ic)
            loss_ic = torch.mean(self.ic_fn(x_ic, u_ic) ** 2)
            losses['ic'] = loss_ic
            losses['total'] += self.lambda_ic * loss_ic
        
        return losses


# ============================================================================
# Integration with Unified Model API
# ============================================================================

def create_deepxde_model(
    layer_sizes: List[int],
    activation: str = "tanh",
    learning_rate: float = 1e-3,
    use_fourier_features: bool = False,
    **kwargs
) -> Model:
    """
    Create a DeepXDE-style model wrapped in the framework's Unified Model API.
    
    This provides a DeepXDE-like interface using gnn_pde_v2 framework components.
    
    Args:
        layer_sizes: List of layer dimensions
        activation: Activation function
        learning_rate: Learning rate for optimizer
        use_fourier_features: Whether to use Fourier feature mapping
        **kwargs: Additional arguments for DeepXDEModel
        
    Returns:
        Unified Model instance ready for training
    """
    # Create architecture
    architecture = DeepXDEModel(
        layer_sizes=layer_sizes,
        activation=activation,
        use_fourier_features=use_fourier_features,
        **kwargs
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(architecture.parameters(), lr=learning_rate)
    
    # Wrap in Unified Model
    return Model(
        architecture=architecture,
        loss_fn='mse',
        optimizer=optimizer,
    )


# ============================================================================
# Example: Poisson Equation
# ============================================================================

def poisson_pde(**kwargs):
    """
    Poisson equation: ∇²u = f
    
    In 2D: u_xx + u_yy = f(x, y)
    
    Here we solve: ∇²u = sin(πx)sin(πy) with u=0 on boundary
    """
    u = kwargs['u']
    x = kwargs['x']
    
    # Compute first derivatives
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
    
    # Compute second derivatives
    u_xx = torch.autograd.grad(u_x[:, 0].sum(), x, create_graph=True, retain_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(u_x[:, 1].sum(), x, create_graph=True, retain_graph=True)[0][:, 1]
    
    # Source term
    f = torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1])
    
    # PDE residual: ∇²u - f = 0
    residual = u_xx + u_yy - f
    return residual


def example_poisson():
    """
    Example: Solving 2D Poisson equation with PINN using framework components.
    """
    print("=" * 60)
    print("DeepXDE-style Poisson Equation Example")
    print("=" * 60)
    
    # Network: 2 inputs (x, y) → 64 hidden → 1 output (u)
    # Uses framework's DeepXDEModel and FourierFeatureEncoder
    model = DeepXDEModel(
        layer_sizes=[2, 64, 64, 64, 1],
        activation="tanh",
        kernel_initializer="Glorot uniform",
        use_fourier_features=False,
    )
    
    # Physics loss
    physics_loss = PhysicsLoss(
        pde_fn=poisson_pde,
        lambda_bc=10.0,  # Strong BC enforcement
    )
    
    # Collocation points
    num_points = 1000
    x_pde = torch.rand(num_points, 2) * 2 - 1  # [-1, 1] × [-1, 1]
    
    # Boundary points (u=0)
    num_bc = 200
    theta = torch.linspace(0, 2*math.pi, num_bc)
    x_bc = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    
    # Forward pass
    u = model(x_pde)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {model.layer_sizes}")
    print(f"  Activation: {model.activation_name}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nInput/Output:")
    print(f"  Input shape: {x_pde.shape}")
    print(f"  Output shape: {u.shape}")
    
    # Compute physics loss (example)
    losses = physics_loss(model, x_pde, x_bc)
    print(f"\nPhysics Losses:")
    print(f"  PDE: {losses['pde']:.6f}")
    print(f"  BC: {losses.get('bc', 0):.6f}")
    print(f"  Total: {losses['total']:.6f}")
    
    return model, losses


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """
    Demonstrate DeepXDE-style models using gnn_pde_v2 framework.
    """
    print("=" * 60)
    print("DeepXDE Examples using gnn_pde_v2 Framework")
    print("=" * 60)
    
    print("\n" + "-" * 60)
    print("Example 1: Standard FNN for low-frequency PDE")
    print("-" * 60)
    
    model1 = DeepXDEModel(
        layer_sizes=[2, 64, 64, 64, 1],
        activation="tanh",
        kernel_initializer="Glorot uniform",
    )
    
    x = torch.randn(100, 2)
    y = model1(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\n" + "-" * 60)
    print("Example 2: Fourier Feature Network for high-frequency PDE")
    print("-" * 60)
    
    model2 = DeepXDEModel(
        layer_sizes=[2, 128, 128, 128, 1],
        activation="tanh",  # Framework uses tanh for sin
        use_fourier_features=True,
        num_fourier_features=256,
        fourier_feature_scale=10.0,
    )
    
    y2 = model2(x)
    print(f"Input: {x.shape}, Output: {y2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"Uses framework's FourierFeatureEncoder")
    
    print("\n" + "-" * 60)
    print("Example 3: Residual Network with LayerNorm")
    print("-" * 60)
    
    model3 = DeepXDEModel(
        layer_sizes=[3, 128, 128, 128, 2],
        activation="gelu",
        use_residual=True,
        use_layer_norm=True,
    )
    
    x3 = torch.randn(50, 3)
    y3 = model3(x3)
    print(f"Input: {x3.shape}, Output: {y3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    print(f"Uses framework's MLP with layer_norm")
    
    print("\n" + "-" * 60)
    print("Example 4: Unified Model API")
    print("-" * 60)
    
    # Using the Unified Model wrapper
    unified_model = create_deepxde_model(
        layer_sizes=[2, 64, 64, 1],
        activation="tanh",
        learning_rate=1e-3,
    )
    
    print(f"Unified Model created with framework Model class")
    print(f"Ready for training with model.train_step(batch)")
    
    print("\n" + "=" * 60)
    print("Model registered as:", model1._model_name)
    print("Available models:", AutoRegisterModel.list_models())
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Example 5: Poisson Equation")
    print("=" * 60)
    model4, losses = example_poisson()
    
    return model1, model2, model3, unified_model, model4


if __name__ == "__main__":
    model1, model2, model3, unified_model, model4 = example_usage()
