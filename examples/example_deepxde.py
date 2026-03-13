"""
Example: DeepXDE-Style Unified API

This example recreates the DeepXDE unified API from:
https://github.com/lululxvi/deepxde

Reference: Lu et al. "DeepXDE: A Deep Learning Library for Solving Differential Equations" (2021)

Key characteristics:
1. Unified Model(data, net) API
2. Separation of Data (sampling, losses) and Network (architecture)
3. Common training loop for different methods (PINN, DeepONet, FNO)
4. Uses framework components (core.MLP, models.FNO, etc.)

This version uses gnn_pde_v2 framework components while preserving
the DeepXDE-style API abstraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Framework imports
from gnn_pde_v2.core import MLP
from gnn_pde_v2.convenient import AutoRegisterModel, get_initializer
from training_utils import Model


# ============================================================================
# Data Interface (DeepXDE-style abstraction)
# ============================================================================

class Data(ABC):
    """
    Abstract base class for DeepXDE-style Data.

    In DeepXDE, Data handles:
    - Training data sampling
    - Loss computation
    - Test data generation

    This abstraction is preserved to maintain the DeepXDE API style.
    """

    @abstractmethod
    def losses(self, targets, outputs, loss_fn, inputs, model):
        """
        Compute losses.

        Args:
            targets: Target values
            outputs: Model outputs
            loss_fn: Loss function
            inputs: Model inputs
            model: The Model instance

        Returns:
            List of losses
        """
        pass

    @abstractmethod
    def train_next_batch(self, batch_size: Optional[int] = None):
        """Get next training batch."""
        pass

    @abstractmethod
    def test(self):
        """Get test data."""
        pass


class TimePDE(Data):
    """
    Time-dependent PDE data handler.

    Example: Burgers equation, Navier-Stokes
    """

    def __init__(
        self,
        geometry,
        pde: Callable,
        ic: Callable,
        bc: Callable,
        num_domain: int = 1000,
        num_boundary: int = 100,
        num_initial: int = 100,
    ):
        self.geometry = geometry
        self.pde = pde
        self.ic = ic
        self.bc = bc
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial

    def losses(self, targets, outputs, loss_fn, inputs, model):
        """Compute PDE, BC, and IC losses."""
        # PDE residual loss
        pde_residual = self.pde(inputs, outputs, model.net)
        pde_loss = loss_fn(pde_residual, torch.zeros_like(pde_residual))

        # Boundary condition loss
        bc_residual = self.bc(inputs, outputs)
        bc_loss = loss_fn(bc_residual, torch.zeros_like(bc_residual))

        # Initial condition loss
        ic_residual = self.ic(inputs, outputs)
        ic_loss = loss_fn(ic_residual, torch.zeros_like(ic_residual))

        return [pde_loss, bc_loss, ic_loss]

    def train_next_batch(self, batch_size=None):
        """Sample training points."""
        # Domain points
        domain_points = self.geometry.sample_domain(self.num_domain)
        # Boundary points
        boundary_points = self.geometry.sample_boundary(self.num_boundary)
        # Initial points
        initial_points = self.geometry.sample_initial(self.num_initial)

        return torch.cat([domain_points, boundary_points, initial_points], dim=0)

    def test(self):
        """Get test data."""
        return self.geometry.sample_domain(self.num_domain)


class PDENetData(Data):
    """
    Data handler for Neural Operator (FNO-style).

    Uses paired input-output data for supervised learning.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_train = len(X_train)

    def losses(self, targets, outputs, loss_fn, inputs, model):
        """Compute supervised loss."""
        return [loss_fn(outputs, targets)]

    def train_next_batch(self, batch_size=None):
        """Get training batch."""
        if batch_size is None:
            return self.X_train, self.y_train

        indices = torch.randperm(self.num_train)[:batch_size]
        return self.X_train[indices], self.y_train[indices]

    def test(self):
        """Get test data."""
        return self.X_test, self.y_test


# ============================================================================
# Network Interface (using framework components)
# ============================================================================

class NN(nn.Module, ABC):
    """
    Abstract base class for DeepXDE-style Neural Networks.
    """

    @abstractmethod
    def forward(self, inputs):
        """Forward pass."""
        pass


class DeepONet(NN, AutoRegisterModel, name='deeponet', namespace='example'):
    """
    Deep Operator Network using framework's MLP.

    Architecture:
        Branch net: encodes input function u(x)
        Trunk net: encodes query points y
        Output: dot product of branch and trunk outputs

    Reference:
        Lu et al. "Learning nonlinear operators via DeepONet based on
        the universal approximation theorem of operators." (2021)
    """

    def __init__(
        self,
        layer_sizes_branch: List[int],
        layer_sizes_trunk: List[int],
        activation: str = "gelu",
        kernel_initializer: str = "Glorot uniform",
        num_outputs: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.num_outputs = num_outputs

        # Branch net using framework's MLP
        branch_input = layer_sizes_branch[0]
        branch_output = layer_sizes_branch[-1]
        branch_hidden = layer_sizes_branch[1:-1]
        weight_init = get_initializer(kernel_initializer)

        self.branch = MLP(
            in_dim=branch_input,
            out_dim=branch_output * num_outputs,
            hidden_dims=branch_hidden,
            activation=activation,
            dropout=dropout_rate,
            use_layer_norm=False,
            weight_init=weight_init,
        )

        # Trunk net using framework's MLP
        trunk_input = layer_sizes_trunk[0]
        trunk_output = layer_sizes_trunk[-1]
        trunk_hidden = layer_sizes_trunk[1:-1]

        self.trunk = MLP(
            in_dim=trunk_input,
            out_dim=trunk_output * num_outputs,
            hidden_dims=trunk_hidden,
            activation=activation,
            dropout=dropout_rate,
            use_layer_norm=False,
            weight_init=weight_init,
        )

        # Bias terms
        self.b = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(num_outputs)
        ])

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: Tuple of (x_func, x_loc)
                x_func: Input function values [batch, branch_input]
                x_loc: Query locations [batch, trunk_input]

        Returns:
            Output [batch, num_outputs]
        """
        x_func, x_loc = inputs

        # Branch and trunk encodings
        branch_out = self.branch(x_func)
        trunk_out = self.trunk(x_loc)

        if self.num_outputs == 1:
            # Single output: dot product
            y = torch.einsum("bi,bi->b", branch_out, trunk_out)
            y = y.unsqueeze(-1) + self.b[0]
        else:
            # Multiple outputs: split and compute
            branch_splits = branch_out.chunk(self.num_outputs, dim=-1)
            trunk_splits = trunk_out.chunk(self.num_outputs, dim=-1)
            ys = []
            for i, (b_split, t_split) in enumerate(zip(branch_splits, trunk_splits)):
                y = torch.einsum("bi,bi->b", b_split, t_split)
                y = y.unsqueeze(-1) + self.b[i]
                ys.append(y)
            y = torch.cat(ys, dim=-1)

        return y


class FNO(NN):
    """
    Fourier Neural Operator using framework's FNO implementation.

    Wraps framework's models.FNO for DeepXDE-style API.
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ):
        super().__init__()

        # Use the FNO implementation from gnn_pde_v2
        from gnn_pde_v2.models.fno_model import FNO as FrameworkFNO

        self.net = FrameworkFNO(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=[modes] * 2,  # 2D
            n_layers=n_layers,
            n_dim=2,
        )

    def forward(self, inputs):
        return self.net(inputs)


# ============================================================================
# Model Interface (Unified API from DeepXDE)
# ============================================================================

class Model:
    """
    DeepXDE-style unified Model API.

    The Model wraps Data and NN to provide a consistent training interface
    regardless of the underlying method (PINN, DeepONet, FNO, etc.).

    This builds on the framework's training.Model with additional
    DeepXDE-specific features like multi-loss handling.

    Args:
        data: Data instance (handles sampling and losses)
        net: NN instance (the neural network architecture)
    """

    def __init__(self, data: Data, net: NN):
        self.data = data
        self.net = net

        # Training state
        self.opt_name = None
        self.batch_size = None
        self.loss_weights = None
        self.metrics = None
        self.optimizer = None

        # History
        self.train_state = TrainState()
        self.losshistory = LossHistory()

    def compile(
        self,
        optimizer: str = "adam",
        lr: float = 1e-3,
        loss: Union[str, Callable] = "MSE",
        metrics: Optional[list] = None,
        loss_weights: Optional[list] = None,
    ):
        """
        Configure the model for training.

        Args:
            optimizer: Optimizer name ('adam', 'sgd', 'lbfgs')
            lr: Learning rate
            loss: Loss function ('MSE', 'L2', etc.)
            metrics: List of metrics to track
            loss_weights: Weights for each loss component
        """
        self.opt_name = optimizer
        self.loss_weights = loss_weights

        # Setup optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.net.parameters(),
                lr=lr,
                max_iter=100,
            )

        # Setup loss function
        if loss == "MSE":
            self.loss_fn = nn.MSELoss()
        elif loss == "L2":
            self.loss_fn = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)
        else:
            self.loss_fn = loss

        self.metrics = metrics or []

    def train(
        self,
        epochs: Optional[int] = None,
        iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        display_every: int = 100,
    ):
        """
        Train the model.

        Args:
            epochs: Number of epochs
            iterations: Number of iterations (overrides epochs)
            batch_size: Batch size
            display_every: Display progress every N steps
        """
        if iterations is None:
            iterations = epochs * 100  # Default

        print("Training model...\n")
        print(f"{'Step':>8} {'Loss':>12} {'Test Loss':>12}")
        print("-" * 40)

        for step in range(iterations):
            # Training step
            loss = self._train_step(batch_size)

            # Update history
            self.losshistory.add_loss(loss)

            # Display progress
            if step % display_every == 0 or step == iterations - 1:
                test_loss = self._test()
                print(f"{step:8d} {loss:12.4e} {test_loss:12.4e}")

        print("\nTraining finished!")

    def _train_step(self, batch_size):
        """Single training step."""
        self.net.train()

        # Get batch
        batch = self.data.train_next_batch(batch_size)

        if isinstance(batch, tuple):
            inputs, targets = batch
        else:
            inputs = batch
            targets = None

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward
        outputs = self.net(inputs)

        # Compute losses
        losses = self.data.losses(targets, outputs, self.loss_fn, inputs, self)

        # Weight and sum losses
        if self.loss_weights is not None:
            loss = sum(w * l for w, l in zip(self.loss_weights, losses))
        else:
            loss = sum(losses)

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _test(self):
        """Evaluate on test data."""
        self.net.eval()

        with torch.no_grad():
            batch = self.data.test()

            if isinstance(batch, tuple):
                inputs, targets = batch
            else:
                inputs = batch
                targets = None

            outputs = self.net(inputs)

            if targets is not None:
                return self.loss_fn(outputs, targets).item()
            else:
                return 0.0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict on new data.

        Args:
            x: Input tensor

        Returns:
            Predictions
        """
        self.net.eval()
        with torch.no_grad():
            return self.net(x)

    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, filepath)

    def restore(self, filepath: str):
        """Restore model."""
        checkpoint = torch.load(filepath)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TrainState:
    """Track training state."""

    def __init__(self):
        self.step = 0
        self.epoch = 0


class LossHistory:
    """Track loss history."""

    def __init__(self):
        self.loss_train = []
        self.loss_test = []

    def add_loss(self, loss):
        self.loss_train.append(loss)


# ============================================================================
# Geometry Helpers
# ============================================================================

class Geometry:
    """Base geometry class."""

    def sample_domain(self, n):
        """Sample points in the domain."""
        raise NotImplementedError

    def sample_boundary(self, n):
        """Sample points on the boundary."""
        raise NotImplementedError

    def sample_initial(self, n):
        """Sample points at initial time."""
        raise NotImplementedError


class Interval(Geometry):
    """1D interval [xmin, xmax]."""

    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def sample_domain(self, n):
        return torch.rand(n, 1) * (self.xmax - self.xmin) + self.xmin

    def sample_boundary(self, n):
        # Two boundary points
        n_per_side = n // 2
        left = torch.full((n_per_side, 1), self.xmin)
        right = torch.full((n - n_per_side, 1), self.xmax)
        return torch.cat([left, right], dim=0)

    def sample_initial(self, n):
        return torch.zeros(n, 1)


class Rectangle(Geometry):
    """2D rectangle [xmin, xmax] x [ymin, ymax]."""

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def sample_domain(self, n):
        x = torch.rand(n, 1) * (self.xmax - self.xmin) + self.xmin
        y = torch.rand(n, 1) * (self.ymax - self.ymin) + self.ymin
        return torch.cat([x, y], dim=1)

    def sample_boundary(self, n):
        # Sample on 4 sides
        n_per_side = n // 4

        # Bottom
        x = torch.rand(n_per_side, 1) * (self.xmax - self.xmin) + self.xmin
        y = torch.full((n_per_side, 1), self.ymin)
        bottom = torch.cat([x, y], dim=1)

        # Top
        x = torch.rand(n_per_side, 1) * (self.xmax - self.xmin) + self.xmin
        y = torch.full((n_per_side, 1), self.ymax)
        top = torch.cat([x, y], dim=1)

        # Left
        x = torch.full((n_per_side, 1), self.xmin)
        y = torch.rand(n_per_side, 1) * (self.ymax - self.ymin) + self.ymin
        left = torch.cat([x, y], dim=1)

        # Right
        x = torch.full((n - 3 * n_per_side, 1), self.xmax)
        y = torch.rand((n - 3 * n_per_side, 1)) * (self.ymax - self.ymin) + self.ymin
        right = torch.cat([x, y], dim=1)

        return torch.cat([bottom, top, left, right], dim=0)

    def sample_initial(self, n):
        x = torch.rand(n, 1) * (self.xmax - self.xmin) + self.xmin
        y = torch.rand(n, 1) * (self.ymax - self.ymin) + self.ymin
        return torch.cat([x, y], dim=1)


# ============================================================================
# Usage Examples
# ============================================================================

def example_pinn():
    """Example: PINN for 1D Poisson equation using framework's MLP."""

    # Define PDE: u_xx = f(x)
    def pde(inputs, outputs, net):
        x = inputs.requires_grad_(True)
        u = net(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = torch.sin(2 * torch.pi * x)  # Source term
        return u_xx - f

    def bc(inputs, outputs):
        # Boundary condition: u(0) = u(1) = 0
        return outputs

    def ic(inputs, outputs):
        # No initial condition for steady state
        return torch.zeros_like(outputs)

    # Create geometry
    geom = Interval(0, 1)

    # Create data handler
    data = TimePDE(
        geometry=geom,
        pde=pde,
        ic=ic,
        bc=bc,
        num_domain=100,
        num_boundary=20,
        num_initial=0,
    )

    # Create network using framework's MLP directly
    # layer_sizes=[1, 50, 50, 50, 1] -> in_dim=1, hidden=[50,50,50], out_dim=1
    net = MLP(
        in_dim=1,
        out_dim=1,
        hidden_dims=[50, 50, 50],
        activation="tanh",
        use_layer_norm=False,  # DeepXDE-style: no LayerNorm
        weight_init=get_initializer("Glorot uniform"),
    )

    # Create model
    model = Model(data, net)
    model.compile(optimizer="adam", lr=1e-3, loss="MSE")

    # Train
    model.train(iterations=1000, display_every=200)

    # Predict
    x_test = torch.linspace(0, 1, 100).unsqueeze(1)
    u_pred = model.predict(x_test)

    print(f"\nPredicted solution shape: {u_pred.shape}")
    print(f"Network uses framework's core.MLP directly")

    return model


def example_fno():
    """Example: FNO for Darcy flow using framework's FNO."""

    # Create synthetic training data
    n_train = 100
    resolution = 32

    # Random permeability fields
    X_train = torch.randn(n_train, 1, resolution, resolution)
    # Corresponding pressure fields (synthetic)
    y_train = torch.randn(n_train, 1, resolution, resolution)

    # Create data handler
    data = PDENetData(X_train, y_train)

    # Create FNO network using framework's implementation
    net = FNO(
        modes=16,
        width=64,
        in_channels=1,
        out_channels=1,
        n_layers=4,
    )

    # Create model
    model = Model(data, net)
    model.compile(optimizer="adam", lr=1e-3, loss="MSE")

    # Train
    model.train(iterations=500, batch_size=10, display_every=100)

    # Predict
    X_test = torch.randn(1, 1, resolution, resolution)
    y_pred = model.predict(X_test)

    print(f"\nPredicted solution shape: {y_pred.shape}")
    print(f"Network uses framework's models.FNO")

    return model


def example_deeponet():
    """Example: DeepONet using framework's MLP components."""

    # Create synthetic data
    n_train = 500
    n_sensors = 50
    n_query = 20

    # Input function values at sensor points
    x_func = torch.randn(n_train, n_sensors)
    # Query locations
    x_loc = torch.rand(n_train, 1)
    # Ground truth outputs
    y_train = torch.randn(n_train, 1)

    # Create DeepONet using framework's MLP
    net = DeepONet(
        layer_sizes_branch=[n_sensors, 128, 128, 64],
        layer_sizes_trunk=[1, 64, 64, 64],
        activation="gelu",
    )

    # Simple forward pass example
    output = net((x_func, x_loc))

    print(f"Branch input: {x_func.shape}")
    print(f"Trunk input: {x_loc.shape}")
    print(f"Output: {output.shape}")
    print(f"Network uses framework's core.MLP for both branch and trunk")

    return net


if __name__ == "__main__":
    print("=" * 50)
    print("Example 1: PINN for 1D Poisson equation")
    print("Using framework's core.MLP directly")
    print("=" * 50)
    model_pinn = example_pinn()

    print("\n" + "=" * 50)
    print("Example 2: FNO for Darcy flow")
    print("Using framework's models.FNO")
    print("=" * 50)
    model_fno = example_fno()

    print("\n" + "=" * 50)
    print("Example 3: DeepONet")
    print("Using framework's core.MLP for branch and trunk")
    print("=" * 50)
    model_deeponet = example_deeponet()
