"""
Training utilities for DeepXDE-style examples.

These utilities are kept in examples/ rather than the framework core
to keep the framework lean. They provide a simple training wrapper
inspired by DeepXDE's unified API.

This module is NOT part of the public framework API.
"""

from typing import Optional, Union, Callable, Dict, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class LossFunction:
    """
    Wrapper for loss functions with optional reduction.
    """

    def __init__(
        self,
        loss_type: Union[str, Callable],
        reduction: str = 'mean',
    ):
        self.loss_type = loss_type
        self.reduction = reduction

        if isinstance(loss_type, str):
            if loss_type == 'mse':
                self.fn = nn.MSELoss(reduction=reduction)
            elif loss_type == 'l1':
                self.fn = nn.L1Loss(reduction=reduction)
            elif loss_type == 'smooth_l1':
                self.fn = nn.SmoothL1Loss(reduction=reduction)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        else:
            self.fn = loss_type

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)


class Model:
    """
    Unified model interface.

    Wraps architecture, loss function, and optimizer for consistent training.
    Inspired by DeepXDE's unified API for PINNs and Neural Operators.

    Example:
        # Create model components
        encoder = MLPEncoder(...)
        processor = GraphNetProcessor(...)
        decoder = MLPDecoder(...)
        architecture = EncodeProcessDecode(encoder, processor, decoder)

        # Create unified model
        model = Model(
            architecture=architecture,
            loss_fn='mse',
            optimizer=torch.optim.Adam(architecture.parameters(), lr=1e-3)
        )

        # Training
        for batch in dataloader:
            metrics = model.train_step(batch)
            print(f"Loss: {metrics['loss']}")

        # Inference
        predictions = model.predict(batch)
    """

    def __init__(
        self,
        architecture: nn.Module,
        loss_fn: Union[str, Callable] = 'mse',
        optimizer: Optional[Optimizer] = None,
    ):
        self.architecture = architecture
        self.loss_fn = LossFunction(loss_fn)
        self.optimizer = optimizer

        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def train_step(
        self,
        batch: Any,
        target_key: str = 'target',
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch (typically tuple of (input, target) or dict)
            target_key: Key for target in dict batch

        Returns:
            Dictionary of metrics
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Pass optimizer to constructor or call set_optimizer().")

        self.architecture.train()
        self.optimizer.zero_grad()

        # Extract input and target
        if isinstance(batch, dict):
            target = batch.pop(target_key)
            # Pass remaining as kwargs
            predictions = self.architecture(**batch)
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            inputs, target = batch
            predictions = self.architecture(inputs)
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

        # Compute loss
        loss = self.loss_fn(predictions, target)

        # Backward
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
        }

    @torch.no_grad()
    def predict(
        self,
        inputs: Any,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Inference.

        Args:
            inputs: Model inputs
            return_latent: If True, also return latent representation

        Returns:
            Predictions (and optionally latent representation)
        """
        self.architecture.eval()

        if return_latent and hasattr(self.architecture, 'get_latent'):
            latent = self.architecture.get_latent(inputs)
            predictions = self.architecture(inputs)
            return predictions, latent
        else:
            return self.architecture(inputs)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        target_key: str = 'target',
    ) -> Dict[str, float]:
        """
        Evaluate on dataset.

        Returns:
            Dictionary of metrics
        """
        self.architecture.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                target = batch[target_key]
                inputs = {k: v for k, v in batch.items() if k != target_key}
                predictions = self.architecture(**inputs)
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, target = batch
                predictions = self.architecture(inputs)
            else:
                continue

            loss = self.loss_fn(predictions, target)
            total_loss += loss.item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches if n_batches > 0 else float('inf'),
        }

    def set_optimizer(self, optimizer: Optimizer):
        """Set or update optimizer."""
        self.optimizer = optimizer

    def save_checkpoint(self, path: str, epoch: int = 0, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.architecture.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, map_location=None):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        self.architecture.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        return checkpoint.get('epoch', 0)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            'model_state_dict': self.architecture.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load from state dict."""
        self.architecture.load_state_dict(state_dict['model_state_dict'])
        if self.optimizer and state_dict.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.history = state_dict.get('history', {'train_loss': [], 'val_loss': []})
