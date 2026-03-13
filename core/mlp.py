"""
Canonical MLP building block for the GNN-PDE framework.

Flexible feedforward network with configurable hidden/final behavior
and per-layer normalization support.
"""

from typing import List, Optional, Callable, Sequence, Union, Any
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init


_UNSET = object()


class SinActivation(nn.Module):
    """Sine activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    """
    Flexible feedforward network with configurable hidden/final behavior.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Hidden-layer activation spec
        dropout: Hidden-layer dropout spec
        norm: Hidden-layer normalization spec
        final_activation: Output-layer activation spec
        final_dropout: Output-layer dropout
        final_norm: Output-layer normalization spec
        norms: Per-layer normalization specs (overrides norm/final_norm if provided)
        linear_factory: Callable creating the affine layer for each stage
        use_layer_norm: Backward-compatible alias for hidden LayerNorm behavior
        pre_activation: If set, uses pre-activation pattern (Act → Linear) instead of
            post-activation (Linear → Act). Value can be an activation spec (str, module,
            or callable) which is applied before each linear layer. This is useful for
            architectures like AdaLN modulation networks.
        weight_init: Weight initialization function (default: xavier_uniform_)
        bias_init: Bias initialization function (default: constant 0)

    Example:
        >>> # Standard usage (post-activation: Linear → Act)
        >>> mlp = MLP(64, 64, hidden_dims=[128, 128], use_layer_norm=False)
        >>>
        >>> # With custom initialization
        >>> mlp = MLP(64, 64, [128], weight_init=init.kaiming_normal_, use_layer_norm=False)
        >>>
        >>> # For single layer (no hidden)
        >>> linear = MLP(64, 64, hidden_dims=[], use_layer_norm=False)
        >>>
        >>> # Final-only LayerNorm (MeshGraphNets / WindFarm-style)
        >>> mlp = MLP(64, 64, hidden_dims=[128, 128, 128], activation='relu', final_norm='layer')
        >>>
        >>> # Per-layer normalization (MeshGraphNets-style with norms)
        >>> mlp = MLP(64, 64, hidden_dims=[128, 128, 128], activation='relu',
        ...           norms=[None, None, None, 'layer'])  # 4 layers, norm only on final
        >>>
        >>> # Pre-activation pattern (Act → Linear) for AdaLN-style modulation
        >>> mlp = MLP(64, 192, hidden_dims=[], pre_activation='silu')
        >>> # Produces: SiLU → Linear(64, 192)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: Union[str, nn.Module, Callable, Sequence[Union[str, nn.Module, Callable]]] = 'gelu',
        dropout: Union[float, Sequence[float]] = 0.0,
        norm: Any = _UNSET,
        final_activation: Any = _UNSET,
        final_dropout: float = 0.0,
        final_norm: Any = _UNSET,
        norms: Optional[Sequence[Any]] = None,
        linear_factory: Optional[Callable[[int, int], nn.Module]] = None,
        use_layer_norm: Optional[bool] = True,
        pre_activation: Any = None,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()

        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        total_layers = len(dims) - 1

        if linear_factory is None:
            linear_factory = lambda a, b: nn.Linear(a, b)

        hidden_layers = max(total_layers - 1, 0)

        # Resolve normalization specs
        # Priority: norms > explicit norm/final_norm > use_layer_norm default
        if norms is not None:
            if len(norms) != total_layers:
                raise ValueError(f"norms sequence must have length {total_layers}, got {len(norms)}")
            layer_norms = list(norms)
        else:
            if norm is _UNSET and final_norm is _UNSET:
                norm = 'layer' if use_layer_norm else None
                final_norm = None
            else:
                if norm is _UNSET:
                    norm = None
                if final_norm is _UNSET:
                    final_norm = None
            hidden_norms = self._expand_spec(norm, hidden_layers, "norm")
            layer_norms = hidden_norms + [final_norm]

        if final_activation is _UNSET:
            final_activation = None

        hidden_activations = self._expand_spec(activation, hidden_layers, "activation")
        hidden_dropouts = self._expand_numeric_spec(dropout, hidden_layers, "dropout")

        # Pre-activation mode: Act → Linear per layer
        if pre_activation is not None:
            pre_act_module = self._make_activation(pre_activation)
            for i in range(total_layers):
                if pre_act_module is not None:
                    layers.append(pre_act_module)
                layers.append(linear_factory(dims[i], dims[i + 1]))
        else:
            # Standard post-activation mode: Linear → Act
            for i in range(total_layers):
                layers.append(linear_factory(dims[i], dims[i + 1]))

                is_last = i == total_layers - 1
                norm_module = self._make_norm(layer_norms[i], dims[i + 1])
                if norm_module is not None:
                    layers.append(norm_module)

                if not is_last:
                    act_module = self._make_activation(hidden_activations[i])
                    if act_module is not None:
                        layers.append(act_module)
                    if hidden_dropouts[i] > 0:
                        layers.append(nn.Dropout(hidden_dropouts[i]))
                else:
                    act_module = self._make_activation(final_activation)
                    if act_module is not None:
                        layers.append(act_module)
                    if final_dropout > 0:
                        layers.append(nn.Dropout(final_dropout))

        self._net = nn.Sequential(*layers)
        self.weight_init = weight_init
        self.bias_init = bias_init

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self.weight_init(m.weight)
            if m.bias is not None:
                self.bias_init(m.bias)

    @staticmethod
    def _expand_spec(spec: Any, n: int, name: str) -> List[Any]:
        if n == 0:
            return []
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, nn.Module)):
            if len(spec) != n:
                raise ValueError(f"{name} sequence must have length {n}, got {len(spec)}")
            return list(spec)
        return [spec] * n

    @staticmethod
    def _expand_numeric_spec(spec: Union[float, Sequence[float]], n: int, name: str) -> List[float]:
        if n == 0:
            return []
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            if len(spec) != n:
                raise ValueError(f"{name} sequence must have length {n}, got {len(spec)}")
            return [float(x) for x in spec]
        return [float(spec)] * n

    @staticmethod
    def _make_activation(spec: Any) -> Optional[nn.Module]:
        if spec is None:
            return None
        if isinstance(spec, nn.Module):
            return spec
        if isinstance(spec, str):
            mapping = {
                'relu': nn.ReLU,
                'gelu': nn.GELU,
                'silu': nn.SiLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'sin': SinActivation,
            }
            if spec not in mapping:
                raise ValueError(f"Unknown activation: {spec}")
            return mapping[spec]()
        if callable(spec):
            module = spec()
            if not isinstance(module, nn.Module):
                raise ValueError("Callable activation spec must return nn.Module")
            return module
        raise ValueError(f"Unsupported activation spec: {spec}")

    @staticmethod
    def _make_norm(spec: Any, dim: int) -> Optional[nn.Module]:
        if spec is None:
            return None
        if isinstance(spec, nn.Module):
            return spec
        if isinstance(spec, str):
            if spec == 'layer':
                return nn.LayerNorm(dim)
            raise ValueError(f"Unknown normalization spec: {spec}")
        if callable(spec):
            module = spec(dim)
            if not isinstance(module, nn.Module):
                raise ValueError("Callable norm spec must return nn.Module")
            return module
        raise ValueError(f"Unsupported normalization spec: {spec}")

    @property
    def layers(self) -> nn.Sequential:
        """The underlying ``nn.Sequential`` of layers (read-only)."""
        return self._net

    @property
    def in_features(self) -> int:
        """Input dimension inferred from the first Linear layer."""
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        raise AttributeError("MLP has no Linear layer")

    @property
    def out_features(self) -> int:
        """Output dimension inferred from the last Linear layer."""
        last_linear: Optional[nn.Linear] = None
        for m in self._net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise AttributeError("MLP has no Linear layer")
        return last_linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)
