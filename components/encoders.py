"""
Encoders for the GNN-PDE framework.

Canonical MLP building blocks and graph encoder wrappers.
"""

from dataclasses import replace
from typing import List, Optional, Callable, Sequence, Union, Any
from functools import partial
import torch
import torch.nn as nn
import torch.nn.init as init

from ..core.graph import GraphsTuple


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
        linear_factory: Callable creating the affine layer for each stage
        use_layer_norm: Backward-compatible alias for hidden LayerNorm behavior
        weight_init: Weight initialization function (default: xavier_uniform_)
        bias_init: Bias initialization function (default: constant 0)
        
    Example:
        >>> # Standard usage
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
        linear_factory: Optional[Callable[[int, int], nn.Module]] = None,
        use_layer_norm: Optional[bool] = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()
        
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]

        if linear_factory is None:
            linear_factory = lambda a, b: nn.Linear(a, b)

        hidden_layers = max(len(dims) - 2, 0)

        if norm is _UNSET and final_norm is _UNSET:
            norm = 'layer' if use_layer_norm else None
            final_norm = None
        else:
            if norm is _UNSET:
                norm = None
            if final_norm is _UNSET:
                final_norm = None

        if final_activation is _UNSET:
            final_activation = None

        hidden_activations = self._expand_spec(activation, hidden_layers, "activation")
        hidden_dropouts = self._expand_numeric_spec(dropout, hidden_layers, "dropout")
        hidden_norms = self._expand_spec(norm, hidden_layers, "norm")
        # Build layers
        for i in range(len(dims) - 1):
            layers.append(linear_factory(dims[i], dims[i + 1]))
            
            is_last = i == len(dims) - 2
            if not is_last:
                norm_module = self._make_norm(hidden_norms[i], dims[i + 1])
                if norm_module is not None:
                    layers.append(norm_module)
                act_module = self._make_activation(hidden_activations[i])
                if act_module is not None:
                    layers.append(act_module)
                if hidden_dropouts[i] > 0:
                    layers.append(nn.Dropout(hidden_dropouts[i]))
            else:
                norm_module = self._make_norm(final_norm, dims[i + 1])
                if norm_module is not None:
                    layers.append(norm_module)
                act_module = self._make_activation(final_activation)
                if act_module is not None:
                    layers.append(act_module)
                if final_dropout > 0:
                    layers.append(nn.Dropout(final_dropout))
        
        self.net = nn.Sequential(*layers)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPEncoder(nn.Module):
    """
    Simple graph encoder with separate node and optional edge MLPs.
    """

    def __init__(
        self,
        node_in_dim: int,
        node_out_dim: int,
        edge_in_dim: Optional[int] = None,
        edge_out_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()

        hidden_dims = [128, 128] if hidden_dims is None else hidden_dims

        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=node_out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )

        if edge_in_dim is not None and edge_out_dim is not None:
            self.edge_encoder = MLP(
                in_dim=edge_in_dim,
                out_dim=edge_out_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                weight_init=weight_init,
                bias_init=bias_init,
            )
        else:
            self.edge_encoder = None

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if self.edge_encoder is not None and graph.edges is not None else None
        return replace(graph, nodes=nodes, edges=edges)


class MLPMeshEncoder(nn.Module):
    """
    MeshGraphNets-style encoder with separate node, edge, and optional global MLPs.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        global_in_dim: Optional[int],
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'gelu',
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        weight_init: Callable = init.xavier_uniform_,
        bias_init: Callable = partial(init.constant_, val=0.0),
    ):
        super().__init__()

        hidden_dims = [128] if hidden_dims is None else hidden_dims

        self.node_encoder = MLP(
            in_dim=node_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.edge_encoder = MLP(
            in_dim=edge_in_dim,
            out_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.global_encoder = (
            MLP(
                in_dim=global_in_dim,
                out_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                weight_init=weight_init,
                bias_init=bias_init,
            )
            if global_in_dim is not None
            else None
        )

    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        nodes = self.node_encoder(graph.nodes) if graph.nodes is not None else None
        edges = self.edge_encoder(graph.edges) if graph.edges is not None else None
        globals_ = self.global_encoder(graph.globals) if self.global_encoder is not None and graph.globals is not None else None
        return replace(graph, nodes=nodes, edges=edges, globals=globals_)


# Convenience functions for common encoder patterns

def make_mlp_encoder(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    **mlp_kwargs
) -> MLP:
    """
    Create an MLP encoder with standard architecture.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers (including output, excluding input)
        **mlp_kwargs: Additional arguments for MLP
        
    Returns:
        MLP encoder
        
    Example:
        >>> encoder = make_mlp_encoder(10, 128, hidden_dim=128, num_layers=2)
        >>> # Creates: Linear(10, 128) -> LayerNorm -> GELU -> Linear(128, 128)
    """
    if num_layers == 1:
        hidden_dims = []
    else:
        hidden_dims = [hidden_dim] * (num_layers - 1)
    
    return MLP(in_dim, out_dim, hidden_dims, **mlp_kwargs)
