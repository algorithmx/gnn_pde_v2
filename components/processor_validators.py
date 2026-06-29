"""Validation helpers for graph processor blocks.

These functions keep construction-time validation and initialization checks out
of the main processor block classes so those classes can focus on message
passing behavior.
"""

from typing import Optional

import torch
import torch.nn as nn

from .edge_processors import EdgeMessageProcessor
from .node_updaters import NodeUpdateStrategy


__all__ = [
    "infer_module_tensor_kwargs",
    "reset_linear_layers",
    "validate_edge_message_processor",
    "validate_node_update_strategy",
    "verify_edge_message_pipeline",
    "verify_edge_transform_output",
]


def validate_edge_message_processor(
    edge_processor: EdgeMessageProcessor,
    latent_dim: int,
) -> int:
    """Validate an edge-message processor against block expectations."""
    if not isinstance(edge_processor, nn.Module):
        raise TypeError(
            "edge_processor must be an nn.Module to preserve parameter registration "
            "and torch.compile() specialization"
        )
    if not isinstance(edge_processor, EdgeMessageProcessor):
        raise TypeError(
            "edge_processor must satisfy EdgeMessageProcessor protocol: "
            "provide weight_out_dim and forward(src_x, edge_weights)"
        )
    weight_out_dim = edge_processor.weight_out_dim
    if not isinstance(weight_out_dim, int) or weight_out_dim <= 0:
        raise ValueError(
            f"edge_processor.weight_out_dim must be a positive int, got {weight_out_dim!r}"
        )
    processor_latent_dim = edge_processor.latent_dim
    if processor_latent_dim != latent_dim:
        raise ValueError(
            "edge_processor latent_dim must match block latent_dim: "
            f"got {processor_latent_dim} vs {latent_dim}"
        )
    return weight_out_dim


def validate_node_update_strategy(
    node_updater: NodeUpdateStrategy,
    latent_dim: int,
) -> None:
    """Validate an injected node-update strategy against block expectations.

    The ``NodeUpdateStrategy`` ABC guarantees ``forward`` and a positive
    ``latent_dim`` via inheritance; this helper additionally checks that the
    instance's ``latent_dim`` matches the block at construction time.
    """
    if not isinstance(node_updater, nn.Module):
        raise TypeError(
            "node_updater must be an nn.Module to preserve parameter registration "
            "and torch.compile() specialization"
        )
    if not isinstance(node_updater, NodeUpdateStrategy):
        raise TypeError(
            "node_updater must satisfy NodeUpdateStrategy protocol: "
            "provide an int latent_dim and forward(nodes, aggregated)"
        )
    updater_latent_dim = node_updater.latent_dim
    if not isinstance(updater_latent_dim, int) or updater_latent_dim <= 0:
        raise ValueError(
            f"node_updater.latent_dim must be a positive int, got {updater_latent_dim!r}"
        )
    if updater_latent_dim != latent_dim:
        raise ValueError(
            "node_updater latent_dim must match block latent_dim: "
            f"got {updater_latent_dim} vs {latent_dim}"
        )



def infer_module_tensor_kwargs(
    module: nn.Module,
) -> dict[str, torch.device | torch.dtype]:
    """Infer device and floating dtype from a module for eager checks."""
    ref: Optional[torch.Tensor] = next(module.parameters(), None)
    if ref is None:
        ref = next(module.buffers(), None)
    if ref is None:
        return {}
    tensor_kwargs: dict[str, torch.device | torch.dtype] = {"device": ref.device}
    if torch.is_floating_point(ref):
        tensor_kwargs["dtype"] = ref.dtype
    return tensor_kwargs



def verify_edge_message_pipeline(
    *,
    edge_weight_net: nn.Module,
    edge_processor: EdgeMessageProcessor,
    latent_dim: int,
    edge_latent_dim: int,
    num_edges: int,
) -> None:
    """Eagerly verify the edge-weight-net → edge-processor pipeline."""
    if num_edges <= 0:
        raise ValueError(f"num_edges must be positive, got {num_edges}")

    tensor_kwargs = infer_module_tensor_kwargs(edge_weight_net)
    edge_features = torch.randn(num_edges, edge_latent_dim, **tensor_kwargs)
    src_x = torch.randn(num_edges, latent_dim, **tensor_kwargs)

    with torch.no_grad():
        edge_weights = edge_weight_net(edge_features)
        out = edge_processor(src_x, edge_weights)

    if edge_weights.ndim != 2:
        raise ValueError(
            "edge_weight_net must return rank-2 tensor [E, weight_out_dim] during verification"
        )
    if edge_weights.shape != (num_edges, edge_processor.weight_out_dim):
        raise ValueError(
            "edge_weight_net and edge_processor disagree on weight shape during verification: "
            f"got {tuple(edge_weights.shape)} vs ({num_edges}, {edge_processor.weight_out_dim})"
        )
    if out.ndim != 2 or out.shape != (num_edges, latent_dim):
        raise ValueError(
            "edge message pipeline must return shape [E, latent_dim] during verification: "
            f"got {tuple(out.shape)} vs ({num_edges}, {latent_dim})"
        )



def verify_edge_transform_output(
    *,
    edge_transform: nn.Module,
    input_dim: int,
    expected_dim: int,
    num_edges: int,
) -> None:
    """Eagerly verify that an edge transform returns the expected latent size."""
    if num_edges <= 0:
        raise ValueError(f"num_edges must be positive, got {num_edges}")

    tensor_kwargs = infer_module_tensor_kwargs(edge_transform)
    dummy_features = torch.randn(num_edges, input_dim, **tensor_kwargs)

    with torch.no_grad():
        output = edge_transform(dummy_features)

    if output.shape != (num_edges, expected_dim):
        raise ValueError(
            "edge_transform output shape mismatch: "
            f"expected ({num_edges}, {expected_dim}), got {tuple(output.shape)}. "
            f"Ensure out_dim={expected_dim} matches latent_dim."
        )



def reset_linear_layers(module: nn.Module) -> None:
    """Reset all linear submodules with Xavier/zero initialization."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
