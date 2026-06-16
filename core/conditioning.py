"""Conditioning primitives for modulation-based architectures.

These types are **nominal** (inheritance-based), unlike the *structural*
``typing.Protocol`` interfaces in :mod:`gnn_pde_v2.core.protocols`.  They live
in their own module precisely so that ``protocols.py`` can remain a pure
structural-typing surface (see ``docs/protocol_issues_2026_06.md`` §5).

:class:`ConditioningProtocol` is an abstract ``nn.Module`` base class:
implementations must *inherit* from it (and bind the ``CondT`` type variable)
rather than merely duck-typing a ``forward`` method.  This is intentional —
conditioning modules need to participate in PyTorch parameter registration, so
nominal typing is the right tool here.

For backwards compatibility, both names are re-exported from
:mod:`gnn_pde_v2.core.protocols`, so existing
``from gnn_pde_v2.core.protocols import Modulation, ConditioningProtocol``
imports continue to work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch.nn as nn
from torch import Tensor

#: Type variable for the condition input accepted by a :class:`ConditioningProtocol`.
#: Bind it to a concrete type when subclassing::
#:
#:     class MyConditioner(ConditioningProtocol[Tensor]):
#:         def forward(self, condition: Tensor) -> Modulation: ...
CondT = TypeVar("CondT")


@dataclass
class Modulation:
    """Container for transformer modulation parameters.

    All fields are optional; absent fields mean no modulation is applied
    for that axis.

    Args:
        shift: Additive bias term  ``[..., dim]``
        scale: Multiplicative scale term ``[..., dim]``
        gate: Post-residual gate term ``[..., dim]``
        cross_kv: Pre-computed key/value tensors for cross-attention
    """
    shift: Tensor | None = None
    scale: Tensor | None = None
    gate: Tensor | None = None
    cross_kv: Tensor | None = None


class ConditioningProtocol(nn.Module, ABC, Generic[CondT]):
    """Abstract base class for conditioning mechanisms.

    .. note::

        Despite the historical name, this is a *nominal* ABC, **not** a
        structural ``typing.Protocol``.  Implementations must subclass it.
        It is kept in :mod:`gnn_pde_v2.core.conditioning` (and only
        re-exported from :mod:`gnn_pde_v2.core.protocols` for backwards
        compatibility) so the protocol module stays purely structural.

    This class is *generic* over ``CondT``, the type of the condition
    accepted by :meth:`forward`.  Subclasses should bind ``CondT`` to a
    concrete type so that callers can tell at a glance what they must
    provide::

        # Accepts any optional input — no condition is required.
        class ZeroConditioning(ConditioningProtocol[object]): ...

        # Requires a floating-point tensor of shape [..., cond_dim].
        class AdaLNConditioning(ConditioningProtocol[Tensor]): ...

    Using the wrong condition type for a given implementation will be
    caught by static analysis (mypy / pyright) rather than producing a
    silent runtime error.

    Example::

        class MyConditioning(ConditioningProtocol[Tensor]):
            def __init__(self, cond_dim: int, out_dim: int):
                super().__init__()
                self.proj = nn.Linear(cond_dim, out_dim * 2)

            def forward(self, condition: Tensor) -> Modulation:
                shift, scale = self.proj(condition).chunk(2, dim=-1)
                return Modulation(shift=shift, scale=scale)
    """

    @abstractmethod
    def forward(self, condition: CondT) -> Modulation:  # type: ignore[override]
        """Convert condition to modulation parameters.

        Args:
            condition: Conditioning input of type ``CondT`` (declared by
                the concrete subclass).

        Returns:
            Modulation with optional shift/scale/gate/cross_kv fields.
        """
        ...


__all__ = [
    "CondT",
    "Modulation",
    "ConditioningProtocol",
]
