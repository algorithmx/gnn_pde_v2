"""
Conditioning components for modulation-based architectures.

Provides AdaLN (Adaptive Layer Normalization) and FiLM (Feature-wise Linear
Modulation) conditioning mechanisms for transformer and other architectures.

Note: Modulation and ConditioningProtocol are defined in core.protocols.
Import them from there: ``from gnn_pde_v2.core.protocols import Modulation, ConditioningProtocol``
"""

import torch.nn as nn
from torch import Tensor

from ..core.protocols import ConditioningProtocol, Modulation


# =============================================================================
# Conditioning Protocol
# =============================================================================
# Modulation and ConditioningProtocol are defined in core/protocols.py.
# Import them from there: ``from gnn_pde_v2.core.protocols import ConditioningProtocol``


class ZeroConditioning(ConditioningProtocol[object]):
    """Identity conditioning — no modulation applied.

    Accepts (and ignores) any condition value, including ``None``.
    Suitable as a drop-in for any slot typed as
    ``ConditioningProtocol[T]`` for any ``T``.
    """

    def forward(self, condition: object = None) -> Modulation:  # type: ignore[override]
        return Modulation()


class _AdaLNConditioningBase(ConditioningProtocol[Tensor]):
    """Base single-source AdaLN conditioning.

    Subclasses set the class variable ``_n_chunks`` to control the output:

    * ``_n_chunks = 3`` — produces ``(shift, scale, gate) × 2`` (one pair per
      block: attention + MLP).  Gate enables in-place residual scaling.
    * ``_n_chunks = 2`` — produces ``(shift, scale) × 2`` with ``gate=None``.
      Suitable for final projection layers without a residual connection.

    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    _n_chunks: int  # 3 → with gate, 2 → gate-free

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * self._n_chunks * out_dim),
        )
        nn.init.zeros_(self.proj[1].weight)
        nn.init.zeros_(self.proj[1].bias)

    def forward(self, condition: Tensor) -> Modulation:
        n = self._n_chunks
        params = self.proj(condition).chunk(2 * n, dim=-1)
        return Modulation(
            shift=torch.cat([params[0], params[n]], dim=-1),
            scale=torch.cat([params[1], params[n + 1]], dim=-1),
            gate=torch.cat([params[2], params[n + 2]], dim=-1) if n == 3 else None,
        )


class AdaLNConditioning(_AdaLNConditioningBase):
    """Single-source AdaLN conditioning with gate: ``(shift, scale, gate) × 2``.

    Standard choice for transformer blocks with residual connections.
    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    _n_chunks = 3


class AdaLNConditioningNoGate(_AdaLNConditioningBase):
    """Single-source AdaLN conditioning without gate: ``(shift, scale) × 2``.

    Use for final projection layers or any context without a residual gate.
    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    _n_chunks = 2


class _DualAdaLNConditioningBase(ConditioningProtocol[Tensor]):
    """Base dual-source AdaLN conditioning (Unisolver-style: μ + f).

    Subclasses set the class variable ``_n_chunks`` to control the output:

    * ``_n_chunks = 3`` — produces ``(shift, scale, gate)`` with gate enabled.
    * ``_n_chunks = 2`` — produces ``(shift, scale)`` with ``gate=None``.

    Condition type: ``Tensor`` of shape ``[..., mu_dim + f_dim]``.
    The last dimension is split as ``condition[..., :mu_dim]`` (domain-wise μ)
    and ``condition[..., mu_dim:]`` (point-wise f). A ``ValueError`` is raised
    at runtime if the last dimension does not equal ``mu_dim + f_dim``.
    """

    _n_chunks: int  # 3 → with gate, 2 → gate-free

    def __init__(
        self,
        mu_dim: int,
        f_dim: int,
        out_dim: int,
        split_ratio: float = 0.25,
    ):
        super().__init__()
        self.mu_dim = mu_dim
        self.f_dim = f_dim
        self.split_ratio = split_ratio

        mu_out = int(out_dim * split_ratio)
        f_out = out_dim - mu_out

        self.proj_mu = nn.Sequential(nn.SiLU(), nn.Linear(mu_dim, self._n_chunks * mu_out))
        self.proj_f = nn.Sequential(nn.SiLU(), nn.Linear(f_dim, self._n_chunks * f_out))

        for proj in [self.proj_mu, self.proj_f]:
            nn.init.zeros_(proj[1].weight)
            nn.init.zeros_(proj[1].bias)

    def forward(self, condition: Tensor) -> Modulation:
        expected = self.mu_dim + self.f_dim
        if condition.shape[-1] != expected:
            raise ValueError(
                f"{type(self).__name__}.forward expects condition.shape[-1] == "
                f"mu_dim + f_dim == {expected}, got {condition.shape[-1]}."
            )
        mu = condition[..., : self.mu_dim]
        f = condition[..., self.mu_dim :]

        n = self._n_chunks
        params_mu = self.proj_mu(mu).chunk(n, dim=-1)
        params_f = self.proj_f(f).chunk(n, dim=-1)

        return Modulation(
            shift=torch.cat([params_mu[0], params_f[0]], dim=-1),
            scale=torch.cat([params_mu[1], params_f[1]], dim=-1),
            gate=torch.cat([params_mu[2], params_f[2]], dim=-1) if n == 3 else None,
        )


class DualAdaLNConditioning(_DualAdaLNConditioningBase):
    """Dual-source AdaLN conditioning with gate: ``(shift, scale, gate)``.

    Standard choice for transformer blocks with residual connections.
    Condition type: ``Tensor`` of shape ``[..., mu_dim + f_dim]``.
    """

    _n_chunks = 3


class DualAdaLNConditioningNoGate(_DualAdaLNConditioningBase):
    """Dual-source AdaLN conditioning without gate: ``(shift, scale)``.

    Use for final projection layers or any context without a residual gate.
    Condition type: ``Tensor`` of shape ``[..., mu_dim + f_dim]``.
    """

    _n_chunks = 2


class FiLMConditioning(ConditioningProtocol[Tensor]):
    """FiLM-style conditioning (feature-wise linear modulation).

    Condition type: ``Tensor`` of shape ``[..., cond_dim]``.
    """

    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, out_dim)
        self.beta_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, condition: Tensor) -> Modulation:
        return Modulation(
            shift=self.beta_proj(condition),
            scale=self.gamma_proj(condition),
            gate=None,
        )


# =============================================================================
# Helper Functions
# =============================================================================

def apply_modulation(x: Tensor, mod: Modulation) -> Tensor:
    """Apply modulation to a tensor.
    
    Applies shift and scale modulation to tensor x:
        out = x * (1 + scale) + shift
    
    Args:
        x: Input tensor of shape [..., D]
        mod: Modulation containing optional shift and scale
        
    Returns:
        Modulated tensor of same shape as x
    """
    if mod.scale is not None:
        x = x * (1 + mod.scale)
    if mod.shift is not None:
        x = x + mod.shift
    return x


# Backward-compatible alias (deprecated, use apply_modulation)
_apply_modulation = apply_modulation


