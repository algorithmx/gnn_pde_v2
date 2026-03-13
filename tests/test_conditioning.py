"""
Tests for conditioning protocol and implementations.
"""

import pytest
import torch
from torch import Tensor


class TestModulation:
    """Test Modulation dataclass behavior."""

    def test_empty_modulation(self):
        """Empty modulation has None for all fields."""
        from gnn_pde_v2.components.transformer import Modulation

        mod = Modulation()
        assert mod.shift is None
        assert mod.scale is None
        assert mod.gate is None
        assert mod.cross_kv is None

    def test_modulation_with_tensors(self):
        """Modulation can hold tensors."""
        from gnn_pde_v2.components.transformer import Modulation

        shift = torch.randn(2, 10, 64)
        scale = torch.randn(3, 10, 64)
        gate = torch.randn(3, 10, 64)

        mod = Modulation(shift=shift, scale=scale, gate=gate)

        assert torch.equal(mod.shift, shift)
        assert torch.equal(mod.scale, scale)
        assert torch.equal(mod.gate, gate)

    def test_modulation_is_dataclass(self):
        """Modulation is a dataclass with expected fields."""
        from dataclasses import fields
        from gnn_pde_v2.components.transformer import Modulation
        field_names = {f.name for f in fields(Modulation)}
        assert field_names == {'shift', 'scale', 'gate', 'cross_kv'}


class TestConditioningProtocol:
    """Test ConditioningProtocol is an abstract base class."""

    def test_is_abstract(self):
        """Cannot instantiate ConditioningProtocol directly."""
        from gnn_pde_v2.components.transformer import ConditioningProtocol

        with pytest.raises(TypeError):
            ConditioningProtocol()

    def test_is_nn_module_subclass(self):
        """ConditioningProtocol inherits from nn.Module."""
        import torch.nn as nn
        from gnn_pde_v2.components.transformer import ConditioningProtocol
        assert issubclass(ConditioningProtocol, nn.Module)

    def test_is_abc_subclass(self):
        """ConditioningProtocol inherits from ABC."""
        from abc import ABC
        from gnn_pde_v2.components.transformer import ConditioningProtocol
        assert issubclass(ConditioningProtocol, ABC)
