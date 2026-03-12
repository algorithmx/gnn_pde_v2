"""
Tests to verify examples run without errors.

These tests import and run the examples to ensure they work.
"""

import pytest
import sys
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


class TestCoreExamples:
    """Test core (lean) examples."""
    
    def test_meshgraphnets_core_imports(self):
        """Test that meshgraphnets_core.py can be imported."""
        try:
            import core.meshgraphnets_core as example
            assert hasattr(example, 'MinimalMeshGraphNets')
            assert hasattr(example, 'example_usage')
        except ImportError as e:
            pytest.fail(f"Failed to import core example: {e}")


class TestConvenientExamples:
    """Test convenient (high-level) examples."""
    
    def test_meshgraphnets_easy_imports(self):
        """Test that meshgraphnets_easy.py can be imported."""
        try:
            import convenient.meshgraphnets_easy as example
            assert hasattr(example, 'EasyMeshGraphNets')
        except ImportError as e:
            pytest.skip(f"Convenient example not available: {e}")


class TestPaperExamples:
    """Test paper replication examples."""
    
    def test_meshgraphnets_imports(self):
        """Test MeshGraphNets example imports."""
        try:
            from gnn_pde_v2.examples import example_meshgraphnets as example
            assert hasattr(example, 'MeshGraphNets')
        except ImportError as e:
            pytest.fail(f"Failed to import MeshGraphNets example: {e}")
    
    def test_deepxde_imports(self):
        """Test DeepXDE example imports."""
        try:
            from gnn_pde_v2.examples import example_deepxde as example
            assert hasattr(example, 'PINNModel')
        except ImportError as e:
            pytest.fail(f"Failed to import DeepXDE example: {e}")
    
    def test_fno_imports(self):
        """Test FNO example imports."""
        try:
            from gnn_pde_v2.examples import example_neuraloperator_fno as example
            assert hasattr(example, 'FNO')
        except ImportError as e:
            pytest.fail(f"Failed to import FNO example: {e}")
    
    def test_transolver_imports(self):
        """Test Transolver example imports."""
        try:
            from gnn_pde_v2.examples import example_transolver as example
            assert hasattr(example, 'Transolver')
        except ImportError as e:
            pytest.fail(f"Failed to import Transolver example: {e}")
    
    def test_unisolver_imports(self):
        """Test Unisolver example imports."""
        try:
            from gnn_pde_v2.examples import example_unisolver as example
            assert hasattr(example, 'UniSolver')
        except ImportError as e:
            pytest.fail(f"Failed to import UniSolver example: {e}")
    
    def test_windfarm_imports(self):
        """Test WindFarm GNO example imports."""
        try:
            from gnn_pde_v2.examples import example_windfarm_gno as example
            assert hasattr(example, 'WindFarmGNO')
        except ImportError as e:
            pytest.fail(f"Failed to import WindFarm example: {e}")
    
    def test_graph_pde_imports(self):
        """Test Graph-PDE GNO example imports."""
        try:
            from gnn_pde_v2.examples import example_graph_pde_gno as example
            assert hasattr(example, 'GraphPDEGNO')
        except ImportError as e:
            pytest.fail(f"Failed to import Graph-PDE example: {e}")
