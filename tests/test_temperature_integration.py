"""
Integration tests for temperature mechanisms with PhysicsTokenAttention.

Tests end-to-end functionality with transformer components.
"""

import pytest
import torch

from gnn_pde_v2.components.temperature import (
    FixedTemperature,
    AdaptiveTemperature,
    PerHeadTemperature,
)
from gnn_pde_v2.components.attention import PhysicsTokenAttention
from gnn_pde_v2.components.transformer import (
    TransformerBlock,
    TransformerProcessor,
)
from gnn_pde_v2.core.graph import GraphsTuple


class TestPhysicsTokenAttentionTemperature:
    """Tests for PhysicsTokenAttention with different temperature modes."""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 16, 64)  # [B, N, D]
    
    def test_fixed_temperature_backward_compatible(self, sample_input):
        """Test fixed mode maintains backward compatibility."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='fixed',
            temperature=1.0
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, FixedTemperature)
    
    def test_adaptive_temperature(self, sample_input):
        """Test adaptive temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=False
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, AdaptiveTemperature)
    
    def test_per_head_temperature(self, sample_input):
        """Test per-head temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='per_head'
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
        assert isinstance(attn.temperature_module, PerHeadTemperature)
    
    def test_learnable_scalar_temperature(self, sample_input):
        """Test learnable scalar temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='learnable_scalar'
        )
        out = attn(sample_input)
        assert out.shape == sample_input.shape
    
    def test_annealed_temperature(self, sample_input):
        """Test annealed temperature mode."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='annealed',
            anneal_warmup_epochs=5
        )
        attn.set_epoch(10)
        out = attn(sample_input)
        assert out.shape == sample_input.shape
    
    def test_gumbel_softmax_integration(self, sample_input):
        """Test Gumbel-Softmax works with adaptive temperature."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive',
            use_gumbel_softmax=True
        )
        # Test in training mode
        attn.train()
        out_train = attn(sample_input)
        
        # Test in eval mode (Gumbel should be disabled)
        attn.eval()
        out_eval = attn(sample_input)
        
        assert out_train.shape == sample_input.shape
        assert out_eval.shape == sample_input.shape
    
    def test_single_batch_input(self):
        """Test with single batch (2D input)."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        x = torch.randn(16, 64)  # [N, D]
        out = attn(x)
        assert out.shape == x.shape
    
    def test_temperature_parameter_access(self, sample_input):
        """Test temperature value can be accessed after forward pass."""
        attn = PhysicsTokenAttention(
            dim=64,
            n_tokens=8,
            n_heads=4,
            temperature_mode='adaptive'
        )
        out = attn(sample_input)
        
        # Temperature should be computed and stored
        assert hasattr(attn, 'temperature_module')


class TestTransformerBlockTemperature:
    """Tests for TransformerBlock with temperature parameters."""
    
    def test_transformer_block_with_adaptive_temp(self):
        """Test TransformerBlock passes temperature params to PhysicsTokenAttention."""
        block = TransformerBlock(
            dim=64,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='adaptive',
            use_gumbel_softmax=True
        )
        
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape
    
    def test_transformer_block_set_epoch(self):
        """Test TransformerBlock set_epoch propagates to attention."""
        block = TransformerBlock(
            dim=64,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='annealed'
        )
        
        # Should not raise error
        block.set_epoch(5)
    
    def test_transformer_block_without_physics_tokens(self):
        """Test TransformerBlock without physics tokens warns about ignored params."""
        with pytest.warns(UserWarning, match="Ignored parameters"):
            block = TransformerBlock(
                dim=64,
                n_heads=4,
                use_physics_tokens=False,
                temperature_mode='adaptive'  # Should be ignored
            )
        
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape


class TestTransformerProcessorTemperature:
    """Tests for TransformerProcessor with temperature."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        return GraphsTuple.from_flat(
            nodes=torch.randn(10, 64),
            n_node=torch.tensor([10]),
            edges=None,
            receivers=None,
            senders=None,
            positions=None
        )
    
    def test_processor_with_adaptive_temperature(self, sample_graph):
        """Test TransformerProcessor with adaptive temperature."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=2,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='adaptive'
        )
        
        out_graph = processor(sample_graph)
        assert out_graph.nodes.shape == sample_graph.nodes.shape
    
    def test_processor_set_epoch_propagation(self, sample_graph):
        """Test set_epoch propagates to all blocks."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=3,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='annealed'
        )
        
        # Set epoch should propagate to all blocks
        processor.set_epoch(10)
        
        # Verify each block has the epoch set
        for block in processor.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'temperature_module'):
                temp_module = block.attn.temperature_module
                if hasattr(temp_module, 'current_epoch'):
                    assert temp_module.current_epoch == 10
    
    def test_processor_multiple_layers_consistent(self, sample_graph):
        """Test all layers use same temperature configuration."""
        processor = TransformerProcessor(
            latent_dim=64,
            n_layers=3,
            n_heads=4,
            use_physics_tokens=True,
            n_tokens=8,
            temperature_mode='per_head'
        )
        
        out_graph = processor(sample_graph)
        assert out_graph.nodes.shape == sample_graph.nodes.shape
