"""
Tests for multiscale components.

Covers: GraphPool, GraphUnpool, GraphUNetProcessor, MGKNProcessor,
        MultiResolutionFNOBlock, UFNOBlock, HierarchicalFNOBlock,
        MiniUNet, HierarchicalGraph utilities, and MultiscaleFNO model.

References:
- Gao & Ji, "Graph U-Nets", ICML 2019
- Li et al., "Multipole Graph Neural Operator", NeurIPS 2020
- Wen et al., "U-FNO", 2022
"""

import pytest
import torch
import torch.nn as nn
import time

from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components.multiscale import (
    GraphPool,
    GraphUnpool,
    GraphUNetProcessor,
    MGKNProcessor,
    MultiResolutionFNOBlock,
    UFNOBlock,
    HierarchicalFNOBlock,
    MiniUNet,
    HierarchicalGraph,
    build_hierarchical_graphs,
    compute_transition_matrix,
    restrict_to_coarse,
    prolong_to_fine,
)
from gnn_pde_v2.models.multiscale_fno import MultiscaleFNO


# =============================================================================
# TestGraphPool - Graph U-Nets (ICML 2019)
# =============================================================================

class TestGraphPool:
    """Test GraphPool layer from Graph U-Nets."""

    def test_forward_basic(self, device):
        """Pool 100 nodes to 50."""
        num_nodes = 100
        feature_dim = 16
        k = 50
        
        # Create sample graph
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        senders = torch.randint(0, num_nodes, (200,), device=device)
        receivers = torch.randint(0, num_nodes, (200,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(200, 4, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([200], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        pooled_graph, indices = pool(graph)
        
        assert pooled_graph.nodes.shape[0] == k
        assert pooled_graph.nodes.shape[1] == feature_dim

    def test_topk_selection(self, device):
        """Verify top-k nodes selected based on projection values."""
        num_nodes = 50
        feature_dim = 8
        k = 20
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        
        # Compute projection values manually
        proj_norm = torch.norm(pool.proj_vector)
        if proj_norm > 0:
            y = torch.matmul(nodes, pool.proj_vector) / proj_norm
        else:
            y = torch.matmul(nodes, pool.proj_vector)
        
        _, expected_indices = torch.topk(y, k, largest=True, sorted=True)
        
        pooled_graph, indices = pool(graph)
        
        # Verify selected indices match top-k
        assert torch.equal(indices, expected_indices)

    def test_gate_operation(self, device):
        """Verify sigmoid gate is applied to selected nodes."""
        num_nodes = 30
        feature_dim = 8
        k = 10
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        # With gate
        pool_with_gate = GraphPool(k=k, feature_dim=feature_dim, use_gate=True).to(device)
        pooled_with_gate, _ = pool_with_gate(graph)
        
        # Without gate
        pool_no_gate = GraphPool(k=k, feature_dim=feature_dim, use_gate=False).to(device)
        pool_no_gate.proj_vector.data = pool_with_gate.proj_vector.data.clone()
        pooled_no_gate, _ = pool_no_gate(graph)
        
        # They should be different due to gate
        assert not torch.allclose(pooled_with_gate.nodes, pooled_no_gate.nodes)

    def test_trainable_projection(self, device):
        """Verify projection vector is trainable."""
        num_nodes = 20
        feature_dim = 8
        k = 10
        
        nodes = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        pooled_graph, _ = pool(graph)
        
        loss = pooled_graph.nodes.sum()
        loss.backward()
        
        assert pool.proj_vector.grad is not None

    def test_graph_power_augmentation(self, device):
        """Test A^2 connectivity augmentation."""
        num_nodes = 20
        feature_dim = 8
        k = 10
        
        # Create a simple chain graph
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        senders = torch.arange(num_nodes - 1, device=device)
        receivers = torch.arange(1, num_nodes, device=device)
        senders = torch.cat([senders, receivers])
        receivers = torch.cat([receivers, senders[:num_nodes-1]])
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(senders.shape[0], 4, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([senders.shape[0]], device=device),
        )
        
        # With augmentation (A^2)
        pool_aug = GraphPool(k=k, feature_dim=feature_dim, connectivity_augmentation=2).to(device)
        pooled_aug, _ = pool_aug(graph)
        
        # Without augmentation
        pool_no_aug = GraphPool(k=k, feature_dim=feature_dim, connectivity_augmentation=1).to(device)
        pool_no_aug.proj_vector.data = pool_aug.proj_vector.data.clone()
        pooled_no_aug, _ = pool_no_aug(graph)
        
        # Both should have k nodes
        assert pooled_aug.nodes.shape[0] == k
        assert pooled_no_aug.nodes.shape[0] == k

    def test_edge_preservation(self, device):
        """Verify edges between selected nodes are preserved and remapped."""
        num_nodes = 20
        feature_dim = 8
        k = 10
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        # Create edges between consecutive nodes
        senders = torch.arange(num_nodes - 1, device=device)
        receivers = torch.arange(1, num_nodes, device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(senders.shape[0], 4, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([senders.shape[0]], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        pooled_graph, indices = pool(graph)
        
        # If there are edges, they should be remapped to new indices
        if pooled_graph.senders is not None and pooled_graph.receivers is not None:
            # All sender/receiver indices should be < k
            assert (pooled_graph.senders < k).all()
            assert (pooled_graph.receivers < k).all()

    def test_k_larger_than_n(self, device):
        """Test when k >= num_nodes, returns identity."""
        num_nodes = 10
        feature_dim = 8
        k = 20  # Larger than num_nodes
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        pooled_graph, indices = pool(graph)
        
        # Should return original graph
        assert pooled_graph.nodes.shape[0] == num_nodes
        assert torch.equal(indices, torch.arange(num_nodes, device=device))

    def test_indices_returned(self, device):
        """Verify indices are returned for unpooling."""
        num_nodes = 30
        feature_dim = 8
        k = 15
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        pool = GraphPool(k=k, feature_dim=feature_dim).to(device)
        pooled_graph, indices = pool(graph)
        
        # Indices should have shape (k,)
        assert indices.shape[0] == k
        # All indices should be in valid range
        assert (indices >= 0).all()
        assert (indices < num_nodes).all()


# =============================================================================
# TestGraphUnpool - Graph U-Nets
# =============================================================================

class TestGraphUnpool:
    """Test GraphUnpool layer from Graph U-Nets."""

    def test_forward_basic(self, device):
        """Unpool 50 nodes to 100."""
        original_num_nodes = 100
        pooled_num_nodes = 50
        feature_dim = 16
        
        pooled_nodes = torch.randn(pooled_num_nodes, feature_dim, device=device)
        pooled_graph = GraphsTuple.from_flat(
            nodes=pooled_nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([pooled_num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        # Create indices (first 50 nodes selected)
        indices = torch.arange(pooled_num_nodes, device=device)
        
        unpool = GraphUnpool().to(device)
        unpooled_graph = unpool(pooled_graph, indices, original_num_nodes)
        
        assert unpooled_graph.nodes.shape[0] == original_num_nodes
        assert unpooled_graph.nodes.shape[1] == feature_dim

    def test_zero_filling(self, device):
        """Verify unselected nodes are filled with zeros."""
        original_num_nodes = 20
        pooled_num_nodes = 10
        feature_dim = 8
        
        pooled_nodes = torch.ones(pooled_num_nodes, feature_dim, device=device)
        pooled_graph = GraphsTuple.from_flat(
            nodes=pooled_nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([pooled_num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        # Select every other node
        indices = torch.arange(0, original_num_nodes, 2, device=device)[:pooled_num_nodes]
        
        unpool = GraphUnpool().to(device)
        unpooled_graph = unpool(pooled_graph, indices, original_num_nodes)
        
        # Check that selected positions have ones
        assert torch.allclose(unpooled_graph.nodes[indices], torch.ones_like(unpooled_graph.nodes[indices]))
        
        # Check that unselected positions are zeros
        unselected_mask = torch.ones(original_num_nodes, dtype=torch.bool, device=device)
        unselected_mask[indices] = False
        assert (unpooled_graph.nodes[unselected_mask] == 0).all()

    def test_position_restoration(self, device):
        """Verify nodes are restored to correct positions."""
        original_num_nodes = 20
        pooled_num_nodes = 5
        feature_dim = 8
        
        # Create distinctive features
        pooled_nodes = torch.randn(pooled_num_nodes, feature_dim, device=device)
        pooled_graph = GraphsTuple.from_flat(
            nodes=pooled_nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([pooled_num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        # Select specific indices
        indices = torch.tensor([2, 5, 8, 12, 17], device=device)
        
        unpool = GraphUnpool().to(device)
        unpooled_graph = unpool(pooled_graph, indices, original_num_nodes)
        
        # Verify nodes are at correct positions
        for i, idx in enumerate(indices):
            assert torch.allclose(unpooled_graph.nodes[idx], pooled_nodes[i])

    def test_inverse_property(self, device):
        """Test that pool(unpool(x)) recovers x approximately."""
        num_nodes = 50
        feature_dim = 8
        k = 25
        
        # Create original graph
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([0], device=device),
        )
        
        # Pool
        pool = GraphPool(k=k, feature_dim=feature_dim, use_gate=False).to(device)
        pooled_graph, indices = pool(graph)
        
        # Unpool
        unpool = GraphUnpool().to(device)
        unpooled_graph = unpool(pooled_graph, indices, num_nodes)
        
        # The unpooled nodes at selected indices should match original
        assert torch.allclose(unpooled_graph.nodes[indices], nodes[indices])


# =============================================================================
# TestGraphUNetProcessor - Graph U-Nets
# =============================================================================

class TestGraphUNetProcessor:
    """Test Graph U-Net processor."""

    def test_forward_basic(self, device):
        """Test 3-level U-Net forward pass."""
        num_nodes = 64
        feature_dim = 16
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(100, feature_dim, device=device),
            receivers=torch.randint(0, num_nodes, (100,), device=device),
            senders=torch.randint(0, num_nodes, (100,), device=device),
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([100], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=3,
            nodes_per_level=[32, 16, 8],
            hidden_dim=32,
            n_layers_per_level=1,
            skip_connection="add",
        ).to(device)
        
        output = processor(graph)
        
        # Output should have same shape as input
        assert output.nodes.shape == graph.nodes.shape

    def test_skip_connections_add(self, device):
        """Test skip connections with add mode."""
        num_nodes = 32
        feature_dim = 16
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        senders = torch.randint(0, num_nodes, (50,), device=device)
        receivers = torch.randint(0, num_nodes, (50,), device=device)
        
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(50, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([50], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=2,
            nodes_per_level=[16],
            skip_connection="add",
        ).to(device)
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape

    def test_skip_connections_concat(self, device):
        """Test skip connections with concat mode."""
        num_nodes = 32
        feature_dim = 16
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        senders = torch.randint(0, num_nodes, (50,), device=device)
        receivers = torch.randint(0, num_nodes, (50,), device=device)
        
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(50, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([50], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=2,
            nodes_per_level=[16],
            skip_connection="concat",
        ).to(device)
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape

    def test_encoder_decoder_balance(self, device):
        """Test same number of encoder and decoder levels, and that the
        U-Net topology is actually functional (forward pass restores shape).

        Checking module count alone cannot catch wiring bugs (e.g. wrong
        unpool indices).  Running a forward pass confirms the full
        encoder→bottleneck→decoder pipeline is consistent.
        """
        num_nodes = 64   # large enough to survive 3× halving (n_levels=4)
        feature_dim = 8

        senders = torch.randint(0, num_nodes, (100,), device=device)
        receivers = torch.randint(0, num_nodes, (100,), device=device)
        graph = GraphsTuple.from_flat(
            nodes=torch.randn(num_nodes, feature_dim, device=device),
            edges=torch.randn(100, feature_dim, device=device),
            senders=senders,
            receivers=receivers,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([100], device=device),
        )

        for n_levels in [2, 3, 4]:
            processor = GraphUNetProcessor(
                latent_dim=feature_dim,
                n_levels=n_levels,
                skip_connection="add",
            ).to(device)

            assert len(processor.encoders) == n_levels
            assert len(processor.decoders) == n_levels

            # Forward pass: skip connections + unpool must exactly reconstruct
            # the original node count, proving the topology is correct.
            output = processor(graph)
            assert output.nodes.shape == graph.nodes.shape, (
                f"n_levels={n_levels}: output shape {output.nodes.shape} "
                f"!= input shape {graph.nodes.shape}"
            )

    def test_gradient_flow(self, device):
        """Test that gradients flow through all parameters."""
        num_nodes = 32
        feature_dim = 8
        
        nodes = torch.randn(num_nodes, feature_dim, device=device, requires_grad=True)
        senders = torch.randint(0, num_nodes, (50,), device=device)
        receivers = torch.randint(0, num_nodes, (50,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(50, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([50], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=2,
            nodes_per_level=[16],
            skip_connection="add",
        ).to(device)
        
        output = processor(graph)
        loss = output.nodes.sum()
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in processor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.parametrize("n_levels", [2, 3, 4])
    def test_different_levels(self, n_levels, device):
        """Test U-Net with different numbers of levels."""
        num_nodes = 64
        feature_dim = 8
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        senders = torch.randint(0, num_nodes, (100,), device=device)
        receivers = torch.randint(0, num_nodes, (100,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(100, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([100], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=n_levels,
            skip_connection="add",
        ).to(device)
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape


# =============================================================================
# TestMGKNProcessor - MGKN (NeurIPS 2020)
# =============================================================================

class TestMGKNProcessor:
    """Test MGKN processor with V-cycle."""

    def test_v_cycle_structure(self, device):
        """Test downward then upward traversal."""
        num_nodes = 100
        feature_dim = 16
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(150, feature_dim, device=device),
            receivers=torch.randint(0, num_nodes, (150,), device=device),
            senders=torch.randint(0, num_nodes, (150,), device=device),
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([150], device=device),
        )
        
        processor = MGKNProcessor(
            latent_dim=feature_dim,
            n_levels=3,
            nodes_per_level=[50, 25],
            hidden_dim=32,
            n_layers_per_level=1,
        ).to(device)
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape

    def test_skip_connections(self, device):
        """Verify that skip connections are wired into the computation graph.

        Skip connections add encoder features (derived from the fine-level
        input) to each decoder stage in the upward V-cycle pass.  Two
        concrete invariants must hold:

        1. Gradient connectivity — gradients from the output must flow back
           to the original input nodes via the skip-connection path.
        2. Non-identity transformation — the V-cycle must actually change
           the node features (not a passthrough).
        """
        num_nodes = 64
        feature_dim = 8

        nodes = torch.randn(num_nodes, feature_dim, device=device,
                            requires_grad=True)
        num_edges = 100
        senders = torch.randint(0, num_nodes, (num_edges,), device=device)
        receivers = torch.randint(0, num_nodes, (num_edges,), device=device)

        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(num_edges, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([num_edges], device=device),
        )

        processor = MGKNProcessor(
            latent_dim=feature_dim,
            n_levels=3,
            nodes_per_level=[32, 16],
            hidden_dim=16,
            n_layers_per_level=1,
        ).to(device)

        output = processor(graph)

        # 1. Shape preservation through the full V-cycle.
        assert output.nodes.shape == graph.nodes.shape

        # 2. Gradient connectivity: skip connections keep input nodes in the
        #    computation graph.  If they were severed, fine-level encoder
        #    features would not reach the decoder and the gradient from
        #    output → input would vanish or be zero.
        loss = output.nodes.sum()
        loss.backward()
        assert nodes.grad is not None, "No gradient reached input nodes"
        assert nodes.grad.abs().max().item() > 0, (
            "Input node gradients are all zero — skip connections may be severed"
        )

        # 3. Non-identity: the V-cycle must transform features.
        assert not torch.allclose(output.nodes.detach(), nodes.detach()), (
            "Output equals input — the V-cycle applied no transformation"
        )

    @pytest.mark.parametrize("n_levels", [2, 3, 4])
    def test_different_levels(self, n_levels, device):
        """Test MGKN with different numbers of levels."""
        num_nodes = 64
        feature_dim = 8
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        num_edges = 100
        senders = torch.randint(0, num_nodes, (num_edges,), device=device)
        receivers = torch.randint(0, num_nodes, (num_edges,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(num_edges, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([num_edges], device=device),
        )
        
        nodes_per_level = [num_nodes // (2 ** (i + 1)) for i in range(n_levels - 1)]
        
        processor = MGKNProcessor(
            latent_dim=feature_dim,
            n_levels=n_levels,
            nodes_per_level=nodes_per_level,
            hidden_dim=16,
            n_layers_per_level=1,
        ).to(device)
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape

    def test_coarsest_processing(self, device):
        """Test that coarsest level is processed."""
        num_nodes = 64
        feature_dim = 8
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        num_edges = 100
        senders = torch.randint(0, num_nodes, (num_edges,), device=device)
        receivers = torch.randint(0, num_nodes, (num_edges,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(num_edges, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([num_edges], device=device),
        )
        
        processor = MGKNProcessor(
            latent_dim=feature_dim,
            n_levels=3,
            nodes_per_level=[32, 16],
            hidden_dim=16,
            n_layers_per_level=1,
        ).to(device)
        
        # Should have processors for each level
        assert len(processor.level_processors) == 3
        
        output = processor(graph)
        assert output.nodes.shape == graph.nodes.shape


# =============================================================================
# TestMultiResolutionFNOBlock
# =============================================================================

class TestMultiResolutionFNOBlock:
    """Test MultiResolutionFNOBlock with parallel frequency bands."""

    def test_forward_basic(self, device):
        """Test forward pass with 3 bands."""
        width = 32
        modes_list = [[8, 8], [16, 16], [32, 32]]
        
        block = MultiResolutionFNOBlock(
            width=width,
            modes_list=modes_list,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 64, 64, device=device)
        output = block(x)
        
        assert output.shape == x.shape

    def test_band_weights_learnable(self, device):
        """Test that band weights are learnable and sum to 1."""
        width = 16
        modes_list = [[8, 8], [16, 16]]
        
        block = MultiResolutionFNOBlock(
            width=width,
            modes_list=modes_list,
            n_dim=2,
        ).to(device)
        
        # Check weights exist
        assert hasattr(block, 'band_weights')
        
        # Check softmax produces weights summing to 1
        weights = torch.nn.functional.softmax(block.band_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
        
        # Check they are trainable
        x = torch.randn(2, width, 32, 32, device=device, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert block.band_weights.grad is not None

    def test_different_modes_per_band(self, device):
        """Test each band has correct number of modes."""
        width = 16
        modes_list = [[8, 8], [16, 16], [32, 32]]
        
        block = MultiResolutionFNOBlock(
            width=width,
            modes_list=modes_list,
            n_dim=2,
        ).to(device)
        
        # Check each band block has correct modes
        for i, band_block in enumerate(block.band_blocks):
            # Modes are passed to FNOBlock which stores them in spectral_conv
            expected_modes = modes_list[i]
            # The spectral conv should have weights with shape based on modes
            assert band_block.spectral_conv.weights.shape[-3:-1] == tuple(expected_modes)

    def test_weighted_combination(self, device):
        """Test that output changes with different weights."""
        width = 16
        modes_list = [[8, 8], [16, 16]]
        
        block = MultiResolutionFNOBlock(
            width=width,
            modes_list=modes_list,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device)
        
        # First forward
        output1 = block(x)
        
        # Change weights and forward again
        block.band_weights.data = torch.randn_like(block.band_weights)
        output2 = block(x)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2)


# =============================================================================
# TestUFNOBlock - U-FNO (Wen et al. 2022)
# =============================================================================

class TestUFNOBlock:
    """Test U-FNO block combining spectral and local processing."""

    def test_forward_basic(self, device):
        """Test forward pass with spectral + U-Net + bias."""
        width = 32
        modes = [16, 16]
        
        block = UFNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
            unet_depth=2,
        ).to(device)
        
        x = torch.randn(2, width, 64, 64, device=device)
        output = block(x)
        
        assert output.shape == x.shape

    def test_three_branches(self, device):
        """Test that all three components contribute."""
        width = 16
        modes = [8, 8]
        
        block = UFNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
            unet_depth=2,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device)
        
        # Get individual branch outputs
        x1 = block.spectral_conv(x)
        x2 = block.unet(x)
        x3 = block.bias(x)
        
        # All should have same shape
        assert x1.shape == x.shape
        assert x2.shape == x.shape
        assert x3.shape == x.shape
        
        # Combined output
        output = block(x)
        assert output.shape == x.shape

    def test_vs_standard_fno(self, device):
        """Test that U-FNO produces different output than standard FNO."""
        from gnn_pde_v2.components.spectral import FNOBlock
        
        width = 16
        modes = [8, 8]
        
        ufno_block = UFNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
            unet_depth=2,
        ).to(device)
        
        fno_block = FNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device)
        
        out_ufno = ufno_block(x)
        out_fno = fno_block(x)
        
        # Both should have same shape
        assert out_ufno.shape == out_fno.shape
        # But different values (U-Net adds local processing)
        assert not torch.allclose(out_ufno, out_fno)

    @pytest.mark.parametrize("unet_depth", [1, 2, 3])
    def test_unet_depth(self, unet_depth, device):
        """Test U-FNO with different U-Net depths."""
        width = 16
        modes = [8, 8]
        
        block = UFNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
            unet_depth=unet_depth,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device)
        output = block(x)
        
        assert output.shape == x.shape

    def test_gradient_flow(self, device):
        """Test that all branches are trainable."""
        width = 16
        modes = [8, 8]
        
        block = UFNOBlock(
            width=width,
            modes=modes,
            n_dim=2,
            unet_depth=2,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        # Check spectral conv has gradients
        assert block.spectral_conv.weights.grad is not None
        
        # Check U-Net has gradients
        has_unet_grad = any(p.grad is not None for p in block.unet.parameters())
        assert has_unet_grad
        
        # Check bias has gradients
        has_bias_grad = any(p.grad is not None for p in block.bias.parameters())
        assert has_bias_grad


# =============================================================================
# TestMiniUNet
# =============================================================================

class TestMiniUNet:
    """Test MiniUNet for local processing."""

    def test_forward_2d(self, device):
        """Test 2D input."""
        in_channels = 16
        out_channels = 16
        
        unet = MiniUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
            depth=2,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, in_channels, 64, 64, device=device)
        output = unet(x)
        
        assert output.shape == x.shape

    def test_forward_3d(self, device):
        """Test 3D input."""
        in_channels = 8
        out_channels = 8
        
        unet = MiniUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=16,
            depth=2,
            n_dim=3,
        ).to(device)
        
        x = torch.randn(2, in_channels, 16, 16, 16, device=device)
        output = unet(x)
        
        assert output.shape == x.shape

    def test_encoder_decoder(self, device):
        """Test features at each level have correct shapes."""
        in_channels = 8
        hidden_channels = 16
        depth = 2
        
        unet = MiniUNet(
            in_channels=in_channels,
            out_channels=8,
            hidden_channels=hidden_channels,
            depth=depth,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, in_channels, 64, 64, device=device)
        
        # Manually run through encoder
        encoder_features = []
        for conv, pool in zip(unet.encoder_convs, unet.encoder_pools):
            x_enc = conv(x)
            encoder_features.append(x_enc)
            x = pool(x_enc)
        
        # Should have depth features
        assert len(encoder_features) == depth
        
        # Feature channels should increase
        for i, feat in enumerate(encoder_features):
            expected_channels = hidden_channels * (2 ** i)
            assert feat.shape[1] == expected_channels

    def test_skip_connections(self, device):
        """Test that skip connections are used in decoder."""
        in_channels = 8
        hidden_channels = 16
        depth = 2
        
        unet = MiniUNet(
            in_channels=in_channels,
            out_channels=8,
            hidden_channels=hidden_channels,
            depth=depth,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, in_channels, 64, 64, device=device)
        
        # Run forward
        output = unet(x)
        assert output.shape[2:] == (64, 64)


# =============================================================================
# TestHierarchicalFNOBlock
# =============================================================================

class TestHierarchicalFNOBlock:
    """Test HierarchicalFNOBlock with coarse-to-fine processing."""

    def test_forward_basic(self, device):
        """Test 3-level hierarchical processing."""
        width = 32
        modes = [16, 16]
        n_levels = 3
        
        block = HierarchicalFNOBlock(
            width=width,
            modes=modes,
            n_levels=n_levels,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 64, 64, device=device)
        output = block(x)
        
        assert output.shape == x.shape

    def test_downsampling(self, device):
        """Test that coarser levels use smaller spatial sizes."""
        width = 16
        modes = [8, 8]
        n_levels = 3
        
        block = HierarchicalFNOBlock(
            width=width,
            modes=modes,
            n_levels=n_levels,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 64, 64, device=device)
        original_size = x.shape[2:]
        
        # Check that each level has progressively smaller modes
        for level, fno_block in enumerate(block.fno_blocks):
            if level == 0:
                continue
            # Modes should be reduced for coarser levels
            expected_modes = [max(4, m // (2 ** level)) for m in modes]
            actual_modes = list(fno_block.spectral_conv.weights.shape[-3:-1])
            assert actual_modes == expected_modes

    def test_upsampling(self, device):
        """Test that outputs are upsampled back to original size."""
        width = 16
        modes = [8, 8]
        n_levels = 3
        
        block = HierarchicalFNOBlock(
            width=width,
            modes=modes,
            n_levels=n_levels,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 64, 64, device=device)
        output = block(x)
        
        # Output should match input size
        assert output.shape == x.shape

    def test_level_combination(self, device):
        """Test that all levels contribute to output."""
        width = 16
        modes = [8, 8]
        n_levels = 3
        
        block = HierarchicalFNOBlock(
            width=width,
            modes=modes,
            n_levels=n_levels,
            n_dim=2,
        ).to(device)
        
        x = torch.randn(2, width, 32, 32, device=device)
        output = block(x)
        
        # Output should be combination of all levels
        assert output.shape == x.shape


# =============================================================================
# TestHierarchicalGraph - MGKN utilities
# =============================================================================

class TestHierarchicalGraph:
    """Test hierarchical graph utilities."""

    def test_build_hierarchy(self, device):
        """Build 3 levels and verify correct number of graphs."""
        num_nodes = 100
        feature_dim = 16
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(150, feature_dim, device=device),
            receivers=torch.randint(0, num_nodes, (150,), device=device),
            senders=torch.randint(0, num_nodes, (150,), device=device),
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([150], device=device),
        )
        
        hierarchy = build_hierarchical_graphs(
            graph,
            levels=3,
            nodes_per_level=[50, 25],
            feature_dim=feature_dim,
        )
        
        # levels=3 must produce exactly 3 graphs (one per resolution level)
        assert len(hierarchy.graphs) == 3

    def test_nodes_per_level(self, device):
        """Verify each level has correct number of nodes."""
        num_nodes = 100
        feature_dim = 16
        nodes_per_level = [50, 25]
        
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        num_edges = 100
        senders = torch.randint(0, num_nodes, (num_edges,), device=device)
        receivers = torch.randint(0, num_nodes, (num_edges,), device=device)
        
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(num_edges, feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([num_edges], device=device),
        )
        
        hierarchy = build_hierarchical_graphs(
            graph,
            levels=3,
            nodes_per_level=nodes_per_level,
            feature_dim=feature_dim,
        )
        
        # Level 0 = original fine graph
        assert hierarchy.graphs[0].nodes.shape[0] == num_nodes
        # Levels 1 and 2 must match nodes_per_level exactly;
        # checking only level 0 would never catch a wrong coarsening ratio.
        assert hierarchy.graphs[1].nodes.shape[0] == nodes_per_level[0]
        assert hierarchy.graphs[2].nodes.shape[0] == nodes_per_level[1]

    def test_restrict_to_coarse(self, device):
        """Test fine to coarse restriction."""
        n_fine = 20
        n_coarse = 10
        feature_dim = 8
        
        fine_features = torch.randn(n_fine, feature_dim, device=device)
        transition = torch.randn(n_fine, n_coarse, device=device)
        transition = torch.softmax(transition, dim=1)
        
        coarse_features = restrict_to_coarse(fine_features, transition)
        
        assert coarse_features.shape == (n_coarse, feature_dim)

    def test_prolong_to_fine(self, device):
        """Test coarse to fine prolongation."""
        n_fine = 20
        n_coarse = 10
        feature_dim = 8
        
        coarse_features = torch.randn(n_coarse, feature_dim, device=device)
        transition = torch.randn(n_fine, n_coarse, device=device)
        transition = torch.softmax(transition, dim=1)
        
        fine_features = prolong_to_fine(coarse_features, transition)
        
        assert fine_features.shape == (n_fine, feature_dim)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for multiscale components."""

    def test_multiscale_fno_forward(self, device):
        """Test full MultiscaleFNO model forward pass."""
        model = MultiscaleFNO(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[16, 16],
            n_layers=2,
            n_dim=2,
            architecture="ufno",
        ).to(device)
        
        x = torch.randn(2, 1, 64, 64, device=device)
        output = model(x)
        
        assert output.shape == x.shape

    @pytest.mark.parametrize("architecture", ["standard", "multiband", "ufno", "hierarchical"])
    def test_multiscale_fno_architectures(self, architecture, device):
        """Test MultiscaleFNO with different architectures."""
        model = MultiscaleFNO(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[16, 16],
            n_layers=2,
            n_dim=2,
            architecture=architecture,
        ).to(device)
        
        x = torch.randn(2, 1, 64, 64, device=device)
        output = model(x)
        
        assert output.shape == x.shape

    def test_super_resolution(self, device):
        """Test super-resolution capability."""
        model = MultiscaleFNO(
            in_channels=1,
            out_channels=1,
            width=32,
            modes=[16, 16],
            n_layers=2,
            n_dim=2,
            architecture="ufno",
        ).to(device)
        
        # Train at low resolution
        x_low = torch.randn(2, 1, 32, 32, device=device)
        
        # Test at high resolution
        output_high = model.forward_super_resolution(x_low, [64, 64])
        
        assert output_high.shape == (2, 1, 64, 64)

    def test_graph_unet_end_to_end(self, device):
        """Test Graph U-Net on synthetic data."""
        num_nodes = 64
        feature_dim = 16
        
        # Create synthetic graph data
        nodes = torch.randn(num_nodes, feature_dim, device=device)
        # Create a simple grid-like graph
        senders = []
        receivers = []
        for i in range(8):
            for j in range(7):
                node_idx = i * 8 + j
                # Connect to right neighbor
                senders.append(node_idx)
                receivers.append(node_idx + 1)
                # Connect to bottom neighbor
                if i < 7:
                    senders.append(node_idx)
                    receivers.append(node_idx + 8)
        
        senders = torch.tensor(senders, device=device)
        receivers = torch.tensor(receivers, device=device)
        
        # GraphNetBlock expects edges with dimension = latent_dim
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(senders.shape[0], feature_dim, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([num_nodes], device=device),
            n_edge=torch.tensor([senders.shape[0]], device=device),
        )
        
        processor = GraphUNetProcessor(
            latent_dim=feature_dim,
            n_levels=3,
            nodes_per_level=[32, 16, 8],
            skip_connection="add",
        ).to(device)
        
        output = processor(graph)
        
        # Should produce valid predictions
        assert output.nodes.shape == graph.nodes.shape
        assert not torch.isnan(output.nodes).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test that components work on CUDA."""
        device = torch.device('cuda')
        
        # Test GraphPool
        pool = GraphPool(k=10, feature_dim=8).to(device)
        nodes = torch.randn(20, 8, device=device)
        senders = torch.randint(0, 20, (30,), device=device)
        receivers = torch.randint(0, 20, (30,), device=device)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=torch.randn(30, 8, device=device),
            receivers=receivers,
            senders=senders,
            n_node=torch.tensor([20], device=device),
            n_edge=torch.tensor([30], device=device),
        )
        pooled, _ = pool(graph)
        assert pooled.nodes.device.type == 'cuda'
        
        # Test FNO blocks
        block = UFNOBlock(width=16, modes=[8, 8], n_dim=2).to(device)
        x = torch.randn(2, 16, 32, 32, device=device)
        output = block(x)
        assert output.device.type == 'cuda'

    def test_device_placement_cpu(self, device):
        """Test that components work on CPU."""
        # Test GraphPool
        pool = GraphPool(k=10, feature_dim=8)
        nodes = torch.randn(20, 8)
        graph = GraphsTuple.from_flat(
            nodes=nodes,
            edges=None,
            receivers=None,
            senders=None,
            n_node=torch.tensor([20]),
            n_edge=torch.tensor([0]),
        )
        pooled, _ = pool(graph)
        assert pooled.nodes.device.type == 'cpu'


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_graph_100(device):
    """Create a sample graph with 100 nodes for pooling tests."""
    num_nodes = 100
    feature_dim = 16
    
    nodes = torch.randn(num_nodes, feature_dim, device=device)
    senders = torch.randint(0, num_nodes, (200,), device=device)
    receivers = torch.randint(0, num_nodes, (200,), device=device)
    
    return GraphsTuple.from_flat(
        nodes=nodes,
        edges=torch.randn(200, 4, device=device),
        receivers=receivers,
        senders=senders,
        n_node=torch.tensor([num_nodes], device=device),
        n_edge=torch.tensor([200], device=device),
    )


@pytest.fixture
def sample_grid_2d(device):
    """Create a sample 2D grid for FNO tests."""
    return torch.randn(2, 1, 32, 32, device=device)
