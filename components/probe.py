"""
Probe-based decoders for arbitrary query points.

This module provides flexible probe decoders that can predict values at
arbitrary spatial locations by message passing from a source graph.

Two main implementations are provided:

1. **ProbeDecoder**: General-purpose decoder with configurable processor
2. **WindFarmGNO**: Paper-faithful Wind-Farm Graph Neural Operator

Reference:
---------
Schøler, J. P., et al. (2025). "Graph Neural Operator for windfarm wake flow."
Wind Energy Science Discussions. https://doi.org/10.5194/wes-2025-261
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn

from ..core.graph import GraphsTuple, GraphTopology, batch_graphs, unbatch_graphs
from ..core.functional import scatter_mean, scatter_sum
from ..core.mlp import MLP


# =============================================================================
# Probe Graph Construction Utility
# =============================================================================

class ProbeGraphBuilder:
    """
    Static utility for constructing probe graphs from source nodes and queries.
    
    Encapsulates the k-NN graph construction logic used by probe decoders.
    Handles both single graphs and batched processing.
    
    The constructed graph has layout:
        nodes = [source_nodes | probe_nodes]
        edges = source → probe (bipartite)
    
    This layout allows easy extraction of probe nodes after processing.
    
    Example:
        >>> builder = ProbeGraphBuilder()
        >>> probe_graph = builder.build(
        ...     source_positions=turbine_pos,  # [N_src, 2]
        ...     source_features=turbine_feat,  # [N_src, latent_dim]
        ...     query_positions=probe_pos,     # [N_qry, 2]
        ...     k_nearest=5,
        ... )
    """
    
    @staticmethod
    def build(
        source_positions: torch.Tensor,
        source_features: torch.Tensor,
        query_positions: torch.Tensor,
        k_nearest: int = 5,
        edge_feature_type: str = 'distance',
    ) -> GraphsTuple:
        """
        Construct a probe GraphsTuple for a **single** source graph.
        
        Builds a bipartite graph where edges go from source nodes to probe
        (query) nodes. Source nodes are placed first; probe nodes are
        appended after them.
        
        Args:
            source_positions: [N_source, n_dim] source node coordinates
            source_features: [N_source, feature_dim] source node features
            query_positions: [N_queries, n_dim] query coordinates
            k_nearest: Number of nearest neighbors to connect
            edge_feature_type: 'distance' | 'pos_diff' | 'both'
            
        Returns:
            GraphsTuple with layout [source_nodes | probe_nodes]
        """
        n_queries = query_positions.shape[0]
        n_source = source_positions.shape[0]
        
        # Find k nearest source nodes for each query
        distances = torch.cdist(query_positions, source_positions)  # [N_queries, N_source]
        _, nearest_indices = torch.topk(distances, k=k_nearest, largest=False, dim=-1)
        # nearest_indices: [N_queries, k_nearest]
        
        # Create edges: source (sender) -> probe (receiver)
        # Receivers are probe nodes (offset by n_source in the combined graph)
        local_receivers = torch.arange(
            n_queries, device=query_positions.device
        ).repeat_interleave(k_nearest)
        receivers = local_receivers + n_source
        senders = nearest_indices.reshape(-1)
        
        # Get distances for edges
        nearest_distances = torch.gather(distances, 1, nearest_indices)  # [N_queries, k_nearest]
        
        # Build edge features based on type
        if edge_feature_type == 'distance':
            edge_features = nearest_distances.reshape(-1, 1)  # [N_edges, 1]
        elif edge_feature_type == 'pos_diff':
            query_expanded = query_positions.repeat_interleave(k_nearest, dim=0)
            pos_diff = query_expanded - source_positions[senders]
            edge_features = pos_diff
        elif edge_feature_type == 'both':
            query_expanded = query_positions.repeat_interleave(k_nearest, dim=0)
            pos_diff = query_expanded - source_positions[senders]
            edge_dist = nearest_distances.reshape(-1, 1)
            edge_features = torch.cat([pos_diff, edge_dist], dim=-1)
        else:
            raise ValueError(f"Unknown edge_feature_type: {edge_feature_type}")
        
        # Initialize probe nodes by mean-aggregating k-nearest source features
        probe_nodes = scatter_mean(
            source_features[senders],
            local_receivers,
            dim=0,
            dim_size=n_queries
        )
        
        # Combine: [source_nodes | probe_nodes]
        all_nodes = torch.cat([source_features, probe_nodes], dim=0)
        all_positions = torch.cat([source_positions, query_positions], dim=0)
        
        return GraphsTuple(
            nodes=all_nodes,
            topology=GraphTopology(
                n_node=torch.tensor([n_source + n_queries], device=query_positions.device),
                senders=senders,
                receivers=receivers,
                n_edge=torch.tensor([len(receivers)], device=query_positions.device),
                positions=all_positions,
            ),
            edges=edge_features,
            globals=None,
        )
    
    @staticmethod
    def extract_probe_nodes(
        probe_graph: GraphsTuple,
        n_query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract probe (query) node features from a batched probe GraphsTuple.
        
        Within each sub-graph, probe nodes occupy the last n_query[b] rows.
        
        Args:
            probe_graph: Batched probe GraphsTuple after processing
            n_query: [B] number of probe nodes per sub-graph
            
        Returns:
            [total_queries, feature_dim] probe node features
        """
        all_nodes = probe_graph.nodes
        
        # Compute segment bounds
        ends = probe_graph.n_node.cumsum(0)
        starts = ends - probe_graph.n_node
        probe_starts = ends - n_query
        
        # Build flat index tensor
        base = probe_starts.repeat_interleave(n_query)
        total_q = int(n_query.sum().item())
        global_idx = torch.arange(total_q, device=all_nodes.device)
        seg_starts = (n_query.cumsum(0) - n_query)
        local_offsets = global_idx - seg_starts.repeat_interleave(n_query)
        
        return all_nodes[base + local_offsets]


# =============================================================================
# General-Purpose Probe Decoder
# =============================================================================

class ProbeDecoder(nn.Module):
    """
    Decoder for arbitrary query points using probe mechanism.
    
    Creates a probe graph connecting query points to nearest source nodes,
    then applies message passing to propagate information and decode predictions.
    
    The processor is **injected** via the constructor, allowing different
    message passing strategies (GraphNetBlock, GENBlock, etc.).
    
    Architecture:
    -------------
    1. Build probe graph: k-NN from queries to source nodes
    2. Encode edges: optional edge encoder (e.g., RBF)
    3. Process: message passing via injected processor
    4. Extract: probe node features
    5. Decode: MLP to output dimension (optional concat with query features)
    
    Args:
        latent_dim: Dimension for node features
        processor: Processor module(s) for message passing. Can be:
                   - Single module (e.g., GraphNetBlock)
                   - ModuleList for multiple steps
                   - Any nn.Module that accepts and returns GraphsTuple
        edge_encoder: Optional edge feature encoder (e.g., LearnableRBFEncoder)
        out_dim: Output dimension for predictions
        hidden_dim: Hidden dimension for output MLP
        k_nearest: Number of neighbors for probe graph construction
        decode_with_query_features: Whether to concat query input features
                                    before final decode
        
    Example:
        >>> # With GEN blocks (paper-faithful)
        >>> processor = nn.ModuleList([
        ...     GENBlock(latent_dim=128) for _ in range(6)
        ... ])
        >>> decoder = ProbeDecoder(
        ...     latent_dim=128,
        ...     processor=processor,
        ...     edge_encoder=LearnableRBFEncoder(num_kernels=20),
        ...     out_dim=1,
        ...     k_nearest=5,
        ... )
        >>>
        >>> # With single GraphNetBlock
        >>> decoder = ProbeDecoder(
        ...     latent_dim=128,
        ...     processor=GraphNetBlock(latent_dim=128),
        ...     out_dim=3,
        ...     k_nearest=3,
        ... )
    """

    #: Discriminator used by :class:`~gnn_pde_v2.models.EncodeProcessDecode`
    #: to decide whether ``query_positions`` should be forwarded. Probe-based
    #: decoders require explicit query positions.
    is_query_decoder: bool = True

    def __init__(
        self,
        latent_dim: int,
        processor: nn.Module,
        edge_encoder: Optional[nn.Module] = None,
        out_dim: int = 1,
        hidden_dim: int = 128,
        k_nearest: int = 5,
        decode_with_query_features: bool = False,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.processor = processor
        self.edge_encoder = edge_encoder
        self.k_nearest = k_nearest
        self.decode_with_query_features = decode_with_query_features

        # Determine the effective edge feature dimension after optional encoding.
        # ProbeGraphBuilder always produces 1-D distance edges; if an encoder is
        # provided that has a num_kernels attribute we use that.
        if edge_dim is None:
            if edge_encoder is not None and hasattr(edge_encoder, 'num_kernels'):
                edge_dim = edge_encoder.num_kernels
            else:
                edge_dim = 1  # raw distance scalar from ProbeGraphBuilder

        # Project edges to latent_dim when they differ (required by most
        # message-passing blocks which expect edge_dim == latent_dim).
        if edge_dim != latent_dim:
            self.edge_projection: Optional[nn.Linear] = nn.Linear(edge_dim, latent_dim)
        else:
            self.edge_projection = None

        # Output MLP
        decoder_in_dim = latent_dim
        if decode_with_query_features:
            # Will be set on first forward pass based on query features
            self._query_feature_dim: Optional[int] = None

        self.output_mlp = MLP(
            in_dim=decoder_in_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation='gelu',
        )
    
    def forward(
        self,
        graph: GraphsTuple,
        query_positions: torch.Tensor,
        query_features: Optional[torch.Tensor] = None,
        n_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode at query positions.
        
        Args:
            graph: Source GraphsTuple with nodes and positions
            query_positions: [total_queries, n_dim] query coordinates
            query_features: Optional [total_queries, feat_dim] query input features
            n_query: [B] number of queries per graph (for batched graphs)
            
        Returns:
            [total_queries, out_dim] predictions at query points
        """
        if graph.nodes is None or graph.positions is None:
            raise ValueError("Graph must have nodes and positions")
        
        B = graph.num_graphs
        
        # Normalize n_query
        if n_query is None:
            n_query = torch.tensor(
                [query_positions.shape[0]], dtype=torch.long, device=query_positions.device
            )
        
        # Build probe graphs
        n_node = graph.n_node
        src_starts, src_ends = self._segment_bounds(n_node)
        qry_starts, qry_ends = self._segment_bounds(n_query)
        
        probe_graphs = [
            ProbeGraphBuilder.build(
                graph.positions[ns:ne],
                graph.nodes[ns:ne],
                query_positions[qs:qe],
                self.k_nearest,
            )
            for ns, ne, qs, qe in zip(src_starts.tolist(), src_ends.tolist(),
                                      qry_starts.tolist(), qry_ends.tolist())
        ]
        
        # Batch probe graphs
        batched_probe = batch_graphs(probe_graphs)
        
        # Encode edges (optional) then project to latent_dim
        edges = batched_probe.edges
        if self.edge_encoder is not None:
            edges = self.edge_encoder(edges.squeeze(-1))
            if edges.dim() > 2:
                edges = edges.view(edges.shape[0], -1)
        if self.edge_projection is not None:
            edges = self.edge_projection(edges)
        batched_probe = batched_probe.replace(edges=edges)

        # Process through processor
        if isinstance(self.processor, nn.ModuleList):
            for block in self.processor:
                batched_probe = block(batched_probe)
        else:
            batched_probe = self.processor(batched_probe)
        
        # Extract probe nodes
        probe_features = ProbeGraphBuilder.extract_probe_nodes(batched_probe, n_query)
        
        # Optionally concat with query input features
        if self.decode_with_query_features and query_features is not None:
            probe_features = torch.cat([probe_features, query_features], dim=-1)
            # Update MLP input dim if needed
            if self._query_feature_dim is None:
                self._query_feature_dim = query_features.shape[-1]
                # Note: MLP needs to be recreated or handle dynamic dims
                # For simplicity, assume fixed dimensions in practice
        
        # Decode
        return self.output_mlp(probe_features)
    
    @staticmethod
    def _segment_bounds(counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (starts, ends) tensors for segment bounds."""
        ends = counts.cumsum(0)
        starts = ends - counts
        return starts, ends


# =============================================================================
# Paper-Faithful Wind-Farm GNO
# =============================================================================

class WindFarmGNO(nn.Module):
    """
    Paper-faithful Wind-Farm Graph Neural Operator.
    
    Two-stage architecture from Schøler et al. (2025):
    
    Stage 1 - Turbine-to-Turbine (T2T):
        - Encodes turbine features and spatial relationships
        - Applies GEN blocks with softmax aggregation
        - Produces latent representations for each turbine
        
    Stage 2 - Probe-to-Turbine (P2T):
        - Connects probe points to nearest turbines (k-NN)
        - Message passes turbine info to probe locations
        - Decodes flow field at probe locations
    
    This implementation uses the exact architecture from the paper:
    - Learnable RBF encoding for distances
    - GEN blocks with epsilon stability and softmax aggregation
    - Shared encoder weights between T2T and P2T
    - Separate decoders for turbines and probes
    
    Args:
        num_turbine_features: Input dimension for turbine node features
        num_edge_features: Input dimension for edge features (typically 1 for distance)
        num_probe_features: Input dimension for probe node features (e.g., free stream U, TI)
        turbine_output_dim: Output dimension for turbine predictions (default: 1 for effective wind speed)
        probe_output_dim: Output dimension for probe predictions (default: 1 for wind speed)
        latent_dim: Latent dimension (paper default: 128)
        hidden_dim: Hidden dimension for MLPs (paper default: 128)
        num_mlp_layers: Layers in MLPs (paper default: 6)
        wt_message_passing_steps: GEN blocks in T2T stage (paper default: 6)
        probe_message_passing_steps: GEN blocks in P2T stage (paper default: 6)
        k_neighbors: k for k-NN probe graph (paper default: 5)
        use_rbf: Whether to use learnable RBF encoding (paper: True)
        rbf_kwargs: Dict with RBF parameters
        epsilon: Numerical stability constant (paper default: 1e-6)
        
    Example:
        >>> model = WindFarmGNO(
        ...     num_turbine_features=10,  # e.g., x, y, D, CT, power, U, TI, ...
        ...     num_edge_features=4,      # e.g., dx, dy, distance, angle
        ...     num_probe_features=6,     # e.g., x, y, U, TI, ...
        ...     latent_dim=128,
        ...     hidden_dim=128,
        ...     num_mlp_layers=6,
        ...     wt_message_passing_steps=6,
        ...     probe_message_passing_steps=6,
        ...     k_neighbors=5,
        ...     use_rbf=True,
        ... )
        >>> output = model(turbine_graph, probe_positions, probe_features)
        >>> output['turbine'].shape  # [N_turbines, 1]
        >>> output['probe'].shape    # [N_probes, 1]
    """
    
    def __init__(
        self,
        num_turbine_features: int,
        num_edge_features: int,
        num_probe_features: int,
        turbine_output_dim: int = 1,
        probe_output_dim: int = 1,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_mlp_layers: int = 6,
        wt_message_passing_steps: int = 6,
        probe_message_passing_steps: int = 6,
        k_neighbors: int = 5,
        use_rbf: bool = True,
        rbf_kwargs: Optional[Dict[str, Any]] = None,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.k_neighbors = k_neighbors
        self.use_rbf = use_rbf
        
        # Default RBF parameters from paper
        if rbf_kwargs is None and use_rbf:
            rbf_kwargs = {
                'num_kernels': 20,
                'd_min': -1.0,
                'd_max': 1.0,
                'learnable': True,
            }
        self.rbf_kwargs = rbf_kwargs
        
        # Stage 1: Turbine-to-Turbine (T2T) components
        
        # Edge encoder (RBF or MLP)
        if use_rbf:
            from .rbf import LearnableRBFEncoder
            self.edge_encoder = LearnableRBFEncoder(**rbf_kwargs)
            # Each of the num_edge_features values is encoded independently by
            # the RBF kernel bank, producing num_kernels values per input scalar.
            # The resulting per-edge feature vector has dimension
            # num_edge_features * num_kernels.
            edge_input_dim = num_edge_features * rbf_kwargs['num_kernels']
        else:
            self.edge_encoder = None
            edge_input_dim = num_edge_features
        
        # Node and edge encoders (shared between T2T and P2T)
        self.node_encoder = MLP(
            in_dim=num_turbine_features,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim] * num_mlp_layers,
            activation='relu',
        )
        
        self.edge_embedder = MLP(
            in_dim=edge_input_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim] * num_mlp_layers,
            activation='relu',
        )
        
        # Stage 1 processor: GEN blocks for turbine-to-turbine
        from .processors import GENBlock
        self.turbine_processor = nn.ModuleList([
            GENBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_mlp_layers=num_mlp_layers,
                activation='relu',
                epsilon=epsilon,
            )
            for _ in range(wt_message_passing_steps)
        ])
        
        # Stage 1 decoder: turbine predictions
        self.turbine_decoder = MLP(
            in_dim=latent_dim,
            out_dim=turbine_output_dim,
            hidden_dims=[hidden_dim] * num_mlp_layers,
            activation='relu',
        )
        
        # Stage 2: Probe-to-Turbine (P2T) components
        
        # Probe decoder with GEN processor
        probe_processor = nn.ModuleList([
            GENBlock(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_mlp_layers=num_mlp_layers,
                activation='relu',
                epsilon=epsilon,
            )
            for _ in range(probe_message_passing_steps)
        ])
        
        self.probe_decoder = ProbeDecoder(
            latent_dim=latent_dim,
            processor=probe_processor,
            edge_encoder=self.edge_encoder if use_rbf else None,
            out_dim=probe_output_dim,
            hidden_dim=hidden_dim,
            k_nearest=k_neighbors,
            decode_with_query_features=True,
        )
        
        # Store probe feature dim for decoder adjustment
        self.num_probe_features = num_probe_features
        
        # Adjust probe decoder output MLP to handle concatenated features
        # Overwrite the output_mlp created in ProbeDecoder.__init__
        self.probe_decoder.output_mlp = MLP(
            in_dim=latent_dim + num_probe_features,
            out_dim=probe_output_dim,
            hidden_dims=[hidden_dim] * num_mlp_layers,
            activation='relu',
        )
    
    def forward(
        self,
        turbine_graph: GraphsTuple,
        probe_positions: torch.Tensor,
        probe_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through two-stage WindFarm GNO.
        
        Args:
            turbine_graph: GraphsTuple with turbine nodes, edges, positions
                - nodes: [N_turbines, num_turbine_features]
                - edges: [N_edges, num_edge_features]
                - positions: [N_turbines, 2]
            probe_positions: [N_probes, 2] probe coordinates
            probe_features: [N_probes, num_probe_features] probe input features
            
        Returns:
            Dict with:
                - 'turbine': [N_turbines, turbine_output_dim] turbine predictions
                - 'probe': [N_probes, probe_output_dim] flow at probe locations
        """
        # ==================== Stage 1: Turbine-to-Turbine ====================
        
        # Encode edges (RBF if enabled)
        if self.edge_encoder is not None:
            encoded_edges = self.edge_encoder(turbine_graph.edges.squeeze(-1))
            if encoded_edges.dim() > 2:
                encoded_edges = encoded_edges.view(encoded_edges.shape[0], -1)
        else:
            encoded_edges = turbine_graph.edges
        
        # Encode to latent space
        nodes = self.node_encoder(turbine_graph.nodes)
        edges = self.edge_embedder(encoded_edges)
        
        # Build latent graph
        from dataclasses import replace
        graph = replace(turbine_graph, nodes=nodes, edges=edges)
        
        # Process through GEN blocks
        for block in self.turbine_processor:
            graph = block(graph)
        
        turbine_latent = graph.nodes
        
        # Decode turbine predictions
        turbine_pred = self.turbine_decoder(turbine_latent)
        
        # ==================== Stage 2: Probe-to-Turbine ====================
        
        # Use ProbeDecoder for P2T stage
        probe_pred = self.probe_decoder(
            graph=graph,
            query_positions=probe_positions,
            query_features=probe_features,
        )
        
        return {
            'turbine': turbine_pred,
            'probe': probe_pred,
        }


__all__ = [
    "ProbeGraphBuilder",
    "ProbeDecoder",
    "WindFarmGNO",
]
