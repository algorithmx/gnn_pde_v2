"""
Probe-based decoder for arbitrary query points.

Inspired on but general than Wind-Farm-GNO.
"""

from typing import Optional
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple, batch_graphs
from ..core.functional import scatter_sum, scatter_mean
from ..core.mlp import MLP


class ProbeDecoder(nn.Module):
    """
    Decoder for arbitrary query points using probe mechanism.

    Two-stage approach:
    1. Process source graph (already done by processor)
    2. Message pass from source to probe locations
    3. Decode at probe locations

    Reference: Wind-Farm-GNO probe-based decoder.
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int = 32,
        out_dim: int = 3,
        hidden_dim: int = 128,
        n_probe_layers: int = 2,
        k_nearest: int = 3,
        distance_encoding: str = 'rbf',
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.k_nearest = k_nearest
        self.distance_encoding = distance_encoding

        # Edge encoder: encode distance to edge features
        if distance_encoding == 'rbf':
            # RBF encoding dimension
            edge_input_dim = k_nearest  # One feature per neighbor
        else:
            edge_input_dim = k_nearest * 2  # Position differences

        self.edge_encoder = MLP(
            in_dim=edge_input_dim,
            out_dim=edge_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )

        # Probe processor layers
        self.probe_layers = nn.ModuleList([
            ProbeMessagePassingLayer(latent_dim, edge_dim, hidden_dim, activation)
            for _ in range(n_probe_layers)
        ])

        # Output MLP
        self.output_mlp = MLP(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=activation,
        )

    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None,
        n_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode at query positions, supporting single and batched graphs.

        ``graph`` may be a single graph or a batch of graphs packed into a
        ``GraphsTuple`` (i.e. the output of ``batch_graphs``).  ``query_positions``
        must be a flat packed tensor whose assignment to graphs is described by
        ``n_query``.

        Args:
            graph: Processed source ``GraphsTuple`` — single or batched.
                Must have ``nodes`` and ``positions`` set.
            query_positions: ``[total_queries, n_dim]`` — flat packed query
                coordinates across all graphs in the batch.
            n_query: ``[B]`` number of query points per graph.  If ``None``,
                all query points are assumed to belong to the single graph
                (equivalent to ``n_query = [total_queries]``).

        Returns:
            ``[total_queries, out_dim]`` — predictions at every query point,
            packed in the same order as ``query_positions``.

        Raises:
            ValueError: If ``query_positions`` is ``None``.
            ValueError: If ``graph.nodes`` or ``graph.positions`` is ``None``.
        """
        if query_positions is None:
            raise ValueError(
                "ProbeDecoder requires query_positions to be provided. "
                "Pass the [total_queries, n_dim] tensor of query coordinates."
            )
        if graph.nodes is None:
            raise ValueError("Graph must have nodes for ProbeDecoder")
        if graph.positions is None:
            raise ValueError("Graph must have positions for ProbeDecoder")

        B = graph.num_graphs

        # Normalise n_query: default to treating all queries as one group
        if n_query is None:
            n_query = torch.tensor(
                [query_positions.shape[0]], dtype=torch.long, device=query_positions.device
            )

        # n_node is guaranteed to exist when B >= 1
        n_node = graph.n_node  # [B]

        # ── Build one probe GraphsTuple per source graph, then pack ──────────
        # The only per-graph work is k-NN lookup (inherently local geometry).
        # All MLP and scatter ops execute on the packed result below.
        # Precompute segment bounds as Python int lists (needed for tensor slicing).
        src_starts, src_ends = (t.tolist() for t in self._segment_bounds(n_node))
        qry_starts, qry_ends = (t.tolist() for t in self._segment_bounds(n_query))
        probe_graphs = [
            self._construct_probe_graph(
                graph.positions[ns:ne],
                graph.nodes[ns:ne],
                query_positions[qs:qe],
            )
            for ns, ne, qs, qe in zip(src_starts, src_ends, qry_starts, qry_ends)
        ]

        # Pack all per-graph probe graphs into one batched GraphsTuple.
        # From here on every operation is vectorised across the full batch.
        batched_probe = batch_graphs(probe_graphs)

        # ── Vectorised message passing over the packed batch ─────────────────
        for layer in self.probe_layers:
            batched_probe = layer(batched_probe)

        # ── Extract probe node features and decode ───────────────────────────
        # Each sub-graph in batched_probe has layout [source_nodes | probe_nodes].
        # Probe nodes are the last n_query[b] nodes of sub-graph b.
        probe_node_features = self._extract_probe_nodes(batched_probe, n_query)
        return self.output_mlp(probe_node_features)

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _segment_bounds(counts: torch.Tensor):
        """
        Return ``(starts, ends)`` tensors for each segment in a packed tensor.

        Both tensors are on the same device as ``counts``.

        Args:
            counts: ``[B]`` integer tensor of segment lengths.

        Returns:
            Tuple ``(starts, ends)`` each of shape ``[B]``, where
            ``starts[b] = sum(counts[:b])`` and ``ends[b] = sum(counts[:b+1])``.
        """
        ends   = counts.cumsum(0)
        starts = ends - counts
        return starts, ends

    def _encode_edge_features(
        self,
        nearest_distances: torch.Tensor,
        senders: torch.Tensor,
        source_positions: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode per-edge features and project through ``edge_encoder``.

        Args:
            nearest_distances: ``[N_queries, k]`` — distance from each query
                to each of its k nearest source nodes.
            senders: ``[N_queries * k]`` — flat source-node indices for every
                edge (row-major order matching ``nearest_distances.reshape(-1)``).
            source_positions: ``[N_source, n_dim]``
            query_positions:  ``[N_queries, n_dim]``

        Returns:
            ``[N_queries * k, edge_dim]`` encoded edge features.
        """
        if self.distance_encoding == 'rbf':
            # Each of the k edges for query q carries all k distances from q.
            # nearest_distances[q] repeated k times → [N_queries*k, k].
            edge_raw = nearest_distances.repeat_interleave(self.k_nearest, dim=0)
        else:
            # Position-difference encoding: Δpos (n_dim) ‖ scalar distance (1)
            query_expanded = query_positions.repeat_interleave(self.k_nearest, dim=0)
            pos_diff = query_expanded - source_positions[senders]   # [N_q*k, n_dim]
            edge_dist = nearest_distances.reshape(-1, 1)            # [N_q*k, 1]
            edge_raw = torch.cat([pos_diff, edge_dist], dim=-1)
        return self.edge_encoder(edge_raw)

    def _extract_probe_nodes(
        self,
        probe_graph: GraphsTuple,
        n_query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract probe (query) node features from a batched probe GraphsTuple.

        Within each sub-graph produced by ``_construct_probe_graph``, source
        nodes come first and probe nodes are appended at the end.  Given the
        per-graph node counts (``probe_graph.n_node``) and query counts
        (``n_query``), this slices out only the probe node rows.

        Args:
            probe_graph: Batched probe ``GraphsTuple`` after message passing.
            n_query: ``[B]`` number of probe nodes per sub-graph.

        Returns:
            ``[total_queries, latent_dim]`` — probe node features, packed in
            the same order as the original ``query_positions``.
        """
        all_nodes = probe_graph.nodes  # [total_nodes, D]

        # ── Fully-vectorised probe-node extraction (no Python loop) ───────────
        # Within each sub-graph, probe nodes occupy the last n_query[b] rows.
        # Strategy: build a flat index tensor that selects exactly those rows.
        #
        # 1. End of each sub-graph's node block in the flat tensor.
        #    probe_starts[b] = cumsum(n_node)[b] - n_query[b]
        _, n_node_ends = self._segment_bounds(probe_graph.n_node)  # [B]
        probe_starts = n_node_ends - n_query                       # [B]

        # 2. Expand probe_starts so every query position in graph b gets
        #    the same base offset: [total_queries]
        base = probe_starts.repeat_interleave(n_query)

        # 3. Per-segment local offsets (0, 1, …, n_query[b]-1 for each b)
        #    without a Python loop — the "cumshift" trick:
        #    global arange minus the start-of-segment offset for each element.
        total_q = int(n_query.sum().item())
        global_idx = torch.arange(total_q, device=all_nodes.device)
        seg_starts, _ = self._segment_bounds(n_query)
        local_offsets = global_idx - seg_starts.repeat_interleave(n_query)  # [total_q]

        return all_nodes[base + local_offsets]
    
    def _construct_probe_graph(
        self,
        source_positions: torch.Tensor,
        source_features: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> GraphsTuple:
        """
        Construct a probe GraphsTuple for a **single** source graph.

        Builds a bipartite graph where edges go from source nodes to probe
        (query) nodes.  Source nodes are placed first; probe nodes are
        appended after them.  This layout lets ``_extract_probe_nodes``
        recover them by slicing the last ``n_queries`` rows of ``nodes``.

        Called once per graph in the batch from ``forward``; the resulting
        single-graph ``GraphsTuple``s are then packed via ``batch_graphs``.

        Args:
            source_positions: ``[N_source, n_dim]``
            source_features:  ``[N_source, latent_dim]``
            query_positions:  ``[N_queries, n_dim]``

        Returns:
            Single-graph ``GraphsTuple`` with layout
            ``nodes = [source_nodes | probe_nodes]``.
        """
        n_queries = query_positions.shape[0]
        n_source = source_positions.shape[0]
        
        # Find k nearest source nodes for each query point
        distances = torch.cdist(query_positions, source_positions)  # [N_queries, N_source]
        _, nearest_indices = torch.topk(distances, k=self.k_nearest, largest=False, dim=-1)
        # nearest_indices: [N_queries, k_nearest]
        
        # Create edges: source nodes -> probe nodes
        # Receivers: probe nodes appended after source nodes
        # Senders: source nodes (nearest_indices flattened)
        local_receivers = torch.arange(n_queries, device=query_positions.device).repeat_interleave(self.k_nearest)
        receivers = local_receivers + n_source
        senders = nearest_indices.reshape(-1)
        
        # Edge features: distances
        nearest_distances = torch.gather(
            distances, 1, nearest_indices
        )  # [N_queries, k_nearest]

        edge_features = self._encode_edge_features(
            nearest_distances, senders, source_positions, query_positions
        )
        
        # Create probe graph
        # Nodes: initialized from mean of k-nearest source features
        probe_nodes = scatter_mean(source_features[senders], local_receivers, dim=0, dim_size=n_queries)
        
        all_nodes = torch.cat([source_features, probe_nodes], dim=0)
        all_positions = torch.cat([source_positions, query_positions], dim=0)

        return GraphsTuple(
            nodes=all_nodes,
            edges=edge_features,
            receivers=receivers,
            senders=senders,
            globals=None,
            n_node=torch.tensor([n_source + n_queries], device=query_positions.device),
            n_edge=torch.tensor([len(receivers)], device=query_positions.device),
            positions=all_positions,
        )


class ProbeMessagePassingLayer(nn.Module):
    """
    Single message passing layer for probe graph.
    
    Performs one step of message passing from source nodes to probe nodes:
    1. Update edges based on sender/receiver nodes and current edge features
    2. Aggregate messages to receiver (probe) nodes
    3. Update node features
    
    Args:
        latent_dim: Dimension for node features
        edge_dim: Dimension for edge features
        hidden_dim: Hidden dimension for MLPs
        activation: Activation function ('relu', 'gelu', 'silu', 'tanh')
    """

    def __init__(
        self,
        latent_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.edge_dim = edge_dim

        # Edge update: [sender_node, receiver_node, edge] -> new_edge
        self.edge_mlp = MLP(
            in_dim=2 * latent_dim + edge_dim,
            out_dim=edge_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )

        # Node update: [node, aggregated_edges] -> new_node
        self.node_mlp = MLP(
            in_dim=latent_dim + edge_dim,
            out_dim=latent_dim,
            hidden_dims=[hidden_dim],
            activation=activation,
        )
    
    def forward(self, graph: GraphsTuple) -> GraphsTuple:
        """
        One message passing step.
        
        Args:
            graph: Input GraphsTuple with nodes, edges, senders, receivers
            
        Returns:
            Updated GraphsTuple with new node and edge features (with residual)
        """
        nodes = graph.nodes
        edges = graph.edges
        receivers = graph.receivers
        senders = graph.senders
        
        # Edge update
        sender_features = nodes[senders]
        receiver_features = nodes[receivers]
        edge_inputs = torch.cat([sender_features, receiver_features, edges], dim=-1)
        new_edges = self.edge_mlp(edge_inputs)
        
        # Node update (aggregate to receivers)
        aggregated = scatter_sum(new_edges, receivers, dim=0, dim_size=nodes.shape[0])
        
        node_inputs = torch.cat([nodes, aggregated], dim=-1)
        new_nodes = self.node_mlp(node_inputs)
        
        # Residual connection
        return graph.replace(nodes=nodes + new_nodes, edges=edges + new_edges)
