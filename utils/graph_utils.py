"""
Graph construction utilities.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from ..core.graph import GraphsTuple


def knn_graph(
    positions: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct k-nearest neighbor graph.
    
    Args:
        positions: [N, n_dim] - Node positions
        k: Number of neighbors
        batch: Optional [N] - Batch indices for each node
        
    Returns:
        (edge_index, edge_attr) where edge_index is [2, E]
    """
    from torch_cluster import knn_graph as torch_knn_graph
    
    if batch is None:
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    
    edge_index = torch_knn_graph(positions, k, batch, loop=False)
    
    # Compute edge features (relative positions)
    senders = edge_index[0]
    receivers = edge_index[1]
    
    sender_pos = positions[senders]
    receiver_pos = positions[receivers]
    edge_attr = receiver_pos - sender_pos
    
    return edge_index, edge_attr


def radius_graph(
    positions: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct radius graph (edges within distance r).
    
    Args:
        positions: [N, n_dim] - Node positions
        r: Radius threshold
        batch: Optional [N] - Batch indices
        max_num_neighbors: Maximum neighbors per node
        
    Returns:
        (edge_index, edge_attr)
    """
    from torch_cluster import radius_graph as torch_radius_graph
    
    if batch is None:
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    
    edge_index = torch_radius_graph(
        positions, r, batch, loop=False, max_num_neighbors=max_num_neighbors
    )
    
    senders = edge_index[0]
    receivers = edge_index[1]
    
    sender_pos = positions[senders]
    receiver_pos = positions[receivers]
    edge_attr = receiver_pos - sender_pos
    
    return edge_index, edge_attr


def compute_edge_features(
    positions: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    features: Optional[str] = 'relative',
) -> torch.Tensor:
    """
    Compute edge features from node positions.
    
    Args:
        positions: [N, n_dim] - Node positions
        senders: [E] - Sender node indices
        receivers: [E] - Receiver node indices
        features: Type of features ('relative', 'distance', 'both')
        
    Returns:
        [E, feat_dim] - Edge features
    """
    sender_pos = positions[senders]
    receiver_pos = positions[receivers]
    
    relative = receiver_pos - sender_pos
    distance = torch.norm(relative, dim=-1, keepdim=True)
    
    if features == 'relative':
        return relative
    elif features == 'distance':
        return distance
    elif features == 'both':
        return torch.cat([relative, distance], dim=-1)
    else:
        raise ValueError(f"Unknown features type: {features}")


def mesh_to_graph(
    vertices: torch.Tensor,
    faces: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
) -> GraphsTuple:
    """
    Convert mesh to GraphsTuple.
    
    Args:
        vertices: [N, n_dim] - Vertex positions
        faces: [F, face_size] - Face connectivity (triangles, quads, etc.)
        features: Optional [N, feat_dim] - Per-vertex features
        
    Returns:
        GraphsTuple representation
    """
    if faces is not None:
        # Build edges from faces
        edges_list = []
        for face in faces:
            # Add edges for each pair in face
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    edges_list.append([face[i].item(), face[j].item()])
        
        edges = torch.tensor(edges_list, dtype=torch.long, device=vertices.device).T
        
        # Make bidirectional
        senders = torch.cat([edges[0], edges[1]])
        receivers = torch.cat([edges[1], edges[0]])
    else:
        # No faces provided - use KNN
        from torch_cluster import knn_graph
        edge_index = knn_graph(vertices, k=6, loop=False)
        senders = edge_index[0]
        receivers = edge_index[1]
    
    # Compute edge features
    edge_attr = compute_edge_features(vertices, senders, receivers, features='both')
    
    return GraphsTuple(
        nodes=features,
        edges=edge_attr,
        receivers=receivers,
        senders=senders,
        globals=None,
        n_node=torch.tensor([vertices.shape[0]], device=vertices.device),
        n_edge=torch.tensor([senders.shape[0]], device=vertices.device),
        positions=vertices,
    )
