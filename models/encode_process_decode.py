"""
Encode-Process-Decode architecture.

Core pattern from DeepMind Graph Nets.
"""

from typing import Optional, Union
import torch
from ..core.graph import GraphsTuple
from ..core.base import BaseModel
from ..core.protocols import GraphEncoder, GraphProcessor, NodeDecoder, QueryDecoder


class EncodeProcessDecode(BaseModel):
    """
    Encode-Process-Decode architecture.
    
    Modular 3-stage pipeline:
    1. Encoder: Extract latent features from input graph
    2. Processor: Evolve latent representation via message passing
    3. Decoder: Generate output from processed representation
    
    This separation enables:
    - Reusable processors across different I/O types
    - Probe-based decoding for arbitrary query points
    - Multi-fidelity architectures
    
    Args:
        encoder: Graph encoder satisfying :class:`~gnn_pde_v2.core.GraphEncoder`
            protocol; maps ``GraphsTuple → GraphsTuple``.
        processor: Graph processor satisfying
            :class:`~gnn_pde_v2.core.GraphProcessor` protocol; maps
            ``GraphsTuple → GraphsTuple``.
        decoder: Decoder satisfying either :class:`~gnn_pde_v2.core.NodeDecoder`
            (``GraphsTuple → Tensor``) or :class:`~gnn_pde_v2.core.QueryDecoder`
            (``(GraphsTuple, Tensor) → Tensor``).  ``EncodeProcessDecode``
            dispatches on the decoder's ``is_query_decoder`` class attribute:
            ``query_positions`` is forwarded only when ``is_query_decoder`` is
            ``True``.  Node decoders (the default) never receive
            ``query_positions``.
    """
    
    def __init__(
        self,
        encoder: GraphEncoder,
        processor: GraphProcessor,
        decoder: Union[NodeDecoder, QueryDecoder],
    ):
        super().__init__()
        
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
    
    def forward(
        self,
        graph: GraphsTuple,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through EPD architecture.
        
        Args:
            graph: Input GraphsTuple
            query_positions: Optional query points for probe decoding
            
        Returns:
            Model output (shape depends on decoder)
        """
        # Encode
        latent = self.encoder(graph)
        
        # Process
        processed = self.processor(latent)
        
        # Decode — dispatch on the decoder's declared kind so that NodeDecoder
        # implementations are never called with the query_positions argument
        # they don't accept. The discriminator is an explicit class attribute
        # (``is_query_decoder``) rather than an isinstance check, because
        # ``@runtime_checkable`` protocols cannot distinguish decoder signatures.
        if getattr(self.decoder, "is_query_decoder", False):
            output = self.decoder(processed, query_positions)
        else:
            output = self.decoder(processed)
        
        return output
    
    def get_latent(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Get latent representation after encoding.
        
        Args:
            graph: Input GraphsTuple
            
        Returns:
            GraphsTuple with encoded (latent) features
        """
        return self.encoder(graph)
    
    def get_processed(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Get representation after full processing (encoding + processing).
        
        Args:
            graph: Input GraphsTuple
            
        Returns:
            GraphsTuple with processed features
        """
        latent = self.encoder(graph)
        return self.processor(latent)
