"""
Encode-Process-Decode architecture.

Core pattern from DeepMind Graph Nets.
"""

from typing import Optional, Union, Callable
import torch
import torch.nn as nn
from ..core.graph import GraphsTuple
from ..core.base import BaseModel


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
        encoder: Feature extraction module
        processor: Message passing/evolution module
        decoder: Output generation module
    """
    
    def __init__(
        self,
        encoder: EncoderProtocol,
        processor: ProcessorProtocol,
        decoder: DecoderProtocol,
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
        
        # Decode
        output = self.decoder(processed, query_positions)
        
        return output
    
    def get_latent(self, graph: GraphsTuple) -> GraphsTuple:
        """Get latent representation after encoding."""
        return self.encoder(graph)
    
    def get_processed(self, graph: GraphsTuple) -> GraphsTuple:
        """Get representation after full processing."""
        latent = self.encoder(graph)
        return self.processor(latent)
