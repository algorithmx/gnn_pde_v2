# Core Examples (Lean Approach)

These examples demonstrate the **lean core** usage of the GNN-PDE framework.

## Philosophy

- **Explicit over implicit**: No magic, no registry, no metaclasses
- **Import only what you need**: Fine-grained control over dependencies
- **Standard PyTorch**: Just nn.Module with helpful components

## Examples

### `meshgraphnets_core.py`

Minimal MeshGraphNets implementation using:
- `gnn_pde_v2.GraphsTuple` - Graph data structure
- `gnn_pde_v2.components.MLP` - Building block
- `gnn_pde_v2.components.GraphNetBlock` - Processor
- `gnn_pde_v2.components.Residual` - Skip connection

## Usage

```python
from gnn_pde_v2 import GraphsTuple
from gnn_pde_v2.components import MLP, GraphNetBlock

class MyModel(nn.Module):
    def __init__(self):
        self.encoder = MLP(10, 128, [128])
        self.processor = GraphNetBlock(128, 128)
    
    def forward(self, graph: GraphsTuple):
        nodes = self.encoder(graph.nodes)
        processed = self.processor(GraphsTuple(nodes=nodes, ...))
        return processed.nodes
```

## When to Use Core

- Research experiments needing full control
- Production code requiring explicit dependencies
- When you want to understand every layer
- Minimizing external dependencies
