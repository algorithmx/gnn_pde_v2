# GNN-PDE v2: High-Level Architecture Assessment

Based on `docs/architecture-dependencies.md`, the project follows a clean **layered architecture**:

- **Core** (`core/`): `GraphsTuple`, `BaseModel`, `MLP`, functional scatter ops, aggregation, registry, and structural protocols.
- **Components** (`components/`): interchangeable graph/grid building blocks (processors, attention, spectral, decoders, conditioning, multiscale, etc.).
- **Models** (`models/`): registered end-to-end models (`GraphNet`, `MeshGraphNet`, `FNO`, `MultiscaleFNO`) and `EncodeProcessDecode`.
- **Examples** (`examples/`): driver scripts.
- **Utils** (`utils/`): graph/spatial helpers.

The architecture doc presents this as dependency-clean and ready for strict protocol enforcement. However, the design-issue docs (`protocol_issues_2026_06.md`, `issues_priority_sorted_transformer.md`) and a quick runtime check show several **high-level outstanding design problems**:

## 1. Structural protocols are structurally meaningless

The graph/grid protocols in `core/protocols.py` collapse to "has a `forward` method" at runtime because:

- They are `@runtime_checkable` and only declare `forward(...)`.
- `runtime_checkable` checks method names, not signatures.

Consequences:

- `GraphEncoder`, `GraphProcessor`, `NodeDecoder`, and `GraphModel` are **indistinguishable** (`isinstance(mlp, GraphEncoder) == True` for a decoder).
- `PositionEncoder`, `GridProcessor`, `GridModel` are satisfied by `nn.ReLU()`.
- `Decoder = Union[NodeDecoder, QueryDecoder]` is effectively just `nn.Module`.

This makes both the runtime `isinstance` checks and the static types misleading.

## 2. `EncodeProcessDecode` dispatch is broken

`models/encode_process_decode.py` tries to dispatch probe vs. node decoders with:

```python
if isinstance(self.decoder, QueryDecoder):
    output = self.decoder(processed, query_positions)
else:
    output = self.decoder(processed)
```

Because every `nn.Module` satisfies `QueryDecoder`, this branch is taken for **all** decoders. It only works today because `MLPDecoder.forward` happens to accept an optional `query_positions=None`. The dispatch does not provide the safety it claims.

## 3. Two incompatible extension models in `core/protocols.py`

The file advertises "structural typing without inheritance," but:

- `ConditioningProtocol` is an `nn.Module + ABC + Generic` base class, not a `Protocol`.
- Graph/grid protocols are `Protocol`s.
- `EdgeMessageProcessor` / `NodeUpdateStrategy` are protocols, but there are also ABC/base-class implementations in `components/edge_processors.py` and `components/node_updaters.py`.

So the framework has both nominal-inheritance and structural-typing extension mechanisms for similar concepts, with neither authoritative.

## 4. `ProbeDecoder` does not fit the `QueryDecoder` protocol

`QueryDecoder` only requires `query_positions`, but `ProbeDecoder.forward` also takes `query_features` and `n_query`, which real callers (e.g., `WindFarmGNO`) need. The protocol under-describes the actual contract, so `EncodeProcessDecode` cannot correctly drive a `ProbeDecoder`.

## 5. `FourierFeatureEncoder` still does not satisfy `GraphEncoder`

It exposes `forward(x: Tensor) -> Tensor` (position encoding) and a separate `encode_graph(...)` method, so `isinstance(FourierFeatureEncoder(...), GraphEncoder)` is `False`. This was flagged in earlier reviews and remains unresolved.

## 6. No protocol-conformance test coverage

`tests/` never imports `NodeDecoder`/`QueryDecoder` for `isinstance` assertions. The "all components satisfy protocols" claims in `PROTOCOL_FIX_SUMMARY.md` are not backed by tests.

## 7. Transformer/attention stack has design smells

Per `issues_priority_sorted_transformer.md`:

- `SparseGraphAttention` does reshape gymnastics to fit temperature modules that expect 4D logits.
- `TransformerBlock` and `TransformerProcessor` duplicate ~22 constructor kwargs each, creating a maintenance tax.
- `PhysicsTokenAttention` is still a large class (~250 lines), though `TiledSliceOperation` was already extracted.

## Bottom line

The **module dependency graph is clean**, but the **contract layer (protocols)** is the main architectural weakness. The protocols look precise but are not enforced at runtime or by static analysis, and the `EncodeProcessDecode` dispatch relies on them in a way that is currently broken. The likely architectural decisions ahead are:

- Give protocols distinct method names (e.g., `encode_graph`, `process`, `decode`) so they are actually distinguishable, or
- Admit these are positional roles and collapse them to a single `GraphStage`/`nn.Module` type.
- Fix the `QueryDecoder` dispatch in `EncodeProcessDecode` with an explicit discriminator or a separate class.
- Move `ConditioningProtocol` out of the structural protocols file and align on one extension mechanism (ABC vs. Protocol) per concept.
