## Independent Architecture Review — gnn_pde_v2

### 1. Inconsistent model base-class & registration story (highest impact, resolved)

> **RESOLVED (2026-06-17).** The decorator/imperative write API
> (`@MODEL_REGISTRY.register` / `.add()`) has been removed from `MODEL_REGISTRY`; the
> FNO family and `MultiscaleFNO` now subclass `AutoRegisterModel`; and
> `MultiscaleFNO` is both registered (`multiscalefno`/`multiscale_fno`/`msfno`) and
> added to `_LAZY_MODELS`. There is now exactly one registration mechanism and one
> inheritance root, `isinstance(m, BaseModel)` is `True` for every shipped model, and
> the change is guarded by `tests/test_fno.py::TestRegistryWriteApiRemoved`. Full
> evidence and runtime re-verification:
> `docs/investigation-report-model-base-class-and-registration.md` §0.

**Historical (pre-fix) record below — no longer reflects the code:**

There are two registration paths *and* two inheritance roots, applied unevenly:

| Model | Registration | Base class | Uses EPD | Registered | Exported |
|-------|-------------|-----------|----------|-----------|----------|
| `GraphNet`/`MeshGraphNet` | `AutoRegisterModel` | `BaseModel` | yes | yes | lazy |
| `FNO`/`TFNO`/`AFNO` | `@MODEL_REGISTRY.register` | plain `nn.Module` | no | yes | lazy |
| `MultiscaleFNO` | none | plain `nn.Module` | no | **no** | **no** |
| `EncodeProcessDecode` | none | `BaseModel` | n/a | **no** | eager |

Consequences:
- `isinstance(model, BaseModel)` is unreliable — it silently misses every FNO variant, so any serialization/introspection keyed on `BaseModel` is broken.
- Two creation entry points (`AutoRegisterModel.create('graphnet')` vs `MODEL_REGISTRY.create('fno')`) with no single canonical API.
- `MultiscaleFNO` is a substantial model that is invisible to the registry and package exports (__init__.py `_LAZY_MODELS` omits it).

This is the most user-facing structural weakness.

---

### 2. `EncodeProcessDecode` is not actually the framework's central pipeline

The architecture is presented as Encode-Process-Decode, and the decoder protocols/`is_query_decoder` dispatch exist to serve it — but only graph models route through it (gnn_model.py). Grid models bypass it entirely and wrap `FNOProcessor` directly (fno_model.py). So the entire `QueryDecoder` / `NodeDecoder` / dispatch machinery is graph-only, and the "central pattern" is really one of two unrelated assembly styles. The protocol layer's effort is concentrated on a path most models don't use.

---

### 3. Dual extension mechanism (Protocol + ABC) with a real name collision (resolved)

> **RESOLVED (2026-06-29).** Each of the three pluggable component contracts now
> has exactly one mechanism: a single public ABC living next to its concrete
> implementations in `components/`. The colliding/redundant `typing.Protocol`
> twins were deleted from `core/protocols.py`. `_EdgeMessageProcessorBase` →
> public `EdgeMessageProcessor` (`components/edge_processors.py`),
> `_NodeUpdaterBase` → public `NodeUpdateStrategy` (`components/node_updaters.py`),
> and `EdgeFeatureAssembler` remains the public ABC (`components/edge_assemblers.py`).
> Each ABC carries a `__subclasshook__` so duck-typed `nn.Module`s still pass
> `isinstance` (open for extension without inheritance). Validators
> (`processor_validators.py`), `core/__init__.py`, `components/__init__.py`, and
> `processors.py` import the single name; `core.protocols.__all__` no longer
> lists them. Guarded by `tests/test_protocol_conformance.py::TestProtocolsNotRuntimeCheckable::test_component_contracts_are_single_abc`.
> 448 tests pass (3 pre-existing unrelated failures).

**Historical (pre-fix) record below — no longer reflects the code:**

Three pluggable extension points each have **both** a `typing.Protocol` (in `core/protocols.py`) **and** an abstract base class (`nn.Module, ABC` in `components/`). Every concrete class inherits from the ABC, so the Protocol is structurally redundant — it adds no runtime constraint beyond what the ABC already enforces.

#### 3a. `EdgeFeatureAssembler` — literal name collision (most severe)

Two unrelated types share the same qualified name:

| Location | Declaration |
|----------|------------|
| `core/protocols.py:163` | `class EdgeFeatureAssembler(Protocol)` |
| `components/edge_assemblers.py:46` | `class EdgeFeatureAssembler(nn.Module, ABC)` |

`components/__init__.py:66–72` imports the **ABC** from `.edge_assemblers` and exports it in `__all__` (line 172), but does **not** re-export the Protocol (the Protocol re-export block at lines 126–137 omits `EdgeFeatureAssembler`). So `from gnn_pde_v2.components import EdgeFeatureAssembler` gives you the ABC — the Protocol is shadowed. Meanwhile `architecture-dependencies.md` references "`EdgeFeatureAssembler` protocol" without specifying which one. Code that imports the Protocol from `gnn_pde_v2.core.protocols` and code that imports the ABC from `gnn_pde_v2.components` are operating on two completely different types with the same name.

The concrete assemblers (`NodeDifferenceAssembler`, `ConcatAssembler`, `DifferenceOnlyAssembler`, `ConcatWithEdgesAssembler`) all inherit from the ABC. The Protocol is tested nowhere; the ABC is enforced by `tests/test_edge_assemblers.py::TestEdgeFeatureAssemblerABC` (instantiation rejection, missing-method rejection).

#### 3b. `EdgeMessageProcessor` — private ABC, redundant Protocol

| Location | Declaration |
|----------|------------|
| `core/protocols.py:114` | `class EdgeMessageProcessor(Protocol)` — `@runtime_checkable`, declares `latent_dim`, `weight_out_dim` property, `forward` |
| `components/edge_processors.py:23` | `class _EdgeMessageProcessorBase(nn.Module, ABC)` — **private** (`_` prefix), declares `latent_dim`, `weight_out_dim`, `forward`, plus `verify_shape_contract` |

All four concrete processors (`FullEdgeMessageProcessor`, `VectorEdgeMessageProcessor`, `ScalarEdgeMessageProcessor`, `LowRankEdgeMessageProcessor`) inherit from `_EdgeMessageProcessorBase`. The Protocol is re-exported in `components/__init__.py` (line 129, 232).

The validator `validate_edge_message_processor` (`processor_validators.py:26`) checks `isinstance(edge_processor, EdgeMessageProcessor)` — but this passes trivially because inheriting from `_EdgeMessageProcessorBase` already guarantees the required methods/properties, making the objects structurally satisfy the Protocol. The Protocol adds zero additional safety. The real runtime safety comes from:
1. The ABC forcing `weight_out_dim` and `forward` to be implemented (or instantiation fails with `TypeError`).
2. The `verify_shape_contract` method on `_EdgeMessageProcessorBase` (line 38–67), which runs an eager forward-pass shape check.
3. The dedicated `verify_edge_message_pipeline` pipeline-level check (line 104–137).

The `isinstance` check against the Protocol is decorative — removing it would change nothing, because the `nn.Module` check on line 31 already catches non-Module inputs and the ABC already ensures the interface.

#### 3c. `NodeUpdateStrategy` — private ABC, redundant Protocol, orphaned export

| Location | Declaration |
|----------|------------|
| `core/protocols.py:141` | `class NodeUpdateStrategy(Protocol)` — `@runtime_checkable`, declares `latent_dim`, `forward` |
| `components/node_updaters.py:75` | `class _NodeUpdaterBase(nn.Module, ABC)` — **private**, declares `latent_dim`, `forward` |

All four concrete updaters (`ConcatMLPNodeUpdater`, `RootWeightNodeUpdater`, `PassThroughNodeUpdater`, `ResidualMLPNodeUpdater`) inherit from `_NodeUpdaterBase`. The Protocol is **not** re-exported from `components/__init__.py` (omitted from the Protocol re-export block at lines 126–137 and from `__all__` at lines 230–239). This was documented as issue #9 in `docs/legacy/protocol_issues_2026_06.md` (“missing from protocols.__all__”).

The validator `validate_node_update_strategy` (`processor_validators.py:55`) mirrors the `EdgeMessageProcessor` pattern exactly: `isinstance(node_updater, NodeUpdateStrategy)` passes trivially because the ABC already provides the interface. The docstring on line 61–63 even acknowledges the gap: “`@runtime_checkable` cannot verify the `latent_dim` attribute's type or value, nor that the strategy is an `nn.Module`; this helper closes that gap at construction time.” The actual enforcement is the validator's own `nn.Module` check and manual `latent_dim` type/value inspection — the `isinstance` Protocol check is a no-op.

#### Summary

All three pairs follow the same pattern: a `Protocol` describes an interface, a private (`_`-prefixed) `nn.Module, ABC` enforces it via inheritance, and the concrete classes use the ABC. The Protocol is either shadowed (#3a), a redundant `isinstance` pass-through (#3b), or both redundant and orphaned from the public export surface (#3c). Pick one mechanism per concept and remove the other.

---

### 4. Structural graph/grid protocols carry no enforceable contract (resolved)

> **RESOLVED (2026-06-26).** The five graph-world stage protocols
> (`GraphEncoder`, `GraphProcessor`, `NodeDecoder`, `QueryDecoder`, `GraphModel`)
> are no longer `@runtime_checkable` — they are plain `Protocol` classes serving
> as static-typing hints only. The orphaned grid trio (`PositionEncoder`,
> `GridProcessor`, `GridModel`) and the deprecated `Decoder = Union[NodeDecoder,
> QueryDecoder]` alias have been deleted entirely (zero production/test
> consumers). Runtime dispatch in `EncodeProcessDecode` continues to use the
> `is_query_decoder` class attribute, the only mechanism it ever trusted. The
> three component-contract protocols (`EdgeMessageProcessor`,
> `NodeUpdateStrategy`, `EdgeFeatureAssembler`) are unchanged — they belong to
> issue #3. The change is guarded by
> `tests/test_protocol_conformance.py::TestProtocolsNotRuntimeCheckable` (12 new
> regression tests). Full plan and evidence:
> `docs/remediation-plan-issue4-structural-protocols.md`.

**Historical (pre-fix) record below — no longer reflects the code:**

Nine `@runtime_checkable` protocols defined in `core/protocols.py` fall into four structurally identical clusters:

| Cluster | Protocols | Signature | Distinguishable at runtime? |
|---------|-----------|-----------|----------------------------|
| A | `GraphEncoder`, `GraphProcessor` | `forward(GraphsTuple) → GraphsTuple` | No — identical |
| B | `NodeDecoder`, `GraphModel` | `forward(GraphsTuple) → Tensor` | No — identical |
| C | `PositionEncoder`, `GridProcessor`, `GridModel` | `forward(Tensor) → Tensor` | No — identical (4th bullet below) |
| D | `QueryDecoder` | `forward(GraphsTuple, Tensor, ...) → Tensor` | Marginal — `@runtime_checkable` can't check signatures, only method existence |

The clusters differ *on paper* (A → GraphsTuple, B → Tensor, C → Tensor×Tensor, D → varied), but `@runtime_checkable` only verifies that named methods **exist** — it does not inspect signatures, return types, or distinguish `GraphsTuple` from `Tensor`.

#### 4a. Concrete consequences

1. **`nn.ReLU()` satisfies the entire C cluster.** Any `nn.Module` whose `forward` accepts and returns a `Tensor` — which is every `nn.Module` — passes `isinstance(x, PositionEncoder)`, `isinstance(x, GridProcessor)`, and `isinstance(x, GridModel)` simultaneously. A random `nn.Linear(64, 64)` structurally "satisfies" the grid processor contract.

2. **An encoder can't tell a processor from itself.** Both `GraphEncoder` and `GraphProcessor` declare `forward(graph: GraphsTuple) -> GraphsTuple`. The only way to know which stage a module belongs to is *where it's wired in the pipeline*, not any property of the module itself.

3. **Production code never uses these as runtime discriminators.** A grep for `isinstance(... GraphEncoder|GraphProcessor|GraphModel|GridProcessor|GridModel|PositionEncoder|NodeDecoder)` across the entire codebase hits **zero** results in production code. The only hits are in `core/protocols.py` *docstring examples* that unwittingly demonstrate the problem:
   
   ```python
   # protocols.py:104-107 — the example that proves nothing
   proc: GraphProcessor = GraphNetProcessor(latent_dim=128, n_layers=6)
   assert isinstance(proc, GraphProcessor)  # True at runtime
   # ^ Would also pass for GraphEncoder, NodeDecoder, and any nn.Module
   ```

   The sole runtime consumer is `EncodeProcessDecode.__init__` (line 43–47), which uses `GraphEncoder`, `GraphProcessor`, and `Union[NodeDecoder, QueryDecoder]` as **type annotations only** — static mypy hints, not runtime guards.

4. **`EncodeProcessDecode.forward` explicitly rejects protocol-based dispatch.** The decoder-branching logic (line 81) reads `getattr(self.decoder, "is_query_decoder", False)` — an explicit class attribute — rather than `isinstance(decoder, QueryDecoder)`. This is deliberate, documented in both the method (lines 79–80) and the test suite (`test_protocol_conformance.py:139–142`):
   
   ```python
   # test_protocol_conformance.py:138-142
   def test_dispatch_uses_flag_not_isinstance(self):
       """The dispatch must read the discriminator attribute, not a protocol."""
       src = inspect.getsource(EncodeProcessDecode.forward)
       assert "is_query_decoder" in src
       assert "isinstance(self.decoder, QueryDecoder)" not in src
   ```

5. **The `Decoder` Union is doubly broken.** `Decoder = Union[NodeDecoder, QueryDecoder]` (line 262) combines two `@runtime_checkable` protocols into a Union. In CPython, `isinstance(x, Union[A, B])` where both A and B are runtime-checkable degrades to `isinstance(x, A) or isinstance(x, B)`. Since *both* only check for `forward` method existence, `isinstance(x, Decoder)` is true for literally any object with a `forward` method — it can't discriminate between the two protocols, between a decoder and an encoder, or between a model and a ReLU. The code itself labels it deprecated (line 253–261): "useless as a runtime discriminator."

#### 4b. Real-world workarounds already exist in the codebase

Two distinct patterns have evolved to cope with the gap between these protocols and runtime needs:

| Pattern | File | What it does |
|---------|------|-------------|
| **Explicit method name** | `components/fourier_encoder.py:27–32` | `FourierFeatureEncoder.forward` goes `Tensor→Tensor` (satisfies `PositionEncoder`), but its graph-level method is deliberately named `encode_graph` instead of `forward`, so it does **not** satisfy `GraphEncoder`. Docstring explains: "Call `encode_graph` explicitly rather than passing this object where a `GraphEncoder` is expected." |
| **Adapter wrapper** | `examples/example_gnn_solver_simplified.py:51–65` | `MLPEncoder` wraps an `MLP` (which is `Tensor→Tensor`) and provides `forward(GraphsTuple)→GraphsTuple` — the adapter pattern needed precisely because a plain `MLP` structurally satisfies protocols it shouldn't. |
| **Class attribute discriminator** | `components/decoders.py`, `components/probe.py` | `MLPDecoder.is_query_decoder = False`; `ProbeDecoder.is_query_decoder = True`. The only mechanism `EncodeProcessDecode` actually trusts at runtime. |

#### 4c. The grid trio is completely orphaned (compounded by issue #2)

`PositionEncoder`, `GridProcessor`, and `GridModel` are defined in `core/protocols.py` and re-exported from `components/__init__.py` and `core/__init__.py` — but:

- **No grid model** (FNO, TFNO, AFNO, MultiscaleFNO) uses `EncodeProcessDecode` (issue #2).
- **No grid model** imports these protocols (`fno_model.py` imports from `..core.registry` and `..components.spectral` only).
- **No grid component** (`spectral.py`, `FNOProcessor`, `AFNOBlock`) imports them.
- **No test** references them.

They exist solely as seven lines of docstring that look like a contract layer but are actually an aspirational architecture diagram that was never implemented. Combined with the structural indistinguishability, they are *documentation shell classes* — in 100% of use cases, a reviewer seeing `def forward(self, x: Tensor) -> Tensor` already knows what the function does better than a `PositionEncoder` type hint could possibly express.

#### 4d. Remediation options

**Option 1 — Distinct method names (genuinely enforceable).** Give each role a method with a unique name that `@runtime_checkable` can distinguish:

```python
class GraphEncoder(Protocol):
    def encode_graph(self, graph: GraphsTuple) -> GraphsTuple: ...

class GraphProcessor(Protocol):
    def process(self, graph: GraphsTuple) -> GraphsTuple: ...

class NodeDecoder(Protocol):
    def decode(self, graph: GraphsTuple) -> Tensor: ...
```

Then `isinstance(obj, GraphEncoder)` genuinely requires `encode_graph` to exist. A `GraphNetProcessor` with only `forward` would fail. The cost is that `EncodeProcessDecode` must call `encoder.encode_graph(latent)` instead of `encoder(latent)`, and every concrete component must rename its forward path for the role it fills. This is the only option that makes protocols enforceable.

**Option 2 — Collapse to type aliases (honest documentation).** Remove `@runtime_checkable`, keep the protocols as `TypedDict`-style type aliases that serve only as documentation:

```python
GraphEncoder = Callable[[GraphsTuple], GraphsTuple]  # or keep Protocol without @runtime_checkable
GraphProcessor = Callable[[GraphsTuple], GraphsTuple]
```

This makes the cost zero while preserving the documentation value. The existing `getattr(decoder, "is_query_decoder", False)` dispatch continues to work.

**Option 3 — Remove the grid trio entirely.** Since no code uses them and issue #2 means the EPD pipeline doesn't apply to grid models, `PositionEncoder`/`GridProcessor`/`GridModel` can be deleted with zero production impact. Re-add them with distinct method names if/when a grid EPD pipeline materializes.

**Recommendation:** Option 2 + Option 3. The graph-world protocols have documentation value for `EncodeProcessDecode`'s type annotations; drop `@runtime_checkable` and the docstring examples that imply enforcement. Remove the grid trio until there's a grid EPD pipeline that needs them. The `Decoder` Union should be removed — it's already marked deprecated and no new code should use it.

---

### 5. Temperature interface leaks dense-attention tensor layout

Temperature modules assume a dense `[B, H, N, G]` logits layout. `SparseGraphAttention` must perform a `[E, H] → [1, H, E, 1] → [E, H]` reshape dance to reuse them (attention.py), and `adaptive` temperature is *forbidden* in sparse attention because the per-edge vs per-node semantics don't line up — a `ValueError` enforces a constraint that only exists because of the shape coupling. The temperature abstraction is coupled to one attention implementation's tensor shape rather than to a neutral "logits + features" contract.

---

### 6. Config objects coexist with flat-kwarg duplication in the transformer stack

`TransformerBlock` / `TransformerProcessor` correctly accept `PhysicsTokenConfig` / `RelativePositionConfig`, but both still carry `**legacy_kwargs` and route the same ~18 flat parameters at two levels (transformer.py). Worse, `PhysicsTokenAttention.__init__` re-declares the 10 `PhysicsTokenConfig` fields inline in a 20-parameter constructor — adding one config field requires editing both the dataclass and the constructor. The config refactor is half-done.

---

**Bottom line:** the *module dependency graph* remains clean and the *protocol docstrings* are now honest, so the old review's headline ("contracts are broken/misleading") is largely resolved. The real current weaknesses are **inconsistency**, not deception: an uneven model base-class/registration story (#1), a "central" EPD pipeline that only half the models use (#2), and redundant/colliding extension mechanisms (#3–#4). #5 and #6 are localized coupling/duplication smells in the attention/transformer stack.
