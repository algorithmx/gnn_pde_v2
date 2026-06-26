## Independent Architecture Review — gnn_pde_v2

### 1. Inconsistent model base-class & registration story (highest impact)

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

### 3. Dual extension mechanism (Protocol + ABC) with a real name collision

For edge processors, node updaters, and edge assemblers there are **both** a `Protocol` (in protocols.py) **and** an abstract base class (in components), and every concrete class inherits from the ABC — so the Protocol is redundant. The worst case is a literal name clash:

- `EdgeFeatureAssembler` as a `Protocol` — protocols.py
- `EdgeFeatureAssembler` as an `nn.Module, ABC` — edge_assemblers.py

__init__.py exports the ABC under that name, shadowing the identically-named protocol that `architecture-dependencies.md` also lists. Two different types share one name. The validators' `isinstance(x, EdgeMessageProcessor/NodeUpdateStrategy)` checks (processor_validators.py) pass trivially because the ABC already guarantees the methods — the structural protocol adds no real constraint beyond the ABC. Pick one mechanism per concept.

---

### 4. Structural graph/grid protocols carry no enforceable contract

`GraphEncoder`, `GraphProcessor`, `NodeDecoder`, `GraphModel` are all `forward(GraphsTuple) -> GraphsTuple|Tensor`, and `PositionEncoder`/`GridProcessor`/`GridModel` are all `forward(Tensor) -> Tensor` — structurally identical, so `nn.ReLU()` satisfies the whole grid trio. The code itself admits they can't discriminate, and grepping shows they're used **only as type hints**, never as runtime checks. They are documentation that masquerades as a contract layer. Either give them distinct method names (`encode_graph`/`process`/`decode`) so they're genuinely distinguishable, or collapse them to a single `GraphStage` alias and stop implying enforcement.

---

### 5. Temperature interface leaks dense-attention tensor layout

Temperature modules assume a dense `[B, H, N, G]` logits layout. `SparseGraphAttention` must perform a `[E, H] → [1, H, E, 1] → [E, H]` reshape dance to reuse them (attention.py), and `adaptive` temperature is *forbidden* in sparse attention because the per-edge vs per-node semantics don't line up — a `ValueError` enforces a constraint that only exists because of the shape coupling. The temperature abstraction is coupled to one attention implementation's tensor shape rather than to a neutral "logits + features" contract.

---

### 6. Config objects coexist with flat-kwarg duplication in the transformer stack

`TransformerBlock` / `TransformerProcessor` correctly accept `PhysicsTokenConfig` / `RelativePositionConfig`, but both still carry `**legacy_kwargs` and route the same ~18 flat parameters at two levels (transformer.py). Worse, `PhysicsTokenAttention.__init__` re-declares the 10 `PhysicsTokenConfig` fields inline in a 20-parameter constructor — adding one config field requires editing both the dataclass and the constructor. The config refactor is half-done.

---

**Bottom line:** the *module dependency graph* remains clean and the *protocol docstrings* are now honest, so the old review's headline ("contracts are broken/misleading") is largely resolved. The real current weaknesses are **inconsistency**, not deception: an uneven model base-class/registration story (#1), a "central" EPD pipeline that only half the models use (#2), and redundant/colliding extension mechanisms (#3–#4). #5 and #6 are localized coupling/duplication smells in the attention/transformer stack.
