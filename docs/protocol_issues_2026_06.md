# Protocol Issues (2026-06-16)

An audit of `core/protocols.py` and its consumers on `main` (commit
`d87b8b1`, verified under Python 3.13.9 with `conda env ml_env`).

## Background

The framework defines a set of `typing.Protocol`-based interfaces in
`core/protocols.py` with the stated goal of enabling *strict protocol
enforcement* — i.e. component types like `GraphEncoder`, `GraphProcessor`,
and `Decoder` should reject incompatible modules at static-analysis time
*and* at runtime via `isinstance`.

The protocol surface (excerpt):

```
Graph-world:   GraphEncoder, GraphProcessor, NodeDecoder, QueryDecoder,
               Decoder (= Union[NodeDecoder, QueryDecoder]), GraphModel,
               EdgeMessageProcessor, EdgeFeatureAssembler, NodeUpdateStrategy
Grid-world:    PositionEncoder, GridProcessor, GridModel
Conditioning:  ConditioningProtocol (nn.Module + ABC + Generic[CondT]),
               Modulation
```

All graph/grid protocols are marked `@runtime_checkable` with a single
`forward` method. This document catalogues where that design breaks down.

Every claim below was verified by running the code, not by reading
docstrings.

## Scope

This document is the current, authoritative catalogue of protocol issues.
Earlier internal review notes have been superseded and are no longer
canonical.

---

## 0. Summary of findings

| # | Issue | Severity | Category |
|---|-------|----------|----------|
| 1 | `isinstance(decoder, QueryDecoder)` is True for **every** `nn.Module` — the EPD dispatch is meaningless | **Critical** | Correctness |
| 2 | `GraphEncoder`, `GraphProcessor`, `NodeDecoder`, `GraphModel` are indistinguishable at runtime | **Critical** | Design |
| 3 | `PositionEncoder` / `GridProcessor` / `GridModel` all collapse to "has `forward`"; `nn.ReLU` satisfies all three | **High** | Design |
| 4 | `Decoder = Union[NodeDecoder, QueryDecoder]` cannot be used for runtime dispatch and lies to static checkers | **High** | Typing |
| 5 | `ConditioningProtocol` inherits from `nn.Module + ABC + Generic`, contradicting the file's stated "structural typing" philosophy | **High** | Architecture |
| 6 | `runtime_checkable` does not verify signatures, attribute types, or `@property` members — `EdgeMessageProcessor.weight_out_dim` and `NodeUpdateStrategy.latent_dim` are effectively unenforced | **High** | Enforcement gap |
| 7 | `ProbeDecoder.forward` takes `query_features`/`n_query` not present in `QueryDecoder`, so the protocol under-describes the real contract | **Medium** | API drift |
| 8 | `FourierFeatureEncoder.encode_graph()` still does not satisfy `GraphEncoder` (method name mismatch) — unfixed since the first review | **Medium** | Conformance |
| 9 | `NodeUpdateStrategy` is exported from `core/__init__.py` but missing from `protocols.__all__` — inconsistent public surface | **Medium** | Export hygiene |
| 10 | Protocol conformance is asserted in docs but has **no test suite**; `tests/` never imports `NodeDecoder`/`QueryDecoder` for isinstance assertions on real components | **Medium** | Verification gap |
| 11 | `ProbeDecoder.processor` is typed as bare `nn.Module`, not `GraphProcessor` — protocol discipline is applied selectively | **Low** | Inconsistency |
| 12 | No protocol for `Aggregation`, `EdgeFeatureAssembler` consumers, or the probe-graph builder, even though these are designed as pluggable extension points | **Low** | Coverage gap |
| 13 | Protocols carry no shape/dim information, so static analysis cannot catch latent_dim mismatches between encoder→processor→decoder | **Low** | Type expressiveness |

---

## 1. CRITICAL: the `QueryDecoder` dispatch in `EncodeProcessDecode` does not work

`models/encode_process_decode.py:76` does:

```python
if isinstance(self.decoder, QueryDecoder):
    output = self.decoder(processed, query_positions)
else:
    output = self.decoder(processed)
```

The intent is: route to `QueryDecoder`-shaped decoders only when the decoder
really takes `query_positions`. The implementation does not do this.

### Why

`@runtime_checkable` protocols, by Python's design, check **only that the
method name exists** — they do not inspect the signature
([CPython docs — `typing.Protocol`, "runtime_checkable"](https://docs.python.org/3/library/typing.html#typing.runtime_checkable)).
Every `nn.Module` already defines `forward`, so every `nn.Module` satisfies
every protocol in this file whose only member is `forward`.

Verified at runtime under Python 3.13:

```
MLPDecoder is NodeDecoder: True
MLPDecoder is QueryDecoder: True   # <-- should be False
ProbeDecoder is NodeDecoder: True  # <-- should be False
ProbeDecoder is QueryDecoder: True
MLPDecoder is GraphModel: True     # <-- a decoder "is" an end-to-end model
```

So the `if isinstance(self.decoder, QueryDecoder)` branch is taken for
**every** decoder, including `MLPDecoder` and `IndependentMLPDecoder`. The
only reason tests do not explode is that `MLPDecoder.forward` happens to
accept an optional `query_positions=None` argument and silently ignores it.
The intended "strict dispatch" provides zero protection:

- If a user writes a `NodeDecoder` with the strict signature
  `forward(self, graph: GraphsTuple) -> Tensor` (no `query_positions`
  parameter), `EncodeProcessDecode` will still call it with two positional
  args and crash at runtime — exactly the case the dispatch was supposed to
  prevent.
- If `query_positions` is `None` and the decoder really is a `ProbeDecoder`,
  the dispatch will still happily pass `None` through.

### Fix options

Pick one; do not leave the broken dispatch in place.

1. **Drop the union entirely.** Make `EncodeProcessDecode` take a single
   concrete `decoder: nn.Module` and *always* call
   `decoder(processed, query_positions=query_positions)`. Decoders accept or
   ignore `query_positions` via their own signature. Simplest, no runtime
   introspection, no false claims.
2. **Use an explicit discriminator.** Add `is_query_decoder: bool = False`
   (or a small `DecoderKind` enum) on decoder classes; dispatch on that.
   Cheap, honest, and unaffected by signature drift.
3. **Split the EPD class.** `EncodeProcessDecode` for node decoders,
   `ProbeEncodeProcessDecode` for query decoders. Removes the union at the
   type level — which is what the doc claimed to do but didn't.

Do **not** keep `isinstance(.., QueryDecoder)` as the discriminator.

---

## 2. CRITICAL: `GraphEncoder` ≡ `GraphProcessor` ≡ `NodeDecoder` ≡ `GraphModel` at runtime

All four are defined as `@runtime_checkable` protocols whose only member is
`def forward(self, ...) -> ...`. Because `runtime_checkable` does not check
signatures, the four protocols are **indistinguishable** by `isinstance`:

```python
# MLPDecoder (a node decoder) satisfies:
isinstance(mlp, GraphEncoder)     # True  -- "encodes" a graph?
isinstance(mlp, GraphProcessor)   # True  -- "processes" a graph?
isinstance(mlp, NodeDecoder)      # True
isinstance(mlp, GraphModel)       # True  -- end-to-end model?
```

Consequences:

- The type annotations on `EncodeProcessDecode.__init__`
  (`encoder: GraphEncoder`, `processor: GraphProcessor`) provide **no static
  guarantee either**, because the protocols are structurally identical.
  Mypy and pyright cannot tell them apart either: any object with a
  `forward(GraphsTuple) -> X` method satisfies all of them.
- A user can swap encoder and processor without any type error, even though
  the semantics are different (encoder typically lifts raw features into
  latent space; processor does not).

### Fix

Give each protocol a **non-`forward`** marker that the type checker can use
to differentiate, OR accept that these are the same protocol and collapse
them. Examples:

```python
@runtime_checkable
class GraphEncoder(Protocol):
    def encode_graph(self, graph: GraphsTuple) -> GraphsTuple: ...
```

i.e. mandate a distinct method name per role. Then `isinstance` and the
static checkers can actually distinguish them. This requires renaming
`forward` to `encode_graph`/`process`/`decode` on the implementations,
which is a one-time mechanical change.

The alternative is to acknowledge that "graph-stage" is a *positional*
property (where in the EPD pipeline the module sits), not a *structural*
one, and stop pretending the protocol carries that information.

---

## 3. HIGH: `PositionEncoder` / `GridProcessor` / `GridModel` all collapse to `nn.Module`

Same root cause as §2, even more embarrassing:

```python
isinstance(nn.ReLU(), PositionEncoder)   # True
isinstance(nn.ReLU(), GridProcessor)     # True
isinstance(nn.ReLU(), GridModel)         # True
```

A scalar activation satisfies every grid protocol. The protocols convey no
information beyond "is an `nn.Module`."

### Fix

Same as §2: either rename the protocol methods (`encode_positions`,
`process_grid`, `forward_grid`), or admit these protocols are decorative and
just use `nn.Module` directly.

---

## 4. HIGH: `Decoder = Union[NodeDecoder, QueryDecoder]` is the wrong tool

`core/protocols.py` defines:

```python
Decoder = Union[NodeDecoder, QueryDecoder]
```

and used it as the annotation for `EncodeProcessDecode.decoder`. The union
is misleading in two ways:

1. **Static typing:** Union-of-Protocols forces every call site to narrow
   the type before calling `forward`, because the two arms have incompatible
   signatures. The only honest way to narrow is the very `isinstance` check
   we just showed (in §1) does not work at runtime. So the union pretends
   to offer type safety that the runtime cannot deliver.
2. **Runtime:** `isinstance(x, Decoder)` against a Union of
   `runtime_checkable` protocols silently degrades to "does it satisfy
   *either* arm" — which, per §1, means "does it have a `forward` method,"
   which is true for every `nn.Module`. Verified:

   ```
   isinstance(nn.ReLU(), Decoder)        # True
   ```

So `Decoder` is functionally equivalent to `nn.Module` at runtime, while
*appearing* to be a precise sum type in the source. This is the worst of
both worlds.

### Fix

Pick one concrete decoder protocol (probably `NodeDecoder`, since most use
cases are node-output) and have a *separate* class for the probe variant
(see §1 fix option 3). Drop the Union.

---

## 5. HIGH: `ConditioningProtocol` breaks the file's own design claim

The module docstring of `core/protocols.py` advertises:

> Using `typing.Protocol` allows structural (duck-type) checking without
> requiring inheritance, making the system open for extension.

Then the first protocol defined is:

```python
class ConditioningProtocol(nn.Module, ABC, Generic[CondT]):
    @abstractmethod
    def forward(self, condition: CondT) -> Modulation: ...
```

This is a nominal ABC that requires inheritance (`ZeroConditioning`,
`AdaLNConditioning`, etc. all subclass it), not a structural Protocol. It
sits in `protocols.py` next to actual `Protocol` classes, creating two
incompatible extension models in the same file:

- "Implement `forward` with the right signature" (graph/grid protocols).
- "Inherit from `ConditioningProtocol[CondT]` and register as an
  `nn.Module`" (conditioning).

The two worlds cannot be checked the same way, cannot be substituted the
same way, and confuse readers about which rules apply.

### Fix

Move `ConditioningProtocol` (and `Modulation`, if it must live near it) out
of `protocols.py` — either to `core/conditioning.py` or keep it in
`components/conditioning.py` only. Leave `protocols.py` for *structural*
protocols only. Alternatively, convert it to a real
`Protocol[CondT]` and have the implementations stop inheriting from it.

---

## 6. HIGH: `runtime_checkable` does not enforce attribute/property contracts

`EdgeMessageProcessor` and `NodeUpdateStrategy` declare structural members
beyond `forward`:

```python
class EdgeMessageProcessor(Protocol):
    latent_dim: int
    @property
    def weight_out_dim(self) -> int: ...
    def forward(self, src_x: Tensor, edge_weights: Tensor) -> Tensor: ...

class NodeUpdateStrategy(Protocol):
    latent_dim: int
    def forward(self, nodes: Tensor, aggregated: Tensor) -> Tensor: ...
```

Two problems:

1. **`@property` members on `runtime_checkable` protocols are unreliable.**
   `isinstance` only verifies the attribute *name* exists on the instance;
   it cannot distinguish a property, a class attribute, or an instance
   attribute. Worse, on Python ≤ 3.11, inspecting data-attribute-only
   members of `runtime_checkable` protocols raises
   `TypeError: Protocols with non-method members don't support issubclass()`
   and degrades `isinstance` to method-name-only checks.
2. **Types are unchecked.** `latent_dim: int` on the protocol gives no
   runtime guarantee that the attribute is an `int`, only that something
   named `latent_dim` exists. A class with `latent_dim = "banana"` passes
   `isinstance(.., NodeUpdateStrategy)`.

Verified:

```python
class Fake:
    latent_dim = 8
    def forward(self, n, a): return n
isinstance(Fake(), NodeUpdateStrategy)   # True — even though Fake is not an nn.Module
```

The docstring for these two protocols says implementations "should also be
`nn.Module` instances" — but that contract is not encoded anywhere.

### Fix

If you want these contracts to be real, stop relying on `runtime_checkable`
for them. Either:

- Add a small `validate_edge_message_processor()` helper (which already
  exists in `components/processor_validators.py:25` — it is just never
  invoked from the protocol surface) and call it in
  `EdgeConditionedConvBlock.__init__`.
- Or add a base class (`_EdgeMessageProcessorBase`, which also already
  exists in `components/edge_processors.py:23`) and require inheritance.
  Then drop the protocol entirely — the base class already enforces
  `latent_dim` validation in `__init__`.

The current state is "we have both a Protocol and an ABC for the same thing,
and neither is authoritative."

---

## 7. MEDIUM: `ProbeDecoder.forward` does not match `QueryDecoder`

`QueryDecoder` declares:

```python
def forward(self, graph: GraphsTuple, query_positions: Tensor) -> Tensor: ...
```

`ProbeDecoder.forward` is actually:

```python
def forward(
    self,
    graph: GraphsTuple,
    query_positions: torch.Tensor,
    query_features: Optional[torch.Tensor] = None,
    n_query: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
```

The extra parameters are *required for correct usage* in `WindFarmGNO`,
which calls `self.probe_decoder(graph=..., query_positions=...,
query_features=...)`. The protocol under-describes the real contract, so
`EncodeProcessDecode` cannot drive a `ProbeDecoder` correctly — it has no
way to pass `query_features` or `n_query`.

### Fix

Either widen `QueryDecoder` to include the optional parameters (and accept
that the protocol now describes ProbeDecoder specifically rather than any
query-based decoder), or introduce a separate `ProbeDecoderProtocol` that
matches reality and stop calling `ProbeDecoder` a `QueryDecoder`.

---

## 8. MEDIUM: `FourierFeatureEncoder` does not satisfy `GraphEncoder`

`components/fourier_encoder.py` exposes two methods:

```python
class FourierFeatureEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...     # PositionEncoder-shaped
    def encode_graph(self, graph: GraphsTuple) -> GraphsTuple: ...  # NOT named forward
```

`isinstance(FourierFeatureEncoder(...), GraphEncoder)` returns **False**
because the graph-encoding method is named `encode_graph`, not `forward`.
This is the one case in the codebase where the protocol accidentally
behaves correctly — but only by accident. Either:

- Rename `encode_graph` → `forward` and overload `forward` to accept either
  `Tensor` or `GraphsTuple` (dispatch on type), making the class satisfy
  both `PositionEncoder` and `GraphEncoder`. Brittle but workable.
- Or split the class into a `FourierPositionEncoder` and a
  `FourierGraphEncoder` that share a private kernel bank.

---

## 9. MEDIUM: `NodeUpdateStrategy` is missing from `protocols.__all__`

`core/protocols.py:325` exports `EdgeMessageProcessor` but not
`NodeUpdateStrategy`. Meanwhile `core/__init__.py:29` *does* re-export
`NodeUpdateStrategy`. So:

- `from gnn_pde_v2.core.protocols import NodeUpdateStrategy` — works.
- `from gnn_pde_v2.core.protocols import *` — does **not** bring in
  `NodeUpdateStrategy`.

Public surface is inconsistent. Add `"NodeUpdateStrategy"` to `__all__`.

---

## 10. MEDIUM: there is no test that real components satisfy the protocols

`grep -rn "NodeDecoder\|QueryDecoder" tests/` returns zero hits — no test
in `tests/` asserts that the registered components actually satisfy their
claimed protocols. The design intent of "strict enforcement" is therefore
unverified at the CI level.

### Fix

Add `tests/test_protocol_conformance.py` that, for each registered
component in `MODEL_REGISTRY` and each component exported from
`components/__init__.py`, asserts:

- `isinstance(component, expected_protocol)` is True,
- `isinstance(component, unrelated_protocol)` is **False** (this will fail
  today, by design — see §2 — and that failure is the point),
- The component's `forward` signature, inspected via `inspect.signature`,
  is compatible with the protocol's `forward` signature (this is the check
  `runtime_checkable` cannot do).

Without these tests, the next refactor will silently break protocol
conformance again.

---

## 11. LOW: `ProbeDecoder.processor` is typed as bare `nn.Module`

`components/probe.py:232`:

```python
processor: nn.Module,
```

If the project's stated goal is protocol enforcement, this should be
`GraphProcessor` (or a probe-specific protocol). Using `nn.Module` here
while using `GraphProcessor` in `EncodeProcessDecode` is inconsistent.

---

## 12. LOW: missing protocols for other pluggable extension points

`Aggregation` (in `core/aggregation.py`) is a pluggable strategy consumed
by message-passing blocks but has no Protocol — it is an abstract base
class instead. `ProbeGraphBuilder` is also injectable in principle but has
no protocol. If the codebase is committed to protocols as the primary
extension mechanism, these gaps should be filled (or the project should
admit that ABCs are fine and stop using Protocols at all — see §5).

---

## 13. LOW: protocols do not encode tensor shapes

None of the protocols carry dimension information. A `GraphProcessor` that
outputs 64-dim features fed into a `GraphProcessor` that expects 128-dim
input satisfies the protocol perfectly and explodes at runtime. This is a
fundamental limitation of Python's type system, but worth stating plainly
in the protocols' docstrings so users do not expect what is not there.

One occasionally-suggested remedy is `Protocol[NodeT, EdgeT]` generics.
That idea is currently unimplemented, and would not actually solve the
shape problem even if added — it would just push it one level deeper,
because `NodeT`/`EdgeT` would have to be bound to nominal marker types
that still say nothing about tensor extent. A more honest fix is a
one-time `latent_dim` assertion at EPD construction time, similar to what
`validate_edge_message_processor` already does for edge processors.

---

## Recommended next steps (in order)

1. **Fix the `isinstance(decoder, QueryDecoder)` dispatch in
   `EncodeProcessDecode.forward` today.** It is the only finding that can
   produce a user-visible crash with a single-line change to a decoder
   signature. (§1)
2. **Decide between "real Protocols with distinct method names" vs.
   "collapse the indistinguishable protocols to a single `GraphStage`
   type."** Pick one and apply it consistently to §2 and §3.
3. **Move `ConditioningProtocol` out of `protocols.py`** so the file's
   stated design rule ("structural typing, no inheritance") matches its
   contents. (§5)
4. **Add `tests/test_protocol_conformance.py`** covering both positive and
   negative isinstance assertions. (§10)
5. **Drop `Decoder = Union[NodeDecoder, QueryDecoder]`** once the EPD
   dispatch is fixed. (§4)
6. **Resolve the Protocol-vs-ABC duplication for `EdgeMessageProcessor`
   and `NodeUpdateStrategy`** by picking one enforcement mechanism.
   (§6)
