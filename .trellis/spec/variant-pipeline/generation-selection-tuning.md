# Generation, Selection, And Tuning

## Core Principle

Do not generate generic `tcrv` compute ops first. Extension plugins propose execution variants directly, and core passes organize, verify, select, dispatch, and lower them.

Correct:

```text
high-level op -> plugin-proposed execution variants -> capability-driven selection/lowering
```

Wrong:

```text
high-level op -> generic tcrv op -> backend-specific lowering
```

## Input

Inputs may include:

```text
linalg.matmul
linalg.generic reduction
stablehlo dot_general
stablehlo softmax-like region
tosa ops
custom high-level kernel dialect
```

TianChen-RV assumes the high-level op already expresses computation semantics. It does not recover or reinvent algorithm semantics.

## Pipeline

```text
High-level MLIR op
    |
    v
Collect target capabilities
    |
    v
Query extension plugins
    |
    v
Plugins propose execution variants
    |
    v
Capability and legality verification
    |
    v
Variant-local tuning and cost estimation
    |
    v
Select variant or generate dispatch
    |
    v
Lower selected variants through plugin emission paths
```

## Plugin-Driven Proposal

For each high-level op, core asks enabled plugins:

```text
RVV plugin: can it generate an RVV variant?
IME plugin: can it generate an IME variant?
Offload plugin: can it generate an offload variant?
Scalar/default plugin: can it generate fallback?
Future plugin: can it generate another variant?
```

The first C++ interface slice may collect proposal metadata objects before
materializing textual IR. Each proposal must still be directly mappable to
`tcrv.exec.variant` and dispatch metadata: variant symbol/name, origin plugin,
required capability ids or symbol references, and optional generic
condition/guard/policy metadata. A proposal may also carry plugin-owned MLIR
attributes as a generic named-attribute bag, but those names must be
dialect-qualified/discardable and must not collide with required
`tcrv.exec.variant` attributes. Proposal collection is orchestration only; it
does not select variants, run tuning, lower extension ops, emit runtime glue, or
rewrite IR. Registry orchestration must still validate that any proposal
required capability ids or symbol references are non-empty, known to the request
capability set, and available before the proposal can proceed to later legality
or materialization slices.

## Generic Proposal Materialization

After proposal collection and generic capability validation, a core C++/MLIR
materialization helper may turn validated `VariantProposal` metadata into real
`tcrv.exec.variant` operations under the relevant `tcrv.exec.kernel`.

The materialization slice is intentionally bounded:

- it uses MLIR builders and ODS-generated `tcrv.exec` op APIs, not textual
  string rewriting;
- it preserves proposal order when creating variants;
- it maps required capability ids through the generic `TargetCapabilitySet`
  into `FlatSymbolRefAttr` references to `tcrv.exec.capability` symbols;
- it preserves required capability symbol references after checking they resolve
  in the kernel capability scope;
- it attaches only generic metadata supported by the current `tcrv.exec` ODS
  contract, such as variant symbol/name, origin plugin, `requires`, and
  non-empty generic `condition`/`guard`/`policy` decision metadata;
- it preserves plugin-owned dialect-qualified named attributes from the proposal
  onto the materialized `tcrv.exec.variant` as opaque MLIR attributes, after
  generic validation and without interpreting target-family semantics;
- it diagnoses malformed input before mutating IR when practical, including
  missing kernel anchors, duplicate variant symbols, invalid variant names,
  empty origins, unresolved capability ids or symbols, duplicate plugin-owned
  attributes, malformed plugin-owned attribute names, and collisions with core
  variant attributes.

This helper does not select variants, build dispatch, run tuning, lower
extension dialects, emit runtime glue, or implement extension-specific
semantics. Core materialization must remain free of target-family branches.

## Generic Dispatch Synthesis

After variants are already materialized, a bounded core C++/MLIR dispatch
synthesis helper may organize sibling `tcrv.exec.variant` operations into a
single `tcrv.exec.dispatch`. This slice consumes real MLIR ops and the generic
`TargetCapabilitySet`; it does not collect plugin proposals, materialize new
variants, tune, lower, emit, probe hardware, or interpret target-family
semantics.

The deterministic first-slice policy is:

- scan direct `tcrv.exec.variant` children of each `tcrv.exec.kernel` in IR
  order;
- leave kernels with an existing direct `tcrv.exec.dispatch` unchanged rather
  than creating competing dispatch ops;
- choose the first direct variant whose `requires` are generically available as
  the `tcrv.exec.fallback` target;
- emit `tcrv.exec.case` entries for the remaining variants in original variant
  order;
- inherit any present generic `condition`, `guard`, or `policy` metadata from
  the target variant onto the generated `tcrv.exec.case`;
- attach a synthesized non-empty generic guard policy only when a case target has
  unavailable required capabilities and no inherited generic decision metadata,
  so `--tcrv-check-capability-requires` can treat it as runtime-guarded without
  overwriting plugin-proposed metadata;
- diagnose and leave IR unmodified when no direct variant is generically
  available as an executable fallback.

Dispatch synthesis must stay target-neutral. It must not branch on RVV, IME,
offload, scalar fallback, vendors, accelerators, or any other target family.

Example without IME:

```text
linalg.matmul
  -> @rvv variant
  -> @sophgo_offload variant
  -> @fallback variant
```

Example with IME available:

```text
linalg.matmul
  -> @rvv variant
  -> @ime variant
  -> @sophgo_offload variant
  -> @fallback variant
```

## Generic Materialized Variant Legality Routing

After variants are materialized and before cost/tuning/lowering/emission, the
core registry may orchestrate plugin-local legality verification for real
`tcrv.exec.variant` IR.

The first C++/MLIR legality routing contract is:

- build or receive the generic `TargetCapabilitySet` from the enclosing
  `tcrv.exec.kernel`;
- create a `VariantLegalityRequest` containing the materialized
  `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and that capability
  set;
- route the request only to the plugin named by the variant `origin` attribute;
- reject missing variants, missing kernels, missing or empty origins, unknown
  origin plugins, and disabled origin plugins with generic diagnostics;
- propagate plugin-local legality failures while wrapping them with plugin,
  variant, and kernel context;
- verify direct variants in kernel IR order when checking a whole kernel;
- leave proposal collection, materialization, dispatch synthesis, selection,
  tuning, lowering, and emission as separate pipeline responsibilities.

This routing layer complements `--tcrv-check-capability-requires`. The generic
capability pass continues to check availability and dispatch/fallback guarding;
plugin legality checks extension-owned body details through plugin interfaces.
Core legality orchestration must not hard-code RVV, IME, offload, scalar
fallback, vendor, dtype, shape, layout, or target-family semantics.

## Generic Materialized Variant Cost Routing

After variants are materialized and legality can be routed plugin-locally, the
core registry may also orchestrate plugin-local cost estimation for real
`tcrv.exec.variant` IR. This is an input to later selection, not selection
itself.

The first C++/MLIR cost routing contract is:

- build or receive the generic `TargetCapabilitySet` from the enclosing
  `tcrv.exec.kernel`;
- create a `VariantCostRequest` containing the materialized
  `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and that capability
  set;
- route the request only to the plugin named by the variant `origin` attribute;
- reject missing variants, missing kernels, variants not enclosed by the
  requested kernel, missing or empty origins, unknown origin plugins, and
  disabled origin plugins with generic diagnostics;
- propagate plugin-local cost failures while wrapping them with plugin,
  variant, and kernel context;
- reject invalid estimates, including missing scores, non-finite scores,
  negative scores, mismatched origin/variant identity, and present-but-empty
  generic explanation or policy text;
- collect direct kernel variant costs in original IR order without mutating IR;
- return deterministic rankings by ascending generic score, with equal-score
  ties kept stable by original kernel IR order;
- leave proposal collection, materialization, legality, dispatch synthesis,
  full selection, tuning, lowering, emission, runtime ABI, and hardware
  probing as separate pipeline responsibilities.

The default plugin cost hook is deterministic and safe: it returns a neutral
finite score for the request variant and origin plugin. Concrete analytic,
empirical, profile-guided, shape-aware, dtype-aware, runtime-aware, or
hardware-specific cost semantics belong inside plugins and later selection
slices, not in core registry orchestration.

Core cost orchestration must not hard-code RVV, IME, offload, scalar fallback,
vendor, dtype, shape, layout, runtime ABI, microarchitecture, or target-family
semantics.

## Generic Cost-Aware Selection Planning

### 1. Scope / Trigger

After materialized variant costs are ranked, a bounded core C++/MLIR selection
planner may turn direct `tcrv.exec.variant` children into an explicit compiler
decision plan. This planner consumes real MLIR ops, a `tcrv.exec.kernel`, the
generic `TargetCapabilitySet`, and `ExtensionPluginRegistry` ranking output. It
does not collect proposals, materialize variants, verify plugin-local legality,
lower extension dialects, emit runtime glue, run hardware probes, or implement
target-specific selection semantics.

### 2. Signatures

The durable C++ API shape is:

```cpp
enum class VariantSelectionKind {
  StaticVariant,
  RuntimeDispatch,
  FallbackOnly,
  NoViableVariant,
};

struct VariantSelectionCase {
  tcrv::exec::VariantOp variant;
  plugin::VariantCostEstimate cost;
  std::size_t originalIndex;
  bool genericallyAvailable;
  bool hasGenericDecisionMetadata;
};

struct VariantSelectionPlan {
  VariantSelectionKind kind;
  tcrv::exec::KernelOp kernel;
  tcrv::exec::VariantOp selectedVariant;
  tcrv::exec::VariantOp fallback;
  SmallVector<VariantSelectionCase> dispatchCases;
  SmallVector<VariantSelectionCase> rankedVariants;
};

Expected<VariantSelectionPlan> planKernelVariantSelection(
    tcrv::exec::KernelOp kernel,
    const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry);

Error materializeRuntimeDispatchPlan(OpBuilder &builder,
                                     const VariantSelectionPlan &plan,
                                     tcrv::exec::DispatchOp *createdDispatch);

Error materializeSelectedVariantMarker(
    OpBuilder &builder,
    const VariantSelectionPlan &plan,
    tcrv::exec::DiagnosticOp *createdMarker);
```

### 3. Contracts

The generic selection-planning contract is:

- collect only direct `tcrv.exec.variant` children of the request
  `tcrv.exec.kernel`;
- obtain score order through `ExtensionPluginRegistry::rankKernelVariantsByCost`
  and preserve its stable original-IR-order tie break;
- require every selected or dispatched variant to have structured generic
  `requires` metadata and origin-owned cost information;
- treat a variant as generically available only when all required capability
  symbols resolve in the supplied `TargetCapabilitySet` and are available;
- reject unavailable variants that have no non-empty generic `condition`,
  `guard`, or `policy` metadata, rather than silently selecting them;
- choose the best generically available fallback by cost ranking, not by old
  kernel IR order;
- produce an explicit no-variant plan when a kernel has no direct variants;
- diagnose runtime-dispatch situations that have guarded candidates but no
  generically available fallback;
- produce a static plan when the lowest ranked executable path is available and
  no lower-cost guarded candidate must be retained;
- produce a runtime dispatch plan when lower-cost guarded candidates must be
  retained and a generically available fallback exists.

### 4. Validation & Error Matrix

- missing kernel or missing kernel body -> return an `llvm::Error`;
- no direct variants -> return a `NoViableVariant` plan, not a crash;
- cost ranking failure from unknown origin, disabled plugin, plugin-local
  failure, or invalid estimate -> propagate the registry error with plugin,
  variant, and kernel context;
- selected/dispatched variant without structured `requires` metadata -> return
  an `llvm::Error`;
- unavailable variant without generic decision metadata -> return an
  `llvm::Error`;
- runtime-dispatch plan without a generically available fallback -> return an
  `llvm::Error`;
- dispatch materialization for a non-runtime-dispatch plan -> return an
  `llvm::Error`;
- dispatch materialization when the kernel already contains a direct
  `tcrv.exec.dispatch` -> return an `llvm::Error` and leave IR unchanged;
- dispatch materialization with case/fallback variants not directly enclosed by
  the plan kernel -> return an `llvm::Error`.
- selected-path marker materialization for a non-`StaticVariant` /
  non-`FallbackOnly` plan -> return an `llvm::Error`;
- selected-path marker materialization when the kernel already contains a direct
  `tcrv.exec.dispatch` or a different direct selected marker -> return an
  `llvm::Error`.

### 5. Good / Base / Bad Cases

- Good: lower-cost guarded variants carry non-empty generic metadata, a
  generically available fallback exists, and materialization creates ordered
  `tcrv.exec.case` ops plus one `tcrv.exec.fallback`.
- Base: all candidates are generically available and unguarded; the planner
  chooses the lowest-cost static variant and preserves stable equal-score order.
- Bad: an unavailable unguarded variant appears in the ranked set; the planner
  rejects it instead of silently selecting or dispatching it.

### 6. Tests Required

- C++ tests must parse or build real MLIR modules with `tcrv.exec.kernel`,
  `tcrv.exec.capability`, and materialized `tcrv.exec.variant` ops.
- Mock plugins must provide costs through the real
  `ExtensionPluginRegistry::rankKernelVariantsByCost` path.
- Positive tests must cover static lowest-cost selection, equal-cost stable
  ties, guarded runtime dispatch, fallback-by-cost rather than IR order, typed
  dispatch materialization, metadata copying, and verifier/capability-check
  acceptance.
- Negative tests must cover no direct variants, no available fallback,
  unavailable unguarded variants, existing direct dispatch rejection, ranking
  failure propagation, missing kernels, and cross-kernel case/fallback variants.

### 7. Wrong vs Correct

Wrong:

```cpp
if (target.hasRVV()) choose(rvvVariant);
else if (target.hasIME()) choose(imeVariant);
```

Correct:

```cpp
Expected<VariantSelectionPlan> plan =
    planKernelVariantSelection(kernel, capabilities, registry);
```

A selection materialization helper may turn a runtime dispatch plan into typed
`tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback` operations. The
helper preserves planned case order, copies non-empty generic
`condition`/`guard`/`policy` metadata verbatim, rejects kernels that already
contain a direct dispatch, and does not erase variants. It must not parse,
switch on, or infer target-family meaning from generic metadata strings.

Static and fallback-only selection plans may instead materialize one direct
`tcrv.exec.diagnostic` selected-path marker:

```mlir
tcrv.exec.diagnostic {
  reason = "variant-selected",
  message = "static variant selected by generic cost and capability planning",
  severity = "note",
  status = "selected",
  target = @selected_variant,
  selection_kind = "static-variant"
}
```

The marker is generic control metadata. `target` references the selected direct
variant, and `selection_kind` is `static-variant` or `fallback-only` in the
first slice. It is not lowering IR, runtime ABI glue, or target-family logic.
Re-running selection must reuse an equivalent direct marker rather than
duplicating it.

Selection planning must stay target-neutral. It must not branch on RVV, IME,
offload, scalar fallback, vendors, accelerators, dtype, shape, layout, runtime
ABI, microarchitecture, or any other target family.

### 8. Pass Integration and Registry Injection

The generic selection planner may be exposed through an MLIR module pass, but the
pass boundary must preserve plugin-owned cost semantics:

- `createSelectVariantsPass(const ExtensionPluginRegistry &registry)` is the
  production construction path. Tooling or future plugin loaders inject the
  populated registry before pass execution.
- `createSelectVariantsPass()` may be used for public `tcrv-opt` registration,
  but it owns only an empty registry. It must diagnose variants with
  unregistered `origin` plugins instead of inventing core-side attribute costs,
  target-family fallbacks, or Python-only ranking.
- The pass must build `TargetCapabilitySet` from each `tcrv.exec.kernel`, call
  the existing selection planner, materialize runtime-dispatch plans with typed
  `tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback`, and
  materialize static/fallback-only plans with one generic selected-path
  `tcrv.exec.diagnostic` marker.
- Static, fallback-only, and no-direct-variant plans do not erase variants,
  lower extension dialects, or inject target-specific IR.
- Tests should cover both injected-registry pass execution and public default
  pass diagnostics for missing origin plugins.

## Variant IR Required Fields

Each variant must include:

```text
variant name
origin plugin
required capabilities
shape/dtype/layout preconditions
extension dialect ops
cost/tuning metadata
emission path
fallback or dispatch relation
```

Reference shape:

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin"
  tuning = #tcrv.tuning<...>
  cost = #tcrv.cost<...> {
  ... tcrv.rvv ops ...
}
```

## Legality

### Core legality

Core checks:

- variant declares `requires`;
- `requires` is satisfied or guarded by runtime dispatch;
- variant has origin plugin;
- body does not contain unknown/unregistered extension op;
- dispatch/fallback is complete.

### Plugin legality

Plugins check extension-specific details:

- RVV: SEW/LMUL/VL/mask/tail/memory/toolchain legality;
- IME: VLEN-backed fragment shape, dtype, accumulator, layout, toolchain;
- offload: runtime ABI, transfer, shape, sync, fallback;
- future plugin: its own extension rules.

Core verifier orchestrates plugin verifiers. It must not hard-code extension internals.

## Selection

Selection chooses:

```text
single static variant
runtime dispatch set
fallback-only path
```

Inputs:

```text
target capability
shape/dtype/layout
cost model
profile data
runtime availability
tuning result
user policy: performance / portability / debug / no-offload
```

## Runtime Dispatch

Dispatch conditions come from capability object, runtime probe, shape guard, dtype guard, cost threshold, or user policy.

Reference:

```mlir
tcrv.exec.dispatch {
  tcrv.exec.case @sophgo_offload {
    condition = "runtime_available",
    guard = "large_shape",
    policy = "prefer_offload"
  }
  tcrv.exec.case @ime {condition = "extension_capability_available"}
  tcrv.exec.case @rvv {condition = "vector_capability_available"}
  tcrv.exec.fallback @scalar_or_default
}
```

The structured `tcrv.exec.case` form is a compiler-visible reference to a
declared variant. The core verifier checks dispatch/fallback completeness and
symbol resolution only; it does not implement selection, cost modeling, tuning,
or extension-specific condition semantics.

Capability legality may treat a `tcrv.exec.case` as a runtime guard for an
unavailable required capability only when the case carries at least one
non-empty generic `condition`, `guard`, or `policy` attribute. The core pass
does not parse those strings and must not switch on target families. A
`tcrv.exec.fallback` target is different: it must be generically available under
the same target capability set, so runtime dispatch always has an executable
fallback path when guarded cases are unavailable.

## Emission Readiness After Selection

After materialization, legality, cost ranking, selection, and dispatch
synthesis, the compiler may run a generic emission-readiness check before final
lowering. This pass does not create lowering IR or runtime glue. It verifies
that the current execution path can be routed back to the origin plugin that
owns each selected materialized variant.

First-slice behavior:

- kernels with a direct `tcrv.exec.dispatch` check every `tcrv.exec.case`
  target and the `tcrv.exec.fallback` target;
- kernels without a dispatch but with one direct selected-path
  `tcrv.exec.diagnostic` marker check only that marker target;
- kernels without a dispatch or selected-marker surface conservatively check all
  direct `tcrv.exec.variant` children;
- selected-path marker targets must resolve to direct sibling variants in the
  same kernel before plugin routing, and duplicate direct selected markers are
  invalid;
- dispatch and fallback targets must resolve to direct sibling variants in the
  same kernel before plugin routing;
- missing, duplicate, non-variant, or non-sibling dispatch references are
  diagnosed generically before plugin emission-readiness hooks are called;
- routing is by the generic variant `origin` attribute through
  `ExtensionPluginRegistry`, with no target-family branches in core code.

The same selected-path traversal may be reused to collect plugin-owned emission
plans after readiness. Emission-plan collection must keep dispatch cases and
fallbacks in dispatch body order, consume a single selected-path marker before
falling back to conservative all-direct-variant traversal, and diagnose the
same malformed selected-path structure before invoking plugin hooks. Each
collected plan is a structured compiler decision object for the selected path:
it records origin plugin, kernel symbol, variant symbol, selected-path role,
support status, and either supported route metadata or an unsupported
diagnostic. It does not lower IR, generate executable code, prove runtime
correctness, or justify performance claims.

## Capability-Aware Tuning

Tuning has two layers.

Variant selection tuning:

```text
choose RVV vs IME vs offload vs fallback
```

Variant-local tuning:

```text
RVV: LMUL, SEW, VL policy, unroll, packing, thread partition
IME: fragment shape, K-block, accumulator policy, packing
Offload: transfer threshold, batch size, async overlap, buffer reuse
```

Tuning must remain part of selection/variant quality. Do not present ordinary parameter search as the main theory.

## Cost Model Boundary

Cost model may be analytic, empirical, profile-guided, or hybrid. It must explain:

- why RVV is selected;
- why IME is selected;
- why offload is selected;
- why dispatch is retained;
- why fallback is used.

## Diagnostics

Structured diagnostics should cover:

- no plugin supports the high-level op;
- plugin supports op but capability is absent;
- plugin proposes a variant with an empty, unknown, or unavailable required
  capability id or symbol reference;
- plugin generated variant but legality failed;
- variant legal but emission path missing;
- multiple variants have close cost and dispatch is retained;
- offload runtime unavailable and RVV fallback selected.
