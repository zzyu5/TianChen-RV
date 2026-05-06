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
condition/guard/policy metadata. Proposal collection is orchestration only; it
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
  contract, such as variant symbol/name, origin plugin, and `requires`;
- it diagnoses malformed input before mutating IR when practical, including
  missing kernel anchors, duplicate variant symbols, invalid variant names,
  empty origins, and unresolved capability ids or symbols.

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
- attach a non-empty generic `condition`, `guard`, or `policy` string to any
  case whose target has unavailable required capabilities, so
  `--tcrv-check-capability-requires` can treat it as runtime-guarded;
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
