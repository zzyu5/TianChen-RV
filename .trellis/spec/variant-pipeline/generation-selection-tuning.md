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
  case @sophgo_offload if #tcrv.cond<"runtime_available && large_shape">
  case @ime            if #tcrv.cond<"ime_available && dtype_supported">
  case @rvv            if #tcrv.cond<"rvv_available">
  fallback @scalar_or_default
}
```

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
- plugin generated variant but legality failed;
- variant legal but emission path missing;
- multiple variants have close cost and dispatch is retained;
- offload runtime unavailable and RVV fallback selected.
