# Generation Selection Tuning

## Scope

The variant pipeline turns TianChen-RV execution envelopes and plugin-owned
extension bodies into selected executable paths. It does not create a generic
high-level compute IR and does not treat metadata as route authority.

Current RVV path:

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin legality
  -> optional RVV selected-body realization
  -> RVV plugin route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC
```

## Current Inputs

Current Stage1/Stage2 plugin work may start from:

- hand-written or generated TianChen-RV MLIR;
- `tcrv.exec` envelope and selected variants;
- typed extension-family bodies such as `tcrv_rvv`;
- selected boundaries that the origin plugin legally consumes;
- structured capability/profile facts;
- runtime SSA / ABI declarations.

Selected-path metadata is diagnostic/control mirror only. It may explain why a
variant was chosen, but it cannot be compute input, dtype authority, route
authority, body authority, or artifact authority.

High-level MLIR frontend lowering is a future integration path unless an
explicit task selects it after RVV maturity.

## Stage Gates

### Stage 1: RVV Route-Authority Reset

Replace or fail-close active paths that treat the following as RVV route
authority:

```text
bounded i32m1 arithmetic
RVVI32M1* route specs/slices
finite tcrv_rvv.i32_* ops
!tcrv_rvv.i32m1 helper types
rvv-i32m1 route ids
artifact names
source-front-door/source-artifact patterns
descriptor residue
exact __riscv_*_i32m1 spellings
```

Do not preserve an executable legacy i32 compatibility route.

### Stage 2: Typed RVV Coverage And Performance Realization

Expand route-supported coverage on corrected typed `tcrv_rvv` bodies.
Structured computation classes calibrate coverage:

```text
elementwise
broadcast
reduction / accumulation
contraction-like accumulation
movement/layout
dtype conversion
mask/tail
runtime shape/control
```

Stage 2 also includes selected-body realization:

```text
selected pre-realized tcrv_rvv body
  + target capability
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> realized tcrv_rvv selected body
```

Hints/config/profile are not final products. If they affect generated code,
they must be consumed into body structure before route construction.

After a route family is production-validated, Stage 2 progress must not degrade
into an unbounded sequence of artifact ABI evidence closeouts. A generated-bundle
or `ssh rvv` proof is useful when emitted code, ABI/header behavior, runtime
correctness, or performance is newly claimed. It is not a substitute for
advancing typed primitive coverage, selected-body realization, resource-aware
tuning, or a measured performance comparison path.

### Stage 3: Extension Generalization

IME, Offload, TensorExtLite, Template/Toy source-front-door examples, and
future plugin positives are Stage3/later unless an explicit task opens them.

## Variant Required Fields

Each executable variant must have:

- origin plugin;
- structured `requires`;
- typed extension-family body or selected boundary;
- plugin legality result;
- ABI role declarations and explicit body imports/consumption;
- optional cost/tuning hints as realization inputs;
- route-provider output when emission is supported;
- fallback/dispatch relation where needed.

Shape/dtype/layout preconditions and cost/tuning facts may be mirrored in
metadata, but executable RVV dtype/config/operation facts must be structural in
typed `tcrv_rvv` body or consumed into realized body. Emission path metadata is
not route support.

## Selection

Selection may choose:

- one static selected variant;
- runtime dispatch among guarded cases;
- conservative fallback;
- unsupported diagnostic when no legal executable route exists.

Selection must not:

- infer compute from `tcrv.exec`;
- infer dtype from ABI strings or parameter names;
- select routes by artifact name;
- rank source-front-door metadata as an executable route;
- treat readiness/status dashboards as progress.

## Tuning

Tuning is capability-aware and variant-local. It may propose hints/policy, but
code-affecting choices must be realized into body structure.

For RVV, realization may materialize:

```text
setvl/VL placement
SEW/LMUL/policy
memory form
mask/tail behavior
unroll/prefetch structure
accumulator/reduction layout
```

### Resource-Aware RVV Closure

For performance-sensitive RVV Stage 2 families, especially low-precision
direct-contraction paths, tuning must be resource-aware before it is described as
autotuning or performance maturity. The resource model may start static and
bounded, but it must reason over compiler-visible body facts such as:

```text
target VLEN / ELEN / vector register budget
reserved mask / v0 usage
SEW / LMUL / EMUL
widening or narrowing pressure
peak live vector groups
load/store and mask live ranges
accumulator count and reduction layout
vsetvl region count
memory form / stride
tail and mask policy
```

A selector may keep schedule facts as owner-local metadata only when the facts
are consumed by RVV selected-body realization, provider planning, or target
artifact validation. If a choice changes generated code, it must be realized into
`tcrv_rvv` structure or a provider-consumed owner-local plan before
`TCRVEmitCLowerableRoute` construction.

The current bounded Gearbox schedule materialization style is an MVP unless it
enumerates and prunes candidates with such resource facts. A fixed static
candidate, fixed unroll, or fixed LMUL schedule is not a completed
resource-aware autotuning pass.

llama.cpp q8/q4 examples are representative pressure tests for this contract.
They must be treated as low-precision direct-contraction maturity signals, not
as q8-named route authority, benchmark-name authority, or permission to bypass
typed `tcrv_rvv` body/provider facts.

For future IME/Offload/family work, analogous tuning remains plugin-local and
Stage3-gated unless explicitly selected.

## Emission-Plan Mirrors

Emission-plan diagnostics are optional mirrors after plugin route construction.
They are useful for reproducibility and failure reporting, but they are not:

- route authority;
- dtype/config authority;
- selected body;
- progress status;
- target artifact authority;
- runtime/correctness/performance evidence.

## Good / Bad Cases

Good:

```text
typed tcrv_rvv body
  -> RVV legality
  -> selected-body realization consumes hints
  -> provider-built route
  -> common EmitC
```

Bad:

```text
variant metadata says dtype=i32, route=rvv-i32m1-add
  -> emit object
```

Bad:

```text
source-front-door marker
  -> selected RVV artifact during Stage 1
```

Bad:

```text
emission_plan status
  -> readiness/progress state
```

## Tests Required

Pipeline changes require focused tests for:

- selected body presence and legality;
- fail-closed legacy i32 route-table and source-front-door cases;
- selected-body realization consuming code-affecting hints/config;
- route provider output before common EmitC;
- metadata-only paths failing closed;
- dispatch/fallback coherence.

Runtime/correctness/performance claims require the corresponding hardware or
runtime evidence; for RVV that means real `ssh rvv` evidence.
