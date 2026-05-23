# MLIR Testing Contract

## Scope

Testing must validate real TianChen-RV compiler behavior: dialect syntax,
verification, pass behavior, plugin interfaces, route-provider materialization,
common EmitC lowering, target artifact packaging, and runtime evidence when
claimed.

Tests are not dashboards, readiness states, artifact ledgers, or substitute
architecture authority.

## RVV Stage Rules

During RVV Stage 1:

- no positive legacy `RVVI32M1*` route-table tests;
- no positive `rvv-i32m1-*` object/header/bundle tests;
- no positive generated artifacts from `!tcrv_rvv.i32m1` or
  `tcrv_rvv.i32_*` helper-family authority;
- no positive RVV source-front-door/source-artifact bundle route tests;
- negative/fail-closed tests are required for stale legacy authority.

Positive RVV generated artifact tests are allowed only for corrected generic
typed `tcrv_rvv` routes after Stage1 reset:

```text
selected RVV variant
  -> explicit typed vector-level tcrv_rvv body
  -> RVV plugin legality / realization
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> target artifact
```

Use "explicit typed vector-level `tcrv_rvv` body"; do not use stale phrases
such as "explicit microkernel" or "selected RVV capacity metadata" as
authority.

## Required Test Types

### lit / FileCheck

Use for:

- dialect syntax and parsing;
- verifier success/failure;
- pass rewrite behavior;
- fail-closed diagnostics;
- selected-body realization visible in IR;
- route-provider handoff visible through emitted IR/mirrors;
- common EmitC materialization.

### C++ Tests

Use for:

- plugin registry APIs;
- route provider APIs;
- selected-body realization result objects;
- capability helper APIs;
- non-textual compiler utilities.

### Runtime / Hardware Evidence

Runtime, correctness, or performance claims require actual execution evidence.
For RVV claims, use real `ssh rvv` output. Local compile-only, static MLIR
checks, or Python smoke tests are not RVV runtime evidence.

For RVV generated-bundle evidence over runtime `n`, memory-writing routes must
check both active-lane arithmetic/data movement and guard/tail preservation.
Harnesses should initialize output storage beyond `n` (or inactive/passthrough
lanes when applicable) with sentinels and fail if the generated route writes
outside the runtime element count. These sentinel checks are evidence quality
guards only; they do not become route, dtype, or artifact authority.

## Source Front Door Tests

Default source-front-door policy is explicit-only or disabled. During RVV
Stage 1, RVV source-front-door tests must prove fail-closed behavior unless an
explicit future task has enabled a mature corrected typed route.

Toy, Template, TensorExtLite, IME, Offload, and future plugin positive
source-front-door tests are Stage3/later examples and must not be introduced as
current RVV progress.

## Metadata-Only Tests

Metadata-only surfaces may be tested only as mirrors or diagnostics. A test
must not assert that these alone authorize executable output:

```text
emission_plan status/result
route id
artifact kind/name
manifest
semantic role graph
construction template
source-front-door marker
selected-path metadata
descriptor
```

## Good / Bad Cases

Good:

```text
typed tcrv_rvv body -> plugin route provider -> common EmitC
```

Good:

```text
legacy RVVI32M1 route table -> unsupported diagnostic
```

Good:

```text
source-artifact RVV Stage1 marker -> fail-closed diagnostic
```

Bad:

```text
rvv-i32m1 route id -> supported object artifact
```

Bad:

```text
emission_plan status -> readiness/progress acceptance
```

Bad:

```text
test name or artifact name implies dtype/operation
```

## Review Checklist

- [ ] Does each positive test exercise a production compiler path?
- [ ] Does each RVV positive artifact test use corrected typed `tcrv_rvv` body authority?
- [ ] Are legacy i32m1 and source-front-door RVV paths negative/fail-closed?
- [ ] Are C ABI strings, parameter names, route ids, and artifact names treated as mirrors only?
- [ ] Are runtime/correctness/performance claims backed by appropriate run evidence?
- [ ] Is broad smoke/dashboard/report coverage avoided unless it verifies a real touched behavior?
