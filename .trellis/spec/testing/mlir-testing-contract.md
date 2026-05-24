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

For positive selected-body target artifact fixtures, keep kernel and variant
symbol names concise. The generated C/EmitC function name is derived from the
kernel and selected variant symbols, and target export must fail closed when
that derived identifier is outside the bounded C/EmitC identifier contract.
Do not treat a shortened fixture symbol as route authority; it is only a
packaging constraint around an already provider-built route.

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

## Direct Pre-Realized Route-Entry Generated-Bundle Evidence

### 1. Scope / Trigger

Use direct pre-realized route-entry generated-bundle evidence only for bounded
RVV pre-realized selected-body families whose production emission-plan entry is
already specified to realize before route facts are collected. It proves:

```text
selected pre-realized tcrv_rvv body
  -> RVV route-entry realization bridge during emission-plan construction
  -> provider-built route
  -> target artifact bundle
  -> external C ABI harness
```

It must not be used to imply direct route-entry support for every pre-realized
family.

### 2. Signatures

The durable evidence command shape is:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --direct-pre-realized-route-entry \
  --op-kind <bounded-supported-kind> \
  [--op-kind <bounded-supported-kind> ...] \
  [--dry-run] \
  --runtime-count <n> --runtime-count <n> \
  [--stride-bytes <bytes> ...]
```

`--direct-pre-realized-route-entry` requires
`--pre-realized-selected-body`. The option means the script must not insert
`--tcrv-materialize-selected-lowering-boundaries`; the production
`--tcrv-materialize-emission-plans` route-entry path must consume the
pre-realized body.

### 3. Contracts

- The selected input remains a pre-realized selected `tcrv.exec` RVV fixture.
- The local bundle generation pipeline must contain
  `--tcrv-materialize-emission-plans` and target artifact bundle export.
- The local bundle generation pipeline must not contain
  `--tcrv-materialize-selected-lowering-boundaries`.
- Evidence must label `materializer` as
  `rvv-route-entry-selected-body-realization`.
- Evidence must set `route_entry_realization` to `true`.
- Materialized body checks must prove the pre-realized op was consumed and the
  realized typed `tcrv_rvv` body remains.
- Bundle checks must validate provider-owned route metadata, runtime ABI order,
  generated header/object paths, and absence of descriptor/direct-C/source
  export residue.
- Runtime/correctness claims still require non-dry-run `ssh rvv` compile/run
  evidence.

### 4. Validation & Error Matrix

- `--direct-pre-realized-route-entry` without
  `--pre-realized-selected-body` -> fail before bundle generation.
- Direct route-entry option with an op kind outside the bounded supported set
  -> fail before bundle generation.
- Materialized output still contains the selected pre-realized body -> fail.
- Pipeline contains `--tcrv-materialize-selected-lowering-boundaries` while in
  direct route-entry mode -> fail.
- Bundle metadata contains descriptor/direct-C/source-export residue -> fail.
- Non-dry-run remote compile or run fails -> report blocked/failed evidence
  with command output; do not claim runtime correctness.

### 5. Good/Base/Bad Cases

- Good: pre-realized compare/select or base-memory route-entry fixture ->
  direct emission-plan route-entry realization -> generated bundle -> ABI
  harness, with optional `ssh rvv` run evidence.
- Base: pre-realized families outside the bounded direct route-entry set keep
  using the explicit selected-boundary materialization mode until a later task
  adds route-entry artifact evidence and tests.
- Bad: all pre-realized selected-body fixtures silently skip the selected
  boundary pass and claim direct route-entry support.
- Bad: script evidence treats the artifact name, route id, status field,
  runtime ABI string, or test name as route support authority.

### 6. Tests Required

- lit/FileCheck for at least one direct pre-realized route-entry generated
  bundle dry-run.
- The test must check `materializer`,
  `route_entry_realization`, materialized body consumption, provider route
  metadata, generated harness invocation, and absence of descriptor/direct-C/
  source-export/source-front-door authority.
- Negative CLI checks must cover missing `--pre-realized-selected-body` and an
  unsupported direct route-entry op kind when the option changes.
- Keep at least one existing pre-realized selected-boundary materializer test
  for families that still require the explicit boundary path.

### 7. Wrong vs Correct

Wrong:

```text
--pre-realized-selected-body for any op
  -> skip selected-boundary pass
  -> infer route support from generated artifact metadata
```

Correct:

```text
--pre-realized-selected-body --direct-pre-realized-route-entry
  -> bounded supported route-entry op
  -> emission-plan route-entry realization
  -> provider-built route and target bundle
```

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
