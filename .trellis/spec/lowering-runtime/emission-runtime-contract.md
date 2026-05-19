# Emission Runtime Contract

## Scope

This contract defines how selected TianChen-RV variants become emitted code,
runtime ABI surfaces, target artifacts, and evidence. It applies to RVV now and
to IME, Offload, TensorExtLite, Template/Toy, scalar fallback, and future
families only when their stage gate is explicitly open.

Emission is not a metadata state machine. The executable authority chain is:

```text
tcrv.exec envelope
  -> selected extension variant
  -> typed extension-family body or selected boundary
  -> origin plugin legality / optional selected-body realization
  -> origin plugin route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> C/C++ or equivalent backend representation
  -> target artifact
  -> runtime / hardware evidence when claimed
```

For the current RVV mainline, the selected body is a typed low-level
`tcrv_rvv` body and the origin plugin is the RVV plugin.

## Non-Authority Rule

The following surfaces are diagnostic/handoff mirrors only:

```text
emission_plan diagnostics
diagnostic result/status fields
route ids
artifact kind/name metadata
manifests
semantic role graphs
construction templates
source-front-door markers
selected-path metadata
test names
```

They must not define route support, dtype, compute semantics, schedule,
intrinsic spelling, ABI value use, target artifact authority, correctness,
runtime, performance, or progress.

## RVV Emission Authority

RVV target export may consume only a provider-built route over a legal
typed/realized RVV body:

```text
selected RVV variant
  -> typed tcrv_rvv body
  -> RVV plugin legality
  -> optional RVV selected-body realization
  -> RVV plugin-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target export packaging
```

RVV dtype/config/operation facts must be structural in the typed
`tcrv_rvv` body or consumed into the realized body before route construction.
They must not be recovered from `RVVI32M1*` metadata, `rvv-i32m1` route ids,
`tcrv_rvv.i32_*` helper names, C ABI spellings, exact `__riscv_*_i32m1`
spellings, source-front-door patterns, descriptor residue, artifact names, or
test names.

## Legacy i32m1 Route-Table Policy

There is no supported explicit typed RVV `i32m1` legacy route-table object
plan. All legacy `i32m1` route-table cases must be unsupported/fail-closed
unless already rewritten onto the corrected generic typed `tcrv_rvv` surface.

Forbidden current-positive outputs include:

```text
rvv-i32m1-* materialized EmitC object/header/bundle plans
RVVI32M1* route tables as executable compatibility route
artifact kind: riscv-elf-relocatable-object for legacy i32m1 route-table cases
positive target export from !tcrv_rvv.i32m1 or tcrv_rvv.i32_* helper families
```

Legacy terms may remain only in clearly labeled fail-closed inventory,
negative tests, parser/verifier fixtures without artifact generation, or
migration notes.

## Emission Diagnostics

`tcrv.exec.diagnostic {reason = "emission_plan"}` may be emitted as an optional
mirror after plugin route/provider decisions. It is not a required route input.

Allowed purpose:

- explain why a selected path is unsupported;
- mirror the provider-built route and materialized EmitC handoff;
- carry deterministic provenance for reproducibility;
- help target export report failure reasons.

Forbidden purpose:

- deciding selected route;
- deciding dtype/config/body shape;
- proving executable generation;
- acting as readiness/progress state;
- substituting for plugin legality or route provider output;
- authorizing target artifacts without materialized EmitC route.

If a diagnostic mirror exists, fields such as `result`, `origin`, `role`,
`lowering_boundary`, `runtime_abi`, `required_capabilities`, `artifact_kind`,
and `lowering_pipeline` must exactly mirror already-made plugin decisions.

## Target Artifact Export

Target artifact export consumes materialized EmitC produced from a
`TCRVEmitCLowerableRoute`. Metadata is an exact mirror of that route and the
selected variant.

Export must fail closed when:

- no selected extension variant exists;
- no typed/selected extension body exists;
- plugin legality rejects the body;
- route provider does not return `TCRVEmitCLowerableRoute`;
- common EmitC materialization fails;
- only emission-plan metadata, selected-path metadata, route ids, or manifests
  exist;
- legacy RVV i32m1 route-table data is the only route authority;
- source-front-door/source-artifact markers are the only input during RVV
  Stage 1.

Target export must not synthesize C bodies, headers, objects, bundles, runtime
ABI, dispatch source, or self-checks from descriptors, route ids, selected
metadata, artifact names, or family records.

## Runtime ABI Boundary

`tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI roles and C
export spelling. The selected extension body must explicitly import and consume
those values.

Correct:

```text
mem_window/runtime_param declare lhs/rhs/out/n roles
-> tcrv_rvv body imports those ABI values
-> typed loads/stores/control/dataflow consume them
-> RVV provider maps body + ABI roles to emitted C/C++
```

Wrong:

```text
c_name/c_type/abi_role/artifact name -> infer dtype/operation/route
```

## Source Front Doors

Source-front-door and source-artifact bundle flows are future/Stage3 or
explicit opt-in flows. They are disabled by default for current RVV Stage 1.

During Stage 1, source-only RVV inputs must fail closed unless they already
materialize a corrected typed `tcrv_rvv` body through an explicit opt-in path
after the mature route exists. Positive RVV generated artifact tests must not
start from source-front-door metadata.

Template, Toy, TensorExtLite, IME, Offload, and future plugin source-front-door
examples are Stage3/later examples. They must not become the current source or
route authority before RVV maturity.

## Family-Specific Stage Gates

- RVV: active Stage1/Stage2 path, typed body and plugin route provider
  required.
- Scalar fallback: no active executable scalar body; unsupported diagnostics
  only until later rebuild.
- IME: Stage3/later after RVV maturity and real hardware/toolchain evidence.
- Offload: Stage3/later; current no-active-route validation must not export
  artifacts.
- TensorExtLite / Template / Toy: Stage3/later examples only unless explicitly
  promoted after RVV maturity.
- Future plugins: evidence scenarios only until admitted by explicit spec/task.

## Good / Bad Cases

Good:

```text
typed tcrv_rvv body
  -> RVV legality
  -> realized body if needed
  -> provider-built route
  -> common EmitC
  -> target artifact mirror
```

Good:

```text
legacy RVVI32M1 route-table input
  -> unsupported diagnostic
  -> no EmitC object/header/bundle
```

Bad:

```text
emission_plan metadata + route id
  -> target object
```

Bad:

```text
source-artifact marker
  -> positive RVV Stage1 artifact
```

Bad:

```text
manifest / semantic role graph / construction template
  -> executable route authority
```

## Required Tests

Production-path changes require tests attached to the changed compiler behavior:

- MLIR/lit/FileCheck for typed body legality, route-provider consumption, and
  fail-closed diagnostics;
- C++ tests for route provider / registry APIs where textual MLIR is not
  sufficient;
- negative coverage for legacy `RVVI32M1*`, `rvv-i32m1`, source-front-door, and
  metadata-only target export paths;
- `ssh rvv` run output for runtime/correctness/performance claims.

Broad dashboard, report, ledger, manifest-only, or test-only changes are not
emission/runtime progress unless a production compiler path changed.
