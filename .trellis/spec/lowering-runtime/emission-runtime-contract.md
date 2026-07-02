# Emission Runtime Contract

## Scope

This contract defines how selected TianChen-RV variants become emitted code,
runtime ABI surfaces, target artifacts, and evidence. It applies identically to
every extension family — RVV, IME, Offload, TensorExtLite, Template/Toy, scalar
fallback, and future families.

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

`tcrv.exec.diagnostic {reason = "emission_plan"}` is emitted as a
non-authoritative mirror after plugin route/provider decisions. It is never a
route INPUT; but when target-artifact export runs, exactly one emission-plan
diagnostic per selected path is a REQUIRED deterministic mirror (见 Candidate
Gate below).

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
- source-front-door/source-artifact markers are the only RVV input
  (these routes fail closed, 见 core-invariants I7).

Target export must not synthesize C bodies, headers, objects, bundles, runtime
ABI, dispatch source, or self-checks from descriptors, route ids, selected
metadata, artifact names, or family records.

## Selected Path Artifact Candidate Gate

### 1. Scope / Trigger

Use this contract when target artifact export consumes selected-path
`tcrv.exec` surfaces and their emission-plan diagnostics. It applies to static
selected variants, selected dispatch cases, dispatch fallbacks, and
fallback-only selections.

### 2. Signatures

The target export collector consumes:

```text
tcrv.exec.dispatch / selected diagnostic marker
emission_plan diagnostic keyed by selected variant symbol + selected path role
status = supported | unsupported
origin
role = direct variant | dispatch case | dispatch fallback
emission_kind
artifact_kind
```

It produces `TargetArtifactCandidate` values only for selected paths whose
provider-built route + materialized EmitC exist; `status = supported` is a
faithful mirror of that route, validated against the real route before packaging
(见 Validation & Error Matrix below), not the gating authority (I4).

### 3. Contracts

- Every selected path must have exactly one emission-plan diagnostic before
  target artifact export.
- A selected kernel must produce at least one supported executable target
  artifact candidate before export can continue.
- Unsupported selected paths are diagnostics only. They must not become target
  candidates, runtime ABI authority, headers, objects, bundles, or executable
  fallback evidence.
- If a selected dispatch case is unsupported and the dispatch fallback is also
  unsupported, export must fail closed at candidate collection and report the
  unsupported selected paths.
- If a selected dispatch case is supported and its scalar fallback is currently
  unsupported, export may continue for the supported selected case only. The
  fallback remains a mirror/diagnostic boundary, not executable fallback
  evidence.

### 4. Validation & Error Matrix

- Missing selected-path emission-plan diagnostic -> fail before target artifact
  candidate construction.
- Selected kernel has no supported candidates -> fail with the selected
  variant symbols, path roles, statuses, origins, emission kinds, and artifact
  kinds for the unsupported paths.
- Unsupported scalar fallback diagnostic attempts to produce a header/object ->
  fail; scalar fallback has no current executable body or route.
- Supported candidate has stale origin, role, runtime ABI, metadata, or
  provider mirrors -> fail in target artifact validation before packaging.

### 5. Good/Base/Bad Cases

- Good: selected RVV dispatch case supported, scalar fallback unsupported ->
  export the RVV artifact; record fallback only as validated mirror metadata.
- Base: fallback-only scalar selection with unsupported scalar emission ->
  export fails with an unsupported selected-path diagnostic.
- Bad: unsupported RVV case plus unsupported scalar fallback -> empty bundle or
  generic "found none" result.

### 6. Tests Required

- Add lit/FileCheck coverage for at least one selected dispatch route where all
  selected paths are unsupported and export rejects the executable artifact
  claim.
- Keep positive selected-case artifact tests proving that an unsupported
  fallback mirror does not block a supported selected case.
- Use C++ target artifact tests when the behavior is not visible through MLIR
  diagnostics.

### 7. Wrong vs Correct

Wrong:

```text
selected dispatch case unsupported
  + scalar fallback unsupported
  -> target export returns no candidates or an empty artifact result
```

Correct:

```text
selected paths all unsupported
  -> target export rejects the kernel
  -> diagnostic lists @variant, role, status, origin, emission_kind, artifact_kind
```

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

Source-front-door and source-artifact bundle flows are disabled by default and
fail closed (见 core-invariants I7); they are not a buildable future deliverable.

Source-only RVV inputs must fail closed unless they already materialize a
corrected typed `tcrv_rvv` body through an explicit opt-in path over an existing
mature route. Positive RVV generated artifact tests must not start from
source-front-door metadata.

Source-front-door examples for any family must not become the current source or
route authority.

## Family Export Gate

A family exports only via a typed extension body + plugin-built
`TCRVEmitCLowerableRoute` + materialized EmitC; absent these it fails closed
(I7). This gate applies identically to every family. RVV is the first and
broadest realized family; scalar fallback has no active executable body or route
(unsupported diagnostics only); any family without a real producer route fails
closed at candidate collection rather than exporting artifacts.

## 发射与构造 / Construction Discipline (Track B)

This umbrella section defines how a selected body becomes emitted code and when
that emission counts as *construction* in the strong sense. It is family-generic
(applies identically to RVV, IME, scalar, and future families) and consumes the
master-doc Track B clauses by ID; it does not restate the six-state ladder
([K-4]) or the pattern registry schema ([PAT-1]) — both are defined in the
objectives 总纲; cite them by ID.

### [K-0] Dual Emission Paths And Convergence

Two emission paths coexist and are both acknowledged (总纲 [B-3]):

```text
(a) monolithic block-quantized operator
      -> flat body emitter (descriptor *selection*)
(b) decomposed N-operand construction path
```

Path (a) selects a descriptor and dispatches to a flat body; it is
**descriptor-selected composition** — constructed-weak ([K-4]), never labeled
`constructed`. Selecting a descriptor is dispatch/wiring, not construction
(调度≠构造, 见 [L-6]①). Path (b) builds the body from operand-level structure.

The terminal state is convergence, not permanent duality: the **generic operand
layer becomes the sole emission authority** ([K-3b]) and both paths unify beneath
it. During the transition both paths may run, but each kernel's path is honestly
labeled by its [K-4] state (constructed-weak vs constructed, per [L-8]); a
flat-path body is never reported as strong construction. Convergence acceptance
criteria are owned by [K-3b] in the objectives 总纲; this contract only fixes the
authority direction (operand layer wins; flat emitter subordinates to it).

### [K-2] Body Pattern Library As Construction Primitives

A strong-sense `constructed` body ([K-4]) is materialized by **referencing
body-pattern-library primitives**, not by calling an opaque handwritten
arithmetic helper. The primitive families are (总纲 [K-2]):

```text
widening dot-product-reduce
dequant LUT / bit-unpack
repack / blocking
reduction tree
masked tail handling
[K-2b] wide-clamp / dual-zone body (widen-clamp-narrow)
```

Each primitive is parameterized by `(SEW, LMUL, VLEN-symbolic)` so that the same
registry entry migrates across boards by key change, not rewrite (迁移性 owned by
[PAT-3]). The pattern-registry data model is owned by [PAT-1] (cite by ID; do not
restate its fields here); this
contract only fixes the **emission-side discipline**: the arithmetic body of a
constructed variant comes from pattern primitives, and any residual handwritten
arithmetic helper downgrades the body to constructed-weak.

### Construction Provenance Manifest

Every mechanism-constructed body emits a provenance manifest
`provenance = [pattern_id, ...]` naming the pattern primitives it was built from.

The manifest is a **mirror** (I4) of a **structural fact** (I5) — that the body
is genuinely assembled from typed pattern-library primitives. It is the
判断依据 for placing a kernel at the `constructed` state ([K-4]); it is never
itself route, dtype, compute, or progress authority, and it never *proves*
construction on its own.

Strong-sense `constructed` ([L-8]/[K-4]) requires the **conjunction**:

```text
provenance manifest exists  ∧  no opaque handwritten arithmetic helper in the body
```

The manifest is deliberately necessary-but-not-sufficient: this is what stops it
being gamed as authority-by-itself. A body carrying a manifest but still routing
its arithmetic through a selected handwritten helper is
**descriptor-selected composition / constructed-weak**, and must be labeled so —
reporting it as strong construction is 违宪 ([L-8]).

### [K-6] Construction Gap-Closure Loop ([GAP-1])

Any performance loss (or unexpected parity — 持平≠击败) against the same-ISA
framework kernel is triaged into exactly one of:

```text
{ 缺能力事实 | 缺模式 | 选择错误 | 物理(带宽墙) }
```

Triage attribution requires disassembly evidence ([L-7]; compile-lit evidence is
not silicon-sealed evidence, 见 [L-6]③). For the two construction-facing verdicts
this layer owns:

```text
缺模式  -> name a new body-pattern-library entry (into [PAT-1] registry)
缺能力事实 -> name a new capability/schema fact
  -> close the gap -> re-emit the constructed body
  -> same-kernel same-board retest
  -> attribution log references the gap-ID
```

`[K-2b]` wide-clamp/dual-zone is the **canonical first gap-closure pattern** (the
exemplar of a named-then-closed construction gap): a clamp regression becomes
"gap named → closed → re-measured vs auto-vectorization." The performance
measurement harness and the three-level attribution log ([D-4]) are owned by the
variant-pipeline / validation layers; this contract only fixes the construction
half — a 缺模式/缺事实 verdict must resolve to a *named registry/schema entry*
that is then reconstructed and retested, not to an opaque one-off helper.

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
constructed body from pattern primitives
  -> provenance = [widening-dot-reduce, dequant-lut, ...]
  -> no opaque arithmetic helper
  -> labeled constructed (strong, [K-4])
```

Bad:

```text
flat body emitter picks a descriptor
  -> body arithmetic is a selected handwritten helper
  -> reported as constructed (strong)   # 违宪 [L-8]; this is constructed-weak
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
  -> positive RVV artifact
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
