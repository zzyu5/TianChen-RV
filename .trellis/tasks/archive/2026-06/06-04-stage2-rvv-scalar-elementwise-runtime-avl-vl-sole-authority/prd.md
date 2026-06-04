# Stage2 RVV scalar and elementwise runtime AVL/VL sole-authority cleanup

## Goal

Make RVV runtime scalar splat-store and elementwise arithmetic target artifact
validation visibly treat `RVVRuntimeAVLVLSelectedBoundaryContract` as the sole
acceptance authority for runtime `n` / AVL / VL facts. Existing route-local
runtime/control copies may remain only as generated mirrors checked after the
shared selected-boundary contract has matched the rebuilt provider route.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv scalar splat-store or elementwise body
  -> RVV plugin-owned runtime AVL/VL selected-boundary contract
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation consumes runtimeAVLVLContract first
  -> route-local runtime/control mirror consistency checks
  -> route payload/header/type/ABI/statement/mirror validation
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV scalar and elementwise runtime AVL/VL sole-authority cleanup`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `aafe65e3 rvv: make memory runtime AVL VL mirrors non-authoritative`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps route authority in the selected `tcrv.exec`
  envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned legality /
  realization / provider, provider-built `TCRVEmitCLowerableRoute`, neutral
  common EmitC materialization, and target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires target artifact
  validators to dispatch from rebuilt provider descriptions and treat artifact
  metadata, route ids, ABI strings, generated filenames, diagnostics, and
  mirrors as post-route evidence only.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned selected
  boundary authority for runtime AVL source, runtime VL contract, selected
  `with_vl` scope, `setvl`, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI parameter, remaining AVL metadata, pointer advancement,
  bounded slice, and multi-VL facts.
* Archived runtime-scalar splat-store and elementwise arithmetic runtime
  AVL/VL migration tasks already embedded `runtimeAVLVLContract` in
  `RVVRuntimeScalarSplatStoreRouteValidationContract` and
  `RVVElementwiseArithmeticRouteValidationContract`.
* Archived memory-family sole-authority cleanup introduced the current
  target-side pattern: validate `runtimeAVLVLContract` first, then compare
  retained route-local runtime control plan, runtime ABI order, setvl callee,
  VL C type, full-chunk VL, loop VL, and loop induction copies as mirrors only
  through a shared helper.
* Live header inspection shows memory-family contracts already document
  retained runtime/control copies as consistency mirrors, while scalar
  splat-store and elementwise arithmetic contracts still expose
  `runtimeControlPlanID` and `runtimeABIOrder` without the mirror-only comment.
* Live target inspection shows scalar splat-store and elementwise validation
  call `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, but still follow
  with route-local runtime/control comparisons through family-specific
  provider-field checks and diagnostics rather than the shared mirror helper.
* Existing target C++ coverage proves selected-boundary stale runtime facts
  fail closed for scalar splat-store and representative plain/masked/
  scalar-broadcast elementwise routes. This round should add focused
  provider-contract mirror assertions and keep existing fail-closed coverage
  intact.

## Requirements

* Cover exactly two promoted non-memory route families:
  runtime scalar splat-store and elementwise arithmetic, including existing
  plain, masked, scalar-broadcast, and strided elementwise variants already
  represented by the provider contract.
* Keep `RVVRuntimeAVLVLSelectedBoundaryContract` as the first runtime
  acceptance check in both target payload validators.
* Document the retained route-local runtime/control fields in
  `RVVRuntimeScalarSplatStoreRouteValidationContract` and
  `RVVElementwiseArithmeticRouteValidationContract` as target-side consistency
  mirrors. `runtimeAVLVLContract` remains the acceptance authority.
* Replace family-specific direct runtime control / ABI order / setvl / VL name
  comparisons with the shared route-local runtime AVL/VL mirror helper after
  selected-boundary validation.
* Stale, missing, or mismatched route-local runtime/control mirrors must fail
  closed with diagnostics naming route-local runtime AVL/VL mirror mismatch,
  and must not become substitute acceptance authority.
* Preserve scalar splat-store and elementwise-specific validation for
  operation kind, memory/source form, scalar-broadcast, masked/strided facts,
  typed compute op, dtype/config, headers, type mappings, ABI mappings,
  operand bindings, intrinsic leaves, statement-plan shape, and stale
  cross-family mirrors.
* Preserve emitted C/C++, runtime ABI order, runtime counts, statement
  ordering, scalar store behavior, elementwise arithmetic behavior,
  correctness behavior, and performance behavior.
* Keep common EmitC/export neutral. Do not infer runtime control or family
  semantics from route ids, artifact names, manifests, descriptor residue,
  C strings, tests, scripts, status fields, exact intrinsic spelling, or
  metadata mirrors.

## Acceptance Criteria

* [x] Scalar splat-store and elementwise validation contracts document retained
      runtime/control copies as target-side consistency mirrors whose authority
      is `runtimeAVLVLContract`.
* [x] Scalar splat-store target validation consumes
      `runtimeAVLVLContract` before route-local runtime/control mirror checks
      and uses the shared mirror helper for runtime control plan, runtime ABI
      order, setvl callee, VL C type, full-chunk VL, loop VL, and loop
      induction.
* [x] Elementwise target validation consumes `runtimeAVLVLContract` before
      route-local runtime/control mirror checks and uses the shared mirror
      helper for runtime control plan, runtime ABI order, setvl callee, VL C
      type, full-chunk VL, loop VL, and loop induction.
* [x] Focused target C++ coverage asserts scalar splat-store and
      representative plain, masked, and scalar-broadcast elementwise provider
      contracts keep retained route-local runtime/control mirrors equal to the
      embedded selected-boundary contract.
* [x] Existing stale selected-boundary negative coverage for scalar
      splat-store and elementwise runtime AVL/VL facts remains green.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass if provider/target code
      changes.
* [x] Focused scalar splat-store and elementwise lit/generated-bundle dry-run
      checks pass.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      common-EmitC, route-id/artifact-name, exact `__riscv_*_i32m1`, or
      mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Technical Approach

1. Reuse the memory cleanup's shared target helper for route-local runtime
   AVL/VL mirrors instead of adding new family-specific comparison code.
2. Add mirror-only comments in the two validation contract structs.
3. In scalar splat-store payload validation, after shared selected-boundary
   validation, compare retained local runtime/control mirrors against
   `contract.runtimeAVLVLContract`; keep non-runtime family checks unchanged.
4. In elementwise description/payload validation, remove runtime/control local
   direct acceptance from the generic description check and compare retained
   runtime/control mirrors through the shared helper after selected-boundary
   validation; keep operation/config/family checks unchanged.
5. Add focused C++ provider-contract mirror assertions for scalar splat-store
   and representative plain, masked, and scalar-broadcast elementwise routes.
6. Run focused build/test/lit checks, old-authority scan, `git diff --check`,
   Trellis validation, archive, and commit if complete.

## Non-Goals

* No new arithmetic operations, broadcast coverage, compare/select, reduction,
  conversion, MAcc, widening-dot, memory-family rework, source-front-door
  routes, high-level frontend lowering, broad smoke matrices, dashboards, or
  report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, test names,
  status fields, exact intrinsic spelling, or metadata mirrors.
* No `ssh rvv` unless emitted C, runtime ABI order/counts, statement ordering,
  scalar-store/arithmetic behavior, correctness, or performance behavior
  changes.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused scalar splat-store and elementwise lit/generated-bundle filters
  after final touched route slices are known.
* Run added-line old-authority scan over touched production/test/spec files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-scalar-elementwise-runtime-avl-vl-sole-authority`

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a provider/target
validation naming/order/test cleanup. The expected diff should not change
emitted C/C++, runtime ABI order, runtime counts, statement ordering,
scalar-store behavior, elementwise arithmetic behavior, correctness behavior,
or performance behavior.

## Definition Of Done

RVV scalar splat-store and elementwise arithmetic target artifact validation
accept runtime `n` / AVL / VL facts only through the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`; route-local runtime/control fields
are visibly mirror-only consistency checks. Focused checks and scans pass, the
task is finished/archived, and a coherent commit records the production diff
and evidence.

## Implementation Results

* Added mirror-only comments to
  `RVVElementwiseArithmeticRouteValidationContract` and
  `RVVRuntimeScalarSplatStoreRouteValidationContract` so retained
  runtime/control copies are documented as target-side consistency mirrors.
* Rewired runtime scalar splat-store target payload validation to call
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)` after
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, covering local
  runtime control plan, runtime ABI order, setvl callee, VL C type,
  full-chunk VL, loop VL, and loop induction mirrors.
* Rewired elementwise arithmetic target payload validation to use the same
  shared route-local runtime AVL/VL mirror helper after selected-boundary
  validation, removing family-specific direct checks for runtime control,
  runtime ABI order, setvl, VL type, and EmitC AVL/VL names.
* Preserved scalar splat-store and elementwise-specific validation for route
  id, operation, memory/source form, dtype/config, family plans, scalar
  broadcast, masked/strided facts, headers, type mappings, ABI mappings,
  operand bindings, intrinsic leaves, statement-plan shape, and stale
  cross-family mirrors.
* Added target C++ provider-contract assertions that scalar splat-store and
  representative plain, masked, and scalar-broadcast elementwise retained
  route-local runtime/control mirrors match the embedded runtime AVL/VL
  selected-boundary contract.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings remained in target/test code: unused helper
warnings in target validation and enum switch coverage warnings in
`TargetArtifactExportTest.cpp`; no build errors occurred.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: passed.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with:

```text
RVV extension plugin smoke test passed
```

### Focused Lit / Generated-Bundle Dry-Run

From `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(add|sub|mul|masked-add|masked-sub|masked-mul|scalar-broadcast-add|scalar-broadcast-sub|scalar-broadcast-mul)'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 459
  Passed  :  18
```

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-splat-store'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 471
  Passed  :   6
```

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(selected-body|pre-realized|masked-add|pre-realized-masked-add|scalar-broadcast-add|pre-realized-scalar-broadcast-add|pre-realized-scalar-broadcast-sub)-dry-run'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 470
  Passed  :   7
```

### Old-Authority Scan

Added-line scan over touched production/test files:

```text
rtk proxy bash -lc 'if git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"; then exit 1; else exit 0; fi'
```

Result: passed with no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### SSH RVV Decision

`ssh rvv` was not run. This round changed provider/target validation ordering,
mirror-only documentation, and target C++ contract assertions only. It did not
change emitted C/C++, runtime ABI order, runtime counts, statement ordering,
scalar-store behavior, elementwise arithmetic behavior, correctness behavior,
or performance behavior.

### Spec Update Decision

No `.trellis/spec/` change was needed. The existing
`.trellis/spec/lowering-runtime/emitc-route.md` runtime scalar splat-store and
elementwise arithmetic sections already require embedded
`RVVRuntimeAVLVLSelectedBoundaryContract` authority, route-local runtime fields
as mirrors only, target-side selected-boundary-first validation, and the exact
focused C++/lit coverage this task implemented.
