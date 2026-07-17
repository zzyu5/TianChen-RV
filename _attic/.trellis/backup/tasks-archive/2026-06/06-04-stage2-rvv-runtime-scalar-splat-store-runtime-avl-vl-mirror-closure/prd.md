# Stage2 RVV runtime scalar splat-store runtime AVL/VL mirror closure

## Goal

Close the remaining runtime-boundary mirror authority leaks for the RVV runtime
scalar splat-store target-artifact consumer. The provider contract and target
validation must keep runtime control, runtime ABI order, setvl/VL/loop fields,
and candidate runtime metadata as route-local mirrors only; the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract` is the sole acceptance authority for
runtime `n`, AVL, and VL facts.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv runtime scalar splat-store body
  -> RVV plugin-owned runtime AVL/VL selected-boundary contract
  -> provider-built runtime scalar splat-store TCRVEmitCLowerableRoute
  -> target artifact validation consumes runtimeAVLVLContract first
  -> route-local runtime AVL/VL mirror consistency checks
  -> candidate metadata runtime mirrors checked only as mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime scalar splat-store runtime AVL/VL mirror closure`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `3a2729bf rvv: close memory runtime mirror metadata labels`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps RVV route authority in the selected
  `tcrv.exec` envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  neutral common EmitC materialization, and target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires runtime scalar
  splat-store provider construction to consume selected typed body facts,
  family plan facts, runtime AVL/VL control facts, residual ABI bindings, and
  statement plans, not route ids, artifact names, ABI strings, exact intrinsic
  spelling, or metadata mirrors.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeScalarSplatStoreRouteValidationContract` with an embedded
  `RVVRuntimeAVLVLSelectedBoundaryContract`, and requires target validation to
  consume that embedded contract before any local runtime ABI, setvl/pre-loop,
  loop, statement-plan, or mirror facts.
* Recent archived runtime AVL/VL tasks already moved scalar splat-store payload
  validation to selected-boundary-first target validation and added the shared
  `route-local runtime AVL/VL mirror` helper.
* Live target validation still labels scalar splat-store candidate metadata for
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` as selected
  typed runtime facts instead of route-local runtime AVL/VL mirrors.
* Live planning code still has a scalar-splat-store-local runtime ABI order
  constant path. The value may remain as a bounded mirror value, but it must be
  visibly derived into and checked against the runtime AVL/VL selected-boundary
  contract rather than treated as a separate acceptance authority.

## Requirements

* Scope is exactly the runtime scalar splat-store runtime-boundary consumer:
  provider facts, target payload validation, candidate metadata mirror labels,
  and focused tests for this family.
* Preserve the existing selected-boundary-first target payload validation:
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` must run before
  route-local runtime/control mirror checks.
* Retained route-local copies of `runtimeControlPlanID`, `runtimeABIOrder`,
  setvl callee, VL C type, full-chunk VL, loop VL, and loop induction must be
  documented or validated only as mirrors after selected-boundary validation.
* Candidate metadata labels for `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` must identify route-local runtime AVL/VL
  mirrors, not selected typed runtime acceptance facts.
* Stale or missing route-local runtime metadata/mirrors must fail closed with
  diagnostics containing `route-local runtime AVL/VL mirror`.
* Runtime ABI parameter-list checks may remain after selected-boundary
  validation, but only as ABI-binding consistency checks for `rhs_scalar`,
  `out`, and `n`; they must not make runtime ABI order a second acceptance
  authority.
* Preserve emitted C/C++, runtime ABI order/counts, statement ordering,
  scalar-splat-store behavior, correctness behavior, and performance behavior.
* Do not infer runtime control from route ids, artifact names, manifests,
  descriptor residue, C strings, scripts, tests, status fields, exact
  intrinsic spelling, or metadata mirrors.

## Acceptance Criteria

* [x] Runtime scalar splat-store candidate metadata validation labels
      `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` as
      route-local runtime AVL/VL mirrors.
* [x] Focused target tests assert the two runtime metadata labels are
      mirror-only labels for scalar splat-store.
* [x] Focused negative target tests for stale scalar splat-store runtime
      metadata/mirrors fail closed with diagnostics containing
      `route-local runtime AVL/VL mirror`.
* [x] Provider/target runtime ABI parameter-list checks remain after
      selected-boundary validation and are described as ABI-binding
      consistency checks, not runtime ABI order authority.
* [x] No emitted C/C++, runtime ABI order/count, statement ordering, or runtime
      behavior changes are introduced.
* [x] Focused scalar-splat-store target/export and plugin checks pass.
* [x] `rtk git diff --check` passes.
* [x] Bounded grep/added-line old-authority scan over touched files finds no
      new positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      common-EmitC, route-id/artifact-name, exact `__riscv_*_i32m1`, or
      mirror-only metadata as authority.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Technical Approach

1. Keep the existing scalar splat-store payload validation order:
   selected-boundary contract first, then route-local runtime AVL/VL mirrors,
   then ABI binding, headers/types, statement shape, and candidate mirrors.
2. Rename scalar splat-store candidate metadata runtime labels so stale/missing
   metadata failures name route-local runtime AVL/VL mirrors.
3. Tighten focused target tests around scalar splat-store metadata labels and
   stale metadata diagnostics.
4. If source comments are needed around runtime ABI parameter checks, clarify
   that they are ABI-binding consistency checks after selected-boundary
   acceptance.

## Non-Goals

* No new scalar-splat-store operations, dtype/LMUL cases, memory/reduction/
  conversion/contraction/compare-select/elementwise/source-front-door/
  high-level frontend/intrinsic coverage, or broad route-family sweeps.
* No movement of RVV semantics into common EmitC/export.
* No runtime behavior, generated C/C++, ABI order/count, or statement ordering
  change unless a focused test exposes an existing defect.
* No `ssh rvv` unless emitted C, runtime ABI order/counts, statement ordering,
  scalar-splat-store correctness, or runtime behavior changes.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused scalar splat-store lit/export checks if the changed behavior
  affects generated bundle or target-artifact route validation expectations.
* Run before/after bounded grep for scalar-splat-store runtime-control and
  runtime-ABI labels in touched validation/planning/test paths.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-splat-store-runtime-avl-vl-mirror-closure`

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a provider/target
validation label and diagnostic cleanup. The expected diff should not change
emitted C/C++, runtime ABI order/counts, statement ordering, scalar-splat-store
behavior, correctness behavior, or performance behavior.

## Definition Of Done

Runtime scalar splat-store target artifact validation visibly accepts runtime
`n` / AVL / VL facts only through the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`; route-local runtime/control fields
and candidate runtime metadata are checked only as mirrors with fail-closed
mirror diagnostics. Focused checks and scans pass, the task is finished/
archived, and one coherent commit records the bounded cleanup and evidence.

## Implementation Results

* Changed scalar splat-store target candidate metadata validation labels for
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` to
  `route-local runtime AVL/VL control plan mirror` and
  `route-local runtime AVL/VL ABI order mirror`.
* Kept scalar splat-store payload validation order unchanged:
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)`, then ABI-binding,
  header/type, ABI mapping, and statement-plan checks run.
* Clarified `validateRVVRuntimeScalarSplatStoreRuntimeABIFacts(...)` as an
  ABI-binding list consistency check after selected-boundary acceptance.
* Made missing candidate metadata diagnostics include the mirror label, so
  missing runtime metadata can fail with route-local runtime AVL/VL mirror
  wording instead of only generic provenance wording.
* Added target C++ regression coverage for stale scalar splat-store runtime ABI
  metadata, stale runtime control metadata, and missing runtime ABI metadata
  diagnostics carrying the route-local runtime AVL/VL mirror labels.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings remained: two unused helper warnings in
`RVVTargetArtifactRouteFamilyValidation.cpp` and enum switch coverage warnings
in `TargetArtifactExportTest.cpp`; no build errors occurred.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: passed.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with `RVV extension plugin smoke test passed`.

### Focused Lit

Initial source-root lit invocation failed because `test/lit.cfg.py` requires
the generated build-site config field `tianchenrv_obj_root`. The check was
rerun from `build/test`, matching the Ninja test working directory:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-splat-store
```

Result: passed, 6 selected runtime-scalar-splat-store tests passed and 471
unrelated tests were excluded.

### Scans

```text
rtk rg -n "selected typed RVV runtime scalar splat-store runtime|route-local runtime AVL/VL (control plan|ABI order) mirror|validateRVVRuntimeScalarSplatStoreRuntimeABIFacts" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp
```

Result: runtime scalar splat-store target labels now use route-local runtime
AVL/VL mirror wording; the old `selected typed RVV runtime scalar splat-store
runtime ...` label pattern is absent in touched target/test files.

```text
rtk proxy bash -lc 'git diff --unified=0 -- lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route id|artifact name|metadata.*authority|mirror-only.*authority)" || true'
```

Result: no added production/test line matched the old-authority scan.

```text
rtk git diff --check
```

Result: passed.

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-splat-store-runtime-avl-vl-mirror-closure
```

Result: passed.

## Spec Update Decision

No `.trellis/spec/` update was needed. The current
`.trellis/spec/lowering-runtime/emitc-route.md` already states that scalar
splat-store candidate metadata for `tcrv_rvv.runtime_control_plan` and
`tcrv_rvv.runtime_abi_order` must use route-local runtime AVL/VL mirror labels,
and this task implemented that existing contract.

## SSH RVV Decision

No `ssh rvv` run was needed. This task changed target validation labels,
candidate metadata diagnostics, and focused C++ tests only; it did not change
emitted C/C++, runtime ABI order/counts, statement ordering, scalar
splat-store behavior, correctness behavior, or performance behavior.
