# Stage2 RVV compare/select runtime AVL/VL sole-authority cleanup

## Goal

Make RVV compare/select target artifact validation visibly treat
`RVVRuntimeAVLVLSelectedBoundaryContract` as the sole acceptance authority for
runtime `n` / AVL / VL facts. Existing compare/select route-local
runtime/control copies and candidate metadata values may remain only as
generated mirrors checked after the shared selected-boundary contract has
matched the rebuilt provider route.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv compare/select body
  -> RVV plugin-owned runtime AVL/VL selected-boundary contract
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation consumes runtimeAVLVLContract first
  -> route-local runtime/control mirror consistency checks
  -> compare/select payload/header/type/ABI/statement/mirror validation
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select runtime AVL/VL sole-authority cleanup`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: `ok`, meaning no short status entries
  were reported through RTK.
* Initial `rtk git log --oneline -8` started at
  `f7ad1708 rvv: make scalar elementwise runtime mirrors non-authoritative`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps RVV route authority in the selected
  `tcrv.exec` envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned
  legality / realization / provider, provider-built `TCRVEmitCLowerableRoute`,
  neutral common EmitC materialization, and target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires runtime ABI roles
  from `tcrv.exec.runtime_param` to be explicitly consumed by typed RVV
  control/dataflow; route ids, artifact names, ABI strings, diagnostics, and
  metadata mirrors are not route authority.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned selected
  boundary authority for runtime AVL source, runtime VL contract, selected
  `with_vl` scope, `setvl`, VL C type, full-chunk VL, loop VL, loop
  induction, runtime `n` ABI parameter, remaining AVL metadata, pointer
  advancement, bounded slice, and multi-VL facts.
* The archived compare/select runtime AVL/VL contract migration already added
  `runtimeAVLVLContract` to `RVVCompareSelectRouteValidationContract` and made
  target validation call
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before compare/select
  payload/header/type/ABI/statement/mirror acceptance.
* The archived scalar splat-store and elementwise arithmetic sole-authority
  cleanup introduced the current target-side pattern: validate the shared
  selected-boundary contract first, then compare retained route-local runtime
  control plan, runtime ABI order, setvl callee, VL C type, full-chunk VL, loop
  VL, and loop induction copies through
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)`.
* Live header inspection shows `RVVCompareSelectRouteValidationContract` still
  exposes `runtimeControlPlanID` and `runtimeABIOrder` without the mirror-only
  comment used by promoted memory/elementwise contracts.
* Live target inspection shows compare/select validation calls the shared
  selected-boundary validator, then still directly compares route-local
  runtime ABI order, runtime AVL/VL control plan, VL C type, and setvl callee
  through family-specific provider-field checks.
* Live provider metadata mirror inspection shows compare/select candidate
  metadata labels still describe `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` as selected typed RVV facts, rather than
  explicit route-local runtime AVL/VL mirrors.

## Requirements

* Cover exactly the compare/select producer route family:
  plain compare-select, computed-mask select, runtime-scalar compare-select,
  and runtime-scalar dual compare-mask-and-select.
* Keep `RVVRuntimeAVLVLSelectedBoundaryContract` as the first runtime
  acceptance check in compare/select target payload validation.
* Document retained compare/select route-local runtime/control fields in
  `RVVCompareSelectRouteValidationContract` as target-side consistency
  mirrors whose authority is `runtimeAVLVLContract`.
* Replace compare/select family-specific direct runtime control / ABI order /
  setvl / VL type checks with the shared route-local runtime AVL/VL mirror
  helper after selected-boundary validation.
* Keep stale, missing, or mismatched route-local runtime/control mirrors
  fail-closed with diagnostics that name route-local runtime AVL/VL mirror
  mismatch rather than treating route-description runtime facts as authority.
* Rename compare/select candidate metadata mirror labels for runtime control
  plan and runtime ABI order so they are explicitly route-local runtime AVL/VL
  mirrors, not selected authority.
* Preserve compare/select-specific validation for operation kind, memory form,
  dtype/config, compare predicates, select layout, dual compare facts, mask
  facts, headers, type mappings, ABI mappings, operand bindings, intrinsic
  leaves, statement-plan shape, and stale cross-family mirrors.
* Preserve emitted C/C++, runtime ABI order, runtime counts, statement
  ordering, compare/select behavior, correctness behavior, and performance
  behavior.
* Keep common EmitC/export neutral. Do not infer runtime control or
  compare/select semantics from route ids, artifact names, manifests,
  descriptor residue, C strings, tests, scripts, status fields, exact
  intrinsic spelling, or metadata mirrors.

## Acceptance Criteria

* [x] `RVVCompareSelectRouteValidationContract` documents retained
      runtime/control copies as target-side consistency mirrors whose
      acceptance authority is `runtimeAVLVLContract`.
* [x] Compare/select target validation consumes `runtimeAVLVLContract` before
      route-local runtime/control mirror checks.
* [x] Compare/select target validation uses
      `validateRVVRouteLocalRuntimeAVLVLMirrors(...)` for retained runtime
      control plan, runtime ABI order, setvl callee, VL C type, full-chunk VL,
      loop VL, and loop induction mirrors.
* [x] Compare/select metadata mirror contracts label runtime control plan and
      runtime ABI order as route-local runtime AVL/VL mirrors.
* [x] Focused target C++ coverage asserts plain, runtime-scalar, and
      runtime-scalar dual compare/select provider contracts keep retained
      route-local runtime/control mirrors equal to the embedded
      selected-boundary contract.
* [x] Focused target C++ stale-mirror coverage rejects stale compare/select
      candidate runtime control plan and runtime ABI order metadata as mirror
      mismatches.
* [x] Existing stale selected-boundary negative coverage for compare/select
      runtime AVL/VL facts remains green.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass if provider/target code
      changes.
* [x] Focused compare/select lit/generated-bundle dry-run checks pass.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      common-EmitC, route-id/artifact-name, exact `__riscv_*_i32m1`, or
      mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Technical Approach

1. Add the mirror-only comment to `RVVCompareSelectRouteValidationContract`.
2. In compare/select payload contract validation, keep
   `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` first, then call
   `validateRVVRouteLocalRuntimeAVLVLMirrors(...)` with the retained
   compare/select contract copies.
3. Remove compare/select family-specific direct checks for runtime ABI order,
   runtime control plan, setvl callee, and VL C type from the post-contract
   provider-field list; keep non-runtime compare/select checks unchanged.
4. Rename compare/select metadata mirror labels for
   `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` to
   route-local runtime AVL/VL mirror wording.
5. Extend focused target C++ compare/select provider-contract assertions to
   check route-local runtime AVL/VL mirror equality against the embedded
   selected-boundary contract.
6. Add or tighten focused candidate stale-mirror coverage for compare/select
   runtime control plan and runtime ABI order metadata.
7. Run focused build/test/lit checks, old-authority scan, `git diff --check`,
   Trellis validation, archive, and commit if complete.

## Out Of Scope

* No new compare operations, predicates, select variants, memory-family work,
  scalar splat-store work, elementwise work, conversion, reduction, MAcc,
  widening-dot, source-front-door routes, high-level frontend lowering, broad
  smoke matrices, dashboards, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, test names,
  status fields, exact intrinsic spelling, or metadata mirrors.
* No `ssh rvv` unless emitted C, runtime ABI order/counts, statement ordering,
  compare/select behavior, correctness, or performance behavior changes.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused compare/select lit/generated-bundle filters after confirming the
  live test names.
* Run added-line old-authority scan over touched production/test/spec files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-runtime-avl-vl-sole-authority`

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a provider/target
validation naming/order/test cleanup. The expected diff should not change
emitted C/C++, runtime ABI order, runtime counts, statement ordering,
compare/select behavior, correctness behavior, or performance behavior.

## Definition Of Done

RVV compare/select target artifact validation accepts runtime `n` / AVL / VL
facts only through the embedded `RVVRuntimeAVLVLSelectedBoundaryContract`;
route-local runtime/control fields and candidate metadata are visibly
mirror-only consistency checks. Focused checks and scans pass, the task is
finished/archived, and a coherent commit records the production diff and
evidence.

## Implementation Results

* Added mirror-only documentation to
  `RVVCompareSelectRouteValidationContract` so retained runtime/control copies
  are explicitly target-side consistency mirrors and
  `runtimeAVLVLContract` is the acceptance authority.
* Rewired compare/select target contract validation to call
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)` immediately after
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* Removed compare/select family-specific direct checks for route-local runtime
  ABI order, runtime AVL/VL control plan, setvl callee, and VL C type.
  Non-runtime compare/select checks for operation, memory form, config,
  predicates, select/mask facts, headers, type mappings, ABI mappings, operand
  bindings, intrinsic leaves, statement plans, and stale cross-family mirrors
  remain in place.
* Renamed compare/select metadata mirror labels for
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` to
  route-local runtime AVL/VL mirror wording.
* Extended target C++ compare/select provider-contract checks so plain,
  runtime-scalar, and runtime-scalar dual compare/select routes assert retained
  route-local runtime/control mirrors match the embedded selected-boundary
  contract.
* Added focused stale candidate metadata coverage for compare/select
  `tcrv_rvv.runtime_control_plan`, and tightened stale
  `tcrv_rvv.runtime_abi_order` coverage to assert route-local runtime AVL/VL
  mirror diagnostics.

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

Result: passed with no failure output.

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
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 457 (95.81%)
  Passed  :  20 (4.19%)
```

### Old-Authority Scan

Added-line scan over touched production/test files:

```text
rtk proxy bash -lc 'matches=$(git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"); status=$?; if [ $status -eq 0 ]; then printf "%s\n" "$matches"; exit 1; fi; exit 0'
```

Result: passed with no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-runtime-avl-vl-sole-authority
```

Result: passed for `implement.jsonl` and `check.jsonl`.

### SSH RVV Decision

`ssh rvv` was not run. This round changed provider/target validation order,
mirror-only labels/comments, and target C++ contract assertions only. It did
not change emitted C/C++, runtime ABI order, runtime counts, statement
ordering, compare/select behavior, correctness behavior, or performance
behavior.

## Spec Update Decision

No `.trellis/spec/` change was needed. The existing
`.trellis/spec/lowering-runtime/emitc-route.md` runtime AVL/VL
selected-boundary contract already requires target validation to consume the
embedded selected-boundary contract before route-local runtime/control mirror
checks, and already classifies route-local runtime/control copies as mirrors
only.

## Continuation Point

No continuation is required for this bounded compare/select sole-authority
cleanup. Future work can promote another route family only from a new bounded
Direction Brief.
