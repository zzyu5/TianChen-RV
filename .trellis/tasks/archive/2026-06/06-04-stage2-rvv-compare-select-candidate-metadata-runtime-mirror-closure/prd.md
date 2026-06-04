# Stage2 RVV compare/select candidate-metadata runtime mirror closure

## Goal

Close the residual compare/select candidate metadata runtime mirror wording path
left after commit `fc10332f`. The compare/select provider-owned metadata mirror
contract already labels `tcrv_rvv.runtime_control_plan` and
`tcrv_rvv.runtime_abi_order` as route-local runtime AVL/VL mirrors, and target
validation already consumes `RVVRuntimeAVLVLSelectedBoundaryContract` before
route-family payload acceptance. This task makes the remaining target-side
fallback candidate metadata checks use the same explicit mirror-only wording so
stale metadata cannot be read as runtime AVL/VL authority.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select candidate-metadata runtime mirror closure`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported
  before this task was created.
* Initial `rtk git log --oneline -8` started at
  `fc10332f rvv: make compare select runtime mirrors non-authoritative`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` places RVV authority in the selected `tcrv.exec`
  envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned legality /
  realization / route provider, provider-built `TCRVEmitCLowerableRoute`, and
  neutral common EmitC/target validation. Metadata mirrors are not route,
  compute, dtype, runtime, or evidence authority.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select and
  other RVV execution facts to come from typed body/config/runtime facts and
  plugin-owned provider contracts, not route ids, artifact names, ABI strings,
  test names, descriptors, C snippets, or mirror metadata.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires compare/select
  target artifact validation to consume
  `RVVCompareSelectRouteValidationContract::runtimeAVLVLContract` before
  route payloads, header/type mappings, runtime ABI mappings, statement plans,
  predicate/select facts, or candidate metadata mirrors.
* The archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-compare-select-runtime-avl-vl-sole-authority/`
  rewired the primary compare/select route validation contract path to selected
  boundary first, added provider/candidate tests, and declared no spec update
  needed.
* Live provider planning inspection shows
  `buildRVVCompareSelectRouteMetadataMirrorContract(...)` already labels:
  `tcrv_rvv.runtime_control_plan` as
  `route-local runtime AVL/VL control plan mirror`, and
  `tcrv_rvv.runtime_abi_order` as
  `route-local runtime AVL/VL ABI order mirror`.
* Live target inspection shows
  `validateRVVCompareSelectMaskTargetArtifactCandidateMirrors(...)` first uses
  the provider-owned compare/select metadata mirror contract for
  compare/select producer routes. Its older fallback checks still label
  `tcrv_rvv.runtime_control_plan` as
  `selected typed RVV compare/select mask runtime AVL/VL control plan` and
  `tcrv_rvv.runtime_abi_order` as
  `selected typed RVV compare/select mask runtime ABI order`.
* Existing focused C++ coverage already mutates stale compare/select candidate
  metadata for these two keys on the provider-owned mirror-contract path and
  expects route-local runtime AVL/VL mirror diagnostics.

## Requirements

* Limit production edits to compare/select candidate metadata mirror validation
  and directly coupled focused tests if needed.
* Keep `RVVRuntimeAVLVLSelectedBoundaryContract` as the first runtime
  acceptance authority in compare/select route/provider validation.
* Preserve the provider-owned compare/select metadata mirror contract path.
* Make the remaining target-side compare/select fallback candidate metadata
  labels for `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` explicitly route-local runtime AVL/VL mirrors.
* Ensure stale/missing candidate metadata cannot become runtime AVL/VL
  authority. Any stale candidate mirror should fail with mirror diagnostics.
* Do not change emitted C/C++, runtime ABI order/counts, statement ordering,
  compare/select behavior, correctness behavior, or performance behavior.

## Acceptance Criteria

* [x] Before/after grep over
      `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` shows the
      compare/select fallback labels for `tcrv_rvv.runtime_control_plan` and
      `tcrv_rvv.runtime_abi_order` use route-local runtime AVL/VL mirror
      wording rather than selected typed RVV runtime authority wording.
* [x] The provider-owned compare/select metadata mirror contract in
      `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` remains intact and keeps
      route-local runtime AVL/VL mirror labels for the same two keys.
* [x] Focused C++ target coverage still rejects stale compare/select candidate
      metadata for `tcrv_rvv.runtime_control_plan` and
      `tcrv_rvv.runtime_abi_order` with route-local runtime AVL/VL mirror
      diagnostics.
* [x] The primary compare/select selected-boundary-first route validation
      remains intact.
* [x] Focused target artifact and compare/select lit checks pass.
* [x] `tianchenrv-rvv-extension-plugin-test` is run if provider/planning code
      changes; otherwise the no-provider-change rationale is recorded.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptors/direct-C/source-front-door,
      route-id/artifact-name, exact `__riscv_*_i32m1`, common EmitC
      semantics, or mirror-only authority.
* [x] `rtk git diff --check` passes.

## Technical Approach

1. Replace the two residual compare/select fallback label strings in
   `validateRVVCompareSelectMaskTargetArtifactCandidateMirrors(...)` with:
   `route-local runtime AVL/VL control plan mirror` and
   `route-local runtime AVL/VL ABI order mirror`.
2. Re-grep the compare/select target validation path for the old selected typed
   runtime labels.
3. Run the focused C++ target artifact test and compare/select lit filter. Run
   the RVV extension plugin test only if provider/planning code changes.
4. Record evidence, validate/archive the Trellis task, and commit one coherent
   change if complete.

## Out Of Scope

* No new compare/select operations, predicates, select variants, memory work,
  scalar splat-store work, elementwise work, conversion, reduction, MAcc,
  widening-dot, source-front-door routes, high-level frontend lowering, broad
  smoke matrices, dashboards, or report-only evidence.
* No movement of RVV runtime semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  descriptor residue, C strings, scripts, test names, status fields, exact
  intrinsic spelling, or metadata mirrors.
* No `ssh rvv` unless emitted C, runtime ABI order/counts, statement ordering,
  compare/select behavior, correctness, or performance behavior changes.

## Evidence Plan

* Before/after grep:
  `rtk rg -n "runtime_control_plan|runtime_abi_order|selected typed RVV compare/select mask runtime|route-local runtime AVL/VL" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp test/Target/TargetArtifactExportTest.cpp`
* Build and run focused target test:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16`
  `rtk build/bin/tianchenrv-target-artifact-export-test`
* Focused compare/select lit filter from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)'`
* Provider/planning code unchanged rationale, or if it changes:
  `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 16`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Added-line old-authority scan over touched files.
* `rtk git diff --check`.
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-candidate-metadata-runtime-mirror-closure`.

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a target validation
diagnostic/label cleanup. The intended diff does not change emitted C/C++,
runtime ABI order, runtime counts, statement ordering, compare/select runtime
behavior, correctness behavior, or performance behavior.

## Definition Of Done

The residual compare/select candidate metadata fallback labels are visibly
mirror-only, focused target/lit checks pass, no new old-authority wording is
introduced, the Trellis task is truthful and archived, and one commit records
the bounded cleanup.

## Implementation Results

* Updated the residual fallback checks in
  `validateRVVCompareSelectMaskTargetArtifactCandidateMirrors(...)`:
  * `tcrv_rvv.runtime_control_plan` label now reads
    `route-local runtime AVL/VL control plan mirror`.
  * `tcrv_rvv.runtime_abi_order` label now reads
    `route-local runtime AVL/VL ABI order mirror`.
* Left the provider-owned compare/select metadata mirror contract unchanged;
  it already carried the same route-local runtime AVL/VL mirror labels.
* Did not change emitted C/C++, provider planning, runtime ABI order/counts,
  statement ordering, compare/select semantics, correctness behavior, or
  performance behavior.

## Evidence Results

### Grep

Before source edit, live target inspection found the fallback labels:

```text
selected typed RVV compare/select mask runtime AVL/VL control plan
selected typed RVV compare/select mask runtime ABI order
```

After source edit:

```text
rtk rg -n "selected typed RVV compare/select mask runtime|route-local runtime AVL/VL|tcrv_rvv\\.runtime_control_plan|tcrv_rvv\\.runtime_abi_order" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp test/Target/TargetArtifactExportTest.cpp
```

Result: the old `selected typed RVV compare/select mask runtime...` labels no
longer appear in the target fallback path; the target fallback and provider
contract both show route-local runtime AVL/VL mirror wording for the two keys.

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
```

Result: passed. Existing unused-function warnings remained in
`RVVTargetArtifactRouteFamilyValidation.cpp`; no build errors occurred.

### C++ Target Test

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: passed with no failure output.

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

### RVV Extension Plugin Test Decision

`tianchenrv-rvv-extension-plugin-test` was not run because this round did not
modify provider/planning code or plugin route construction. The provider-owned
compare/select metadata mirror contract was only re-grepped for consistency.

### Old-Authority Scan

Added-line scan over the touched production source:

```text
rtk proxy bash -lc 'matches=$(git diff -U0 -- lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"); status=$?; if [ $status -eq 0 ]; then printf "%s\\n" "$matches"; exit 1; fi; exit 0'
```

Result: passed with no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-candidate-metadata-runtime-mirror-closure
```

Result: passed for `implement.jsonl` and `check.jsonl`.

## Spec Update Decision

No `.trellis/spec/` change is needed. The existing EmitC route spec already
requires compare/select candidate metadata to be checked only after the
embedded runtime AVL/VL selected-boundary contract has accepted runtime facts,
and already classifies family-local runtime fields as mirrors.

## Continuation Point

No continuation is required for this bounded compare/select candidate metadata
mirror label closure. Future work should switch route families only from a new
bounded Direction Brief.
