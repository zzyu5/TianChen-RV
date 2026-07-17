# Stage2 RVV compare/select runtime AVL/VL contract migration

## Goal

Migrate the selected-body RVV compare/select route-family validation contract to
embed the provider-owned `RVVRuntimeAVLVLSelectedBoundaryContract`, then require
target artifact validation to consume that contract before accepting
compare/select mask route payloads, headers, C type mappings, runtime ABI
mappings, statement-plan facts, and metadata mirrors.

This round's bounded consumer slice is the production-active compare/select
route family: plain compare-select, computed-mask select, runtime-scalar
computed-mask select, and runtime-scalar dual compare-mask-and-select.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: `ok`, meaning the worktree had no short
  status entries through RTK.
* Initial `rtk git log --oneline -8` started at
  `6be47366 rvv: consume runtime AVL VL contract for base memory`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / realization / provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  provider construction to consume verified family plans, materialization
  facts, elementwise/select operand-binding facts, route-control provider facts,
  and the RVV-owned compare/select statement plan before route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires route-family validation
  contracts that consume runtime AVL/VL facts to embed it rather than duplicate
  target-side runtime checks.
* The archived segment2 runtime AVL/VL promotion added the named runtime
  contract and wired segment2 target validation to consume it before route
  payload and mirror acceptance.
* The archived base-memory migration embedded the same contract in
  `RVVBaseMemoryMovementRouteValidationContract`, consumed it before base-memory
  target validation accepted headers/types/ABI/statement payloads, and kept the
  diff contract/test-only with no `ssh rvv` rerun.
* Live inspection shows `RVVCompareSelectRouteValidationContract` lacks both
  `configContractID` and an embedded `RVVRuntimeAVLVLSelectedBoundaryContract`,
  while `RVVSelectedBodyEmitCRouteDescription` already carries
  `configContractID`, `runtimeVLContractID`, runtime AVL source, selected
  `with_vl`/VL op facts, loop VL names, remaining AVL metadata, pointer
  advancement metadata, and runtime ABI parameters.
* Live target validation has a shared
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` helper already used by
  base-memory and segment2 consumers.
* Existing compare/select target validation reconstructs runtime ABI and
  statement-plan checks from the local compare/select contract and description;
  it does not yet require the named runtime AVL/VL selected-boundary contract
  before accepting the compare/select route.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract` to
  `RVVCompareSelectRouteValidationContract`.
* Add `configContractID` to the compare/select validation contract so the
  embedded runtime contract can mirror the rebuilt selected-body description.
* Build the embedded runtime contract from provider-owned compare/select facts
  and rebuilt route description fields: typed config SEW/LMUL, tail/mask
  policy, config contract id, runtime-VL contract id, selected `with_vl`
  boundary, runtime AVL source, `setvl` intrinsic/type, full-chunk VL, loop VL,
  loop induction, remaining AVL metadata, pointer advancement metadata,
  bounded-slice/multi-VL support, runtime ABI order, and the runtime `n` ABI
  parameter.
* Update compare/select target validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  compare/select headers, type mappings, runtime ABI mappings, statement plans,
  compare predicate facts, select layout, dual-compare facts, and candidate
  metadata mirrors.
* Make compare/select statement-plan validation consume the embedded runtime
  contract for provider-expected runtime `n`, `setvl` callee, VL C type,
  full-chunk VL, loop VL, loop induction, loop upper bound, and remaining AVL
  expression checks when the compare/select contract is available.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` boundary/scope,
  `setvl` intrinsic, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI role/order/ownership, remaining AVL metadata, and pointer
  advancement metadata where applicable.
* Keep common EmitC/export neutral. Do not infer runtime control from route ids,
  artifact names, manifests, descriptor residue, C strings, test names, status
  fields, or metadata mirrors.

## Acceptance Criteria

* [x] `RVVCompareSelectRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The compare/select contract builder derives the embedded runtime contract
      via `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` from provider route
      facts and rebuilt selected-body description facts.
* [x] Compare/select target artifact validation consumes the runtime AVL/VL
      contract before accepting route payloads, headers, type mappings, ABI
      mappings, statement plans, compare/select family facts, and metadata
      mirrors.
* [x] Compare/select statement-plan validation uses the embedded runtime
      contract for pre-loop setvl, loop setvl, full-chunk VL, loop VL,
      induction, runtime n/AVL, and remaining AVL checks when the contract is
      present.
* [x] Negative C++ target coverage rejects stale or missing runtime AVL/VL
      contract facts, including runtime AVL source, runtime VL contract id,
      selected `with_vl` scope, `setvl` callee, VL C type, full-chunk VL, loop
      VL, loop induction, runtime `n` ABI facts, and pointer advancement
      metadata.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused lit or generated-bundle dry-run coverage for compare/select
      selected-body artifacts remains green.
* [x] Added-line old-authority scan over touched production/test files finds no
      new positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door
      authority, mirror-only authority, route-id/artifact-name authority, or
      exact `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build focused C++ tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused lit/generated-bundle dry-run tests for compare/select
  selected-body artifacts after locating the exact filter in live tests.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, pointer advancement expressions, harness behavior, or
correctness semantics. The planned diff is contract/validation/test-only. If
that remains true, state the no-runtime-change rationale and cite the existing
compare/select generated-bundle evidence that remains unchanged.

## Out Of Scope

* No new RVV route coverage, predicates, compare variants, select variants,
  dtype/LMUL clone batches, reductions, MAcc, segment2 expansion,
  source-front-door routes, high-level frontend lowering, dashboards, broad
  smoke matrices, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, tests, or
  metadata mirrors.
* No cross-family migration beyond compare/select in this round.

## Definition Of Done

Compare/select target route validation accepts only provider-built
compare/select routes whose embedded runtime AVL/VL selected-boundary contract
matches the rebuilt route description and statement payload. Focused C++/lit
checks pass, old-authority scan is clean, the task is finished and archived,
and a coherent commit records the production diff plus PRD evidence.

## Implementation Results

* Added `configContractID` and
  `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVCompareSelectRouteValidationContract`.
* Rewired
  `getRVVCompareSelectRouteValidationContract(...)` to build the embedded
  runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after compare/select
  runtime ABI parameters are populated.
* Rewired compare/select target route validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before
  route-family headers, type mappings, ABI mappings, compare predicate facts,
  select layout, dual-compare facts, statement-plan validation, and candidate
  mirror acceptance.
* Rewired compare/select statement-plan validation to use the embedded runtime
  contract for runtime `n`, `setvl` callee, VL C type, full-chunk VL, loop VL,
  loop induction, loop bounds, and remaining AVL expression checks when the
  compare/select validation contract is present.
* Updated compare/select target tests to assert the embedded runtime AVL/VL
  selected-boundary contract mirrors rebuilt route facts.
* Added fail-closed target coverage for stale runtime AVL source, runtime VL
  contract, selected `with_vl` scope, `setvl` callee, VL C type, full-chunk VL,
  loop VL, loop induction, runtime `n` ABI role, and pointer advancement
  metadata.
* Updated manual compare/select target fixtures to use the canonical RVV
  config/VL contract names for selected boundary, full-chunk VL, loop VL,
  loop induction, remaining AVL metadata, pointer advancement metadata,
  bounded slice, and multi-VL facts.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the
  compare/select route validation contract, target consumer order, error
  matrix, tests, and wrong/correct flow.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: initially failed because `configContractID` had not been added to the
compare/select contract header even though provider/target/test code used it.
The missing header field was added, then the focused target test was rebuilt.
Existing warnings remained in target/test code; no final build errors remain.

After self-repair:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: initially failed the new embedded runtime contract positive check
because manual compare/select target fixtures still used older local runtime
names such as `vl_full` / `i` instead of the canonical RVV config/VL contract
names. The fixtures were repaired to use provider-owned config/VL facts, and
the test then passed.

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
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"'
```

Result: no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-runtime-avl-vl-contract
```

Result: passed for `implement.jsonl` and `check.jsonl`.

### SSH RVV

Not rerun. This round changed provider/target validation contracts, target C++
tests, manual target fixtures, Trellis task files, and spec text only. It did
not change emitted C/C++, runtime ABI order, runtime counts, pointer
advancement expressions, harness behavior, or correctness semantics. The
focused compare/select generated-bundle artifact tests remained green under
the lit filter above.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` because this task
changed a cross-layer provider/target validation contract. The spec now records
that `RVVCompareSelectRouteValidationContract` embeds
`RVVRuntimeAVLVLSelectedBoundaryContract`, carries `configContractID`, and that
target validation must consume the runtime contract before compare/select
payloads, statement plans, predicate/select facts, ABI/header/type facts, and
mirrors.

## Continuation Point

Full cross-family runtime AVL/VL promotion remains open. The next bounded
consumer can migrate another existing route-family validation contract, such
as standalone reduction/accumulation, to embed and consume
`RVVRuntimeAVLVLSelectedBoundaryContract`.
