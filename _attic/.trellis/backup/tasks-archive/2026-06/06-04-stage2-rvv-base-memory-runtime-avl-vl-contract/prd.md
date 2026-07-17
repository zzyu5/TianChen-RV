# Stage2 RVV base-memory runtime AVL/VL contract migration

## Goal

Migrate the selected-body RVV base-memory movement route-family validation
contract to embed the named provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract`, then require target artifact
validation to consume that contract before accepting base-memory route payloads,
statement plans, metadata mirrors, headers, type mappings, ABI order, stride /
index / mask facts, and pointer advancement facts.

This round's bounded consumer slice is the existing base-memory movement route
family: strided load / unit store, unit load / strided store, indexed gather /
unit store, indexed scatter / unit load, static-mask unit load / store, and
static-mask unit store where the family is already production-active.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV base-memory runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no output, meaning the worktree was clean.
* Initial `rtk git log --oneline -8` started at
  `99ea72f8 rvv: promote runtime AVL VL boundary contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / realization / provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says base-memory movement is
  a selected-body realization / route-family provider consumer, and runtime
  AVL/VL, policy, ABI order, and selected capability facts must pass through
  RVV-owned route-control and statement-plan boundaries before provider route
  construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires route-family
  validation contracts that consume runtime AVL/VL facts to embed it instead
  of duplicating target-side fields.
* `.trellis/spec/lowering-runtime/emitc-route.md` also defines the base-memory
  fact surface. The provider accessor is the canonical source for operation,
  memory form, runtime ABI order, target leaf profile, provider-supported
  mirror, header/type summaries, operand binding plan/summary, route-family
  plan, stride/index/mask facts, and runtime ABI parameters.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  evidence only when runtime, correctness, or performance behavior is changed
  or claimed. This round should not change generated C/C++, runtime ABI order,
  runtime counts, pointer advancement expressions, or correctness behavior.
* The archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-runtime-avl-vl-selected-boundary-contract/`
  added the named runtime AVL/VL selected-boundary contract and embedded it in
  the segment2 validation contract.
* The same archived task names base-memory movement as the next continuation
  point for cross-family runtime AVL/VL promotion.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract` to
  `RVVBaseMemoryMovementRouteValidationContract`.
* Build the embedded runtime contract from provider-owned base-memory facts:
  typed config SEW/LMUL, tail/mask policy, config contract id, runtime-VL
  contract id, selected `with_vl` boundary, runtime AVL source,
  `setvl` intrinsic/type, full-chunk VL, loop VL, loop induction, remaining
  AVL metadata, pointer advancement metadata, bounded-slice/multi-VL support,
  runtime ABI order, and the runtime `n` ABI parameter.
* Update base-memory target validation so runtime AVL/VL contract validation
  runs before route payload, header/type mapping, ABI mapping, statement plan,
  stride/index/mask facts, pointer advancement metadata, and candidate mirror
  acceptance.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` scope,
  `setvl` intrinsic, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI role/order/ownership, remaining AVL metadata, and pointer
  advancement metadata where applicable.
* Keep common EmitC/export neutral. Do not infer runtime control from route
  ids, artifact names, manifests, descriptor residue, C strings, test names,
  status fields, or metadata mirrors.
* Add focused target/provider tests for base-memory positive runtime contract
  embedding and negative stale/missing contract fields.

## Acceptance Criteria

* [x] `RVVBaseMemoryMovementRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The base-memory contract builder derives the runtime contract from the
      provider route description / base-memory facts rather than target-local
      reconstruction.
* [x] Base-memory target artifact validation consumes the runtime AVL/VL
      contract before accepting headers, type mappings, ABI mappings,
      statement plans, route payloads, stride/index/mask facts, pointer
      advancement metadata, and metadata mirrors.
* [x] Negative C++ target coverage rejects stale or missing runtime AVL/VL
      contract facts, including runtime AVL source, runtime VL contract id,
      selected `with_vl` scope, `setvl` intrinsic or VL type, full-chunk VL,
      loop VL, loop induction, runtime `n` ABI facts, and pointer advancement
      metadata where applicable.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused lit or generated-bundle dry-run coverage for base-memory
      selected-body artifacts remains green.
* [x] Added-line old-authority scan over touched production/test files finds
      no new positive dependency on `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/
      source-front-door authority, mirror-only authority, route-id/artifact-name
      authority, or exact `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build focused C++ tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused lit/generated-bundle dry-run tests for base-memory selected-body
  artifacts after locating the exact filter in live tests.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-base-memory-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, pointer advancement expressions, harness behavior, or
correctness semantics. The planned diff is contract/validation/test-only. If
that remains true, state the no-runtime-change rationale and cite the existing
base-memory generated-bundle / executable evidence that remains unchanged.

## Out Of Scope

* No new RVV route coverage, dtype/LMUL clone batches, segment2 expansion,
  compare/select expansion, reduction/MAcc expansion, source-front-door routes,
  high-level frontend lowering, dashboards, broad smoke matrices, or report-only
  evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, or tests.
* No cross-family migration beyond base-memory movement in this round.

## Definition Of Done

Base-memory target route validation accepts only provider-built base-memory
routes whose embedded runtime AVL/VL selected-boundary contract matches the
rebuilt route description and statement payload. Focused C++/lit checks pass,
old-authority scan is clean, the task is finished and archived, and a coherent
commit records the production diff plus PRD evidence.

## Implementation Results

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVBaseMemoryMovementRouteValidationContract`.
* Rewired
  `getRVVBaseMemoryMovementRouteValidationContract(...)` to build the embedded
  runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after base-memory runtime
  ABI parameters are populated.
* Rewired base-memory target route validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before route id,
  payload, header/type mapping, ABI mapping, statement plan, layout, and
  candidate mirror acceptance.
* Rewired base-memory statement-plan validation to consume the embedded runtime
  contract for runtime `n`, `setvl` callee, VL C type, full-chunk VL, loop VL,
  loop induction, loop bounds, remaining AVL expression inputs, and
  pointer-advance induction.
* Kept common EmitC/export neutral. The diff did not move RVV semantics into
  common code and did not change emitted route statement construction.
* Added target C++ coverage asserting all production-active base-memory
  validation contracts embed matching runtime AVL/VL selected-boundary facts.
* Added fail-closed target C++ coverage for stale runtime AVL source, runtime
  VL contract, selected `with_vl` boundary/scope, `setvl` callee, VL C type,
  full-chunk VL, loop VL, loop induction, runtime `n` ABI role, and pointer
  advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the base-memory
  validation contract signature, target consumer order, error matrix, tests,
  and wrong/correct flow.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings remained in target/test code:
unused function warnings in target validation and switch coverage warnings in
`TargetArtifactExportTest.cpp`; no build errors occurred.

After one test expectation self-repair:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
```

Result: passed.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: initially failed one new negative-test text expectation because the
runtime `n` ABI role mutation now fails earlier at the shared runtime AVL/VL
selected-boundary contract. The expectation was corrected to the earlier
fail-closed diagnostic, then the test passed.

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
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-load-store)|pre-realized-selected-body-artifact-(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-load-store|masked-unit-store)'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 466 (97.69%)
  Passed  :  11 (2.31%)
```

### Old-Authority Scan

Added-line scan over touched production/test files:

```text
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"'
```

Result: no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-base-memory-runtime-avl-vl-contract
```

Result: passed for `implement.jsonl` and `check.jsonl`.

### SSH RVV

Not rerun. This round changed provider/target validation contracts, C++ tests,
and spec text only. It did not change emitted C/C++, runtime ABI order,
runtime counts, pointer advancement expressions, harness behavior, or
correctness semantics. Existing base-memory generated-bundle artifact tests
remained green under the focused lit filter above.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` because this task
changed a cross-layer provider/target validation contract. The base-memory fact
surface now records that `RVVBaseMemoryMovementRouteValidationContract`
embeds `RVVRuntimeAVLVLSelectedBoundaryContract`, and that target validation
must consume it before route-family payloads and mirrors.

## Continuation Point

Full cross-family runtime AVL/VL promotion remains open. The next bounded
consumer can migrate another existing route-family validation contract, such as
compare/select or standalone reduction, to embed and consume
`RVVRuntimeAVLVLSelectedBoundaryContract`.
