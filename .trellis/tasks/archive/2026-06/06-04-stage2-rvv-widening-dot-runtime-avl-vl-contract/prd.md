# Stage2 RVV widening-dot runtime AVL/VL contract migration

## Direction

Migrate the selected-body RVV widening-dot reduction route-family validation
contract to embed the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract`, then require target artifact
validation to consume that contract before accepting widening-dot route
payloads, runtime ABI facts, source/accumulator/result type facts, memory-form
facts, contraction facts, statement-plan facts, and metadata mirrors.

## Module Goal

Make runtime `n` / AVL / VL authority flow through the RVV plugin-owned
selected-boundary contract for widening dot-reduction:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv widening-dot body
  -> RVVRuntimeAVLVLSelectedBoundaryContract embedded in
     RVVWideningDotReduceRouteValidationContract
  -> provider-built TCRVEmitCLowerableRoute setvl/loop/dot-reduction statements
  -> target artifact validation consumes that contract
  -> common EmitC materialization remains neutral
```

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: clean through RTK (`ok` / no short status
  entries).
* Initial `rtk git log --oneline -8` started at
  `52cd73a2 rvv: consume runtime AVL VL contract for MAcc`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / realization / provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` classifies active
  widening-dot reductions as direct contraction route-provider consumers.
  Runtime AVL/VL, SEW/LMUL, policy, capability, ABI order, materialized leaves,
  math operand binding, and statement construction must be provider-owned and
  must not be inferred from names, metadata, ABI strings, route ids, or common
  EmitC.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires route-family
  validation contracts that consume runtime AVL/VL facts to embed it instead of
  duplicating target-side runtime checks.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  evidence only when runtime, correctness, or performance behavior is changed
  or claimed. This round should be contract/validation/test-only.
* Archived runtime AVL/VL contract tasks show the established migration
  pattern: segment2, base-memory, compare/select, standalone reduction, and
  MAcc now embed `runtimeAVLVLContract` and make target validation consume it
  before payload/header/type/ABI/statement/mirror acceptance.
* Live inspection shows `RVVWideningDotReduceRouteValidationContract` still
  carries family-local runtime fields such as `runtimeControlPlanID`,
  `runtimeABIOrder`, `setVLIntrinsic`, `vlCType`, `emitCFullChunkVLName`,
  `emitCLoopVLName`, and `emitCLoopInductionName`, but it lacks both
  `configContractID` and embedded `RVVRuntimeAVLVLSelectedBoundaryContract`.
* Live target validation already has
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, but widening-dot
  target validation currently checks runtime control through widening-dot-local
  fields before validating statement plans.

## Requirements

* Add `configContractID` and
  `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVWideningDotReduceRouteValidationContract`.
* Build the embedded runtime contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` from provider-owned
  widening-dot facts and rebuilt selected-body description fields: result
  SEW/LMUL, tail/mask policy, config contract id, runtime-VL contract id,
  selected `with_vl` boundary/scope, runtime AVL source, setvl intrinsic, VL C
  type, full-chunk VL, loop VL, loop induction, remaining AVL metadata,
  pointer advancement metadata, bounded-slice/multi-VL support, runtime ABI
  order, and the runtime `n` ABI parameter.
* Update widening-dot target validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  runtime ABI facts, route operand binding facts, contraction family plan,
  source/accumulator/result SEW/LMUL, memory-form facts, mask facts,
  headers/type mappings, route payloads, statement plans, and metadata mirrors.
* Make widening-dot statement-plan validation consume the embedded runtime
  contract for provider-expected runtime `n`, pre-loop setvl, loop setvl,
  full-chunk VL, loop VL, loop induction, loop upper bound, remaining AVL
  expression, selected-body provenance, pointer advancement, and runtime ABI
  role/order checks.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` scope, setvl
  callee, VL C type, full-chunk VL, loop VL, loop induction, runtime `n` ABI
  role/order/ownership, remaining AVL metadata, pointer advancement metadata,
  and widening-dot statement-plan facts.
* Keep common EmitC/export neutral. Do not infer runtime control from route ids,
  artifact names, manifests, descriptor residue, C strings, tests, status
  fields, exact intrinsic spelling, or metadata mirrors.

## Acceptance Criteria

* [x] `RVVWideningDotReduceRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The widening-dot contract builder derives the embedded runtime contract
      through the shared provider-owned helper rather than target-local
      reconstruction.
* [x] Widening-dot target artifact validation consumes the runtime AVL/VL
      contract before accepting runtime ABI, route payload, header/type
      mapping, statement-plan, contraction, source/accumulator/result,
      memory-form, mask, and mirror facts.
* [x] Widening-dot statement-plan validation uses the embedded runtime contract
      for pre-loop setvl, loop setvl, full-chunk VL, loop VL, induction,
      runtime n/AVL, remaining AVL, and pointer advancement checks.
* [x] Negative C++ target coverage rejects stale or missing runtime AVL/VL
      contract facts, including runtime AVL source, runtime VL contract id,
      selected `with_vl` scope, setvl callee, VL C type, full-chunk VL, loop
      VL, loop induction, runtime `n` ABI facts, remaining AVL metadata, and
      pointer advancement metadata.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused widening-dot lit or generated-bundle dry-run coverage remains
      green.
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
* Run focused widening-dot generated-bundle/lit dry-run after locating the live
  filter.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-widening-dot-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, widening-dot accumulation/result
behavior, harness behavior, or correctness semantics. The intended diff is
contract/validation/test-only. If that remains true, state the
no-runtime-change rationale and cite existing widening-dot executable evidence
from prior artifact ABI / production validation tasks.

## Out Of Scope

* No new widening-dot operations, dtype/LMUL clone batches, conversion/dtype
  routes, MAcc expansion, standalone reduction expansion, segment2,
  compare/select, source-front-door routes, high-level frontend lowering,
  dashboards, broad smoke matrices, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, tests, exact
  intrinsic spelling, or metadata mirrors.
* No cross-family migration beyond widening-dot in this round.

## Definition Of Done

Widening-dot target route validation accepts only provider-built widening-dot
routes whose embedded runtime AVL/VL selected-boundary contract matches the
rebuilt route description and statement payload. Focused C++/lit checks pass,
old-authority scan is clean, the task is finished and archived, and a coherent
commit records the production diff plus PRD evidence.

## Implementation Results

* Added `configContractID` and
  `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVWideningDotReduceRouteValidationContract`.
* Rewired
  `populateRVVWideningDotValidationContract(...)` to build the embedded
  runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after widening-dot runtime
  ABI parameters are populated.
* The embedded runtime contract uses provider-derived widening-dot result
  SEW/LMUL, tail/mask policy, selected config contract id, setvl intrinsic, VL
  C type, runtime ABI order, and the provider-owned runtime `n` ABI parameter.
* Rewired widening-dot target route validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before runtime
  ABI, route payload, source/accumulator/result SEW-LMUL, contraction,
  memory-form, mask, header/type, statement-plan, and mirror acceptance.
* Rewired widening-dot statement-plan validation to consume the embedded
  runtime contract for runtime `n`, pre-loop setvl, loop setvl, full-chunk VL,
  loop VL, loop induction, loop bounds/step, remaining AVL expression inputs,
  and induction-based pointer advancement.
* Added target C++ positive coverage asserting the embedded runtime AVL/VL
  selected-boundary contract mirrors rebuilt widening-dot route facts.
* Added fail-closed widening-dot target coverage for stale runtime AVL source,
  missing runtime VL contract, stale selected `with_vl` scope, stale setvl
  callee, stale VL C type, stale full-chunk VL, stale loop VL, stale loop
  induction, stale runtime `n` ABI role, stale remaining AVL metadata, and
  stale pointer advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` so the existing
  Widening Dot-Reduce Route Validation Contract section records the embedded
  runtime AVL/VL selected-boundary contract field and target consumer order.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed after self-repairing the initial header field placement error.
Existing warnings were emitted in generated/test code, but no build errors
remained.

After updating target test expectations for the new first rejection point:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
```

Result: passed.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: failed once because old stride-role negative assertions expected the
later widening-dot ABI index check. The new runtime AVL/VL contract correctly
failed earlier with `exactly one provider-owned runtime n/AVL ABI parameter`.
The test expectations were updated to match the new consumer order.

Rerun result: passed.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with `RVV extension plugin smoke test passed`.

### Focused Lit / Generated-Bundle Dry-Run

From `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widening-dot-reduce|widening-dot-reduction'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 454
  Passed  :  23
```

### Old-Authority Scan

Added-line scan:

```text
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/spec/lowering-runtime/emitc-route.md .trellis/tasks/06-04-stage2-rvv-widening-dot-runtime-avl-vl-contract | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\\\.i32_|!tcrv_rvv\\\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"'
```

Result: no matches.

### Whitespace And Trellis

```text
rtk git diff --check
```

Result: passed.

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-widening-dot-runtime-avl-vl-contract
```

Result: passed.

## SSH RVV

Not rerun. This round changed provider/target validation contracts, target C++
tests, Trellis records, and spec text only. It did not change emitted C/C++,
runtime ABI order, runtime counts, statement ordering, widening-dot
accumulation/result behavior, mask/tail behavior, stride behavior, harness, or
correctness/performance semantics.

Existing executable evidence remains the authority for unchanged runtime
behavior:

* Plain `widening_dot_reduce_add`:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/prd.md`
  records real `ssh rvv` counts `0,1,16,23,257` passing with signed horizontal
  dot, seed-added, scalar-output, tail-preserved, and distinguishing checks.
* Strided-input `strided_input_widening_dot_reduce_add`:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/prd.md`
  records real `ssh rvv` counts `0,1,16,17,257` passing for stride/data pairs.
* Computed-mask `computed_masked_widening_dot_reduce_add`:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/prd.md`
  records real `ssh rvv` counts `0,1,16,17,257` passing.
* Computed-mask strided-input
  `computed_masked_strided_input_widening_dot_reduce_add`:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/prd.md`
  records real `ssh rvv` counts `0,1,16,17,257` passing for stride/data/mask
  cases.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` because this task
changed a concrete cross-layer contract. The Widening Dot-Reduce Route
Validation Contract section now includes the `configContractID` and
`RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` fields, the
target consumer order, fail-closed runtime AVL/VL diagnostics, and required
test assertions.

## Self-Repair

* Initial build failed because `configContractID` was not placed inside
  `RVVWideningDotReduceRouteValidationContract`. The header field was added in
  the correct contract location and the build passed.
* Initial target test run failed because existing stride-role negative tests
  still expected later widening-dot ABI index diagnostics. Since the embedded
  runtime AVL/VL contract now correctly rejects duplicate runtime element-count
  parameters first, those expected fragments were updated.
