# Stage2 RVV elementwise arithmetic runtime AVL/VL contract migration

## Goal

Migrate the existing RVV elementwise arithmetic route-family validation
contract to embed the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract`, then require target artifact
validation to consume that contract before accepting elementwise route payload,
runtime/control facts, statement-plan facts, header/type mappings, ABI
mappings, scalar-broadcast, masked/strided forms, and metadata mirrors.

This round's bounded consumer slice is the existing production-active
elementwise arithmetic family: ordinary vector/vector add/sub/mul, masked
vector/vector add/sub/mul, scalar-broadcast add/sub/mul, and already-connected
strided elementwise arithmetic where the current provider contract covers it.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV elementwise arithmetic runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no dirty file list, meaning the worktree
  was clean.
* Initial `rtk git log --oneline -8` started at
  `14c37e5a rvv: consume runtime AVL VL contract for splat store`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires elementwise
  arithmetic statement planning to consume verified route-control, typed
  config, materialization, and operand-binding facts. Common EmitC/export must
  carry provider output only.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires promoted
  route-family validation contracts to embed it instead of duplicating runtime
  AVL/VL acceptance in target-local fields.
* Archived runtime AVL/VL migration tasks show the established pattern:
  add `runtimeAVLVLContract` to the route-family validation contract, build it
  through `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after provider-owned
  runtime ABI parameters are populated, and make target validation consume the
  shared runtime contract before family-local facts.
* The archived elementwise/broadcast arithmetic contract extraction task
  already added `RVVElementwiseArithmeticRouteValidationContract` and rewired
  target elementwise validation to consume provider-owned route-family facts.
  Live inspection shows that contract still carries runtime-control fields such
  as `runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL names, memory-form,
  scalar-broadcast, masked/strided, and statement-plan facts as family-local
  fields rather than embedding `runtimeAVLVLContract`.
* This round should not change generated C/C++, runtime ABI order, runtime
  counts, statement ordering, arithmetic semantics, scalar-broadcast behavior,
  mask/tail semantics, or correctness behavior. If that remains true, real
  `ssh rvv` is not required for this validation-only migration.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVElementwiseArithmeticRouteValidationContract`.
* Build the embedded runtime contract from provider-owned elementwise route
  facts: typed SEW/LMUL/policy/config, runtime control plan, runtime VL
  contract, runtime AVL source, selected `with_vl` boundary, setvl intrinsic,
  VL C type, full-chunk VL, loop VL, loop induction, remaining AVL metadata,
  pointer advancement metadata, runtime ABI order, and runtime `n` ABI
  parameter.
* Rewire target elementwise validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  route payload, headers, type mappings, ABI mappings, elementwise operation
  kind, memory/source form, scalar-broadcast facts, masked/strided facts,
  intrinsic leaves, operand binding facts, statement-plan facts, or metadata
  mirrors.
* Statement-plan validation for elementwise arithmetic must consume the
  embedded runtime contract for pre-loop setvl, loop setvl, full-chunk VL,
  loop VL, loop induction, runtime `n` ABI parameter, remaining AVL metadata,
  and pointer advancement checks where the validation contract is present.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` scope,
  setvl intrinsic/type, full-chunk VL, loop VL, loop induction, runtime `n`
  ABI role/order/ownership, remaining AVL metadata, pointer advancement
  metadata, and elementwise statement-plan facts where applicable.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields, but must
  not infer elementwise runtime control or arithmetic semantics from route ids,
  artifact names, manifests, descriptor residue, generated C strings, scripts,
  test names, status fields, or metadata mirrors.

## Acceptance Criteria

* [x] `RVVElementwiseArithmeticRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] `getRVVElementwiseArithmeticRouteValidationContract(...)` derives the
      embedded runtime contract from provider-owned route description and
      runtime ABI facts rather than target-local reconstruction.
* [x] Elementwise target artifact validation consumes the runtime AVL/VL
      contract before accepting route payload, runtime ABI facts, setvl/VL
      facts, operation/source/memory form facts, scalar-broadcast, masked,
      strided, headers, type mappings, statement plans, and metadata mirrors.
* [x] Negative C++ target coverage rejects stale or missing runtime AVL/VL
      contract facts, including runtime AVL source, runtime VL contract id,
      selected `with_vl` scope, setvl callee or VL type, full-chunk VL, loop
      VL, loop induction, runtime `n` ABI facts, and pointer advancement or
      statement-plan facts where applicable.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test`.
* [x] Focused elementwise arithmetic lit or generated-bundle dry-run coverage
      remains green, including at least one scalar-broadcast or masked/strided
      existing case if touched.
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
* Run focused lit/generated-bundle dry-run filters for selected-body
  elementwise arithmetic artifacts after locating the current exact test names.
* Run added-line old-authority scan over touched production/test files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-elementwise-arithmetic-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, arithmetic computation,
scalar-broadcast behavior, mask/tail behavior, layout, correctness, or
performance behavior. The planned diff is contract/validation/test-only. If
that remains true, state the no-runtime-change rationale and cite existing
elementwise executable/dry-run evidence for unchanged paths.

## Out Of Scope

* No new elementwise operations, dtype/LMUL clone batches, vector reduction
  work, masked/indexed/strided memory-family migrations, conversion work,
  MAcc or widening-dot work, source-front-door routes, high-level frontend
  lowering, broad smoke matrices, dashboards, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, test names,
  status fields, or metadata mirrors.
* No cross-family runtime AVL/VL migration beyond elementwise arithmetic in
  this round.

## Definition Of Done

Elementwise arithmetic target route validation accepts only provider-built
routes whose embedded runtime AVL/VL selected-boundary contract matches the
rebuilt route description and statement payload. Focused C++/lit checks pass,
old-authority scan is clean, the task is finished and archived, and one
coherent commit records the production diff plus PRD evidence.

## Implementation Results

Completed in this round:

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVElementwiseArithmeticRouteValidationContract`.
* Rewired
  `getRVVElementwiseArithmeticRouteValidationContract(...)` to build the
  embedded runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after elementwise runtime
  ABI parameters are populated.
* Promoted the runtime contract inputs for elementwise `setvl` and VL C type
  to provider-derived typed config facts (`__riscv_vsetvl_e{SEW}{LMUL}` and
  `size_t`) instead of copying mutable route-description mirrors.
* Rewired elementwise target route validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before route id,
  payload, headers, type mappings, ABI mappings, operation/source/memory form,
  scalar-broadcast, masked, strided, statement-plan, and mirror acceptance.
* Rewired elementwise statement-plan validation to consume the embedded runtime
  contract for runtime `n`, pre-loop setvl, full-chunk VL, loop setvl, loop
  VL, loop induction, loop bounds, remaining AVL expression, and pointer
  advancement facts.
* Added target C++ coverage proving representative plain, masked, and
  scalar-broadcast elementwise contracts embed matching runtime AVL/VL
  selected-boundary facts.
* Added scalar-broadcast elementwise fail-closed target coverage for stale
  runtime AVL source, missing runtime VL contract, stale `with_vl` scope,
  stale setvl callee, stale VL C type, stale full-chunk VL, stale loop VL,
  stale loop induction, stale runtime `n` ABI role, and stale pointer
  advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the elementwise
  arithmetic route validation contract signature, runtime contract consumer
  order, error matrix, required tests, and wrong/correct authority flow.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings remained in target/test code: unused helper
warnings in target validation and enum switch coverage warnings in
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

Result: initially failed one new negative-test text expectation because stale
runtime `n` ABI role now fails at the shared runtime AVL/VL
selected-boundary contract completeness gate. The expectation was corrected to
the earlier fail-closed diagnostic, then the test passed.

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
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(selected-body|pre-realized|masked-add|pre-realized-masked-add|scalar-broadcast-add|pre-realized-scalar-broadcast-add|pre-realized-scalar-broadcast-sub)-dry-run'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 470
  Passed  :   7
```

### Trellis Context

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-elementwise-arithmetic-runtime-avl-vl-contract
```

Result: passed after correcting one archived PRD path in `implement.jsonl`.

### Old-Authority Scan

Added-line scan:

```text
rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/spec/lowering-runtime/emitc-route.md | rg '^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)'
```

Result: no matches. No added positive legacy i32m1, descriptor,
source-front-door, source-artifact, exact i32m1 intrinsic, common-EmitC, or
mirror-only route authority tokens were found.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

## SSH RVV Decision

`ssh rvv` was not run. This round changed provider/target validation
contracts, fail-closed target checks, tests, and specs only. It did not change
emitted C/C++, runtime ABI order, runtime counts, statement ordering,
arithmetic computation, scalar-broadcast behavior, source/result layout,
dtype/config relation, mask/tail behavior, correctness, or performance
behavior. Existing elementwise executable/dry-run evidence remains the
runtime-behavior basis for unchanged generated paths.
