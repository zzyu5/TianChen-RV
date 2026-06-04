# Stage2 RVV runtime AVL/VL selected-boundary contract promotion

## Goal

Promote runtime `n` / AVL / VL control facts into an explicit RVV
provider-owned selected-boundary contract and prove one existing route-family
consumer uses it before target artifact validation accepts a route.

This round's bounded consumer slice is
`computed_masked_segment2_update_unit_load`, because the previous archived task
already proved the explicit and pre-realized selected-body paths generate
artifact/header/harness bundles and pass real `ssh rvv` correctness without
needing compiler-path changes.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime AVL/VL selected-boundary contract promotion`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: `ok` (clean through RTK).
* Initial `rtk git log --oneline -8` started at
  `addac4bf chore(task): archive segment2 update executable evidence`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact -> `ssh rvv` evidence for
  runtime/correctness/performance claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says runtime ABI values must
  be explicitly imported into the selected `tcrv_rvv` body and consumed by
  `setvl` / `with_vl` / route provider facts. Segment2 planning is selected
  body route-family provider planning, not route-entry ownership.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the
  provider-owned route fact surface pattern. Common EmitC carries provider
  output only; it must not infer RVV runtime control, dtype, intrinsic names,
  route ids, or ABI semantics.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  evidence only when runtime/correctness/performance behavior is claimed or
  changed. This round should not change generated C, ABI order, runtime counts,
  pointer advancement, or correctness behavior.
* The archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-segment2-update-executable-closeout/`
  records successful real `ssh rvv` runs for both pre-realized and explicit
  `computed_masked_segment2_update_unit_load` paths over runtime counts
  `0,1,7,16,23,257` and mask patterns `0,1`.
* Live inspection shows `RVVRuntimeAVLVLControlPlan` already derives and
  verifies core runtime facts from explicit `tcrv_rvv.runtime_abi_value`, the
  selected variant's single runtime-element-count binding, `setvl`, `with_vl`,
  SEW/LMUL, policy, and the selected-body config/VL contract.
* Segment2 provider preflight already checks that the route-control provider
  plan and segment2 provider plan point at the same `RVVRuntimeAVLVLControlPlan`
  before constructing `TCRVEmitCLowerableRoute`.
* Target segment2 validation currently checks runtime AVL/VL facts across
  separate description/statement-plan comparisons. It lacks a single named
  provider-owned runtime AVL/VL selected-boundary contract object that target
  validation must consume before artifact acceptance.

## Requirements

* Add a named RVV provider-owned runtime AVL/VL selected-boundary contract that
  can be built from a rebuilt `RVVSelectedBodyEmitCRouteDescription`.
* The contract must carry runtime n/AVL source, runtime ABI role/order, SEW,
  LMUL, tail/mask policy, config/VL contract IDs, setvl/with_vl boundary ops,
  setvl intrinsic/type, full-chunk VL name, loop remaining-AVL VL name, loop
  induction, remaining-AVL expression metadata, pointer/element advancement
  metadata, bounded-slice/multi-VL facts, selected-body provenance, and the
  provider-derived runtime AVL ABI parameter.
* Integrate that contract into the existing segment2 validation contract for
  `computed_masked_segment2_update_unit_load` and sibling segment2 consumers
  without changing generated route behavior.
* Target artifact validation for segment2 must consume the runtime AVL/VL
  contract before checking artifact/export mirrors and fail closed for missing
  or stale runtime control fields.
* Add focused C++ target coverage proving:
  * `computed_masked_segment2_update_unit_load` exposes the runtime AVL/VL
    selected-boundary contract and the contract matches the rebuilt route
    description.
  * stale `runtimeAVLASource`, stale `runtimeVLContractID`, stale
    `vlScopeOpName`, or stale pointer advancement metadata fails before route
    acceptance.
* Keep common EmitC/export neutral: no common code may infer runtime control
  from route ids, artifact names, metadata strings, descriptors, C snippets, or
  tests.

## Acceptance Criteria

* [x] `RVVRuntimeAVLVLSelectedBoundaryContract` or equivalent named provider
      contract exists in RVV provider-facing code.
* [x] `getRVVSegment2MemoryRouteValidationContract(...)` includes and validates
      the runtime AVL/VL contract for the chosen segment2 consumer slice.
* [x] Segment2 target artifact provider-fact validation consumes the runtime
      AVL/VL contract before validating route headers/types/ABI/statement
      payloads.
* [x] Negative target coverage rejects stale or missing runtime AVL/VL facts
      for `computed_masked_segment2_update_unit_load`.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused generated-bundle/lit dry-run filter for computed-mask segment2
      update still passes.
* [x] Old-authority scan over touched production/test files finds no new
      positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      mirror-only authority, route-id/artifact-name authority, or exact
      `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build/run focused C++ tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/generated-bundle dry-run filter from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2-update|pre-realized-selected-body-artifact-computed-masked-segment2-update|explicit-selected-body-artifact-computed-masked-segment2-update'`
* Bounded old-authority scan over touched files.
* `rtk git diff --check`
* Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-avl-vl-selected-boundary-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes generated C, runtime ABI
order, runtime counts, pointer advancement, or correctness behavior. The
planned diff should only add provider/target contract validation. If that stays
true, cite the archived live evidence from
`06-04-stage2-rvv-computed-mask-segment2-update-executable-closeout` for the
unchanged executable path.

## Out Of Scope

* No new RVV operations, dtype/LMUL clone batches, segment3+, compare/select
  coverage, reduction/MAcc coverage, source-front-door routes, high-level
  frontend lowering, broad smoke matrices, dashboards, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, C strings, descriptors, or tests.
* No full cross-family runtime AVL/VL promotion in this round. Remaining route
  families should be named as continuation work if this slice completes.

## Definition Of Done

The segment2 `computed_masked_segment2_update_unit_load` target route consumer
accepts only provider-built routes whose runtime AVL/VL selected-boundary
contract matches the rebuilt route description and statement payload. Focused
C++/lit checks pass, old-authority scan is clean, the task is finished and
archived, and a coherent commit records the production diff plus PRD evidence.

## Implementation Results

* Added `RVVRuntimeAVLVLSelectedBoundaryContract` to the RVV runtime control
  provider-facing API.
* Added `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`, which derives the
  runtime selected-boundary contract from provider-expected SEW/LMUL,
  tail/mask policy, config contract, `setvl` intrinsic/type facts, runtime ABI
  order, and the provider-owned runtime ABI parameters.
* Embedded the runtime AVL/VL selected-boundary contract into
  `RVVSegment2MemoryRouteValidationContract`.
* Rewired segment2 target route validation so provider-fact validation consumes
  the runtime contract before route headers, type mappings, ABI mappings, and
  statement payload validation.
* Rewired segment2 statement-plan validation to use the runtime contract's
  full-chunk VL, loop VL, and loop induction facts for the loop/setvl/pointer
  checks.
* Added focused target tests for the computed-mask segment2 update route:
  positive runtime contract matching, stale runtime AVL source, stale runtime
  VL contract, stale `with_vl` scope, and stale pointer advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with a 7-section
  code-spec for the RVV runtime AVL/VL selected-boundary contract.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings were emitted in target/test code, but no
build errors occurred.

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
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2-update|pre-realized-selected-body-artifact-computed-masked-segment2-update|explicit-selected-body-artifact-computed-masked-segment2-update'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 472
  Passed  :   5
```

### Old-Authority Scan

Full touched-file scan:

```text
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only" include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/tasks/06-04-stage2-rvv-runtime-avl-vl-selected-boundary-contract
```

Result: matches in `test/Target/TargetArtifactExportTest.cpp` are pre-existing
test inventory or negative guardrails. Task-file matches are non-authority
requirements and scan terms.

Added-line scan:

```text
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\\\.i32_|!tcrv_rvv\\\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"'
```

Result: no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### SSH RVV

Not rerun. This round changed provider/target validation contracts and C++
tests only. It did not change emitted C/C++, runtime ABI order, runtime counts,
pointer advancement expressions, harness behavior, or correctness semantics.
The unchanged executable path is covered by archived live evidence in
`.trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-segment2-update-executable-closeout/prd.md`,
which records successful explicit and pre-realized
`computed_masked_segment2_update_unit_load` `ssh rvv` runs over counts
`0,1,7,16,23,257` and patterns `0,1`.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the new
provider-facing runtime AVL/VL selected-boundary contract. This was required
because the task added a concrete provider API signature and route-family
contract payload fields. The spec section records scope, signature, contract
fields, fail-closed error matrix, good/base/bad cases, tests, and wrong-vs-
correct flow.

## Continuation Point

Full cross-family runtime AVL/VL promotion remains open. The next continuation
point is to migrate another existing consumer family, such as base-memory
movement or compare/select, to consume `RVVRuntimeAVLVLSelectedBoundaryContract`
in its route validation contract and target validator.
