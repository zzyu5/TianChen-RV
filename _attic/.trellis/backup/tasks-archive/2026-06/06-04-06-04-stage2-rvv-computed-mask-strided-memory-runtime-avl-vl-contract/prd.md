# Stage2 RVV computed-mask strided memory runtime AVL/VL contract migration

## Goal

Migrate the existing computed-mask strided memory route-family validation
contract so `RVVComputedMaskStridedMemoryRouteValidationContract` embeds
`RVVRuntimeAVLVLSelectedBoundaryContract`, and require target artifact
validation to consume that embedded runtime contract before accepting strided
load/store route payloads, runtime/control facts, computed-mask facts,
stride/memory layout facts, header/type/ABI mappings, statement-plan facts,
and artifact mirrors.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv computed-mask strided load/store body
  -> RVV plugin-owned computed-mask strided memory route facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVComputedMaskStridedMemoryRouteValidationContract
  -> target artifact validation consumes the embedded runtime contract
  -> provider-built TCRVEmitCLowerableRoute setvl/mask/strided-load-or-store/unit-load-or-store statements
  -> common EmitC materialization
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask strided memory runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `bb02f56c rvv: consume runtime AVL VL contract for computed indexed memory`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no spawned
  subagents or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps RVV route authority in the selected
  `tcrv.exec` envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned
  legality/route provider, common EmitC materialization, and target artifact
  validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires non-segment
  computed-mask memory route planning to flow through verified family plans,
  materialization facts, memory operand-binding facts, route-control provider
  plan, and computed-mask memory statement-plan facts before provider route
  construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned authority for
  runtime `n`/AVL/VL selected-boundary facts before target artifact acceptance.
* Archived runtime AVL/VL tasks show the established migration pattern: route
  validation contracts embed `runtimeAVLVLContract`, builders populate it via
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`, and target validators run
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before route-local
  acceptance.
* Live inspection shows
  `RVVComputedMaskStridedMemoryRouteValidationContract` exists from the prior
  provider-contract extraction task, but still lacks `runtimeAVLVLContract`.
* Live inspection shows
  `getRVVComputedMaskStridedMemoryRouteValidationContract(...)` fills
  route-local runtime fields but does not call
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* Live inspection shows computed-mask strided target validation checks
  runtime/control and statement shape through family-local fields, and its
  statement-plan validation path currently calls
  `validateRVVCompareSelectMaskRouteStatementPlan(...)` without passing the
  provider runtime contract.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVComputedMaskStridedMemoryRouteValidationContract`.
* Populate the embedded contract in
  `getRVVComputedMaskStridedMemoryRouteValidationContract(...)` from canonical
  provider-owned facts: SEW, LMUL, tail/mask policy, config contract, setvl
  intrinsic, VL C type, runtime ABI order, runtime ABI parameter list, and the
  computed-mask strided memory consumer label.
* Target validation must call
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before accepting
  route-local runtime/control, computed-mask producer, mask type/source/form,
  stride source/type/unit facts, memory-form facts, header/type, ABI,
  statement-plan, and mirror facts.
* Computed-mask strided statement validation must consume setvl callee, VL
  type, full-chunk VL, loop VL, loop induction, and runtime n/AVL ABI
  parameter from the embedded runtime contract rather than mutable
  route-description mirrors.
* Preserve the existing provider-built strided store/load-unit-store statement
  sequence and do not change emitted C, runtime ABI order, runtime counts,
  mask/stride behavior, correctness behavior, or performance behavior.
* Fail closed for missing, stale, or mismatched runtime AVL source, runtime VL
  contract id, selected `with_vl` scope, setvl intrinsic, VL type,
  full-chunk VL, loop VL, loop induction, runtime n ABI role/order, remaining
  AVL metadata, pointer advancement metadata, computed-mask producer facts,
  stride facts, strided memory-form facts, and statement-plan facts where
  applicable.
* Keep common EmitC/export neutral. Do not infer runtime control from route
  ids, artifact names, test names, manifests, C strings, descriptor residue,
  source-front-door metadata, or mirror metadata.

## Acceptance Criteria

* [x] `RVVComputedMaskStridedMemoryRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The provider builder fills the embedded runtime contract through
      `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` using provider-owned
      typed/config/runtime facts.
* [x] Target computed-mask strided memory validation calls the shared runtime
      AVL/VL selected-boundary validator before route-local checks and before
      candidate route acceptance can succeed.
* [x] Rebuilt route statement validation for strided store/load-unit-store uses
      the embedded runtime contract for setvl/VL/loop/runtime n facts and still
      checks strided statement-plan structure.
* [x] Focused C++ target coverage proves positive embedded contract
      population and negative stale/missing runtime AVL/VL contract facts for
      computed-mask strided memory.
* [x] Focused computed-mask strided generated-bundle/lit coverage remains
      green.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy i32, route-id, artifact-name, descriptor,
      source-front-door/source-export/direct-C, common-EmitC, exact-intrinsic,
      or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Non-Goals

* Do not add new memory operations.
* Do not revisit computed-mask indexed memory, segment2 memory, reductions,
  elementwise, conversion, MAcc, widening-dot, scalar/offload/IME,
  source-front-door routes, or high-level frontend lowering.
* Do not move RVV semantics into common EmitC/export.
* Do not run broad smoke matrices unless focused checks expose a shared
  regression.
* Do not run `ssh rvv` unless emitted C, runtime ABI order, runtime counts,
  statement ordering, strided load/store behavior, correctness behavior, or
  performance behavior changes.

## Evidence Plan

* Build-focused checks:
  * `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
  * `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* Focused test checks:
  * `rtk ./build/bin/tianchenrv-target-artifact-export-test`
  * `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
  * Focused computed-mask strided generated-bundle dry-run lit tests when
    local lit tooling is available.
* Hygiene:
  * `rtk git diff --check`
  * Added-line old-authority scan over touched files.
  * `rtk git status --short`

## Technical Notes

* Primary provider files:
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* Primary target consumer file:
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* Primary focused tests:
  * `test/Target/TargetArtifactExportTest.cpp`
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-strided-store-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-strided-load-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-store-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-load-dry-run.test`

## Implementation Summary

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVComputedMaskStridedMemoryRouteValidationContract`.
* Populated the embedded contract in
  `getRVVComputedMaskStridedMemoryRouteValidationContract(...)` via
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  computed-mask strided runtime ABI parameters are attached to the contract.
* Rewired computed-mask strided target validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before
  route-local runtime control, computed-mask, stride/memory, header/type, ABI,
  and statement-plan checks.
* Rewired strided store/load-unit-store payload validation to pass
  `contract->runtimeAVLVLContract` into
  `validateRVVCompareSelectMaskRouteStatementPlan(...)`, so setvl/VL/loop and
  runtime n facts come from the embedded runtime selected-boundary contract.
* Updated computed-mask strided target fixtures to use canonical selected
  runtime AVL/VL facts from `RVVSelectedBodyConfigVLContract` rather than
  hand-coded `i` / `vl_full` loop names.
* Added positive and negative target C++ coverage for embedded runtime AVL/VL
  contract population and stale runtime AVL source, runtime VL contract,
  selected `with_vl` scope, setvl callee, loop induction, loop VL, runtime n
  ABI role, and pointer advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` to record
  computed-mask strided memory as a promoted runtime AVL/VL selected-boundary
  contract consumer with concrete target validation and test requirements.

## Evidence Results

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16`
  passed after self-repair. Existing switch/unused-function warnings remained.
* `rtk ./build/bin/tianchenrv-target-artifact-export-test` passed after
  self-repair of stale test expectation text from `i` to canonical `offset`.
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 16`
  passed.
* `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
* Focused lit passed from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-computed-masked-strided-(store|load)|rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-store-dry-run'`;
  5 passed, 472 excluded.
* Direct generated-bundle dry-runs passed:
  * `computed_masked_strided_store` pre-realized selected-body dry-run
    returned `rvv_generated_bundle_abi_e2e: dry_run_success`.
  * `computed_masked_strided_load_unit_store` pre-realized selected-body
    dry-run returned `rvv_generated_bundle_abi_e2e: dry_run_success`.
* `rtk git diff --check` passed.
* Added-line old-authority scan over touched source/spec/task files returned
  no matches.
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-strided-memory-runtime-avl-vl-contract`
  passed.
* No `ssh rvv` run was performed because this round changes provider/target
  validation contracts and manual/dry-run test fixtures only; it does not
  alter emitted C, runtime ABI order, runtime counts, statement ordering,
  strided load/store semantics, correctness behavior, or performance behavior.
