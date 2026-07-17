# Stage2 RVV computed-mask indexed memory runtime AVL/VL contract migration

## Goal

Migrate the existing computed-mask indexed memory route-family validation
contract so `RVVComputedMaskIndexedMemoryRouteValidationContract` embeds
`RVVRuntimeAVLVLSelectedBoundaryContract`, and require target artifact
validation to consume that embedded runtime contract before accepting indexed
gather/scatter route payloads, runtime/control facts, computed-mask facts,
index/memory layout facts, header/type/ABI mappings, statement-plan facts, and
artifact mirrors.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv computed-mask indexed gather/scatter body
  -> RVV plugin-owned computed-mask indexed memory route facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVComputedMaskIndexedMemoryRouteValidationContract
  -> target artifact validation consumes the embedded runtime contract
  -> provider-built TCRVEmitCLowerableRoute setvl/mask/index-load/indexed-load-or-store/unit-load-or-store statements
  -> common EmitC materialization
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask indexed memory runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `1a198712 rvv: consume runtime AVL VL contract for unit masked memory`.
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
  plan, and the computed-mask memory statement-plan boundary before provider
  route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned authority for
  runtime `n`/AVL/VL selected-boundary facts before target artifact acceptance.
* Archived runtime AVL/VL selected-boundary, base-memory, unit-stride masked
  memory, elementwise, splat-store, conversion, MAcc, widening-dot, and
  standalone-reduction tasks show the established migration pattern: route
  validation contracts embed `runtimeAVLVLContract`, builders populate it via
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`, and target validators run
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before route-local
  acceptance.
* Live inspection shows
  `RVVComputedMaskIndexedMemoryRouteValidationContract` still lacks
  `runtimeAVLVLContract`.
* Live inspection shows
  `getRVVComputedMaskIndexedMemoryRouteValidationContract(...)` fills
  route-local runtime fields but does not call
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* Live inspection shows computed-mask indexed target validation checks
  `runtimeControlPlanID`, runtime ABI order, setvl/VL fields, and statement
  shape through family-local fields, and the indexed statement validation path
  currently calls `validateRVVCompareSelectMaskRouteStatementPlan(...)` without
  passing the provider runtime contract.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVComputedMaskIndexedMemoryRouteValidationContract`.
* Populate the embedded contract in
  `getRVVComputedMaskIndexedMemoryRouteValidationContract(...)` from canonical
  provider-owned facts: SEW, LMUL, tail/mask policy, config contract,
  setvl intrinsic, VL C type, runtime ABI order, runtime ABI parameter list,
  and the computed-mask indexed memory consumer label.
* Target validation must call
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before accepting
  route-local runtime/control, computed-mask producer, mask type/source/form,
  index/vector/layout, header/type, ABI, statement-plan, and mirror facts.
* Computed-mask indexed statement validation must consume setvl callee, VL
  type, full-chunk VL, loop VL, loop induction, and runtime n/AVL ABI parameter
  from the embedded runtime contract rather than mutable route-description
  mirrors.
* Keep the existing provider-built indexed gather/scatter statement sequence:
  full-chunk setvl, runtime AVL/VL loop setvl, compare lhs load, index load,
  index scale, compare rhs load, payload or passthrough load, compare mask,
  masked indexed load/store, and final unit store where applicable.
* Fail closed for missing, stale, or mismatched runtime AVL source, runtime VL
  contract id, selected `with_vl` scope, setvl intrinsic, VL type,
  full-chunk VL, loop VL, loop induction, runtime n ABI role/order, remaining
  AVL metadata, pointer advancement metadata, computed-mask producer facts,
  index EEW/offset/source facts, indexed memory-form facts, and statement-plan
  facts where applicable.
* Keep common EmitC/export neutral. Do not infer runtime control from route
  ids, artifact names, test names, manifests, C strings, descriptor residue,
  source-front-door metadata, or mirror metadata.

## Acceptance Criteria

* [x] `RVVComputedMaskIndexedMemoryRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The provider builder fills the embedded runtime contract through
      `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` using provider-owned
      typed/config/runtime facts.
* [x] Target computed-mask indexed memory validation calls the shared runtime
      AVL/VL selected-boundary validator before route-local checks and before
      candidate route acceptance can succeed.
* [x] Rebuilt route statement validation for indexed gather/scatter uses the
      embedded runtime contract for setvl/VL/loop/runtime n facts and still
      checks indexed statement-plan structure.
* [x] Focused C++ target coverage proves positive embedded contract population
      and negative stale/missing runtime AVL/VL contract facts for computed-mask
      indexed memory.
* [x] Focused computed-mask indexed generated-bundle/lit coverage remains
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
* Do not migrate computed-mask strided memory, segment2 memory, reductions,
  elementwise, conversion, MAcc, widening-dot, scalar/offload/IME,
  source-front-door routes, or high-level frontend lowering.
* Do not move RVV semantics into common EmitC/export.
* Do not run broad smoke matrices unless focused checks expose a shared
  regression.
* Do not run `ssh rvv` unless emitted C, runtime ABI order, runtime counts,
  statement ordering, indexed gather/scatter behavior, correctness behavior, or
  performance behavior changes.

## Evidence Plan

* Build-focused checks:
  * `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
  * `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* Focused test checks:
  * `rtk ./build/test/Target/tianchenrv-target-artifact-export-test`
  * `rtk ./build/test/Plugin/tianchenrv-rvv-extension-plugin-test`
  * Focused computed-mask indexed generated-bundle dry-run lit tests, if the
    local build tree exposes `llvm-lit`.
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
  * `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-gather-load-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-scatter-store-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-gather-load-dry-run.test`
  * `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test`

## Implementation Summary

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVComputedMaskIndexedMemoryRouteValidationContract`.
* Populated the embedded contract in
  `getRVVComputedMaskIndexedMemoryRouteValidationContract(...)` via
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  computed-mask indexed runtime ABI parameters are attached to the contract.
* Rewired computed-mask indexed target validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before
  route-local runtime control, header/type, ABI, computed-mask, index/memory,
  and statement-plan checks.
* Rewired indexed gather/scatter payload validation to pass
  `contract->runtimeAVLVLContract` into
  `validateRVVCompareSelectMaskRouteStatementPlan(...)`, so setvl/VL/loop and
  runtime n facts come from the embedded runtime selected-boundary contract.
* Updated computed-mask indexed target fixtures to use canonical selected
  runtime AVL/VL facts from `RVVSelectedBodyConfigVLContract` rather than
  hand-coded `i` / `vl_full` loop names.
* Added positive and negative target C++ coverage for embedded runtime AVL/VL
  contract population and stale runtime AVL source, runtime VL contract,
  selected `with_vl` boundary, setvl callee, loop induction, loop VL, and
  runtime n ABI role facts.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` to record
  computed-mask indexed memory as a promoted runtime AVL/VL selected-boundary
  contract consumer with concrete target validation and test requirements.

## Evidence Results

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
  passed.
* `rtk ./build/bin/tianchenrv-target-artifact-export-test` passed.
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
  passed.
* `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
* Focused generated-bundle dry-runs passed:
  * explicit computed-mask indexed gather:
    `rvv_generated_bundle_abi_e2e: dry_run_success`
  * explicit computed-mask indexed scatter:
    `rvv_generated_bundle_abi_e2e: dry_run_success`
  * pre-realized computed-mask indexed gather:
    `rvv_generated_bundle_abi_e2e: dry_run_success`
  * pre-realized computed-mask indexed scatter:
    `rvv_generated_bundle_abi_e2e: dry_run_success`
* `llvm-lit` and `FileCheck` were not available in PATH or `build/bin`, so the
  focused script tests were run directly in `--dry-run` mode with
  `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and empty
  `--llvm-readobj`.
* `rtk git diff --check` passed.
* Added-line old-authority scan over touched source/test/task files produced
  no matches.
* No `ssh rvv` run was performed because this round changes provider/target
  validation contracts and manual/dry-run test fixtures only; it does not alter
  emitted C, runtime ABI order, runtime counts, statement ordering, indexed
  gather/scatter semantics, correctness behavior, or performance behavior.
