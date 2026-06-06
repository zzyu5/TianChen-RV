# Stage2 RVV computed-masked segment2 store executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked segment2 store selected-body route family truthful at the executable artifact ABI boundary. The selected or pre-realized typed `tcrv_rvv` segment2 store body facts must line up with RVV plugin-owned route validation, statement planning, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence when executable behavior is claimed. If any executable-boundary fact is missing, stale, or derived from mirrors rather than typed body/config/runtime/provider facts, the path must fail closed with a targeted diagnostic.

## What I Already Know

* The previous completed task closed the paired computed-masked segment2 load evidence gap by requiring mask-tail route-family plan/owner mirrors in generated bundle evidence and C++ target validation.
* The current task switches to the computed-masked segment2 store route family only.
* Store-specific facts include compare-produced mask, mask-tail/inactive-lane policy, field0/field1 source roles and unit-load payloads, destination interleaved output memory role, segment tuple creation, masked segment store, dtype/SEW/LMUL/config/policy, runtime AVL/VL, header/prototype bindings, runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`, and old destination preservation on inactive/tail lanes.
* Repo inspection shows the C++ production seam already has provider-owned computed-mask segment2 route facts, segment2 statement plans, and target artifact validation contracts for load/store/update.
* The bounded gap is evidence hardening for the store seam: generated-bundle dry-run/self-test coverage does not yet force the computed-masked segment2 store evidence to expose `tcrv_rvv.mask_tail_policy_route_family_plan` and `tcrv_rvv.mask_tail_policy_owner`, and C++ target export tests have load fail-closed mutations for stale mask-tail provider/candidate facts but not the paired store mutations.

## Requirements

* Scope implementation to `computed_masked_segment2_store_unit_load`.
* Preserve the existing production route chain:
  typed `tcrv_rvv` selected body -> RVV provider-owned route/fact validation -> segment2 statement plan -> `TCRVEmitCLowerableRoute` -> common EmitC materialization -> target artifact -> generated bundle ABI.
* Do not make route ids, helper names, test names, descriptors, artifact metadata, emission-plan mirrors, or common EmitC code into store semantic authority.
* Keep common EmitC/export mechanically neutral.
* Strengthen generated-bundle store evidence so mask-tail route-family plan and owner are checked for explicit and pre-realized selected bodies.
* Strengthen script self-test coverage so fake store bundles cannot pass if mask-tail metadata or boundary summary is missing/stale.
* Strengthen C++ target artifact tests so stale provider and candidate mask-tail plan/owner facts fail closed for computed-masked segment2 store.
* Keep generated harness checks for active field0/field1 writes, inactive old interleaved destination preservation, source preservation, tail preservation, and multi-pattern active/inactive mask coverage.

## Acceptance Criteria

* [x] Explicit and pre-realized generated-bundle dry-run evidence for computed-masked segment2 store checks `tcrv_rvv.mask_tail_policy_route_family_plan`.
* [x] Explicit and pre-realized generated-bundle dry-run evidence for computed-masked segment2 store checks `tcrv_rvv.mask_tail_policy_owner`.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` fails if computed-masked segment2 store fake-bundle mask-tail plan/owner facts disappear or diverge.
* [x] `test/Target/TargetArtifactExportTest.cpp` rejects stale provider mask-tail route-family plan and owner for computed-masked segment2 store.
* [x] `test/Target/TargetArtifactExportTest.cpp` rejects stale candidate metadata mirrors for computed-masked segment2 store mask-tail route-family plan and owner.
* [x] Store active/inactive harness evidence still covers field0/field1 interleaved destination order, source preservation, inactive old destination preservation, tail preservation, runtime `n`, and patterns `0,1`.
* [x] Focused plugin/target checks pass: `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test`.
* [x] Relevant generated-bundle dry-run/lit tests pass for explicit, pre-realized, and direct pre-realized fail-closed store cases.
* [x] If the generated-bundle script changes, `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` pass.
* [x] If runtime correctness is claimed, explicit and pre-realized non-dry-run `ssh rvv` generated-bundle runs pass.
* [x] Bounded old-authority scan over touched source/test/script diff finds no new `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or source-front-door authority in added lines.
* [x] `git diff --check` passes; `git diff --cached --check` and final `git status --short` are checked during finish/commit.

## Definition of Done

* The computed-masked segment2 store executable artifact/ABI seam is hardened with truthful store-specific generated-bundle and target-validation evidence.
* Any stale store mask-tail executable-boundary fact fails closed before artifact acceptance.
* Trellis task context and workspace journal are updated truthfully.
* One coherent commit records the implementation and evidence, unless a blocker is found and documented with the exact continuation point.

## Out of Scope

* No computed-masked segment2 update expansion in this round.
* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No rework of the completed computed-masked segment2 load seam except as bounded reference.
* No MAcc/reduction/dequant/compare-select expansion.
* No high-level Linalg/Vector/StableHLO frontend.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.
* No descriptor-driven or direct-C authority.
* No performance tuning database or dashboard/report-only work.

## Technical Notes

Read and used:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* Archived load task: `.trellis/tasks/archive/2026-06/06-06-rvv-computed-masked-segment2-load-artifact-abi/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-segment2-store-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Target/TargetArtifactExportTest.cpp`

Implementation approach:

* Add store mask-tail plan/owner checks to explicit and pre-realized generated-bundle dry-run tests.
* Add store mask-tail boundary self-test checks in `scripts/rvv_generated_bundle_abi_e2e.py`, mirroring the completed load evidence pattern but keeping store-specific operation and harness checks.
* Add C++ target artifact stale provider/candidate failures for computed-masked segment2 store mask-tail plan and owner.
* Run focused script, C++ target, generated-bundle dry-run/lit, and `ssh rvv` checks.

## Results

Files changed:

* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Target/TargetArtifactExportTest.cpp`

Checks run:

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv /home/kingdom/phdworks/TianchenRV/build/test --filter 'computed-masked-segment2-store'`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/06-07-rvv-computed-masked-segment2-store-artifact-abi/explicit-ssh-rvv-v1 --run-id explicit-computed-mask-segment2-store-ssh-rvv --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-07-rvv-computed-masked-segment2-store-artifact-abi/pre-realized-ssh-rvv-v1 --run-id pre-realized-computed-mask-segment2-store-ssh-rvv --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20`
* Bounded added-line old-authority scan over touched script/test/source diff.
* `rtk git diff --check`

Runtime evidence:

* Explicit selected-body `ssh rvv`: PASS for `computed_masked_segment2_store_unit_load`, counts `0,1,7,16,23,257`, patterns `0,1`; artifact directory `artifacts/tmp/06-07-rvv-computed-masked-segment2-store-artifact-abi/explicit-ssh-rvv-v1/explicit-computed-mask-segment2-store-ssh-rvv`.
* Pre-realized selected-body `ssh rvv`: PASS for `computed_masked_segment2_store_unit_load`, counts `0,1,7,16,23,257`, patterns `0,1`; artifact directory `artifacts/tmp/06-07-rvv-computed-masked-segment2-store-artifact-abi/pre-realized-ssh-rvv-v1/pre-realized-computed-mask-segment2-store-ssh-rvv`.
