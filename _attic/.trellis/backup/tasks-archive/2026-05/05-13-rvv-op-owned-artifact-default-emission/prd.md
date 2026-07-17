# RVV op-owned artifact default emission

## Goal

Complete the bounded RVV i32-vadd production artifact route so source/header/
object export treats verified `tcrv_rvv` op/source identity plus plugin-owned
selected-boundary state as required production authority. The default exporter
must no longer accept descriptor-only or source-identity-missing selected
surfaces as enough information to emit RVV compute artifacts.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `8654416 feat(rvv): validate selected boundary source identity`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-selected-boundary-extension-op-production-route/prd.md`
  added selected-boundary source identity fields and target preflight checks.
* Current target export still treats those boundary identity fields as optional:
  `validateBoundarySourceIdentityForRecord` returns success when no
  source-identity field is present.
* The default i32-vadd planning path already materializes
  `tcrv_rvv.lowering_boundary` and `tcrv_rvv.i32_vadd_microkernel`, and the
  emission plan already records typed selected-plan metadata for
  `tcrv_rvv.i32_add` and `TCRVEmitCLowerableOpInterface`.

## Requirements

* Keep the scope to the existing i32-vadd selected RVV family/config slice.
* Artifact export must require complete selected RVV binary source identity on
  the matching `tcrv_rvv.lowering_boundary` before emitting source/header/object
  artifacts.
* The required identity must match the selected typed RVV microkernel body:
  dtype, family, operator, microkernel op, EmitC source op, and generated
  EmitC lowerable interface.
* The source kind must be a recognized typed source authority:
  frontend lowering, default typed-body materialization, or direct typed
  microkernel body.
* Descriptor-only production attempts remain rejected or quarantined.
* Core `tcrv.exec`, generic transforms, and shared construction code must not
  gain RVV semantic branches.
* No new dtype, LMUL, family, performance, or broad smoke-matrix expansion.

## Acceptance Criteria

* [x] The default i32-vadd selected RVV artifact path exports only when the
      matching lowering boundary carries complete selected binary source
      identity.
* [x] Missing selected-boundary source identity fails before generated C source
      is emitted.
* [x] Existing stale/wrong source identity failures continue to fail before
      generated C source is emitted.
* [x] Selected-plan metadata and runtime ABI mirrors remain validated against
      the typed RVV microkernel body and IR-backed callable ABI.
* [x] Descriptor-only RVV i32-vadd production remains fail-closed.
* [x] Focused lit/FileCheck and C++/CMake checks for touched RVV target/plugin
      paths pass.
* [x] `git diff --check` and staged diff check pass before commit.
* [x] Trellis task status/archive and final commit truthfully reflect this
      bounded round.

## Out Of Scope

* Adding i64, vsub, vmul, new LMUL, or new RVV family behavior as the main
  result.
* Moving compute semantics into `tcrv.exec`.
* Descriptor-to-C production export.
* RVV runtime/correctness/performance claims without focused `ssh rvv`
  evidence.
* Template/Toy/TensorExtLite changes except regressions from shared interfaces.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Likely implementation surface:

* `lib/Target/RVV/RVVMicrokernel.cpp`
* `test/Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir`
* focused regression tests under `test/Target/RVVMicrokernel/` and
  `test/Transforms/LoweringBoundary/`

## Completion Summary

This round tightened the default i32-vadd RVV target artifact route without
adding core-pass RVV branches:

* Generic target artifact candidate preflight now rebuilds the selected RVV
  source-authority record in a strict mode for the i32-vadd route.
* In that strict mode, a matching `tcrv_rvv.lowering_boundary` must carry
  complete selected binary source identity before source/header/object helper
  routes can reuse the callable source candidate.
* The required identity is checked against the typed microkernel body:
  selected dtype, family, arithmetic operator, microkernel op, EmitC source op,
  and generated `TCRVEmitCLowerableOpInterface`.
* Descriptor-only and stale-descriptor behavior remains fail-closed after typed
  body authority is established.
* Existing hand-authored i32-vadd test fixtures were updated to carry the same
  source identity now required by the default production route.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` now records the
  i32-vadd selected source identity preflight contract for default artifacts.

No `ssh rvv` evidence was run. This round changes target preflight validation
and test fixtures only; it does not change generated RVV C intrinsic body,
header, object bytes, runtime behavior, correctness, or performance claims.

## Checks Run

* `cmake --build build --target TianChenRVRVVTarget tcrv-translate tcrv-opt -j2`
* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir Target/RVVMicrokernel/rvv-microkernel-header.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir Target/RVVMicrokernel/rvv-microkernel-header.mlir Target/RVVMicrokernel/rvv-microkernel-object.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Transforms/LoweringBoundary/rvv-i32-vadd-descriptor-only-legacy-quarantine.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-auto-materialization|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-op-owned-artifact-default-emission`
