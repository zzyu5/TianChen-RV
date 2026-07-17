# RVV extension-family EmitC emission route

## Goal

Route the migrated dynamic vector `i32-vadd` and `i32-vsub` RVV selected
artifact path through a plugin-owned extension-family EmitC emission surface.
The executable C semantics for these selected routes must come from
materialized `tcrv_rvv` op state plus selected-state contracts, then flow
through the common target artifact/source/header/object/bundle handoff, instead
of leaving direct RVV target string emission as the only owner of the generated
RVV intrinsic C body.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round is clean at HEAD
  `c42715b fix(rvv): quarantine selected route descriptor authority`.
* No `.trellis/.current-task` existed before this task. This task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/05-14-rvv-extension-family-emitc-emission-route`.
* Recent completed rounds already made the migrated vadd/vsub selected default
  artifact route require compiler-produced RVV selected state: source identity,
  op-family identity, selected SEW/LMUL/tail/mask config, runtime AVL/VL role
  data, and runtime ABI role data.
* Descriptor-only compute/config/runtime authority is already quarantined for
  the migrated default route. This round must not re-open descriptor-to-C as a
  production exporter.
* The next architectural bottleneck is emission ownership: the finite RVV
  binary selected routes still need a plugin-owned EmitC/source-op mapping
  surface that the target artifact route consumes for generated RVV intrinsic C
  evidence.

## Requirements

* Scope is limited to existing migrated selected routes:
  `vector-dynamic-i32-vadd` and `vector-dynamic-i32-vsub`.
* Materialized `tcrv_rvv` ops and selected-state contracts must drive the
  bounded EmitC emission route for the migrated slice.
* Selected emission planning or the RVV target handoff must map op-family
  identity, selected vector config, source identity, runtime length roles, and
  runtime ABI roles into a common EmitC/artifact route payload.
* Generated C must remain clang/LLVM-compatible, include `riscv_vector.h`, and
  use RVV C intrinsics for the bounded selected families, including
  `__riscv_vadd_vv_i32m1` and `__riscv_vsub_vv_i32m1` for the current m1
  dynamic vector fixtures.
* Descriptor fields may only remain bounded mirror/cross-check metadata after
  typed op/source authority is established. Descriptor-only production
  attempts, stale op identity, missing selected config, stale runtime length
  role data, and ABI mismatches must fail closed before complete artifact
  output.
* Core `tcrv.exec` must remain orchestration-only. Generic transforms and
  generic artifact/export front doors must not gain RVV semantic branches.
* If full replacement of the existing direct RVV target emission is too large
  for one round, complete one coherent production submodule: the RVV
  op-family EmitC mapping plus at least one active production consumer path
  for migrated vadd/vsub, and document any remaining direct target helper as a
  compatibility path with selected-state fail-closed checks.

## Acceptance Criteria

* [x] The migrated vadd/vsub artifact path has a plugin-owned RVV
      extension-family EmitC/source-op mapping surface consumed by production
      target artifact export.
* [x] The mapping is driven by materialized `tcrv_rvv` typed op/source identity
      plus selected config and runtime length contracts, not by descriptor-only
      route ids or descriptor element counts.
* [x] Generated source/header/object/bundle output for vadd and vsub still
      preserves selected op-family state, selected config, runtime ABI
      signature, runtime AVL/VL role data, `riscv_vector.h`, and the expected
      RVV intrinsic spellings.
* [x] Direct target emission is either bypassed for the migrated slice,
      factored behind the plugin-owned EmitC route mapping, or explicitly
      retained as a temporary compatibility helper whose active consumers and
      fail-closed selected-state checks are documented by code/tests.
* [x] Focused fail-closed coverage rejects descriptor-only production
      attempts, missing EmitC/source-op mapping, stale vadd/vsub op identity,
      vadd/vsub ABI mismatch, missing/conflicting selected config, and
      missing/stale runtime length role data before complete artifact output.
* [x] Focused vadd/vsub VectorToExec, LinalgToExec, RVVMicrokernel,
      RVVScalarDispatch, TargetArtifactExport, TargetArtifactBundleExport, and
      `rvv-microkernel-bundle-e2e.test` regressions remain green where touched.
* [x] Bounded ref-scan over touched RVV plugin/target/export files confirms
      descriptor-driven direct C emission is not the migrated default route,
      and no generic `tcrv.exec` or common transform RVV semantic branch was
      added.
* [x] If generated RVV source/object bytes change, focused `ssh rvv`
      compile/run evidence is collected through the existing e2e path. If not,
      the final report must avoid runtime correctness/performance claims.
* [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean worktree, and one coherent
      commit complete the round if finished.

## Non-Goals

* No new dtype, i64 expansion, LMUL matrix, third operation, broad family
  matrix, broad smoke suite, or performance tuning.
* No descriptor-to-C production exporter and no descriptor element count or
  descriptor vector shape as authoritative compute/config/runtime control.
* No computation semantics in `tcrv.exec`.
* No RVV semantic branches in generic core orchestration or generic transforms.
* No GCC/vendor compiler default route.
* No Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin work
  except narrow regressions caused by shared validation.
* No runtime, correctness, or performance claims beyond focused evidence
  actually run for this finite selected vadd/vsub route.

## Technical Notes

Specs read before PRD:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-route-default-quarantine/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-op-family-production-route/prd.md`
* `.trellis/workspace/codex/journal-5.md` latest RVV compiler-produced,
  selected op-family, full-state, and descriptor-quarantine sections.

Likely implementation surface:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp`
* existing target artifact/export route owners under `include/TianChenRV/Target/`,
  `lib/Target/`, and `tools/tcrv-translate/tcrv-translate.cpp`
* focused vadd/vsub VectorToExec, LinalgToExec, RVVMicrokernel,
  RVVScalarDispatch, TargetArtifactExport, TargetArtifactBundleExport, and
  `test/Scripts/rvv-microkernel-bundle-e2e.test` tests.

## Definition Of Done

The finite migrated dynamic vector `i32-vadd`/`i32-vsub` selected route has an
active plugin-owned RVV EmitC/source-op mapping consumed by the production
artifact route. The generated source/header/object/bundle path still emits
clang-compatible RVV intrinsic C from selected `tcrv_rvv` state, while
descriptor-only compute/config/runtime authority remains quarantined and core
orchestration remains target-neutral.

## Implementation Summary

This round completed the coherent production submodule allowed by the task
brief: RVV selected emission planning now publishes a typed EmitC route mapping
for finite `i32-vadd`/`i32-vsub` selected routes, and the production RVV
source/header/object/dispatch artifact handoff validates and carries it.

The new selected-plan metadata is:

* `tcrv_rvv.emitc_route_kind =
  extension-family-ops-to-emitc-call-opaque`
* `tcrv_rvv.emitc_source_authority = mlir-emitc-cpp-emitter`
* `tcrv_rvv.emitc_required_header = riscv_vector.h`
* `tcrv_rvv.emitc_arithmetic_intrinsic =
  __riscv_vadd_vv_i32m1` or `__riscv_vsub_vv_i32m1` for the migrated m1
  dynamic vector slice, with i32m2/i64m1 planner tests proving the selected
  vector shape still drives the intrinsic suffix.

Production consumers updated:

* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` appends the
  plugin-selected EmitC route metadata after selected source identity.
* `lib/Target/RVV/RVVMicrokernel.cpp` and
  `lib/Target/RVV/RVVScalarDispatch.cpp` require the same EmitC route metadata
  before source/header/object/bundle export.
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h` owns the shared
  RVV selected EmitC route metadata spelling, role, notes, and intrinsic
  derivation from the selected config contract.
* `scripts/rvv_microkernel_e2e.py` treats the route kind, source authority,
  required header, and arithmetic intrinsic as required manifest authority for
  bundle/direct e2e evidence.

Descriptor-only authority remains quarantined: the direct helper remains a
temporary RVV target compatibility path for explicit bounded finite routes, but
the migrated default route now fails closed unless selected typed source
identity, selected config, runtime length, source identity, and EmitC route
metadata all match.

No files under `lib/Transforms`, `include/TianChenRV/Dialect/Exec`, or
`lib/Dialect/Exec` were changed.

## Validation Summary

Local checks run:

* `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin
  tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test
  tianchenrv-rvv-extension-plugin-test -j2`
* `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  after fixture repair
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir
  Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
  from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Scripts/rvv-microkernel-bundle-e2e.test` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir
  Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir
  Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir
  Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir
  Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir
  Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir
  Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir
  Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir
  Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir
  Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir
  Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir`
  from `build/test`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-14-rvv-extension-family-emitc-emission-route`
* `python3 ./.trellis/scripts/task.py finish`
* `python3 ./.trellis/scripts/task.py archive --no-commit
  05-14-rvv-extension-family-emitc-emission-route`
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/archive/2026-05/05-14-rvv-extension-family-emitc-emission-route`
* `git diff --cached --check`

Self-repair performed:

* Fixed the plugin planner/target metadata vector type mismatch by adding a
  local `VariantSelectedPlanMetadata` append helper that reuses the shared
  target contract descriptors.
* Updated C++ direct-candidate fixtures so they carry the same EmitC route
  metadata required by production validation.
* Moved vsub bundle source FileCheck assertions to the actual metadata comment
  location.
* Relaxed stale-descriptor negative tests so they no longer forbid intrinsic
  names inside diagnostic selected-plan metadata, while still forbidding
  completed generated C bodies.
* Updated `rvv_microkernel_e2e.py --self-test` synthetic manifests to include
  the new route metadata instead of weakening evidence checks.

`ssh rvv` evidence run because generated RVV source/object evidence changed:

* `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd
  --lower-vector-i32-vadd-frontend --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --expect-selected-kernel
  frontend_vector_dynamic_i32_vadd --runtime-count 7 --runtime-count 16
  --runtime-count 23 --run-id codex-emitc-route-vadd --overwrite
  --timeout 120 --connect-timeout 10`
* `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vsub
  --lower-vector-i32-vsub-frontend --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --expect-selected-kernel
  frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16
  --runtime-count 23 --run-id codex-emitc-route-vsub --overwrite
  --timeout 120 --connect-timeout 10`

Both returned `status: success`, `mode: ssh`, `ssh_evidence: true`, and
runtime counts `[7, 16, 23]`.

Artifact roots:

* `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-emitc-route-vadd`
* `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-emitc-route-vsub`

Observed source hashes:

* vadd bundle source:
  `e021095ac0f18ba84dda180f110263fc059ab408907d0f801388b24e506f61fa`
* vsub bundle source:
  `a63f26559eb5f088f9547462b9ba9c14792698d3e4d56f06695fe61a43828948`
