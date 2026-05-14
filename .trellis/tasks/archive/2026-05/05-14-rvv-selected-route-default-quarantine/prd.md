# RVV selected route default quarantine

## Goal

Make the migrated dynamic vector `i32-vadd` and `i32-vsub` RVV selected routes
the production/default artifact route. Default source/header/object/bundle
export for these slices must require compiler-produced, materialized
`tcrv_rvv` state: selected source identity, finite binary op-family identity,
selected SEW/LMUL/tail/mask config, runtime AVL/VL role data, and runtime ABI
role data.

This round is structural migration of the default route. It must not close by
adding more descriptor-preserving coverage, another operation family, a dtype
matrix, or broad smoke evidence.

## Current State

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round was clean at HEAD
  `d3bccab feat(rvv): extend selected op-family route to vsub`.
* No `.trellis/.current-task` existed before this task. This task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/05-14-rvv-selected-route-default-quarantine`.
* The previous selected op-family round extended the selected production route
  from dynamic vector `i32-vadd` to `i32-vsub` and made target-artifact
  candidate preflight require selected lowering-boundary source identity for
  finite typed RVV binary artifact routes by dtype instead of only for vadd.
* The remaining bottleneck is not family coverage. It is ensuring stale
  descriptor-only compute/config/runtime authority cannot silently become the
  production/default authority for migrated vadd/vsub slices.

## Requirements

* Scope is limited to the existing migrated dynamic vector `i32-vadd` and
  `i32-vsub` selected routes and their default artifact export path.
* Default target artifact source/header/object/bundle export must consume the
  same selected-state contracts from materialization through selected emission
  planning, microkernel export, scalar dispatch, and bundle records.
* Compiler-produced `tcrv_rvv` state is authoritative for:
  * binary op identity and source identity;
  * selected config: SEW, LMUL, tail policy, mask policy, vector type suffix,
    setvl suffix, intrinsic suffixes;
  * runtime length: external element-count ABI parameter, AVL source/role, VL
    source/scope, dynamic source-tail authority.
* Descriptor element count, descriptor vector shape, descriptor body/control
  policy, or direct descriptor-to-C behavior must not be authoritative for
  compute/config/runtime decisions on migrated vadd/vsub default routes.
* Descriptor mirrors may remain only as bounded, validated mirrors after typed
  authority or as clearly quarantined legacy/direct-helper fixtures outside
  the default production route.
* Stale descriptor-only inputs, missing selected source identity, stale op
  identity, selected-config conflict, runtime-length conflict, or ABI mismatch
  must fail closed before artifact completion.
* Core `tcrv.exec` remains orchestration-only. Generic transforms and generic
  artifact bundle routing must not gain RVV semantic branches.
* Python remains evidence/tooling only; compiler core behavior stays in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.

## Acceptance Criteria

* [x] Default export for migrated vadd/vsub requires materialized `tcrv_rvv`
      op/source identity plus selected config and runtime length contract data.
* [x] At least one obsolete descriptor-authority path or fixture for migrated
      vadd/vsub is deleted, bypassed, or explicitly quarantined outside the
      default route.
* [x] Descriptor-only or source-identity-stripped frontend-lowering attempts
      fail before complete source/header/object/bundle artifact output.
* [x] Selected emission planning, microkernel export, scalar dispatch, bundle
      export, and the e2e script consume the same selected-state contracts for
      vadd and vsub.
* [x] Positive vadd/vsub compiler-produced artifact/e2e tests remain green and
      still show selected source identity, selected config, runtime length data,
      generated source/header/object output, and RVV intrinsic route.
* [x] Focused fail-closed coverage includes missing source identity and relies
      on existing or new tests for stale op identity, ABI mismatch, missing or
      conflicting selected config, and missing or stale runtime length role
      data.
* [x] A bounded ref-scan over touched RVV plugin/target/export files confirms
      descriptor fields are not authoritative on the migrated default route and
      no new vadd/vsub semantic branch was added to generic core orchestration.
* [x] Focused build/lit/script checks pass for touched RVV dialect/plugin/
      target/export tools and affected transform/artifact/e2e tests.
* [x] `ssh rvv` evidence is collected only if generated RVV source/object
      behavior changes. If this round changes fail-closed preflight/quarantine
      only, no runtime/correctness/performance claim is made.
* [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, clean final worktree, and one coherent
      commit complete the round if finished.

## Non-Goals

* No new dtype, i64 expansion, LMUL matrix, third operation, broad family
  matrix, broad smoke suite, or performance tuning.
* No descriptor-to-C production exporter.
* No descriptor element count or descriptor vector shape as authoritative
  compute/config/runtime control for migrated vadd/vsub.
* No computation semantics in `tcrv.exec`.
* No RVV semantic branches in generic core orchestration.
* No GCC/vendor compiler default route.
* No Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin changes
  except narrow regressions caused by shared validation.

## Read Context

Specs read before implementation:

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

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-op-family-production-route/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-05-14-rvv-compiler-produced-artifact-e2e-closure/prd.md`
* `.trellis/workspace/codex/journal-5.md` latest RVV compiler-produced and
  selected op-family sections.

Likely implementation surface:

* `lib/Target/RVV/RVVMicrokernel.cpp`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `scripts/rvv_microkernel_e2e.py`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
* `test/Scripts/rvv-microkernel-bundle-e2e.test`

## Definition Of Done

The migrated dynamic vector vadd/vsub route no longer has any default artifact
path where descriptor-only compute/config/runtime authority can stand in for
compiler-produced selected RVV state. Any remaining direct-helper or legacy
fixture behavior is marked and quarantined outside the default production
route, and focused evidence shows the compiler-produced vadd/vsub selected
routes still generate the expected source/header/object bundle artifacts.

## Implementation Summary

Completed in this round:

* `lib/Target/RVV/RVVMicrokernel.cpp` now treats a selected variant that
  declares `tcrv_rvv.selected_binary_source_kind = "frontend-lowering"` as
  requiring matching selected lowering-boundary source identity even when the
  caller reaches `buildModuleRecord()` through helper-only self-check export
  instead of the generic artifact front door.
* RVV microkernel artifact route metadata now carries explicit quarantine
  claims for descriptor compute, selected-config, and runtime authority. These
  claims mark descriptor mirrors as non-authoritative after typed RVV source
  identity, selected-config contract, and IR-backed runtime ABI authority.
* RVV+scalar dispatch artifact route metadata now carries matching quarantine
  claims for descriptor compute/config/runtime authority, which is the default
  vector-dynamic vadd/vsub bundle route.
* The dynamic vector vadd and vsub bundle tests now assert the quarantine route
  claims in generated bundle indexes.
* The dynamic vector vadd and vsub bundle tests now fail closed when selected
  source identity is stripped from the lowering boundary and microkernel before
  invoking the explicit RVV self-check helper route. Before this change, the
  vsub self-check helper emitted C for the stripped frontend-lowering module.

No generic core orchestration, `tcrv.exec` computation semantics, new dtype,
new family, LMUL matrix, descriptor-to-C exporter, GCC/vendor default, runtime
behavior, correctness claim, or performance path was added.

## Validation Summary

Focused build and checks run:

* `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir` from `build/test`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-microkernel-bundle-e2e.test` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/target-artifact-bundle-positive.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir` from `build/test`
* Manual self-check-helper fail-closed probe for stripped vsub selected source
  identity: rejected with `requires selected RVV binary source identity before
  target artifact export`.
* Bounded ref-scan: changed source files are limited to
  `lib/Target/RVV/RVVMicrokernel.cpp` and
  `lib/Target/RVV/RVVScalarDispatch.cpp`; no `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec` sources changed.
* `git diff --check`

No `ssh rvv` evidence was run because generated RVV source/object bytes,
runtime behavior, correctness claims, and performance claims were not changed
in this round.
