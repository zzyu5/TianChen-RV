# RVV EmitC Body Emission Production Route

## Goal

Promote selected RVV EmitC metadata for the migrated vector-dynamic-i32-vadd
and vector-dynamic-i32-vsub routes from artifact annotation into an executable,
plugin-owned RVV EmitC body emission route. The production route must consume
materialized `tcrv.rvv` op-family state, selected SEW/LMUL/tail/mask config,
runtime AVL/VL role data, and source identity to generate clang-compatible RVV
intrinsic C through the common artifact/export path.

## Why Now

The previous completed slice routed selected EmitC metadata through RVV
planning and artifact/export evidence. That makes selected state visible, but
the migrated vadd/vsub default route must no longer depend on direct RVV target
string ownership or descriptor-only compute/config/runtime authority for the
generated executable C body.

## Scope

- Cover the existing i32 vadd/vsub selected routes only.
- Implement or wire one production RVV EmitC body mapper owned by the RVV
  plugin/target boundary.
- Rewire one real artifact/export consumer so generated function/body C comes
  from materialized selected RVV state rather than metadata-only checks.
- Preserve common artifact/source/header/object export as the handoff boundary.
- Keep `tcrv.exec` and generic orchestration family-neutral.
- Keep direct RVV target string emission, if still present, behind explicit
  compatibility checks and out of the migrated default route.

## Out Of Scope

- No new dtype, i64, LMUL matrix, third operation, broad family expansion, or
  performance tuning.
- No descriptor-to-C production exporter resurrection.
- No descriptor element count or descriptor vector shape as authoritative
  compute/config/runtime control for the migrated default route.
- No computation semantics in `tcrv.exec`.
- No RVV family branches in generic core orchestration.
- No GCC/vendor compiler default route.
- No report-only, helper-only, metadata-only, FileCheck-only, or negative-test
  only milestone.
- No runtime/correctness/performance claims beyond focused evidence actually
  run.

## Requirements

- The migrated vadd/vsub route emits generated C function bodies from
  materialized `tcrv.rvv` op-family state.
- Emitted C includes and uses clang/LLVM-compatible RVV C intrinsics through
  `riscv_vector.h`.
- Selected EmitC metadata is consumed by emission code, not only printed,
  exported, or asserted in tests.
- Selected config and runtime length role data are fail-closed inputs for the
  emitted body route.
- Descriptor-only production attempts remain quarantined for the selected
  route.
- vadd/vsub ABI and source identity mismatches fail closed.
- Existing vadd/vsub artifact, bundle, object, scalar dispatch, and e2e
  regressions remain green or are repaired within this task.

## Acceptance Criteria

- [x] vadd and vsub default selected routes materialize `tcrv.rvv` state with
  op-family identity and source identity.
- [x] RVV EmitC body generation consumes selected config and runtime length
  role data.
- [x] Generated C bodies use `riscv_vector.h` RVV intrinsics for vadd/vsub.
- [x] At least one common artifact/export path writes the generated body from
  the plugin-owned mapper.
- [x] Missing EmitC mapping, stale op identity, vadd/vsub ABI mismatch, missing
  or conflicting selected config, and missing/stale runtime length role data
  fail closed in focused tests.
- [x] Descriptor-only compute/config/runtime authority is not the migrated
  default route.
- [x] Core `tcrv.exec` and generic passes remain orchestration-only with no new
  RVV semantic branches.
- [x] Focused local build/lit/e2e checks pass.
- [x] `ssh rvv` compile/run evidence is collected if generated RVV C behavior
  changes and the remote is reachable; otherwise the exact blocker is recorded.

## Implementation Summary

This round adds an explicit `RVVBinaryEmitCBodyMapping` contract and wires
`RVVMicrokernel` source generation so migrated vadd/vsub body emission consumes
the selected emission-plan metadata for route kind, source authority, required
header, and arithmetic intrinsic. The common EmitC lowerable route now receives
that mapping and uses it to add `riscv_vector.h` and choose the compute
`emitc.call_opaque` callee.

For frontend-selected vadd/vsub, missing or stale selected EmitC body metadata
now fails before source emission. Compatibility fallback from selected config
remains available only for non-frontend selected records without selected plan
metadata, and descriptor-only compute/config/runtime authority remains
quarantined behind the existing typed RVV body checks.

Generated valid vadd/vsub RVV intrinsic behavior is unchanged: the emitted
compute intrinsics remain `__riscv_vadd_vv_i32m1` and
`__riscv_vsub_vv_i32m1`. This round changes the production ownership and
fail-closed authority path, plus emitted comments/object evidence, so no new
runtime/correctness/performance claim or `ssh rvv` run was made.

## Validation

- `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-microkernel-bundle-e2e.test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir Scripts/rvv-microkernel-e2e.test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Scripts/rvv-scalar-dispatch-bundle-e2e.test`
- Bounded reference scan over touched RVV plugin/target/export files for descriptor/direct-string authority terms.
- `git diff --check`

## Evidence Plan

- Read the selected-route specs and previous archived PRDs before source edits.
- Inspect RVV dialect, selected planning, selected config/runtime length
  contracts, microkernel emission, scalar dispatch, artifact export, and
  vadd/vsub transform/e2e tests.
- Run focused C++/TableGen build coverage for touched RVV plugin/target/export
  code and generated headers.
- Run affected lit/FileCheck tests for VectorToExec, TargetArtifactExport,
  TargetArtifactBundleExport, RVVMicrokernel, RVVScalarDispatch, selected
  quarantine, and `rvv-microkernel-bundle-e2e.test`.
- Run bounded reference scans in touched RVV plugin/target/export files for
  direct descriptor-driven or direct string-owned production C emission.
- Run `git diff --check` and `git diff --cached --check`.

## Definition Of Done

- Task PRD, implementation context, check context, and journal status are
  truthful.
- Production code is rewired for the scoped route rather than adding an unused
  helper.
- Focused checks pass or exact blockers are recorded.
- Task is finished/archived if complete.
- One coherent commit is created when complete.
