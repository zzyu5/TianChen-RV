# RVV C++ artifact/runtime production invocation

## Goal

Move one remaining source-frontdoor invocation handoff from script/bundle-index
evidence into the C++ production dispatch artifact route. For the bounded MLIR
vector i32 vadd/vsub source-frontdoor path, `RVVScalarDispatch.cpp` must consume
the validated selected RVV source identity when emitting generated dispatch
source/header/object artifacts, so the artifacts themselves carry source
identity, op-family identity, selected config, and runtime length roles through
the normal target artifact/bundle route.

This round is not a script-only or metadata-only milestone. The coherent
submodule is the C++ RVV+scalar dispatch artifact owner, with focused
FileCheck/script evidence that generated source/header/object or bundle
artifacts reuse that owner.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- At task creation, `git status --short` was clean and HEAD was
  `41e1dbd feat(rvv): close vector source dispatch invocation`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief.
- The previous commit changed Trellis metadata,
  `scripts/rvv_scalar_dispatch_e2e.py`, and
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`; it did not touch
  `LowerSourceRVVBinaryToExec.cpp`, TargetArtifactExport/BundleExport,
  RVVMicrokernel emission, RuntimeABI, or `RVVScalarDispatch.cpp`.
- Archived PRDs show that earlier same-day rounds already added C++ selected
  source identity validation in `RVVScalarDispatch.cpp` and script-level
  generated bundle invocation evidence.
- Current code already validates selected-plan source identity for RVV dispatch
  candidates and puts `tcrv_rvv.dispatch_contract_selected_source_identity` in
  bundle index metadata for vector dynamic vadd/vsub.
- Current dispatch source/header artifacts print selected config, runtime
  length, EmitC mapping, runtime ABI parameters, and candidate metadata, but
  header output does not expose a single dispatch-owned selected source
  identity contract; source output relies on embedded component/candidate
  metadata rather than an explicit dispatch artifact contract line.

## Requirements

- Add an active C++ production owner change in `lib/Target/RVV/RVVScalarDispatch.cpp`.
- Generated RVV+scalar dispatch source and header artifacts must require and
  print a dispatch-owned selected RVV source identity summary derived from the
  validated selected config contract plus selected-plan source-kind and
  microkernel-op metadata.
- The summary must include source kind, dtype, family, arithmetic operator,
  selected RVV microkernel op, EmitC source op, and
  `TCRVEmitCLowerableOpInterface`.
- Missing selected source identity, stale source kind, missing/stale
  microkernel op, selected config mismatch, runtime length role mismatch, and
  stale ABI signature must continue to fail closed before artifact or bundle
  success.
- Both dynamic vector i32-vadd and i32-vsub source-frontdoor bundle tests must
  prove generated dispatch source/header artifacts carry the dispatch-owned
  selected source identity and the existing bundle index contract.
- Scripts may remain evidence consumers only; no new Python compiler semantics
  or surrogate kernels.
- Generic core orchestration must remain target-neutral: no new RVV semantic
  branch in shared core passes and no compute semantics in `tcrv.exec`.

## Acceptance Criteria

- [x] `RVVScalarDispatch.cpp` actively consumes selected source identity while
      emitting dispatch artifacts, not only during bundle index generation or
      script validation.
- [x] Generated dispatch source and header artifacts for vector dynamic i32-vadd
      print a dispatch-owned selected source identity line matching
      `source_kind=frontend-lowering,dtype=i32,family=i32-vadd,operator=add,...`.
- [x] Generated dispatch source and header artifacts for vector dynamic i32-vsub
      print a dispatch-owned selected source identity line matching
      `source_kind=frontend-lowering,dtype=i32,family=i32-vsub,operator=subtract,...`.
- [x] Existing bundle index metadata still carries
      `tcrv_rvv.dispatch_contract_selected_source_identity` for both vadd and
      vsub.
- [x] Existing fail-closed coverage for stale/missing selected source identity,
      selected config mismatch, runtime length role mismatch, stale ABI
      signatures, descriptor-only authority, and explicit-route misuse remains
      green.
- [x] Focused C++/TableGen build covers touched target/export support tools.
- [x] Focused lit/FileCheck covers vector dynamic vadd/vsub source-frontdoor,
      TargetArtifactBundleExport, RVVMicrokernel, RVVScalarDispatch, and script
      e2e routes touched by this round.
- [x] At least one focused `ssh rvv` generated artifact invocation passes, or an
      exact blocker is recorded after local generation and compile.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, and one coherent commit complete if the task is finished.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 expansion, LMUL matrix expansion, third dynamic vector operation, broad
  family matrix, broad smoke suite, or performance tuning.
- No script-only, e2e-harness-only, task-only, journal-only, test-only,
  negative-test-only, wrapper-only, report-only, or metadata-only milestone.
- No new Python compiler semantics.
- No descriptor-to-C production exporter and no descriptor element count or
  vector shape as compute/config/runtime/ABI authority.
- No moving computation semantics into `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No GCC/vendor compiler default route.
- No Template/Toy/TensorExtLite/IME/Offload work except narrow regressions
  caused by shared validation.
- No runtime, correctness, or performance claim beyond focused evidence
  actually run.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior PRDs read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-generated-artifact-invocation-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-production-artifact-runtime-closure/prd.md`.
- Primary implementation surface:
  `lib/Target/RVV/RVVScalarDispatch.cpp`.
- Focused test surfaces:
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  and existing RVVScalarDispatch/script regressions.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit from `build/test` for:
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  `Target/RVVScalarDispatch`,
  `Target/RVVMicrokernel`,
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
- Run script self-tests:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`,
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run focused local generated dispatch bundle dry-runs for dynamic vector
  i32-vadd and i32-vsub.
- Run focused `ssh rvv` generated artifact invocation for at least one dynamic
  vector source-frontdoor dispatch bundle route.
- Run bounded ref-scans for explicit-only routing, descriptor-only authority,
  stale op-family identity, and generic core semantic branches.
- Run `git diff --check`, `git diff --cached --check`, and Trellis validation
  before finish/archive.

## Implementation Summary

- Production owner changed: `lib/Target/RVV/RVVScalarDispatch.cpp`.
- Added `buildDispatchSelectedSourceIdentityContractSummary` and
  `printDispatchSelectedSourceIdentityContract`.
- Dispatch source/header export now requires unique
  `tcrv_rvv.selected_binary_source_kind` and
  `tcrv_rvv.selected_binary_microkernel_op` selected-plan metadata, reuses the
  existing dispatch selected-source identity validator, and prints a
  dispatch-owned `dispatch_selected_source_identity` contract line before
  generated source/header/object artifact success.
- The generated contract line is derived from the validated
  `RVVBinarySelectedConfigContract`, so dtype, family, operator, EmitC source
  op, generated lowerable interface, selected source kind, and microkernel op
  remain tied to production C++ state rather than script parsing or descriptor
  shape.
- Focused vector dynamic vadd/vsub TargetArtifactBundleExport tests now require
  the generated dispatch source and header artifacts to carry the
  dispatch-owned selected source identity, in addition to the existing bundle
  index contract.
- No generic core orchestration, `tcrv.exec` compute semantics, descriptor-to-C
  exporter, or script compiler semantics were added.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- C++ smoke tests passed:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed from `build/test`:
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.
- Focused lit passed from `build/test`:
  `Target/RVVScalarDispatch`, `Target/RVVMicrokernel`, and
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test` (`46/46`).
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local generated dispatch bundle dry-runs passed for dynamic vector i32-vadd
  and i32-vsub:
  `codex-cpp-source-id-vadd-local` and
  `codex-cpp-source-id-vsub-local`.
- Direct artifact inspection found `dispatch_selected_source_identity` in both
  generated source and header artifacts for vadd and vsub under
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/`.
- `ssh rvv` evidence passed for dynamic vector i32-vsub generated dispatch
  bundle invocation:
  `codex-cpp-source-id-vsub-ssh`, `status: success`,
  `ssh_evidence: true`, `ssh_evidence_verified: true`.
- Bounded ref-scan found no new generic core `tcrv.exec` or common-pass RVV
  semantic branch; RVV matches are limited to the existing source-frontdoor
  transform and RVV target-local owner surfaces.
- `git diff --check` and Trellis context validation passed before finish.

## Self-Repair Notes

- The first FileCheck update placed the new header identity expectation after
  runtime-length comments, but generated headers print the dispatch-owned
  selected source identity immediately after manifest route metadata. The
  header checks were reordered to match the production artifact order.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new
dialect op, command signature, cross-layer schema field, plugin protocol, or
architecture rule. It tightened an existing RVVScalarDispatch artifact emission
contract using already-specified selected-source identity, selected config,
runtime length, EmitC-route, and descriptor-quarantine rules.
